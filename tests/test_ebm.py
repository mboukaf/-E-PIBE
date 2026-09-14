"""Energy-based residual models and the EPIBE schedule, Section 3.2."""

from __future__ import annotations

import math

import pytest
import torch

from pibe.config import EBMConfig, RunConfig
from pibe.core.bank import EstimatorBank, Mode
from pibe.core.energy_bank import EnergyEstimatorBank
from pibe.data.noise import build_noise_model
from pibe.ebm import CompositeGaussLegendre, ScalarEBM, residual_radii, resolve_radii
from pibe.experiment import build_experiment
from pibe.training.epibe_phases import Phase, apply_epibe_phase, phase_for_iteration
from pibe.training.epibe_trainer import EPIBETrainer
from pibe.training.trainer import PIBETrainer
from pibe.utils.seeding import make_generator


# ----------------------------------------------------------------------
# quadrature, Eq. (59) / Remark 4
# ----------------------------------------------------------------------


def test_quadrature_integrates_polynomials_exactly() -> None:
    """A composite rule of order m is exact for degree 2m-1 on each panel."""
    rule = CompositeGaussLegendre(radius=1.5, panels=4, nodes_per_panel=8)
    for power in (0, 2, 4, 6):
        exact = 2 * 1.5 ** (power + 1) / (power + 1)
        assert float(rule.integrate(rule.nodes**power)) == pytest.approx(exact, rel=1e-12)


def test_log_integral_matches_a_direct_integral() -> None:
    """The log-domain path agrees with the direct one where both are safe."""
    rule = CompositeGaussLegendre(radius=1.0, panels=8, nodes_per_panel=8)
    log_f = -2.0 * rule.nodes**2
    assert float(rule.log_integral(log_f)) == pytest.approx(
        float(torch.log(rule.integrate(torch.exp(log_f)))), rel=1e-12
    )


def test_log_integral_survives_energies_that_would_overflow() -> None:
    """Why the log domain is used: exp(-beta E) overflows across the class."""
    rule = CompositeGaussLegendre(radius=1.0, panels=8, nodes_per_panel=8)
    log_f = torch.full_like(rule.nodes, 800.0)
    assert torch.isfinite(rule.log_integral(log_f))
    assert not torch.isfinite(rule.integrate(torch.exp(log_f)))


# ----------------------------------------------------------------------
# Assumption 3: the constrained energy class
# ----------------------------------------------------------------------


def test_gauge_normalization_is_exact() -> None:
    """Eq. (47): the energy vanishes at the origin by construction."""
    ebm = ScalarEBM(radius=0.4)
    zero = torch.zeros((), dtype=torch.float64)
    assert float(ebm.energy(zero).detach()) == 0.0


def test_energy_respects_its_sup_norm_bound() -> None:
    """Eq. (48), enforced by the output parameterization rather than penalized."""
    ebm = ScalarEBM(radius=0.4, energy_bound=3.0)
    with torch.no_grad():
        for parameter in ebm.energy.parameters():
            parameter.mul_(50.0)  # drive the raw network far past the bound
        values = ebm.energy(torch.linspace(-0.4, 0.4, 2001, dtype=torch.float64))
    assert float(values.abs().max()) <= 3.0 + 1e-12


def test_projection_makes_the_parameter_set_compact() -> None:
    """Algorithm 2 lines 17 and 23 constrain each update to K_k."""
    ebm = ScalarEBM(radius=0.4)
    with torch.no_grad():
        for parameter in ebm.parameters():
            parameter.add_(100.0)
    ebm.project(2.0)
    assert max(float(p.abs().max()) for p in ebm.parameters()) <= 2.0


def test_spectral_norm_gives_a_guaranteed_lipschitz_constant() -> None:
    """The optional route to (48)'s second clause; the bound must actually hold."""
    ebm = ScalarEBM(radius=0.5, energy_scale=4.0, spectral_norm_layers=True)
    grid = torch.linspace(-0.5, 0.5, 20001, dtype=torch.float64)
    with torch.no_grad():
        values = ebm.energy(grid)
    measured = float((values[1:] - values[:-1]).abs().max() / (grid[1] - grid[0]))
    assert measured <= ebm.energy.lipschitz_constant + 1e-9
    assert ebm.energy.lipschitz_constant == pytest.approx(4.0 / 0.5)


# ----------------------------------------------------------------------
# the density (49) and its likelihood (53)
# ----------------------------------------------------------------------


def test_density_is_normalized() -> None:
    ebm = ScalarEBM(radius=0.3)
    assert ebm.normalization_error() < 1e-10


def test_untrained_density_is_essentially_uniform() -> None:
    """A near-zero energy must give the uniform law on R_k, not something else."""
    torch.manual_seed(0)
    radius = 0.3
    ebm = ScalarEBM(radius=radius)
    with torch.no_grad():
        for parameter in ebm.energy.parameters():
            parameter.mul_(1e-6)
    moments = ebm.moments()
    assert moments.mean == pytest.approx(0.0, abs=1e-6)
    assert moments.std == pytest.approx(radius / math.sqrt(3.0), rel=1e-4)
    assert moments.log_partition == pytest.approx(math.log(2 * radius), abs=1e-6)


def test_negative_log_likelihood_is_the_mean_negative_log_density() -> None:
    """Eq. (53) is exactly -mean(log p); the two routes must agree."""
    torch.manual_seed(1)
    ebm = ScalarEBM(radius=0.5)
    residual = 0.1 * torch.randn(500, dtype=torch.float64)
    residual = residual.clamp(-0.4, 0.4)
    assert float(ebm.negative_log_likelihood(residual)) == pytest.approx(
        float(-ebm.log_density(residual).mean()), rel=1e-12
    )


@pytest.mark.parametrize(
    "family,bias",
    [("gaussian", 0.0), ("gaussian", 0.05), ("skewed", 0.0), ("contaminated", 0.04)],
)
def test_fitted_density_recovers_the_sample_mean(family: str, bias: float) -> None:
    """Eq. (54) on a known law: this is what PIBE's quadratic term cannot do.

    The whole purpose of the energy-based data term is that the location of the
    residual law is *inferred* rather than assumed to be zero, so recovering a
    nonzero mean is the property to test.
    """
    torch.manual_seed(0)
    noise = build_noise_model(family, 0.02, bias=bias, bias_relative=False)
    sample = noise.sample((100_000,), generator=make_generator(0))

    ebm = ScalarEBM(radius=0.5, energy_bound=12.0)
    optimizer = torch.optim.Adam(ebm.parameters(), lr=2e-3)
    for _ in range(1200):
        optimizer.zero_grad(set_to_none=True)
        ebm.negative_log_likelihood(sample).backward()
        optimizer.step()
        ebm.project(10.0)

    moments = ebm.moments()
    assert moments.mean == pytest.approx(float(sample.mean()), abs=2e-3)
    assert moments.std == pytest.approx(float(sample.std()), rel=0.15)


# ----------------------------------------------------------------------
# the out-of-support barrier
# ----------------------------------------------------------------------


def test_barrier_pushes_residuals_back_into_the_support() -> None:
    """"Likelihood zero" outside must mean an inward force, not a flat region.

    A clamp leaves the energy constant outside, so an escaped residual receives
    no gradient at all and the estimate is free to drift further --- the failure
    this barrier exists to prevent.
    """
    ebm = ScalarEBM(radius=0.5, barrier_scale=0.1)
    outside = torch.tensor([0.8, -0.8], dtype=torch.float64, requires_grad=True)
    ebm.energy_with_barrier(outside).sum().backward()
    # Positive residual gets a positive slope (pushed down), and vice versa.
    assert float(outside.grad[0]) > 100.0
    assert float(outside.grad[1]) < -100.0

    clamping = ScalarEBM(radius=0.5, barrier_scale=0.0)
    flat = torch.tensor([0.8, -0.8], dtype=torch.float64, requires_grad=True)
    clamping.energy_with_barrier(flat).sum().backward()
    assert float(flat.grad.abs().max()) == 0.0


def test_barrier_leaves_the_density_untouched_inside_the_support() -> None:
    """The barrier is a training device, not part of Eq. (49)."""
    torch.manual_seed(2)
    with_barrier = ScalarEBM(radius=0.5, barrier_scale=0.1)
    inside = torch.linspace(-0.49, 0.49, 101, dtype=torch.float64)
    assert torch.allclose(
        with_barrier.energy_with_barrier(inside), with_barrier.energy(inside)
    )


def test_out_of_support_fraction_reports_violations() -> None:
    ebm = ScalarEBM(radius=0.5)
    residual = torch.tensor([0.0, 0.2, 0.6, -0.9], dtype=torch.float64)
    assert ebm.out_of_support_fraction(residual) == pytest.approx(0.5)


# ----------------------------------------------------------------------
# residual supports, Assumption 3
# ----------------------------------------------------------------------


def test_a_priori_radii_follow_assumption_3() -> None:
    """Box widths, plus the noise support on the measured cell only."""
    from pibe.systems.registry import build_system

    system = build_system("automatica_n3v2")
    radii = residual_radii(system, noise_bound=0.01)
    widths = [
        float(system.state_bounds[j, 1] - system.state_bounds[j, 0])
        for j in range(system.n)
    ]
    assert radii[2] == pytest.approx(widths[0] + 0.01)
    assert radii[3] == pytest.approx(widths[1])
    assert radii[4] == pytest.approx(widths[2])


def test_radius_override_accepts_a_scalar_or_one_per_cell() -> None:
    from pibe.systems.registry import build_system

    system = build_system("automatica_n3v2")
    assert set(resolve_radii(system, 0.0, override=0.3).values()) == {0.3}
    assert resolve_radii(system, 0.0, override=[0.1, 0.2, 0.3])[3] == pytest.approx(0.2)
    with pytest.raises(ValueError, match="expected 3 radii"):
        resolve_radii(system, 0.0, override=[0.1, 0.2])


# ----------------------------------------------------------------------
# the schedule, Algorithm 2 / Remark 5
# ----------------------------------------------------------------------


def test_phase_map_reproduces_algorithm_2() -> None:
    """With n_fit = 0 the split is exactly the three phases of Remark 5."""
    phases = [phase_for_iteration(i, n_ebm=10, n_par=20) for i in range(30)]
    assert phases[:10] == [Phase.WARMUP] * 10
    assert phases[10:20] == [Phase.LOCAL] * 10
    assert phases[20:] == [Phase.GLOBAL] * 10


def test_optional_fit_phase_is_inserted_after_activation() -> None:
    phases = [phase_for_iteration(i, n_ebm=10, n_par=20, n_fit=5) for i in range(25)]
    assert phases[9] is Phase.WARMUP
    assert phases[10:15] == [Phase.FIT] * 5
    assert phases[15:20] == [Phase.LOCAL] * 5
    assert phases[20] is Phase.GLOBAL


def test_warmup_uses_the_quadratic_term_and_the_others_do_not() -> None:
    assert not Phase.WARMUP.uses_energy
    assert all(p.uses_energy for p in (Phase.FIT, Phase.LOCAL, Phase.GLOBAL))
    assert not Phase.FIT.trains_pinn
    assert all(p.trains_pinn for p in (Phase.WARMUP, Phase.LOCAL, Phase.GLOBAL))
    assert Phase.GLOBAL.mode is Mode.GLOBAL
    assert all(p.mode is Mode.LOCAL for p in (Phase.WARMUP, Phase.FIT, Phase.LOCAL))


def _small_epibe_experiment(**overrides):
    config = RunConfig.from_dict(
        {
            "system": {"name": "automatica_n3v2"},
            "data": {"n_trajectories": 4, "n_samples": 21, "horizon": 2.0,
                     "noise_sigma": 0.002, "noise_bias": 0.05,
                     "noise_bias_relative": False},
            "basis": {"kind": "fourier", "q": 3, "omega": 1.2566370614359172,
                      "coeff_lo": -0.1, "coeff_hi": 0.1},
            "architecture": {"latent_dim": 4, "encoder_hidden": [8],
                             "decoder_hidden": [8], "head_hidden": [8]},
            "training": {"n_total": 6, "n_par": 4, "n_collocation": 8,
                         "batch_size": 2, "log_every": 0},
            "ebm": {"enabled": True, "n_ebm": 2, "radius": 0.2,
                    "hidden": [8], "panels": 4, "nodes_per_panel": 8},
        }
    )
    for key, value in overrides.items():
        setattr(config.ebm, key, value)
    return build_experiment(config)


def test_phase_freezes_the_right_blocks() -> None:
    experiment = _small_epibe_experiment()
    bank = experiment.bank

    pinn, ebm = apply_epibe_phase(bank, 3, Phase.WARMUP)
    assert pinn and not ebm and not bank.use_energy

    pinn, ebm = apply_epibe_phase(bank, 3, Phase.FIT)
    assert not pinn and ebm and bank.use_energy

    pinn, ebm = apply_epibe_phase(bank, 3, Phase.LOCAL)
    assert pinn and ebm and bank.use_energy
    # Upstream stays frozen while a cell is trained locally.
    assert not any(p.requires_grad for p in bank.cell_parameters(2))

    pinn, ebm = apply_epibe_phase(bank, 3, Phase.GLOBAL)
    assert any(p.requires_grad for p in bank.cell_parameters(2))
    assert any(p.requires_grad for p in bank.ebm_parameters(2))
    # Cells the outer loop has not reached take no part in any objective.
    assert not any(p.requires_grad for p in bank.cell_parameters(4))


# ----------------------------------------------------------------------
# the bank
# ----------------------------------------------------------------------


def test_energy_term_can_be_restricted_to_the_measurement_cell() -> None:
    """``ebm.cells: [2]`` keeps quadratic consistency terms downstream.

    Only the first cell's residual carries measurement noise; Eqs. (29)/(38)
    are deterministic consistency penalties whose correct value is zero.
    """
    experiment = _small_epibe_experiment(cells=[2])
    bank = experiment.bank
    bank.use_energy = True
    assert bank.uses_energy_at(2)
    assert not bank.uses_energy_at(3)
    assert set(bank.density_moments()) == {2}

    data = experiment.train_data
    with torch.enable_grad():
        outputs = bank(target_cell=bank.final_index, y=data.y, t_data=data.t,
                       t_coll=experiment.t_coll, mode=Mode.GLOBAL, create_graph=False)
    loss = bank.cell_loss(3, data.y, experiment.t_coll, outputs, lam=1.0)
    residual = bank.consistency_residual(3, data.y, outputs)
    assert float(loss.data) == pytest.approx(float((residual**2).mean()))


def test_fit_phase_falls_back_when_a_cell_has_no_energy_model() -> None:
    """A non-energy cell must still have trainable parameters in every phase."""
    bank = _small_epibe_experiment(cells=[2]).bank
    pinn, ebm = apply_epibe_phase(bank, 3, Phase.FIT)
    assert pinn and not ebm


def test_unknown_energy_cells_are_rejected() -> None:
    with pytest.raises(ValueError, match="outside 2\\.\\.4"):
        _small_epibe_experiment(cells=[2, 9])


def test_energy_bank_is_a_pibe_bank_until_the_energy_term_is_enabled() -> None:
    """Section 3.2 changes only the consistency term; everything else is shared."""
    experiment = _small_epibe_experiment()
    bank = experiment.bank
    assert isinstance(bank, EstimatorBank)

    data = experiment.train_data
    with torch.enable_grad():
        outputs = bank(target_cell=bank.final_index, y=data.y, t_data=data.t,
                       t_coll=experiment.t_coll, mode=Mode.GLOBAL, create_graph=False)

    bank.use_energy = False
    quadratic = bank.cell_loss(2, data.y, experiment.t_coll, outputs, lam=1.0)
    residual = bank.consistency_residual(2, data.y, outputs)
    assert float(quadratic.data) == pytest.approx(float((residual**2).mean()))

    bank.use_energy = True
    energetic = bank.cell_loss(2, data.y, experiment.t_coll, outputs, lam=1.0)
    # The physics half is untouched by the choice of data term.
    assert float(energetic.physics) == pytest.approx(float(quadratic.physics))


def test_consistency_residual_signs_follow_the_paper() -> None:
    """eps_2 = y - x_1^2 and eps_k = x_{k-1}^{k-1} - x_{k-1}^k; Eq. (54) needs the sign."""
    experiment = _small_epibe_experiment()
    bank = experiment.bank
    data = experiment.train_data
    with torch.enable_grad():
        outputs = bank(target_cell=bank.final_index, y=data.y, t_data=data.t,
                       t_coll=experiment.t_coll, mode=Mode.GLOBAL, create_graph=False)
    assert torch.allclose(
        bank.consistency_residual(2, data.y, outputs), data.y - outputs[2].x_prev_data
    )
    assert torch.allclose(
        bank.consistency_residual(3, data.y, outputs),
        outputs[2].x_new_data - outputs[3].x_prev_data,
    )


def test_ebm_parameters_are_not_part_of_the_pinn_blocks() -> None:
    """Algorithm 2 schedules the two independently, so they must not overlap."""
    bank = _small_epibe_experiment().bank
    pinn = {id(p) for p in bank.parameters_upto(bank.final_index)}
    ebm = {id(p) for p in bank.ebm_parameters_upto(bank.final_index)}
    assert pinn and ebm and not (pinn & ebm)


def test_experiment_selects_the_algorithm_from_the_config() -> None:
    assert isinstance(_small_epibe_experiment().make_trainer(), EPIBETrainer)


def test_disabled_ebm_gives_a_plain_pibe_run() -> None:
    """The default must leave every existing configuration untouched."""
    config = RunConfig.from_dict(
        {
            "system": {"name": "automatica_n3v2"},
            "data": {"n_trajectories": 4, "n_samples": 21, "horizon": 2.0},
            "basis": {"kind": "fourier", "q": 3, "omega": 1.2566370614359172,
                      "coeff_lo": -0.1, "coeff_hi": 0.1},
            "architecture": {"latent_dim": 4, "encoder_hidden": [8],
                             "decoder_hidden": [8], "head_hidden": [8]},
            "training": {"n_total": 4, "n_par": 2, "n_collocation": 8,
                         "batch_size": 2, "log_every": 0},
        }
    )
    assert config.ebm.enabled is False
    experiment = build_experiment(config)
    assert not isinstance(experiment.bank, EnergyEstimatorBank)
    assert type(experiment.make_trainer()) is PIBETrainer


def test_remark_5_ordering_is_enforced() -> None:
    with pytest.raises(ValueError, match="0 < n_ebm < n_par"):
        EBMConfig(enabled=True, n_ebm=500).validate(n_par=100, n_total=200)
    EBMConfig(enabled=True, n_ebm=50).validate(n_par=100, n_total=200)


def test_epibe_trainer_rejects_a_bank_without_energy_models() -> None:
    config = RunConfig.from_dict(
        {
            "system": {"name": "automatica_n3v2"},
            "data": {"n_trajectories": 4, "n_samples": 21, "horizon": 2.0},
            "basis": {"kind": "fourier", "q": 3, "omega": 1.2566370614359172,
                      "coeff_lo": -0.1, "coeff_hi": 0.1},
            "architecture": {"latent_dim": 4, "encoder_hidden": [8],
                             "decoder_hidden": [8], "head_hidden": [8]},
            "training": {"n_total": 4, "n_par": 2, "n_collocation": 8,
                         "batch_size": 2, "log_every": 0},
        }
    )
    experiment = build_experiment(config)
    with pytest.raises(TypeError, match="requires an EnergyEstimatorBank"):
        EPIBETrainer(experiment.bank, train_data=experiment.train_data,
                     t_coll=experiment.t_coll, config=config.training)


def test_a_short_epibe_run_touches_every_phase_and_moves_both_blocks() -> None:
    """End to end: Algorithm 2 runs and updates the PINN and the EBM."""
    import copy

    experiment = _small_epibe_experiment(n_fit=1)
    before = copy.deepcopy(experiment.bank.ebm(2).state_dict())
    trainer = experiment.make_trainer()
    trainer.config.log_every = 1  # record every iteration, so all phases appear
    history = trainer.train()

    phases = {record.mode for record in history.iterations}
    assert phases == {"warmup", "fit", "local", "global"}

    after = experiment.bank.ebm(2).state_dict()
    moved = max(
        float((after[k] - before[k]).abs().max())
        for k in after
        if after[k].is_floating_point()
    )
    assert moved > 0.0
    assert math.isfinite(experiment.bank.noise_mean())


def test_centered_warmup_is_the_quadratic_term_with_the_location_profiled_out() -> None:
    """Proposition 1: minimizing R_D over m leaves the residual's variance.

    The plain warm-up pins the location at zero, which is the assumption EPIBE
    exists to drop; the centered one leaves it to the physics term.
    """
    experiment = _small_epibe_experiment(cells=[2], centered_warmup=True)
    bank = experiment.bank
    bank.use_energy = False
    data = experiment.train_data
    with torch.enable_grad():
        outputs = bank(target_cell=bank.final_index, y=data.y, t_data=data.t,
                       t_coll=experiment.t_coll, mode=Mode.GLOBAL, create_graph=False)

    residual = bank.consistency_residual(2, data.y, outputs)
    loss = bank.cell_loss(2, data.y, experiment.t_coll, outputs, lam=1.0)
    assert float(loss.data) == pytest.approx(float(residual.var(unbiased=False)))

    # Invariant under a constant shift of the estimate -- that is the point.
    shifted = residual + 0.37
    assert float(shifted.var(unbiased=False)) == pytest.approx(
        float(residual.var(unbiased=False))
    )
    # Cells that kept the quadratic consistency term are unaffected.
    downstream = bank.cell_loss(3, data.y, experiment.t_coll, outputs, lam=1.0)
    r3 = bank.consistency_residual(3, data.y, outputs)
    assert float(downstream.data) == pytest.approx(float((r3**2).mean()))


# ----------------------------------------------------------------------
# sensor-offset location: profiled, offset parameter, amortized search
# ----------------------------------------------------------------------


def test_location_options_are_validated() -> None:
    with pytest.raises(ValueError, match="location"):
        EBMConfig(enabled=True, location="anywhere").validate(n_par=4, n_total=6)
    with pytest.raises(ValueError, match="offset_score"):
        EBMConfig(enabled=True, offset_score="guess").validate(n_par=4, n_total=6)
    with pytest.raises(ValueError, match="offset_window"):
        EBMConfig(enabled=True, offset_window=[1.0, 0.0]).validate(n_par=4, n_total=6)


def test_profiled_energy_term_is_invariant_to_a_common_shift() -> None:
    """The data term cannot see the location; that is left to the physics."""
    experiment = _small_epibe_experiment(cells=[2], location="profiled")
    bank = experiment.bank
    bank.use_energy = True
    bank.eval()
    data = experiment.train_data
    with torch.enable_grad():
        outputs = bank(target_cell=2, y=data.y, t_data=data.t,
                       t_coll=experiment.t_coll, mode=Mode.GLOBAL, create_graph=False)
        base = bank.cell_loss(2, data.y, experiment.t_coll, outputs, lam=1.0)
        moved = bank.cell_loss(2, data.y + 0.03, experiment.t_coll, outputs, lam=1.0)
    assert float(moved.data) == pytest.approx(float(base.data), rel=1e-10)


def test_offset_parameter_shifts_only_the_measured_reconstruction() -> None:
    experiment = _small_epibe_experiment(cells=[2], offset_parameter=True)
    bank = experiment.bank
    data = experiment.train_data
    kwargs = dict(t_data=data.t, t_coll=experiment.t_coll, mode=Mode.GLOBAL, create_graph=False)
    with torch.enable_grad():
        before = bank(target_cell=bank.final_index, y=data.y, **kwargs)
        with torch.no_grad():
            bank.offset.fill_(0.25)
        after = bank(target_cell=bank.final_index, y=data.y, **kwargs)
    assert torch.allclose(after[2].x_prev_data, before[2].x_prev_data + 0.25)
    assert torch.allclose(after[2].x_prev_dot_coll, before[2].x_prev_dot_coll)
    assert torch.allclose(after[2].x_new_data, before[2].x_new_data)


def test_amortized_bank_reads_out_the_offset_corrected_measurement() -> None:
    experiment = _small_epibe_experiment(cells=[2], location="amortized")
    bank = experiment.bank
    data = experiment.train_data
    bank.location.fill_(0.3)
    stored = bank.estimate(data.y, data.t, experiment.t_coll)
    plain = EstimatorBank.estimate(bank, data.y - 0.3, data.t, experiment.t_coll)
    explicit = bank.estimate(data.y, data.t, experiment.t_coll, offset=0.3)
    assert torch.equal(stored.x, plain.x) and torch.equal(explicit.x, plain.x)
    assert "location" in bank.state_dict()


def test_offset_buffers_do_not_change_existing_checkpoints() -> None:
    """A run without the new options must keep loading old state dicts strictly."""
    bank = _small_epibe_experiment(cells=[2]).bank
    keys = set(bank.state_dict())
    assert "location" not in keys and "residual_mean" not in keys and "offset" not in keys


def test_simulated_output_matches_the_data_generator() -> None:
    from pibe.data.disturbance import BasisDisturbance
    from pibe.data.simulate import rk4_integrate
    from pibe.eval.offset_shooting import simulate_output

    experiment = _small_epibe_experiment()
    data = experiment.data
    x0 = data.x[:, 0, :]
    a = data.coefficients
    theta = experiment.system.theta_true
    ours = simulate_output(experiment.system, experiment.basis, x0, theta, a, data.t, substeps=8)
    for i in range(len(data)):
        reference = rk4_integrate(experiment.system, data.t, x0[i:i + 1], theta.reshape(1, -1),
                                  BasisDisturbance(experiment.basis, a[i]), substeps=8)[0, :, 0]
        assert torch.allclose(ours[i], reference, atol=1e-10)


def test_amortization_feeds_offset_measurements_only_while_amortizing() -> None:
    experiment = _small_epibe_experiment(cells=[2], location="amortized",
                                         offset_window=[0.5, 0.6])
    trainer = experiment.make_trainer()
    y = experiment.train_data.y
    assert torch.equal(trainer.measurement_for_step(y), y)
    trainer._amortizing = True
    shift = y - trainer.measurement_for_step(y)
    assert float(shift.min()) >= 0.5 - 1e-12 and float(shift.max()) <= 0.6 + 1e-12
    # One offset per trajectory, constant along it.
    assert torch.allclose(shift, shift[:, :1].expand_as(shift))
