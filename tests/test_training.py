"""Algorithm 1: phase schedule, freezing, and an end-to-end training run."""

from __future__ import annotations

import pytest
import torch

from pibe.config import RunConfig, TrainingConfig
from pibe.core.bank import Mode
from pibe.experiment import build_experiment
from pibe.training.phases import apply_phase, mode_for_iteration


# ----------------------------------------------------------------------
# schedule
# ----------------------------------------------------------------------


def test_mode_switches_at_the_cutoff() -> None:
    """``i < N_par`` is local pre-training, the rest is fine-tuning."""
    n_par = 5
    assert [mode_for_iteration(i, n_par) for i in range(4)] == [Mode.LOCAL] * 4
    assert mode_for_iteration(4, n_par) is Mode.LOCAL
    assert mode_for_iteration(5, n_par) is Mode.GLOBAL
    assert mode_for_iteration(9, n_par) is Mode.GLOBAL


def test_training_config_enforces_the_cutoff_inequality() -> None:
    """Algorithm 1's input requires ``0 < N_par < N_tot``."""
    with pytest.raises(ValueError, match="0 < n_par < n_total"):
        TrainingConfig(n_total=100, n_par=100)
    with pytest.raises(ValueError, match="0 < n_par < n_total"):
        TrainingConfig(n_total=100, n_par=0)


# ----------------------------------------------------------------------
# freezing
# ----------------------------------------------------------------------


def small_experiment(**overrides):
    payload = {
        "system": {"name": "sin_chain", "kwargs": {"n": 3}},
        "data": {
            "n_trajectories": 8,
            "n_samples": 24,
            "horizon": 3.0,
            "noise_sigma": 0.01,
            "seed": 1,
        },
        "basis": {"q": 5, "degree": 3},
        "architecture": {
            "latent_dim": 8,
            "encoder_hidden": [32],
            "decoder_hidden": [32, 32],
            "head_hidden": [32],
        },
        "training": {
            "n_total": 8,
            "n_par": 6,
            "n_collocation": 20,
            "batch_size": 4,
            "log_every": 0,
        },
        "device": "cpu",
    }
    for section, values in overrides.items():
        payload[section] = {**payload.get(section, {}), **values}
    return build_experiment(RunConfig.from_dict(payload))


def test_apply_phase_local_freezes_upstream() -> None:
    """Line 6: only the current cell is trainable during pre-training."""
    experiment = small_experiment()
    bank = experiment.bank
    active = apply_phase(bank, target_cell=3, mode=Mode.LOCAL)

    assert all(not p.requires_grad for p in bank.cell_parameters(2))
    assert all(p.requires_grad for p in bank.cell_parameters(3))
    assert all(not p.requires_grad for p in bank.cell_parameters(4))
    assert len(active) == len(list(bank.cell_parameters(3)))


def test_apply_phase_global_unfreezes_the_prefix() -> None:
    """Line 11: cells ``2..k`` become trainable, later cells stay frozen."""
    experiment = small_experiment()
    bank = experiment.bank
    active = apply_phase(bank, target_cell=3, mode=Mode.GLOBAL)

    assert all(p.requires_grad for p in bank.cell_parameters(2))
    assert all(p.requires_grad for p in bank.cell_parameters(3))
    assert all(not p.requires_grad for p in bank.cell_parameters(4))
    assert len(active) == len(list(bank.parameters_upto(3)))


def test_local_phase_leaves_upstream_weights_untouched() -> None:
    """A full local step must not move a single upstream parameter."""
    experiment = small_experiment()
    bank = experiment.bank
    trainer = experiment.make_trainer()

    before = [p.detach().clone() for p in bank.cell_parameters(2)]
    active = apply_phase(bank, target_cell=3, mode=Mode.LOCAL)
    optimizer = torch.optim.Adam(active, lr=1e-2)
    trainer.step(3, Mode.LOCAL, optimizer, active)

    after = list(bank.cell_parameters(2))
    for old, new in zip(before, after):
        assert torch.equal(old, new), "an upstream weight changed during pre-training"


def test_global_phase_moves_upstream_weights() -> None:
    """Fine-tuning must actually reach the upstream cells."""
    experiment = small_experiment()
    bank = experiment.bank
    trainer = experiment.make_trainer()

    before = [p.detach().clone() for p in bank.cell_parameters(2)]
    active = apply_phase(bank, target_cell=3, mode=Mode.GLOBAL)
    optimizer = torch.optim.Adam(active, lr=1e-2)
    trainer.step(3, Mode.GLOBAL, optimizer, active)

    after = list(bank.cell_parameters(2))
    assert any(not torch.equal(old, new) for old, new in zip(before, after)), (
        "no upstream weight moved during end-to-end fine-tuning"
    )


# ----------------------------------------------------------------------
# end to end
# ----------------------------------------------------------------------


def test_training_reduces_the_first_cell_objective() -> None:
    """The measurement fit of cell 2 must improve over a short run."""
    experiment = small_experiment(
        training={"n_total": 220, "n_par": 200, "log_every": 10, "lr_local": 3e-3}
    )
    trainer = experiment.make_trainer()
    trainer.train_cell(2)

    records = trainer.history.for_cell(2)
    assert len(records) >= 5
    start = records[0].local
    end = min(record.local for record in records[-3:])
    assert end < 0.5 * start, (
        f"cell 2's local loss barely moved: {start:.3e} -> {end:.3e}"
    )


def test_full_run_produces_finite_estimates() -> None:
    """Every cell trains and the bank yields usable estimates and ``d_hat``."""
    experiment = small_experiment(
        training={"n_total": 12, "n_par": 8, "log_every": 0}
    )
    trainer = experiment.make_trainer()
    trainer.train()

    estimates = experiment.bank.estimate(
        experiment.val_data.y, experiment.val_data.t, experiment.t_coll
    )
    for name, tensor in [
        ("x", estimates.x),
        ("theta", estimates.theta),
        ("a", estimates.a),
        ("d", estimates.d),
    ]:
        assert torch.isfinite(tensor).all(), f"{name} contains non-finite values"

    assert estimates.d.shape == experiment.val_data.y.shape
    assert len(trainer.history.validation) == len(list(experiment.bank.cell_indices))


def test_validation_does_not_leak_gradients() -> None:
    """``evaluate`` returns detached scalars despite needing grad mode."""
    experiment = small_experiment()
    trainer = experiment.make_trainer()
    loss = trainer.evaluate(2, experiment.val_data)
    assert not loss.local.requires_grad
    assert torch.isfinite(loss.local)


# ----------------------------------------------------------------------
# local-only ablation
# ----------------------------------------------------------------------


def test_local_only_pins_every_iteration_to_the_local_regime() -> None:
    """The ablation never enters the global phase, whatever ``n_par`` says."""
    for iteration in (0, 5, 500, 10_000):
        assert mode_for_iteration(iteration, n_par=5, local_only=True) is Mode.LOCAL


def test_local_only_waives_the_algorithm_1_cutoff_inequality() -> None:
    """``n_par`` is meaningless without a global phase, so it is not validated."""
    config = TrainingConfig(n_total=100, n_par=100, local_only=True)
    assert config.local_only
    # ... but the inequality is still enforced for the standard schedule.
    with pytest.raises(ValueError, match="0 < n_par < n_total"):
        TrainingConfig(n_total=100, n_par=100, local_only=False)


def test_local_only_never_moves_upstream_weights() -> None:
    """The point of the ablation: upstream cells stay exactly as trained.

    Under the global loss (27) the sole measurement anchor, cell 2's data term
    (23), is down-weighted by ``exp(-(k-2)/k)`` while consistency terms keep
    weight 1, so the chain can drift collectively.  Freezing upstream removes
    that freedom entirely.
    """
    experiment = small_experiment(training={"n_total": 6, "n_par": 3, "local_only": True})
    bank = experiment.bank
    trainer = experiment.make_trainer()

    trainer.train_cell(2)
    before = [p.detach().clone() for p in bank.cell_parameters(2)]
    trainer.train_cell(3)
    after = list(bank.cell_parameters(2))

    for old, new in zip(before, after):
        assert torch.equal(old, new), (
            "cell 2 changed while cell 3 was training under the local-only ablation"
        )


def test_local_only_run_completes_and_estimates() -> None:
    experiment = small_experiment(
        training={"n_total": 10, "n_par": 5, "local_only": True, "log_every": 0}
    )
    trainer = experiment.make_trainer()
    trainer.train()
    assert all(record.mode == "local" for record in trainer.history.iterations)

    estimates = experiment.bank.estimate(
        experiment.val_data.y, experiment.val_data.t, experiment.t_coll
    )
    assert torch.isfinite(estimates.d).all()


def test_final_global_iters_extends_only_the_last_cell() -> None:
    """The extra budget lands on cell ``n+1``, in the global regime."""
    experiment = small_experiment(
        training={"n_total": 6, "n_par": 3, "final_global_iters": 8, "log_every": 1}
    )
    trainer = experiment.make_trainer()
    final = experiment.bank.final_index

    trainer.train_cell(2)
    trainer.train_cell(final)

    assert len(trainer.history.for_cell(2)) == 6
    assert len(trainer.history.for_cell(final)) == 14  # 6 + 8
    extra = [r for r in trainer.history.for_cell(final) if r.iteration >= 6]
    assert extra and all(r.mode == "global" for r in extra), (
        "the extra final-cell iterations must all be end-to-end"
    )


def test_final_global_iters_ignored_under_local_only() -> None:
    """With no global phase there is nothing to extend."""
    experiment = small_experiment(
        training={
            "n_total": 5, "n_par": 3, "final_global_iters": 7,
            "local_only": True, "log_every": 1,
        }
    )
    trainer = experiment.make_trainer()
    trainer.train_cell(experiment.bank.final_index)
    assert len(trainer.history.for_cell(experiment.bank.final_index)) == 5


def test_joint_mode_trains_every_cell_at_once() -> None:
    """The whole bank is optimized against L^(n+1)_Tot from the first step."""
    experiment = small_experiment(
        training={"n_total": 8, "n_par": 1, "joint": True, "log_every": 1}
    )
    bank = experiment.bank
    trainer = experiment.make_trainer()

    before = {k: [p.detach().clone() for p in bank.cell_parameters(k)] for k in bank.cell_indices}
    trainer.train()

    # Only the final cell appears in the history: it *is* the joint objective.
    assert {r.cell for r in trainer.history.iterations} == {bank.final_index}
    # ... yet every cell's weights moved.
    for k in bank.cell_indices:
        after = list(bank.cell_parameters(k))
        assert any(not torch.equal(o, n) for o, n in zip(before[k], after)), (
            f"cell {k} was not trained under joint mode"
        )


def test_joint_and_local_only_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        TrainingConfig(n_total=10, n_par=5, joint=True, local_only=True)
