"""Data generation: integrator accuracy, noise law, admissibility, splitting."""

from __future__ import annotations

import math

import pytest
import torch

from pibe.basis.bspline import BSplineBasis
from pibe.data.dataset import TrajectoryBatcher, generate_dataset
from pibe.data.disturbance import BasisDisturbance, sample_coefficients
from pibe.data.noise import NoiseFree, TruncatedGaussianNoise
from pibe.data.simulate import (
    check_admissible,
    fill_distance,
    rk4_integrate,
    uniform_grid,
)
from pibe.systems.examples.sin_chain import SinChainSystem
from pibe.utils.seeding import make_generator


# ----------------------------------------------------------------------
# integrator
# ----------------------------------------------------------------------


def test_rk4_is_fourth_order() -> None:
    """Halving the step must cut the error by roughly 16.

    Validates the integrator generically, without needing a closed-form
    solution for the nonlinear chain.
    """
    system = SinChainSystem(n=3)
    t = uniform_grid(0.0, 2.0, 9)
    x0 = torch.tensor([[0.3, -0.2, 0.1]], dtype=torch.float64)

    reference = rk4_integrate(system, t, x0, system.theta_true, substeps=256)
    coarse = rk4_integrate(system, t, x0, system.theta_true, substeps=4)
    fine = rk4_integrate(system, t, x0, system.theta_true, substeps=8)

    error_coarse = float((coarse - reference).abs().max())
    error_fine = float((fine - reference).abs().max())
    order = math.log2(error_coarse / error_fine)
    assert 3.5 < order < 4.5, f"observed convergence order {order:.2f}, expected ~4"


def test_rk4_reproduces_a_linear_solution() -> None:
    r"""With :math:`\kappa = 0` and :math:`\theta = 0` the chain is linear.

    The companion matrix has all poles at :math:`-1`, so the solution is
    :math:`x(t) = e^{At}x_0` and can be checked against ``matrix_exp``.
    """
    n = 3
    system = SinChainSystem(n=n, kappa=0.0, theta_true=[0.0] * (n - 1))
    matrix = torch.zeros(n, n, dtype=torch.float64)
    for j in range(n - 1):
        matrix[j, j + 1] = 1.0
    matrix[n - 1] = -system.companion

    t = uniform_grid(0.0, 3.0, 13)
    x0 = torch.tensor([[0.4, -0.3, 0.2]], dtype=torch.float64)
    numeric = rk4_integrate(system, t, x0, system.theta_true, substeps=64)

    for index, time in enumerate(t.tolist()):
        analytic = torch.matrix_exp(matrix * time) @ x0[0]
        assert torch.allclose(numeric[0, index], analytic, atol=1e-9), (
            f"mismatch at t={time:g}"
        )


def test_rk4_rejects_non_increasing_grid() -> None:
    system = SinChainSystem(n=2)
    bad = torch.tensor([0.0, 0.5, 0.5], dtype=torch.float64)
    with pytest.raises(ValueError, match="strictly increasing"):
        rk4_integrate(system, bad, torch.zeros(1, 2, dtype=torch.float64), system.theta_true)


def test_uniform_grid_includes_endpoints() -> None:
    """Section 4.2 requires the grids to contain the endpoints of ``[0, T]``."""
    grid = uniform_grid(0.0, 5.0, 11)
    assert float(grid[0]) == 0.0 and float(grid[-1]) == 5.0
    assert fill_distance(grid) == pytest.approx(0.25)


# ----------------------------------------------------------------------
# noise
# ----------------------------------------------------------------------


def test_truncated_gaussian_respects_its_support() -> None:
    r"""Every realization satisfies :math:`\|\omega\|_\infty \le \bar w`."""
    noise = TruncatedGaussianNoise(sigma=1.0, bound=1.5)
    sample = noise.sample((200_000,), generator=make_generator(0))
    assert float(sample.abs().max()) <= noise.bound


def test_truncated_gaussian_is_zero_mean() -> None:
    """Symmetric truncation preserves the zero mean, so the data term is unbiased."""
    noise = TruncatedGaussianNoise(sigma=0.5, bound=1.0)
    sample = noise.sample((400_000,), generator=make_generator(1))
    assert abs(float(sample.mean())) < 5e-3


def test_truncated_gaussian_variance_matches_the_closed_form() -> None:
    noise = TruncatedGaussianNoise(sigma=1.0, bound=2.0)
    sample = noise.sample((400_000,), generator=make_generator(2))
    assert float(sample.var(unbiased=False)) == pytest.approx(noise.variance, rel=2e-2)
    assert noise.variance < noise.sigma**2, "truncation must reduce the variance"


def test_noise_free_model() -> None:
    assert NoiseFree().bound == 0.0
    assert float(NoiseFree().sample((10,)).abs().max()) == 0.0


# ----------------------------------------------------------------------
# admissibility, Assumption 1
# ----------------------------------------------------------------------


def test_check_admissible_accepts_bounded_trajectories() -> None:
    system = SinChainSystem(n=3, state_bound=8.0, x0_bound=0.4)
    t = uniform_grid(0.0, 4.0, 41)
    x0 = system.sample_x0(6, generator=make_generator(0))
    x = rk4_integrate(system, t, x0, system.theta_true, substeps=16)
    margin = check_admissible(system, x)
    assert bool((margin > 0).all())


def test_check_admissible_rejects_escaping_trajectories() -> None:
    """A box too tight for the trajectory violates Assumption 1 and must raise."""
    system = SinChainSystem(n=3, state_bound=0.05, x0_bound=0.04)
    t = uniform_grid(0.0, 4.0, 41)
    x = torch.ones(2, 41, 3, dtype=torch.float64) * 5.0
    with pytest.raises(ValueError, match="admissible set"):
        check_admissible(system, x)


# ----------------------------------------------------------------------
# dataset
# ----------------------------------------------------------------------


def build_dataset(n_traj: int = 10, n_samples: int = 24):
    system = SinChainSystem(n=3)
    basis = BSplineBasis(q=5, degree=3, t_start=0.0, t_end=3.0)
    generator = make_generator(0)
    disturbance = BasisDisturbance(
        basis, sample_coefficients(basis, (-2.0, 2.0), generator=generator)
    )
    t = uniform_grid(0.0, 3.0, n_samples)
    return generate_dataset(
        system=system,
        t_grid=t,
        n_trajectories=n_traj,
        noise=TruncatedGaussianNoise(sigma=0.02),
        disturbance=disturbance,
        generator=generator,
    )


def test_dataset_shapes_and_output_equation() -> None:
    r"""``y = x_1 + omega`` with the noise inside its support."""
    data = build_dataset()
    assert data.x.shape == (10, 24, 3)
    assert data.y.shape == (10, 24)
    # d is per trajectory: Eq. (2)'s ``a`` is an unknown to be inferred from y,
    # so a shared disturbance would let the coefficient head learn a constant.
    assert data.d.shape == (10, 24)
    assert data.theta.shape == (10, 2)
    residual = (data.y - data.x[..., 0]).abs().max()
    assert float(residual) <= 3.0 * 0.02 + 1e-12


def test_split_is_disjoint_and_covers_everything() -> None:
    r""":math:`\Omega^{train} \cap \Omega^{test} = \emptyset` (Section 5.1, step 3)."""
    data = build_dataset(n_traj=10)
    train, val = data.split(0.7, generator=make_generator(3))
    assert len(train) + len(val) == len(data)
    assert len(train) == 7 and len(val) == 3
    # No trajectory may appear in both halves.
    train_rows = {tuple(row.tolist()) for row in train.y}
    val_rows = {tuple(row.tolist()) for row in val.y}
    assert train_rows.isdisjoint(val_rows)


def test_batcher_respects_batch_size_and_uniqueness() -> None:
    batcher = TrajectoryBatcher(20, batch_size=6, generator=make_generator(0))
    for _ in range(10):
        index = batcher.sample()
        assert index.numel() == 6
        assert index.unique().numel() == 6
        assert int(index.max()) < 20


def test_batcher_full_batch_is_stable() -> None:
    batcher = TrajectoryBatcher(5, batch_size=0)
    assert batcher.is_full_batch
    assert torch.equal(batcher.sample(), torch.arange(5))


def test_generate_dataset_requires_ground_truth_parameters() -> None:
    system = SinChainSystem(n=3)
    system.theta_true = None
    with pytest.raises(ValueError, match="theta_true"):
        generate_dataset(system, uniform_grid(0.0, 1.0, 5), 2)


# ----------------------------------------------------------------------
# per-trajectory disturbances
# ----------------------------------------------------------------------


def test_disturbance_can_differ_per_trajectory() -> None:
    r"""Each trajectory carries its own ``a``, as Eq. (2) intends.

    The paper notes that :math:`\hat a^\ell` "may differ between trajectories",
    which only means something if the *true* ``a`` does too.  With one shared
    ``a`` the coefficient head can satisfy the objective by emitting a constant,
    and the disturbance-estimation problem is never posed.
    """
    system = SinChainSystem(n=3)
    basis = BSplineBasis(q=5, degree=3, t_start=0.0, t_end=3.0)
    generator = make_generator(0)
    coefficients = torch.rand(10, 5, generator=generator, dtype=torch.float64) - 0.5
    data = generate_dataset(
        system=system, t_grid=uniform_grid(0.0, 3.0, 24), n_trajectories=10,
        disturbance=BasisDisturbance(basis, coefficients), generator=generator,
    )
    assert data.d.shape == (10, 24)
    assert data.coefficients is not None and data.coefficients.shape == (10, 5)
    # Distinct coefficients must give distinct disturbances, hence distinct states.
    assert float((data.d[0] - data.d[1]).abs().max()) > 1e-6
    assert torch.allclose(data.coefficients, coefficients)


def test_split_carries_matching_coefficients() -> None:
    """A subset's truth must follow it, or the oracle scores the wrong ``a``."""
    system = SinChainSystem(n=3)
    basis = BSplineBasis(q=5, degree=3, t_start=0.0, t_end=3.0)
    generator = make_generator(1)
    coefficients = torch.rand(10, 5, generator=generator, dtype=torch.float64) - 0.5
    data = generate_dataset(
        system=system, t_grid=uniform_grid(0.0, 3.0, 24), n_trajectories=10,
        disturbance=BasisDisturbance(basis, coefficients), generator=generator,
    )
    train, val = data.split(0.7, generator=make_generator(2))
    assert train.coefficients.shape == (7, 5)
    assert val.coefficients.shape == (3, 5)
    # Every row of the split must be a row of the original, paired with its own d.
    for subset in (train, val):
        for i in range(len(subset)):
            match = (data.coefficients == subset.coefficients[i]).all(dim=1).nonzero()
            assert match.numel() == 1
            assert torch.allclose(data.d[int(match)], subset.d[i])


def test_per_trajectory_theta_broadcasts_through_the_vector_field() -> None:
    r"""``theta`` of shape ``(P, n-1)`` must align with the trajectory axis.

    ``x`` may carry a time axis ``(P, M, n)``; a per-trajectory ``theta`` has to
    be inserted before it, not misread as a time axis.
    """
    system = SinChainSystem(n=3)
    torch.manual_seed(0)
    x = torch.randn(4, 7, 3, dtype=torch.float64)
    theta = torch.randn(4, 2, dtype=torch.float64) * 0.2
    d = torch.zeros(4, 7, dtype=torch.float64)
    out = system.vector_field(x, theta, d)
    assert out.shape == (4, 7, 3)
    # Row p must equal the field evaluated with that row's parameters alone.
    for p in range(4):
        single = system.vector_field(x[p], theta[p], d[p])
        assert torch.allclose(out[p], single, atol=1e-14)
