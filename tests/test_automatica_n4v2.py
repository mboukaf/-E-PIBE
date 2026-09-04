r"""Fourth-order benchmark, revision 2 --- identifiability-hardened."""

from __future__ import annotations

import math

import pytest
import torch

from pibe.basis import FourierBasis
from pibe.core.cell import CellOutput
from pibe.core.residuals import final_residual, state_residual
from pibe.data.disturbance import BasisDisturbance
from pibe.data.simulate import check_admissible, rk4_integrate, uniform_grid
from pibe.systems.examples.automatica_n4 import AutomaticaN4System
from pibe.systems.examples.automatica_n4v2 import (
    FREQUENCIES,
    AutomaticaN4V2System,
)
from pibe.utils.seeding import make_generator

OMEGA = 2.0 * math.pi / 5.0
HORIZON = 20.0
COEFF_BOUNDS = torch.tensor(
    [[-0.15, 0.15], [-0.15, 0.15], [-0.12, 0.12], [-0.12, 0.12]], dtype=torch.float64
)


def reference_rhs(x, theta, d):
    """The revised right-hand side, transcribed independently."""
    x1, x2, x3, x4 = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    t1, t2, t3 = theta[..., 0], theta[..., 1], theta[..., 2]
    dx1 = x2 - 0.35 * torch.tanh(x1) + t1 * (1.3 + 0.7 * torch.cos(x1))
    dx2 = (x3 - 0.30 * torch.tanh(x2) + 0.10 * torch.sin(x1)
           + t2 * (1.2 + 0.6 * torch.cos(6.0 * x2)))
    dx3 = (x4 - 0.25 * torch.tanh(x3) + 0.10 * torch.sin(x1 + x2)
           + t3 * (1.15 + 0.55 * torch.cos(10.0 * x3)))
    dx4 = (-0.576 * x1 - 2.736 * x2 - 4.76 * x3 - 3.6 * x4
           + 0.15 * torch.sin(x1) + 0.08 * torch.tanh(x2 * x3)
           + 0.08 * t1 * torch.sin(x2) + 0.06 * t2 * torch.sin(x3)
           + 0.05 * t3 * torch.tanh(x4) + d)
    return torch.stack([dx1, dx2, dx3, dx4], dim=-1)


def test_vector_field_matches_the_specification() -> None:
    system = AutomaticaN4V2System()
    torch.manual_seed(0)
    x = torch.randn(64, 4, dtype=torch.float64)
    theta = torch.randn(64, 3, dtype=torch.float64) * 0.25
    d = torch.randn(64, dtype=torch.float64) * 0.1
    assert torch.allclose(
        system.vector_field(x, theta, d), reference_rhs(x, theta, d), atol=1e-14
    )


def test_only_the_sensitivities_differ_from_revision_1() -> None:
    """`f_4`, the poles and every range are untouched --- including `X_j`."""
    a, b = AutomaticaN4System(), AutomaticaN4V2System()
    assert a.companion.tolist() == b.companion.tolist()
    assert a.state_bounds.tolist() == b.state_bounds.tolist()
    assert a.x0_bounds.tolist() == b.x0_bounds.tolist()
    assert a.theta_bounds.tolist() == b.theta_bounds.tolist()
    assert torch.equal(a.theta_true, b.theta_true)
    torch.manual_seed(0)
    x = torch.randn(32, 4, dtype=torch.float64)
    th = torch.randn(32, 3, dtype=torch.float64) * 0.25
    assert torch.allclose(a.f_last(x, th), b.f_last(x, th), atol=1e-14)


def simulate(system, n_traj=32, seed=7):
    basis = FourierBasis(q=4, omega=OMEGA, t_start=0.0, t_end=HORIZON)
    g = make_generator(seed)
    u = torch.rand(4, generator=g, dtype=torch.float64)
    a = COEFF_BOUNDS[:, 0] + u * (COEFF_BOUNDS[:, 1] - COEFF_BOUNDS[:, 0])
    dist = BasisDisturbance(basis, a)
    t = uniform_grid(0.0, HORIZON, 201)
    x0 = system.sample_x0(n_traj, generator=g)
    x = rk4_integrate(system, t, x0, system.theta_true, dist, substeps=8)
    return basis, dist, t, x


def test_trajectories_stay_inside_the_unchanged_ranges() -> None:
    """Design E needs no wider box than revision 1."""
    system = AutomaticaN4V2System()
    _, _, _, x = simulate(system, n_traj=48)
    assert bool((check_admissible(system, x) > 0).all())


def test_every_sensitivity_varies_far_more_than_in_revision_1() -> None:
    r"""Revision 1's `s_2` and `s_3` were effectively constant (4% and 1%).

    A constant compensating shift is exactly what the final residual's
    companion row annihilates, so near-constant sensitivities are the worst
    case for identifiability even though Assumption 6 is satisfied.
    """
    old, new = AutomaticaN4System(), AutomaticaN4V2System()
    _, _, _, x_old = simulate(old, n_traj=32)
    _, _, _, x_new = simulate(new, n_traj=32)

    def variation(system, x, j):
        s = system.parameter_sensitivity(j, x.reshape(-1, 4))
        return float(s.max() / s.min()) - 1.0

    # Measured on this 32-trajectory sample: revision 1 varies 30% / 19% / 21%,
    # revision 2 varies 118% / 200% / 183%.  (On the narrower trajectories used
    # for the curvature measurement, revision 1's s_2 and s_3 move only 4% and
    # 1% -- the variation depends on how much of the state space is explored,
    # which is precisely why it must be assessed on realized trajectories
    # rather than over the admissible box.)
    for j in (1, 2, 3):
        assert variation(old, x_old, j) < 0.35, f"revision 1 s_{j} varies too much to be the baseline"
        assert variation(new, x_new, j) > 1.0, (
            f"revision 2 s_{j} varies only {variation(new, x_new, j):.2f}"
        )
        assert variation(new, x_new, j) > 3.0 * variation(old, x_old, j)


def test_each_sensitivity_keys_off_its_own_coordinate() -> None:
    """`s_j` depends on `x_j` alone, so the compensations cannot re-align."""
    system = AutomaticaN4V2System()
    torch.manual_seed(0)
    x = torch.randn(64, 4, dtype=torch.float64) * 0.3
    for j in (1, 2, 3):
        for other in range(4):
            if other == j - 1:
                continue
            moved = x.clone()
            moved[:, other] += 0.25
            assert torch.allclose(
                system.parameter_sensitivity(j, x),
                system.parameter_sensitivity(j, moved), atol=1e-14
            ), f"s_{j} must not depend on x_{other + 1}"


def test_frequencies_sweep_enough_phase_for_each_span() -> None:
    r"""A multiplier is useless if the coordinate's span is small.

    `x_3` only ranges over about +-0.4, so `cos(3 x_3)` sweeps 2.4 rad and
    barely moves; the multiplier must make the argument sweep of order
    :math:`2\pi` across the *observed* span.
    """
    system = AutomaticaN4V2System()
    _, _, _, x = simulate(system, n_traj=48)
    flat = x.reshape(-1, 4)
    for j, m in enumerate(FREQUENCIES):
        span = float(flat[:, j].max() - flat[:, j].min())
        assert m * span > 2.5, (
            f"s_{j + 1}: multiplier {m} over span {span:.2f} sweeps only "
            f"{m * span:.2f} rad"
        )


def test_residuals_vanish_on_the_true_trajectory() -> None:
    system = AutomaticaN4V2System()
    basis, dist, t, x = simulate(system, n_traj=6)
    p, n = x.shape[0], x.shape[1]
    x_dot = system.vector_field(x, system.theta_true, dist(t).expand(p, n))
    theta = system.theta_true.expand(p, 3)

    def cell(prev, new, head):
        return CellOutput(
            latent=torch.zeros(p, 1, dtype=x.dtype),
            x_prev_data=x[..., prev - 1], x_prev_coll=x[..., prev - 1],
            x_prev_dot_coll=x_dot[..., prev - 1],
            x_new_data=None if new is None else x[..., new - 1],
            x_new_coll=None if new is None else x[..., new - 1],
            head=head,
        )

    outs = {k: cell(k - 1, k, theta[:, [k - 2]]) for k in (2, 3, 4)}
    outs[5] = cell(4, None, dist.coefficients.expand(p, 4))
    for k in (2, 3, 4):
        assert float(state_residual(system, k, outs[k], outs).abs().max()) < 1e-9
    assert float(final_residual(system, basis, outs[5], outs, t).abs().max()) < 1e-9
