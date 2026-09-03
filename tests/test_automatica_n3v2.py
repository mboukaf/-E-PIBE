r"""Third-order benchmark, revision 2 --- identifiability-hardened."""

from __future__ import annotations

import math

import pytest
import torch

from pibe.basis import FourierBasis
from pibe.core.cell import CellOutput
from pibe.core.residuals import final_residual, state_residual
from pibe.data.disturbance import BasisDisturbance
from pibe.data.simulate import check_admissible, rk4_integrate, uniform_grid
from pibe.systems.examples.automatica_n3 import AutomaticaN3System
from pibe.systems.examples.automatica_n3v2 import AutomaticaN3V2System
from pibe.utils.seeding import make_generator

OMEGA = 2.0 * math.pi / 5.0
HORIZON = 20.0
COEFF_BOUNDS = torch.tensor(
    [[-0.18, 0.18], [-0.18, 0.18], [-0.12, 0.12]], dtype=torch.float64
)


def reference_rhs(x, theta, d):
    """The revised right-hand side, transcribed independently."""
    x1, x2, x3 = x[..., 0], x[..., 1], x[..., 2]
    th1, th2 = theta[..., 0], theta[..., 1]
    dx1 = x2 - 0.35 * torch.tanh(x1) + th1 * (1.3 + 0.7 * torch.cos(x1))
    dx2 = (
        x3 - 0.30 * torch.tanh(x2) + 0.10 * torch.sin(x1)
        + th2 * (1.2 + 0.6 * torch.cos(2.0 * x2))
    )
    dx3 = (
        -0.648 * x1 - 2.34 * x2 - 2.7 * x3
        + 0.15 * torch.sin(x1) + 0.08 * torch.tanh(x1 * x2)
        + 0.08 * th1 * torch.sin(x2) + 0.06 * th2 * torch.tanh(x3) + d
    )
    return torch.stack([dx1, dx2, dx3], dim=-1)


def test_vector_field_matches_the_specification() -> None:
    system = AutomaticaN3V2System()
    torch.manual_seed(0)
    x = torch.randn(64, 3, dtype=torch.float64)
    theta = torch.randn(64, 2, dtype=torch.float64) * 0.25
    d = torch.randn(64, dtype=torch.float64) * 0.1
    assert torch.allclose(
        system.vector_field(x, theta, d), reference_rhs(x, theta, d), atol=1e-14
    )


def test_only_the_sensitivities_and_X1_differ_from_revision_1() -> None:
    """`f_3`, the poles and every other range are untouched."""
    a, b = AutomaticaN3System(), AutomaticaN3V2System()
    assert a.companion.tolist() == b.companion.tolist()
    assert a.x0_bounds.tolist() == b.x0_bounds.tolist()
    assert a.theta_bounds.tolist() == b.theta_bounds.tolist()
    assert torch.equal(a.theta_true, b.theta_true)
    # f_3 is identical.
    torch.manual_seed(0)
    x = torch.randn(32, 3, dtype=torch.float64)
    th = torch.randn(32, 2, dtype=torch.float64) * 0.25
    assert torch.allclose(a.f_last(x, th), b.f_last(x, th), atol=1e-14)
    # X_1 is widened; the others are not.
    assert b.state_bounds[0].tolist() == [-2.5, 2.5]
    assert b.state_bounds[1:].tolist() == a.state_bounds[1:].tolist()


def simulate(system, n_traj=32, seed=5):
    basis = FourierBasis(q=3, omega=OMEGA, t_start=0.0, t_end=HORIZON)
    g = make_generator(seed)
    u = torch.rand(3, generator=g, dtype=torch.float64)
    a = COEFF_BOUNDS[:, 0] + u * (COEFF_BOUNDS[:, 1] - COEFF_BOUNDS[:, 0])
    dist = BasisDisturbance(basis, a)
    t = uniform_grid(0.0, HORIZON, 201)
    x0 = system.sample_x0(n_traj, generator=g)
    x = rk4_integrate(system, t, x0, system.theta_true, dist, substeps=8)
    return basis, dist, t, x


def test_trajectories_stay_inside_the_widened_ranges() -> None:
    system = AutomaticaN3V2System()
    _, _, _, x = simulate(system, n_traj=48)
    assert bool((check_admissible(system, x) > 0).all())


def test_sensitivity_varies_far_more_than_in_revision_1() -> None:
    r"""The point of the revision: :math:`\partial_\theta f` must *vary*.

    Assumption 6 only asks for :math:`\gamma_k > 0`.  A sensitivity bounded
    away from zero but nearly constant satisfies it while leaving the
    state-parameter split practically unidentifiable, because the compensating
    state shift is then constant --- exactly what the final residual's
    companion row cancels.
    """
    old, new = AutomaticaN3System(), AutomaticaN3V2System()
    _, _, _, x_old = simulate(old, n_traj=32)
    _, _, _, x_new = simulate(new, n_traj=32)

    def spread(system, x, j, coord):
        s = system.parameter_sensitivity(j, x.reshape(-1, 3))
        return float(s.max() / s.min()) - 1.0

    # Measured on this 32-trajectory sample: 8.1% for revision 1, 47.0% for
    # revision 2.  (Over a 300-trial ensemble, which explores more of the state
    # space, the figures are 5% and 101%.)
    old_var = spread(old, x_old, 1, 0)
    new_var = spread(new, x_new, 1, 0)
    assert old_var < 0.12, f"revision 1's s_1 should be near-constant, got {old_var:.3f}"
    assert new_var > 0.40, f"revision 2's s_1 should vary strongly, got {new_var:.3f}"
    assert new_var > 4.0 * old_var


def test_sensitivities_key_off_different_coordinates() -> None:
    """`s_1` depends on `x_1` only, `s_2` on `x_2` only.

    Shared dependence lets the two compensations re-align; measured, that
    recovers most of the degeneracy (3.2x versus 25.7x).
    """
    system = AutomaticaN3V2System()
    torch.manual_seed(0)
    x = torch.randn(64, 3, dtype=torch.float64) * 0.4
    perturbed = x.clone()
    perturbed[:, 1] += 0.3  # move x_2 only
    assert torch.allclose(
        system.parameter_sensitivity(1, x),
        system.parameter_sensitivity(1, perturbed), atol=1e-14
    ), "s_1 must not depend on x_2"

    perturbed = x.clone()
    perturbed[:, 0] += 0.3  # move x_1 only
    assert torch.allclose(
        system.parameter_sensitivity(2, x),
        system.parameter_sensitivity(2, perturbed), atol=1e-14
    ), "s_2 must not depend on x_1"


def test_excitation_floors_hold_on_realized_trajectories() -> None:
    system = AutomaticaN3V2System()
    _, _, _, x = simulate(system, n_traj=48)
    flat = x.reshape(-1, 3)
    for j, floor in enumerate(system.sensitivity_floors(), start=1):
        assert float(system.parameter_sensitivity(j, flat).min()) >= floor - 1e-6
    # Both exceed revision 1's 0.36 / 0.49.
    assert system.excitation_floors()[0] > 0.36
    assert system.excitation_floors()[1] > 0.49


def test_residuals_vanish_on_the_true_trajectory() -> None:
    system = AutomaticaN3V2System()
    basis, dist, t, x = simulate(system, n_traj=6)
    p, n = x.shape[0], x.shape[1]
    x_dot = system.vector_field(x, system.theta_true, dist(t).expand(p, n))
    theta = system.theta_true.expand(p, 2)

    def cell(prev, new, head):
        return CellOutput(
            latent=torch.zeros(p, 1, dtype=x.dtype),
            x_prev_data=x[..., prev - 1], x_prev_coll=x[..., prev - 1],
            x_prev_dot_coll=x_dot[..., prev - 1],
            x_new_data=None if new is None else x[..., new - 1],
            x_new_coll=None if new is None else x[..., new - 1],
            head=head,
        )

    outs = {k: cell(k - 1, k, theta[:, [k - 2]]) for k in (2, 3)}
    outs[4] = cell(3, None, dist.coefficients.expand(p, 3))
    for k in (2, 3):
        assert float(state_residual(system, k, outs[k], outs).abs().max()) < 1e-9
    assert float(final_residual(system, basis, outs[4], outs, t).abs().max()) < 1e-9
