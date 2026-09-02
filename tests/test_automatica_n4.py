r"""The fourth-order benchmark system.

The central test transcribes the specification's right-hand side *independently*
of the implementation and compares the two.  A typo in a coefficient or a
misplaced state index changes the trajectory and is caught here rather than
surfacing as an unexplained accuracy loss during training.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from pibe.basis import FourierBasis
from pibe.core.cell import CellOutput
from pibe.core.residuals import final_residual, state_residual
from pibe.data.disturbance import BasisDisturbance, ChirpRemainder
from pibe.data.simulate import check_admissible, rk4_integrate, uniform_grid
from pibe.systems.examples.automatica_n4 import (
    COMPANION,
    AutomaticaN4System,
)
from pibe.utils.seeding import make_generator

OMEGA = 2.0 * math.pi / 5.0
HORIZON = 20.0
COEFF_BOUNDS = torch.tensor(
    [[-0.15, 0.15], [-0.15, 0.15], [-0.12, 0.12], [-0.12, 0.12]], dtype=torch.float64
)


def reference_rhs(x: torch.Tensor, theta: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """The specification's right-hand side, transcribed independently."""
    x1, x2, x3, x4 = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    th1, th2, th3 = theta[..., 0], theta[..., 1], theta[..., 2]

    dx1 = x2 - 0.35 * torch.tanh(x1) + th1 * (0.8 + 0.2 * torch.cos(x1))
    dx2 = (
        x3
        - 0.30 * torch.tanh(x2)
        + 0.10 * torch.sin(x1)
        + th2 * (0.9 + 0.1 * torch.sin(x1) + 0.1 * torch.cos(x2))
    )
    dx3 = (
        x4
        - 0.25 * torch.tanh(x3)
        + 0.10 * torch.sin(x1 + x2)
        + th3 * (1.0 + 0.1 * torch.sin(x1 + x3))
    )
    dx4 = (
        -0.576 * x1
        - 2.736 * x2
        - 4.76 * x3
        - 3.6 * x4
        + 0.15 * torch.sin(x1)
        + 0.08 * torch.tanh(x2 * x3)
        + 0.08 * th1 * torch.sin(x2)
        + 0.06 * th2 * torch.sin(x3)
        + 0.05 * th3 * torch.tanh(x4)
        + d
    )
    return torch.stack([dx1, dx2, dx3, dx4], dim=-1)


def test_vector_field_matches_the_specification() -> None:
    """The assembled dynamics reproduce the written equations exactly."""
    system = AutomaticaN4System()
    torch.manual_seed(0)
    x = torch.randn(64, 4, dtype=torch.float64)
    theta = torch.randn(64, 3, dtype=torch.float64) * 0.25
    d = torch.randn(64, dtype=torch.float64) * 0.1

    assert torch.allclose(
        system.vector_field(x, theta, d), reference_rhs(x, theta, d), atol=1e-14
    )


def test_output_is_the_first_coordinate() -> None:
    system = AutomaticaN4System()
    x = torch.randn(5, 7, 4, dtype=torch.float64)
    assert torch.equal(system.output(x), x[..., 0])


def test_linear_skeleton_has_the_stated_poles() -> None:
    """The companion coefficients are those of ``(s+0.6)(s+0.8)(s+1)(s+1.2)``."""
    expected = np.poly([-0.6, -0.8, -1.0, -1.2])  # [1, 3.6, 4.76, 2.736, 0.576]
    # COMPANION multiplies (x_1, x_2, x_3, x_4), i.e. ascending powers of s.
    assert COMPANION == pytest.approx(tuple(expected[:0:-1]))

    matrix = torch.zeros(4, 4, dtype=torch.float64)
    for j in range(3):
        matrix[j, j + 1] = 1.0
    matrix[3] = -torch.tensor(COMPANION, dtype=torch.float64)
    poles = torch.linalg.eigvals(matrix).real.sort().values
    assert poles.tolist() == pytest.approx([-1.2, -1.0, -0.8, -0.6], abs=1e-10)
    assert bool((poles < 0).all()), "the nominal backbone must be stable"


def test_dimensions_and_admissible_sets() -> None:
    system = AutomaticaN4System()
    assert system.n == 4 and system.theta_dim == 3
    assert system.state_bounds.tolist() == [
        [-4.0, 4.0], [-1.2, 1.2], [-0.8, 0.8], [-1.2, 1.2]
    ]
    assert system.x0_bounds.tolist() == [
        [-0.8, 0.8], [-0.5, 0.5], [-0.4, 0.4], [-0.3, 0.3]
    ]
    # Theta_j is deliberately wider than the sampling range.
    assert system.theta_bounds.tolist() == [[-0.4, 0.4]] * 3


def test_sampled_parameters_lie_in_their_box() -> None:
    r""":math:`\theta_i \sim \mathcal{U}[-0.25, 0.25]`, drawn once and held constant."""
    for seed in range(6):
        system = AutomaticaN4System(theta_seed=seed)
        assert system.theta_true.shape == (3,)
        assert bool((system.theta_true.abs() <= 0.25).all())
    # A given seed is reproducible; different seeds differ.
    assert torch.equal(
        AutomaticaN4System(theta_seed=3).theta_true,
        AutomaticaN4System(theta_seed=3).theta_true,
    )
    assert not torch.equal(
        AutomaticaN4System(theta_seed=3).theta_true,
        AutomaticaN4System(theta_seed=4).theta_true,
    )


def test_true_parameters_stay_clear_of_the_admissible_boundary() -> None:
    r"""The truth must be interior to :math:`\Theta_j`, not on its edge.

    The parameter head is a tanh reparameterization onto :math:`\Theta_j`.  A
    target at the boundary is only reachable in the saturated limit, where the
    gradient vanishes and the estimate pins to the edge permanently.  Keeping
    the sampling range strictly inside :math:`\Theta_j` bounds the required
    pre-activation and keeps the head trainable.
    """
    for seed in range(8):
        system = AutomaticaN4System(theta_seed=seed)
        half_width = system.theta_bounds[:, 1]
        fraction = (system.theta_true.abs() / half_width).max()
        assert float(fraction) <= 0.7, (
            f"seed {seed}: true theta reaches {float(fraction):.0%} of the "
            "admissible half-width; the tanh head would saturate"
        )


# ----------------------------------------------------------------------
# excitation, Assumption 6
# ----------------------------------------------------------------------


def test_parameter_sensitivities_match_autograd() -> None:
    r"""The closed-form :math:`\partial_{\theta_j} f_j` agrees with autograd."""
    system = AutomaticaN4System()
    torch.manual_seed(0)
    x = torch.randn(32, 4, dtype=torch.float64)
    for j in (1, 2, 3):
        theta = torch.randn(32, 3, dtype=torch.float64, requires_grad=True) * 0.0
        theta = theta.detach().requires_grad_(True)
        value = system.f(j, x[..., :j], theta[..., :j])
        (grad,) = torch.autograd.grad(value.sum(), theta)
        assert torch.allclose(
            grad[..., j - 1], system.parameter_sensitivity(j, x), atol=1e-12
        )


def test_sensitivity_floors_hold_on_the_admissible_set() -> None:
    r"""The stated floors bound :math:`\partial_{\theta_j} f_j` from below.

    Evaluated on a dense grid over the admissible box :math:`\mathcal{X}`,
    which is what Assumption 6's uniform infimum ranges over.
    """
    system = AutomaticaN4System()
    lo, hi = system.state_bounds[:, 0], system.state_bounds[:, 1]
    axes = [torch.linspace(lo[j], hi[j], 31, dtype=torch.float64) for j in range(3)]
    grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3)

    for j, floor in enumerate(system.sensitivity_floors(), start=1):
        minimum = float(system.parameter_sensitivity(j, grid).min())
        assert minimum >= floor - 1e-9, (
            f"d f_{j}/d theta_{j} dips to {minimum:.4f}, below the stated floor {floor}"
        )


def test_excitation_floors_are_the_squared_sensitivities() -> None:
    r""":math:`\gamma_k` of Assumption 6 is the square of the sensitivity floor."""
    system = AutomaticaN4System()
    for floor, gamma in zip(system.sensitivity_floors(), system.excitation_floors()):
        assert gamma == pytest.approx(floor**2, abs=1e-12)
    assert system.excitation_floors() == pytest.approx((0.36, 0.49, 0.81))


# ----------------------------------------------------------------------
# trajectories
# ----------------------------------------------------------------------


def simulate(n_traj: int = 24, amplitude: float = 0.0):
    system = AutomaticaN4System()
    basis = FourierBasis(q=4, omega=OMEGA, t_start=0.0, t_end=HORIZON)
    generator = make_generator(0)
    coefficients = COEFF_BOUNDS[:, 0] + torch.rand(
        4, generator=generator, dtype=torch.float64
    ) * (COEFF_BOUNDS[:, 1] - COEFF_BOUNDS[:, 0])
    remainder = ChirpRemainder(amplitude) if amplitude > 0 else None
    disturbance = BasisDisturbance(basis, coefficients, remainder=remainder)

    t = uniform_grid(0.0, HORIZON, 201)
    x0 = system.sample_x0(n_traj, generator=generator)
    x = rk4_integrate(system, t, x0, system.theta_true, disturbance, substeps=8)
    return system, basis, disturbance, t, x


def test_trajectories_stay_inside_the_decoder_ranges() -> None:
    r"""Assumption 1 on the stated :math:`\mathcal{X}_j`, with margin."""
    system, _, _, _, x = simulate(n_traj=48)
    margin = check_admissible(system, x)
    assert bool((margin > 0).all()), f"margins {margin.tolist()}"


@pytest.mark.slow
def test_empirical_maxima_reproduce_the_specification() -> None:
    r"""Randomised sweep over :math:`x_0`, :math:`\theta` and :math:`a`.

    The specification reports :math:`\max|x_j| \approx (2.65, 0.72, 0.41,
    0.75)` across 300 random trials.  Reproducing that requires varying all
    three random inputs --- a single disturbance draw reaches well under the
    stated maxima, so this is a separate test from the admissibility check
    above.

    The direction that matters is the upper one: more sampling can only raise
    the observed maxima, so the assertion is that they stay below the stated
    values (with slack) *and* inside :math:`\mathcal{X}`.
    """
    system = AutomaticaN4System()
    basis = FourierBasis(q=4, omega=OMEGA, t_start=0.0, t_end=HORIZON)
    generator = make_generator(7)
    t = uniform_grid(0.0, HORIZON, 201)

    theta_lo, theta_hi = system.theta_bounds[:, 0], system.theta_bounds[:, 1]
    maxima = torch.zeros(4, dtype=torch.float64)
    for _ in range(150):
        unit = torch.rand(4, generator=generator, dtype=torch.float64)
        coefficients = COEFF_BOUNDS[:, 0] + unit * (
            COEFF_BOUNDS[:, 1] - COEFF_BOUNDS[:, 0]
        )
        theta = theta_lo + torch.rand(
            3, generator=generator, dtype=torch.float64
        ) * (theta_hi - theta_lo)
        x0 = system.sample_x0(1, generator=generator)
        x = rk4_integrate(
            system, t, x0, theta, BasisDisturbance(basis, coefficients), substeps=8
        )
        maxima = torch.maximum(maxima, x.abs().amax(dim=(0, 1)))

    stated = torch.tensor([2.65, 0.72, 0.41, 0.75], dtype=torch.float64)
    assert bool((maxima <= stated + 0.25).all()), (
        f"observed maxima {maxima.tolist()} exceed the specification's "
        f"{stated.tolist()}"
    )
    assert bool((maxima >= 0.5 * stated).all()), (
        f"observed maxima {maxima.tolist()} are far below the specification's "
        f"{stated.tolist()}; the sweep may not be exercising the dynamics"
    )
    # The decoder ranges must retain the promised margin.
    assert bool((maxima < system.state_bounds[:, 1]).all())


def test_residuals_vanish_on_the_true_trajectory() -> None:
    """Eqs. (31) and (40) hold exactly for this system's true solution."""
    system, basis, disturbance, t, x = simulate(n_traj=6)
    n_traj, n_samples = x.shape[0], x.shape[1]
    d_values = disturbance(t).expand(n_traj, n_samples)
    x_dot = system.vector_field(x, system.theta_true, d_values)
    theta = system.theta_true.expand(n_traj, 3)

    def cell(prev: int, new: int | None, head: torch.Tensor) -> CellOutput:
        return CellOutput(
            latent=torch.zeros(n_traj, 1, dtype=x.dtype),
            x_prev_data=x[..., prev - 1],
            x_prev_coll=x[..., prev - 1],
            x_prev_dot_coll=x_dot[..., prev - 1],
            x_new_data=None if new is None else x[..., new - 1],
            x_new_coll=None if new is None else x[..., new - 1],
            head=head,
        )

    outputs = {k: cell(k - 1, k, theta[:, [k - 2]]) for k in (2, 3, 4)}
    outputs[5] = cell(4, None, disturbance.coefficients.expand(n_traj, 4))

    for k in (2, 3, 4):
        residual = state_residual(system, k, outputs[k], outputs)
        assert float(residual.abs().max()) < 1e-9, f"r_{k} does not vanish"

    residual = final_residual(system, basis, outputs[5], outputs, t)
    assert float(residual.abs().max()) < 1e-9, "r_5 does not vanish"


def test_remainder_makes_exact_recovery_impossible() -> None:
    r"""With :math:`\varepsilon_{d,4} = 0.03` the final residual cannot vanish.

    Eq. (4) bounds :math:`\|r_q\|_\infty \le \varepsilon_{d,q}`, and exact
    recovery holds only when that constant is zero.
    """
    system, basis, disturbance, t, x = simulate(n_traj=4, amplitude=0.03)
    assert disturbance.remainder_bound(t) == pytest.approx(0.03, rel=1e-3)

    n_traj, n_samples = x.shape[0], x.shape[1]
    d_values = disturbance(t).expand(n_traj, n_samples)
    x_dot = system.vector_field(x, system.theta_true, d_values)
    theta = system.theta_true.expand(n_traj, 3)

    def cell(prev, new, head):
        return CellOutput(
            latent=torch.zeros(n_traj, 1, dtype=x.dtype),
            x_prev_data=x[..., prev - 1], x_prev_coll=x[..., prev - 1],
            x_prev_dot_coll=x_dot[..., prev - 1],
            x_new_data=None if new is None else x[..., new - 1],
            x_new_coll=None if new is None else x[..., new - 1],
            head=head,
        )

    outputs = {k: cell(k - 1, k, theta[:, [k - 2]]) for k in (2, 3, 4)}
    outputs[5] = cell(4, None, disturbance.coefficients.expand(n_traj, 4))

    # The state residuals are untouched: r_q enters the last equation only.
    for k in (2, 3, 4):
        assert float(state_residual(system, k, outputs[k], outputs).abs().max()) < 1e-9

    # The final residual is exactly the unrepresentable remainder.
    residual = final_residual(system, basis, outputs[5], outputs, t)
    assert torch.allclose(residual, ChirpRemainder(0.03)(t).expand_as(residual), atol=1e-9)
    assert float(residual.abs().max()) == pytest.approx(0.03, rel=1e-2)
