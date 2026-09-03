r"""The third-order benchmark system.

As for the fourth-order case, the right-hand side is transcribed a second time
from the specification and compared against the implementation, so a coefficient
typo fails here rather than surfacing as unexplained accuracy loss.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from pibe.basis import FourierBasis
from pibe.core.cell import CellOutput
from pibe.core.residuals import final_residual, state_residual
from pibe.data.disturbance import BasisDisturbance
from pibe.data.simulate import check_admissible, rk4_integrate, uniform_grid
from pibe.eval.observability import basis_observability
from pibe.systems.examples.automatica_n3 import COMPANION, AutomaticaN3System
from pibe.systems.examples.automatica_n4 import AutomaticaN4System
from pibe.utils.seeding import make_generator

OMEGA = 2.0 * math.pi / 5.0
HORIZON = 20.0
COEFF_BOUNDS = torch.tensor(
    [[-0.18, 0.18], [-0.18, 0.18], [-0.12, 0.12]], dtype=torch.float64
)


def reference_rhs(x, theta, d):
    """The specification's right-hand side, transcribed independently."""
    x1, x2, x3 = x[..., 0], x[..., 1], x[..., 2]
    th1, th2 = theta[..., 0], theta[..., 1]
    dx1 = x2 - 0.35 * torch.tanh(x1) + th1 * (0.8 + 0.2 * torch.cos(x1))
    dx2 = (
        x3 - 0.30 * torch.tanh(x2) + 0.10 * torch.sin(x1)
        + th2 * (0.9 + 0.1 * torch.sin(x1) + 0.1 * torch.cos(x2))
    )
    dx3 = (
        -0.648 * x1 - 2.34 * x2 - 2.7 * x3
        + 0.15 * torch.sin(x1)
        + 0.08 * torch.tanh(x1 * x2)
        + 0.08 * th1 * torch.sin(x2)
        + 0.06 * th2 * torch.tanh(x3)
        + d
    )
    return torch.stack([dx1, dx2, dx3], dim=-1)


def test_vector_field_matches_the_specification() -> None:
    system = AutomaticaN3System()
    torch.manual_seed(0)
    x = torch.randn(64, 3, dtype=torch.float64)
    theta = torch.randn(64, 2, dtype=torch.float64) * 0.25
    d = torch.randn(64, dtype=torch.float64) * 0.1
    assert torch.allclose(
        system.vector_field(x, theta, d), reference_rhs(x, theta, d), atol=1e-14
    )


def test_linear_skeleton_is_hurwitz_with_the_stated_poles() -> None:
    expected = np.poly([-0.6, -0.9, -1.2])  # [1, 2.7, 2.34, 0.648]
    assert COMPANION == pytest.approx(tuple(expected[:0:-1]))
    matrix = torch.zeros(3, 3, dtype=torch.float64)
    matrix[0, 1] = matrix[1, 2] = 1.0
    matrix[2] = -torch.tensor(COMPANION, dtype=torch.float64)
    poles = torch.linalg.eigvals(matrix).real.sort().values
    assert poles.tolist() == pytest.approx([-1.2, -0.9, -0.6], abs=1e-10)
    assert bool((poles < 0).all())


def test_dimensions_and_admissible_sets() -> None:
    system = AutomaticaN3System()
    assert system.n == 3 and system.theta_dim == 2
    assert system.state_bounds.tolist() == [[-2.0, 2.0], [-0.8, 0.8], [-0.8, 0.8]]
    assert system.x0_bounds.tolist() == [[-0.8, 0.8], [-0.5, 0.5], [-0.35, 0.35]]
    # Theta_j is wider than the sampling range, keeping the truth interior.
    assert system.theta_bounds.tolist() == [[-0.4, 0.4]] * 2
    for seed in range(6):
        assert bool((AutomaticaN3System(theta_seed=seed).theta_true.abs() <= 0.25).all())


def test_sensitivity_floors_hold_on_the_admissible_set() -> None:
    system = AutomaticaN3System()
    lo, hi = system.state_bounds[:, 0], system.state_bounds[:, 1]
    axes = [torch.linspace(lo[j], hi[j], 61, dtype=torch.float64) for j in range(2)]
    grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 2)
    for j, floor in enumerate(system.sensitivity_floors(), start=1):
        assert float(system.parameter_sensitivity(j, grid).min()) >= floor - 1e-9
    assert system.excitation_floors() == pytest.approx((0.36, 0.49))


def simulate(n_traj: int = 24):
    system = AutomaticaN3System()
    basis = FourierBasis(q=3, omega=OMEGA, t_start=0.0, t_end=HORIZON)
    generator = make_generator(0)
    coefficients = COEFF_BOUNDS[:, 0] + torch.rand(
        3, generator=generator, dtype=torch.float64
    ) * (COEFF_BOUNDS[:, 1] - COEFF_BOUNDS[:, 0])
    disturbance = BasisDisturbance(basis, coefficients)
    t = uniform_grid(0.0, HORIZON, 201)
    x0 = system.sample_x0(n_traj, generator=generator)
    x = rk4_integrate(system, t, x0, system.theta_true, disturbance, substeps=8)
    return system, basis, disturbance, t, x


def test_trajectories_stay_inside_the_decoder_ranges() -> None:
    system, _, _, _, x = simulate(n_traj=48)
    assert bool((check_admissible(system, x) > 0).all())


@pytest.mark.slow
def test_empirical_maxima_reproduce_the_specification() -> None:
    r"""The specification reports max|x| <~ (1.29, 0.52, 0.44) over 300 trials."""
    system = AutomaticaN3System()
    basis = FourierBasis(q=3, omega=OMEGA, t_start=0.0, t_end=HORIZON)
    generator = make_generator(11)
    t = uniform_grid(0.0, HORIZON, 201)
    lo, hi = system.theta_bounds[:, 0], system.theta_bounds[:, 1]
    maxima = torch.zeros(3, dtype=torch.float64)
    for _ in range(150):
        u = torch.rand(3, generator=generator, dtype=torch.float64)
        a = COEFF_BOUNDS[:, 0] + u * (COEFF_BOUNDS[:, 1] - COEFF_BOUNDS[:, 0])
        theta = 0.25 * (2 * torch.rand(2, generator=generator, dtype=torch.float64) - 1)
        x0 = system.sample_x0(1, generator=generator)
        x = rk4_integrate(system, t, x0, theta, BasisDisturbance(basis, a), substeps=8)
        maxima = torch.maximum(maxima, x.abs().amax(dim=(0, 1)))
    stated = torch.tensor([1.29, 0.52, 0.44], dtype=torch.float64)
    assert bool((maxima <= stated + 0.15).all()), f"observed {maxima.tolist()}"
    assert bool((maxima < system.state_bounds[:, 1]).all())


def test_residuals_vanish_on_the_true_trajectory() -> None:
    system, basis, disturbance, t, x = simulate(n_traj=6)
    p, n_samples = x.shape[0], x.shape[1]
    x_dot = system.vector_field(x, system.theta_true, disturbance(t).expand(p, n_samples))
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

    outputs = {k: cell(k - 1, k, theta[:, [k - 2]]) for k in (2, 3)}
    outputs[4] = cell(3, None, disturbance.coefficients.expand(p, 3))
    for k in (2, 3):
        assert float(state_residual(system, k, outputs[k], outputs).abs().max()) < 1e-9
    assert float(final_residual(system, basis, outputs[4], outputs, t).abs().max()) < 1e-9


def test_disturbance_is_more_observable_than_in_the_fourth_order_system() -> None:
    r"""One fewer integration means a stronger trace of ``d`` at the output.

    This is the structural reason to prefer the third-order benchmark: the
    disturbance enters the last equation and is measured only through ``x_1``,
    so every extra chain link attenuates it.
    """
    basis3 = FourierBasis(q=3, omega=OMEGA, t_start=0.0, t_end=HORIZON)
    basis4 = FourierBasis(q=4, omega=OMEGA, t_start=0.0, t_end=HORIZON)
    bounds4 = torch.tensor(
        [[-0.15, 0.15], [-0.15, 0.15], [-0.12, 0.12], [-0.12, 0.12]],
        dtype=torch.float64,
    )
    third = basis_observability(
        AutomaticaN3System(), basis3, COEFF_BOUNDS, noise_sigma=0.002, n_samples=201
    )
    fourth = basis_observability(
        AutomaticaN4System(), basis4, bounds4, noise_sigma=0.002, n_samples=201
    )
    # The weakest component of the shorter chain is far better resolved.
    assert float(third.snr.min()) > 2.0 * float(fourth.snr.min())
    # ... and it clears the noise floor, which the fourth-order one does not.
    assert float(third.snr.min()) > 1.0
    assert float(fourth.snr.min()) < 1.0
