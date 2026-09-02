r"""The physics residuals must vanish on the true trajectory.

This is the strongest available check on the argument assembly of Eqs. (26),
(31) and (40).  We build synthetic cell outputs carrying the *true* states,
parameters and coefficients, and supply the true time derivative computed
independently from :meth:`TriangularSystem.vector_field`.  Then

.. math::

    r_k    &= \dot x_{k-1} - x_k - f_{k-1}(\dots) = 0, \\
    r_{n+1} &= \dot x_n - f_n(x,\theta) - \Gamma_q^\top a = d - \Gamma_q^\top a = 0

identically, the last equality holding because the test disturbance is exactly
a basis expansion (:math:`r_q = 0`).

The test has teeth because the two sides come from different code paths: the
derivative is row ``k-2`` of the assembled vector field, while the residual
reconstructs :math:`f_{k-1}`'s arguments by *indexing into a dictionary of cell
outputs*.  Any misordering --- taking :math:`x_{k-1}` from cell ``k-1``'s new
output instead of the current cell's reconstruction, or pairing
:math:`\theta_j` with the wrong cell --- makes the residual nonzero, because
the example system's :math:`f_j` genuinely depends on several arguments.
"""

from __future__ import annotations

import pytest
import torch

from pibe.basis.bspline import BSplineBasis
from pibe.core.cell import CellOutput
from pibe.core.residuals import (
    disturbance_estimate,
    final_residual,
    state_residual,
    upstream_states,
    upstream_thetas,
)
from pibe.data.disturbance import BasisDisturbance
from pibe.data.simulate import rk4_integrate, uniform_grid
from pibe.systems.examples.sin_chain import SinChainSystem

TOL = 1e-9


def build_truth(n: int, horizon: float = 3.0, n_samples: int = 41, n_traj: int = 4):
    """Simulate the example system and package the truth as cell outputs."""
    torch.manual_seed(0)
    system = SinChainSystem(n=n)
    basis = BSplineBasis(q=5, degree=3, t_start=0.0, t_end=horizon)
    coefficients = torch.linspace(-1.5, 1.5, basis.q, dtype=torch.float64)
    disturbance = BasisDisturbance(basis, coefficients)

    t = uniform_grid(0.0, horizon, n_samples)
    generator = torch.Generator().manual_seed(0)
    x0 = system.sample_x0(n_traj, generator=generator)
    x = rk4_integrate(system, t, x0, system.theta_true, disturbance, substeps=16)

    theta = system.theta_true.expand(n_traj, system.theta_dim)
    d_values = disturbance(t).expand(n_traj, n_samples)
    x_dot = system.vector_field(x, system.theta_true, d_values)  # (P, N, n)

    def cell_output(prev: int, new: int | None, head: torch.Tensor) -> CellOutput:
        """A cell reconstructing coordinate ``prev`` and estimating ``new`` (1-based)."""
        x_prev = x[..., prev - 1]
        x_new = None if new is None else x[..., new - 1]
        return CellOutput(
            latent=torch.zeros(n_traj, 1, dtype=torch.float64),
            x_prev_data=x_prev,
            x_prev_coll=x_prev,
            x_prev_dot_coll=x_dot[..., prev - 1],
            x_new_data=x_new,
            x_new_coll=x_new,
            head=head,
        )

    outputs: dict[int, CellOutput] = {}
    for k in range(2, n + 1):
        outputs[k] = cell_output(prev=k - 1, new=k, head=theta[:, [k - 2]])
    outputs[n + 1] = cell_output(
        prev=n, new=None, head=coefficients.expand(n_traj, basis.q)
    )
    return system, basis, t, x, outputs, coefficients


@pytest.mark.parametrize("n", [2, 3, 4, 6])
def test_state_residual_vanishes_on_truth(n: int) -> None:
    """Eq. (31) is satisfied exactly by the true trajectory."""
    system, _, _, _, outputs, _ = build_truth(n)
    for k in range(2, n + 1):
        residual = state_residual(system, k, outputs[k], outputs)
        assert residual.abs().max() < TOL, (
            f"r_{k} does not vanish on the true trajectory: "
            f"max |r| = {float(residual.abs().max()):.3e}"
        )


@pytest.mark.parametrize("n", [2, 3, 4, 6])
def test_final_residual_vanishes_on_truth(n: int) -> None:
    """Eq. (40) is satisfied exactly when the disturbance lies in the basis span."""
    system, basis, t, _, outputs, _ = build_truth(n)
    residual = final_residual(system, basis, outputs[n + 1], outputs, t)
    assert residual.abs().max() < TOL, (
        f"r_(n+1) does not vanish: max |r| = {float(residual.abs().max()):.3e}"
    )


@pytest.mark.parametrize("n", [3, 5])
def test_upstream_assembly_recovers_true_coordinates(n: int) -> None:
    r"""``I^x`` and ``I^\theta`` of Eq. (26) select the right quantities."""
    system, _, _, x, outputs, _ = build_truth(n)

    # I^x_{upto} must equal (x_1, ..., x_upto).
    for upto in range(0, n):
        states = upstream_states(outputs, upto, grid="coll")
        assert len(states) == upto
        for j, state in enumerate(states, start=1):
            assert torch.allclose(state, x[..., j - 1]), (
                f"upstream_states picked the wrong tensor for x_{j}"
            )

    # I^theta_{upto} must equal (theta_1, ..., theta_upto).
    for upto in range(0, n):
        thetas = upstream_thetas(outputs, upto)
        assert len(thetas) == upto
        for j, theta in enumerate(thetas, start=1):
            expected = system.theta_true[j - 1]
            assert torch.allclose(theta, torch.full_like(theta, float(expected))), (
                f"upstream_thetas picked the wrong tensor for theta_{j}"
            )


def _replace(output: CellOutput, **changes) -> CellOutput:
    """A copy of ``output`` with selected fields overridden."""
    fields = {
        "latent": output.latent,
        "x_prev_data": output.x_prev_data,
        "x_prev_coll": output.x_prev_coll,
        "x_prev_dot_coll": output.x_prev_dot_coll,
        "x_new_data": output.x_new_data,
        "x_new_coll": output.x_new_coll,
        "head": output.head,
    }
    fields.update(changes)
    return CellOutput(**fields)


def test_residual_detects_swapped_parameters() -> None:
    """A misordered parameter list must produce a nonzero residual.

    Guards the vanishing-residual tests against being vacuous: if the example
    system's ``f_j`` ignored most of its arguments, those tests would pass even
    for a wrong assembly.
    """
    system, _, _, _, outputs, _ = build_truth(n=4)
    corrupted = dict(outputs)
    # theta_2 where theta_1 belongs.
    corrupted[2] = _replace(outputs[2], head=outputs[3].head)
    residual = state_residual(system, 3, corrupted[3], corrupted)
    assert residual.abs().max() > 1e-3, (
        "swapping theta_1 for theta_2 left the residual unchanged"
    )


def test_residual_detects_swapped_states() -> None:
    r"""Taking :math:`x_1` from the wrong decoder output must be detected.

    Eq. (26) specifies :math:`x_1 = \hat x^2_1`, cell 2's *reconstruction*.
    Substituting cell 2's *new* coordinate :math:`\hat x^2_2` must change the
    residual.
    """
    system, _, _, _, outputs, _ = build_truth(n=4)
    corrupted = dict(outputs)
    corrupted[2] = _replace(
        outputs[2],
        x_prev_coll=outputs[2].x_new_coll,
        x_prev_data=outputs[2].x_new_data,
    )
    residual = state_residual(system, 3, corrupted[3], corrupted)
    assert residual.abs().max() > 1e-3, (
        "substituting x_2 for x_1 left the residual unchanged"
    )


def test_final_residual_detects_wrong_coefficients() -> None:
    """A perturbed coefficient vector must break Eq. (40)."""
    system, basis, t, _, outputs, coefficients = build_truth(n=3)
    perturbed = coefficients.clone()
    perturbed[0] += 0.5
    corrupted = dict(outputs)
    corrupted[4] = _replace(
        outputs[4], head=perturbed.expand(outputs[4].head.shape[0], basis.q)
    )
    residual = final_residual(system, basis, corrupted[4], corrupted, t)
    assert residual.abs().max() > 1e-3, (
        "perturbing a disturbance coefficient left the final residual unchanged"
    )


def test_disturbance_estimate_matches_basis_expansion() -> None:
    """Eq. (41) reproduces the true disturbance when the coefficients are exact."""
    _, basis, t, _, _, coefficients = build_truth(n=3)
    a_hat = coefficients.expand(4, basis.q)
    estimated = disturbance_estimate(basis, a_hat, t)
    truth = BasisDisturbance(basis, coefficients)(t)
    assert torch.allclose(estimated, truth.expand_as(estimated), atol=1e-12)
