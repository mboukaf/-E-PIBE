r"""Remark 7: a local cell does not identify the state-parameter split.

The paper is explicit that one local loss is not enough to separate a free
state output from a parameter:

    "No separate bound on :math:`\epsilon^{int}_{s_k}` or
    :math:`\epsilon^{int}_{p_{k-1}}` follows from (101), even under
    Assumption 6.  Indeed, at the function-space level, consider the noise-free
    exact-upstream setting and choose a nonzero constant :math:`c` [...]  Set
    :math:`\bar x^k_{k-1} = x_{k-1}`, :math:`\bar\theta_{k-1} = \theta_{k-1} +
    c`, and choose the second decoder output as [...]  Both the data loss and
    the physics loss are then zero, while the parameter error equals :math:`c`
    and the state error compensates for it."

Two things are worth pinning down as tests.

1. **The model class is right.**  Holding the states at their true values, the
   physics loss is minimized at the true parameter.  If this ever fails, the
   residual or the system is wrong.

2. **The degeneracy is real.**  Remark 7's construction --- perturb
   :math:`\theta_{k-1}` by :math:`c` and let the free output :math:`x_k` absorb
   it --- drives the local loss to zero at a wrong parameter.  This is a
   property of the method as specified, not a defect of the implementation, and
   it is why Assumption 4 is stated as "a post-training accuracy hypothesis,
   not a consequence of the Universal Approximation Theorem".

Observing an estimate :math:`\hat\theta` that sits near the centre of
:math:`\Theta_j` while the physics loss is small is therefore the expected
signature of weak excitation (Assumption 6), not evidence of a bug.
"""

from __future__ import annotations

import pytest
import torch

from pibe.core.cell import CellOutput
from pibe.core.residuals import state_residual
from pibe.data.simulate import rk4_integrate, uniform_grid
from pibe.systems.examples.sin_chain import SinChainSystem
from pibe.utils.seeding import make_generator


def truth(n: int = 3, n_traj: int = 4, n_samples: int = 81, horizon: float = 4.0):
    """Simulate the example system and return the exact states and derivatives."""
    system = SinChainSystem(n=n)
    t = uniform_grid(0.0, horizon, n_samples)
    x0 = system.sample_x0(n_traj, generator=make_generator(0))
    x = rk4_integrate(system, t, x0, system.theta_true, substeps=32)
    zeros = torch.zeros(n_traj, n_samples, dtype=x.dtype)
    x_dot = system.vector_field(x, system.theta_true, zeros)
    return system, x, x_dot


def cell_output(x_prev, x_new, x_dot_prev, theta_value) -> CellOutput:
    return CellOutput(
        latent=torch.zeros(x_prev.shape[0], 1, dtype=x_prev.dtype),
        x_prev_data=x_prev,
        x_prev_coll=x_prev,
        x_prev_dot_coll=x_dot_prev,
        x_new_data=x_new,
        x_new_coll=x_new,
        head=torch.full((x_prev.shape[0], 1), float(theta_value), dtype=x_prev.dtype),
    )


def test_physics_loss_is_minimized_at_the_true_parameter() -> None:
    """With the true states held fixed, the residual identifies ``theta_1``.

    Guards the model class: the true parameter must be recoverable in
    principle, whatever the optimizer does in practice.
    """
    system, x, x_dot = truth()
    true_theta = float(system.theta_true[0])

    grid = torch.linspace(true_theta - 1.0, true_theta + 1.0, 201, dtype=torch.float64)
    losses = []
    for value in grid:
        probe = cell_output(x[..., 0], x[..., 1], x_dot[..., 0], value)
        residual = state_residual(system, 2, probe, {2: probe})
        losses.append(float((residual**2).mean()))

    best = float(grid[int(torch.tensor(losses).argmin())])
    assert best == pytest.approx(true_theta, abs=0.02), (
        f"physics loss minimized at theta={best:.4f}, true value {true_theta:.4f}"
    )
    assert min(losses) < 1e-8, "the residual does not vanish at the true parameter"


def test_free_state_output_absorbs_a_parameter_perturbation() -> None:
    r"""Remark 7's counterexample: a wrong ``theta`` costs nothing.

    Perturb :math:`\theta_1` by :math:`c` and set the free output to
    :math:`\bar x_2 = \dot x_1 - f_1(x_1, \theta_1 + c)`.  The data term is
    untouched (:math:`\bar x^2_1 = x_1`) and the physics residual is still
    identically zero, yet the parameter error is exactly :math:`c`.
    """
    system, x, x_dot = truth()
    true_theta = float(system.theta_true[0])

    for c in (0.1, -0.25, 0.5):
        perturbed = true_theta + c
        theta_vector = torch.full(
            (*x.shape[:2], 1), perturbed, dtype=x.dtype
        )
        # x_2 chosen to cancel the perturbed drift exactly.
        absorbing = x_dot[..., 0] - system.f(1, x[..., :1], theta_vector)

        probe = cell_output(x[..., 0], absorbing, x_dot[..., 0], perturbed)
        residual = state_residual(system, 2, probe, {2: probe})

        assert float((residual**2).mean()) < 1e-24, (
            f"Remark 7's construction should give an exactly zero residual for c={c}"
        )
        # ... while the state estimate is wrong by exactly the compensating amount.
        state_error = float((absorbing - x[..., 1]).abs().max())
        assert state_error > 0.01, (
            "the construction is vacuous: the absorbing output equals the true x_2"
        )


def test_degeneracy_needs_the_free_output() -> None:
    """Pinning ``x_2`` to the truth restores identifiability of ``theta_1``.

    Confirms the degeneracy is exactly the one Remark 7 describes --- it lives
    in the *free* second decoder output, not in the residual itself.
    """
    system, x, x_dot = truth()
    true_theta = float(system.theta_true[0])

    # theta perturbed but x_2 held at the truth: the residual can no longer vanish.
    probe = cell_output(x[..., 0], x[..., 1], x_dot[..., 0], true_theta + 0.3)
    residual = state_residual(system, 2, probe, {2: probe})
    assert float((residual**2).mean()) > 1e-4
