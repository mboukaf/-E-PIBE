r"""Scoring the *true* solution on the training objective.

When a trained bank produces poor estimates while reporting a small loss, there
are two very different explanations, and they call for opposite responses:

**Optimization failure.**  The true :math:`(x, \theta, a)` achieves a *lower*
objective than what training found, but the optimizer never got there.  The
levers are the schedule --- more end-to-end fine-tuning, a larger
``lr_global``, a smaller :math:`N_{par}` fraction --- and the reparameterization
ranges.  Nothing about the losses needs to change.

**Loss-design failure.**  The objective genuinely prefers the wrong answer, so
no amount of optimization will recover the truth.  This is the situation
Remark 7 constructs: a free state output absorbs a parameter perturbation at
zero cost.  Only extra information --- excitation (Assumption 6), a coercivity
inequality (Remark 10), or a different objective --- can help.

This module settles which one is happening by evaluating
:math:`\mathcal{L}^k_{Loc}` and :math:`\mathcal{L}^{n+1}_{Tot}` at the exact
solution, and comparing against the trained bank on the same data.

Two reference points are worth knowing when reading the output.  At the truth,
every physics residual vanishes identically (up to integration error), so the
physics terms are ~1e-30 rather than 0.  And the first cell's data term cannot
go below the measurement-noise variance, since :math:`\hat x_1 = x_1` while
:math:`y = x_1 + \omega`; a *trained* cell-2 data term below that floor means
the decoder is fitting noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from pibe.basis.base import DisturbanceBasis
from pibe.core.bank import EstimatorBank, Mode
from pibe.core.cell import CellOutput
from pibe.core.losses import (
    CellLoss,
    data_loss,
    local_loss,
    physics_loss,
    total_loss,
)
from pibe.core.residuals import final_residual, state_residual
from pibe.data.dataset import TrajectoryData
from pibe.data.disturbance import BasisDisturbance
from pibe.data.simulate import rk4_integrate
from pibe.systems.base import TriangularSystem


@dataclass
class OracleComparison:
    """Per-cell local losses at the truth and at the trained bank."""

    oracle: dict[int, CellLoss]
    trained: dict[int, CellLoss]
    oracle_total: float
    trained_total: float
    noise_floor: float | None

    @property
    def truth_scores_better(self) -> bool:
        """Whether the exact solution beats the trained one on the objective."""
        return self.oracle_total < self.trained_total

    @property
    def verdict(self) -> str:
        """Which of the two failure modes the numbers indicate."""
        if self.truth_scores_better:
            return (
                "optimization failure: the true solution attains a lower objective "
                "than training reached, so it is representable and preferred but "
                "was not found. Lengthen end-to-end fine-tuning (lower N_par / "
                "raise lr_global) and check that no head is pinned to the "
                "boundary of its admissible box."
            )
        return (
            "loss-design failure: the objective scores the trained solution at "
            "or below the truth, so optimization cannot recover the parameters. "
            "This is the Remark 7 degeneracy; it needs excitation "
            "(Assumption 6) or extra information, not more training."
        )

    def summary(self) -> str:
        lines = [
            f"{'cell':>5} | {'L_Loc @ truth':>15} | {'L_Loc @ trained':>15} | winner",
            "-" * 60,
        ]
        for k in sorted(self.oracle):
            at_truth = float(self.oracle[k].local)
            at_trained = float(self.trained[k].local)
            winner = "truth" if at_truth < at_trained else "trained"
            lines.append(
                f"{k:>5} | {at_truth:>15.4e} | {at_trained:>15.4e} | {winner}"
            )
        lines.append("-" * 60)
        lines.append(f"L_Tot @ truth   : {self.oracle_total:.4e}")
        lines.append(f"L_Tot @ trained : {self.trained_total:.4e}")
        if self.noise_floor is not None:
            lines.append(
                f"noise floor     : {self.noise_floor:.4e} "
                "(the cell-2 data term cannot honestly go below this)"
            )
        lines.append("")
        lines.append(f"verdict: {self.verdict}")
        return "\n".join(lines)


def oracle_losses(
    system: TriangularSystem,
    basis: DisturbanceBasis,
    disturbance: BasisDisturbance,
    data: TrajectoryData,
    t_coll: Tensor,
    lam: float,
    substeps: int = 16,
) -> dict[int, CellLoss]:
    r"""Evaluate every :math:`\mathcal{L}^k_{Loc}` at the exact solution.

    The true trajectory is re-integrated onto the collocation grid so the
    physics terms are evaluated at the same points training uses.
    """
    x_data = data.x
    # Truth is per trajectory: each carries its own ``a`` (and possibly its own
    # ``theta``), so the exact solution must be built from ``data``, not from
    # the system's single nominal value.
    theta = data.theta
    # Rebuild the disturbance from the *subset's own* coefficients: after a
    # train/val split the passed-in object still carries every trajectory's.
    if data.coefficients is not None:
        disturbance = BasisDisturbance(
            basis, data.coefficients, remainder=disturbance.remainder
        )
    x_coll = rk4_integrate(
        system,
        t_grid=t_coll,
        x0=x_data[:, 0, :],
        theta=theta,
        disturbance=disturbance,
        substeps=substeps,
    )
    n_traj, n_coll = x_coll.shape[0], t_coll.numel()
    d_coll = disturbance(t_coll)
    if d_coll.ndim == 1:
        d_coll = d_coll.reshape(1, -1).expand(n_traj, n_coll)
    x_dot = system.vector_field(x_coll, theta, d_coll)
    final = system.n + 1

    def make(prev: int, new: int | None, head: Tensor) -> CellOutput:
        return CellOutput(
            latent=torch.zeros(n_traj, 1, dtype=x_data.dtype, device=x_data.device),
            x_prev_data=x_data[..., prev - 1],
            x_prev_coll=x_coll[..., prev - 1],
            x_prev_dot_coll=x_dot[..., prev - 1],
            x_new_data=None if new is None else x_data[..., new - 1],
            x_new_coll=None if new is None else x_coll[..., new - 1],
            head=head,
        )

    outputs = {
        k: make(k - 1, k, theta[:, [k - 2]]) for k in range(2, system.n + 1)
    }
    coefficients = disturbance.coefficients
    if coefficients.shape[0] == 1:
        coefficients = coefficients.expand(n_traj, basis.q)
    coefficients = coefficients.to(x_data.dtype)
    outputs[final] = make(system.n, None, coefficients)

    losses: dict[int, CellLoss] = {}
    for k in range(2, final + 1):
        target = data.y if k == 2 else outputs[k - 1].x_new_data
        residual = (
            final_residual(system, basis, outputs[k], outputs, t_coll)
            if k == final
            else state_residual(system, k, outputs[k], outputs)
        )
        losses[k] = local_loss(
            data_loss(outputs[k].x_prev_data, target), physics_loss(residual), lam
        )
    return losses


def compare_to_oracle(
    bank: EstimatorBank,
    disturbance: BasisDisturbance,
    data: TrajectoryData,
    t_coll: Tensor,
    lam: float,
    noise_floor: float | None = None,
) -> OracleComparison:
    """Score the trained bank against the exact solution on the same data."""
    oracle = oracle_losses(
        bank.system, bank.basis, disturbance, data, t_coll, lam
    )

    was_training = bank.training
    bank.eval()
    try:
        with torch.enable_grad():
            outputs = bank(
                target_cell=bank.final_index,
                y=data.y,
                t_data=data.t,
                t_coll=t_coll,
                mode=Mode.GLOBAL,
                create_graph=False,
            )
            trained = {
                k: bank.cell_loss(k, data.y, t_coll, outputs, lam)
                for k in bank.cell_indices
            }
    finally:
        bank.train(was_training)
    trained = {
        k: CellLoss(
            data=loss.data.detach(),
            physics=loss.physics.detach(),
            local=loss.local.detach(),
        )
        for k, loss in trained.items()
    }

    final = bank.final_index
    return OracleComparison(
        oracle=oracle,
        trained=trained,
        oracle_total=float(
            total_loss({k: v.local for k, v in oracle.items()}, final)
        ),
        trained_total=float(
            total_loss({k: v.local for k, v in trained.items()}, final)
        ),
        noise_floor=noise_floor,
    )
