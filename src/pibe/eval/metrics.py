r"""Estimation-accuracy metrics.

Errors are reported in the norms the analysis of Section 4 uses: the
normalized discrete :math:`L^2` norm of Eq. (60),

.. math:: \|\mathbf{v}\|_{N,2} := \Bigl(\frac{1}{N}\sum_{i=1}^N |v_i|^2\Bigr)^{1/2},

which is grid-size independent, alongside supremum norms, which is the form the
componentwise bounds (114)-(118) of Corollary 2 take.

Because the estimator is trajectory-conditioned, the parameter and coefficient
estimates vary with :math:`\ell`; both the mean and the spread across
trajectories are reported.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import torch
from torch import Tensor

from pibe.core.bank import Estimates
from pibe.data.dataset import TrajectoryData


def normalized_l2(v: Tensor, dim: int = -1) -> Tensor:
    r"""The norm :math:`\|\cdot\|_{N,2}` of Eq. (60), along ``dim``."""
    return torch.sqrt(torch.mean(v**2, dim=dim))


@dataclass
class EstimationMetrics:
    r"""Accuracy of one trained bank on one set of trajectories.

    Attributes
    ----------
    state_l2, state_sup
        Per-coordinate state errors, shape ``(n,)``: :math:`\|\hat x_j -
        x_j\|_{N,2}` and :math:`\|\hat x_j - x_j\|_\infty`, averaged and
        maximised over trajectories respectively.
    theta_abs_error
        :math:`|\hat\theta_j - \theta_j|` averaged over trajectories, shape
        ``(n - 1,)``.
    theta_std
        Spread of :math:`\hat\theta_j` across trajectories, shape ``(n - 1,)``.
        The true parameters are constants shared by every trajectory, so a
        large spread indicates the bank is fitting trajectory-specific
        artefacts rather than the parameter.
    disturbance_l2, disturbance_sup
        Errors of :math:`\hat d = \Gamma_q^\top \hat a` against the true
        :math:`d`.  When the true disturbance leaves the span of
        :math:`\Gamma_q`, these are bounded below by
        :math:`\varepsilon_{d,q}`, Eq. (4).
    output_l2
        :math:`\|\hat x_1 - y\|_{N,2}`, the residual of the only genuine
        measurement fit.
    """

    state_l2: Tensor
    state_sup: Tensor
    theta_abs_error: Tensor
    theta_std: Tensor
    disturbance_l2: float
    disturbance_sup: float
    output_l2: float

    def as_dict(self) -> dict[str, Any]:
        """Plain-Python view, suitable for JSON."""
        payload = asdict(self)
        for key, value in payload.items():
            if isinstance(value, Tensor):
                payload[key] = value.tolist()
        return payload

    def summary(self) -> str:
        """A short human-readable table."""
        lines = ["state errors (L2 / sup):"]
        for j in range(self.state_l2.numel()):
            lines.append(
                f"  x_{j + 1}: {float(self.state_l2[j]):.4e} / "
                f"{float(self.state_sup[j]):.4e}"
            )
        lines.append("parameter errors (|err| / spread):")
        for j in range(self.theta_abs_error.numel()):
            lines.append(
                f"  theta_{j + 1}: {float(self.theta_abs_error[j]):.4e} / "
                f"{float(self.theta_std[j]):.4e}"
            )
        lines.append(
            f"disturbance: L2={self.disturbance_l2:.4e}, sup={self.disturbance_sup:.4e}"
        )
        lines.append(f"output fit: L2={self.output_l2:.4e}")
        return "\n".join(lines)


@torch.no_grad()
def compute_metrics(estimates: Estimates, data: TrajectoryData) -> EstimationMetrics:
    """Compare a bank's estimates against ground truth.

    Parameters
    ----------
    estimates
        Output of :meth:`~pibe.core.bank.EstimatorBank.estimate` on ``data.y``.
    data
        The trajectories the estimates were produced from, carrying the true
        states, parameters and disturbance.
    """
    if estimates.x.shape != data.x.shape:
        raise ValueError(
            f"estimate shape {tuple(estimates.x.shape)} does not match the "
            f"data {tuple(data.x.shape)}"
        )

    state_error = estimates.x - data.x  # (B, N, n)
    state_l2 = normalized_l2(state_error, dim=1).mean(dim=0)  # (n,)
    state_sup = state_error.abs().amax(dim=1).amax(dim=0)  # (n,)

    theta_error = estimates.theta - data.theta  # (B, n-1)
    theta_abs_error = theta_error.abs().mean(dim=0)
    theta_std = (
        estimates.theta.std(dim=0, unbiased=False)
        if estimates.theta.shape[0] > 1
        else torch.zeros_like(theta_abs_error)
    )

    disturbance_error = estimates.d - data.d  # (B, N) against (N,)
    output_error = estimates.x[..., 0] - data.y

    return EstimationMetrics(
        state_l2=state_l2,
        state_sup=state_sup,
        theta_abs_error=theta_abs_error,
        theta_std=theta_std,
        disturbance_l2=float(normalized_l2(disturbance_error, dim=1).mean()),
        disturbance_sup=float(disturbance_error.abs().amax()),
        output_l2=float(normalized_l2(output_error, dim=1).mean()),
    )
