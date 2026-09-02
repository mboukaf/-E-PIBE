r"""PIBE loss functions.

Each cell carries a local objective combining a data-fit term with a physics
term, :math:`\mathcal{L}^k_{Loc} = \mathcal{L}^k_{Data} + \lambda
\mathcal{L}^k_{Physics}`, following the hybrid PINN loss (6).

Data terms
----------
The three data terms are one rule: cell ``k`` compares its reconstruction of
coordinate ``k-1`` against the target supplied from upstream.

.. math::

    \mathcal{L}^2_{Data} &= \frac{1}{N}\sum_{i=1}^N
        \bigl(y(t_i) - \hat x^2_1(t_i)\bigr)^2, & &\text{(23)} \\
    \mathcal{L}^k_{Data} &= \frac{1}{N}\sum_{i=1}^N
        \bigl(\hat x^{k-1}_{k-1}(t_i) - \hat x^k_{k-1}(t_i)\bigr)^2,
        & &\text{(29)} \\
    \mathcal{L}^{n+1}_{Data} &= \frac{1}{N}\sum_{i=1}^N
        \bigl|\hat x^n_n(t_i) - \hat x^{n+1}_n(t_i)\bigr|^2. & &\text{(38)}

Only the first is a genuine measurement fit.  The downstream terms are
"deterministic inter-cell consistency penalties; no Gaussian propagation
through the bank is assumed" --- comparing the two estimators' outputs
"enforces consistency at the :math:`x_{k-1}` coordinate and enables
fine-tuning of upstream weights once gradients are propagated through the
global loss (27)".

Physics terms
-------------
.. math:: \mathcal{L}^k_{Physics} = \frac{1}{N_r}\sum_{j=1}^{N_r}
          |r_k(\tau_j)|^2, \qquad \text{(24), (30), (39)}

Global loss
-----------
.. math:: \mathcal{L}^k_{Tot} = \sum_{m=2}^{k} e^{-\frac{k-m}{k}}
          \mathcal{L}^m_{Loc}, \qquad \text{(27), (42)}

used *only* during joint fine-tuning.  The weight is 1 on the current cell and
decays geometrically towards the head of the chain, so the cell being trained
dominates while upstream cells are still allowed to adapt.

All losses are averaged over the selected trajectory minibatch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class CellLoss:
    """The three scalars produced by one cell's local objective."""

    data: Tensor
    physics: Tensor
    local: Tensor

    def as_floats(self) -> dict[str, float]:
        return {
            "data": float(self.data),
            "physics": float(self.physics),
            "local": float(self.local),
        }


def data_loss(prediction: Tensor, target: Tensor) -> Tensor:
    r"""Mean squared discrepancy on the data grid, Eqs. (23)/(29)/(38).

    Parameters
    ----------
    prediction
        Cell ``k``'s reconstruction :math:`\hat x^k_{k-1}`, shape ``(B, N)``.
    target
        The upstream target: :math:`y` for ``k = 2``, otherwise
        :math:`\hat x^{k-1}_{k-1}`, shape ``(B, N)``.

    Returns
    -------
    Tensor
        Scalar, averaged over both time samples and trajectories.
    """
    if prediction.shape != target.shape:
        raise ValueError(
            f"shape mismatch between prediction {tuple(prediction.shape)} "
            f"and target {tuple(target.shape)}"
        )
    return torch.mean((target - prediction) ** 2)


def physics_loss(residual: Tensor) -> Tensor:
    r"""Mean squared physics residual on the collocation grid, Eqs. (24)/(30)/(39)."""
    return torch.mean(residual**2)


def local_loss(data: Tensor, physics: Tensor, lam: float) -> CellLoss:
    r""":math:`\mathcal{L}^k_{Loc} = \mathcal{L}^k_{Data} + \lambda \mathcal{L}^k_{Physics}`.

    Eqs. (22), (28) and (37).
    """
    if lam <= 0:
        raise ValueError(f"the physics weight lambda must be positive, got {lam}")
    return CellLoss(data=data, physics=physics, local=data + lam * physics)


def global_weights(cell_index: int) -> dict[int, float]:
    r"""The weights :math:`e^{-(k-m)/k}` of Eq. (27) for :math:`m = 2,\dots,k`.

    The current cell ``m = k`` has weight exactly 1.
    """
    if cell_index < 2:
        raise ValueError(f"cell index must be at least 2, got {cell_index}")
    return {
        m: math.exp(-(cell_index - m) / cell_index)
        for m in range(2, cell_index + 1)
    }


def total_loss(local_losses: dict[int, Tensor], cell_index: int) -> Tensor:
    r"""The weighted global loss :math:`\mathcal{L}^k_{Tot}`, Eqs. (27)/(42).

    Parameters
    ----------
    local_losses
        :math:`\mathcal{L}^m_{Loc}` for every ``m = 2..k``, keyed by cell index.
    cell_index
        ``k``.

    Returns
    -------
    Tensor
        Scalar.
    """
    weights = global_weights(cell_index)
    missing = set(weights) - set(local_losses)
    if missing:
        raise ValueError(
            f"global loss for cell {cell_index} needs local losses from cells "
            f"{sorted(weights)}, missing {sorted(missing)}"
        )
    terms = [weight * local_losses[m] for m, weight in weights.items()]
    return torch.stack(terms).sum()
