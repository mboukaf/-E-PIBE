r"""The two-phase schedule of Algorithm 1.

For each cell ``k``, the per-cell iteration budget :math:`N_{tot}` is split at
the local pre-training cutoff :math:`N_{par}`, with
:math:`0 < N_{par} < N_{tot}`:

``i < N_par`` --- local pre-training (lines 5-9)
    Freeze :math:`\{\Theta^m_{PINN}\}_{m=2}^{k-1}`, build
    :math:`\mathbf{U}_{k-1}` from detached upstream outputs, and update *only*
    :math:`\Theta^k_{PINN}` with :math:`\nabla_{\Theta^k_{PINN}}
    \mathcal{L}^k_{Loc}`.

``i >= N_par`` --- end-to-end fine-tuning (lines 10-15)
    Unfreeze :math:`\{\Theta^m_{PINN}\}_{m=2}^{k}`, evaluate the chain without
    detaching, and update every block with :math:`\nabla \mathcal{L}^k_{Tot}`.

Freezing is applied through ``requires_grad_`` in addition to the ``no_grad``
evaluation of upstream cells in :meth:`~pibe.core.bank.EstimatorBank.forward`.
The two are redundant by design: the flag is what Algorithm 1 states, and the
context manager is what makes the detachment structural rather than a property
of the optimizer's parameter list.
"""

from __future__ import annotations

from torch import nn

from pibe.core.bank import EstimatorBank, Mode


def mode_for_iteration(iteration: int, n_par: int) -> Mode:
    """Which regime iteration ``i`` of a cell's budget belongs to."""
    if iteration < 0:
        raise ValueError(f"iteration must be non-negative, got {iteration}")
    return Mode.LOCAL if iteration < n_par else Mode.GLOBAL


def apply_phase(bank: EstimatorBank, target_cell: int, mode: Mode) -> list[nn.Parameter]:
    r"""Set every cell's ``requires_grad`` for the regime and return the active block.

    Parameters
    ----------
    bank
        The estimator bank.
    target_cell
        ``k``, the cell currently being trained.
    mode
        The regime.

    Returns
    -------
    list[nn.Parameter]
        :math:`\Theta^k_{PINN}` under :attr:`~pibe.core.bank.Mode.LOCAL`, and
        :math:`\mathfrak{W}_{2:k}` under :attr:`~pibe.core.bank.Mode.GLOBAL`.

    Notes
    -----
    Cells downstream of ``k`` are always frozen: they have not been reached by
    the outer loop and take no part in :math:`\mathcal{L}^k_{Tot}`.
    """
    if mode is Mode.LOCAL:
        # Lines 6 and 9: upstream frozen, only the current cell updated.
        bank.set_trainable_upto(target_cell - 1, False)
        bank.cell(target_cell).set_trainable(True)
        active = list(bank.cell_parameters(target_cell))
    else:
        # Line 11: the whole prefix of the chain becomes trainable.
        bank.set_trainable_upto(target_cell, True)
        active = list(bank.parameters_upto(target_cell))

    for k in range(target_cell + 1, bank.final_index + 1):
        bank.cell(k).set_trainable(False)

    return active
