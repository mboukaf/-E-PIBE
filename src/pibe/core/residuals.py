r"""Physics residuals of the PIBE cells, Eqs. (25), (31) and (40).

Argument assembly
-----------------
Eq. (26) groups the continuous upstream arguments of cell ``k`` as

.. math::

    \mathcal{I}^x_{k-2} &:= \operatorname{col}\bigl(\hat x^2_1(t),
        \{\hat x^j_j(t)\}_{j=2}^{k-2}\bigr), \\
    \mathcal{I}^\theta_{k-2} &:= \operatorname{col}\bigl(
        \{\hat\theta^{j+1}_j(t)\}_{j=1}^{k-2}\bigr),

with empty indexed sequences omitted.  So :math:`\mathcal{I}^x_{k-2}`
"contains precisely the upstream state arguments :math:`x_1,\dots,x_{k-2}`,
while :math:`\mathcal{I}^\theta_{k-2}` contains precisely
:math:`\theta_1,\dots,\theta_{k-2}`".  Concretely, coordinate 1 comes from cell
2's *reconstructed* output and coordinate ``j >= 2`` from cell ``j``'s *new*
output, while :math:`\theta_j` comes from cell ``j+1``'s head.

The residual of cell ``k``, Eq. (31), is then

.. math::

    r_k(t) = \dot{\hat x}^k_{k-1}(t) - \hat x^k_k(t)
           - f_{k-1}\bigl(\mathcal{I}^x_{k-2}, \hat x^k_{k-1}(t),
                          \mathcal{I}^\theta_{k-2}, \hat\theta^k_{k-1}\bigr),

whose arguments are ordered exactly as :math:`f_{k-1}`'s signature
:math:`(x_1,\dots,x_{k-1},\theta_1,\dots,\theta_{k-1})`.  Note that
:math:`x_{k-1}` is supplied by the *current* cell's reconstruction
:math:`\hat x^k_{k-1}`, not by cell ``k-1``'s :math:`\hat x^{k-1}_{k-1}`:
"thus, :math:`r_k` depends only on (i) the frozen upstream estimates in
:math:`\mathcal{I}_{k-1}` and (ii) the current PINN outputs".

For ``k = 2`` both upstream sequences are empty and this reduces to Eq. (25),
:math:`r_2 = \dot{\hat x}^2_1 - \hat x^2_2 - f_1(\hat x^2_1, \hat\theta^2_1)`.

The final cell, Eq. (40), uses the full state vector of Eq. (123) and the
basis expansion of the disturbance:

.. math::

    r_{n+1}(t) = \dot{\hat x}^{n+1}_n(t)
               - f_n\bigl(\hat{\boldsymbol{x}}^{n+1}(t), \hat{\boldsymbol\theta}\bigr)
               - \Gamma_q(t)^\top \hat a.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

from pibe.basis.bspline import BSplineBasis
from pibe.core.cell import CellOutput
from pibe.systems.base import TriangularSystem

Grid = Literal["data", "coll"]


def _state_on(output: CellOutput, which: Literal["prev", "new"], grid: Grid) -> Tensor:
    """Pick one decoder output of one cell on one grid."""
    attribute = f"x_{which}_{grid}"
    value = getattr(output, attribute)
    if value is None:
        raise ValueError(
            f"cell output has no '{which}' coordinate on the '{grid}' grid "
            "(the final cell produces no new coordinate)"
        )
    return value


def upstream_states(
    outputs: dict[int, CellOutput], upto: int, grid: Grid = "coll"
) -> list[Tensor]:
    r"""Assemble :math:`(x_1, \dots, x_{\text{upto}})` from upstream cells.

    Implements :math:`\mathcal{I}^x_{\text{upto}}` of Eq. (26):
    :math:`x_1 = \hat x^2_1` is cell 2's reconstruction, and
    :math:`x_j = \hat x^j_j` for :math:`j \ge 2` is cell ``j``'s new
    coordinate.  Returns an empty list when ``upto <= 0``.
    """
    if upto <= 0:
        return []
    states = [_state_on(outputs[2], "prev", grid)]
    states.extend(_state_on(outputs[j], "new", grid) for j in range(2, upto + 1))
    assert len(states) == upto, (len(states), upto)
    return states


def upstream_thetas(outputs: dict[int, CellOutput], upto: int) -> list[Tensor]:
    r"""Assemble :math:`(\theta_1, \dots, \theta_{\text{upto}})` from upstream heads.

    Implements :math:`\mathcal{I}^\theta_{\text{upto}}` of Eq. (26):
    :math:`\theta_j = \hat\theta^{j+1}_j` comes from cell ``j+1``'s parameter
    head.  Each entry has shape ``(B, 1)``.
    """
    if upto <= 0:
        return []
    return [outputs[j + 1].head for j in range(1, upto + 1)]


def _stack_states(states: list[Tensor]) -> Tensor:
    """Stack ``(B, M)`` coordinates into ``(B, M, len(states))``."""
    return torch.stack(states, dim=-1)


def _stack_thetas(thetas: list[Tensor], time_steps: int) -> Tensor:
    """Broadcast ``(B, 1)`` parameters along time into ``(B, M, len(thetas))``."""
    expanded = [theta.expand(-1, time_steps) for theta in thetas]
    return torch.stack(expanded, dim=-1)


def state_residual(
    system: TriangularSystem,
    cell_index: int,
    current: CellOutput,
    outputs: dict[int, CellOutput],
    grid: Grid = "coll",
) -> Tensor:
    r"""The residual :math:`r_k` of Eq. (31) (Eq. (25) when ``k = 2``).

    Parameters
    ----------
    system
        Supplies :math:`f_{k-1}`.
    cell_index
        ``k``, with ``2 <= k <= n``.
    current
        Cell ``k``'s own output; must carry ``x_prev_dot_coll``.
    outputs
        Outputs of cells ``2..k-1``, keyed by cell index.
    grid
        Grid the residual is evaluated on; the physics loss uses ``"coll"``.

    Returns
    -------
    Tensor
        Shape ``(B, M)``.
    """
    if not 2 <= cell_index <= system.n:
        raise ValueError(
            f"state residual is defined for 2 <= k <= n = {system.n}, got {cell_index}"
        )
    derivative = current.x_prev_dot_coll
    if derivative is None:
        raise ValueError(
            f"cell {cell_index} was evaluated without its time derivative; "
            "the physics residual needs it"
        )
    x_new = _state_on(current, "new", grid)
    x_prev = _state_on(current, "prev", grid)

    # f_{k-1} takes (x_1, ..., x_{k-1}, theta_1, ..., theta_{k-1}).
    upto = cell_index - 2
    states = upstream_states(outputs, upto, grid) + [x_prev]
    thetas = upstream_thetas(outputs, upto) + [current.head]
    assert len(states) == cell_index - 1 == len(thetas), (len(states), len(thetas))

    time_steps = x_prev.shape[1]
    drift = system.f(
        cell_index - 1, _stack_states(states), _stack_thetas(thetas, time_steps)
    )
    return derivative - x_new - drift


def final_residual(
    system: TriangularSystem,
    basis: BSplineBasis,
    current: CellOutput,
    outputs: dict[int, CellOutput],
    t_grid: Tensor,
    grid: Grid = "coll",
) -> Tensor:
    r"""The residual :math:`r_{n+1}` of Eq. (40).

    Parameters
    ----------
    system
        Supplies :math:`f_n`.
    basis
        The disturbance basis :math:`\Gamma_q`.
    current
        The final cell's output; ``x_prev`` is the auxiliary
        :math:`\hat x^{n+1}_n` and ``head`` is :math:`\hat a`.
    outputs
        Outputs of cells ``2..n``.
    t_grid
        The times the residual is evaluated at, shape ``(M,)``.

    Returns
    -------
    Tensor
        Shape ``(B, M)``.
    """
    derivative = current.x_prev_dot_coll
    if derivative is None:
        raise ValueError(
            "the final cell was evaluated without its time derivative; "
            "the physics residual needs it"
        )
    x_final = _state_on(current, "prev", grid)

    # Eq. (123): x^{n+1} = col(x_1, {x_j}_{j=2}^{n-1}, x^{n+1}_n).
    states = upstream_states(outputs, system.n - 1, grid) + [x_final]
    thetas = upstream_thetas(outputs, system.n - 1)
    assert len(states) == system.n, (len(states), system.n)
    assert len(thetas) == system.theta_dim, (len(thetas), system.theta_dim)

    time_steps = x_final.shape[1]
    drift = system.f_last(
        _stack_states(states), _stack_thetas(thetas, time_steps)
    )
    return derivative - drift - disturbance_estimate(basis, current.head, t_grid)


def disturbance_estimate(basis: BSplineBasis, a_hat: Tensor, t_grid: Tensor) -> Tensor:
    r"""The estimated disturbance :math:`\hat d(t) = \Gamma_q(t)^\top \hat a`, Eq. (41).

    Parameters
    ----------
    basis
        The basis :math:`\Gamma_q`.
    a_hat
        Estimated coefficients, shape ``(B, q)``.
    t_grid
        Times, shape ``(M,)``.

    Returns
    -------
    Tensor
        Shape ``(B, M)``.

    Notes
    -----
    This recovers only the structured component of Eq. (2); the remainder
    :math:`r_q` is the irreducible finite-basis approximation error, bounded by
    :math:`\varepsilon_{d,q}` in (4).
    """
    if a_hat.shape[-1] != basis.q:
        raise ValueError(
            f"expected {basis.q} coefficients, got {a_hat.shape[-1]}"
        )
    gamma = basis.evaluate(t_grid)  # (M, q)
    return torch.einsum("mq,bq->bm", gamma, a_hat)
