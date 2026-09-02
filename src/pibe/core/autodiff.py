r"""Time derivatives of decoder outputs by automatic differentiation.

Eqs. (21) and (36) define

.. math:: \dot{\hat x}^{k,\ell}_{k-1}(t)
          = \frac{\partial}{\partial t}\mathcal{S}_k(t, \mathbf{z}^\ell_{k-1}),

"computed by automatic differentiation while :math:`\mathbf{z}^\ell_{k-1}` is
held fixed".  This module provides the two primitives that make that exact.

Two correctness requirements drive the implementation.

**Independent time entries.**  :func:`make_time_input` materializes a distinct
tensor element per ``(trajectory, collocation point)`` pair.  A broadcast or
``expand``-ed grid would share storage across the batch, and the reverse pass
would then accumulate :math:`\sum_b \partial_t \hat x[b, m]` into every entry
instead of the per-trajectory derivative.

**Graph retention.**  :func:`time_derivative` passes ``create_graph=True`` by
default.  The physics loss is a function of :math:`\dot{\hat x}`, so the
parameter gradient :math:`\nabla_{\Theta} \mathcal{L}_{Physics}` differentiates
*through* this derivative; without the retained graph the physics term would
contribute no gradient at all.
"""

from __future__ import annotations

import torch
from torch import Tensor


def make_time_input(
    t_grid: Tensor, batch_size: int, requires_grad: bool = True
) -> Tensor:
    """Broadcast a shared time grid to a per-trajectory differentiable input.

    Parameters
    ----------
    t_grid
        Shared grid, shape ``(M,)``.
    batch_size
        ``B``, the number of trajectories in the minibatch.
    requires_grad
        Whether the result is a differentiable leaf.  Set ``False`` on grids
        whose derivative is never taken (e.g. the data grid), which avoids
        building an unused graph.

    Returns
    -------
    Tensor
        Shape ``(B, M, 1)``, contiguous with independent storage per entry.
    """
    if t_grid.ndim != 1:
        raise ValueError(f"t_grid must be 1-D, got shape {tuple(t_grid.shape)}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    # expand() aliases storage; clone() gives each (b, m) entry its own element,
    # which is what makes the per-trajectory derivative well defined.
    t = t_grid.view(1, -1, 1).expand(batch_size, -1, 1).clone()
    if requires_grad:
        t.requires_grad_(True)
    return t


def time_derivative(
    output: Tensor, t: Tensor, create_graph: bool = True
) -> Tensor:
    r"""Differentiate a decoder output with respect to time.

    Parameters
    ----------
    output
        A single decoder output coordinate, shape ``(B, M)``.
    t
        The time input it was evaluated at, shape ``(B, M, 1)``, as produced by
        :func:`make_time_input` with ``requires_grad=True``.
    create_graph
        Retain the graph so the result is itself differentiable with respect to
        the network parameters.  Required during training.

    Returns
    -------
    Tensor
        :math:`\partial_t` of ``output``, shape ``(B, M)``.

    Notes
    -----
    Differentiating ``output.sum()`` yields the elementwise derivative rather
    than a sum of cross terms because ``output[b, m]`` depends on ``t`` only
    through ``t[b, m]``: the latent code is constant along the time axis and
    the decoder acts pointwise in ``t``.  Hence

    .. math:: \frac{\partial}{\partial t[b,m]} \sum_{b',m'} \text{output}[b',m']
              = \frac{\partial\,\text{output}[b,m]}{\partial t[b,m]},

    and one reverse pass recovers the whole derivative field.
    """
    if output.ndim != 2:
        raise ValueError(f"output must have shape (B, M), got {tuple(output.shape)}")
    if t.ndim != 3 or t.shape[-1] != 1:
        raise ValueError(f"t must have shape (B, M, 1), got {tuple(t.shape)}")
    if output.shape != t.shape[:2]:
        raise ValueError(
            f"output shape {tuple(output.shape)} does not match time grid "
            f"{tuple(t.shape[:2])}"
        )
    if not t.requires_grad:
        raise RuntimeError(
            "time input does not require grad; build it with "
            "make_time_input(..., requires_grad=True) before differentiating"
        )

    (grad,) = torch.autograd.grad(
        outputs=output.sum(),
        inputs=t,
        create_graph=create_graph,
        retain_graph=True,
    )
    if grad is None:  # pragma: no cover - decoder always depends on t
        raise RuntimeError("decoder output is independent of t; cannot differentiate")
    return grad[..., 0]
