r"""Bounded output reparameterizations.

Section 3.1 constrains every state decoder and parameter head to the
corresponding admissible set:

    "All state decoders and parameter heads are constrained to the
    corresponding admissible sets :math:`\mathcal{X}_j` and :math:`\Theta_j`;
    bounded output reparameterizations may be used when these sets are
    intervals."

and, for the final cell, "The output of :math:`\mathcal{Q}_{n+1}` is constrained
to the admissible set :math:`\mathcal{A}`, for instance through an output
reparameterization when :math:`\mathcal{A}` is a box."

For an interval :math:`[\ell, u]` we use

.. math:: \pi(z) = c + r\tanh(z), \qquad c = \tfrac{\ell + u}{2},
          \quad r = \tfrac{u - \ell}{2},

which is smooth (so the decoder stays :math:`C^2` in :math:`t`, as Eq. (21)
requires), surjective onto the open interval, and has bounded derivative
:math:`r`, contributing the factor picked up by the output-scale term in the
Lipschitz constants :math:`L_{\mathcal{S}_k}, L_{\mathcal{Q}_k}` of (75).

One numerical caveat: for :math:`|z| \gtrsim 19` in float64, ``tanh``
saturates to exactly :math:`\pm 1`, so the output attains the boundary and its
derivative is exactly zero.  A cell driven into saturation therefore stops
receiving gradient on that coordinate.  This is not reachable from ordinary
training dynamics, but it is the failure mode to look for if a state or
parameter estimate pins to the edge of its admissible interval and stays
there --- the usual cause is admissible bounds that are too tight for the true
trajectory, which also violates Assumption 1.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class BoxReparameterization(nn.Module):
    r"""Map :math:`\mathbb{R}^m` onto the open box :math:`\prod_j (\ell_j, u_j)`.

    Parameters
    ----------
    bounds
        Shape ``(m, 2)`` with columns ``(lo, hi)``.
    """

    def __init__(self, bounds: Tensor, saturation_limit: float | None = None) -> None:
        super().__init__()
        bounds = torch.as_tensor(bounds)
        if bounds.ndim != 2 or bounds.shape[1] != 2:
            raise ValueError(f"bounds must have shape (m, 2), got {tuple(bounds.shape)}")
        if bool((bounds[:, 0] >= bounds[:, 1]).any()):
            raise ValueError("every bound must satisfy lo < hi")
        self.register_buffer("center", 0.5 * (bounds[:, 0] + bounds[:, 1]))
        self.register_buffer("radius", 0.5 * (bounds[:, 1] - bounds[:, 0]))
        self.saturation_limit = (
            None if saturation_limit is None else float(saturation_limit)
        )

    @property
    def dim(self) -> int:
        return int(self.center.numel())

    @property
    def gradient_floor(self) -> float:
        r""":math:`\operatorname{sech}^2(A)`, the smallest gradient multiplier."""
        if self.saturation_limit is None:
            return 0.0
        return float(1.0 - math.tanh(self.saturation_limit) ** 2)

    def forward(self, z: Tensor) -> Tensor:
        r"""Apply the reparameterization to ``z`` of shape ``(..., m)``.

        With ``saturation_limit`` set, ``z`` is clamped in the forward pass while
        the backward pass sees the identity --- a straight-through clamp.  The
        output is then confined to :math:`c \pm r\tanh(A)` and, crucially, the
        gradient multiplier is floored at :math:`\operatorname{sech}^2(A)`
        instead of decaying as :math:`4e^{-2|z|}`.

        Without it this map has a trap that is fatal rather than merely slow.
        Nothing bounds :math:`z`, so a head pushed towards the edge of its
        admissible set keeps going; by :math:`|z| \approx 18` the multiplier is
        :math:`4\times 10^{-16}`, below float64 resolution, and the unit can
        never come back however wrong it is.  Measured in a collapsed run: the
        state decoder at :math:`|z| = 18.4` and both parameter heads pinned, with
        gradient norms of :math:`10^{-7}`.  The floor makes that recoverable ---
        Adam normalizes per parameter, so a small but *honest* gradient still
        produces full-size steps, while :math:`4\times 10^{-16}` is numerical
        noise.

        A saturating unit is still a signal that something is wrong upstream;
        this keeps it from becoming permanent.
        """
        if z.shape[-1] != self.dim:
            raise ValueError(
                f"expected last dimension {self.dim}, got {z.shape[-1]}"
            )
        if self.saturation_limit is not None:
            limit = self.saturation_limit
            z = z + (z.clamp(-limit, limit) - z).detach()
        return self.center + self.radius * torch.tanh(z)

    def extra_repr(self) -> str:  # pragma: no cover - trivial
        lo = (self.center - self.radius).tolist()
        hi = (self.center + self.radius).tolist()
        return f"box={list(zip(lo, hi))}"


class UnboundedOutput(nn.Module):
    """Identity, for admissible sets that are not boxed."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)

    def forward(self, z: Tensor) -> Tensor:
        return z

    def extra_repr(self) -> str:  # pragma: no cover - trivial
        return f"dim={self.dim}"


def make_output_map(
    bounds: Tensor | None, dim: int, saturation_limit: float | None = None
) -> nn.Module:
    """Return a :class:`BoxReparameterization` if ``bounds`` is given, else identity."""
    if bounds is None:
        return UnboundedOutput(dim)
    bounds = torch.as_tensor(bounds)
    if bounds.shape[0] != dim:
        raise ValueError(f"expected {dim} bound rows, got {bounds.shape[0]}")
    return BoxReparameterization(bounds, saturation_limit=saturation_limit)


def box_from_interval(lo: float, hi: float, dim: int, dtype: torch.dtype) -> Tensor:
    """Replicate a scalar interval into a ``(dim, 2)`` bounds tensor."""
    return torch.tensor([[lo, hi]], dtype=dtype).expand(dim, 2).clone()
