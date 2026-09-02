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
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class BoxReparameterization(nn.Module):
    r"""Map :math:`\mathbb{R}^m` onto the open box :math:`\prod_j (\ell_j, u_j)`.

    Parameters
    ----------
    bounds
        Shape ``(m, 2)`` with columns ``(lo, hi)``.
    """

    def __init__(self, bounds: Tensor) -> None:
        super().__init__()
        bounds = torch.as_tensor(bounds)
        if bounds.ndim != 2 or bounds.shape[1] != 2:
            raise ValueError(f"bounds must have shape (m, 2), got {tuple(bounds.shape)}")
        if bool((bounds[:, 0] >= bounds[:, 1]).any()):
            raise ValueError("every bound must satisfy lo < hi")
        self.register_buffer("center", 0.5 * (bounds[:, 0] + bounds[:, 1]))
        self.register_buffer("radius", 0.5 * (bounds[:, 1] - bounds[:, 0]))

    @property
    def dim(self) -> int:
        return int(self.center.numel())

    def forward(self, z: Tensor) -> Tensor:
        """Apply the reparameterization to ``z`` of shape ``(..., m)``."""
        if z.shape[-1] != self.dim:
            raise ValueError(
                f"expected last dimension {self.dim}, got {z.shape[-1]}"
            )
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


def make_output_map(bounds: Tensor | None, dim: int) -> nn.Module:
    """Return a :class:`BoxReparameterization` if ``bounds`` is given, else identity."""
    if bounds is None:
        return UnboundedOutput(dim)
    bounds = torch.as_tensor(bounds)
    if bounds.shape[0] != dim:
        raise ValueError(f"expected {dim} bound rows, got {bounds.shape[0]}")
    return BoxReparameterization(bounds)


def box_from_interval(lo: float, hi: float, dim: int, dtype: torch.dtype) -> Tensor:
    """Replicate a scalar interval into a ``(dim, 2)`` bounds tensor."""
    return torch.tensor([[lo, hi]], dtype=dtype).expand(dim, 2).clone()
