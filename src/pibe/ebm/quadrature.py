r"""Fixed quadrature for the one-dimensional partition function, Eq. (59).

Training an energy-based model normally requires Monte-Carlo machinery ---
Langevin dynamics or contrastive divergence --- because the partition function

.. math:: Z(E_\zeta) = \int e^{-\beta E_\zeta(\tilde y)}\,d\tilde y

is intractable in high dimension.  Remark 4 observes that EPIBE never needs
that: every cell :math:`k \in \{2,\dots,n+1\}` models a **scalar** consistency
residual on the fixed compact support :math:`\mathcal{R}_k = [-\bar\rho_k,
\bar\rho_k]` of Assumption 3, so

.. math:: Z_k(\tilde E_{\zeta_k})
          = \int_{-\bar\rho_k}^{\bar\rho_k} e^{-\beta \tilde E_{\zeta_k}(\xi)}\,d\xi

is a one-dimensional integral over a known interval.  A fixed Gauss--Legendre
rule evaluates it at constant cost per gradient step, and the *same* rule
supplies

.. math:: \nabla_{\zeta_k}\log Z_k
          = -\beta\,\mathbb{E}_{\xi\sim p_{k,\zeta_k}}[\nabla_{\zeta_k}\tilde E_{\zeta_k}(\xi)],

which is what backpropagation through the negative log-likelihood needs.  That
gradient identity is not implemented by hand here: differentiating the
:func:`log_partition` estimate below *is* the quadrature approximation of it,
so autograd produces the right quantity automatically.

Composite rather than single-panel
----------------------------------
Remark 4 suggests :math:`10^2`--:math:`10^3` nodes.  A single Gauss--Legendre
panel of that order spreads its nodes over the whole of :math:`\mathcal{R}_k`,
which is fine for a smooth energy but poor when the residual density is
concentrated on a small part of a conservatively chosen support --- exactly the
regime of a low-noise run, where :math:`\bar\rho_k` comes from the decoder box
while the residuals themselves are three orders of magnitude smaller.  Splitting
the interval into equal panels and applying a modest rule on each keeps the
node *spacing* uniform and fine, which is what resolves a narrow peak wherever
it sits.  The total node count is ``panels * nodes_per_panel``, so the cost
statement of Remark 4 is unchanged.

Everything is computed in the log domain: :func:`log_partition` returns
:math:`\log Z_k` via ``logsumexp``, never :math:`Z_k` itself, because
:math:`e^{-\beta\tilde E}` spans :math:`e^{\pm\beta B_{E,k}}` and overflows in
the middle of the admissible energy range.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn


class CompositeGaussLegendre(nn.Module):
    r"""A fixed composite Gauss--Legendre rule on :math:`[-\bar\rho, \bar\rho]`.

    Nodes and log-weights are buffers, so they follow the module through
    ``.to()`` and are written to checkpoints alongside the energy parameters ---
    the quadrature is part of the model definition, not a runtime detail, and a
    reloaded checkpoint must integrate against the same rule it was trained on.

    Parameters
    ----------
    radius
        :math:`\bar\rho`, the support half-width from Assumption 3.
    panels
        Number of equal subintervals.
    nodes_per_panel
        Gauss--Legendre order on each subinterval.  A rule of order ``m`` is
        exact for polynomials of degree ``2m - 1`` on its panel.
    dtype
        Working precision; the nodes are built in double and cast.
    """

    def __init__(
        self,
        radius: float,
        panels: int = 64,
        nodes_per_panel: int = 16,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        if radius <= 0:
            raise ValueError(f"the support radius must be positive, got {radius}")
        if panels < 1 or nodes_per_panel < 1:
            raise ValueError(
                f"invalid rule: panels={panels}, nodes_per_panel={nodes_per_panel}"
            )
        self.radius = float(radius)
        self.panels = int(panels)
        self.nodes_per_panel = int(nodes_per_panel)

        # Reference rule on [-1, 1], then affinely mapped onto each panel.
        reference_nodes, reference_weights = np.polynomial.legendre.leggauss(
            self.nodes_per_panel
        )
        edges = np.linspace(-self.radius, self.radius, self.panels + 1)
        half = 0.5 * (edges[1] - edges[0])
        centres = 0.5 * (edges[:-1] + edges[1:])

        nodes = (centres[:, None] + half * reference_nodes[None, :]).reshape(-1)
        weights = np.broadcast_to(
            half * reference_weights[None, :], (self.panels, self.nodes_per_panel)
        ).reshape(-1)

        self.register_buffer("nodes", torch.as_tensor(nodes, dtype=dtype))
        self.register_buffer("log_weights", torch.as_tensor(np.log(weights), dtype=dtype))

    @property
    def n_nodes(self) -> int:
        """Total number of evaluation points, ``panels * nodes_per_panel``."""
        return self.panels * self.nodes_per_panel

    @property
    def spacing(self) -> float:
        """Mean node spacing --- the resolution a narrow density peak sees."""
        return 2.0 * self.radius / self.n_nodes

    def log_integral(self, log_integrand: Tensor) -> Tensor:
        r"""``log`` of :math:`\int f`, given :math:`\log f` at the nodes.

        Parameters
        ----------
        log_integrand
            :math:`\log f` evaluated at :attr:`nodes`, shape ``(..., n_nodes)``.

        Returns
        -------
        Tensor
            :math:`\log \int_{-\bar\rho}^{\bar\rho} f`, shape ``(...)``.

        Notes
        -----
        Gauss--Legendre weights are strictly positive, so the whole integrand
        can be carried in the log domain and combined with ``logsumexp``; there
        is no cancellation to worry about and no overflow at either end of the
        admissible energy range.
        """
        if log_integrand.shape[-1] != self.n_nodes:
            raise ValueError(
                f"expected {self.n_nodes} node values, got {log_integrand.shape[-1]}"
            )
        return torch.logsumexp(log_integrand + self.log_weights, dim=-1)

    def integrate(self, values: Tensor) -> Tensor:
        """:math:`\\int g` for a directly evaluated (signed) integrand.

        Used for the descriptive moments of Eq. (54), where the integrand
        :math:`\\xi p(\\xi)` changes sign and the log domain does not apply.
        """
        if values.shape[-1] != self.n_nodes:
            raise ValueError(
                f"expected {self.n_nodes} node values, got {values.shape[-1]}"
            )
        return torch.sum(values * torch.exp(self.log_weights), dim=-1)

    def extra_repr(self) -> str:  # pragma: no cover - trivial
        return (
            f"radius={self.radius:g}, panels={self.panels}, "
            f"nodes_per_panel={self.nodes_per_panel}, spacing={self.spacing:.3g}"
        )
