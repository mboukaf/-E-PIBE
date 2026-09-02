r"""Common interface for disturbance bases :math:`\Gamma_q`.

Eq. (2) decomposes the disturbance as :math:`d(t) = \Gamma_q(t)^\top a +
r_q(t)`, where

.. math:: \Gamma_q(t) = [\gamma_0(t)\ \cdots\ \gamma_{q-1}(t)]^\top

is "a known vector of basis functions :math:`\gamma_i \in C^1([0,T])`".  The
only structural requirement the framework places on the basis is linear
independence on the horizon, Eq. (3):

.. math:: W_\Gamma := \int_0^T \Gamma_q(t)\Gamma_q(t)^\top\,dt
          \succeq \underline{\gamma}_d I_q, \qquad \underline{\gamma}_d > 0.

Anything satisfying that contract works.  Two are provided:
:class:`~pibe.basis.bspline.BSplineBasis` (Definition 1, the choice described
in Section 5.1) and :class:`~pibe.basis.fourier.FourierBasis` (a trigonometric
basis, which is perfectly conditioned when the horizon spans a whole number of
periods).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch
from torch import Tensor


class DisturbanceBasis(ABC):
    r"""Abstract base for :math:`\Gamma_q`.

    Parameters
    ----------
    q
        Number of basis functions, i.e. the dimension of ``a``.
    t_start, t_end
        Endpoints of the estimation horizon; ``(0, T)`` in the paper.
    """

    def __init__(
        self,
        q: int,
        t_start: float = 0.0,
        t_end: float = 1.0,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
    ) -> None:
        if q < 1:
            raise ValueError(f"q must be positive, got {q}")
        if not t_end > t_start:
            raise ValueError(f"require t_start < t_end, got ({t_start}, {t_end})")
        self._q = int(q)
        self.t_start = float(t_start)
        self.t_end = float(t_end)
        self._dtype = dtype
        self._device = torch.device(device) if device is not None else torch.device("cpu")

    # ------------------------------------------------------------------
    # properties
    # ------------------------------------------------------------------

    @property
    def q(self) -> int:
        """Number of basis functions."""
        return self._q

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def horizon(self) -> float:
        """``T - t_start``."""
        return self.t_end - self.t_start

    # ------------------------------------------------------------------
    # evaluation
    # ------------------------------------------------------------------

    @abstractmethod
    def evaluate(self, t: Tensor) -> Tensor:
        r"""Evaluate :math:`\Gamma_q(t)`; ``(...)`` in, ``(..., q)`` out."""

    @abstractmethod
    def derivative(self, t: Tensor) -> Tensor:
        r"""Evaluate :math:`\dot\Gamma_q(t)`; ``(...)`` in, ``(..., q)`` out."""

    def __call__(self, t: Tensor) -> Tensor:
        return self.evaluate(t)

    # ------------------------------------------------------------------
    # placement
    # ------------------------------------------------------------------

    def to(self, device: torch.device | str | None = None, dtype: torch.dtype | None = None):
        """Move the basis in place and return ``self``."""
        if dtype is not None:
            self._dtype = dtype
        if device is not None:
            self._device = torch.device(device)
        self._move(self._device, self._dtype)
        return self

    def _move(self, device: torch.device, dtype: torch.dtype) -> None:
        """Hook for subclasses holding tensors; default is a no-op."""

    # ------------------------------------------------------------------
    # linear independence, Eq. (3)
    # ------------------------------------------------------------------

    def _gram_panels(self) -> int:
        """Number of quadrature panels the default :meth:`gram` should use."""
        return 16

    def _gram_nodes(self) -> int:
        """Gauss-Legendre nodes per panel."""
        return 10

    def gram(
        self, n_panels: int | None = None, nodes_per_panel: int | None = None
    ) -> Tensor:
        r"""The Gram matrix :math:`W_\Gamma` of Eq. (3).

        Composite Gauss-Legendre over uniform panels.  Subclasses whose basis
        is piecewise polynomial override this with a panel decomposition that
        makes the quadrature exact.
        """
        n_panels = n_panels or self._gram_panels()
        n_nodes = nodes_per_panel or self._gram_nodes()
        nodes_np, weights_np = np.polynomial.legendre.leggauss(n_nodes)
        nodes = torch.as_tensor(nodes_np, dtype=self._dtype, device=self._device)
        weights = torch.as_tensor(weights_np, dtype=self._dtype, device=self._device)

        edges = torch.linspace(
            self.t_start, self.t_end, n_panels + 1, dtype=self._dtype, device=self._device
        )
        gram = torch.zeros(self._q, self._q, dtype=self._dtype, device=self._device)
        for lo, hi in zip(edges[:-1].tolist(), edges[1:].tolist()):
            if hi <= lo:
                continue
            half = 0.5 * (hi - lo)
            mid = 0.5 * (hi + lo)
            values = self.evaluate(mid + half * nodes)  # (n_nodes, q)
            gram = gram + half * torch.einsum("g,gi,gj->ij", weights, values, values)
        return 0.5 * (gram + gram.T)

    def min_gram_eigenvalue(self, **kwargs) -> float:
        r"""The constant :math:`\underline{\gamma}_d` of Eq. (3).

        Positive iff the basis functions are linearly independent on the
        horizon.
        """
        return float(torch.linalg.eigvalsh(self.gram(**kwargs)).min())

    def check_linear_independence(self, tol: float = 1e-12, **kwargs) -> float:
        r"""Assert Eq. (3) and return :math:`\underline{\gamma}_d`."""
        gamma_d = self.min_gram_eigenvalue(**kwargs)
        if gamma_d <= tol:
            raise ValueError(
                f"basis functions are not linearly independent on "
                f"[{self.t_start}, {self.t_end}]: "
                f"lambda_min(W_Gamma) = {gamma_d:.3e}"
            )
        return gamma_d
