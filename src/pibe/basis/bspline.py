r"""B-spline basis for the structured disturbance component.

Implements Definition 1 of the paper.  Given a nondecreasing knot vector
:math:`\Xi = \{\xi_0,\dots,\xi_{q+m}\}` and a degree :math:`m \ge 2`,

.. math::

    \gamma_{i,0}(t) &= \begin{cases} 1, & \xi_i \le t < \xi_{i+1} \\
                                     0, & \text{otherwise} \end{cases} \\
    \gamma_{i,\ell}(t) &= \frac{t - \xi_i}{\xi_{i+\ell} - \xi_i}\,
                          \gamma_{i,\ell-1}(t)
                        + \frac{\xi_{i+\ell+1} - t}{\xi_{i+\ell+1} - \xi_{i+1}}\,
                          \gamma_{i+1,\ell-1}(t)

for :math:`\ell = 1,\dots,m` and :math:`i = 0,\dots,q+m-\ell-1`, where a term
is zero whenever its denominator is zero.  Values at :math:`t = T` are defined
by continuous extension from the left.  The basis is collected as

.. math:: \Gamma_q(t) = [\gamma_{0,m}(t)\ \cdots\ \gamma_{q-1,m}(t)]^\top,

so that the structured disturbance is :math:`d_q(t) = \Gamma_q(t)^\top a`,
Eq. (138).

Linear independence on :math:`[0, T]`, Eq. (3), is checkable exactly via
:meth:`BSplineBasis.gram`.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from pibe.basis.base import DisturbanceBasis

_KNOT_STYLES = ("clamped", "uniform")


class BSplineBasis(DisturbanceBasis):
    r"""The vector of basis functions :math:`\Gamma_q` on :math:`[t_0, t_1]`.

    Parameters
    ----------
    q
        Number of basis functions, i.e. the dimension of the coefficient
        vector ``a``.  Must satisfy ``q >= degree + 1``.
    degree
        Spline degree :math:`m`.  The paper requires ``m >= 2`` so that simple
        interior knots give a :math:`C^{m-1} \subseteq C^1` basis, consistent
        with the regularity :math:`\gamma_i \in C^1([0,T])` imposed in (2).
    t_start, t_end
        Endpoints of the estimation horizon; ``(0, T)`` in the paper.
    knot_style
        ``"clamped"`` repeats the endpoints ``degree + 1`` times, so the basis
        is a partition of unity on the whole of ``[t_0, t_1]`` and reaches the
        boundary.  ``"uniform"`` uses equispaced knots extending past both
        ends, giving a cardinal basis with the same span.
    """

    def __init__(
        self,
        q: int,
        degree: int = 3,
        t_start: float = 0.0,
        t_end: float = 1.0,
        knot_style: str = "clamped",
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
    ) -> None:
        if degree < 2:
            raise ValueError(
                f"degree must be at least 2 for a C^1 basis (Eq. 2), got {degree}"
            )
        if q < degree + 1:
            raise ValueError(
                f"need q >= degree + 1 for a well-defined basis, got q={q}, degree={degree}"
            )
        if not t_end > t_start:
            raise ValueError(f"require t_start < t_end, got ({t_start}, {t_end})")
        if knot_style not in _KNOT_STYLES:
            raise ValueError(
                f"unknown knot_style {knot_style!r}, expected one of {_KNOT_STYLES}"
            )
        super().__init__(q=q, t_start=t_start, t_end=t_end, dtype=dtype, device=device)

        self._degree = int(degree)
        self.knot_style = knot_style

        self.knots = self._build_knots().to(device=self.device, dtype=dtype)

        # Index of the last knot span with positive length; used to extend the
        # level-0 indicators continuously from the left at t = t_end.
        nonempty = torch.nonzero(self.knots[:-1] < self.knots[1:], as_tuple=False)
        if nonempty.numel() == 0:  # pragma: no cover - excluded by constructor checks
            raise ValueError("knot vector has no span of positive length")
        self._last_span = int(nonempty[-1].item())

    # ------------------------------------------------------------------
    # properties
    # ------------------------------------------------------------------

    @property
    def degree(self) -> int:
        """Spline degree :math:`m`."""
        return self._degree

    # ------------------------------------------------------------------
    # evaluation
    # ------------------------------------------------------------------

    def evaluate(self, t: Tensor) -> Tensor:
        r"""Evaluate :math:`\Gamma_q(t)`.

        Parameters
        ----------
        t
            Times, shape ``(...)``.

        Returns
        -------
        Tensor
            Shape ``(..., q)``.
        """
        return self._recursion(t, self._degree)

    def derivative(self, t: Tensor) -> Tensor:
        r"""Evaluate :math:`\dot\Gamma_q(t)`.

        Uses the standard derivative formula

        .. math::

            \dot\gamma_{i,m}(t) = m\left[
                \frac{\gamma_{i,m-1}(t)}{\xi_{i+m} - \xi_i}
              - \frac{\gamma_{i+1,m-1}(t)}{\xi_{i+m+1} - \xi_{i+1}}\right],

        again with the convention that a term with zero denominator vanishes.
        Needed for the weak-form certificate of Proposition 5, not for
        training.

        Returns
        -------
        Tensor
            Shape ``(..., q)``.
        """
        m, q, xi = self._degree, self._q, self.knots
        lower = self._recursion(t, m - 1)  # (..., q + 1)

        d1 = xi[m : m + q] - xi[0:q]
        d2 = xi[m + 1 : m + 1 + q] - xi[1 : 1 + q]
        term1 = self._safe_ratio(lower[..., 0:q], d1)
        term2 = self._safe_ratio(lower[..., 1 : q + 1], d2)
        return m * (term1 - term2)

    # ------------------------------------------------------------------
    # linear independence, Eq. (3)
    # ------------------------------------------------------------------

    def gram(self, nodes_per_span: int | None = None, **_) -> Tensor:
        r"""The Gram matrix :math:`W_\Gamma = \int_{t_0}^{t_1} \Gamma_q \Gamma_q^\top\,dt`.

        Integrated span by span with Gauss-Legendre quadrature.  Since
        :math:`\Gamma_q` is a polynomial of degree ``m`` on each knot span,
        ``m + 1`` nodes integrate the degree-``2m`` product *exactly* (up to
        round-off), so the result is not a quadrature approximation.
        """
        n_nodes = nodes_per_span if nodes_per_span is not None else self._degree + 1
        nodes_np, weights_np = np.polynomial.legendre.leggauss(n_nodes)
        nodes = torch.as_tensor(nodes_np, dtype=self._dtype, device=self.device)
        weights = torch.as_tensor(weights_np, dtype=self._dtype, device=self.device)

        gram = torch.zeros(self._q, self._q, dtype=self._dtype, device=self.device)
        lo_all, hi_all = self.knots[:-1], self.knots[1:]
        for lo, hi in zip(lo_all.tolist(), hi_all.tolist()):
            if hi <= lo:
                continue
            lo = max(lo, self.t_start)
            hi = min(hi, self.t_end)
            if hi <= lo:
                continue
            half = 0.5 * (hi - lo)
            mid = 0.5 * (hi + lo)
            t = mid + half * nodes
            values = self.evaluate(t)  # (n_nodes, q)
            gram = gram + half * torch.einsum("g,gi,gj->ij", weights, values, values)
        return 0.5 * (gram + gram.T)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _move(self, device: torch.device, dtype: torch.dtype) -> None:
        self.knots = self.knots.to(device=device, dtype=dtype)

    def _build_knots(self) -> Tensor:
        """Return the knot vector, of length ``q + degree + 1``."""
        m, q = self._degree, self._q
        if self.knot_style == "clamped":
            n_interior = q - m - 1
            interior = np.linspace(self.t_start, self.t_end, n_interior + 2)[1:-1]
            knots = np.concatenate(
                [
                    np.full(m + 1, self.t_start),
                    interior,
                    np.full(m + 1, self.t_end),
                ]
            )
        else:  # uniform
            step = (self.t_end - self.t_start) / (q - m)
            knots = self.t_start + step * (np.arange(q + m + 1) - m)
        assert knots.shape == (q + m + 1,), knots.shape
        return torch.as_tensor(knots, dtype=torch.float64)

    def _recursion(self, t: Tensor, level: int) -> Tensor:
        """Run Cox-de Boor up to ``level``, returning ``(..., q + degree - level)``."""
        t = torch.as_tensor(t, dtype=self._dtype, device=self.device)
        batch_shape = t.shape
        flat = t.reshape(-1)
        xi = self.knots
        m, q = self._degree, self._q

        col = flat.unsqueeze(-1)  # (M, 1)
        # Level 0: half-open indicators over the q + m knot spans.
        table = ((col >= xi[:-1]) & (col < xi[1:])).to(self._dtype)

        # Continuous extension from the left at (and beyond) the last knot.
        at_right = flat >= xi[-1]
        if bool(at_right.any()):
            table = table.clone()
            table[at_right] = 0.0
            table[at_right, self._last_span] = 1.0

        for ell in range(1, level + 1):
            count = q + m - ell
            d1 = xi[ell : ell + count] - xi[0:count]
            d2 = xi[ell + 1 : ell + 1 + count] - xi[1 : 1 + count]
            num1 = col - xi[0:count]
            num2 = xi[ell + 1 : ell + 1 + count] - col
            left = self._safe_ratio(num1, d1) * table[..., 0:count]
            right = self._safe_ratio(num2, d2) * table[..., 1 : 1 + count]
            table = left + right

        return table.reshape(*batch_shape, table.shape[-1])

    @staticmethod
    def _safe_ratio(numerator: Tensor, denominator: Tensor) -> Tensor:
        """``numerator / denominator``, defined as zero where the denominator vanishes."""
        positive = denominator > 0
        safe = torch.where(positive, denominator, torch.ones_like(denominator))
        return torch.where(
            positive, numerator / safe, torch.zeros_like(numerator)
        )

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (
            f"BSplineBasis(q={self._q}, degree={self._degree}, "
            f"horizon=[{self.t_start}, {self.t_end}], knot_style={self.knot_style!r})"
        )
