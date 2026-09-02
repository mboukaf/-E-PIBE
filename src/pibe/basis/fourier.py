r"""Trigonometric basis for the structured disturbance component.

.. math::

    \Gamma_q(t) = \bigl[\sin(\Omega t)\ \cos(\Omega t)\
                        \sin(2\Omega t)\ \cos(2\Omega t)\ \cdots\
                        \sin(H\Omega t)\ \cos(H\Omega t)\bigr]^\top,
    \qquad q = 2H,

so that :math:`d_q(t) = \Gamma_q(t)^\top a` is a real trigonometric polynomial
with fundamental frequency :math:`\Omega`.  Every component is :math:`C^\infty`,
comfortably satisfying the :math:`\gamma_i \in C^1([0,T])` regularity that
Eq. (2) requires.

Conditioning
------------
When the horizon spans a whole number of fundamental periods,
:math:`T = m\,(2\pi/\Omega)` for integer :math:`m \ge 1`, orthogonality gives

.. math:: W_\Gamma = \int_0^T \Gamma_q\Gamma_q^\top\,dt = \frac{T}{2} I_q,

so Eq. (3) holds with :math:`\underline{\gamma}_d = T/2` and the basis is
perfectly conditioned --- the best possible constant for a basis of this scale.
Off-resonance horizons still give a positive :math:`\underline{\gamma}_d`, but
:math:`W_\Gamma` is no longer diagonal and the constant degrades; the
constructor warns when the horizon is not commensurate.

Note that Section 5.1 of the paper describes a B-spline basis
(:class:`~pibe.basis.bspline.BSplineBasis`).  This module exists because a
trigonometric basis is the natural choice for a periodic disturbance and gives
an exactly known :math:`W_\Gamma`; the two are interchangeable behind
:class:`~pibe.basis.base.DisturbanceBasis`.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from pibe.basis.base import DisturbanceBasis
from pibe.utils.logging import get_logger

logger = get_logger(__name__)


class FourierBasis(DisturbanceBasis):
    r"""Sine/cosine pairs at harmonics of :math:`\Omega`.

    Parameters
    ----------
    q
        Number of basis functions; must be even, ``q = 2H``.
    omega
        Fundamental angular frequency :math:`\Omega`.  If omitted it is set to
        :math:`2\pi / (t_{end} - t_{start})`, one period over the horizon.
    t_start, t_end
        Horizon endpoints.  Phases are measured from ``t_start``.
    """

    def __init__(
        self,
        q: int = 4,
        omega: float | None = None,
        t_start: float = 0.0,
        t_end: float = 1.0,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
    ) -> None:
        if q % 2 != 0:
            raise ValueError(
                f"a sine/cosine basis needs an even q = 2H, got q={q}"
            )
        super().__init__(q=q, t_start=t_start, t_end=t_end, dtype=dtype, device=device)

        self.n_harmonics = self._q // 2
        if omega is None:
            omega = 2.0 * math.pi / self.horizon
        if omega <= 0:
            raise ValueError(f"omega must be positive, got {omega}")
        self.omega = float(omega)

        periods = self.horizon / (2.0 * math.pi / self.omega)
        self.periods = periods
        if abs(periods - round(periods)) > 1e-9:
            logger.warning(
                "horizon spans %.4f fundamental periods (not an integer): "
                "W_Gamma is not diagonal and lambda_min will be below T/2",
                periods,
            )

        self._harmonics = torch.arange(
            1, self.n_harmonics + 1, dtype=dtype, device=self.device
        )

    # ------------------------------------------------------------------
    # evaluation
    # ------------------------------------------------------------------

    def _phases(self, t: Tensor) -> Tensor:
        """Harmonic phases ``h * omega * (t - t_start)``, shape ``(..., H)``."""
        t = torch.as_tensor(t, dtype=self._dtype, device=self._device)
        shifted = (t - self.t_start).unsqueeze(-1)
        return self.omega * shifted * self._harmonics

    @staticmethod
    def _interleave(sine: Tensor, cosine: Tensor) -> Tensor:
        """Interleave into ``[sin_1, cos_1, sin_2, cos_2, ...]``."""
        stacked = torch.stack([sine, cosine], dim=-1)
        return stacked.reshape(*sine.shape[:-1], 2 * sine.shape[-1])

    def evaluate(self, t: Tensor) -> Tensor:
        r"""Evaluate :math:`\Gamma_q(t)`; ``(...)`` in, ``(..., q)`` out."""
        phase = self._phases(t)
        return self._interleave(torch.sin(phase), torch.cos(phase))

    def derivative(self, t: Tensor) -> Tensor:
        r"""Evaluate :math:`\dot\Gamma_q(t)`.

        :math:`\frac{d}{dt}\sin(h\Omega t) = h\Omega\cos(h\Omega t)` and
        :math:`\frac{d}{dt}\cos(h\Omega t) = -h\Omega\sin(h\Omega t)`.
        """
        phase = self._phases(t)
        scale = self.omega * self._harmonics
        return self._interleave(
            scale * torch.cos(phase), -scale * torch.sin(phase)
        )

    # ------------------------------------------------------------------
    # conditioning
    # ------------------------------------------------------------------

    def _gram_panels(self) -> int:
        """Enough panels to resolve the highest harmonic on every period."""
        return max(8, 4 * self.n_harmonics * max(1, round(self.periods)))

    def analytic_gram(self) -> Tensor | None:
        r"""The exact :math:`W_\Gamma = (T/2) I_q`, or ``None`` off resonance.

        Available only when the horizon spans a whole number of fundamental
        periods, in which case distinct harmonics are orthogonal and each
        squared component integrates to :math:`T/2`.
        """
        if abs(self.periods - round(self.periods)) > 1e-9:
            return None
        return 0.5 * self.horizon * torch.eye(
            self._q, dtype=self._dtype, device=self._device
        )

    # ------------------------------------------------------------------
    # placement
    # ------------------------------------------------------------------

    def _move(self, device: torch.device, dtype: torch.dtype) -> None:
        self._harmonics = self._harmonics.to(device=device, dtype=dtype)

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (
            f"FourierBasis(q={self._q}, H={self.n_harmonics}, "
            f"omega={self.omega:.6g}, horizon=[{self.t_start:g}, {self.t_end:g}], "
            f"periods={self.periods:g})"
        )
