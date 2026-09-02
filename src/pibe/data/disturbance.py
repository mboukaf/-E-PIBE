r"""True disturbances driving the last state equation.

Eq. (2) decomposes the disturbance on the estimation horizon as

.. math:: d(t) = \Gamma_q(t)^\top a + r_q(t),

where :math:`\Gamma_q` is the known basis, :math:`a \in \mathcal{A}` the unknown
constant coefficient vector, and :math:`r_q \in L^\infty([0,T])` the part the
selected basis cannot represent, bounded by :math:`\varepsilon_{d,q}` in (4).

:class:`BasisDisturbance` generates the exactly recoverable case
(:math:`r_q = 0`, hence :math:`\varepsilon_{d,q} = 0`); passing a ``remainder``
introduces a controlled model mismatch for studying the bounds of Section 4.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

from pibe.basis.base import DisturbanceBasis


class BasisDisturbance:
    r"""The disturbance :math:`d = \Gamma_q^\top a + r_q`.

    Parameters
    ----------
    basis
        The basis :math:`\Gamma_q`.
    coefficients
        The true ``a``, shape ``(q,)``.
    remainder
        Optional :math:`r_q`, a callable ``t -> (...)``.  Defaults to zero, the
        case in which the finite-dimensional disturbance class is recovered
        exactly.
    """

    def __init__(
        self,
        basis: DisturbanceBasis,
        coefficients: Tensor,
        remainder: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        coefficients = torch.as_tensor(coefficients, dtype=basis.dtype).reshape(-1)
        if coefficients.numel() != basis.q:
            raise ValueError(
                f"expected {basis.q} coefficients for this basis, got {coefficients.numel()}"
            )
        self.basis = basis
        self.coefficients = coefficients.to(basis.device)
        self.remainder = remainder

    def __call__(self, t: Tensor) -> Tensor:
        value = self.basis.evaluate(t) @ self.coefficients
        if self.remainder is not None:
            value = value + self.remainder(t)
        return value

    def structured_part(self, t: Tensor) -> Tensor:
        r"""Only :math:`\Gamma_q(t)^\top a`, the part the estimator can represent."""
        return self.basis.evaluate(t) @ self.coefficients

    def remainder_bound(self, t: Tensor) -> float:
        r"""Empirical :math:`\varepsilon_{d,q} \approx \|r_q\|_{L^\infty}` on ``t``."""
        if self.remainder is None:
            return 0.0
        return float(self.remainder(t).abs().max())

    def to(self, device: torch.device | str | None = None, dtype: torch.dtype | None = None):
        self.basis = self.basis.to(device=device, dtype=dtype)
        self.coefficients = self.coefficients.to(device=device, dtype=dtype)
        return self

    def __repr__(self) -> str:  # pragma: no cover - trivial
        kind = "exact" if self.remainder is None else "with remainder"
        return f"BasisDisturbance(q={self.basis.q}, {kind})"


def sample_coefficients(
    basis: DisturbanceBasis,
    bounds: Tensor | tuple[float, float],
    generator: torch.Generator | None = None,
) -> Tensor:
    r"""Draw ``a`` uniformly from the box :math:`\mathcal{A}`.

    Parameters
    ----------
    basis
        Supplies ``q``, the dtype and the device.
    bounds
        Either a ``(q, 2)`` tensor of per-component ``(lo, hi)`` pairs, or a
        single ``(lo, hi)`` pair applied to every coefficient.
    """
    if isinstance(bounds, tuple):
        low, high = bounds
        if not high > low:
            raise ValueError(f"require lo < hi for the coefficient box, got {bounds}")
        box = torch.tensor(
            [[low, high]], dtype=basis.dtype, device=basis.device
        ).expand(basis.q, 2)
    else:
        box = torch.as_tensor(bounds, dtype=basis.dtype, device=basis.device)
        if box.shape != (basis.q, 2):
            raise ValueError(
                f"coefficient bounds must have shape ({basis.q}, 2), "
                f"got {tuple(box.shape)}"
            )
        if bool((box[:, 0] >= box[:, 1]).any()):
            raise ValueError("every coefficient bound must satisfy lo < hi")

    u = torch.rand(basis.q, generator=generator, dtype=basis.dtype, device=basis.device)
    return box[:, 0] + u * (box[:, 1] - box[:, 0])


class ChirpRemainder:
    r"""The out-of-class disturbance component
    :math:`r_q(t) = \varepsilon_{d,q}\sin(\text{rate}\cdot t^2)`.

    A quadratic-phase chirp is a deliberate choice for probing model mismatch:
    its instantaneous frequency grows without bound, so it cannot be
    represented by a fixed finite basis at any :math:`q`.  The amplitude is
    exactly the constant :math:`\varepsilon_{d,q}` bounding
    :math:`\|r_q\|_{L^\infty(0,T)}` in Eq. (4), which makes it a clean knob for
    studying how the bounds of Section 4 degrade with model mismatch.
    """

    def __init__(self, amplitude: float, rate: float = 0.15) -> None:
        if amplitude < 0:
            raise ValueError(f"amplitude must be non-negative, got {amplitude}")
        self.amplitude = float(amplitude)
        self.rate = float(rate)

    def __call__(self, t: Tensor) -> Tensor:
        return self.amplitude * torch.sin(self.rate * t**2)

    @property
    def sup_norm(self) -> float:
        r"""The exact :math:`\varepsilon_{d,q} = \|r_q\|_{L^\infty}` bound."""
        return self.amplitude

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"ChirpRemainder(amplitude={self.amplitude}, rate={self.rate})"


class FunctionDisturbance:
    """Wrap an arbitrary ``t -> d(t)`` callable as a disturbance.

    Useful for driving the system with a signal that lies *outside* the span of
    :math:`\\Gamma_q`, so that :math:`r_q \\ne 0` and the exact-recovery claim
    below Eq. (4) no longer applies.
    """

    def __init__(self, fn: Callable[[Tensor], Tensor], label: str = "custom") -> None:
        self.fn = fn
        self.label = label

    def __call__(self, t: Tensor) -> Tensor:
        return self.fn(t)

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"FunctionDisturbance({self.label!r})"
