r"""Measurement noise models.

Section 2 restricts attention to compactly supported measurement noise: every
admissible realization satisfies :math:`\|\omega\|_{L^\infty(0,T)} \le \bar w`.
The Gaussian case treated by PIBE (Section 3.1) is

    "a zero-mean Gaussian law symmetrically truncated to a fixed compact
    interval",

which is :class:`TruncatedGaussianNoise` below.  Symmetric truncation preserves
the zero mean, so the noise contributes no bias to the quadratic data term ---
exactly the situation Proposition 1 describes, and the situation EPIBE is later
introduced to relax.

No temporal independence is assumed anywhere; samples are drawn i.i.d. here
purely as a convenient admissible realization.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor


class NoiseModel(ABC):
    r"""A measurement-noise law with compact support :math:`[-\bar w, \bar w]`."""

    @abstractmethod
    def sample(
        self,
        shape: tuple[int, ...],
        generator: torch.Generator | None = None,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
    ) -> Tensor:
        """Draw a noise realization of the given shape."""

    @property
    @abstractmethod
    def bound(self) -> float:
        r"""The support bound :math:`\bar w`."""


class NoiseFree(NoiseModel):
    """The degenerate noise-free case, :math:`\\omega \\equiv 0`."""

    def sample(
        self,
        shape: tuple[int, ...],
        generator: torch.Generator | None = None,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
    ) -> Tensor:
        return torch.zeros(shape, dtype=dtype, device=device)

    @property
    def bound(self) -> float:
        return 0.0

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return "NoiseFree()"


class TruncatedGaussianNoise(NoiseModel):
    r"""Zero-mean Gaussian truncated symmetrically to :math:`[-\bar w, \bar w]`.

    Sampling is by inverse CDF, which is exact and vectorized (no rejection
    loop, so the draw is reproducible from the generator state alone).

    Parameters
    ----------
    sigma
        Standard deviation of the *untruncated* Gaussian.
    bound
        Truncation half-width :math:`\bar w`.  Defaults to ``truncation_sigmas
        * sigma``.
    truncation_sigmas
        Used only when ``bound`` is not given.
    """

    def __init__(
        self,
        sigma: float,
        bound: float | None = None,
        truncation_sigmas: float = 3.0,
    ) -> None:
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        if bound is None:
            bound = truncation_sigmas * sigma
        if bound <= 0:
            raise ValueError(f"bound must be positive, got {bound}")
        self.sigma = float(sigma)
        self._bound = float(bound)

    @property
    def bound(self) -> float:
        return self._bound

    def sample(
        self,
        shape: tuple[int, ...],
        generator: torch.Generator | None = None,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
    ) -> Tensor:
        alpha = torch.tensor(self._bound / self.sigma, dtype=dtype, device=device)
        upper = torch.special.ndtr(alpha)
        lower = torch.special.ndtr(-alpha)

        u = torch.rand(shape, generator=generator, dtype=dtype, device=device)
        p = lower + u * (upper - lower)
        # ndtri is infinite at the open endpoints; the draw never attains them
        # in exact arithmetic, but clamp to stay safe in float32.
        eps = torch.finfo(dtype).tiny
        p = p.clamp(min=eps, max=1.0 - torch.finfo(dtype).eps)
        return self.sigma * torch.special.ndtri(p)

    @property
    def variance(self) -> float:
        r"""Variance of the truncated law.

        For a symmetric truncation at :math:`\alpha = \bar w / \sigma`,

        .. math::
            \operatorname{Var} = \sigma^2
            \left[1 - \frac{2\alpha\varphi(\alpha)}{2\Phi(\alpha) - 1}\right].

        Smaller than ``sigma ** 2``; this is the value to compare against when
        assessing estimator accuracy.
        """
        alpha = torch.tensor(self._bound / self.sigma, dtype=torch.float64)
        phi = torch.exp(-0.5 * alpha**2) / torch.sqrt(
            torch.tensor(2.0 * torch.pi, dtype=torch.float64)
        )
        mass = 2.0 * torch.special.ndtr(alpha) - 1.0
        return float(self.sigma**2 * (1.0 - 2.0 * alpha * phi / mass))

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"TruncatedGaussianNoise(sigma={self.sigma}, bound={self._bound})"
