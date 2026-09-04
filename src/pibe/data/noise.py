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


# ----------------------------------------------------------------------
# Non-Gaussian laws
# ----------------------------------------------------------------------
#
# PIBE's Section 3.1 analysis assumes the Gaussian case above, and Proposition 1
# leans on one property of it in particular: the noise is zero-mean, so it adds
# no bias to the quadratic data term.  Nothing there needs the noise to be
# *Gaussian* --- which makes the shape of the law an empirical question rather
# than a theoretical one, and the laws below are what answer it.
#
# They span the three ways a real sensor departs from the Gaussian assumption:
# a different tail weight at the same variance (uniform, Laplace), rare large
# excursions (contaminated), and asymmetry (skewed).  Only the last of these,
# combined with :class:`BiasedNoise`, actually breaks Proposition 1's premise;
# the rest test whether the estimator cares about shape at all.
#
# Every law is sampled by inverse transform from a single uniform draw, so it is
# reproducible from the generator state alone, and every one is compactly
# supported, keeping Section 2's assumption intact.


class QuantileNoise(NoiseModel):
    r"""Base class for laws defined by a quantile function on ``(0, 1)``.

    A subclass supplies :meth:`quantile`, the inverse CDF of some *unscaled*
    shape.  This class then centres and rescales it so the realized law has zero
    mean and standard deviation exactly ``sigma``, which is what makes different
    shapes comparable at a matched noise level: any difference in the results is
    then attributable to the shape rather than to the noise power.

    The centring shift and the scale are computed once at construction by
    midpoint quadrature of the quantile function over ``(0, 1)``.  Because
    sampling *is* evaluation of that same function at uniform draws, this is
    quadrature of the law's own moments rather than a Monte-Carlo estimate of
    them, and is accurate to the quadrature order.

    Parameters
    ----------
    sigma
        Target standard deviation of the realized law.
    quadrature_points
        Number of midpoints used to calibrate the mean and variance.
    """

    def __init__(self, sigma: float, quadrature_points: int = 200_001) -> None:
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        self.sigma = float(sigma)

        p = (torch.arange(quadrature_points, dtype=torch.float64) + 0.5) / quadrature_points
        values = self.quantile(p)
        mean = values.mean()
        std = torch.sqrt(torch.mean((values - mean) ** 2))
        if not bool(torch.isfinite(std)) or float(std) <= 0:
            raise ValueError(f"{type(self).__name__} has degenerate spread")
        self._shift = float(mean)
        self._scale = self.sigma / float(std)
        # The support bound is attained at the ends of ``(0, 1)``, which the
        # midpoint grid above never reaches; evaluate there directly so the
        # stated bound really does contain every draw.
        eps = torch.finfo(torch.float64).eps
        ends = self.quantile(torch.tensor([eps, 1.0 - eps], dtype=torch.float64))
        self._bound = float(((ends - mean) * self._scale).abs().max())

    @abstractmethod
    def quantile(self, p: Tensor) -> Tensor:
        """Inverse CDF of the unscaled shape, for ``p`` in ``(0, 1)``."""

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
        # Drawn in float64 and cast down: the quantile functions below involve
        # logs of numbers near zero, where float32 loses the tail.
        u = torch.rand(shape, generator=generator, dtype=torch.float64, device=device)
        eps = torch.finfo(torch.float64).eps
        u = u.clamp(min=eps, max=1.0 - eps)
        value = (self.quantile(u) - self._shift) * self._scale
        return value.to(dtype)

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"{type(self).__name__}(sigma={self.sigma:g}, bound={self._bound:.4g})"


class UniformNoise(QuantileNoise):
    r"""Zero-mean uniform noise on :math:`[-\sqrt{3}\sigma, \sqrt{3}\sigma]`.

    The lightest-tailed compactly supported law at a given variance, and the
    natural opposite end from the contaminated law below: it has no tail at all,
    every realization being of comparable size.
    """

    def quantile(self, p: Tensor) -> Tensor:
        return 2.0 * p - 1.0


class TruncatedLaplaceNoise(QuantileNoise):
    r"""Laplace noise truncated symmetrically, then rescaled to ``sigma``.

    Heavier-tailed than the Gaussian at equal variance and sharply peaked at
    zero, so most samples are smaller than the Gaussian's and a few are much
    larger.  Still symmetric, so it leaves Proposition 1's premise intact.

    Parameters
    ----------
    truncation_sigmas
        Truncation point, in units of the *untruncated* Laplace standard
        deviation.  Laplace tails decay more slowly than Gaussian ones, so the
        default is wider than the Gaussian's 3.
    """

    def __init__(self, sigma: float, truncation_sigmas: float = 4.0, **kwargs) -> None:
        if truncation_sigmas <= 0:
            raise ValueError("truncation_sigmas must be positive")
        # A standard Laplace (scale 1) has standard deviation sqrt(2).
        self._limit = float(truncation_sigmas) * 2.0**0.5
        self.truncation_sigmas = float(truncation_sigmas)
        super().__init__(sigma, **kwargs)

    def quantile(self, p: Tensor) -> Tensor:
        # CDF of the standard Laplace at +/- limit, to renormalize onto the
        # truncated support.
        mass = 1.0 - torch.exp(-torch.tensor(self._limit, dtype=p.dtype))
        u = 0.5 + (p - 0.5) * mass  # maps (0,1) onto the untruncated CDF range
        return torch.where(
            u < 0.5,
            torch.log(2.0 * u),
            -torch.log(2.0 * (1.0 - u)),
        )


class ContaminatedGaussianNoise(QuantileNoise):
    r"""A Gaussian core with a fraction of much larger excursions.

    The classical epsilon-contamination model of robust statistics, and the
    closest of these laws to a real sensor fault: most samples are ordinary,
    a few are outliers an order of magnitude larger.  Symmetric, hence still
    zero-mean, but very far from Gaussian in shape --- and the case where a
    *quadratic* data term is most exposed, since it weights a rare large
    residual by its square.

    Parameters
    ----------
    contamination
        Fraction of samples drawn from the wide component.
    outlier_scale
        Width of that component relative to the core.
    truncation_sigmas
        Truncation of each component, in units of its own width.
    """

    def __init__(
        self,
        sigma: float,
        contamination: float = 0.05,
        outlier_scale: float = 8.0,
        truncation_sigmas: float = 3.0,
        **kwargs,
    ) -> None:
        if not 0.0 < contamination < 1.0:
            raise ValueError(f"contamination must lie in (0, 1), got {contamination}")
        if outlier_scale <= 1.0:
            raise ValueError("outlier_scale must exceed 1 to be a contamination")
        self.contamination = float(contamination)
        self.outlier_scale = float(outlier_scale)
        self._alpha = float(truncation_sigmas)
        super().__init__(sigma, **kwargs)

    def quantile(self, p: Tensor) -> Tensor:
        # Splitting the uniform range in proportion to the mixture weights
        # reproduces the mixture exactly, while keeping a single uniform draw
        # and a quantile function the base class can integrate.
        eps = self.contamination
        is_outlier = p < eps
        rescaled = torch.where(is_outlier, p / eps, (p - eps) / (1.0 - eps))
        width = torch.where(
            is_outlier,
            torch.full_like(p, self.outlier_scale),
            torch.ones_like(p),
        )
        return width * self._truncated_normal_quantile(rescaled)

    def _truncated_normal_quantile(self, p: Tensor) -> Tensor:
        """Inverse CDF of a unit normal truncated at ``+/- alpha``."""
        alpha = torch.tensor(self._alpha, dtype=p.dtype)
        lower, upper = torch.special.ndtr(-alpha), torch.special.ndtr(alpha)
        u = (lower + p * (upper - lower)).clamp(
            min=torch.finfo(p.dtype).tiny, max=1.0 - torch.finfo(p.dtype).eps
        )
        return torch.special.ndtri(u)


class SkewedNoise(QuantileNoise):
    r"""A one-sided exponential law, recentred to zero mean.

    Asymmetric: the median sits on the opposite side of zero from the tail, so
    the *typical* error and the *mean* error have opposite signs.  Zero-mean by
    construction, so Proposition 1's premise survives --- this isolates skewness
    from bias, which :class:`BiasedNoise` supplies separately.

    Parameters
    ----------
    truncation_sigmas
        Truncation of the exponential tail, in units of its standard deviation.
    """

    def __init__(self, sigma: float, truncation_sigmas: float = 4.0, **kwargs) -> None:
        self._limit = float(truncation_sigmas)  # unit exponential has std 1
        self.truncation_sigmas = float(truncation_sigmas)
        super().__init__(sigma, **kwargs)

    def quantile(self, p: Tensor) -> Tensor:
        mass = 1.0 - torch.exp(-torch.tensor(self._limit, dtype=p.dtype))
        return -torch.log(1.0 - p * mass)


class BiasedNoise(NoiseModel):
    r"""Any noise law shifted off zero mean --- the case PIBE does not cover.

    Proposition 1's argument is that a symmetric zero-mean noise contributes no
    systematic term to the quadratic data loss, so the estimator is unbiased in
    the limit.  A constant sensor offset breaks precisely that: the data term is
    then minimized by an :math:`\hat x_1` displaced by the offset, and Eq. (1)'s
    chain propagates that displacement into the states, the parameters and the
    disturbance.  This is the situation EPIBE (Section 3.2) exists to handle, so
    the failure here is expected and is the point of measuring it.

    Parameters
    ----------
    base
        The underlying zero-mean law.
    bias
        Constant offset added to every sample, in units of the base law's
        ``sigma`` when ``relative`` is set, else absolute.
    """

    def __init__(self, base: NoiseModel, bias: float, relative: bool = True) -> None:
        self.base = base
        scale = getattr(base, "sigma", 1.0) if relative else 1.0
        self.bias = float(bias) * float(scale)

    @property
    def sigma(self) -> float:
        """The base law's spread; the bias is a mean, not a spread."""
        return float(getattr(self.base, "sigma", float("nan")))

    @property
    def bound(self) -> float:
        return self.base.bound + abs(self.bias)

    def sample(
        self,
        shape: tuple[int, ...],
        generator: torch.Generator | None = None,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
    ) -> Tensor:
        return self.base.sample(shape, generator, dtype, device) + self.bias

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"BiasedNoise({self.base!r}, bias={self.bias:+.4g})"


#: Constructors for the laws above, keyed by the name used on the command line.
NOISE_FAMILIES = {
    "gaussian": TruncatedGaussianNoise,
    "uniform": UniformNoise,
    "laplace": TruncatedLaplaceNoise,
    "contaminated": ContaminatedGaussianNoise,
    "skewed": SkewedNoise,
}


def build_noise_model(
    kind: str,
    sigma: float,
    bias: float = 0.0,
    bias_relative: bool = True,
    **kwargs,
) -> NoiseModel:
    """Construct a noise law by name; ``sigma <= 0`` gives the noise-free case.

    Parameters
    ----------
    kind
        One of :data:`NOISE_FAMILIES`.
    sigma
        Standard deviation.  Every family is calibrated to it, so laws of
        different shape are compared at equal noise power.
    bias
        If nonzero, the law is wrapped in :class:`BiasedNoise` with this offset.
    bias_relative
        Whether ``bias`` is in units of ``sigma`` (the default) or absolute.
        An absolute offset is what lets a sensor's *mean* error be compared
        against its *spread* at equal magnitude.
    """
    if sigma <= 0 and not bias:
        return NoiseFree()
    if kind not in NOISE_FAMILIES:
        raise ValueError(
            f"unknown noise family {kind!r}; expected one of {sorted(NOISE_FAMILIES)}"
        )
    if sigma <= 0:
        # A pure offset with no spread: degenerate as a law, but the cleanest
        # possible isolation of the mean, so it is allowed rather than refused.
        return BiasedNoise(NoiseFree(), bias, relative=False)
    noise = NOISE_FAMILIES[kind](sigma, **kwargs)
    return BiasedNoise(noise, bias, relative=bias_relative) if bias else noise
