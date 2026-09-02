r"""Nonlinear systems in triangular (observability) canonical form.

This module defines the *only* system-specific surface of the framework.
Everything downstream --- data generation, the estimator bank, the physics
residuals --- is generic in the state dimension ``n``.

The class of systems is Eq. (1) of the paper:

.. math::

    \dot x_j &= x_{j+1} + f_j(x_1,\dots,x_j,\; \theta_1,\dots,\theta_j),
        \quad j = 1,\dots,n-1 \\
    \dot x_n &= f_n(x,\theta) + d(t) \\
    y        &= x_1 + \omega(t)

with unknown constant parameter vector
:math:`\theta = (\theta_1,\dots,\theta_{n-1}) \in \Theta \subset \mathbb{R}^{n-1}`
and an additive time-varying disturbance :math:`d` entering the last equation
only.

Index convention
----------------
Coordinates and parameters are **1-based** in the public API, matching the
paper: :meth:`TriangularSystem.f` is called with ``j = 1, ..., n-1`` and
receives ``x_1..x_j`` and ``theta_1..theta_j``.  Tensor storage is 0-based as
usual, so ``x[..., j - 1]`` holds :math:`x_j`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor


@runtime_checkable
class Disturbance(Protocol):
    """A time-varying disturbance :math:`d(t)` entering the last state equation."""

    def __call__(self, t: Tensor) -> Tensor:
        """Evaluate the disturbance.

        Parameters
        ----------
        t
            Times, shape ``(...,)``.

        Returns
        -------
        Tensor
            Disturbance values, shape ``(...,)``, broadcastable against ``t``.
        """
        ...


class ZeroDisturbance:
    """The identically zero disturbance, used when ``d`` is absent."""

    def __call__(self, t: Tensor) -> Tensor:
        return torch.zeros_like(t)

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return "ZeroDisturbance()"


class TriangularSystem(ABC):
    r"""Base class for systems of the form (1).

    Subclasses implement :meth:`f` (the lower-triangular nonlinearities
    :math:`f_1,\dots,f_{n-1}`) and :meth:`f_last` (:math:`f_n`).  Nothing else
    is required: the bank, the residuals and the data generator are written
    against this interface alone.

    Parameters
    ----------
    n
        State dimension, ``n >= 2``.
    state_bounds
        Admissible interval per coordinate, shape ``(n, 2)`` as ``(lo, hi)``.
        These are the sets :math:`\mathcal{X}_j` of Section 3.1; the state
        decoders are reparameterized to land inside them.
    theta_bounds
        Admissible interval per parameter, shape ``(n - 1, 2)``.  These are the
        sets :math:`\Theta_j`, constraining the parameter heads.
    theta_true
        Ground-truth parameters, shape ``(n - 1,)``.  Used for data generation
        and evaluation only; the estimator never sees them.
    disturbance
        The true :math:`d(t)`.  Defaults to :class:`ZeroDisturbance`.  Note
        that this is a property of the *system*: the estimator's finite basis
        :math:`\Gamma_q` is configured separately, and any part of ``d`` the
        basis cannot represent is the remainder :math:`r_q` of Eq. (2).
    x0_bounds
        Box from which initial conditions are drawn, shape ``(n, 2)``.  This is
        :math:`\chi_0 \subseteq \chi`; defaults to ``state_bounds``.
    name
        Human-readable identifier used in logs and checkpoints.
    """

    def __init__(
        self,
        n: int,
        state_bounds: Tensor,
        theta_bounds: Tensor,
        theta_true: Tensor | None = None,
        disturbance: Disturbance | None = None,
        x0_bounds: Tensor | None = None,
        name: str | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        if n < 2:
            raise ValueError(f"state dimension must be at least 2, got n={n}")
        self.n = int(n)
        self.name = name or type(self).__name__
        self._dtype = dtype

        self.state_bounds = self._as_bounds(state_bounds, self.n, "state_bounds")
        self.theta_bounds = self._as_bounds(
            theta_bounds, self.theta_dim, "theta_bounds"
        )
        self.x0_bounds = (
            self.state_bounds.clone()
            if x0_bounds is None
            else self._as_bounds(x0_bounds, self.n, "x0_bounds")
        )
        if not (
            torch.all(self.x0_bounds[:, 0] >= self.state_bounds[:, 0])
            and torch.all(self.x0_bounds[:, 1] <= self.state_bounds[:, 1])
        ):
            raise ValueError("x0_bounds must be contained in state_bounds (chi_0 subset chi)")

        if theta_true is None:
            self.theta_true: Tensor | None = None
        else:
            theta_true = torch.as_tensor(theta_true, dtype=dtype).reshape(-1)
            if theta_true.numel() != self.theta_dim:
                raise ValueError(
                    f"theta_true must have {self.theta_dim} entries, "
                    f"got {theta_true.numel()}"
                )
            lo, hi = self.theta_bounds[:, 0], self.theta_bounds[:, 1]
            if bool(torch.any(theta_true < lo) or torch.any(theta_true > hi)):
                raise ValueError("theta_true lies outside theta_bounds")
            self.theta_true = theta_true

        self.disturbance: Disturbance = disturbance or ZeroDisturbance()

    # ------------------------------------------------------------------
    # dimensions
    # ------------------------------------------------------------------

    @property
    def theta_dim(self) -> int:
        """Number of unknown parameters, :math:`n - 1`."""
        return self.n - 1

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    # ------------------------------------------------------------------
    # the two abstract nonlinearities
    # ------------------------------------------------------------------

    @abstractmethod
    def f(self, j: int, x: Tensor, theta: Tensor) -> Tensor:
        r"""Evaluate :math:`f_j(x_1,\dots,x_j,\ \theta_1,\dots,\theta_j)`.

        Parameters
        ----------
        j
            Equation index, ``1 <= j <= n - 1`` (1-based, as in the paper).
        x
            The first ``j`` state coordinates, shape ``(..., j)``.
        theta
            The first ``j`` parameters, shape ``(..., j)``.

        Returns
        -------
        Tensor
            Shape ``(...)`` --- the batch shape of ``x`` without its last axis.

        Notes
        -----
        Implementations must be written with plain differentiable torch ops:
        this method is called both to integrate the true dynamics and, inside
        the physics residuals (25)/(31), on network outputs that carry an
        autograd graph.
        """

    @abstractmethod
    def f_last(self, x: Tensor, theta: Tensor) -> Tensor:
        r"""Evaluate :math:`f_n(x, \theta)`, the drift of the last equation.

        Parameters
        ----------
        x
            Full state, shape ``(..., n)``.
        theta
            Full parameter vector, shape ``(..., n - 1)``.

        Returns
        -------
        Tensor
            Shape ``(...)``.
        """

    # ------------------------------------------------------------------
    # derived dynamics
    # ------------------------------------------------------------------

    def vector_field(self, x: Tensor, theta: Tensor, d: Tensor) -> Tensor:
        r"""Assemble :math:`\dot x` from (1).

        Parameters
        ----------
        x
            State, shape ``(..., n)``.
        theta
            Parameters, shape ``(..., n - 1)`` or ``(n - 1,)`` (broadcast).
        d
            Disturbance value(s), shape ``(...)`` or scalar (broadcast).

        Returns
        -------
        Tensor
            :math:`\dot x`, shape ``(..., n)``.
        """
        if x.shape[-1] != self.n:
            raise ValueError(f"expected state of width {self.n}, got {x.shape[-1]}")
        if theta.shape[-1] != self.theta_dim:
            raise ValueError(
                f"expected parameters of width {self.theta_dim}, got {theta.shape[-1]}"
            )
        theta = theta.expand(*x.shape[:-1], self.theta_dim)

        rows = [
            x[..., j] + self.f(j, x[..., :j], theta[..., :j])
            for j in range(1, self.n)
        ]
        rows.append(self.f_last(x, theta) + torch.as_tensor(d, dtype=x.dtype, device=x.device))
        return torch.stack(rows, dim=-1)

    def output(self, x: Tensor) -> Tensor:
        r"""Noise-free output :math:`y_0 = Cx = x_1`, shape ``(...)``."""
        return x[..., 0]

    # ------------------------------------------------------------------
    # sampling and scales
    # ------------------------------------------------------------------

    def sample_x0(self, count: int, generator: torch.Generator | None = None) -> Tensor:
        r"""Draw ``count`` initial conditions uniformly from :math:`\chi_0`.

        Returns a tensor of shape ``(count, n)``.  Override for a different
        initial-condition distribution.
        """
        lo, hi = self.x0_bounds[:, 0], self.x0_bounds[:, 1]
        u = torch.rand(
            count, self.n, generator=generator, dtype=self._dtype,
            device=self.x0_bounds.device,
        )
        return lo + u * (hi - lo)

    def reference_scales(self) -> tuple[Tensor, Tensor]:
        r"""Reference scales :math:`(s_{x_1},\dots,s_{x_n})` and :math:`(s_{\theta_j})`.

        Section 4.1 fixes positive reference scales to non-dimensionalize the
        trajectory input (61) and the cell-output norms (67)-(69), and sets
        :math:`s_y := s_{x_1}`.  We take the half-width of each admissible
        interval, which makes every normalized quantity live in ``[-1, 1]``.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(s_x, s_theta)`` of shapes ``(n,)`` and ``(n - 1,)``.
        """
        s_x = 0.5 * (self.state_bounds[:, 1] - self.state_bounds[:, 0])
        s_theta = 0.5 * (self.theta_bounds[:, 1] - self.theta_bounds[:, 0])
        if bool(torch.any(s_x <= 0)) or bool(torch.any(s_theta <= 0)):
            raise ValueError("admissible intervals must have positive width")
        return s_x, s_theta

    # ------------------------------------------------------------------
    # placement
    # ------------------------------------------------------------------

    def to(self, device: torch.device | str | None = None, dtype: torch.dtype | None = None):
        """Move the system's tensors in place and return ``self``."""
        if dtype is not None:
            self._dtype = dtype
        self.state_bounds = self.state_bounds.to(device=device, dtype=dtype)
        self.theta_bounds = self.theta_bounds.to(device=device, dtype=dtype)
        self.x0_bounds = self.x0_bounds.to(device=device, dtype=dtype)
        if self.theta_true is not None:
            self.theta_true = self.theta_true.to(device=device, dtype=dtype)
        return self

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _as_bounds(self, value: Tensor, rows: int, label: str) -> Tensor:
        bounds = torch.as_tensor(value, dtype=self._dtype)
        if bounds.shape != (rows, 2):
            raise ValueError(f"{label} must have shape ({rows}, 2), got {tuple(bounds.shape)}")
        if bool(torch.any(bounds[:, 0] >= bounds[:, 1])):
            raise ValueError(f"{label} requires lo < hi in every row")
        return bounds.clone()

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"{self.name}(n={self.n})"
