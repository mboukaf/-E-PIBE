r"""Numerical integration of the true system, Eq. (1).

Implements step (2) of the data-generation procedure in Section 5.1: for each
sampled initial condition, integrate

.. math:: \dot x = f(x,\theta) + E_d d(t), \qquad y = Cx + \omega(t)

from :math:`t_0 = 0` to :math:`t_F = T` and record the trajectory on the data
grid.  A fixed-step RK4 rule is used, with ``substeps`` sub-intervals between
consecutive grid points so that grid resolution and integration accuracy stay
independent.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

from pibe.systems.base import Disturbance, TriangularSystem
from pibe.utils.logging import get_logger

logger = get_logger(__name__)

VectorField = Callable[[Tensor, Tensor], Tensor]


def _rk4_step(field: VectorField, t: Tensor, x: Tensor, h: Tensor) -> Tensor:
    """One classical Runge-Kutta step of size ``h`` from ``(t, x)``."""
    k1 = field(t, x)
    k2 = field(t + 0.5 * h, x + 0.5 * h * k1)
    k3 = field(t + 0.5 * h, x + 0.5 * h * k2)
    k4 = field(t + h, x + h * k3)
    return x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


@torch.no_grad()
def rk4_integrate(
    system: TriangularSystem,
    t_grid: Tensor,
    x0: Tensor,
    theta: Tensor,
    disturbance: Disturbance | None = None,
    substeps: int = 8,
) -> Tensor:
    r"""Integrate (1) for a batch of initial conditions.

    Parameters
    ----------
    system
        The system to integrate.
    t_grid
        Strictly increasing output times, shape ``(N,)``.  The first entry is
        the initial time.
    x0
        Initial conditions, shape ``(P, n)``.
    theta
        True parameters, shape ``(n - 1,)`` or ``(P, n - 1)``.
    disturbance
        The true :math:`d(t)`.  Defaults to the system's own.
    substeps
        RK4 sub-steps per output interval.

    Returns
    -------
    Tensor
        Trajectories, shape ``(P, N, n)``.
    """
    if x0.ndim != 2 or x0.shape[1] != system.n:
        raise ValueError(f"x0 must have shape (P, {system.n}), got {tuple(x0.shape)}")
    if t_grid.ndim != 1 or t_grid.numel() < 2:
        raise ValueError("t_grid must be a 1-D tensor with at least two entries")
    if bool(torch.any(t_grid[1:] <= t_grid[:-1])):
        raise ValueError("t_grid must be strictly increasing")
    if substeps < 1:
        raise ValueError(f"substeps must be at least 1, got {substeps}")

    d_fn = disturbance if disturbance is not None else system.disturbance
    theta = theta.reshape(-1, system.theta_dim) if theta.ndim > 1 else theta

    def field(t: Tensor, x: Tensor) -> Tensor:
        return system.vector_field(x, theta, d_fn(t))

    trajectory = [x0]
    x = x0
    for i in range(t_grid.numel() - 1):
        h = (t_grid[i + 1] - t_grid[i]) / substeps
        t = t_grid[i]
        for _ in range(substeps):
            x = _rk4_step(field, t, x, h)
            t = t + h
        trajectory.append(x)

    return torch.stack(trajectory, dim=1)


def uniform_grid(
    t_start: float,
    t_end: float,
    count: int,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> Tensor:
    """A uniform grid on ``[t_start, t_end]`` *including both endpoints*.

    Section 4.2 requires the data and collocation grids to contain the
    endpoints of :math:`[0,T]` and to be quasi-uniform; a uniform grid has
    mesh ratio 1 and fill distance ``(t_end - t_start) / (2 * (count - 1))``.
    """
    if count < 2:
        raise ValueError(f"grid needs at least two points, got {count}")
    return torch.linspace(t_start, t_end, count, dtype=dtype, device=device)


def fill_distance(t_grid: Tensor) -> float:
    r"""The fill distance :math:`h = \sup_t \min_i |t - t_i|` of Eq. (87)."""
    gaps = t_grid[1:] - t_grid[:-1]
    return float(gaps.max() / 2.0)


def check_admissible(
    system: TriangularSystem,
    x: Tensor,
    tolerance: float = 0.0,
    raise_on_violation: bool = True,
) -> Tensor:
    r"""Verify that trajectories remain in the admissible set :math:`\chi`.

    Assumption 1 requires the solution to stay in :math:`\chi` on ``[0, T]``.
    This matters operationally, not just theoretically: the state decoders are
    reparameterized onto ``state_bounds``, so a trajectory that leaves the box
    is *unrepresentable* and the estimator cannot converge to it.

    Returns
    -------
    Tensor
        Per-coordinate margin ``min(x - lo, hi - x)`` minimised over trajectory
        and time, shape ``(n,)``.  Negative entries indicate a violation.
    """
    lo = system.state_bounds[:, 0]
    hi = system.state_bounds[:, 1]
    margin = torch.minimum(x - lo, hi - x).amin(dim=(0, 1))
    violated = margin < -tolerance
    if bool(violated.any()):
        offenders = [
            f"x_{j + 1}: margin {float(margin[j]):+.3e}, "
            f"range [{float(lo[j]):g}, {float(hi[j]):g}], "
            f"observed [{float(x[..., j].min()):g}, {float(x[..., j].max()):g}]"
            for j in range(system.n)
            if bool(violated[j])
        ]
        message = (
            "simulated trajectories leave the admissible set chi "
            "(Assumption 1); the box-constrained decoders cannot represent "
            "them. Widen state_bounds, shrink x0_bounds, or shorten T.\n  "
            + "\n  ".join(offenders)
        )
        if raise_on_violation:
            raise ValueError(message)
        logger.warning(message)
    return margin
