r"""Trajectory datasets.

Follows the three steps of Section 5.1:

1. sample initial conditions :math:`x_0 \in \chi_0`;
2. simulate (1) on :math:`[0, T]` to obtain :math:`x^i(t)` and :math:`y^i(t)`,
   Eq. (135);
3. split trajectories into disjoint training and validation sets,
   :math:`\Omega^{train} \cap \Omega^{test} = \emptyset`.

The estimator is *trajectory-level*: one sample is an entire sampled
trajectory, and a minibatch is a set of trajectory indices (Algorithm 1,
line 4).
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterator

import torch
from torch import Tensor

from pibe.data.noise import NoiseFree, NoiseModel
from pibe.data.simulate import check_admissible, rk4_integrate
from pibe.systems.base import Disturbance, TriangularSystem


@dataclass
class TrajectoryData:
    r"""A set of ``P`` sampled trajectories on a shared time grid.

    Attributes
    ----------
    t
        Data grid :math:`\{t_i\}_{i=1}^N`, shape ``(N,)``.
    x
        True states, shape ``(P, N, n)``.  Ground truth: used for evaluation
        only, never by the estimator.
    y
        Noisy measurements :math:`y = x_1 + \omega`, shape ``(P, N)``.  This is
        the vector :math:`\mathbf{Y}^\ell` of Eq. (13) and the sole input to
        the first cell.
    d
        True disturbance on the data grid, shape ``(P, N)``.  Per trajectory,
        because Eq. (2)'s ``a`` is an unknown the estimator must *infer* from
        ``y``; a single shared disturbance would let the coefficient head learn
        a constant instead.
    theta
        True parameters, shape ``(P, n - 1)``.  Per trajectory for the same
        reason when parameters are sampled; broadcast when they are shared.
    coefficients
        True disturbance coefficients ``a``, shape ``(P, q)``.  Carried here so
        the dataset is self-contained: after a train/val split the disturbance
        *object* still holds every trajectory's coefficients, so anything
        needing the truth for a subset (the oracle, the metrics) must read it
        from the subset itself.
    """

    t: Tensor
    x: Tensor
    y: Tensor
    d: Tensor
    theta: Tensor
    coefficients: Tensor | None = None

    def __post_init__(self) -> None:
        p, n_samples, _ = self.x.shape
        if self.t.shape != (n_samples,):
            raise ValueError(f"t must have shape ({n_samples},), got {tuple(self.t.shape)}")
        if self.y.shape != (p, n_samples):
            raise ValueError(f"y must have shape {(p, n_samples)}, got {tuple(self.y.shape)}")
        if self.d.shape != (p, n_samples):
            raise ValueError(f"d must have shape {(p, n_samples)}, got {tuple(self.d.shape)}")
        if self.theta.shape != (p, self.x.shape[2] - 1):
            raise ValueError(
                f"theta must have shape {(p, self.x.shape[2] - 1)}, "
                f"got {tuple(self.theta.shape)}"
            )
        if self.coefficients is not None and self.coefficients.shape[0] != p:
            raise ValueError(
                f"coefficients must have {p} rows, got {self.coefficients.shape[0]}"
            )

    @property
    def n_trajectories(self) -> int:
        """``P``."""
        return self.x.shape[0]

    @property
    def n_samples(self) -> int:
        """``N``, the number of time samples per trajectory."""
        return self.x.shape[1]

    @property
    def state_dim(self) -> int:
        """``n``."""
        return self.x.shape[2]

    @property
    def horizon(self) -> tuple[float, float]:
        """``(0, T)``."""
        return float(self.t[0]), float(self.t[-1])

    def __len__(self) -> int:
        return self.n_trajectories

    def select(self, index: Tensor) -> TrajectoryData:
        """A view onto a subset of trajectories; ``d`` and ``theta`` follow it."""
        return TrajectoryData(
            t=self.t, x=self.x[index], y=self.y[index],
            d=self.d[index], theta=self.theta[index],
            coefficients=None if self.coefficients is None else self.coefficients[index],
        )

    def split(
        self, train_fraction: float = 0.8, generator: torch.Generator | None = None
    ) -> tuple[TrajectoryData, TrajectoryData]:
        """Disjoint train/validation split over trajectory indices."""
        if not 0.0 < train_fraction < 1.0:
            raise ValueError(f"train_fraction must lie in (0, 1), got {train_fraction}")
        count = self.n_trajectories
        n_train = int(round(train_fraction * count))
        n_train = min(max(n_train, 1), count - 1)
        perm = torch.randperm(count, generator=generator).to(self.x.device)
        return self.select(perm[:n_train]), self.select(perm[n_train:])

    def to(
        self, device: torch.device | str | None = None, dtype: torch.dtype | None = None
    ) -> TrajectoryData:
        return TrajectoryData(
            t=self.t.to(device=device, dtype=dtype),
            x=self.x.to(device=device, dtype=dtype),
            y=self.y.to(device=device, dtype=dtype),
            d=self.d.to(device=device, dtype=dtype),
            theta=self.theta.to(device=device, dtype=dtype),
            coefficients=None if self.coefficients is None
            else self.coefficients.to(device=device, dtype=dtype),
        )

    def __repr__(self) -> str:  # pragma: no cover - trivial
        t0, t1 = self.horizon
        return (
            f"TrajectoryData(P={self.n_trajectories}, N={self.n_samples}, "
            f"n={self.state_dim}, horizon=[{t0:g}, {t1:g}])"
        )


def generate_dataset(
    system: TriangularSystem,
    t_grid: Tensor,
    n_trajectories: int,
    noise: NoiseModel | None = None,
    disturbance: Disturbance | None = None,
    generator: torch.Generator | None = None,
    substeps: int = 8,
    check_bounds: bool = True,
    theta: Tensor | None = None,
) -> TrajectoryData:
    """Generate trajectories and noisy measurements, Eq. (135).

    Parameters
    ----------
    system
        The system to simulate; supplies ``theta_true`` and its disturbance.
    t_grid
        Data grid, shape ``(N,)``.
    n_trajectories
        ``P``, the number of initial conditions drawn from :math:`\\chi_0`.
    noise
        Measurement-noise law; defaults to noise-free.
    disturbance
        Overrides the system's own :math:`d(t)`.
    check_bounds
        Verify Assumption 1 (trajectories remain in :math:`\\chi`) and raise
        otherwise.
    theta
        Per-trajectory parameters, shape ``(P, n - 1)``.  Defaults to the
        system's ``theta_true`` broadcast over trajectories.
    """
    if system.theta_true is None:
        raise ValueError(
            "system.theta_true is required to generate data; "
            "it is ground truth and is never exposed to the estimator"
        )
    noise = noise or NoiseFree()
    d_fn = disturbance if disturbance is not None else system.disturbance

    if theta is None:
        theta = system.theta_true.reshape(1, -1).expand(n_trajectories, -1).clone()
    theta = torch.as_tensor(theta, dtype=system.state_bounds.dtype)
    if theta.shape != (n_trajectories, system.theta_dim):
        raise ValueError(
            f"theta must have shape {(n_trajectories, system.theta_dim)}, "
            f"got {tuple(theta.shape)}"
        )

    x0 = system.sample_x0(n_trajectories, generator=generator)
    x = rk4_integrate(
        system,
        t_grid=t_grid,
        x0=x0,
        theta=theta,
        disturbance=d_fn,
        substeps=substeps,
    )
    if check_bounds:
        check_admissible(system, x)

    omega = noise.sample(
        (n_trajectories, t_grid.numel()),
        generator=generator,
        dtype=x.dtype,
        device=x.device,
    )
    y = system.output(x) + omega

    d_true = d_fn(t_grid)
    if d_true.ndim == 1:                      # shared disturbance
        d_true = d_true.reshape(1, -1).expand(n_trajectories, -1).clone()
    coefficients = getattr(d_fn, "coefficients", None)
    if coefficients is not None and coefficients.shape[0] == 1:
        coefficients = coefficients.expand(n_trajectories, -1).clone()
    return TrajectoryData(
        t=t_grid, x=x, y=y, d=d_true, theta=theta, coefficients=coefficients
    )


class TrajectoryBatcher:
    """Yields random minibatches of trajectory indices (Algorithm 1, line 4).

    ``batch_size <= 0`` or a batch at least as large as the dataset means full
    batch, in which case the same index tensor is returned every time.
    """

    def __init__(
        self,
        n_trajectories: int,
        batch_size: int,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        if n_trajectories < 1:
            raise ValueError("need at least one trajectory")
        self.n_trajectories = n_trajectories
        self.generator = generator
        self.device = device
        self.batch_size = (
            n_trajectories
            if batch_size <= 0 or batch_size >= n_trajectories
            else int(batch_size)
        )
        self._full = self.batch_size == n_trajectories
        self._all = torch.arange(n_trajectories, device=device)

    @property
    def is_full_batch(self) -> bool:
        return self._full

    def sample(self) -> Tensor:
        """One minibatch of trajectory indices."""
        if self._full:
            return self._all
        # Drawn on CPU so a CPU generator stays valid regardless of where the
        # data lives, then moved to the data's device for indexing.
        perm = torch.randperm(self.n_trajectories, generator=self.generator)
        return perm[: self.batch_size].to(self._all.device)

    def __iter__(self) -> Iterator[Tensor]:  # pragma: no cover - convenience
        while True:
            yield self.sample()
