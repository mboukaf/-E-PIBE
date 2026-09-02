r"""Typed configuration objects, loadable from YAML.

Groups the inputs listed in Algorithm 1 --- sampled trajectories, the per-cell
iteration budget :math:`N_{tot}`, the local pre-training cutoff
:math:`N_{par}` with :math:`0 < N_{par} < N_{tot}`, and the weighting factor
:math:`\lambda` --- together with the architectural and basis choices.

This module deliberately imports nothing from the rest of the package, so it
can be imported from anywhere without a cycle.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ArchitectureConfig:
    r"""Widths of the three networks making up a cell, Eq. (20).

    Attributes
    ----------
    latent_dim
        :math:`r_k`, the width of the trajectory code.
    encoder_hidden, decoder_hidden, head_hidden
        Hidden layer widths of :math:`\mathcal{E}_k`, :math:`\mathcal{S}_k` and
        :math:`\mathcal{Q}_k`.
    activation
        Must be twice continuously differentiable, per Eq. (21).
    """

    latent_dim: int = 32
    encoder_hidden: tuple[int, ...] = (128, 128)
    decoder_hidden: tuple[int, ...] = (64, 64, 64)
    head_hidden: tuple[int, ...] = (64, 64)
    activation: str = "tanh"

    def __post_init__(self) -> None:
        if self.latent_dim < 1:
            raise ValueError(f"latent_dim must be positive, got {self.latent_dim}")
        self.encoder_hidden = tuple(self.encoder_hidden)
        self.decoder_hidden = tuple(self.decoder_hidden)
        self.head_hidden = tuple(self.head_hidden)


@dataclass
class BasisConfig:
    r"""The disturbance basis :math:`\Gamma_q` and its coefficient box :math:`\mathcal{A}`.

    ``q`` and the knot vector are "fixed during estimation" (Section 5.1).
    Decreasing the knot spacing or increasing ``q`` may reduce the
    approximation error :math:`\varepsilon_{d,q}`, but also enlarges the class
    of admissible disturbances and may worsen distinguishability between
    parameter and disturbance effects.
    """

    q: int = 6
    degree: int = 3
    knot_style: str = "clamped"
    coeff_lo: float = -5.0
    coeff_hi: float = 5.0

    def __post_init__(self) -> None:
        if self.coeff_hi <= self.coeff_lo:
            raise ValueError(
                f"require coeff_lo < coeff_hi, got ({self.coeff_lo}, {self.coeff_hi})"
            )


@dataclass
class DataConfig:
    r"""Trajectory generation, Section 5.1.

    Attributes
    ----------
    n_trajectories
        ``P``.
    n_samples
        ``N``, the size of the data grid on ``[0, T]`` (endpoints included).
    horizon
        ``T``.
    substeps
        RK4 sub-steps per data interval.
    noise_sigma
        Standard deviation of the untruncated Gaussian; ``0`` means noise-free.
    noise_bound
        Truncation half-width :math:`\bar w`; defaults to
        ``noise_truncation_sigmas * noise_sigma``.
    train_fraction
        Fraction of trajectories in :math:`\Omega^{train}`.
    """

    n_trajectories: int = 64
    n_samples: int = 128
    horizon: float = 5.0
    substeps: int = 8
    noise_sigma: float = 0.0
    noise_bound: float | None = None
    noise_truncation_sigmas: float = 3.0
    train_fraction: float = 0.8
    seed: int = 0

    def __post_init__(self) -> None:
        if self.n_trajectories < 2:
            raise ValueError("need at least two trajectories to form a split")
        if self.n_samples < 2:
            raise ValueError(f"n_samples must be at least 2, got {self.n_samples}")
        if self.horizon <= 0:
            raise ValueError(f"horizon must be positive, got {self.horizon}")
        if self.noise_sigma < 0:
            raise ValueError(f"noise_sigma must be non-negative, got {self.noise_sigma}")


@dataclass
class TrainingConfig:
    r"""Algorithm 1's schedule and optimizer settings.

    Attributes
    ----------
    n_total
        :math:`N_{tot}`, the per-cell iteration budget.
    n_par
        :math:`N_{par}`, the local pre-training cutoff; Algorithm 1 requires
        :math:`0 < N_{par} < N_{tot}`.  Iterations ``i < n_par`` update only
        the current cell with :math:`\mathcal{L}^k_{Loc}`; the rest fine-tune
        cells :math:`2..k` end-to-end with :math:`\mathcal{L}^k_{Tot}`.
    lam
        The physics weight :math:`\lambda > 0` of Eq. (6).
    n_collocation
        :math:`N_r`, the size of the collocation grid.
    lr_local, lr_global
        Learning rates for the two phases; fine-tuning perturbs already-trained
        upstream cells and is usually given the smaller rate.
    """

    n_total: int = 4000
    n_par: int = 3000
    lam: float = 1.0
    n_collocation: int = 256
    batch_size: int = 16
    lr_local: float = 1e-3
    lr_global: float = 2e-4
    weight_decay: float = 0.0
    grad_clip: float | None = 1.0
    log_every: int = 200
    seed: int = 0

    def __post_init__(self) -> None:
        if not 0 < self.n_par < self.n_total:
            raise ValueError(
                f"Algorithm 1 requires 0 < n_par < n_total, "
                f"got n_par={self.n_par}, n_total={self.n_total}"
            )
        if self.lam <= 0:
            raise ValueError(f"lambda must be positive, got {self.lam}")
        if self.n_collocation < 2:
            raise ValueError(
                f"n_collocation must be at least 2, got {self.n_collocation}"
            )


@dataclass
class SystemConfig:
    """Names the registered system and its constructor arguments."""

    name: str = "polynomial_chain"
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunConfig:
    """Top-level configuration for one training run."""

    system: SystemConfig = field(default_factory=SystemConfig)
    data: DataConfig = field(default_factory=DataConfig)
    basis: BasisConfig = field(default_factory=BasisConfig)
    architecture: ArchitectureConfig = field(default_factory=ArchitectureConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    device: str = "auto"
    dtype: str = "float64"
    seed: int = 0
    output_dir: str = "outputs/run"

    # ------------------------------------------------------------------
    # serialization
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunConfig:
        """Build from a nested plain dictionary, validating unknown keys."""
        return _build(cls, data)

    @classmethod
    def from_yaml(cls, path: Path | str) -> RunConfig:
        """Load from a YAML file."""
        with open(path) as handle:
            payload = yaml.safe_load(handle) or {}
        if not isinstance(payload, dict):
            raise ValueError(f"{path} must contain a YAML mapping at the top level")
        return cls.from_dict(payload)

    def to_dict(self) -> dict[str, Any]:
        """Recursively convert to plain dictionaries and lists."""
        return dataclasses.asdict(self)

    def to_yaml(self, path: Path | str) -> None:
        """Write to a YAML file, creating parent directories."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as handle:
            yaml.safe_dump(self.to_dict(), handle, sort_keys=False)


def _build(cls: type, data: dict[str, Any]) -> Any:
    """Instantiate a (possibly nested) dataclass from a dictionary."""
    if not dataclasses.is_dataclass(cls):
        return data
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(data) - set(fields)
    if unknown:
        raise ValueError(
            f"unknown key(s) {sorted(unknown)} for {cls.__name__}; "
            f"valid keys are {sorted(fields)}"
        )
    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        field_type = fields[name].type
        # SystemConfig.kwargs is a free-form mapping, not a nested dataclass.
        if isinstance(value, dict) and name != "kwargs":
            nested = _resolve(field_type)
            kwargs[name] = _build(nested, value) if nested is not None else value
        else:
            kwargs[name] = value
    return cls(**kwargs)


_NESTED = {
    "SystemConfig": SystemConfig,
    "DataConfig": DataConfig,
    "BasisConfig": BasisConfig,
    "ArchitectureConfig": ArchitectureConfig,
    "TrainingConfig": TrainingConfig,
}


def _resolve(annotation: Any) -> type | None:
    """Map a dataclass field annotation to the nested dataclass it names."""
    if isinstance(annotation, type):
        return annotation if dataclasses.is_dataclass(annotation) else None
    return _NESTED.get(str(annotation))
