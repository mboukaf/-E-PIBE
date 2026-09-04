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
    time_fourier_features
        Number of harmonics lifting the decoder's time input; ``0`` disables
        it.  See :class:`~pibe.nets.decoder.FourierTimeFeatures` --- it exists
        because disturbance recovery is limited by how well the *first*
        coordinate is resolved, not by network width.
    """

    latent_dim: int = 32
    encoder_hidden: tuple[int, ...] = (128, 128)
    decoder_hidden: tuple[int, ...] = (64, 64, 64)
    head_hidden: tuple[int, ...] = (64, 64)
    activation: str = "tanh"
    time_fourier_features: int = 0
    time_feature_scaling: bool = True

    def __post_init__(self) -> None:
        if self.time_fourier_features < 0:
            raise ValueError(
                "time_fourier_features must be non-negative, got "
                f"{self.time_fourier_features}"
            )
        if self.latent_dim < 1:
            raise ValueError(f"latent_dim must be positive, got {self.latent_dim}")
        self.encoder_hidden = tuple(self.encoder_hidden)
        self.decoder_hidden = tuple(self.decoder_hidden)
        self.head_hidden = tuple(self.head_hidden)


@dataclass
class BasisConfig:
    r"""The disturbance basis :math:`\Gamma_q` and its coefficient box :math:`\mathcal{A}`.

    ``q`` and the basis are "fixed during estimation" (Section 5.1).  Increasing
    ``q`` may reduce the approximation error :math:`\varepsilon_{d,q}`, but also
    enlarges the class of admissible disturbances and may worsen
    distinguishability between parameter and disturbance effects.

    Attributes
    ----------
    kind
        ``"bspline"`` (Definition 1) or ``"fourier"`` (trigonometric).
    degree, knot_style
        B-spline options; ignored --- and rejected if set --- for ``"fourier"``.
    omega
        Fourier fundamental frequency :math:`\Omega`; likewise B-spline-invalid.
    coeff_lo, coeff_hi
        The box :math:`\mathcal{A}`.  Either a scalar, applied to every
        coefficient, or a list of ``q`` values for a per-component box.
    remainder_amplitude, remainder_rate
        The out-of-class disturbance component
        :math:`r_q(t) = \varepsilon_{d,q}\sin(\text{rate}\cdot t^2)` of Eq. (2).
        An amplitude of ``0`` gives the exactly recoverable case
        :math:`\varepsilon_{d,q} = 0`; a nonzero value makes exact recovery
        impossible and bounds degrade by :math:`\varepsilon_{d,q}` per (4).
    """

    kind: str = "bspline"
    q: int = 6
    degree: int | None = None
    knot_style: str | None = None
    omega: float | None = None
    coeff_lo: float | list[float] = -5.0
    coeff_hi: float | list[float] = 5.0
    remainder_amplitude: float = 0.0
    remainder_rate: float = 0.15

    def __post_init__(self) -> None:
        if self.q < 1:
            raise ValueError(f"q must be positive, got {self.q}")
        if self.remainder_amplitude < 0:
            raise ValueError(
                f"remainder_amplitude must be non-negative, got {self.remainder_amplitude}"
            )
        # Validates the interval list eagerly so a malformed box fails at
        # config load rather than midway through data generation.
        self.coefficient_intervals()

    def basis_kwargs(self) -> dict[str, Any]:
        """Basis-specific options, omitting those left unset."""
        options = {
            "degree": self.degree,
            "knot_style": self.knot_style,
            "omega": self.omega,
        }
        return {name: value for name, value in options.items() if value is not None}

    def coefficient_intervals(self) -> list[tuple[float, float]]:
        r"""The box :math:`\mathcal{A}` as ``q`` ``(lo, hi)`` pairs."""

        def spread(value: float | list[float], label: str) -> list[float]:
            if isinstance(value, (int, float)):
                return [float(value)] * self.q
            values = [float(entry) for entry in value]
            if len(values) != self.q:
                raise ValueError(
                    f"{label} must be a scalar or a list of q={self.q} values, "
                    f"got {len(values)}"
                )
            return values

        lows = spread(self.coeff_lo, "coeff_lo")
        highs = spread(self.coeff_hi, "coeff_hi")
        for index, (low, high) in enumerate(zip(lows, highs)):
            if high <= low:
                raise ValueError(
                    f"coefficient {index}: require lo < hi, got ({low}, {high})"
                )
        return list(zip(lows, highs))


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
    sample_disturbance_per_trajectory
        Draw an independent ``a`` for every trajectory.  This is the general
        case and the default: Eq. (2)'s ``a`` is an *unknown* the estimator must
        infer from :math:`y`, and the paper notes that :math:`\hat a^\ell`
        "may differ between trajectories".  With a single shared ``a`` the
        coefficient head can satisfy the objective by learning a constant, so
        the disturbance-estimation problem is never actually posed.
    sample_theta_per_trajectory
        Likewise for :math:`\theta`.  Off by default, since :math:`\theta` is
        a constant of the *system* rather than of a trajectory; turning it on
        asks the bank to infer parameters for an unseen system, which is a
        strictly harder problem than Eq. (1) poses.
    """

    n_trajectories: int = 64
    n_samples: int = 128
    horizon: float = 5.0
    substeps: int = 8
    noise_sigma: float = 0.0
    noise_bound: float | None = None
    noise_truncation_sigmas: float = 3.0
    train_fraction: float = 0.8
    sample_disturbance_per_trajectory: bool = True
    sample_theta_per_trajectory: bool = False
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
    local_only
        Ablation: skip end-to-end fine-tuning entirely and train every cell
        with its own :math:`\mathcal{L}^k_{Loc}` against frozen upstream cells.
        This *departs from Algorithm 1*, which always runs a global phase.

        It is worth having because the global loss (27) weights the current
        cell by 1 and cell 2 by :math:`e^{-(k-2)/k}`, while cell 2's data term
        (23) is the only one comparing against a *measurement*: every other
        data term is a consistency penalty between two estimates, satisfiable
        by the whole chain drifting together.  Down-weighting the sole anchor
        can therefore trade a real measurement fit for mutual agreement.
        Freezing upstream removes that freedom --- in particular it pins the
        final cell's auxiliary state to its consistency target, leaving
        :math:`\hat a` as the only free variable in the last residual.

        When ``True``, ``n_par`` is ignored.
    final_global_iters
        Extra end-to-end iterations appended to the *final* cell's global
        phase.  This cell is the only place the bank closes: at every other
        cell the residual
        :math:`\dot{\hat x}_{k-1} = \hat x_k + f_{k-1}(\cdot,\hat\theta_{k-1})`
        has two unknowns and one equation, a one-parameter family the local
        loss cannot resolve (Remark 7).  At cell :math:`n+1` the auxiliary
        state is pinned by its consistency target and :math:`\hat a` must fit
        the last residual over the whole horizon --- over-determined, hence
        identifying.  Since that information reaches cells :math:`2..n` only
        through :math:`\mathcal{L}^{n+1}_{Tot}`, this phase deserves most of
        the budget.
    joint
        Ablation: skip the outer loop over cells and train the whole bank at
        once against :math:`\mathcal{L}^{n+1}_{Tot}`.

        Motivated by where the bank actually closes.  Under Algorithm 1 cells
        :math:`2..n` are trained to convergence *before* cell :math:`n+1`
        exists, and each of them faces a one-equation/two-unknown residual
        (Remark 7), so each commits to an arbitrary point of its degenerate
        family.  The final cell's residual --- the only over-determined one,
        and hence the only thing that can pick within those families --- then
        arrives too late, and its information has to travel back up a chain
        that has already settled.  Training jointly lets that constraint act
        from the first step.

        Incompatible with ``local_only``.
    checkpoint_every
        Write a resumable checkpoint every this many iterations; ``0`` disables
        it.  Set it on a cluster, where jobs are preempted and time-limited.
    lr_schedule
        ``"none"`` keeps the phase learning rate fixed; ``"cosine"`` anneals it
        to zero over the remaining iterations of the phase.  Annealing matters
        for this objective because the physics residual differentiates the
        decoder in ``t``: a late-training step large enough to perturb the
        fitted trajectory perturbs its derivative far more, so a fixed rate
        leaves the loss bouncing on a noise floor set by the step size rather
        than by the measurement noise.
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
    local_only: bool = False
    joint: bool = False
    lr_schedule: str = "none"
    final_global_iters: int = 0
    checkpoint_every: int = 0

    def __post_init__(self) -> None:
        if self.lr_schedule not in ("none", "cosine"):
            raise ValueError(
                f"lr_schedule must be 'none' or 'cosine', got {self.lr_schedule!r}"
            )
        if self.n_total < 1:
            raise ValueError(f"n_total must be positive, got {self.n_total}")
        # Algorithm 1's requirement, waived for the local-only ablation which
        # deliberately has no global phase.
        if self.joint and self.local_only:
            raise ValueError("joint and local_only are mutually exclusive")
        if not self.local_only and not 0 < self.n_par < self.n_total:
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
