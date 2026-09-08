r"""Assembly of a complete PIBE experiment from a :class:`~pibe.config.RunConfig`.

This is the one place that knows how the pieces fit together: system, basis,
disturbance, data, bank and trainer.  Everything is built on CPU in the target
dtype and moved to the compute device at the end, which keeps RNG generators
device-independent and makes runs reproducible from the seeds alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from pibe.basis import DisturbanceBasis, build_basis
from pibe.config import RunConfig
from pibe.core.bank import EstimatorBank
from pibe.core.energy_bank import EnergyEstimatorBank
from pibe.data.dataset import TrajectoryData, generate_dataset
from pibe.data.disturbance import (
    BasisDisturbance,
    ChirpRemainder,
    sample_coefficients,
)
from pibe.data.noise import (
    NoiseFree,
    NoiseModel,
    TruncatedGaussianNoise,
    build_noise_model,
)
from pibe.data.simulate import fill_distance, uniform_grid
from pibe.systems.base import TriangularSystem
from pibe.systems.registry import build_system
from pibe.training.epibe_trainer import EPIBETrainer
from pibe.training.trainer import PIBETrainer
from pibe.utils.device import check_dtype_support, resolve_device, resolve_dtype
from pibe.utils.logging import get_logger
from pibe.utils.seeding import make_generator, seed_everything

logger = get_logger(__name__)


@dataclass
class Experiment:
    """Everything a run needs, wired together and placed on one device."""

    config: RunConfig
    system: TriangularSystem
    basis: DisturbanceBasis
    disturbance: BasisDisturbance
    noise: NoiseModel
    data: TrajectoryData
    train_data: TrajectoryData
    val_data: TrajectoryData
    bank: EstimatorBank
    t_coll: Tensor
    coefficient_bounds: Tensor
    device: torch.device
    dtype: torch.dtype

    def make_trainer(self) -> PIBETrainer:
        """A trainer over this experiment's data and schedule.

        Returns an :class:`~pibe.training.epibe_trainer.EPIBETrainer`
        (Algorithm 2) when the bank carries energy models, and the plain
        Algorithm 1 trainer otherwise.
        """
        common = dict(
            train_data=self.train_data,
            t_coll=self.t_coll,
            config=self.config.training,
            val_data=self.val_data,
            generator=make_generator(self.config.training.seed),
        )
        if isinstance(self.bank, EnergyEstimatorBank):
            return EPIBETrainer(self.bank, ebm_config=self.config.ebm, **common)
        return PIBETrainer(bank=self.bank, **common)

    def describe(self) -> str:
        """A short report of the constructed problem."""
        gamma_d = self.basis.min_gram_eigenvalue()
        n_params = sum(p.numel() for p in self.bank.parameters())
        return "\n".join(
            [
                f"system            : {self.system} on {self.device} ({self.dtype})",
                f"true theta        : {self.system.theta_true.tolist()}"
                + ("  (sampled per trajectory)"
                   if self.config.data.sample_theta_per_trajectory else ""),
                f"disturbance       : "
                + ("independent a per trajectory"
                   if self.config.data.sample_disturbance_per_trajectory
                   else "one a shared by all trajectories"),
                f"basis             : {self.basis}",
                f"  lambda_min(W)   : {gamma_d:.4e}   (Eq. 3, must be > 0)",
                f"  eps_(d,q)       : {self.disturbance.remainder_bound(self.data.t):.4e}",
                f"noise             : {self.noise}",
                f"data              : {self.data}",
                f"  train / val     : {len(self.train_data)} / {len(self.val_data)}",
                f"  data fill h_d   : {fill_distance(self.data.t):.4e}",
                f"  coll fill h_r   : {fill_distance(self.t_coll):.4e}",
                f"bank              : cells 2..{self.bank.final_index}, "
                f"{n_params} parameters",
            ]
            + (
                [
                    f"estimator         : EPIBE (Algorithm 2), "
                    f"N_EBM={self.config.ebm.n_ebm}, beta={self.config.ebm.beta:g}",
                    "residual supports (Assumption 3):",
                    self.bank.support_report(),
                ]
                if isinstance(self.bank, EnergyEstimatorBank)
                else ["estimator         : PIBE (Algorithm 1)"]
            )
        )


def build_noise(config: RunConfig) -> NoiseModel:
    """Construct the measurement-noise law described by the config.

    The truncated-Gaussian path is kept exactly as it was --- it is the only one
    that honours ``noise_bound``, and every existing run resolves to it --- so
    the non-Gaussian families and the sensor bias are strictly additive.  A
    nonzero ``noise_bias`` is the case PIBE cannot represent and EPIBE exists
    for; see :mod:`pibe.data.noise`.
    """
    data = config.data
    if data.noise_sigma <= 0 and not data.noise_bias:
        return NoiseFree()
    if data.noise_family == "gaussian" and not data.noise_bias:
        return TruncatedGaussianNoise(
            sigma=data.noise_sigma,
            bound=data.noise_bound,
            truncation_sigmas=data.noise_truncation_sigmas,
        )
    return build_noise_model(
        data.noise_family,
        data.noise_sigma,
        bias=data.noise_bias,
        bias_relative=data.noise_bias_relative,
    )


def build_experiment(config: RunConfig) -> Experiment:
    """Build every component of a run from its configuration."""
    seed_everything(config.seed)
    dtype = resolve_dtype(config.dtype)
    device = resolve_device(config.device, dtype=dtype)
    check_dtype_support(device, dtype)

    # --- system -------------------------------------------------------
    system = build_system(config.system.name, dtype=dtype, **config.system.kwargs)

    # --- disturbance basis, Eq. (2) -----------------------------------
    horizon = config.data.horizon
    basis = build_basis(
        kind=config.basis.kind,
        q=config.basis.q,
        t_start=0.0,
        t_end=horizon,
        dtype=dtype,
        **config.basis.basis_kwargs(),
    )
    gamma_d = basis.check_linear_independence()
    logger.debug("basis is linearly independent, lambda_min(W_Gamma)=%.4e", gamma_d)

    coefficient_bounds = torch.tensor(
        config.basis.coefficient_intervals(), dtype=dtype
    )

    # --- true disturbance, Eq. (2) ------------------------------------
    # A zero remainder amplitude gives the exactly recoverable case
    # (eps_{d,q} = 0); a nonzero one puts d outside the basis span.
    data_generator = make_generator(config.data.seed)
    n_traj = config.data.n_trajectories
    if config.data.sample_disturbance_per_trajectory:
        unit = torch.rand(n_traj, basis.q, generator=data_generator, dtype=dtype)
        coefficients = coefficient_bounds[:, 0] + unit * (
            coefficient_bounds[:, 1] - coefficient_bounds[:, 0]
        )
    else:
        coefficients = sample_coefficients(
            basis, coefficient_bounds, generator=data_generator
        )
    remainder = (
        ChirpRemainder(config.basis.remainder_amplitude, config.basis.remainder_rate)
        if config.basis.remainder_amplitude > 0
        else None
    )
    disturbance = BasisDisturbance(basis, coefficients, remainder=remainder)
    system.disturbance = disturbance

    # --- data, Eq. (135) ----------------------------------------------
    noise = build_noise(config)
    t_data = uniform_grid(0.0, horizon, config.data.n_samples, dtype=dtype)
    theta_per_traj = None
    if config.data.sample_theta_per_trajectory:
        # Drawn from a box strictly inside the admissible Theta_j the parameter
        # heads map onto.  Sampling to the edge would place the truth where the
        # head's tanh is already saturated and its gradient is numerically zero.
        bound = config.data.theta_sampling_bound
        limit = 0.5 * (system.theta_bounds[:, 1] - system.theta_bounds[:, 0])
        if bound >= float(limit.min()):
            raise ValueError(
                f"theta_sampling_bound={bound} reaches or exceeds the admissible "
                f"half-width {float(limit.min()):.4g}; leave a margin so the "
                f"parameter heads are not initialised against their box edge"
            )
        unit = torch.rand(n_traj, system.theta_dim, generator=data_generator, dtype=dtype)
        theta_per_traj = bound * (2.0 * unit - 1.0)

    dataset = generate_dataset(
        system=system,
        t_grid=t_data,
        n_trajectories=config.data.n_trajectories,
        theta=theta_per_traj,
        noise=noise,
        disturbance=disturbance,
        generator=data_generator,
        substeps=config.data.substeps,
    )
    train_data, val_data = dataset.split(
        config.data.train_fraction, generator=data_generator
    )

    t_coll = uniform_grid(0.0, horizon, config.training.n_collocation, dtype=dtype)

    # --- bank ---------------------------------------------------------
    # EPIBE (Section 3.2) differs only in the data term, so it is the same bank
    # with one scalar EBM bolted onto each cell; everything downstream --- the
    # estimates, the metrics, the figures --- is untouched by the choice.
    if config.ebm.enabled:
        config.ebm.validate(config.training.n_par, config.training.n_total)
        bank = EnergyEstimatorBank(
            system=system,
            basis=basis,
            n_samples=config.data.n_samples,
            coefficient_bounds=coefficient_bounds,
            architecture=config.architecture,
            ebm=config.ebm,
            noise_bound=noise.bound,
            dtype=dtype,
        )
    else:
        bank = EstimatorBank(
            system=system,
            basis=basis,
            n_samples=config.data.n_samples,
            coefficient_bounds=coefficient_bounds,
            architecture=config.architecture,
        )

    # --- placement ----------------------------------------------------
    system.to(device=device, dtype=dtype)
    basis.to(device=device, dtype=dtype)
    disturbance.to(device=device, dtype=dtype)
    bank.to(device=device, dtype=dtype)
    dataset = dataset.to(device=device, dtype=dtype)
    train_data = train_data.to(device=device, dtype=dtype)
    val_data = val_data.to(device=device, dtype=dtype)
    t_coll = t_coll.to(device=device, dtype=dtype)
    coefficient_bounds = coefficient_bounds.to(device=device, dtype=dtype)

    return Experiment(
        config=config,
        system=system,
        basis=basis,
        disturbance=disturbance,
        noise=noise,
        data=dataset,
        train_data=train_data,
        val_data=val_data,
        bank=bank,
        t_coll=t_coll,
        coefficient_bounds=coefficient_bounds,
        device=device,
        dtype=dtype,
    )
