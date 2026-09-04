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
from pibe.data.dataset import TrajectoryData, generate_dataset
from pibe.data.disturbance import (
    BasisDisturbance,
    ChirpRemainder,
    sample_coefficients,
)
from pibe.data.noise import NoiseFree, NoiseModel, TruncatedGaussianNoise
from pibe.data.simulate import fill_distance, uniform_grid
from pibe.systems.base import TriangularSystem
from pibe.systems.registry import build_system
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
        """A trainer over this experiment's data and schedule."""
        return PIBETrainer(
            bank=self.bank,
            train_data=self.train_data,
            t_coll=self.t_coll,
            config=self.config.training,
            val_data=self.val_data,
            generator=make_generator(self.config.training.seed),
        )

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
        )


def build_noise(config: RunConfig) -> NoiseModel:
    """Construct the measurement-noise law described by the config."""
    data = config.data
    if data.noise_sigma <= 0:
        return NoiseFree()
    return TruncatedGaussianNoise(
        sigma=data.noise_sigma,
        bound=data.noise_bound,
        truncation_sigmas=data.noise_truncation_sigmas,
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
        lo, hi = system.theta_bounds[:, 0], system.theta_bounds[:, 1]
        unit = torch.rand(n_traj, system.theta_dim, generator=data_generator, dtype=dtype)
        theta_per_traj = 0.25 * (2.0 * unit - 1.0)

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
