r"""Algorithm 2: training the EPIBE bank.

Algorithm 2 and Algorithm 1 share their outer structure exactly --- the same
loop over cells :math:`k = 2,\dots,n+1`, the same minibatches of trajectory
indices, the same detached trajectory input :math:`\mathbf{U}_{k-1}`, the same
weighted global loss shape.  They differ in three places, and this class
overrides exactly those:

* the per-cell budget is split three ways instead of two
  (:mod:`pibe.training.epibe_phases`);
* the data term switches from quadratic to energy-based at :math:`N_{EBM}`, and
  the warm-up runs with a physics weight of one whatever :math:`\lambda` is
  configured;
* every step that updated an EBM is followed by a projection onto
  :math:`\mathcal{K}_k`.

Everything else --- checkpointing and resumption, the LR schedule, gradient
clipping, logging, validation --- is inherited from
:class:`~pibe.training.trainer.PIBETrainer` and behaves identically.
"""

from __future__ import annotations

from typing import Any

import torch

from pibe.core.bank import Mode
from pibe.core.energy_bank import EnergyEstimatorBank
from pibe.core.losses import CellLoss
from pibe.data.dataset import TrajectoryData
from pibe.training.epibe_phases import Phase, apply_epibe_phase, phase_for_iteration
from pibe.training.trainer import PIBETrainer
from pibe.utils.logging import get_logger

logger = get_logger(__name__)


class EPIBETrainer(PIBETrainer):
    r"""Trains an :class:`~pibe.core.energy_bank.EnergyEstimatorBank` per Algorithm 2.

    Parameters
    ----------
    bank
        The EPIBE bank.  Must be an
        :class:`~pibe.core.energy_bank.EnergyEstimatorBank`; a plain PIBE bank
        has no EBMs to schedule.
    ebm_config
        Supplies :math:`N_{EBM}`, the EBM learning rate and the projection box.
    **kwargs
        Forwarded to :class:`~pibe.training.trainer.PIBETrainer`.
    """

    def __init__(self, bank: EnergyEstimatorBank, *args, ebm_config=None, **kwargs) -> None:
        if not isinstance(bank, EnergyEstimatorBank):
            raise TypeError(
                "EPIBETrainer requires an EnergyEstimatorBank; use PIBETrainer "
                "for a bank without energy models"
            )
        super().__init__(bank, *args, **kwargs)
        self.ebm_config = ebm_config if ebm_config is not None else bank.ebm_config
        self.ebm_config.validate(self.config.n_par, self.config.n_total)
        self.bank: EnergyEstimatorBank = bank

    # ------------------------------------------------------------------
    # schedule hooks
    # ------------------------------------------------------------------

    def phase_for_iteration(self, iteration: int) -> Phase:
        """Remark 5's three-way split of the per-cell budget."""
        return phase_for_iteration(
            iteration,
            n_ebm=self.ebm_config.n_ebm,
            n_par=self.config.n_par,
            n_fit=self.ebm_config.n_fit,
            local_only=self.config.local_only,
        )

    def enter_phase(
        self, cell_index: int, phase: Phase
    ) -> tuple[list[torch.nn.Parameter], list[dict[str, Any]]]:
        r"""Apply the phase and build its optimizer groups.

        The PINN and EBM blocks become separate parameter groups so they can
        carry different learning rates.  They are fitted *to each other* --- the
        energy chases the residual distribution while the PINN moves the
        residuals --- and that coupled problem is far better behaved when the
        two step sizes can be set independently.
        """
        pinn, ebm = apply_epibe_phase(self.bank, cell_index, phase)

        pinn_lr = (
            self.config.lr_global if phase is Phase.GLOBAL else self.config.lr_local
        )
        if not pinn:  # the EBM-only stretch has no PINN group at all
            pinn_lr = 0.0
        ebm_lr = self.ebm_config.lr_ebm
        if ebm_lr is None:
            ebm_lr = pinn_lr

        groups: list[dict[str, Any]] = []
        if pinn:
            groups.append({"params": pinn, "lr": pinn_lr})
        if ebm:
            groups.append({"params": ebm, "lr": ebm_lr})

        logger.debug(
            "cell %d entering %s: %d PINN tensors, %d EBM tensors, energy=%s",
            cell_index, phase.value, len(pinn), len(ebm), self.bank.use_energy,
        )
        return pinn + ebm, groups

    def phase_mode(self, phase: Phase) -> Mode:
        """Warm-up and local both evaluate the bank with upstream detached."""
        return phase.mode

    def phase_label(self, phase: Phase) -> str:
        return phase.value

    def after_step(self, phase: Phase) -> None:
        """Project every EBM update onto :math:`\\mathcal{K}_k` (lines 17, 23)."""
        if phase.uses_energy:
            self.bank.project(self.ebm_config.weight_bound)

    # ------------------------------------------------------------------
    # the warm-up's physics weight
    # ------------------------------------------------------------------

    def step(self, cell_index: int, phase: Phase, optimizer, active):
        r"""One step, with Remark 5's warm-up physics weight of one.

        "For :math:`i < N_{EBM}`, the EBM parameters are frozen and only the
        current PINN is trained with the corresponding PIBE local objective,
        using a warm-up physics weight equal to one."  The configured
        :math:`\lambda` applies from the moment the EBM is activated onward.
        """
        if phase is Phase.WARMUP:
            configured = self.config.lam
            self.config.lam = 1.0
            try:
                return super().step(cell_index, phase, optimizer, active)
            finally:
                self.config.lam = configured
        return super().step(cell_index, phase, optimizer, active)

    # ------------------------------------------------------------------
    # validation and reporting
    # ------------------------------------------------------------------

    def evaluate(self, cell_index: int, data: TrajectoryData) -> CellLoss:
        """Held-out local loss, scored under the objective actually in force.

        The data term is whatever :attr:`~pibe.core.energy_bank.EnergyEstimatorBank.use_energy`
        currently selects, so a validation number recorded during warm-up is a
        mean square and one recorded afterwards is a likelihood.  They are not
        comparable across that boundary, which is why the phase is recorded
        alongside every history entry.
        """
        return super().evaluate(cell_index, data)

    def train_cell(self, cell_index: int, start_iteration: int = 0) -> None:
        """Train one cell through all three phases, then report its density."""
        super().train_cell(cell_index, start_iteration=start_iteration)

        if cell_index not in self.bank.energy_cells:
            return
        model = self.bank.ebm(cell_index)
        moments = model.moments()
        logger.info(
            "  cell %d EBM | mu=%+.4e | sd=%.4e | log Z=%+.4e | |1-int p|=%.2e",
            cell_index, moments.mean, moments.std, moments.log_partition,
            model.normalization_error(),
        )
        if self.history.validation:
            self.history.validation[-1].update(
                {
                    "ebm_mean": moments.mean,
                    "ebm_std": moments.std,
                    "ebm_log_partition": moments.log_partition,
                    "ebm_normalization_error": model.normalization_error(),
                }
            )
        if cell_index == 2:
            # Eq. (54): the first cell's residual is the measurement residual,
            # so its density mean is the estimate of the noise mean.
            logger.info("  mu_omega_hat = %+.6e  (Eq. 54)", moments.mean)
