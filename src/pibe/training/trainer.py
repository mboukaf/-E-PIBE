r"""Algorithm 1 --- Training Procedure for the PIBE framework.

::

     1: for k <- 2 to n + 1 do
     2:     Initialize the current cell parameters Theta^k_PINN
     3:     for i <- 0 to N_tot - 1 do
     4:         Select a mini-batch of trajectory indices
     5:         if i < N_par then                        > local pre-training
     6:             Freeze {Theta^m_PINN}_{m=2}^{k-1}
     7:             Construct U_{k-1} from detached upstream outputs,
     8:             Evaluate Phi^k and compute L^k_Loc using (22), (28) or (37)
     9:             Update only Theta^k_PINN using grad L^k_Loc
    10:         else                                     > end-to-end fine-tuning
    11:             Unfreeze {Theta^m_PINN}_{m=2}^{k}
    12:             Evaluate {Phi^m}_{m=2}^k sequentially without detaching
    13:             Compute {L^m_Loc}_{m=2}^k
    14:             L^k_Tot <- sum_{m=2}^k e^{-(k-m)/k} L^m_Loc
    15:             Update {Theta^m_PINN}_{m=2}^k using grad L^k_Tot
    16:         end if
    17:     end for
    18: end for
    19: Set d(t) <- Gamma_q(t)^T a

Line 2 needs no explicit action here: the bank is constructed with all cells
freshly initialized, and cell ``k`` is not touched by any optimizer before the
outer loop reaches it, so its parameters are still at their initialization when
its turn comes.

A note on ``torch.no_grad``: validation deliberately does *not* use it.  Every
physics residual differentiates the decoder in ``t`` via autograd, which
requires grad mode to be enabled; only ``create_graph`` is switched off, so no
backward graph is retained.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from pibe.config import TrainingConfig
from pibe.core.bank import EstimatorBank, Mode
from pibe.core.losses import CellLoss, total_loss
from pibe.data.dataset import TrajectoryBatcher, TrajectoryData
from pibe.training.callbacks import (
    History,
    IterationRecord,
    load_training_state,
    save_checkpoint,
    save_training_state,
)
from pibe.training.phases import apply_phase, mode_for_iteration
from pibe.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class StepResult:
    """The outcome of one optimizer step."""

    objective: float
    target: CellLoss
    grad_norm: float


class PIBETrainer:
    r"""Trains an :class:`~pibe.core.bank.EstimatorBank` per Algorithm 1.

    Parameters
    ----------
    bank
        The bank to train.
    train_data
        Training trajectories, :math:`\Omega^{train}`.
    t_coll
        The collocation grid :math:`\{\tau_j\}_{j=1}^{N_r}`, shape ``(N_r,)``.
    config
        Schedule and optimizer settings.
    val_data
        Optional held-out trajectories, :math:`\Omega^{test}`.
    generator
        RNG for minibatch selection.
    """

    def __init__(
        self,
        bank: EstimatorBank,
        train_data: TrajectoryData,
        t_coll: Tensor,
        config: TrainingConfig,
        val_data: TrajectoryData | None = None,
        generator: torch.Generator | None = None,
        history: History | None = None,
        checkpoint_path: Path | None = None,
    ) -> None:
        if train_data.n_samples != bank.n_samples:
            raise ValueError(
                f"bank was built for N={bank.n_samples} samples but the data "
                f"has N={train_data.n_samples}"
            )
        if t_coll.ndim != 1 or t_coll.numel() < 2:
            raise ValueError("t_coll must be a 1-D grid with at least two points")

        self.bank = bank
        self.train_data = train_data
        self.val_data = val_data
        self.t_coll = t_coll
        self.config = config
        self.history = history if history is not None else History()
        self.checkpoint_path = checkpoint_path
        self.generator = generator
        self._resume: dict | None = None
        self.batcher = TrajectoryBatcher(
            n_trajectories=train_data.n_trajectories,
            batch_size=config.batch_size,
            generator=generator,
            device=train_data.x.device,
        )

    # ------------------------------------------------------------------
    # schedule hooks
    # ------------------------------------------------------------------
    #
    # Algorithm 1 has two phases per cell; Algorithm 2 has three, and splits the
    # learning rate between the PINN and its EBM.  These five methods are the
    # only places that differ, so :class:`~pibe.training.epibe_trainer.EPIBETrainer`
    # overrides them instead of restating the loop.  For PIBE a "phase" is simply
    # a :class:`~pibe.core.bank.Mode`.

    def phase_for_iteration(self, iteration: int) -> Any:
        """Which phase iteration ``i`` of a cell's budget belongs to."""
        return mode_for_iteration(
            iteration, self.config.n_par, local_only=self.config.local_only
        )

    def enter_phase(
        self, cell_index: int, phase: Any
    ) -> tuple[list[torch.nn.Parameter], list[dict[str, Any]]]:
        """Set the freeze flags for a phase and return its optimizer groups."""
        active = apply_phase(self.bank, cell_index, phase)
        lr = (
            self.config.lr_local
            if phase is Mode.LOCAL
            else self.config.lr_global
        )
        return active, [{"params": active, "lr": lr}]

    def phase_mode(self, phase: Any) -> Mode:
        """The bank-evaluation regime a phase runs under."""
        return phase

    def phase_label(self, phase: Any) -> str:
        """Name recorded in the history and in resume checkpoints."""
        return phase.value

    def after_step(self, phase: Any) -> None:
        """Hook run after every optimizer step; a no-op for PIBE."""

    # ------------------------------------------------------------------
    # outer loop
    # ------------------------------------------------------------------

    def resume(self) -> tuple[int, int]:
        """Restore from ``checkpoint_path`` if present; return where to continue.

        Returns ``(cell_index, iteration)`` --- the point training stopped, so
        the caller resumes at the *next* iteration.  Restores the optimizer
        moments, the LR-schedule position, the history and both RNG streams, not
        only the weights: on a preempted cluster job the difference between
        resuming and restarting a cell is hours.
        """
        if self.checkpoint_path is None:
            return (2, 0)
        state = load_training_state(self.checkpoint_path)
        if state is None:
            return (2, 0)
        self.bank.load_state_dict(state["state_dict"])
        self.bank.to(device=self.train_data.x.device, dtype=self.train_data.x.dtype)
        self.history = History.from_dict(state.get("history", {}))
        if state.get("generator") is not None and self.generator is not None:
            self.generator.set_state(state["generator"])
        if state.get("torch_rng") is not None:
            torch.set_rng_state(state["torch_rng"])
        self._resume = state
        cell, iteration = int(state["cell_index"]), int(state["iteration"])
        logger.info(
            "resuming from %s at cell %d, iteration %d (%s phase)",
            self.checkpoint_path, cell, iteration + 1, state.get("mode", "?"),
        )
        return (cell, iteration + 1)

    def train(self) -> History:
        """Run the full procedure over cells ``2..n+1`` (lines 1-18).

        Under ``joint`` the outer loop is skipped: training the final cell in
        the global regime already optimizes every cell ``2..n+1`` against
        :math:`\mathcal{L}^{n+1}_{Tot}`, which is exactly joint training.
        """
        start_cell, start_iteration = self.resume()
        if self.config.joint:
            logger.info(
                "joint mode: training cells 2..%d together against L^%d_Tot",
                self.bank.final_index, self.bank.final_index,
            )
            self.train_cell(self.bank.final_index, start_iteration=start_iteration)
            return self.history
        for cell_index in self.bank.cell_indices:
            if cell_index < start_cell:
                continue
            begin = start_iteration if cell_index == start_cell else 0
            self.train_cell(cell_index, start_iteration=begin)
        return self.history

    def train_cell(self, cell_index: int, start_iteration: int = 0) -> None:
        """Train one cell through both regimes (lines 2-17)."""
        config = self.config
        started = time.perf_counter()
        if config.local_only:
            logger.info(
                "cell %d/%d: %d local iterations (local-only ablation, "
                "no end-to-end fine-tuning)",
                cell_index,
                self.bank.final_index,
                config.n_total,
            )
        else:
            logger.info(
                "cell %d/%d: %d local iterations then %d fine-tuning iterations",
                cell_index,
                self.bank.final_index,
                config.n_par,
                config.n_total - config.n_par,
            )

        # The final cell's global phase closes the identification for the
        # whole bank, so it may be given extra iterations.
        total_iterations = config.n_total + (
            config.final_global_iters
            if cell_index == self.bank.final_index and not config.local_only
            else 0
        )

        optimizer: torch.optim.Optimizer | None = None
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        active: list[torch.nn.Parameter] = []
        current_mode: Any = None

        for iteration in range(start_iteration, total_iterations):
            mode = self.phase_for_iteration(iteration)
            if mode is not current_mode:
                # Phase boundary: reset the freeze flags and build an optimizer
                # over the newly active parameter block.
                active, groups = self.enter_phase(cell_index, mode)
                optimizer = torch.optim.Adam(
                    groups, weight_decay=config.weight_decay
                )
                scheduler = (
                    torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=max(1, total_iterations - iteration)
                    )
                    if config.lr_schedule == "cosine"
                    else None
                )
                # Resuming mid-phase: restore the optimizer moments and the
                # schedule position rather than restarting them.
                if (
                    self._resume is not None
                    and self._resume.get("mode") == self.phase_label(mode)
                ):
                    if self._resume.get("optimizer") is not None:
                        optimizer.load_state_dict(self._resume["optimizer"])
                    if scheduler is not None and self._resume.get("scheduler") is not None:
                        scheduler.load_state_dict(self._resume["scheduler"])
                    self._resume = None
                current_mode = mode
                logger.debug(
                    "cell %d entering %s phase with %d trainable tensors",
                    cell_index,
                    self.phase_label(mode),
                    len(active),
                )

            assert optimizer is not None
            result = self.step(cell_index, mode, optimizer, active)
            if scheduler is not None:
                scheduler.step()

            if (
                self.checkpoint_path is not None
                and config.checkpoint_every > 0
                and (iteration + 1) % config.checkpoint_every == 0
            ):
                save_training_state(
                    self.checkpoint_path, self.bank, optimizer, scheduler,
                    cell_index=cell_index, iteration=iteration,
                    mode=self.phase_label(mode),
                    history=self.history, generator=self.generator,
                )
                # Also leave weights where every evaluation path looks for
                # them.  A job killed by its wall clock used to leave only the
                # resume file, so the run was fully recoverable but looked
                # unusable to the tooling; writing both costs one file copy per
                # checkpoint and removes that trap.
                save_checkpoint(
                    self.checkpoint_path.parent / "bank.pt", self.bank,
                    metadata={"cell_index": cell_index, "iteration": iteration,
                              "mode": self.phase_label(mode), "partial": True},
                )
                self.history.save(self.checkpoint_path.parent / "history.json")

            if config.log_every > 0 and (
                iteration % config.log_every == 0 or iteration == total_iterations - 1
            ):
                self.history.log_iteration(
                    IterationRecord(
                        cell=cell_index,
                        iteration=iteration,
                        mode=self.phase_label(mode),
                        objective=result.objective,
                        data=float(result.target.data),
                        physics=float(result.target.physics),
                        local=float(result.target.local),
                        grad_norm=result.grad_norm,
                    )
                )
                logger.info(
                    "  cell %d | i=%5d | %-6s | obj=%.4e | data=%.4e | phys=%.4e | |g|=%.2e",
                    cell_index,
                    iteration,
                    self.phase_label(mode),
                    result.objective,
                    float(result.target.data),
                    float(result.target.physics),
                    result.grad_norm,
                )

        elapsed = time.perf_counter() - started
        entry: dict[str, float | int | str] = {
            "cell": cell_index,
            "seconds": elapsed,
        }
        if self.val_data is not None:
            validation = self.evaluate(cell_index, self.val_data)
            entry.update(
                {f"val_{name}": value for name, value in validation.as_floats().items()}
            )
            logger.info(
                "  cell %d done in %.1fs | val local=%.4e (data=%.4e, phys=%.4e)",
                cell_index,
                elapsed,
                float(validation.local),
                float(validation.data),
                float(validation.physics),
            )
        else:
            logger.info("  cell %d done in %.1fs", cell_index, elapsed)
        self.history.log_validation(entry)

    # ------------------------------------------------------------------
    # one step
    # ------------------------------------------------------------------

    def step(
        self,
        cell_index: int,
        phase: Any,
        optimizer: torch.optim.Optimizer,
        active: list[torch.nn.Parameter],
    ) -> StepResult:
        """One optimizer step (lines 4-15)."""
        index = self.batcher.sample()
        y = self.train_data.y[index]
        mode = self.phase_mode(phase)

        outputs = self.bank(
            target_cell=cell_index,
            y=y,
            t_data=self.train_data.t,
            t_coll=self.t_coll,
            mode=mode,
        )
        losses = self.bank.losses(
            target_cell=cell_index,
            y=y,
            t_coll=self.t_coll,
            outputs=outputs,
            lam=self.config.lam,
            mode=mode,
        )

        if mode is Mode.LOCAL:
            objective = losses[cell_index].local
        else:
            objective = total_loss(
                {m: loss.local for m, loss in losses.items()}, cell_index
            )

        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        if self.config.grad_clip is not None:
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(active, self.config.grad_clip)
            )
        else:
            grad_norm = float(
                torch.sqrt(
                    sum(
                        (p.grad.detach() ** 2).sum()
                        for p in active
                        if p.grad is not None
                    )
                )
            )
        optimizer.step()
        self.after_step(phase)

        # Detach before reporting: the recorded scalars must not keep the
        # step's graph alive past the optimizer update.
        target = losses[cell_index]
        return StepResult(
            objective=float(objective.detach()),
            target=CellLoss(
                data=target.data.detach(),
                physics=target.physics.detach(),
                local=target.local.detach(),
            ),
            grad_norm=grad_norm,
        )

    # ------------------------------------------------------------------
    # evaluation
    # ------------------------------------------------------------------

    def evaluate(self, cell_index: int, data: TrajectoryData) -> CellLoss:
        """The target cell's local loss on held-out trajectories.

        Runs with grad mode enabled but ``create_graph=False``: the physics
        residual needs the autograd time derivative, while no backward graph
        is required.
        """
        was_training = self.bank.training
        self.bank.eval()
        try:
            with torch.enable_grad():
                outputs = self.bank(
                    target_cell=cell_index,
                    y=data.y,
                    t_data=data.t,
                    t_coll=self.t_coll,
                    mode=Mode.GLOBAL,
                    create_graph=False,
                )
                loss = self.bank.cell_loss(
                    cell_index=cell_index,
                    y=data.y,
                    t_coll=self.t_coll,
                    outputs=outputs,
                    lam=self.config.lam,
                )
        finally:
            self.bank.train(was_training)
        return CellLoss(
            data=loss.data.detach(),
            physics=loss.physics.detach(),
            local=loss.local.detach(),
        )
