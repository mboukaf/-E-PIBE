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
        # The global offset moves only in the final cell's end-to-end phase.
        # Upstream of it every cell's new coordinate absorbs a shift of x_hat_1
        # exactly, so the offset would see nothing but minibatch noise, and
        # Adam turns a noise-only gradient into a random walk.
        offset = self.bank.offset
        extra: list[torch.nn.Parameter] = []
        if offset is not None:
            moves = phase is Phase.GLOBAL and cell_index == self.bank.final_index
            offset.requires_grad_(moves)
            if moves:
                lr = self.ebm_config.lr_offset
                groups.append({"params": [offset],
                               "lr": self.config.lr_global if lr is None else lr})
                extra = [offset]

        logger.debug(
            "cell %d entering %s: %d PINN tensors, %d EBM tensors, energy=%s",
            cell_index, phase.value, len(pinn), len(ebm), self.bank.use_energy,
        )
        return pinn + ebm + extra, groups

    def phase_mode(self, phase: Phase) -> Mode:
        """Warm-up and local both evaluate the bank with upstream detached."""
        return phase.mode

    def phase_label(self, phase: Phase) -> str:
        return phase.value

    def after_step(self, phase: Phase) -> None:
        """Project every EBM update onto :math:`\\mathcal{K}_k` (lines 17, 23)."""
        if phase.uses_energy:
            self.bank.project(self.ebm_config.weight_bound)

    def measurement_for_step(self, y: torch.Tensor) -> torch.Tensor:
        r"""During the amortization stage, shift each trajectory by its own random offset.

        The bank then learns the chain for every sensor offset in the prior
        window, not only for the one the data happen to carry, which is what
        lets :meth:`locate` compare offsets by feeding the bank ``y - mu``.
        """
        if not getattr(self, "_amortizing", False):
            return y
        lo, hi = self.ebm_config.offset_window
        unit = torch.rand(y.shape[0], 1, generator=self.generator, dtype=y.dtype)
        return y - (lo + (hi - lo) * unit).to(y.device)

    def train(self):
        """Algorithm 2; under the amortized location, then amortization and location search.

        The stages after Algorithm 2 are resumable, like Algorithm 2 itself.
        Each completed stage is recorded in ``stages.json`` next to the
        training state with the bank's weights at that point; offset training
        also checkpoints mid-stage, and every scored offset is cached, so a
        requeued job loses at most one checkpoint interval.
        """
        cfg = self.ebm_config
        if cfg.location != "amortized":
            return super().train()

        record = self._load_stages()
        if "algorithm2" in record["done"]:
            # resume() brings back the history Algorithm 2 recorded; the weights
            # it loads are then replaced by those of the last finished stage.
            self.resume()
            self._restore_stage_bank()
            logger.info("resuming after Algorithm 2 (stages done: %s)", ", ".join(record["done"]))
            history = self.history
        else:
            history = super().train()
            self._finish_stage(record, "algorithm2")

        if "amortize" not in record["done"]:
            self.amortize(stage="amortize")
            self._finish_stage(record, "amortize")

        if "locate" not in record["done"]:
            record["first"] = self.locate(stage="locate", record=record)
            self._finish_stage(record, "locate")
        else:
            self.bank.location.fill_(record["first"])
        first = record["first"]

        if cfg.recenter_iters > 0:
            # A true offset near the edge of the prior window sits where the
            # amortized bank is least accurate.  Re-amortize on a window of the
            # same width centred on the first estimate -- the width is kept
            # because the spread of offsets itself regularizes the bank (a
            # narrow window, or a fine-tune at a single offset, measurably loses
            # accuracy) -- and search again, only the fine pass around it.
            half = 0.5 * (cfg.offset_window[1] - cfg.offset_window[0])
            prior = list(cfg.offset_window)
            cfg.offset_window = [first - half, first + half]
            try:
                if "recenter" not in record["done"]:
                    self.amortize(iterations=cfg.recenter_iters, stage="recenter")
                    self._finish_stage(record, "recenter")
                if "relocate" not in record["done"]:
                    record["final"] = self.locate(centre=first, stage="relocate", record=record)
                    self._finish_stage(record, "relocate")
                else:
                    self.bank.location.fill_(record["final"])
            finally:
                cfg.offset_window = prior
        return history

    # ------------------------------------------------------------------
    # stage bookkeeping for the amortized location
    # ------------------------------------------------------------------

    def _stage_dir(self):
        return None if self.checkpoint_path is None else self.checkpoint_path.parent

    def _load_stages(self) -> dict:
        import json

        directory = self._stage_dir()
        path = None if directory is None else directory / "stages.json"
        if path is not None and path.exists():
            return json.loads(path.read_text())
        return {"done": [], "scores": {}}

    def _save_stages(self, record: dict) -> None:
        import json

        directory = self._stage_dir()
        if directory is None:
            return
        directory.mkdir(parents=True, exist_ok=True)
        tmp = directory / "stages.json.tmp"
        tmp.write_text(json.dumps(record, indent=1))
        tmp.replace(directory / "stages.json")

    def _finish_stage(self, record: dict, name: str) -> None:
        directory = self._stage_dir()
        if directory is not None:
            tmp = directory / "stage_bank.pt.tmp"
            torch.save({"state_dict": self.bank.state_dict()}, tmp)
            tmp.replace(directory / "stage_bank.pt")
            (directory / "stage_state.pt").unlink(missing_ok=True)
        record["done"].append(name)
        self._save_stages(record)
        logger.info("stage '%s' complete", name)

    def _restore_stage_bank(self) -> None:
        directory = self._stage_dir()
        payload = torch.load(directory / "stage_bank.pt", map_location="cpu", weights_only=False)
        self.bank.load_state_dict(payload["state_dict"])
        self.bank.to(device=self.train_data.x.device, dtype=self.train_data.x.dtype)

    def _search_subset(self):
        """The trajectories the offset search and density refit run on."""
        data, k = self.train_data, self.ebm_config.offset_trajectories
        if k is None or k >= len(data):
            return data.y, data.t
        generator = torch.Generator().manual_seed(int(self.config.seed))
        index = torch.randperm(len(data), generator=generator)[:k].to(data.y.device)
        return data.y[index], data.t

    # ------------------------------------------------------------------
    # amortization and location search
    # ------------------------------------------------------------------

    def amortize(self, iterations: int | None = None, stage: str = "amortize") -> None:
        r"""End-to-end fine-tuning on randomly offset measurements.

        Runs after Algorithm 2 has converged, on the final cell's global
        objective, with every trajectory fed :math:`y - \tilde\mu` for its own
        :math:`\tilde\mu` drawn from :attr:`~pibe.config.EBMConfig.offset_window`.

        Two choices matter and both were forced by failures.  It starts from a
        converged bank: drawing offsets from the first iteration leaves the
        cell-by-cell stages, which cannot identify anything about the location,
        with a moving target, and the parameter heads end pinned against their
        boxes.  And the first cell is scored on the quadratic term, not on the
        location-free likelihood: the point is that feeding ``y - mu`` yields
        the chain *at* offset ``mu``, so the reconstruction must be anchored to
        its input rather than left free to slide.
        """
        cfg, final = self.ebm_config, self.bank.final_index
        iterations = cfg.amortize_iters if iterations is None else iterations
        if iterations <= 0:
            return
        logger.info("%s: %d end-to-end iterations on offsets in [%+.3f, %+.3f]",
                    stage, iterations, *cfg.offset_window)
        active, groups = self.enter_phase(final, Phase.GLOBAL)
        # Only the PINNs move; the densities are refitted after the search.
        ebm_params = set(self.bank.ebm_parameters_upto(final))
        groups = [g for g in groups if not any(p in ebm_params for p in g["params"])]
        self.bank.set_ebm_trainable_upto(final, False)
        if self.bank.offset is not None:
            self.bank.offset.requires_grad_(False)
            groups = [g for g in groups if not any(p is self.bank.offset for p in g["params"])]
        params = [p for g in groups for p in g["params"]]
        optimizer = torch.optim.Adam(groups, weight_decay=self.config.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)

        begin = 0
        directory = self._stage_dir()
        state_path = None if directory is None else directory / "stage_state.pt"
        if state_path is not None and state_path.exists():
            state = torch.load(state_path, map_location="cpu", weights_only=False)
            if state.get("stage") == stage:
                self.bank.load_state_dict(state["state_dict"])
                self.bank.to(device=self.train_data.x.device, dtype=self.train_data.x.dtype)
                optimizer.load_state_dict(state["optimizer"])
                scheduler.load_state_dict(state["scheduler"])
                if state.get("generator") is not None and self.generator is not None:
                    self.generator.set_state(state["generator"])
                begin = int(state["iteration"]) + 1
                logger.info("  resuming %s at iteration %d", stage, begin)

        self._amortizing = True
        self.bank.use_energy = False
        try:
            for iteration in range(begin, iterations):
                result = self.step(final, Phase.GLOBAL, optimizer, params)
                scheduler.step()
                if self.config.log_every > 0 and (
                    iteration % self.config.log_every == 0 or iteration == iterations - 1
                ):
                    logger.info("  %s | i=%5d | obj=%.4e | data=%.4e | phys=%.4e | |g|=%.2e",
                                stage, iteration, result.objective, float(result.target.data),
                                float(result.target.physics), result.grad_norm)
                every = self.config.checkpoint_every
                if state_path is not None and every > 0 and (iteration + 1) % every == 0:
                    tmp = state_path.with_suffix(".tmp")
                    torch.save({
                        "stage": stage, "iteration": iteration,
                        "state_dict": self.bank.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "generator": None if self.generator is None else self.generator.get_state(),
                    }, tmp)
                    tmp.replace(state_path)
        finally:
            self._amortizing = False
            self.bank.use_energy = True

    def locate(self, centre: float | None = None, stage: str = "locate",
               record: dict | None = None) -> float:
        r"""Select the sensor offset by profiling the physics over the prior window.

        The first cell's data term is fitted equally well at every candidate
        offset, so the profile is decided by the physics alone.  Each candidate
        is scored as configured by ``ebm.offset_score`` (see
        :mod:`pibe.eval.offset_shooting`), the minimizer is found by the
        two-pass search of :func:`~pibe.eval.offset_shooting.select_offset` and
        stored as ``bank.location``, and the first cell's density is then
        refitted to the residual that remains, so that Eq. (54) reports location
        and shape together.  Runs on :attr:`~pibe.config.EBMConfig.offset_trajectories`
        training trajectories.
        """
        from pibe.eval.offset_shooting import select_offset

        bank, cfg = self.bank, self.ebm_config
        y, t = self._search_subset()
        logger.info("%s: offset search on %d trajectories", stage, y.shape[0])
        record = record if record is not None else {"done": [], "scores": {}}
        cache = record["scores"].setdefault(stage, {})
        was_training = bank.training
        bank.eval()
        if cfg.offset_score == "shooting":
            best, _, _ = select_offset(
                bank, y, t, self.t_coll, tuple(cfg.offset_window), cfg.offset_grid,
                coarse_steps=cfg.offset_steps, fine_points=cfg.offset_fine,
                fine_steps=cfg.offset_fine_steps, log=logger.info, centre=centre,
                cache=cache, on_score=lambda _: self._save_stages(record),
            )
        else:
            offsets = torch.linspace(cfg.offset_window[0], cfg.offset_window[1],
                                     cfg.offset_grid, dtype=y.dtype)
            costs = [bank.chain_cost(y - float(c), t, self.t_coll, self.config.lam)
                     for c in offsets]
            for c, cost in zip(offsets.tolist(), costs):
                logger.info("  offset %+.4f | chain cost %.5e", c, cost)
            best = float(offsets[int(torch.tensor(costs).argmin())])
        bank.train(was_training)
        bank.location.fill_(best)
        logger.info("selected sensor offset = %+.5f", best)

        if cfg.offset_refit > 0 and 2 in bank.energy_cells:
            with torch.no_grad():
                outputs = bank(bank.final_index, y - bank.location, t, self.t_coll,
                               mode=Mode.GLOBAL, need_derivative=False)
                residual = (y - bank.location - outputs[2].x_prev_data).reshape(-1)
            model = bank.ebm(2)
            bank.set_ebm_trainable(2, True)
            lr = cfg.lr_ebm if cfg.lr_ebm is not None else self.config.lr_local
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            for _ in range(cfg.offset_refit):
                optimizer.zero_grad(set_to_none=True)
                model.negative_log_likelihood(residual).backward()
                optimizer.step()
                model.project(cfg.weight_bound)
            logger.info("mu_omega_hat = %+.6e after refitting the density at the selected offset",
                        bank.noise_mean())
        return best

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
