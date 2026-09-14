r"""The EPIBE bank: a PIBE cascade whose data terms are learned likelihoods.

Section 3.2 changes exactly one thing about the estimator.  Step 3 is explicit
that "the EPIBE uses the same final cell (35), coefficient vector
:math:`\hat a`, physics residual (40), and disturbance estimate (41) as the
PIBE.  Only its consistency term is replaced by the energy-based data loss", and
the same holds cell by cell for Eqs. (52) and (55).  So this class is a subclass
rather than a parallel implementation: the cells, the trajectory input (15), the
physics residuals (25)/(31)/(40), the reported estimates and the global weighting
are inherited unchanged, and only :meth:`~EnergyEstimatorBank.cell_loss` is
overridden.

.. math::

    \mathcal{L}^2_{Loc,E}     &= \mathcal{L}^{ebm,2}_{Data}(\{y - \hat x^2_1\})
                               + \lambda \mathcal{L}^2_{Physics} & &\text{(52)}\\
    \mathcal{L}^k_{Loc,E}     &= \mathcal{L}^{ebm,k}_{Data}
                                 (\{\hat x^{k-1}_{k-1} - \hat x^k_{k-1}\})
                               + \lambda \mathcal{L}^k_{Physics} & &\text{(55)}\\
    \mathcal{L}^{n+1}_{Loc,E} &= \mathcal{L}^{ebm,n+1}_{Data}
                                 (\{\hat x^n_n - \hat x^{n+1}_n\})
                               + \lambda \mathcal{L}^{n+1}_{Physics} & &\text{(57)}

One EBM per cell, not one for the measurement
---------------------------------------------
Only :math:`\varepsilon_2` is a measurement residual.  The rest are inter-cell
consistency residuals, and Section 3.2 declines to assume anything about how the
measurement's law propagates into them: "because the measurement enters only the
first coordinate, how its uncertainty propagates to the downstream states is
unknown; we therefore fit a separate residual model at each cell rather than
assuming that the original measurement-noise law is preserved throughout the
bank."  Hence :attr:`EnergyEstimatorBank.ebms`, keyed by cell exactly like
:attr:`~pibe.core.bank.EstimatorBank.cells`.

That said, the downstream residuals differ from the first in a way that matters
in practice.  Only :math:`\varepsilon_2` contains measurement noise; Eqs. (29)
and (38) are, in PIBE's own words, "deterministic inter-cell consistency
penalties", whose correct value is identically zero.  Replacing a penalty that
*should* be zero with a likelihood under a density that is free to sit anywhere
removes the only thing pinning two independently parameterized reconstructions
of the same coordinate to each other, and they drift.  :attr:`energy_cells`
exists so the energy term can be restricted to the cells that actually see
noise; ``ebm.cells: [2]`` is that configuration.

Parameter blocks stay separate
------------------------------
Algorithm 2 freezes and unfreezes :math:`\Theta^k_{PINN}` and
:math:`\Theta^k_{EBM}` independently and on different schedules, so the EBMs
live in their own module dictionary.  The inherited
:meth:`~pibe.core.bank.EstimatorBank.cell_parameters` and
:meth:`~pibe.core.bank.EstimatorBank.parameters_upto` therefore keep returning
PINN parameters only, and :meth:`ebm_parameters` /
:meth:`ebm_parameters_upto` mirror them for the energy models.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace

import torch
from torch import Tensor, nn

from pibe.basis.base import DisturbanceBasis
from pibe.config import ArchitectureConfig, EBMConfig
from pibe.core.bank import EstimatorBank, Mode
from pibe.core.cell import CellOutput
from pibe.core.losses import CellLoss, data_loss, local_loss, physics_loss
from pibe.core.residuals import final_residual, state_residual
from pibe.ebm.density import DensityMoments, ScalarEBM
from pibe.ebm.support import resolve_radii
from pibe.systems.base import TriangularSystem
from pibe.utils.logging import get_logger

logger = get_logger(__name__)


class EnergyEstimatorBank(EstimatorBank):
    r"""A PIBE bank augmented with one scalar EBM per cell.

    Parameters
    ----------
    system, basis, n_samples, coefficient_bounds, architecture
        As for :class:`~pibe.core.bank.EstimatorBank`; the PINN half is
        unchanged.
    ebm
        EBM hyperparameters, including the support-radius override.
    noise_bound
        :math:`\bar w`, needed for the first cell's a priori radius under
        Assumption 3.  See :mod:`pibe.ebm.support`.

    Attributes
    ----------
    use_energy
        Whether :meth:`cell_loss` uses the energy-based data term.  ``False``
        selects PIBE's quadratic term, which is what Algorithm 2's PINN-only
        warm-up (``i < N_EBM``) runs on; the trainer flips it.  It is *not* a
        way to disable EPIBE --- for that, use :class:`EstimatorBank`.
    """

    def __init__(
        self,
        system: TriangularSystem,
        basis: DisturbanceBasis,
        n_samples: int,
        coefficient_bounds: Tensor,
        architecture: ArchitectureConfig | None = None,
        ebm: EBMConfig | None = None,
        noise_bound: float = 0.0,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__(
            system=system,
            basis=basis,
            n_samples=n_samples,
            coefficient_bounds=coefficient_bounds,
            architecture=architecture,
        )
        self.ebm_config = ebm or EBMConfig()
        self.radii = resolve_radii(
            system, noise_bound=noise_bound, override=self.ebm_config.radius
        )
        self.ebms = nn.ModuleDict(
            {
                str(k): ScalarEBM(
                    radius=self.radii[k],
                    beta=self.ebm_config.beta,
                    hidden=tuple(self.ebm_config.hidden),
                    activation=self.ebm_config.activation,
                    energy_scale=self.ebm_config.energy_scale,
                    energy_bound=self.ebm_config.energy_bound,
                    spectral_norm_layers=self.ebm_config.spectral_norm,
                    symmetric=self.ebm_config.symmetric,
                    panels=self.ebm_config.panels,
                    nodes_per_panel=self.ebm_config.nodes_per_panel,
                    barrier_scale=self.ebm_config.barrier_scale,
                    dtype=dtype,
                )
                for k in self.cell_indices
            }
        )
        # Which cells actually use their EBM.  ``None`` is Step 2 as written:
        # every cell.  Restricting it to {2} keeps the quadratic consistency
        # terms downstream; see the class docstring.
        self.energy_cells = (
            set(self.cell_indices)
            if self.ebm_config.cells is None
            else {int(k) for k in self.ebm_config.cells}
        )
        unknown = self.energy_cells - set(self.cell_indices)
        if unknown:
            raise ValueError(
                f"ebm.cells refers to cells {sorted(unknown)} outside "
                f"2..{self.final_index}"
            )
        # Warm-up first: Remark 5 activates the EBMs only after the PINN has
        # produced residuals worth fitting.
        self.use_energy = False

        # Optional global shift of x_hat^2_1, see EBMConfig.offset_parameter.
        # Registered only when requested so existing checkpoints load unchanged.
        if self.ebm_config.offset_parameter:
            self.offset = nn.Parameter(torch.zeros((), dtype=dtype))
        else:
            self.offset = None
        # Running mean of the first cell's raw residual.  Under a profiled
        # location the density never sees it, so it is what carries the
        # location into mu_omega_hat.
        self.register_buffer(
            "residual_mean", torch.zeros((), dtype=dtype),
            persistent=self.ebm_config.location == "profiled",
        )
        # The sensor offset selected by the amortized location search; the
        # bank sees y - location at inference.
        self.register_buffer(
            "location", torch.zeros((), dtype=dtype),
            persistent=self.ebm_config.location == "amortized",
        )

    # ------------------------------------------------------------------
    # structure
    # ------------------------------------------------------------------

    def uses_energy_at(self, index: int) -> bool:
        """Whether cell ``index`` currently scores its data term with its EBM."""
        return self.use_energy and index in self.energy_cells

    def ebm(self, index: int) -> ScalarEBM:
        r"""The energy model :math:`\Theta^k_{EBM}` of cell ``index``."""
        if index not in self.cell_indices:
            raise KeyError(
                f"cell index {index} out of range; expected 2..{self.final_index}"
            )
        return self.ebms[str(index)]  # type: ignore[return-value]

    def ebm_parameters(self, index: int) -> Iterator[nn.Parameter]:
        """Parameters of a single cell's EBM."""
        return self.ebm(index).parameters()

    def ebm_parameters_upto(self, index: int) -> Iterator[nn.Parameter]:
        """Parameters of the EBMs of cells ``2..index``."""
        for k in range(2, index + 1):
            yield from self.ebm(k).parameters()

    def set_ebm_trainable(self, index: int, trainable: bool) -> None:
        """Freeze or unfreeze one cell's EBM."""
        for parameter in self.ebm(index).parameters():
            parameter.requires_grad_(trainable)

    def set_ebm_trainable_upto(self, index: int, trainable: bool) -> None:
        """Freeze or unfreeze the EBMs of cells ``2..index``."""
        for k in range(2, index + 1):
            self.set_ebm_trainable(k, trainable)

    @torch.no_grad()
    def project(self, weight_bound: float | None = None) -> None:
        r"""Project every EBM's parameters onto :math:`\mathcal{K}_k`.

        Algorithm 2 lines 17 and 23 constrain each EBM update to its compact
        parameter set; the trainer calls this after every optimizer step in
        which an EBM was active.
        """
        bound = self.ebm_config.weight_bound if weight_bound is None else weight_bound
        for k in self.cell_indices:
            self.ebm(k).project(bound)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(self, target_cell, y, t_data, t_coll, mode=Mode.GLOBAL,
                need_derivative=True, create_graph=True):
        """The inherited chain, with the optional global offset added to x_hat^2_1.

        The offset is constant in time, so the derivative is unchanged.  Cell 3
        onwards never reads x_hat^2_1 as an encoder input, so shifting it after
        the chain has run is identical to shifting it inside.
        """
        outputs = super().forward(
            target_cell, y, t_data, t_coll, mode=mode,
            need_derivative=need_derivative, create_graph=create_graph,
        )
        if self.offset is not None:
            shift = self.offset
            if mode is Mode.LOCAL and target_cell != 2:
                shift = shift.detach()
            first = outputs[2]
            outputs[2] = replace(
                first,
                x_prev_data=first.x_prev_data + shift,
                x_prev_coll=first.x_prev_coll + shift,
            )
        return outputs

    @torch.no_grad()
    def estimate(self, y: Tensor, t_data: Tensor, t_coll: Tensor, offset: float | None = None):
        """The inherited readout of ``y - offset``; the stored location when ``offset`` is None."""
        shift = self.location if offset is None else offset
        return super().estimate(y - shift, t_data, t_coll)

    # ------------------------------------------------------------------
    # amortized location search
    # ------------------------------------------------------------------

    def chain_cost(self, y: Tensor, t_data: Tensor, t_coll: Tensor, lam: float) -> float:
        r"""Every term of :math:`\mathcal{L}^{n+1}_{Tot}` except the first cell's data term.

        That term is fitted at every candidate offset alike, so it carries no
        information about the location; what is left is the physics of every
        cell and the downstream consistency penalties, weighted as in Eq. (27).
        Always scored on PIBE's quadratic consistency terms.
        """
        from pibe.core.losses import global_weights

        was_training, was_energy = self.training, self.use_energy
        self.eval()
        self.use_energy = False
        try:
            with torch.enable_grad():
                outputs = self.forward(self.final_index, y, t_data, t_coll,
                                       mode=Mode.GLOBAL, create_graph=False)
                weights = global_weights(self.final_index)
                total = 0.0
                for k in self.cell_indices:
                    loss = super().cell_loss(k, y, t_coll, outputs, lam)
                    total += weights[k] * (
                        lam * float(loss.physics) + (0.0 if k == 2 else float(loss.data))
                    )
        finally:
            self.train(was_training)
            self.use_energy = was_energy
        return total

    def offset_profile(self, y: Tensor, t_data: Tensor, t_coll: Tensor, lam: float,
                       offsets: Tensor) -> Tensor:
        """:meth:`chain_cost` of ``y - c`` for every candidate offset ``c``."""
        return torch.tensor(
            [self.chain_cost(y - float(c), t_data, t_coll, lam) for c in offsets],
            dtype=y.dtype,
        )

    # ------------------------------------------------------------------
    # residuals and losses
    # ------------------------------------------------------------------

    def upstream_target(
        self, cell_index: int, y: Tensor, outputs: dict[int, CellOutput]
    ) -> Tensor:
        """What cell ``k`` reconstructs: the measurement, or cell ``k-1``'s output."""
        if cell_index == 2:
            return y
        target = outputs[cell_index - 1].x_new_data
        if target is None:  # pragma: no cover - unreachable by construction
            raise ValueError(f"cell {cell_index - 1} produced no new coordinate")
        return target

    def consistency_residual(
        self, cell_index: int, y: Tensor, outputs: dict[int, CellOutput]
    ) -> Tensor:
        r"""The scalar residual :math:`\varepsilon_k` that cell ``k``'s EBM models.

        :math:`\varepsilon_2 = y - \hat x^2_1` is the measurement residual; for
        ``k >= 3`` it is :math:`\hat x^{k-1}_{k-1} - \hat x^k_{k-1}`, the
        difference between the upstream estimate of coordinate ``k-1`` and this
        cell's reconstruction of it.

        The sign matters and is the paper's: :math:`\hat\mu_\omega` in Eq. (54)
        is the mean of *this* quantity, so a sensor reading high must give a
        positive mean.  PIBE's quadratic term squares the same difference and is
        blind to the convention, which is precisely the information EPIBE keeps.
        """
        target = self.upstream_target(cell_index, y, outputs)
        return target - outputs[cell_index].x_prev_data

    def cell_loss(
        self,
        cell_index: int,
        y: Tensor,
        t_coll: Tensor,
        outputs: dict[int, CellOutput],
        lam: float,
    ) -> CellLoss:
        r"""The local objective, Eqs. (52)/(55)/(57), or PIBE's (22)/(28)/(37).

        The physics half is inherited verbatim --- Section 3.2 changes only the
        consistency term --- so the two branches differ in one line.  Under
        :attr:`use_energy` the reported ``data`` field is a negative
        log-likelihood and may be negative; under warm-up it is the usual mean
        square.
        """
        target = self.upstream_target(cell_index, y, outputs)
        prediction = outputs[cell_index].x_prev_data

        if cell_index == self.final_index:
            physics = final_residual(
                self.system, self.basis, outputs[cell_index], outputs, t_coll
            )
        else:
            physics = state_residual(
                self.system, cell_index, outputs[cell_index], outputs
            )

        if self.uses_energy_at(cell_index):
            residual = target - prediction
            model = self.ebm(cell_index)
            if cell_index == 2:
                if self.training:
                    with torch.no_grad():
                        self.residual_mean.mul_(0.99).add_(0.01 * residual.mean())
                if self.ebm_config.location in ("profiled", "amortized"):
                    residual = residual - residual.mean()
            data = model.negative_log_likelihood(residual)
            if cell_index == 2 and self.ebm_config.nll_scale == "variance":
                data = data * model.moments().std ** 2
        elif (
            self.ebm_config.centered_warmup and cell_index in self.energy_cells
        ):
            # The warm-up with the location profiled out.  Proposition 1 gives
            # the quadratic data risk as
            #
            #     R_D(y_hat, m) = sigma^2 + (1/T) int (y_hat - y + m - mu)^2,
            #
            # whose minimum over the unknown location m is attained at
            # m = mu - mean(y_hat - y).  Substituting it back leaves exactly the
            # *variance* of the residual: the quadratic term with the location
            # eliminated rather than assumed.  Using it here means the warm-up
            # fixes the shape of x_hat_1 without committing to where it sits,
            # so the location is left to the physics term -- which is the
            # division of labour Proposition 1 actually prescribes.  The plain
            # quadratic term instead pins the location at m = 0, handing EPIBE
            # PIBE's biased answer as its starting point.
            residual = target - prediction
            data = ((residual - residual.mean()) ** 2).mean()
        else:
            # Algorithm 2 line 9: the warm-up runs "the corresponding PIBE local
            # objective", i.e. the quadratic term over the same residual.
            data = data_loss(prediction, target)

        return local_loss(data, physics_loss(physics), lam)

    # ------------------------------------------------------------------
    # diagnostics
    # ------------------------------------------------------------------

    @torch.no_grad()
    def noise_mean(self) -> float:
        r""":math:`\hat\mu_\omega` of Eq. (54), the first cell's density mean.

        Reported only.  Eq. (54) "is not an independently optimized offset and
        equals the true noise mean only when the physical signal is correctly
        identified and the learned first-cell residual density is statistically
        consistent", so this is a moment of a fitted density and inherits every
        caveat attached to that fit.
        """
        mean = self.ebm(2).moments().mean
        if self.ebm_config.location == "profiled":
            # The density was fitted to centred residuals; the location lives
            # in the residual mean instead.
            mean += float(self.residual_mean)
        elif self.ebm_config.location == "amortized":
            # The density was refitted to y - location - x_hat_1.
            mean += float(self.location)
        return mean

    @torch.no_grad()
    def density_moments(self) -> dict[int, DensityMoments]:
        """Moments of the learned residual density of every energy cell."""
        return {k: self.ebm(k).moments() for k in sorted(self.energy_cells)}

    @torch.no_grad()
    def support_report(self) -> str:
        """The Assumption 3 constants actually in force, one line per cell."""
        lines = [
            f"{'cell':>5} | {'rho_k':>10} | {'B_E':>7} | {'L_E':>10} | "
            f"{'nodes':>6} | {'spacing':>10} | {'|1-int p|':>10}"
        ]
        lines.append("-" * len(lines[0]))
        for k in sorted(self.energy_cells):
            model = self.ebm(k)
            lines.append(
                f"{k:>5} | {model.radius:>10.4g} | {model.energy.energy_bound:>7.3g} | "
                f"{model.energy.lipschitz_constant:>10.4g} | "
                f"{model.quadrature.n_nodes:>6} | {model.quadrature.spacing:>10.3g} | "
                f"{model.normalization_error():>10.2e}"
            )
        return "\n".join(lines)

    def extra_repr(self) -> str:  # pragma: no cover - trivial
        return (
            f"n={self.state_dim}, cells=2..{self.final_index}, "
            f"N={self.n_samples}, q={self.basis.q}, "
            f"EBM beta={self.ebm_config.beta:g}"
        )
