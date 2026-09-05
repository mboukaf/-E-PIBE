r"""The three-phase schedule of Algorithm 2 / Remark 5.

PIBE splits each cell's budget once, at :math:`N_{par}`.  EPIBE splits it twice,
at :math:`N_{EBM}` and then at :math:`N_{par}`, with

.. math:: 0 < N_{EBM} < N_{par} < N_{tot},

because the joint PINN--EBM optimization "is sensitive to initialization because
the EBM is fitted to residuals produced by an initially untrained PINN".  The
failure mode is concrete: at initialization the residual is dominated by
reconstruction error rather than by noise, so an EBM switched on immediately
learns the *error's* distribution and then rewards the PINN for reproducing it.

The three phases, from Remark 5 and Algorithm 2 lines 5-24:

``i < N_EBM`` --- PINN-only warm-up (lines 5-10)
    "the EBM parameters are frozen and only the current PINN is trained with the
    corresponding PIBE local objective, using a warm-up physics weight equal to
    one."  Upstream cells are frozen and their outputs detached.  Note the two
    specifics: the objective is the *quadratic* one, and :math:`\lambda` is
    forced to 1 regardless of the configured value.

``N_EBM <= i < N_par`` --- local PINN--EBM training (lines 11-17)
    "the current PINN and its EBM are trained jointly with
    :math:`\mathcal{L}^k_{Loc,E}`, while all upstream cells remain frozen."

``i >= N_par`` --- end-to-end fine-tuning (lines 18-23)
    Every PINN *and* EBM up to the current cell is unfrozen and updated against
    the weighted global energy loss (56)/(58), with upstream outputs no longer
    detached.

Both of the latter phases project every EBM update onto :math:`\mathcal{K}_k`
(lines 17 and 23), which is what keeps Assumption 3's constants in force during
training rather than only at construction.
"""

from __future__ import annotations

from enum import Enum

from torch import nn

from pibe.core.bank import Mode
from pibe.core.energy_bank import EnergyEstimatorBank


class Phase(Enum):
    """One of the per-cell regimes; ``FIT`` is optional (see :func:`phase_for_iteration`)."""

    WARMUP = "warmup"
    FIT = "fit"
    LOCAL = "local"
    GLOBAL = "global"

    @property
    def uses_energy(self) -> bool:
        """Whether the data term is the energy-based likelihood (53)."""
        return self is not Phase.WARMUP

    @property
    def trains_pinn(self) -> bool:
        """Whether :math:`\\Theta^k_{PINN}` is updated in this phase."""
        return self is not Phase.FIT

    @property
    def mode(self) -> Mode:
        r"""How the bank is evaluated.

        The warm-up and the local phase both freeze everything upstream and
        detach its outputs, which is exactly :attr:`~pibe.core.bank.Mode.LOCAL`;
        they differ in the objective, not in the evaluation.
        """
        return Mode.GLOBAL if self is Phase.GLOBAL else Mode.LOCAL


def phase_for_iteration(
    iteration: int, n_ebm: int, n_par: int, n_fit: int = 0, local_only: bool = False
) -> Phase:
    r"""Which phase iteration ``i`` of a cell's budget belongs to.

    With ``n_fit = 0`` this is exactly Algorithm 2's three-way split.  A positive
    ``n_fit`` inserts an EBM-only stretch of that length immediately after
    activation, before the joint updates of line 17 begin; see :class:`Phase`
    and the note below.

    ``local_only`` suppresses the end-to-end phase, mirroring
    :attr:`~pibe.config.TrainingConfig.local_only`.

    Why an EBM-only stretch helps
    -----------------------------
    Consider translating a cell's residuals, :math:`\varepsilon \mapsto
    \varepsilon - c`, which is what moving the PINN's output by a constant does.
    Differentiating (53) at :math:`c = 0` and integrating by parts, the mean
    force on the residual when the density is well fitted
    (:math:`p_{data} = p_{model}`) is

    .. math:: -\frac{\partial}{\partial c}\mathcal{L}^{ebm}_{Data}
              = p_{k,\zeta_k}(\bar\rho_k) - p_{k,\zeta_k}(-\bar\rho_k),

    the difference of the learned density at the two ends of its support.  For a
    density concentrated well inside :math:`\mathcal{R}_k` both terms are
    negligible and the force vanishes, as it should.  But an EBM switched on
    cold is still nearly uniform, so both terms are of order
    :math:`1/(2\bar\rho_k)` and any asymmetry between them is a systematic push
    on the state estimate --- against which, for a single cell, the physics term
    supplies no restoring force at all: Proposition 1's unique-zero hypothesis
    fails cell by cell, since :math:`\hat x^k_k` is free to absorb any shift of
    :math:`\hat x^k_{k-1}`.  The estimate then slides until something stops it.

    Letting the density converge on frozen residuals first removes the transient
    while it is largest.  This is a departure from Algorithm 2 as written --- it
    is off by default for that reason --- but it is the same concern Remark 5
    raises one step earlier, that "the joint PINN--EBM optimization is sensitive
    to initialization because the EBM is fitted to residuals produced by an
    initially untrained PINN".
    """
    if iteration < 0:
        raise ValueError(f"iteration must be non-negative, got {iteration}")
    if iteration < n_ebm:
        return Phase.WARMUP
    if iteration < n_ebm + n_fit:
        return Phase.FIT
    if local_only or iteration < n_par:
        return Phase.LOCAL
    return Phase.GLOBAL


def apply_epibe_phase(
    bank: EnergyEstimatorBank, target_cell: int, phase: Phase
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    r"""Set every freeze flag for a phase; return the ``(PINN, EBM)`` blocks.

    The two blocks are returned separately rather than concatenated because
    Algorithm 2 trains them jointly but not identically: they take different
    learning rates, and only the EBM half is projected onto
    :math:`\mathcal{K}_k` after a step.

    Returns
    -------
    tuple[list[nn.Parameter], list[nn.Parameter]]
        The active :math:`\Theta_{PINN}` and :math:`\Theta_{EBM}` parameters.
        The EBM list is empty during the warm-up.
    """
    if phase is Phase.FIT and target_cell not in bank.energy_cells:
        # Nothing to fit here: this cell keeps the quadratic consistency term,
        # so the EBM-only stretch would leave no trainable parameters at all.
        # Spend the iterations on the PINN instead.
        phase = Phase.LOCAL
    bank.use_energy = phase.uses_energy

    if phase is Phase.GLOBAL:
        # Lines 19-20: the whole prefix of the chain, PINNs and EBMs alike.
        bank.set_trainable_upto(target_cell, True)
        bank.set_ebm_trainable_upto(target_cell, False)
        pinn = list(bank.parameters_upto(target_cell))
        ebm = []
        for k in range(2, target_cell + 1):
            if k in bank.energy_cells:
                bank.set_ebm_trainable(k, True)
                ebm.extend(bank.ebm_parameters(k))
    else:
        # Lines 6-7 and 12-14: upstream frozen, current cell active.
        bank.set_trainable_upto(target_cell - 1, False)
        bank.set_ebm_trainable_upto(target_cell, False)
        bank.cell(target_cell).set_trainable(phase.trains_pinn)
        pinn = list(bank.cell_parameters(target_cell)) if phase.trains_pinn else []
        if phase is Phase.WARMUP or target_cell not in bank.energy_cells:
            # Line 6: "Freeze all EBM parameters and all upstream cells" --- and
            # a cell excluded from ``energy_cells`` has no EBM to train at all,
            # so it stays on the quadratic objective throughout.
            ebm = []
        else:
            bank.set_ebm_trainable(target_cell, True)
            ebm = list(bank.ebm_parameters(target_cell))

    # Cells the outer loop has not reached take no part in any objective.
    for k in range(target_cell + 1, bank.final_index + 1):
        bank.cell(k).set_trainable(False)
        bank.set_ebm_trainable(k, False)

    return pinn, ebm
