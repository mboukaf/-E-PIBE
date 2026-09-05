r"""Scoring the true solution on the *energy-based* objective, Section 3.2.

:mod:`pibe.eval.oracle` answers the same question for PIBE, and the reason for
asking it is unchanged: when a trained bank reports a small loss and poor
estimates, "the optimizer did not get there" and "the objective prefers the
wrong answer" call for opposite responses, and only evaluating the objective at
the exact solution tells them apart.

EPIBE needs its own version because its data term is not a fixed function.  The
quadratic term of PIBE can be evaluated anywhere; :math:`\mathcal{L}^{ebm,k}_{Data}`
is a likelihood under a density that is itself learned, so scoring the true
solution means fitting the density *to the true residuals* first.  That is the
honest comparison, and it is well posed: at the exact solution the first cell's
residual is exactly the measurement noise realization,

.. math:: \varepsilon_2 = y - x_1 = \omega,

whose law is the thing EPIBE claims to be able to model.  So the oracle here is
"the true states and parameters, together with the best density the admissible
class can put on the actual noise" --- precisely the pair
:math:`(y, \mu_\omega)` that Proposition 1 says the combined risk should select.

A note on the downstream cells
------------------------------
At the true solution the inter-cell consistency residuals of Eqs. (29)/(38) are
identically zero.  A density fitted to them is a delta, whose likelihood is
bounded only by Eq. (50) --- so under Section 3.2 Step 2 as written, the oracle's
own data term runs to :math:`-\beta B_{E,k}` and the comparison degenerates.
That is not an artefact of this diagnostic; it is the same defect that makes
those cells drift in training, seen from the objective's side.  When a cell
keeps PIBE's quadratic term (``ebm.cells``), its oracle data term is a clean
zero and the comparison is meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from pibe.core.bank import Mode
from pibe.core.energy_bank import EnergyEstimatorBank
from pibe.core.losses import global_weights
from pibe.data.dataset import TrajectoryData
from pibe.data.disturbance import BasisDisturbance
from pibe.data.simulate import rk4_integrate
from pibe.ebm.density import ScalarEBM
from pibe.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EnergyOracleComparison:
    """The energy objective at the truth and at the trained bank."""

    oracle: dict[int, float]
    trained: dict[int, float]
    oracle_total: float
    trained_total: float
    oracle_mu: float
    trained_mu: float
    true_bias: float

    @property
    def truth_scores_better(self) -> bool:
        """Whether the exact solution attains the lower objective."""
        return self.oracle_total < self.trained_total

    @property
    def verdict(self) -> str:
        if self.truth_scores_better:
            return (
                "optimization failure: the true state, the true parameters and a "
                "density fitted to the actual noise attain a LOWER energy "
                "objective than training reached. The objective does identify "
                "the sensor offset; the schedule failed to find it. The levers "
                "are the schedule and the initialization -- note that the "
                "warm-up uses the quadratic data term, which is itself the "
                "zero-mean assumption, so EPIBE starts from PIBE's answer."
            )
        return (
            "loss-design failure: the objective scores the trained solution at "
            "or below the exact one, so no schedule recovers the offset. The "
            "energy data term is location-indifferent by construction -- its "
            "gradient under a uniform shift of the residual is "
            "p(rho) - p(-rho), which vanishes for any density concentrated "
            "inside its support -- so identification must come from a physics "
            "term with a unique zero (Proposition 1), and here it does not."
        )

    def summary(self) -> str:
        lines = [
            f"{'cell':>5} | {'L_Loc,E @ truth':>17} | {'L_Loc,E @ trained':>18} | winner",
            "-" * 66,
        ]
        for k in sorted(self.oracle):
            at_truth, at_trained = self.oracle[k], self.trained[k]
            lines.append(
                f"{k:>5} | {at_truth:>17.6e} | {at_trained:>18.6e} | "
                f"{'truth' if at_truth < at_trained else 'trained'}"
            )
        lines += [
            "-" * 66,
            f"L_Tot,E @ truth   : {self.oracle_total:.6e}",
            f"L_Tot,E @ trained : {self.trained_total:.6e}",
            f"gap (trained - truth) : {self.trained_total - self.oracle_total:+.6e}",
            "",
            f"mu_omega  @ truth   : {self.oracle_mu:+.6f}  "
            f"(a density fitted to the actual noise)",
            f"mu_omega  @ trained : {self.trained_mu:+.6f}",
            f"true sensor bias    : {self.true_bias:+.6f}",
            "",
            f"verdict: {self.verdict}",
        ]
        return "\n".join(lines)


def fit_density_to(
    residual: Tensor, template: ScalarEBM, iterations: int = 3000, lr: float = 2e-3
) -> ScalarEBM:
    """The best density the admissible class can put on a fixed residual sample.

    Built with the same radius, temperature, architecture and quadrature as the
    cell it stands in for, so the oracle is scored in the *same* class the
    trained model was restricted to and no comparison is made across classes.
    """
    model = ScalarEBM(
        radius=template.radius,
        beta=template.beta,
        hidden=tuple(
            m.out_features
            for m in template.energy.mlp.net
            if hasattr(m, "out_features")
        )[:-1],
        activation=template.energy.mlp.activation_name,
        energy_scale=template.energy.energy_scale,
        energy_bound=template.energy.energy_bound,
        spectral_norm_layers=template.energy.spectral_norm_layers,
        panels=template.quadrature.panels,
        nodes_per_panel=template.quadrature.nodes_per_panel,
        barrier_scale=template.barrier_scale,
        dtype=residual.dtype,
    ).to(residual.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    flat = residual.reshape(-1).detach()
    for _ in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        model.negative_log_likelihood(flat).backward()
        optimizer.step()
        model.project(10.0)
    return model


@torch.no_grad()
def _truth_outputs(bank: EnergyEstimatorBank, data: TrajectoryData, t_coll: Tensor):
    """States, parameters and disturbance at the exact solution."""
    disturbance = BasisDisturbance(
        bank.basis, data.coefficients, remainder=None
    )
    x_coll = rk4_integrate(
        bank.system, t_grid=t_coll, x0=data.x[:, 0, :], theta=data.theta,
        disturbance=disturbance, substeps=16,
    )
    return x_coll


def compare_to_energy_oracle(
    bank: EnergyEstimatorBank,
    data: TrajectoryData,
    t_coll: Tensor,
    lam: float,
    true_bias: float = 0.0,
    fit_iterations: int = 3000,
) -> EnergyOracleComparison:
    r"""Score the exact solution and the trained bank on :math:`\mathcal{L}^{n+1}_{Tot,E}`.

    At the truth every physics residual vanishes identically, so each cell's
    local loss reduces to its data term: a likelihood for the cells carrying an
    EBM, and zero for the cells that kept the quadratic consistency term.
    """
    final = bank.final_index
    weights = global_weights(final)

    # --- the exact solution -------------------------------------------
    oracle: dict[int, float] = {}
    oracle_mu = float("nan")
    for k in bank.cell_indices:
        if k == 2:
            residual = data.y - data.x[..., 0]  # exactly omega, bias included
        else:
            residual = torch.zeros_like(data.y)  # consistency is exact at truth
        if k in bank.energy_cells:
            model = fit_density_to(residual, bank.ebm(k), iterations=fit_iterations)
            oracle[k] = float(model.negative_log_likelihood(residual))
            if k == 2:
                oracle_mu = model.moments().mean
        else:
            oracle[k] = float((residual**2).mean())
    oracle_total = sum(weights[k] * oracle[k] for k in oracle)

    # --- the trained bank, on the same data ----------------------------
    was_training = bank.training
    bank.eval()
    try:
        with torch.enable_grad():
            outputs = bank(
                target_cell=final, y=data.y, t_data=data.t, t_coll=t_coll,
                mode=Mode.GLOBAL, create_graph=False,
            )
            trained = {
                k: float(bank.cell_loss(k, data.y, t_coll, outputs, lam).local)
                for k in bank.cell_indices
            }
    finally:
        bank.train(was_training)
    trained_total = sum(weights[k] * trained[k] for k in trained)

    return EnergyOracleComparison(
        oracle=oracle,
        trained=trained,
        oracle_total=oracle_total,
        trained_total=trained_total,
        oracle_mu=oracle_mu,
        trained_mu=bank.noise_mean() if 2 in bank.energy_cells else float("nan"),
        true_bias=float(true_bias),
    )
