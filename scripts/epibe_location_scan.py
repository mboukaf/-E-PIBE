#!/usr/bin/env python3
r"""Resolve EPIBE's location degeneracy by searching the one direction it lives in.

The diagnosis (``pibe.eval.energy_oracle``) is that the energy objective *does*
prefer the true sensor offset --- the exact solution scores below anything
training reached --- but that gradient descent cannot travel there.  The reason
is structural rather than incidental.  Write :math:`\varepsilon = y - \hat x^2_1`
and consider translating it by :math:`c`.  Differentiating Eq. (53),

.. math:: -\frac{\partial}{\partial c}\mathcal{L}^{ebm,2}_{Data}
          = p_{2,\zeta}(\bar\rho_2) - p_{2,\zeta}(-\bar\rho_2) \approx 0

for any density concentrated inside its support.  So once the EBM has settled on
the residuals the PINN currently produces, the data term is *stationary* along
the degenerate direction: the pair (state, density) can slide together at no
cost, and no first-order method will make them.  Only the physics term breaks the
tie, and its gradient there is of order 1e-4.

The saving grace is that the degeneracy is exactly **one scalar**.  So rather
than hoping an optimizer stumbles across it, this script searches it directly:
for each candidate offset ``c`` the first cell's density is initialized centred
at ``c``, which puts the residuals off-centre and creates a real force pulling
:math:`\hat x^2_1` to :math:`y - c`; training then proceeds normally and the
*objective decides* which ``c`` survives.  Nothing about the model or the loss
changes --- this is global optimization of a non-convex objective along a known
degenerate coordinate, the same tactic the parameter-identifiability profiles
used earlier in this project.

The starting point is a converged quadratic fit, which is what Remark 5's
warm-up produces; a trained PIBE run on the same data is exactly that, so it is
loaded rather than recomputed.

Usage::

    python scripts/epibe_location_scan.py \
        --config configs/automatica_n3v2_epibe_cell2.yaml \
        --warm-start outputs/n3v2_biased_pibe/bank.pt \
        --offsets -0.05 0.0 0.05 0.10 --iterations 4000
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402

from pibe.config import RunConfig  # noqa: E402
from pibe.core.bank import Mode  # noqa: E402
from pibe.core.losses import total_loss  # noqa: E402
from pibe.eval.metrics import compute_metrics  # noqa: E402
from pibe.experiment import build_experiment  # noqa: E402
from pibe.training.callbacks import load_checkpoint, save_checkpoint  # noqa: E402
from pibe.training.epibe_phases import Phase, apply_epibe_phase  # noqa: E402
from pibe.utils.logging import get_logger, setup_logging  # noqa: E402
from pibe.utils.seeding import make_generator  # noqa: E402

logger = get_logger("location_scan")


def seed_density_at(bank, cell: int, residual: torch.Tensor, offset: float,
                    iterations: int = 1500, lr: float = 2e-3) -> None:
    r"""Fit cell ``cell``'s density to ``residual + offset``.

    This is the whole intervention.  Placing the density's mode at ``offset``
    while the residuals still sit near zero makes them off-centre under it, and
    the resulting force drags :math:`\hat x^2_1` towards :math:`y - c`.  It
    biases only the *initialization* of a search; whether the offset survives is
    decided afterwards by the objective.
    """
    model = bank.ebm(cell)
    target = (residual.reshape(-1) + offset).clamp(-model.radius, model.radius)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        model.negative_log_likelihood(target).backward()
        optimizer.step()
        model.project(10.0)


@torch.no_grad()
def residuals_of(bank, data, t_coll, cell: int = 2) -> torch.Tensor:
    with torch.enable_grad():
        outputs = bank(target_cell=bank.final_index, y=data.y, t_data=data.t,
                       t_coll=t_coll, mode=Mode.GLOBAL, create_graph=False)
    return bank.consistency_residual(cell, data.y, outputs).detach()


def objective_on(bank, data, t_coll, lam: float) -> float:
    """:math:`\\mathcal{L}^{n+1}_{Tot,E}` on a whole dataset."""
    was_training = bank.training
    bank.eval()
    try:
        with torch.enable_grad():
            outputs = bank(target_cell=bank.final_index, y=data.y, t_data=data.t,
                           t_coll=t_coll, mode=Mode.GLOBAL, create_graph=False)
            losses = {
                k: bank.cell_loss(k, data.y, t_coll, outputs, lam).local
                for k in bank.cell_indices
            }
            value = float(total_loss(losses, bank.final_index))
    finally:
        bank.train(was_training)
    return value


def refine(experiment, iterations: int, lr_pinn: float, lr_ebm: float,
           batch_size: int, seed: int, use_energy: bool = True,
           train_ebm: bool = True) -> None:
    """Joint end-to-end PINN+EBM training, the regime of Algorithm 2 line 18.

    With ``use_energy=False`` the same end-to-end regime runs on the *centered*
    quadratic data term instead, which is the location-free warm-up described in
    :func:`centered_start`.
    """
    bank = experiment.bank
    config = experiment.config
    pinn, ebm = apply_epibe_phase(bank, bank.final_index, Phase.GLOBAL)
    bank.use_energy = use_energy
    if not use_energy or not train_ebm:
        # A frozen density keeps exerting its force on the PINN instead of
        # re-centring onto whatever residuals it currently sees, which is what
        # lets a seeded location actually be reached before it is renegotiated.
        ebm = []
    groups = [{"params": pinn, "lr": lr_pinn}]
    if ebm:
        groups.append({"params": ebm, "lr": lr_ebm})
    optimizer = torch.optim.Adam(groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)
    data = experiment.train_data
    generator = make_generator(seed)
    count = data.n_trajectories

    for step in range(iterations):
        index = torch.randperm(count, generator=generator)[:batch_size]
        y = data.y[index]
        outputs = bank(target_cell=bank.final_index, y=y, t_data=data.t,
                       t_coll=experiment.t_coll, mode=Mode.GLOBAL)
        losses = {
            k: bank.cell_loss(k, y, experiment.t_coll, outputs, config.training.lam).local
            for k in bank.cell_indices
        }
        objective = total_loss(losses, bank.final_index)
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        torch.nn.utils.clip_grad_norm_(
            pinn + ebm, config.training.grad_clip or 1.0
        )
        optimizer.step()
        scheduler.step()
        if ebm:
            bank.project(config.ebm.weight_bound)
        if step % max(1, iterations // 4) == 0:
            logger.info("    step %5d | L_Tot,E = %+.6e", step, float(objective))


def centered_start(experiment, iterations: int, batch_size: int, seed: int) -> None:
    r"""The single-run alternative to the scan: let the physics pick the location.

    Proposition 1 decomposes the quadratic data risk as

    .. math:: R_D(\hat y, m) = \sigma_\omega^2
              + \frac{1}{T}\int_0^T (\hat y - y + m - \mu_\omega)^2,

    and minimizing it over the unknown location :math:`m` leaves precisely the
    *variance* of the residual.  Training on that centered term therefore fits
    the shape of :math:`\hat x^2_1` while saying nothing about where it sits ---
    the quadratic term with the location eliminated rather than assumed --- so
    the location is settled by :math:`\lambda R_P` alone, which is the division
    of labour Proposition 1 actually prescribes.  Only afterwards is the density
    fitted to the residuals that result, and it inherits whatever mean the
    physics chose.

    The contrast with the scan is the point: the scan searches the degenerate
    coordinate and lets the objective pick, while this lets the physics do it in
    one pass.  Both start from the same converged quadratic fit.
    """
    experiment.config.ebm.centered_warmup = True
    refine(
        experiment, iterations,
        lr_pinn=experiment.config.training.lr_global, lr_ebm=0.0,
        batch_size=batch_size, seed=seed, use_energy=False,
    )
    experiment.config.ebm.centered_warmup = False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--warm-start", type=Path, required=True,
                        help="a converged quadratic fit on the same data")
    parser.add_argument("--offsets", type=float, nargs="+",
                        default=[-0.05, 0.0, 0.025, 0.05, 0.075, 0.10, 0.15])
    parser.add_argument("--iterations", type=int, default=4000)
    parser.add_argument(
        "--hold", type=int, default=0,
        help="iterations with the density frozen right after seeding, so the "
             "PINN is pulled all the way to y - c before the density adapts",
    )
    parser.add_argument(
        "--energy-bound", type=float, default=None,
        help="override B_E. Eq. (50) makes the sharpest representable density "
             "e^{2 beta B_E}/(2 rho); if that is sharper than the true noise "
             "law, the objective's infimum is 'fit y exactly' rather than the "
             "truth, and any correct solution erodes back towards it.",
    )
    parser.add_argument(
        "--centered-iters", type=int, default=0,
        help="run the location-free centered warm-up for this many iterations "
             "instead of scanning offsets; see centered_start()",
    )
    parser.add_argument("--out", type=Path, default=Path("outputs/epibe_location_scan"))
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    setup_logging(args.log_level, logfile=args.out / "scan.log")

    config = RunConfig.from_yaml(args.config)
    if args.energy_bound is not None:
        config.ebm.energy_bound = args.energy_bound
        logger.info("B_E overridden to %.3f", args.energy_bound)
    true_bias = config.data.noise_bias
    logger.info("scanning offsets %s against a true sensor bias of %+.4f",
                args.offsets, true_bias)

    rows = []
    best = None
    for offset in args.offsets:
        logger.info("offset c = %+.4f", offset)
        experiment = build_experiment(config)
        # Load the converged quadratic fit into the PINN half; the EBMs are new.
        payload = torch.load(args.warm_start, map_location="cpu", weights_only=False)
        missing, unexpected = experiment.bank.load_state_dict(
            payload["state_dict"], strict=False
        )
        assert not unexpected, f"unexpected keys in warm start: {unexpected[:3]}"
        experiment.bank.to(device=experiment.device, dtype=experiment.dtype)
        experiment.bank.use_energy = True

        if args.centered_iters > 0:
            logger.info("    centered warm-up for %d iterations "
                        "(location left to the physics)", args.centered_iters)
            centered_start(experiment, args.centered_iters,
                           config.training.batch_size, config.training.seed)
            residual = residuals_of(experiment.bank, experiment.train_data,
                                    experiment.t_coll)
            logger.info("    physics chose a residual mean of %+.5f", float(residual.mean()))
            seed_density_at(experiment.bank, 2, residual, 0.0)
        else:
            residual = residuals_of(experiment.bank, experiment.train_data,
                                    experiment.t_coll)
            seed_density_at(experiment.bank, 2, residual, offset)
        logger.info("    density seeded at mu = %+.5f (residuals were %+.5f)",
                    experiment.bank.ebm(2).moments().mean, float(residual.mean()))

        if args.hold > 0:
            logger.info("    holding the density for %d iterations", args.hold)
            refine(experiment, args.hold,
                   lr_pinn=config.training.lr_global, lr_ebm=0.0,
                   batch_size=config.training.batch_size,
                   seed=config.training.seed, train_ebm=False)
            logger.info("    after the hold: residual mean %+.5f",
                        float(residuals_of(experiment.bank, experiment.train_data,
                                           experiment.t_coll).mean()))
        refine(experiment, args.iterations,
               lr_pinn=config.training.lr_global, lr_ebm=config.ebm.lr_ebm or 2e-3,
               batch_size=config.training.batch_size, seed=config.training.seed)

        val = experiment.val_data
        score = objective_on(experiment.bank, experiment.train_data,
                             experiment.t_coll, config.training.lam)
        metrics = compute_metrics(
            experiment.bank.estimate(val.y, val.t, experiment.t_coll), val
        )
        with torch.no_grad():
            estimates = experiment.bank.estimate(val.y, val.t, experiment.t_coll)
            drift = float((estimates.x[..., 0] - val.x[..., 0]).mean())
        row = {
            "offset": offset,
            "objective": score,
            "mu_hat": experiment.bank.noise_mean(),
            "x1_minus_truth": drift,
            "state_l2": metrics.state_l2.tolist(),
            "theta_abs_error": metrics.theta_abs_error.tolist(),
            "disturbance_l2": metrics.disturbance_l2,
        }
        rows.append(row)
        logger.info("    L_Tot,E = %+.6e | mu_hat = %+.5f | x1-truth = %+.5f",
                    score, row["mu_hat"], drift)

        if best is None or score < best[0]:
            best = (score, offset)
            save_checkpoint(args.out / "bank.pt", experiment.bank,
                            metadata={"config": config.to_dict(), "offset": offset})
            (args.out / "config.resolved.yaml").write_text(
                Path(args.config).read_text()
            )

    (args.out / "scan.json").write_text(json.dumps(rows, indent=2))

    print()
    print(f"true sensor bias {true_bias:+.4f}\n")
    header = (f"{'offset':>8} | {'L_Tot,E':>13} | {'mu_hat':>9} | "
              f"{'x1-truth':>9} | {'x1 L2':>10}{'|dth1|':>10}{'|dth2|':>10}{'d L2':>10}")
    print(header)
    print("-" * len(header))
    for row in rows:
        mark = "  <-- best" if row["offset"] == best[1] else ""
        print(f"{row['offset']:>8.4f} | {row['objective']:>13.6e} | "
              f"{row['mu_hat']:>+9.5f} | {row['x1_minus_truth']:>+9.5f} | "
              f"{row['state_l2'][0]:>10.3e}{row['theta_abs_error'][0]:>10.3e}"
              f"{row['theta_abs_error'][1]:>10.3e}{row['disturbance_l2']:>10.3e}{mark}")
    print(f"\nobjective selects c = {best[1]:+.4f}; truth is {true_bias:+.4f}")
    print(f"artifacts in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
