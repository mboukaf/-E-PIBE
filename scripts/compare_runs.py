#!/usr/bin/env python3
r"""Compare several trained runs: per-seed metrics, and per-coefficient recovery.

Written for the cluster grid, which varies two things --- the basis frequency
:math:`\Omega` and the seed --- and asks one question the aggregate disturbance
error cannot answer.

``eval/observability.py`` predicts, before any training, that the third Fourier
coefficient of the disturbance leaves a trace at the output well below the
others: at :math:`\Omega = 2\pi/5` its SNR is 2.05 against 12.56 for
:math:`a_1, a_2`, and halving the frequency raises it roughly fourfold at no
cost to :math:`W_\Gamma`.  Whether that prediction holds is visible only per
coefficient: :math:`\|\hat d - d\|` mixes all three, and a run that recovers
:math:`a_1` and :math:`a_2` perfectly while learning nothing about :math:`a_3`
can still post a respectable disturbance error.

So the table below reports, for every run, the correlation between
:math:`\hat a_j` and the true :math:`a_j` across trajectories.  For a bank that
genuinely *infers* the disturbance from :math:`y` this is near 1; for one that
has learned to emit a constant it is near 0, whatever the aggregate error says.
That distinction has already caught one false positive in this project.

Usage::

    python scripts/compare_runs.py --runs outputs/big_w5_s*  outputs/big_w10_s*
    python scripts/compare_runs.py --runs outputs/big_* --group-by omega
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402

from pibe.config import RunConfig  # noqa: E402
from pibe.eval.metrics import compute_metrics  # noqa: E402
from pibe.experiment import build_experiment  # noqa: E402
from pibe.training.callbacks import (  # noqa: E402
    load_checkpoint,
    resolve_checkpoint,
)
from pibe.utils.logging import setup_logging  # noqa: E402


def coefficient_recovery(estimates, data) -> tuple[list[float], list[float]]:
    r"""Per-coefficient ``corr(a_hat_j, a_true_j)`` and the spread of ``a_hat_j``.

    The spread is the tell-tale: a head that has collapsed to a constant has a
    near-zero spread *and* a near-zero correlation, while the aggregate
    disturbance error may still look fine because the constant is close to the
    mean of the true coefficients.
    """
    if data.coefficients is None:
        return [], []
    correlations, spreads = [], []
    for j in range(estimates.a.shape[-1]):
        est, truth = estimates.a[:, j], data.coefficients[:, j]
        if est.numel() < 2 or float(est.std()) == 0.0 or float(truth.std()) == 0.0:
            correlations.append(float("nan"))
        else:
            correlations.append(float(torch.corrcoef(torch.stack([est, truth]))[0, 1]))
        spreads.append(float(est.std()))
    return correlations, spreads


def describe(run: Path, recompute: bool) -> dict | None:
    """Metrics for one run, from its JSON or recomputed from the checkpoint."""
    config_path = run / "config.resolved.yaml"
    if not config_path.exists():
        print(f"  {run}: no config.resolved.yaml -- skipped", file=sys.stderr)
        return None
    config = RunConfig.from_yaml(config_path)
    row: dict = {
        "run": run.name,
        "omega": config.basis.omega,
        "seed": config.seed,
        "trajectories": config.data.n_trajectories,
        "per_trajectory_a": config.data.sample_disturbance_per_trajectory,
        "finished": (run / "metrics_val.json").exists(),
        "resumable_state_left": (run / "state.pt").exists(),
    }
    metrics_path = run / "metrics_val.json"
    if metrics_path.exists() and not recompute:
        row.update(json.loads(metrics_path.read_text()))

    has_weights = (run / "bank.pt").exists() or (run / "state.pt").exists()
    if has_weights and (recompute or config.data.sample_disturbance_per_trajectory):
        experiment = build_experiment(config)
        load_checkpoint(resolve_checkpoint(run), experiment.bank)
        experiment.bank.to(device=experiment.device, dtype=experiment.dtype)
        experiment.bank.eval()
        data = experiment.val_data
        with torch.no_grad():
            estimates = experiment.bank.estimate(data.y, data.t, experiment.t_coll)
        if recompute or not metrics_path.exists():
            row.update(compute_metrics(estimates, data).as_dict())
        row["corr_a"], row["spread_a"] = coefficient_recovery(estimates, data)
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--recompute", action="store_true",
                        help="recompute metrics from bank.pt instead of reading the JSON")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    setup_logging("WARNING")

    rows = [r for r in (describe(p, args.recompute) for p in sorted(args.runs)) if r]
    if not rows:
        print("no runs found", file=sys.stderr)
        return 1

    # --- did every job actually finish? -------------------------------
    unfinished = [r["run"] for r in rows if not r["finished"]]
    requeued = [r["run"] for r in rows if r["resumable_state_left"]]
    print(f"{len(rows)} run(s); {len(rows) - len(unfinished)} finished")
    if unfinished:
        print(f"  UNFINISHED (no metrics_val.json): {', '.join(unfinished)}")
    if requeued:
        print(f"  state.pt still present, so these stopped early and can be "
              f"resumed by re-submitting: {', '.join(requeued)}")
    print()

    # --- per-run accuracy ---------------------------------------------
    n = len(rows[0].get("state_l2", []))
    head = f"{'run':>18}{'omega':>9}{'seed':>5} | "
    head += "".join(f"{'x' + str(j + 1):>10}" for j in range(n))
    head += "".join(f"{'|dth' + str(j + 1) + '|':>10}" for j in range(n - 1))
    head += f"{'d L2':>10}"
    print(head)
    print("-" * len(head))
    for r in rows:
        if "state_l2" not in r:
            print(f"{r['run']:>18}{r['omega']:>9.4f}{r['seed']:>5} |  (no metrics)")
            continue
        line = f"{r['run']:>18}{r['omega']:>9.4f}{r['seed']:>5} | "
        line += "".join(f"{v:>10.3e}" for v in r["state_l2"])
        line += "".join(f"{v:>10.3e}" for v in r["theta_abs_error"])
        line += f"{r['disturbance_l2']:>10.3e}"
        print(line)

    # --- the question the grid was built to answer ---------------------
    if any("corr_a" in r for r in rows):
        print()
        print("per-coefficient disturbance recovery, corr(a_hat, a_true) across "
              "trajectories")
        print("(near 1 = inferred from y; near 0 = the head is emitting a constant)")
        q = len(next(r["corr_a"] for r in rows if "corr_a" in r))
        head = f"{'run':>18}{'omega':>9} | " + "".join(f"{'a' + str(j + 1):>9}" for j in range(q))
        head += "  | " + "".join(f"{'sd(a' + str(j + 1) + ')':>10}" for j in range(q))
        print(head)
        print("-" * len(head))
        for r in rows:
            if "corr_a" not in r:
                continue
            line = f"{r['run']:>18}{r['omega']:>9.4f} | "
            line += "".join(f"{v:>9.3f}" for v in r["corr_a"])
            line += "  | " + "".join(f"{v:>10.4f}" for v in r["spread_a"])
            print(line)

    # --- averaged over seeds, which is what a multi-seed grid is for ----
    groups: dict[float, list[dict]] = {}
    for r in rows:
        if "state_l2" in r:
            groups.setdefault(round(float(r["omega"]), 6), []).append(r)
    if len(groups) > 1:
        print()
        print("averaged over seeds")
        head = f"{'omega':>9}{'seeds':>7} | " + "".join(f"{'x' + str(j + 1):>10}" for j in range(n))
        head += f"{'d L2':>10}"
        print(head)
        print("-" * len(head))
        for omega in sorted(groups):
            members = groups[omega]
            mean = [sum(m["state_l2"][j] for m in members) / len(members) for j in range(n)]
            d = sum(m["disturbance_l2"] for m in members) / len(members)
            print(f"{omega:>9.4f}{len(members):>7} | "
                  + "".join(f"{v:>10.3e}" for v in mean) + f"{d:>10.3e}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
