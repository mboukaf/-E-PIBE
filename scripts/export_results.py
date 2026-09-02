#!/usr/bin/env python3
"""Export a trained bank's estimates alongside ground truth, as JSON.

Feeds the plotting layer.  Everything needed to draw estimate-vs-truth is
written out: the state trajectories, the disturbance, the parameters, and the
oracle comparison that says whether a poor fit is an optimization or a
loss-design failure.

Usage::

    python scripts/export_results.py --run outputs/cmp_local --out results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402

from pibe.config import RunConfig  # noqa: E402
from pibe.eval.metrics import compute_metrics  # noqa: E402
from pibe.eval.oracle import compare_to_oracle  # noqa: E402
from pibe.experiment import build_experiment  # noqa: E402
from pibe.training.callbacks import load_checkpoint  # noqa: E402
from pibe.utils.logging import get_logger, setup_logging  # noqa: E402

logger = get_logger("export_results")


def thin(values: list, stride: int) -> list:
    """Keep every ``stride``-th sample, always including the last."""
    if stride <= 1:
        return values
    kept = values[::stride]
    if kept[-1] != values[-1]:
        kept.append(values[-1])
    return kept


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="a training output directory")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--label", default=None, help="name for this run in the report")
    parser.add_argument("--trajectories", type=int, default=3, help="how many to export")
    parser.add_argument("--stride", type=int, default=1, help="thin the time grid")
    args = parser.parse_args()

    setup_logging("INFO")
    config = RunConfig.from_yaml(args.run / "config.resolved.yaml")
    experiment = build_experiment(config)
    load_checkpoint(args.run / "bank.pt", experiment.bank)
    experiment.bank.to(device=experiment.device, dtype=experiment.dtype)

    data = experiment.val_data
    estimates = experiment.bank.estimate(data.y, data.t, experiment.t_coll)
    metrics = compute_metrics(estimates, data)
    comparison = compare_to_oracle(
        bank=experiment.bank,
        disturbance=experiment.disturbance,
        data=data,
        t_coll=experiment.t_coll,
        lam=config.training.lam,
        noise_floor=getattr(experiment.noise, "variance", None),
    )

    count = min(args.trajectories, data.n_trajectories)
    t = thin(data.t.tolist(), args.stride)
    payload = {
        "label": args.label or args.run.name,
        "run_dir": str(args.run),
        "schedule": (
            "local-only"
            if config.training.local_only
            else f"global (N_par={config.training.n_par}/{config.training.n_total})"
        ),
        "system": experiment.system.name,
        "n_states": experiment.system.n,
        "horizon": config.data.horizon,
        "noise_sigma": config.data.noise_sigma,
        "noise_floor": getattr(experiment.noise, "variance", None),
        "eps_dq": experiment.disturbance.remainder_bound(data.t),
        "t": t,
        "trajectories": [
            {
                "y": thin(data.y[i].tolist(), args.stride),
                "true": [thin(data.x[i, :, j].tolist(), args.stride) for j in range(experiment.system.n)],
                "est": [thin(estimates.x[i, :, j].tolist(), args.stride) for j in range(experiment.system.n)],
            }
            for i in range(count)
        ],
        "disturbance": {
            "true": thin(data.d.tolist(), args.stride),
            "est": thin(estimates.d.mean(0).tolist(), args.stride),
            "est_spread": thin(estimates.d.std(0).tolist(), args.stride),
        },
        "theta": {
            "true": data.theta.tolist(),
            "est": estimates.theta.mean(0).tolist(),
            "spread": estimates.theta.std(0).tolist(),
            "bounds": experiment.system.theta_bounds.tolist(),
        },
        "coefficients": {
            "true": experiment.disturbance.coefficients.tolist(),
            "est": estimates.a.mean(0).tolist(),
            "bounds": experiment.coefficient_bounds.tolist(),
        },
        "metrics": metrics.as_dict(),
        "oracle": {
            "total_truth": comparison.oracle_total,
            "total_trained": comparison.trained_total,
            "truth_scores_better": comparison.truth_scores_better,
            "per_cell": {
                str(k): {
                    "truth": float(comparison.oracle[k].local),
                    "trained": float(comparison.trained[k].local),
                }
                for k in sorted(comparison.oracle)
            },
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(payload, handle)
    logger.info("wrote %s", args.out)
    logger.info("metrics:\n%s", metrics.summary())
    logger.info("oracle:\n%s", comparison.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
