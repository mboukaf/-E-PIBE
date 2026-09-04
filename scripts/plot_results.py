#!/usr/bin/env python3
"""Plot a trained bank's estimates against ground truth.

Writes PNG figures next to the run directory:

``states.png``
    True vs estimated trajectory for every coordinate, with the noisy
    measurement overlaid on x1.  Only x1 is measured; x2..xn are reconstructed.
``disturbance.png``
    d_hat(t) = Gamma_q(t)^T a_hat against the true d(t).
``parameters.png``
    theta and the basis coefficients, true vs estimated, against the admissible
    boxes the heads are constrained to.
``errors.png``
    Per-coordinate error growth along the chain, on a log scale.

Usage::

    python scripts/plot_results.py --run outputs/t5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from pibe.config import RunConfig  # noqa: E402
from pibe.eval.figures import write_figures  # noqa: E402
from pibe.eval.metrics import compute_metrics  # noqa: E402
from pibe.eval.oracle import compare_to_oracle  # noqa: E402
from pibe.experiment import build_experiment  # noqa: E402
from pibe.training.callbacks import load_checkpoint  # noqa: E402
from pibe.utils.logging import get_logger, setup_logging  # noqa: E402

logger = get_logger("plot_results")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--trajectory", type=int, default=0)
    parser.add_argument("--split", choices=["val", "train"], default="val")
    args = parser.parse_args()

    setup_logging("INFO")
    outdir = args.outdir or args.run / "figures"
    outdir.mkdir(parents=True, exist_ok=True)

    config = RunConfig.from_yaml(args.run / "config.resolved.yaml")
    experiment = build_experiment(config)
    load_checkpoint(args.run / "bank.pt", experiment.bank)
    experiment.bank.to(device=experiment.device, dtype=experiment.dtype)

    data = experiment.val_data if args.split == "val" else experiment.train_data
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

    write_figures(experiment, outdir, trajectory=args.trajectory)

    print()
    print(f"=== {args.run.name}  ({args.split}, P={len(data)}, T={config.data.horizon}) ===")
    print(metrics.summary())
    print()
    print(comparison.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
