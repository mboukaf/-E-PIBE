#!/usr/bin/env python3
"""Evaluate a trained PIBE checkpoint.

Rebuilds the experiment from the checkpoint's stored config, restores the bank
and reports estimation metrics on the training and validation splits.

Usage::

    python scripts/evaluate.py --checkpoint outputs/pibe_run/bank.pt
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
from pibe.experiment import build_experiment  # noqa: E402
from pibe.training.callbacks import load_checkpoint  # noqa: E402
from pibe.utils.logging import get_logger, setup_logging  # noqa: E402

logger = get_logger("evaluate")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="override the config stored in the checkpoint",
    )
    parser.add_argument("--output", type=Path, default=None, help="write metrics as JSON")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if args.config is not None:
        config = RunConfig.from_yaml(args.config)
    else:
        stored = payload.get("metadata", {}).get("config")
        if stored is None:
            raise SystemExit(
                "checkpoint carries no config; pass --config explicitly"
            )
        config = RunConfig.from_dict(stored)

    experiment = build_experiment(config)
    load_checkpoint(args.checkpoint, experiment.bank)
    experiment.bank.to(device=experiment.device, dtype=experiment.dtype)
    logger.info("experiment:\n%s", experiment.describe())

    results = {}
    for label, data in (("train", experiment.train_data), ("val", experiment.val_data)):
        estimates = experiment.bank.estimate(data.y, data.t, experiment.t_coll)
        metrics = compute_metrics(estimates, data)
        logger.info("%s metrics:\n%s", label, metrics.summary())
        results[label] = metrics.as_dict()

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as handle:
            json.dump(results, handle, indent=2)
        logger.info("metrics written to %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
