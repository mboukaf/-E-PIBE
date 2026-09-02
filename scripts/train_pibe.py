#!/usr/bin/env python3
"""Train a PIBE bank (Algorithm 1) from a YAML configuration.

Usage::

    python scripts/train_pibe.py --config configs/train_pibe.yaml
    python scripts/train_pibe.py --config configs/train_pibe.yaml \
        --set training.n_total=8000 training.n_par=6000 --output-dir outputs/long

Writes the resolved config, the training history, a checkpoint and the final
metrics into the output directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:  # allow running without installing
    sys.path.insert(0, str(REPO_ROOT / "src"))

from pibe.config import RunConfig  # noqa: E402
from pibe.eval.metrics import compute_metrics  # noqa: E402
from pibe.experiment import build_experiment  # noqa: E402
from pibe.training.callbacks import save_checkpoint  # noqa: E402
from pibe.utils.logging import get_logger, setup_logging  # noqa: E402

logger = get_logger("train_pibe")


def parse_override(text: str) -> tuple[list[str], object]:
    """Parse ``section.key=value`` into a path and a JSON-decoded value."""
    if "=" not in text:
        raise argparse.ArgumentTypeError(
            f"override {text!r} must have the form section.key=value"
        )
    path, _, raw = text.partition("=")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = raw  # treat as a bare string
    return path.split("."), value


def apply_overrides(payload: dict, overrides: list[str]) -> dict:
    """Apply ``--set`` overrides to a nested config dictionary."""
    for override in overrides:
        keys, value = parse_override(override)
        cursor = payload
        for key in keys[:-1]:
            cursor = cursor.setdefault(key, {})
            if not isinstance(cursor, dict):
                raise ValueError(f"cannot descend into {key!r} while applying {override!r}")
        cursor[keys[-1]] = value
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="YAML config file")
    parser.add_argument(
        "--set",
        dest="overrides",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="override config entries, e.g. training.lam=0.5",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="build the experiment and print its description, then stop",
    )
    args = parser.parse_args()

    import yaml

    with open(args.config) as handle:
        payload = yaml.safe_load(handle) or {}
    payload = apply_overrides(payload, args.overrides)
    config = RunConfig.from_dict(payload)

    output_dir = args.output_dir or Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(args.log_level, logfile=output_dir / "train.log")

    experiment = build_experiment(config)
    logger.info("experiment:\n%s", experiment.describe())
    config.to_yaml(output_dir / "config.resolved.yaml")

    if args.dry_run:
        logger.info("dry run requested; stopping before training")
        return 0

    trainer = experiment.make_trainer()
    history = trainer.train()
    history.save(output_dir / "history.json")
    save_checkpoint(
        output_dir / "bank.pt",
        experiment.bank,
        metadata={"config": config.to_dict()},
    )

    for label, data in (("train", experiment.train_data), ("val", experiment.val_data)):
        estimates = experiment.bank.estimate(data.y, data.t, experiment.t_coll)
        metrics = compute_metrics(estimates, data)
        logger.info("%s metrics:\n%s", label, metrics.summary())
        with open(output_dir / f"metrics_{label}.json", "w") as handle:
            json.dump(metrics.as_dict(), handle, indent=2)

    logger.info("artifacts written to %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
