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
from pibe.eval.observability import basis_observability  # noqa: E402
from pibe.eval.oracle import compare_to_oracle  # noqa: E402
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
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="ignore any existing state.pt and start over (default is to resume)",
    )
    parser.add_argument(
        "--threads", type=int, default=None,
        help="torch intra-op threads; on a cluster set this to --cpus-per-task",
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="skip the figures (they are written to <output-dir>/figures by default)",
    )
    parser.add_argument(
        "--plot-trajectory", type=int, default=0,
        help="which validation trajectory the figures show",
    )
    args = parser.parse_args()

    import torch

    if args.threads:
        torch.set_num_threads(args.threads)

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
    # Reported before training: whether each disturbance coefficient is even
    # visible at the output is a property of the plant and sigma, not of the
    # estimator, and it bounds what any run can achieve.
    if config.data.noise_sigma > 0:
        logger.info(
            "%s",
            basis_observability(
                system=experiment.system,
                basis=experiment.basis,
                coefficient_bounds=experiment.coefficient_bounds,
                noise_sigma=config.data.noise_sigma,
                n_samples=config.data.n_samples,
            ).summary(),
        )
    config.to_yaml(output_dir / "config.resolved.yaml")

    if args.dry_run:
        logger.info("dry run requested; stopping before training")
        return 0

    # Resume checkpoint lives beside the artifacts; its presence is what makes
    # a preempted cluster job continue instead of restarting.
    state_path = output_dir / "state.pt"
    # The amortized-location stages after Algorithm 2 keep their own resume
    # files (see EPIBETrainer.train); they follow state.pt's lifecycle.
    stage_files = [output_dir / name for name in ("stages.json", "stage_bank.pt", "stage_state.pt")]
    if args.no_resume:
        for path in [state_path, *stage_files]:
            if path.exists():
                path.unlink()
                logger.info("--no-resume: discarded %s", path)

    trainer = experiment.make_trainer()
    trainer.checkpoint_path = state_path
    if config.training.checkpoint_every <= 0:
        logger.warning(
            "training.checkpoint_every is 0, so this run cannot be resumed; "
            "set it to a few thousand for a cluster job"
        )
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

    # Score the exact solution on the same objective, so a poor result is
    # attributed to optimization or to the loss rather than left ambiguous.
    # The oracle scores the exact solution with the quadratic data term.  An
    # EPIBE bank's own data term is a likelihood under a density it also
    # learned, so the two sides would not be on the same objective; score the
    # trained bank quadratically too, and say so.
    was_energy = getattr(experiment.bank, "use_energy", False)
    if was_energy:
        experiment.bank.use_energy = False
    try:
        comparison = compare_to_oracle(
            bank=experiment.bank,
            disturbance=experiment.disturbance,
            data=experiment.train_data,
            t_coll=experiment.t_coll,
            lam=config.training.lam,
            noise_floor=getattr(experiment.noise, "variance", None),
        )
    finally:
        if was_energy:
            experiment.bank.use_energy = True
    logger.info(
        "oracle comparison (train%s):\n%s",
        ", scored on the quadratic objective" if was_energy else "",
        comparison.summary(),
    )

    # Eq. (54) and the per-cell residual densities, for an EPIBE run.
    if hasattr(experiment.bank, "density_moments"):
        densities = {
            str(k): {
                "mean": m.mean,
                "std": m.std,
                "log_partition": m.log_partition,
                "normalization_error": experiment.bank.ebm(k).normalization_error(),
                "radius": experiment.bank.radii[k],
            }
            for k, m in experiment.bank.density_moments().items()
        }
        densities["mu_omega_hat"] = experiment.bank.noise_mean()
        densities["true_noise_bias"] = getattr(experiment.noise, "bias", 0.0)
        with open(output_dir / "densities.json", "w") as handle:
            json.dump(densities, handle, indent=2)
        logger.info(
            "learned residual densities:\n%s\n  mu_omega_hat = %+.6e "
            "(Eq. 54)   true sensor bias = %+.6e",
            experiment.bank.support_report(),
            experiment.bank.noise_mean(),
            getattr(experiment.noise, "bias", 0.0),
        )

    # Figures last, and never fatal: a missing matplotlib or a display quirk
    # must not discard a multi-hour training run that has already succeeded.
    if not args.no_plots:
        try:
            from pibe.eval.figures import write_figures

            written = write_figures(
                experiment, output_dir / "figures", trajectory=args.plot_trajectory
            )
            logger.info("figures: %s", ", ".join(p.name for p in written))
        except Exception as exc:  # noqa: BLE001 - deliberately non-fatal
            logger.warning(
                "figures skipped (%s: %s); training results are unaffected and "
                "can be plotted later with scripts/plot_results.py --run %s",
                type(exc).__name__, exc, output_dir,
            )

    logger.info("artifacts written to %s", output_dir)
    # Training finished, so the resume checkpoint is no longer needed; leaving
    # it would make a re-submitted job exit immediately instead of retraining.
    if state_path.exists():
        state_path.unlink()
    # Keep the offset search's record, under a name a resubmitted job ignores.
    if stage_files[0].exists():
        stage_files[0].replace(output_dir / "offset_search.json")
    for path in stage_files[1:]:
        if path.exists():
            path.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
