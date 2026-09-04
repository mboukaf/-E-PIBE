#!/usr/bin/env python3
r"""Plot a trained bank's estimates at one chosen measurement-noise level.

Companion to ``scripts/noise_sweep.py``, which reports that sweep as a table.
This draws the trajectories behind one row of it: states against truth with the
noisy output overlaid, the disturbance, and the parameter/coefficient recovery.

The model is used exactly as trained --- only the measurement it is fed changes.

Usage::

    python scripts/plot_at_sigma.py --run outputs/n3v2_s0 --sigma 0.1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from noise_sweep import clean_trajectories  # noqa: E402
from pibe.config import RunConfig  # noqa: E402
from pibe.data.noise import NOISE_FAMILIES, build_noise_model  # noqa: E402
from pibe.eval.figures import (  # noqa: E402
    plot_disturbance,
    plot_errors,
    plot_parameters,
    plot_states,
)
from pibe.eval.metrics import compute_metrics  # noqa: E402
from pibe.experiment import build_experiment  # noqa: E402
from pibe.training.callbacks import load_checkpoint  # noqa: E402
from pibe.utils.logging import setup_logging  # noqa: E402
from pibe.utils.seeding import make_generator  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--sigma", type=float, required=True)
    parser.add_argument("--trajectories", type=int, default=64)
    parser.add_argument("--index", type=int, default=0,
                        help="which trajectory to draw")
    parser.add_argument("--noise", type=str, default="gaussian",
                        choices=sorted(NOISE_FAMILIES))
    parser.add_argument("--bias", type=float, default=0.0,
                        help="constant sensor offset, in units of sigma")
    parser.add_argument("--seed", type=int, default=909)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    setup_logging("INFO")
    config = RunConfig.from_yaml(args.run / "config.resolved.yaml")
    experiment = build_experiment(config)
    load_checkpoint(args.run / "bank.pt", experiment.bank)
    experiment.bank.to(device=experiment.device, dtype=experiment.dtype)
    experiment.bank.eval()

    # Same unseen trajectories the sweep uses, so the figures and the table row
    # describe the same experiment.
    clean = clean_trajectories(experiment, args.trajectories, args.seed)
    noise = build_noise_model(args.noise, args.sigma, bias=args.bias)
    omega = noise.sample(
        tuple(clean.y.shape), generator=make_generator(args.seed),
        dtype=experiment.dtype,
    ).to(clean.y.device)
    data = type(clean)(
        t=clean.t, x=clean.x, y=clean.x[..., 0] + omega,
        d=clean.d, theta=clean.theta, coefficients=clean.coefficients,
    )

    with torch.no_grad():
        est = experiment.bank.estimate(data.y, data.t, experiment.t_coll)
    metrics = compute_metrics(est, data)

    tag = f"{args.noise}_sigma_{args.sigma:g}".replace(".", "p")
    if args.bias:
        tag += f"_bias{args.bias:g}".replace(".", "p")
    outdir = args.out or (args.run / "figures" / tag)
    outdir.mkdir(parents=True, exist_ok=True)
    t = data.t.cpu().numpy()
    index = min(args.index, len(data) - 1)

    plot_states(t, data.x.cpu().numpy(), est.x.cpu().numpy(),
                data.y.cpu().numpy(), outdir / "states.png", index=index)
    plot_disturbance(
        t, data.d[index].cpu().numpy(), est.d[index].cpu().numpy(),
        est.d.std(dim=0).cpu().numpy(),
        experiment.disturbance.remainder_bound(data.t),
        outdir / "disturbance.png",
    )
    coeff_true = (
        data.coefficients[index] if data.coefficients is not None
        else experiment.disturbance.coefficients[0]
    )
    plot_parameters(
        {"true": data.theta[index].cpu().numpy(),
         "est": est.theta[index].cpu().numpy(),
         "bounds": experiment.system.theta_bounds.cpu().numpy()},
        {"true": coeff_true.cpu().numpy(),
         "est": est.a[index].cpu().numpy(),
         "bounds": experiment.coefficient_bounds.cpu().numpy()},
        outdir / "parameters.png",
    )
    plot_errors(metrics.as_dict(), outdir / "errors.png")

    print()
    print(f"run {args.run}, evaluated under {noise!r} "
          f"(trained on truncated Gaussian at sigma = {config.data.noise_sigma:g}), "
          f"trajectory {index}")
    print(metrics.summary())
    print(f"\nfigures in {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
