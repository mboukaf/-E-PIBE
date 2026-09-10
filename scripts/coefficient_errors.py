#!/usr/bin/env python3
r"""How the disturbance-coefficient error depends on the coefficient's true value.

The aggregate :math:`\|\hat d - d\|_{N,2}` says how wrong the disturbance
estimate is; it says nothing about *how* it is wrong.  Three failure shapes look
alike in that one number and are distinguished here:

**Shrinkage.**  A regularized or under-determined estimator pulls
:math:`\hat a_j` towards the middle of its admissible box, so the regression of
:math:`\hat a_j` on :math:`a_j` has slope below one.  The error then grows
linearly with :math:`|a_j|`, and the estimate is systematically too small rather
than merely noisy.  A slope near zero is the degenerate end of this: the head has
stopped reading :math:`y` and emits a constant.

**Bias.**  A nonzero intercept, or a mean error that does not pass through the
origin, means the estimate is displaced regardless of the truth.

**Heteroscedasticity.**  The scatter about the fit may widen with :math:`|a_j|`,
which matters because a single RMSE then describes neither the small- nor the
large-coefficient regime.

Each coefficient gets three panels: the estimate against the truth with the
identity and the fitted line; the signed error against the truth with binned
mean and spread; and the error's distribution.  Read the first for shrinkage,
the second for bias and heteroscedasticity, the third for tails.

Written for the one run in the cluster grid that recovered every coefficient
(``big_w10_v2_s1``), where the question is no longer "did it work" but "how well,
and where does it degrade".

Usage::

    python scripts/coefficient_errors.py --run outputs/big_w10_v2_s1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from pibe.config import RunConfig  # noqa: E402
from pibe.eval.figures import EST, TRUTH  # noqa: E402
from pibe.eval.observability import basis_observability  # noqa: E402
from pibe.experiment import build_experiment  # noqa: E402
from pibe.training.callbacks import load_checkpoint, resolve_checkpoint  # noqa: E402
from pibe.utils.logging import setup_logging  # noqa: E402


def bandwidth(values: np.ndarray) -> float:
    """Scott's rule, with the IQR as a fallback for heavy-tailed samples.

    ``n ** (-1/5)`` is the usual optimal rate for a Gaussian kernel; taking the
    smaller of the standard deviation and a robust IQR estimate keeps a few
    outliers from over-smoothing the bulk of the distribution.
    """
    n = values.size
    spread = values.std()
    q75, q25 = np.percentile(values, [75, 25])
    robust = (q75 - q25) / 1.349
    scale = min(spread, robust) if robust > 0 else spread
    return float(scale * n ** (-1 / 5)) or 1.0


def kde(samples: np.ndarray, grid: np.ndarray, factor: float = 1.0) -> np.ndarray:
    """Gaussian kernel density estimate of ``samples`` evaluated on ``grid``.

    Written out rather than imported: scipy is not a dependency of this project,
    and a one-dimensional Gaussian KDE is three lines.  ``factor`` scales the
    bandwidth for deliberate over- or under-smoothing.
    """
    h = bandwidth(samples) * factor
    z = (grid[:, None] - samples[None, :]) / h
    return np.exp(-0.5 * z ** 2).sum(axis=1) / (samples.size * h * np.sqrt(2 * np.pi))


def local_moments(x: np.ndarray, y: np.ndarray, grid: np.ndarray,
                  factor: float = 1.0):
    """Kernel-weighted conditional mean and standard deviation of ``y`` given ``x``.

    The smooth counterpart of binning: every point contributes to every grid
    location with a Gaussian weight, so the curve does not jump at bin edges and
    does not depend on where the edges happen to fall.
    """
    h = bandwidth(x) * factor
    w = np.exp(-0.5 * ((grid[:, None] - x[None, :]) / h) ** 2)
    total = w.sum(axis=1)
    mean = (w * y[None, :]).sum(axis=1) / total
    var = (w * (y[None, :] - mean[:, None]) ** 2).sum(axis=1) / total
    # Where the kernel sees almost no data the estimate is meaningless.
    sparse = total < 3.0
    mean[sparse], var[sparse] = np.nan, np.nan
    return mean, np.sqrt(var)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "train"), default="val")
    parser.add_argument("--smooth", type=float, default=1.0,
                        help="bandwidth multiplier for the KDE and the local "
                             "conditional moments; >1 smooths more")
    parser.add_argument("--out", type=Path, default=None,
                        help="defaults to <run>/figures/coefficient_errors.png")
    args = parser.parse_args()
    setup_logging("WARNING")

    config = RunConfig.from_yaml(args.run / "config.resolved.yaml")
    experiment = build_experiment(config)
    load_checkpoint(resolve_checkpoint(args.run), experiment.bank)
    experiment.bank.to(device=experiment.device, dtype=experiment.dtype)
    experiment.bank.eval()

    data = experiment.val_data if args.split == "val" else experiment.train_data
    if data.coefficients is None:
        raise SystemExit(
            "this run shares one disturbance across trajectories, so there is no "
            "per-trajectory truth to plot against; needs "
            "sample_disturbance_per_trajectory: true"
        )
    with torch.no_grad():
        estimates = experiment.bank.estimate(data.y, data.t, experiment.t_coll)

    truth = data.coefficients.cpu().numpy()          # (P, q)
    estimate = estimates.a.cpu().numpy()             # (P, q)
    error = estimate - truth
    q = truth.shape[1]
    bounds = experiment.coefficient_bounds.cpu().numpy()

    # The pre-training prediction, for context: how visible is each coefficient
    # at the output in the first place?
    snr = None
    if config.data.noise_sigma > 0:
        snr = basis_observability(
            experiment.system, experiment.basis, experiment.coefficient_bounds,
            noise_sigma=config.data.noise_sigma, n_samples=config.data.n_samples,
        ).snr.cpu().numpy()

    rows = []
    fig, axes = plt.subplots(3, q, figsize=(4.3 * q, 10.2), squeeze=False)

    for j in range(q):
        a, e, err = truth[:, j], estimate[:, j], error[:, j]
        lo, hi = float(bounds[j, 0]), float(bounds[j, 1])
        slope, intercept = np.polyfit(a, e, 1)
        corr = float(np.corrcoef(a, e)[0, 1])
        summary = {
            "coefficient": j + 1,
            "corr": corr,
            "slope": float(slope),
            "intercept": float(intercept),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "mean_error": float(err.mean()),
            "sd_true": float(a.std()),
            "sd_estimate": float(e.std()),
            "snr": None if snr is None else float(snr[j]),
        }
        rows.append(summary)

        # --- (1) estimate vs truth -------------------------------------
        ax = axes[0][j]
        ax.scatter(a, e, s=6, alpha=0.35, color=EST, lw=0)
        span = np.array([lo, hi])
        ax.plot(span, span, color=TRUTH, lw=1.6, label="identity")
        ax.plot(span, slope * span + intercept, color="0.35", lw=1.4, ls="--",
                label=f"fit: slope {slope:.3f}")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel(f"true $a_{{{j+1}}}$")
        ax.set_ylabel(f"estimated $\\hat a_{{{j+1}}}$")
        title = f"$a_{{{j+1}}}$   corr {corr:.3f}"
        if snr is not None:
            title += f"   (predicted SNR {snr[j]:.2f})"
        ax.set_title(title, loc="left")
        ax.legend(fontsize=7.5, loc="upper left")

        # --- (2) signed error vs truth ---------------------------------
        ax = axes[1][j]
        ax.scatter(a, err, s=6, alpha=0.3, color=EST, lw=0)
        grid = np.linspace(lo, hi, 200)
        mean_curve, sd_curve = local_moments(a, err, grid, factor=args.smooth)
        ax.fill_between(grid, mean_curve - sd_curve, mean_curve + sd_curve,
                        color=TRUTH, alpha=0.18, lw=0,
                        label="local mean $\\pm$ sd")
        ax.plot(grid, mean_curve, color=TRUTH, lw=1.8)
        ax.axhline(0.0, color="0.4", lw=1.0)
        # Pure shrinkage predicts error = (slope - 1) * a + intercept.
        ax.plot(span, (slope - 1) * span + intercept, color="0.35", lw=1.3, ls="--",
                label="shrinkage prediction")
        ax.set_xlim(lo, hi)
        ax.set_xlabel(f"true $a_{{{j+1}}}$")
        ax.set_ylabel(f"$\\hat a_{{{j+1}}} - a_{{{j+1}}}$")
        ax.set_title(f"mean error {err.mean():+.4f}, RMSE {summary['rmse']:.4f}",
                     loc="left")
        ax.legend(fontsize=7.5)

        # --- (3) error distribution ------------------------------------
        ax = axes[2][j]
        pad = 3.5 * err.std()
        grid = np.linspace(err.min() - pad, err.max() + pad, 512)
        density = kde(err, grid, factor=args.smooth)
        ax.fill_between(grid, density, color=EST, alpha=0.35, lw=0)
        ax.plot(grid, density, color=EST, lw=1.8, label="KDE")
        # A Gaussian of the same mean and spread, to judge the tails against.
        gaussian = np.exp(-0.5 * ((grid - err.mean()) / err.std()) ** 2) / (
            err.std() * np.sqrt(2 * np.pi))
        ax.plot(grid, gaussian, color="0.35", lw=1.2, ls="--", label="Gaussian fit")
        # The samples themselves, so the smoothing cannot hide how many there are.
        ax.plot(err, np.full_like(err, -0.02 * density.max()), "|",
                color=EST, alpha=0.35, ms=5)
        ax.axvline(0.0, color=TRUTH, lw=1.6, label="zero")
        ax.axvline(err.mean(), color="0.35", lw=1.0, ls=":",
                   label=f"mean {err.mean():+.4f}")
        ax.set_ylim(bottom=-0.05 * density.max())
        ax.set_xlabel(f"$\\hat a_{{{j+1}}} - a_{{{j+1}}}$")
        ax.set_ylabel("density")
        ax.set_title(f"sd {err.std():.4f}  (truth sd {a.std():.4f})", loc="left")
        ax.legend(fontsize=7.5)

    for ax in axes.ravel():
        ax.grid(alpha=0.25)

    fig.suptitle(
        f"Disturbance-coefficient error against true value — {args.run.name}, "
        f"{args.split} split, {truth.shape[0]} trajectories\n"
        "top: estimate vs truth (slope < 1 = shrinkage)   "
        "middle: signed error vs truth (bias, heteroscedasticity)   "
        "bottom: error distribution",
        fontsize=10.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.945))

    out = args.out or (args.run / "figures" / "coefficient_errors.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)

    payload = out.with_suffix(".json")
    payload.write_text(json.dumps(rows, indent=2))

    header = (f"{'coef':>5}{'corr':>8}{'slope':>8}{'intercept':>11}{'RMSE':>10}"
              f"{'mean err':>10}{'sd(true)':>10}{'sd(est)':>9}{'SNR':>8}")
    print(f"{args.run.name}, {args.split} split, {truth.shape[0]} trajectories\n")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['coefficient']:>5}{r['corr']:>8.3f}{r['slope']:>8.3f}"
              f"{r['intercept']:>11.4f}{r['rmse']:>10.4f}{r['mean_error']:>10.4f}"
              f"{r['sd_true']:>10.4f}{r['sd_estimate']:>9.4f}"
              + (f"{r['snr']:>8.2f}" if r["snr"] is not None else f"{'-':>8}"))
    print(f"\nwrote {out}\nwrote {payload}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
