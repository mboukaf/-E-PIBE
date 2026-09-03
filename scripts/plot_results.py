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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from pibe.config import RunConfig  # noqa: E402
from pibe.eval.metrics import compute_metrics  # noqa: E402
from pibe.eval.oracle import compare_to_oracle  # noqa: E402
from pibe.experiment import build_experiment  # noqa: E402
from pibe.training.callbacks import load_checkpoint  # noqa: E402
from pibe.utils.logging import get_logger, setup_logging  # noqa: E402

logger = get_logger("plot_results")

# Colourblind-safe, checked for adjacent-pair separation.
TRUTH = "#2a78d6"
EST = "#eb6834"
MEAS = "#1baf7a"

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 130,
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 9,
    "axes.titlesize": 9.5,
    "legend.frameon": False,
    "legend.fontsize": 8.5,
})


def plot_states(t, x_true, x_est, y, out: Path, index: int = 0) -> None:
    """True vs estimated trajectory per coordinate.

    The per-panel annotation reports both the RMSE and the *mean offset*.  That
    second number is the one to read: a large offset with a small residual
    scatter is the signature of Remark 7's degeneracy --- a constant parameter
    error absorbed by a constant shift of the next state --- rather than a
    failure to track the dynamics.
    """
    n = x_true.shape[-1]
    fig, axes = plt.subplots(n, 1, figsize=(8.4, 1.95 * n), sharex=True)
    axes = np.atleast_1d(axes)
    handles = None
    for j, ax in enumerate(axes):
        if j == 0:
            ax.plot(t, y[index], color=MEAS, lw=0.7, alpha=0.5,
                    label="$y$ (measured, noisy)", zorder=1)
        ax.plot(t, x_true[index, :, j], color=TRUTH, lw=1.9, label="true", zorder=2)
        ax.plot(t, x_est[index, :, j], color=EST, lw=1.6, ls="--", label="estimated", zorder=3)

        err = x_est[index, :, j] - x_true[index, :, j]
        rmse = float(np.sqrt(np.mean(err ** 2)))
        offset = float(np.mean(err))
        detrended = float(np.sqrt(np.mean((err - offset) ** 2)))
        ax.set_ylabel(f"$x_{{{j + 1}}}$")
        ax.set_title(
            f"$x_{{{j + 1}}}$ — {'measured' if j == 0 else 'unmeasured'}"
            f"      RMSE {rmse:.3e}"
            f"      mean offset {offset:+.3f}"
            f"      after removing it {detrended:.3e}",
            loc="left", fontsize=9,
        )
        if j == 0:
            handles = ax.get_legend_handles_labels()
    axes[-1].set_xlabel("t  [s]")
    fig.legend(*handles, loc="upper right", ncol=3, bbox_to_anchor=(0.995, 0.995))
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(out)
    plt.close(fig)
    logger.info("wrote %s", out)


def plot_disturbance(t, d_true, d_est, d_spread, eps, out: Path) -> None:
    """The disturbance estimate against truth."""
    fig, (ax, axe) = plt.subplots(
        2, 1, figsize=(8.2, 4.2), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]}
    )
    ax.plot(t, d_true, color=TRUTH, lw=1.8, label="true $d(t)$")
    ax.plot(t, d_est, color=EST, lw=1.6, ls="--", label=r"estimated $\hat d = \Gamma_q^\top \hat a$")
    if d_spread is not None and np.any(d_spread > 0):
        ax.fill_between(t, d_est - d_spread, d_est + d_spread, color=EST, alpha=0.16,
                        lw=0, label="spread across trajectories")
    ax.set_ylabel("$d$")
    ax.set_title(
        f"disturbance   sup err {np.max(np.abs(d_est - d_true)):.3e}   "
        rf"($\|d\|_\infty$ = {np.max(np.abs(d_true)):.3f},  $\varepsilon_{{d,q}}$ = {eps:.3g})",
        loc="left",
    )
    ax.legend(loc="upper right", ncol=2)

    axe.plot(t, d_est - d_true, color=EST, lw=1.2)
    axe.axhline(0.0, color=TRUTH, lw=1.0, alpha=0.6)
    axe.set_ylabel(r"$\hat d - d$")
    axe.set_xlabel("t  [s]")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    logger.info("wrote %s", out)


def _dot_panel(ax, names, truth, est, bounds, title):
    """Truth vs estimate against the admissible interval, one row per entry."""
    ypos = np.arange(len(names))[::-1]
    for y, (lo, hi) in zip(ypos, bounds):
        ax.plot([lo, hi], [y, y], color="0.82", lw=5, solid_capstyle="round", zorder=1)
    for y, a, b in zip(ypos, truth, est):
        ax.plot([a, b], [y, y], color="0.55", lw=1.2, zorder=2)
    ax.scatter(truth, ypos, s=52, color=TRUTH, zorder=3, label="true")
    ax.scatter(est, ypos, s=52, color=EST, zorder=3, label="estimated")
    for y, a, b in zip(ypos, truth, est):
        ax.annotate(f"{b - a:+.3f}", (max(a, b), y), textcoords="offset points",
                    xytext=(9, -3), fontsize=8, color=EST)
    ax.set_yticks(ypos)
    ax.set_yticklabels(names)
    ax.set_title(title, loc="left")
    ax.grid(axis="y", visible=False)


def plot_parameters(theta, coeff, out: Path) -> None:
    """Parameter and coefficient recovery."""
    fig, axes = plt.subplots(
        2, 1, figsize=(7.4, 4.6),
        gridspec_kw={"height_ratios": [len(theta["true"]), len(coeff["true"])]},
    )
    _dot_panel(
        axes[0], [rf"$\theta_{{{i + 1}}}$" for i in range(len(theta["true"]))],
        theta["true"], theta["est"], theta["bounds"],
        r"parameters $\theta$   (grey rule = admissible set $\Theta_j$)",
    )
    axes[0].legend(loc="lower right", ncol=2)
    _dot_panel(
        axes[1], [rf"$a_{{{i + 1}}}$" for i in range(len(coeff["true"]))],
        coeff["true"], coeff["est"], coeff["bounds"],
        r"basis coefficients $a$   (grey rule = admissible set $\mathcal{A}$)",
    )
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    logger.info("wrote %s", out)


def plot_errors(metrics, out: Path) -> None:
    """Error growth along the chain, Proposition 4 / Remark 12."""
    l2 = np.asarray(metrics["state_l2"])
    sup = np.asarray(metrics["state_sup"])
    idx = np.arange(1, len(l2) + 1)
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    ax.semilogy(idx, l2, "o-", color=TRUTH, lw=1.8, label=r"$\|\hat x_j - x_j\|_{N,2}$")
    ax.semilogy(idx, sup, "s--", color=EST, lw=1.5, label=r"$\|\hat x_j - x_j\|_\infty$")
    ax.set_xticks(idx)
    ax.set_xticklabels([f"$x_{{{j}}}$" for j in idx])
    ax.set_xlabel("coordinate (cell order along the chain)")
    ax.set_ylabel("error")
    ax.set_title("error growth down the chain", loc="left")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    logger.info("wrote %s", out)


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

    t = data.t.cpu().numpy()
    plot_states(
        t, data.x.cpu().numpy(), estimates.x.cpu().numpy(), data.y.cpu().numpy(),
        outdir / "states.png", index=args.trajectory,
    )
    plot_disturbance(
        t, data.d.cpu().numpy(), estimates.d.mean(0).cpu().numpy(),
        estimates.d.std(0).cpu().numpy(),
        experiment.disturbance.remainder_bound(data.t),
        outdir / "disturbance.png",
    )
    plot_parameters(
        {
            "true": data.theta.cpu().numpy(),
            "est": estimates.theta.mean(0).cpu().numpy(),
            "bounds": experiment.system.theta_bounds.cpu().numpy(),
        },
        {
            "true": experiment.disturbance.coefficients.cpu().numpy(),
            "est": estimates.a.mean(0).cpu().numpy(),
            "bounds": experiment.coefficient_bounds.cpu().numpy(),
        },
        outdir / "parameters.png",
    )
    plot_errors(metrics.as_dict(), outdir / "errors.png")

    print()
    print(f"=== {args.run.name}  ({args.split}, P={len(data)}, T={config.data.horizon}) ===")
    print(metrics.summary())
    print()
    print(comparison.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
