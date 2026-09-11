#!/usr/bin/env python3
r"""The same output signal at a ladder of noise levels, to judge what is realistic.

Choosing :math:`\sigma` from error tables alone is awkward: the numbers say how
the estimator degrades, not whether the measurement still looks like something a
sensor would produce.  This draws one trajectory's clean :math:`x_1` against
:math:`y = x_1 + \omega` at each level, full horizon on top and a short window
below where individual samples are resolved.

Each panel carries three numbers that put the level in context:

``sigma/pp``
    the noise standard deviation as a fraction of the signal's peak-to-peak
    range --- the closest thing to "how noisy does this look".
``sigma/s_x1``
    as a fraction of the *admissible* range, which is the normalization the
    error metrics of Section 5 use, so this is the column that compares directly
    against :math:`E_{x_1}`.
``SNR a3``
    the predicted signal-to-noise ratio of the least observable disturbance
    coefficient, from :mod:`pibe.eval.observability`.  Below 1 that coefficient
    leaves no pointwise trace in :math:`y` and no estimator can recover it, so
    it marks where the disturbance estimate must start failing regardless of how
    the states look.

Usage::

    python scripts/plot_noise_levels.py --run outputs/big_w10_v2_s1
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
from pibe.data.noise import build_noise_model  # noqa: E402
from pibe.eval.figures import MEAS, TRUTH  # noqa: E402
from pibe.eval.observability import basis_observability  # noqa: E402
from pibe.experiment import build_experiment  # noqa: E402
from pibe.utils.logging import setup_logging  # noqa: E402
from pibe.utils.seeding import make_generator  # noqa: E402

DEFAULT_LEVELS = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--levels", type=float, nargs="+", default=DEFAULT_LEVELS)
    parser.add_argument("--index", type=int, default=0,
                        help="which held-out trajectory to draw")
    parser.add_argument("--window", type=float, nargs=2, default=(5.0, 8.0),
                        help="time interval for the zoomed row")
    parser.add_argument("--seed", type=int, default=909)
    parser.add_argument("--out", type=Path, default=None,
                        help="defaults to <run>/figures/noise_levels.png")
    args = parser.parse_args()
    setup_logging("WARNING")

    config = RunConfig.from_yaml(args.run / "config.resolved.yaml")
    experiment = build_experiment(config)
    data = experiment.val_data
    index = min(args.index, len(data) - 1)

    t = data.t.cpu().numpy()
    clean = data.x[index, :, 0].cpu().numpy()          # the true x_1
    peak_to_peak = float(clean.max() - clean.min())
    s_x1 = float(experiment.system.reference_scales()[0][0])
    trained = config.data.noise_sigma

    lo, hi = args.window
    zoom = (t >= lo) & (t <= hi)

    levels = list(args.levels)
    cols = len(levels)
    fig, axes = plt.subplots(2, cols, figsize=(3.1 * cols, 6.4), squeeze=False,
                             sharey="row")

    for col, sigma in enumerate(levels):
        noise = build_noise_model(
            "gaussian", sigma,
            truncation_sigmas=config.data.noise_truncation_sigmas,
        )
        # Same generator seed at every level, so the realizations differ only in
        # scale and the panels are visually comparable.
        omega = noise.sample((t.size,), generator=make_generator(args.seed),
                             dtype=torch.float64).numpy()
        y = clean + omega

        snr = basis_observability(
            experiment.system, experiment.basis, experiment.coefficient_bounds,
            noise_sigma=sigma, n_samples=config.data.n_samples,
        ).snr.cpu().numpy()

        top = axes[0][col]
        top.plot(t, y, color=MEAS, lw=0.7, alpha=0.75, label="$y$ measured")
        top.plot(t, clean, color=TRUTH, lw=1.7, label="$x_1$ true")
        flag = "  <- trained" if abs(sigma - trained) < 1e-12 else ""
        top.set_title(
            f"$\\sigma$ = {sigma:g}{flag}\n"
            f"{100*sigma/peak_to_peak:.1f}% of peak-to-peak,  "
            f"{100*sigma/s_x1:.1f}% of $s_{{x_1}}$\n"
            f"SNR $a_3$ = {snr[-1]:.2f}"
            + ("  (below noise)" if snr[-1] < 1 else ""),
            loc="left", fontsize=8.5)
        top.axvspan(lo, hi, color="0.85", alpha=0.45, lw=0, zorder=0)

        bot = axes[1][col]
        bot.plot(t[zoom], y[zoom], "-o", color=MEAS, lw=0.8, ms=3, alpha=0.85)
        bot.plot(t[zoom], clean[zoom], color=TRUTH, lw=1.9)
        bot.set_xlabel("t  [s]")

    axes[0][0].set_ylabel("full horizon")
    axes[1][0].set_ylabel(f"zoom  t $\\in$ [{lo:g}, {hi:g}]")
    for ax in axes.ravel():
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=7.5)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2,
               bbox_to_anchor=(0.995, 0.998), fontsize=9)

    fig.suptitle(
        f"{args.run.name} — one held-out trajectory (index {index}) measured at "
        f"increasing noise.  Signal peak-to-peak {peak_to_peak:.3f}, "
        f"$s_{{x_1}}$ = {s_x1:g}, $N$ = {t.size} samples over {t[-1]:g} s.\n"
        "Shaded band on the top row is the window enlarged below.",
        fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.9))

    out = args.out or (args.run / "figures" / "noise_levels.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=145)
    plt.close(fig)

    header = (f"{'sigma':>8}{'w_bar=3s':>10}{'% of p-p':>10}{'% of s_x1':>11}"
              f"{'SNR a1':>9}{'SNR a2':>9}{'SNR a3':>9}")
    print(f"trajectory {index}: peak-to-peak {peak_to_peak:.4f}, s_x1 = {s_x1:g}\n")
    print(header)
    print("-" * len(header))
    for sigma in levels:
        snr = basis_observability(
            experiment.system, experiment.basis, experiment.coefficient_bounds,
            noise_sigma=sigma, n_samples=config.data.n_samples,
        ).snr.cpu().numpy()
        print(f"{sigma:>8.4g}{3*sigma:>10.4g}{100*sigma/peak_to_peak:>10.2f}"
              f"{100*sigma/s_x1:>11.2f}{snr[0]:>9.2f}{snr[1]:>9.2f}{snr[2]:>9.2f}"
              + ("  <- a3 below noise" if snr[2] < 1 else ""))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
