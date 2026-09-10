#!/usr/bin/env python3
r"""Estimates against truth for several randomly drawn trajectories.

The per-run ``states.png`` shows one trajectory, which is enough to see whether
the estimator tracks at all and not enough to see how its quality *varies*.  A
trajectory-conditioned estimator produces a different fit for every initial
condition and every disturbance realisation, so the spread across trajectories
is part of the result rather than a detail.

One row per trajectory, four columns: the measured coordinate with the noisy
measurement overlaid, the two unmeasured coordinates, and the disturbance.
Every panel carries its own RMSE, so a row that is unusually good or bad is
visible immediately, and the last row of the figure summarises the whole
validation split rather than only the drawn sample.

Trajectories are drawn from the held-out split with a fixed seed, so the same
command reproduces the same figure.

Usage::

    python scripts/plot_trajectories.py --run outputs/big_w10_v2_s1 --count 10
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
from pibe.eval.figures import EST, MEAS, TRUTH  # noqa: E402
from pibe.eval.metrics import normalized_l2  # noqa: E402
from pibe.experiment import build_experiment  # noqa: E402
from pibe.training.callbacks import load_checkpoint, resolve_checkpoint  # noqa: E402
from pibe.utils.logging import setup_logging  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--split", choices=("val", "train"), default="val")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None,
                        help="defaults to <run>/figures/trajectories.png")
    args = parser.parse_args()
    setup_logging("WARNING")

    config = RunConfig.from_yaml(args.run / "config.resolved.yaml")
    experiment = build_experiment(config)
    load_checkpoint(resolve_checkpoint(args.run), experiment.bank)
    experiment.bank.to(device=experiment.device, dtype=experiment.dtype)
    experiment.bank.eval()

    data = experiment.val_data if args.split == "val" else experiment.train_data
    with torch.no_grad():
        estimates = experiment.bank.estimate(data.y, data.t, experiment.t_coll)

    # Errors over the whole split, so the drawn sample can be put in context.
    state_error = normalized_l2(estimates.x - data.x, dim=1).cpu().numpy()   # (P, n)
    d_error = normalized_l2(estimates.d - data.d, dim=1).cpu().numpy()       # (P,)

    generator = np.random.default_rng(args.seed)
    picks = generator.choice(len(data), size=min(args.count, len(data)),
                             replace=False)

    t = data.t.cpu().numpy()
    x_true = data.x.cpu().numpy()
    x_est = estimates.x.cpu().numpy()
    y = data.y.cpu().numpy()
    d_true = data.d.cpu().numpy()
    d_est = estimates.d.cpu().numpy()
    n = x_true.shape[2]

    rows = len(picks)
    fig, axes = plt.subplots(rows, n + 1, figsize=(3.6 * (n + 1), 1.75 * rows),
                             squeeze=False, sharex=True)

    for row, index in enumerate(picks):
        for j in range(n):
            ax = axes[row][j]
            if j == 0:
                ax.plot(t, y[index], color=MEAS, lw=0.6, alpha=0.45, zorder=1,
                        label="$y$ (noisy)")
            ax.plot(t, x_true[index, :, j], color=TRUTH, lw=1.5, zorder=2,
                    label="true")
            ax.plot(t, x_est[index, :, j], color=EST, lw=1.3, ls="--", zorder=3,
                    label="estimated")
            ax.set_ylabel(f"$x_{{{j+1}}}$" if row == rows - 1 else "")
            ax.set_title(f"$x_{{{j+1}}}$  rmse {state_error[index, j]:.2e}",
                         loc="left", fontsize=8)

        ax = axes[row][n]
        ax.plot(t, d_true[index], color=TRUTH, lw=1.5, label="true $d$")
        ax.plot(t, d_est[index], color=EST, lw=1.3, ls="--", label=r"$\hat d$")
        ax.set_title(f"$d$  rmse {d_error[index]:.2e}", loc="left", fontsize=8)

        axes[row][0].set_ylabel(f"traj {index}", fontsize=8)

    for ax in axes.ravel():
        ax.grid(alpha=0.22)
        ax.tick_params(labelsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("t  [s]")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=3,
               bbox_to_anchor=(0.995, 0.997), fontsize=8.5)

    fig.suptitle(
        f"{args.run.name} — {rows} trajectories drawn at random from the "
        f"{args.split} split ({len(data)} available), seed {args.seed}\n"
        "split-wide medians:  "
        + "   ".join(f"$x_{{{j+1}}}$ {np.median(state_error[:, j]):.2e}"
                     for j in range(n))
        + f"   $d$ {np.median(d_error):.2e}",
        fontsize=10.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.35 / rows))

    out = args.out or (args.run / "figures" / "trajectories.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)

    print(f"{args.run.name}, {args.split} split, {len(data)} trajectories\n")
    header = f"{'traj':>6}" + "".join(f"{'x' + str(j+1):>11}" for j in range(n)) + f"{'d':>11}"
    print(header)
    print("-" * len(header))
    for index in picks:
        print(f"{index:>6}" + "".join(f"{state_error[index, j]:>11.3e}" for j in range(n))
              + f"{d_error[index]:>11.3e}")
    print("-" * len(header))
    print(f"{'median':>6}" + "".join(f"{np.median(state_error[:, j]):>11.3e}" for j in range(n))
          + f"{np.median(d_error):>11.3e}")
    print(f"{'best':>6}" + "".join(f"{state_error[:, j].min():>11.3e}" for j in range(n))
          + f"{d_error.min():>11.3e}")
    print(f"{'worst':>6}" + "".join(f"{state_error[:, j].max():>11.3e}" for j in range(n))
          + f"{d_error.max():>11.3e}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
