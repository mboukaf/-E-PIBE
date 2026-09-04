#!/usr/bin/env python3
"""Plot the noise-robustness curves produced by ``scripts/noise_sweep.py``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", type=Path, nargs="+", required=True,
                        help="noise_sweep.json files")
    parser.add_argument("--label", type=str, nargs="+", required=True)
    parser.add_argument("--trained-sigma", type=float, default=0.002)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    runs = [(lbl, load(p)) for lbl, p in zip(args.label, args.sweep)]
    n_state = len(runs[0][1][0]["state"])
    n_theta = len(runs[0][1][0]["theta"])

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    styles = ["-o", "--s"]

    for ax, (title, key, count) in zip(
        axes,
        [("state estimate", "state", n_state),
         ("parameter estimate", "theta", n_theta),
         ("disturbance estimate", "d_l2", 1)],
    ):
        for (label, rows), style in zip(runs, styles):
            # sigma = 0 cannot be drawn on a log axis; place it at the left edge.
            sig = [r["sigma"] for r in rows if r["sigma"] > 0]
            floor = min(sig) / 2.0
            xs = [r["sigma"] if r["sigma"] > 0 else floor for r in rows]
            if count == 1:
                ax.plot(xs, [r[key] for r in rows], style, ms=4,
                        label=f"{label}")
            else:
                for j in range(count):
                    ax.plot(xs, [r[key][j] for r in rows], style, ms=4,
                            label=f"{label}: "
                                  + (f"$x_{j+1}$" if key == "state" else rf"$\theta_{j+1}$"))
        ax.axvline(args.trained_sigma, color="0.6", lw=1, ls=":")
        ax.set_xscale("log"); ax.set_yscale("log")
        # The noise-free row has no place on a log axis; it is drawn at the
        # left edge and labelled so it is not misread as a small positive sigma.
        ax.set_xticks([floor] + sig)
        ax.set_xticklabels(["0"] + [f"{v:g}" for v in sig], fontsize=6, rotation=45)
        ax.set_xlabel(r"measurement noise $\sigma$")
        ax.set_title(title)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7)

    axes[0].set_ylabel(r"error  $\|\hat\cdot - \cdot\|_{N,2}$")
    fig.suptitle(
        "PIBE at test-time noise levels it was not trained on "
        f"(dotted line: training $\\sigma$ = {args.trained_sigma:g})",
        fontsize=10,
    )
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
