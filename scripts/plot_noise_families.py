#!/usr/bin/env python3
r"""Compare the noise laws themselves and their effect on a trained bank.

Three panels: the densities at a matched sigma, the state error against sigma
for each zero-mean shape, and the same for a sequence of sensor biases.  The
contrast between the second and third panels is the point --- the estimator is
indifferent to the *shape* of a zero-mean law and sensitive to its *mean*, which
is what Proposition 1's premise predicts.
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

from pibe.data.noise import build_noise_model  # noqa: E402
from pibe.utils.seeding import make_generator  # noqa: E402

FAMILIES = ["gaussian", "uniform", "laplace", "contaminated", "skewed"]
BIASES = [0.0, 0.5, 1.0, 2.0]


def curve(rows, key, comp=0):
    sig = [r["sigma"] for r in rows if r["sigma"] > 0]
    floor = min(sig) / 2.0
    xs = [r["sigma"] if r["sigma"] > 0 else floor for r in rows]
    ys = [(r[key][comp] if isinstance(r[key], list) else r[key]) for r in rows]
    return xs, ys, floor, sig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("outputs/n3v2_s0"))
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--out", type=Path, default=Path("outputs/noise_families.png"))
    args = parser.parse_args()

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))

    # (a) the laws themselves, at a matched standard deviation
    ax = axes[0]
    for fam in FAMILIES:
        w = build_noise_model(fam, args.sigma).sample(
            (400_000,), generator=make_generator(0)
        ).numpy()
        ax.hist(w, bins=np.linspace(-4 * args.sigma, 4 * args.sigma, 241),
                histtype="step", density=True, lw=1.4, label=fam)
    ax.axvline(0.0, color="0.6", lw=0.8)
    ax.set_yscale("log")
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel("density")
    ax.set_title(f"(a) the laws, all at sd = {args.sigma:g}", loc="left")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # (b) zero-mean shapes
    ax = axes[1]
    for fam in FAMILIES:
        path = args.run / f"noise_sweep_{fam}.json"
        if not path.exists():
            continue
        xs, ys, floor, sig = curve(json.loads(path.read_text()), "state", 0)
        ax.plot(xs, ys, "-o", ms=3.5, lw=1.4, label=fam)
    ax.set_title("(b) zero-mean laws: shape is irrelevant", loc="left")

    # (c) biased sensor
    ax = axes[2]
    for bias in BIASES:
        name = "noise_sweep_gaussian.json" if bias == 0 else f"noise_sweep_bias{bias}.json"
        path = args.run / name
        if not path.exists():
            continue
        xs, ys, floor, sig = curve(json.loads(path.read_text()), "state", 0)
        ax.plot(xs, ys, "-s", ms=3.5, lw=1.4,
                label="unbiased" if bias == 0 else rf"bias = {bias:g}$\sigma$")
    ax.set_title("(c) a biased sensor: mean is not", loc="left")

    for ax in axes[1:]:
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xticks([floor] + sig)
        ax.set_xticklabels(["0"] + [f"{v:g}" for v in sig], fontsize=6, rotation=45)
        ax.set_xlabel(r"noise level $\sigma$")
        ax.set_ylabel(r"$\|\hat x_1 - x_1\|_{N,2}$")
        ax.axvline(0.002, color="0.6", lw=1, ls=":")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7)
    axes[1].set_ylim(*axes[2].get_ylim())  # same scale, so (b) and (c) compare

    fig.suptitle(
        "PIBE under non-Gaussian measurement noise "
        "(model trained on truncated Gaussian, $\\sigma$ = 0.002, dotted line)",
        fontsize=10,
    )
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
