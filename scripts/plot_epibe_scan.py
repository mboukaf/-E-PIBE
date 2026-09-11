#!/usr/bin/env python3
r"""Plot the EPIBE location scan: does the objective select the true offset?

The energy data term is stationary along the location degeneracy, so gradient
descent cannot travel it --- but the objective is not *flat* there, and that is
the whole question.  This figure answers it directly: the energy objective
against the candidate offset, with the true sensor bias marked.  If the minimum
sits on the mark, EPIBE identifies the offset and only needed to be pointed
along the one coordinate it cannot descend.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

TRUTH = "#2a78d6"
EST = "#eb6834"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", type=Path, default=Path("outputs/_evidence/epibe_scan/scan.json"))
    parser.add_argument("--true-bias", type=float, default=0.05)
    parser.add_argument("--pibe-x1", type=float, default=None,
                        help="the PIBE control's x1 error, for reference")
    parser.add_argument("--out", type=Path, default=Path("outputs/epibe_scan.png"))
    args = parser.parse_args()

    rows = sorted(json.loads(args.scan.read_text()), key=lambda r: r["offset"])
    c = np.array([r["offset"] for r in rows])
    objective = np.array([r["objective"] for r in rows])
    mu = np.array([r["mu_hat"] for r in rows])
    drift = np.array([r["x1_minus_truth"] for r in rows])
    x1 = np.array([r["state_l2"][0] for r in rows])
    best = c[int(objective.argmin())]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))

    ax = axes[0]
    ax.plot(c, objective, "-o", color=EST, lw=1.6, ms=5)
    ax.axvline(args.true_bias, color=TRUTH, lw=1.4, ls="--", label="true sensor bias")
    ax.axvline(best, color="0.45", lw=1.2, ls=":", label=f"objective minimum ({best:+.3f})")
    ax.set_xlabel("candidate offset $c$")
    ax.set_ylabel(r"$\mathcal{L}^{n+1}_{Tot,E}$")
    ax.set_title("(a) the objective does select the offset", loc="left")

    ax = axes[1]
    ax.plot(c, mu, "-o", color=EST, lw=1.6, ms=5, label=r"$\hat\mu_\omega$ (Eq. 54)")
    ax.plot(c, c, ":", color="0.5", lw=1.2, label="$c$")
    ax.axhline(args.true_bias, color=TRUTH, lw=1.4, ls="--", label="true bias")
    ax.set_xlabel("candidate offset $c$")
    ax.set_ylabel(r"$\hat\mu_\omega$")
    ax.set_title("(b) the recovered noise mean", loc="left")

    ax = axes[2]
    ax.semilogy(c, x1, "-o", color=EST, lw=1.6, ms=5, label=r"EPIBE $\|\hat x_1-x_1\|_{N,2}$")
    if args.pibe_x1 is not None:
        ax.axhline(args.pibe_x1, color=TRUTH, lw=1.4, ls="--",
                   label=f"PIBE control ({args.pibe_x1:.2e})")
    ax.axvline(args.true_bias, color=TRUTH, lw=1.0, ls=":", alpha=0.6)
    ax.set_xlabel("candidate offset $c$")
    ax.set_ylabel(r"$\|\hat x_1 - x_1\|_{N,2}$")
    ax.set_title("(c) state accuracy", loc="left")

    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7.5)
    fig.suptitle(
        "EPIBE's location degeneracy is one scalar: seeding the first-cell "
        "density at $c$ and letting the objective choose",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")
    print(f"objective minimum at c = {best:+.4f}, true bias {args.true_bias:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
