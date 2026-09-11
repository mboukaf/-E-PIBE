#!/usr/bin/env python3
r"""Where does the energy-based data term overtake the quadratic one?

PIBE's quadratic data term is the negative log-likelihood of a zero-mean
Gaussian up to constants, so it asserts :math:`\mu_\omega = 0` rather than
inferring it.  Proposition 1 says the consequence is that it cannot separate the
signal from an unknown location shift, and the measured consequence is blunt:
its state error tracks the sensor offset one-for-one.

EPIBE can represent a nonzero mean, and its failure mode --- a drifting density
--- is *bounded* by the residual support rather than growing with the offset.
Two curves with different slopes must cross, and this locates the crossing.

The left panels are the error of each estimator against the offset.  The right
panel is the offset actually recovered, :math:`\hat\mu_\omega` of Eq. (54),
against the truth: on the diagonal means exact, and the seed markers show how
reproducible that is.  Read the two together --- an error win that rests on an
unreproducible location estimate is not a usable method, and this figure is laid
out so that cannot be hidden.
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

PIBE_C, EPIBE_C = "#2a78d6", "#eb6834"


def load(path: Path):
    metrics = path / "metrics_val.json"
    if not metrics.exists():
        return None
    out = json.loads(metrics.read_text())
    densities = path / "densities.json"
    out["mu_hat"] = (json.loads(densities.read_text())["mu_omega_hat"]
                     if densities.exists() else None)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument("--out", type=Path,
                        default=Path("outputs/bias_crossover.png"))
    args = parser.parse_args()

    # (bias, pibe dir, epibe dir) for the single-seed sweep.
    sweep = [
        (0.05, "bb_pibe_b0.05", "bb_epibe_b0.05"),
        (0.15, "bb_pibe_b0.15", "bb_epibe_b0.15"),
        (0.30, "bb_pibe_b0.30", "bb_epibe_b0.30"),
        (0.50, "bb_pibe_b0.5_BE12.0", "bb_epibe_b0.5_BE12.0"),
        (0.75, "bb_pibe_b0.75", "bb_epibe_b0.75"),
    ]
    # Replications, to show the spread behind any single point.
    repeats = {
        0.15: ["bb_epibe_b0.15", "bb_epibe_b0.15_s1", "bb_epibe_b0.15_s2"],
        0.50: ["lw_epibe_b0.5_s0", "lw_epibe_b0.5_s1", "lw_epibe_b0.5_s2"],
    }

    bias, p_rows, e_rows = [], [], []
    for b, pd, ed in sweep:
        p, e = load(args.outputs / pd), load(args.outputs / ed)
        if p is None or e is None:
            continue
        bias.append(b); p_rows.append(p); e_rows.append(e)
    bias = np.array(bias)

    fig, axes = plt.subplots(1, 4, figsize=(17.5, 4.2))
    panels = [("x₁", "state_l2", 0), ("x₂", "state_l2", 1), ("θ₂", "theta_abs_error", 1)]

    for ax, (name, key, idx) in zip(axes[:3], panels):
        pv = np.array([r[key][idx] for r in p_rows])
        ev = np.array([r[key][idx] for r in e_rows])
        ax.plot(bias, pv, "-o", color=PIBE_C, lw=1.8, ms=6, label="PIBE")
        ax.plot(bias, ev, "-s", color=EPIBE_C, lw=1.8, ms=6, label="EPIBE (cell 2)")
        if name == "x₁":
            ax.plot(bias, bias, ":", color="0.45", lw=1.3,
                    label="offset absorbed entirely")
        # Seed spread, where it was measured.
        for b, dirs in repeats.items():
            vals = [load(args.outputs / d) for d in dirs]
            vals = [v[key][idx] for v in vals if v is not None]
            if len(vals) > 1:
                ax.plot([b] * len(vals), vals, "x", color=EPIBE_C, ms=8, mew=1.8,
                        label="EPIBE, other seeds" if name == "x₁" else None)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("sensor offset")
        ax.set_ylabel(f"$\\|\\hat {name[0]} - {name[0]}\\|$" if name != "θ₂"
                      else r"$|\hat\theta_2 - \theta_2|$")
        ax.set_title(f"{name}", loc="left")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7.5)

    # Recovered offset.
    ax = axes[3]
    mu = np.array([r["mu_hat"] if r["mu_hat"] is not None else np.nan
                   for r in e_rows])
    ax.plot(bias, mu, "-s", color=EPIBE_C, lw=1.8, ms=6, label=r"$\hat\mu_\omega$")
    span = np.array([bias.min() * 0.8, bias.max() * 1.2])
    ax.plot(span, span, ":", color="0.45", lw=1.4, label="exact")
    ax.axhline(0.0, color="0.7", lw=1.0)
    for b, dirs in repeats.items():
        vals = [load(args.outputs / d) for d in dirs]
        vals = [v["mu_hat"] for v in vals if v is not None and v["mu_hat"] is not None]
        if len(vals) > 1:
            ax.plot([b] * len(vals), vals, "x", color=EPIBE_C, ms=9, mew=1.8,
                    label="other seeds" if b == min(repeats) else None)
    ax.set_xscale("log")
    ax.set_xlabel("true sensor offset")
    ax.set_ylabel(r"recovered $\hat\mu_\omega$")
    ax.set_title("offset recovery (Eq. 54)", loc="left")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7.5)

    fig.suptitle(
        "Sensor offset: PIBE's error grows one-for-one with the bias it cannot "
        "represent, EPIBE's does not.  σ = 0.05, held-out split.\n"
        "Crosses are repeated seeds — the error win is only as trustworthy as "
        "the location estimate behind it.",
        fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
