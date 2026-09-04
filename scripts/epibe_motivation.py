#!/usr/bin/env python3
r"""Spread versus mean: why PIBE needs EPIBE for biased measurements.

PIBE's Section 3.1 analysis leans on one property of its noise model, and it is
not Gaussianity.  Proposition 1 needs the noise to be **zero-mean**, so that the
quadratic data term (23) carries no systematic component; the shape of the law
around that mean never enters the argument.

This script tests that claim where it is sharpest, by comparing perturbations of
*equal magnitude* that differ only in whether they average out:

``zero-mean``
    A symmetric law of standard deviation :math:`m` and mean zero.
``contaminated``
    The same, but with 5% of samples drawn 8x wider --- the harshest zero-mean
    law available here, included so the comparison is not a Gaussian artefact.
``biased``
    A tight law (the training :math:`\sigma`) displaced by a constant
    :math:`m`, so its mean is exactly the other arms' standard deviation.

Every point is reported with the spread over trajectories and noise draws, since
a claim that one arm is flat is only as good as the scatter behind it.  The last
panel is the mechanism rather than the symptom: the *signed* mean of
:math:`\hat x_1 - x_1`, which stays at zero in the first two arms and tracks
:math:`m` in the third.  A displaced :math:`\hat x_1` is then propagated through
the chain of Eq. (1) into the states, the parameters and the disturbance --- so
the parameter estimate inherits a bias that no amount of data removes.

Usage::

    python scripts/epibe_motivation.py --run outputs/n3v2_s0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from noise_sweep import clean_trajectories  # noqa: E402
from pibe.config import RunConfig  # noqa: E402
from pibe.data.noise import build_noise_model  # noqa: E402
from pibe.eval.metrics import normalized_l2  # noqa: E402
from pibe.experiment import build_experiment  # noqa: E402
from pibe.training.callbacks import load_checkpoint  # noqa: E402
from pibe.utils.logging import setup_logging  # noqa: E402
from pibe.utils.seeding import make_generator  # noqa: E402

# Pushed well past the trained sigma of 0.002: x_1 has amplitude ~1.5, so the
# largest magnitude here is a perturbation of order the signal itself.
MAGNITUDES = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1,
              0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0]

ARMS = {
    "zero-mean (Gaussian)": dict(family="gaussian", kind="spread"),
    "zero-mean (contaminated)": dict(family="contaminated", kind="spread"),
    "biased sensor": dict(family="gaussian", kind="mean"),
}
COLORS = {"zero-mean (Gaussian)": "#2a78d6",
          "zero-mean (contaminated)": "#1baf7a",
          "biased sensor": "#d62728"}


@torch.no_grad()
def errors(experiment, clean, omega) -> dict[str, torch.Tensor]:
    """Per-trajectory errors for one measurement realization."""
    est = experiment.bank.estimate(clean.x[..., 0] + omega, clean.t, experiment.t_coll)
    err = est.x - clean.x
    return {
        "x1": normalized_l2(err[..., 0], dim=1),                  # (P,)
        "theta": (est.theta - clean.theta).abs().mean(dim=1),     # (P,)
        "d": normalized_l2(est.d - clean.d, dim=1),               # (P,)
        "signed_x1": err[..., 0].mean(dim=1),                     # (P,)
    }


@torch.no_grad()
def measure(experiment, clean, family: str, kind: str, magnitude: float,
            realizations: int, seed: int, sigma_train: float,
            baseline: dict[str, torch.Tensor] | None = None) -> dict:
    """Pool per-trajectory errors over independent noise draws.

    Each trajectory is also reported *paired against its own noise-free result*.
    Without that pairing the scatter across trajectories --- which is a fixed
    property of the trained bank, not of the noise --- swamps the effect being
    measured, especially for the parameters, whose per-trajectory error is
    dominated by a constant offset.
    """
    if kind == "spread":
        noise = build_noise_model(family, magnitude)
    else:
        noise = build_noise_model(family, sigma_train, bias=magnitude,
                                  bias_relative=False)

    pooled: dict[str, list[torch.Tensor]] = {k: [] for k in
                                             ("x1", "theta", "d", "signed_x1")}
    excess: dict[str, list[torch.Tensor]] = {k: [] for k in ("x1", "theta", "d")}
    for r in range(realizations):
        omega = noise.sample(
            tuple(clean.y.shape), generator=make_generator(seed + 1000 * r),
            dtype=experiment.dtype,
        ).to(clean.y.device)
        current = errors(experiment, clean, omega)
        for key, value in current.items():
            pooled[key].append(value)
        if baseline is not None:
            for key in excess:
                excess[key].append(current[key] - baseline[key])

    def stats(chunks):
        v = torch.cat(chunks).cpu().numpy()
        # Two different spreads, answering two different questions.  The 10-90
        # percentiles describe the population of trajectories; the standard
        # error describes how well the *mean* is pinned down, which is the
        # quantity the zero-mean claim is about.
        return {"mean": float(v.mean()), "std": float(v.std()),
                "sem": float(v.std(ddof=1) / np.sqrt(v.size)), "n": int(v.size),
                "lo": float(np.percentile(v, 10)), "hi": float(np.percentile(v, 90))}

    row = {"magnitude": magnitude, "arm": f"{family}/{kind}"}
    row.update({key: stats(chunks) for key, chunks in pooled.items()})
    if baseline is not None:
        row.update({f"excess_{key}": stats(chunks) for key, chunks in excess.items()})
    return row


def draw(ax, xs, rows, key, color, label):
    """Mean with a 95% confidence interval, over the 10-90 population band."""
    mean = np.array([r[key]["mean"] for r in rows])
    sem = np.array([r[key]["sem"] for r in rows])
    lo = np.array([r[key]["lo"] for r in rows])
    hi = np.array([r[key]["hi"] for r in rows])
    ax.fill_between(xs, lo, hi, color=color, alpha=0.10, lw=0)
    ax.errorbar(xs, mean, yerr=1.96 * sem, fmt="-o", ms=3.5, lw=1.5,
                elinewidth=1.2, capsize=3.0, color=color, label=label)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("outputs/n3v2_s0"))
    parser.add_argument("--magnitudes", type=float, nargs="+", default=MAGNITUDES)
    parser.add_argument("--trajectories", type=int, default=64)
    parser.add_argument("--realizations", type=int, default=8)
    parser.add_argument("--seed", type=int, default=909)
    parser.add_argument("--out", type=Path, default=Path("outputs/epibe_motivation.png"))
    args = parser.parse_args()

    setup_logging("WARNING")
    config = RunConfig.from_yaml(args.run / "config.resolved.yaml")
    experiment = build_experiment(config)
    load_checkpoint(args.run / "bank.pt", experiment.bank)
    experiment.bank.to(device=experiment.device, dtype=experiment.dtype)
    experiment.bank.eval()
    sigma_train = config.data.noise_sigma
    clean = clean_trajectories(experiment, args.trajectories, args.seed)

    # The noise-free result each perturbed one is paired against.
    baseline = errors(
        experiment, clean, torch.zeros_like(clean.y)
    )

    results = {}
    for label, spec in ARMS.items():
        results[label] = [
            measure(experiment, clean, spec["family"], spec["kind"], m,
                    args.realizations, args.seed, sigma_train, baseline=baseline)
            for m in args.magnitudes
        ]
        print(f"done: {label}")

    xs = args.magnitudes
    fig, axes = plt.subplots(1, 4, figsize=(17.5, 4.2))
    panels = [
        ("x1", r"state error $\|\hat x_1 - x_1\|_{N,2}$", True),
        ("excess_theta",
         r"excess parameter error over noise-free" "\n"
         r"$\overline{|\hat\theta_j-\theta_j|}(m) - \overline{|\hat\theta_j-\theta_j|}(0)$",
         True),
        ("excess_d",
         r"excess disturbance error over noise-free" "\n"
         r"$\|\hat d - d\|_{N,2}(m) - (\cdot)(0)$", True),
        ("signed_x1", r"signed mean of $\hat x_1 - x_1$", False),
    ]
    for ax, (key, ylabel, logy) in zip(axes, panels):
        for label, rows in results.items():
            draw(ax, xs, rows, key, COLORS[label], label)
        ax.set_xscale("log")
        if logy and key.startswith("excess"):
            # A paired excess can legitimately be negative --- a zero-mean
            # perturbation sometimes helps a given trajectory --- so the axis
            # has to render zero and both signs, which a log axis cannot.
            ax.set_yscale("symlog", linthresh=1e-5)
            ax.axhline(0.0, color="0.6", lw=0.8)
        elif logy:
            ax.set_yscale("log")
        else:
            ax.plot(xs, xs, ":", color="0.4", lw=1.2, label="$y = m$ (full inheritance)")
            ax.axhline(0.0, color="0.6", lw=0.8)
            # The reference line runs off the top; the curves are what matters.
            ax.set_ylim(-0.06, 0.30)
        ax.axvline(sigma_train, color="0.6", lw=1, ls=":")
        ax.set_xlabel(r"perturbation magnitude $m$"
                      "\n(sd for the zero-mean arms, offset for the biased one)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7, loc="upper left")
    axes[0].set_title("(a) measured state", loc="left")
    axes[1].set_title("(b) parameters (paired)", loc="left")
    axes[2].set_title("(c) disturbance (paired)", loc="left")
    axes[3].set_title("(d) the mechanism", loc="left")

    fig.suptitle(
        "Equal-magnitude perturbations: a zero-mean law of sd $m$ vs a sensor "
        f"offset of $m$.  Bars are 95% CI on the mean, shading the 10-90 "
        f"percentile, over {args.trajectories} trajectories x "
        f"{args.realizations} noise draws.  "
        f"Model trained at $\\sigma$ = {sigma_train:g} (dotted).",
        fontsize=9.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"\nwrote {args.out}")

    payload = args.out.with_suffix(".json")
    payload.write_text(json.dumps(results, indent=2))
    print(f"wrote {payload}")

    # A compact table of the same numbers.
    print()
    print(f"{'m':>8} | " + " | ".join(f"{lbl.split(' (')[0][:12]:>26}" for lbl in results))
    print(f"{'':>8} | " + " | ".join(f"{'x1 err':>12}{'excess th':>14}" for _ in results))
    print("-" * 100)
    for i, m in enumerate(xs):
        cells = []
        for rows in results.values():
            cells.append(f"{rows[i]['x1']['mean']:>12.3e}"
                         f"{rows[i]['excess_theta']['mean']:>14.3e}")
        print(f"{m:>8.4g} | " + " | ".join(cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
