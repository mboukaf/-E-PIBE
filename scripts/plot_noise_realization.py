#!/usr/bin/env python3
r"""One realization of the measurement noise, drawn on a fine time grid.

The noise laws of :mod:`pibe.data.noise` are i.i.d. across samples: there is no
time structure to resolve, so a finer :math:`\Delta t` does not reveal a
smoother process underneath --- it only draws more independent samples over the
same horizon.  At :math:`\Delta t = 0.001` against the data grid's
:math:`\Delta t = 0.1` that is a hundredfold increase, which is exactly what
makes the *shape* of the law visible: the trace fills its support, and the
histogram beside it has enough draws to show what the moments are made of.

What "respecting the moments" means here
----------------------------------------
Every family is calibrated at construction --- the realized standard deviation
is exactly :math:`\sigma`, and the support bound :math:`\bar w` is the one the
analysis assumes --- so a realization respects the moments by construction
rather than by adjustment.  This script checks that rather than asserting it:
the printed table puts the draw's empirical mean, standard deviation, skewness
and kurtosis against a large reference sample of the same law, and against the
stated bound.  Nothing is rescaled to make them agree.

The third and fourth moments are the ones worth reading.  Mean and variance are
matched across every family by construction, so they cannot distinguish a
Gaussian from a contaminated mixture; skewness and kurtosis can, and they are
what the estimator's quadratic data term is blind to.

Usage::

    python scripts/plot_noise_realization.py --sigma 0.1
    python scripts/plot_noise_realization.py --sigma 0.1 --family mixture
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
from pibe.data.noise import NOISE_FAMILIES, build_noise_model  # noqa: E402
from pibe.utils.logging import setup_logging  # noqa: E402
from pibe.utils.seeding import make_generator  # noqa: E402

DRAW, REFERENCE = "#2c6ab1", "#ff6c27"
INK, INK_SOFT, RULE = "#0b0b0b", "#52514e", "#9a9993"


def paper_style(base: float) -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": base, "axes.labelsize": base,
        "xtick.labelsize": base - 0.5, "ytick.labelsize": base - 0.5,
        "legend.fontsize": base - 0.5,
        "axes.edgecolor": INK_SOFT, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK_SOFT, "ytick.color": INK_SOFT,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True,
        "grid.color": "#d8d8d4", "grid.linewidth": 0.4,
        "legend.frameon": False,
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def moments(values: np.ndarray) -> dict:
    """Mean, standard deviation, and the standardized third and fourth moments."""
    mean = float(values.mean())
    sd = float(values.std())
    z = (values - mean) / sd
    return dict(mean=mean, sd=sd, skew=float((z ** 3).mean()),
                kurtosis=float((z ** 4).mean()), max_abs=float(np.abs(values).max()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("outputs/big_w10_v2_s1"),
                        help="only read for the horizon and the default family")
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--family", default=None, choices=sorted(NOISE_FAMILIES))
    parser.add_argument("--noise-arg", action="append", default=[],
                        metavar="KEY=VALUE")
    parser.add_argument("--bias", type=float, default=0.0,
                        help="additive offset in units of sigma")
    parser.add_argument("--dt", type=float, default=0.001,
                        help="sampling interval of the realization")
    parser.add_argument("--horizon", type=float, default=None)
    parser.add_argument("--zoom", type=float, default=0.2,
                        help="length of the enlarged window, in seconds")
    parser.add_argument("--seed", type=int, default=909)
    parser.add_argument("--reference-draws", type=int, default=2_000_000)
    parser.add_argument("--width", type=float, default=7.0)
    parser.add_argument("--height", type=float, default=4.6)
    parser.add_argument("--fontsize", type=float, default=9.0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    setup_logging("WARNING")
    paper_style(args.fontsize)

    config = RunConfig.from_yaml(args.run / "config.resolved.yaml")
    horizon = args.horizon if args.horizon is not None else config.data.horizon
    family = args.family or config.data.noise_family
    kwargs = {}
    for item in args.noise_arg:
        key, value = item.split("=", 1)
        kwargs[key.strip()] = float(value)
    if "truncation_sigmas" not in kwargs:
        kwargs["truncation_sigmas"] = config.data.noise_truncation_sigmas

    noise = build_noise_model(family, args.sigma, bias=args.bias, **kwargs)

    n = int(round(horizon / args.dt)) + 1
    t = np.linspace(0.0, horizon, n)
    omega = noise.sample((n,), generator=make_generator(args.seed)).numpy()
    # An independent, much larger draw of the *same* law: the yardstick the
    # realization is checked against, so any gap is sampling error and not a
    # miscalibration.
    reference = noise.sample((args.reference_draws,),
                             generator=make_generator(args.seed + 1)).numpy()

    got, want = moments(omega), moments(reference)

    fig = plt.figure(figsize=(args.width, args.height))
    grid = fig.add_gridspec(2, 2, width_ratios=(3.1, 1.0), height_ratios=(1.0, 1.0),
                            hspace=0.42, wspace=0.11)
    full = fig.add_subplot(grid[0, 0])
    zoom = fig.add_subplot(grid[1, 0])
    dens = fig.add_subplot(grid[:, 1], sharey=full)

    for ax, window in ((full, (0.0, horizon)), (zoom, (0.0, args.zoom))):
        mask = (t >= window[0]) & (t <= window[1])
        # Hairline over the full horizon, where 20001 samples overlap into a
        # band; a marked line in the window, where they are individually
        # resolved and the point is that each draw is independent.
        if ax is full:
            ax.plot(t[mask], omega[mask], color=DRAW, lw=0.25, alpha=0.85)
        else:
            ax.plot(t[mask], omega[mask], color=DRAW, lw=0.8, marker="o", ms=2.4,
                    markerfacecolor="white", markeredgewidth=0.6, alpha=0.95)
        for level, style, label in (
            (noise.bound, (0, (5, 3)), r"$\pm\bar w$"),
            (-noise.bound, (0, (5, 3)), None),
            (noise.mean + args.sigma, (0, (1.5, 2)), r"$\mu\pm\sigma$"),
            (noise.mean - args.sigma, (0, (1.5, 2)), None),
        ):
            ax.axhline(level, color=RULE, lw=0.8, ls=style, zorder=1,
                       label=label if ax is full else None)
        ax.axhline(noise.mean, color=INK_SOFT, lw=0.9, zorder=1,
                   label=r"$\mu_\omega$" if ax is full else None)
        ax.set_xlim(*window)
        ax.set_ylabel(r"$\omega$")
    full.set_title(f"full horizon,  {n} samples at $\\Delta t$ = {args.dt:g} s",
                   loc="left", fontsize=args.fontsize - 1, color=INK_SOFT)
    zoom.set_title(f"first {args.zoom:g} s,  one marker per sample",
                   loc="left", fontsize=args.fontsize - 1, color=INK_SOFT)
    zoom.set_xlabel(r"$t$  [s]")
    # Fewer ticks on the window: the default set runs its last label into the
    # density panel beside it.
    zoom.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
    full.legend(loc="lower center", bbox_to_anchor=(0.5, 1.22), ncol=3,
                columnspacing=1.8, handlelength=2.4)

    # The law, seen side-on: the realization's histogram against the reference's
    # density.  Rotated to share the trace's axis, so a reader checks the two
    # against one scale.
    edges = np.linspace(-noise.bound, noise.bound, 121)
    dens.hist(omega, bins=edges, orientation="horizontal", density=True,
              color=DRAW, alpha=0.5, label="this draw")
    counts, _ = np.histogram(reference, bins=edges, density=True)
    centres = 0.5 * (edges[:-1] + edges[1:])
    dens.plot(counts, centres, color=REFERENCE, lw=1.4,
              label=f"law\n({args.reference_draws:.0e} draws)")
    dens.set_xlabel("density")
    dens.tick_params(labelleft=False)
    dens.legend(loc="upper right", fontsize=args.fontsize - 2.0)
    dens.set_title("shape", loc="left", fontsize=args.fontsize - 1, color=INK_SOFT)

    fig.tight_layout(rect=(0, 0, 1, 0.93), pad=0.4)

    parts = [f"sig{args.sigma:g}", f"dt{args.dt:g}"]
    if family != "gaussian":
        parts.insert(1, family)
    tag = "_".join(parts).replace(".", "p")
    out = args.out or (args.run / "figures" / f"noise_realization_{tag}.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=400)
    plt.close(fig)

    print(f"{noise!r}\n")
    print(f"realization : {n} samples, dt = {args.dt:g} s over [0, {horizon:g}] s"
          f"   (the data grid uses dt = {horizon / (config.data.n_samples - 1):g} s)")
    print(f"reference   : {args.reference_draws} independent draws of the same law\n")
    head = f"{'':>12}{'this draw':>14}{'law':>14}{'stated':>14}"
    print(head)
    print("-" * len(head))
    stated = {"mean": noise.mean, "sd": args.sigma, "max_abs": noise.bound}
    for key, name in (("mean", "mean"), ("sd", "std dev"),
                      ("skew", "skewness"), ("kurtosis", "kurtosis"),
                      ("max_abs", "max |omega|")):
        row = f"{name:>12}{got[key]:>14.6f}{want[key]:>14.6f}"
        row += f"{stated[key]:>14.6f}" if key in stated else f"{'--':>14}"
        print(row)
    inside = float(np.mean(np.abs(omega) <= noise.bound))
    print(f"\nwithin the stated support: {100 * inside:.4f}% of samples"
          + ("  (all)" if inside == 1.0 else "  <-- SOME ESCAPED"))
    print(f"\nwrote {out}\nwrote {out.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
