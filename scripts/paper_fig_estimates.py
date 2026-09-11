#!/usr/bin/env python3
r"""Paper figure: the unmeasured states and the disturbance, estimated from one
noisy scalar measurement.

Three stacked panels sharing a time axis --- :math:`x_2`, :math:`x_3`,
:math:`d(t)` --- each showing the truth against the estimate.  None of the three
is measured: the estimator sees only :math:`y = x_1 + \omega`, which is what the
panels are reconstructed from but is not itself drawn.  Its scale is an order of
magnitude above :math:`x_2`, so sharing a frame with it would need two y-scales,
and at the trained noise level it is visually indistinguishable from :math:`x_1`
anyway.

The two estimates
-----------------
By default each panel carries *two* estimates of the same quantity, from the same
trained model, differing only in what it was fed:

``noise-free`` (orange, dashed)
    :math:`y = x_1` exactly.  This is the estimator's ceiling --- whatever error
    remains is approximation error in the decoder and the cell chain, not
    measurement noise.
``sigma = X`` (green, dotted)
    the same held-out trajectory re-measured at the chosen level.  The states,
    parameters and disturbance coefficients are untouched, so the gap between
    the two dashed curves is caused by :math:`\omega` and nothing else.

Reading the two together separates the two error sources that a single curve
conflates.  ``--no-noise-free`` drops back to one estimate per panel.

Print conventions
-----------------
Vector PDF alongside the PNG, since journals rasterize anything else badly.
Serif type at paper size so the figure matches the body text.  No figure title
--- the caption carries it in LaTeX.

Distinguishable without colour: solid truth, dashed noise-free estimate, dotted
noisy estimate.  Reviewers print in greyscale, where the blue and the green
converge to L = 98 and 103 of 255, so the line style rather than the hue is what
separates those two.

Colour
------
One hue per series --- blue truth, orange noise-free estimate, green noisy
estimate --- checked with the all-pairs validator, as three curves sharing one
frame require.  Normal-vision separation is comfortable (worst pair ΔE 19.9);
the binding constraint is protan vision, where the orange and the green fall to
ΔE 8.0.  That sits at the floor rather than above it, so the dash patterns are
not decoration: they are the secondary encoding that the floor requires.  The
orange also reads at 2.75:1 against white rather than 3:1, relieved by the
legend.

Usage::

    python scripts/paper_fig_estimates.py --run outputs/big_w10_v2_s1 --sigma 0.05
    python scripts/paper_fig_estimates.py --sigma 0.1 --trajectory 124
    python scripts/paper_fig_estimates.py --no-noise-free          # one estimate
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
import matplotlib.patheffects as pe  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from pibe.config import RunConfig  # noqa: E402
from pibe.data.noise import NOISE_FAMILIES, build_noise_model  # noqa: E402
from pibe.experiment import build_experiment  # noqa: E402
from pibe.training.callbacks import load_checkpoint, resolve_checkpoint  # noqa: E402
from pibe.utils.logging import setup_logging  # noqa: E402
from pibe.utils.seeding import make_generator  # noqa: E402

# Identity is fixed per series: blue is always the truth, orange always the
# estimate from a clean measurement, green always the estimate from a noisy one.
# Validated as a categorical triple -- worst all-pairs normal-vision dE 19.9,
# worst protan dE 8.0 (orange against green), which is at the floor and so
# obliges the dash patterns below as secondary encoding.
TRUTH, ESTIMATE1,ESTIMATE2 = "#2c6ab1", "#ff6c27","#337b3e",
INK, INK_SOFT = "#0b0b0b", "#52514e"

# Dash patterns are load-bearing here, not decoration: in greyscale the blue and
# the green converge (L 98 against 103 of 255), and in protan vision the orange
# and the green sit at the separation floor.  Kept far apart on purpose -- a long
# dash against a fine dot.
DASH_CLEAN = (0, (4.5, 2.2))
DASH_NOISY = (0, (1.3, 2.3))

# Where the noise costs little the two estimates coincide almost exactly, and a
# dotted curve laid straight onto a dashed one reads as a single accidental
# dash-dot line.  A thin surface-coloured ring under the top curve keeps the two
# separable exactly where they overlap, which is where it matters.
HALO = [pe.withStroke(linewidth=3.4, foreground="white")]


def paper_style(base: float, serif: bool) -> None:
    """Type and chrome at journal size, with the grid pushed into the background."""
    family = (["DejaVu Serif", "Times New Roman", "serif"] if serif
              else ["DejaVu Sans", "Helvetica", "sans-serif"])
    plt.rcParams.update({
        "font.family": "serif" if serif else "sans-serif",
        "font.serif" if serif else "font.sans-serif": family,
        "mathtext.fontset": "dejavuserif" if serif else "dejavusans",
        "font.size": base,
        "axes.labelsize": base,
        "xtick.labelsize": base - 0.5,
        "ytick.labelsize": base - 0.5,
        "legend.fontsize": base - 0.5,
        "axes.edgecolor": INK_SOFT,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK_SOFT,
        "ytick.color": INK_SOFT,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#d8d8d4",
        "grid.linewidth": 0.4,
        "grid.alpha": 0.9,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,   # embed as TrueType, so the PDF stays searchable
        "ps.fonttype": 42,
    })


def noise_arguments(pairs: list[str]) -> dict:
    """``["weight=0.4", "separation=3"]`` -> ``{"weight": 0.4, "separation": 3.0}``."""
    kwargs = {}
    for item in pairs:
        if "=" not in item:
            raise SystemExit(f"--noise-arg expects KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        kwargs[key.strip()] = float(value)
    return kwargs


def measure(data, config, experiment, sigma: float | None, seed: int,
            family: str | None = None, noise_kwargs: dict | None = None,
            bias: float = 0.0):
    r"""The measurement :math:`y` fed to the bank, and the law that produced it.

    ``sigma = None`` returns the run's own data, i.e. the level it trained at;
    ``sigma = 0`` returns :math:`y = x_1` exactly.  Any other value re-measures
    the *same* trajectories, so two calls differ by :math:`\omega` alone.

    Returns ``(y, sigma, noise)``, with ``noise`` the law itself so the caller
    can report its realized mean and support rather than restating its name.
    """
    if sigma is None:
        return data.y, config.data.noise_sigma, None
    if sigma <= 0 and not bias:
        return data.x[..., 0], 0.0, None
    noise = build_noise_model(
        family or config.data.noise_family, sigma,
        bias=bias, bias_relative=True,
        **({"truncation_sigmas": config.data.noise_truncation_sigmas}
           if not (noise_kwargs or {}).get("truncation_sigmas") else {}),
        **(noise_kwargs or {}),
    )
    omega = noise.sample(tuple(data.y.shape),
                         generator=make_generator(seed),
                         dtype=experiment.dtype).to(data.y.device)
    return data.x[..., 0] + omega, sigma, noise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("outputs/big_w10_v2_s1"))
    parser.add_argument("--trajectory", type=int, default=0,
                        help="index into the held-out split")
    parser.add_argument("--sigma", type=float, default=None,
                        help="measurement noise std for the noisy estimate. The "
                             "same held-out trajectory is kept and only the "
                             "measurement is re-drawn, so the two estimates "
                             "differ by the noise alone. Omit to use the level "
                             "the run was trained at.")
    parser.add_argument("--noise-family", default=None,
                        choices=sorted(NOISE_FAMILIES),
                        help="law for the noisy measurement. 'mixture' is an "
                             "asymmetric mixture of two Gaussians and is the "
                             "only family here whose mean is not zero. Defaults "
                             "to the family the run trained on.")
    parser.add_argument("--noise-arg", action="append", default=[],
                        metavar="KEY=VALUE",
                        help="parameter for that family, repeatable. For "
                             "'mixture': weight, separation, outlier_scale, "
                             "truncation_sigmas.")
    parser.add_argument("--bias", type=float, default=0.0,
                        help="additive offset in units of sigma, applied on top "
                             "of the family. Shifts a symmetric law bodily, "
                             "which is a different thing from the mixture's own "
                             "asymmetry.")
    parser.add_argument("--no-noise-free", dest="noise_free", action="store_false",
                        help="draw only the noisy estimate, not the sigma = 0 "
                             "reference")
    parser.add_argument("--noise-seed", type=int, default=909)
    parser.add_argument("--width", type=float, default=7.0,
                        help="inches; 7.0 is Elsevier full width, 3.4 single column")
    parser.add_argument("--height", type=float, default=4.8)
    parser.add_argument("--fontsize", type=float, default=9.0)
    parser.add_argument("--sans", action="store_true",
                        help="sans-serif instead of serif")
    parser.add_argument("--annotate", action="store_true",
                        help="print each panel's RMSE on the panel")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    setup_logging("WARNING")
    paper_style(args.fontsize, serif=not args.sans)

    config = RunConfig.from_yaml(args.run / "config.resolved.yaml")
    experiment = build_experiment(config)
    load_checkpoint(resolve_checkpoint(args.run), experiment.bank)
    experiment.bank.to(device=experiment.device, dtype=experiment.dtype)
    experiment.bank.eval()

    data = experiment.val_data
    index = min(args.trajectory, len(data) - 1)

    noisy, sigma, law = measure(data, config, experiment, args.sigma,
                                args.noise_seed, args.noise_family,
                                noise_arguments(args.noise_arg), args.bias)
    with torch.no_grad():
        est_noisy = experiment.bank.estimate(noisy, data.t, experiment.t_coll)
        if args.noise_free:
            clean, *_ = measure(data, config, experiment, 0.0, args.noise_seed)
            est_clean = experiment.bank.estimate(clean, data.t, experiment.t_coll)

    def pick(est, j):
        """Panel j of an estimate: the two unmeasured states, then d."""
        return (est.d[index] if j == 2 else est.x[index, :, j + 1]).cpu().numpy()

    t = data.t.cpu().numpy()
    truths = [data.x[index, :, 1].cpu().numpy(),
              data.x[index, :, 2].cpu().numpy(),
              data.d[index].cpu().numpy()]
    labels = [r"$x_2$", r"$x_3$", r"$d$"]
    names = ["x2", "x3", "d "]

    # Series drawn back to front: the noisy estimate sits on top, since it is the
    # curve the figure is about.
    # (legend label, plain label for the console, colour, dash, linewidth, z-order)
    series = []
    if args.noise_free:
        series.append(("noise-free", "noise-free", ESTIMATE1, DASH_CLEAN, 1.45, 3))
        family = args.noise_family or config.data.noise_family
        # Name the law in the legend whenever it is not the plain zero-mean
        # Gaussian: at a matched sigma the shape is the only thing that differs,
        # so a figure that does not say which shape cannot be read.
        tag = (f"$\\sigma$ = {sigma:g}" if family == "gaussian" and not args.bias
               else f"{family}, $\\sigma$ = {sigma:g}")
        series.append((tag, f"sigma={sigma:g}", ESTIMATE2, DASH_CLEAN, 1.7, 4))
    else:
        series.append(("estimated", "estimated", ESTIMATE1, DASH_CLEAN, 1.4, 4))
    estimates = ([est_clean] if args.noise_free else []) + [est_noisy]

    fig, axes = plt.subplots(3, 1, figsize=(args.width, args.height), sharex=True,
                             gridspec_kw={"hspace": 0.16})
    for j, ax in enumerate(axes):
        ax.plot(t, truths[j], color=TRUTH, lw=2, zorder=2,
                label="true" if j == 0 else None)
        for est, (name, _, colour, dash, lw, z) in zip(estimates, series):
            ax.plot(t, pick(est, j), color=colour, ls=dash, lw=2, zorder=z,
                    label=name if j == 0 else None,
                    path_effects=HALO if z == 4 and len(series) > 1 else None)
        ax.set_ylabel(labels[j])
        if args.annotate:
            text = "   ".join(
                f"{name}  {np.sqrt(np.mean((pick(est, j) - truths[j]) ** 2)):.1e}"
                for est, (name, *_) in zip(estimates, series))
            ax.annotate(f"RMSE  {text}", xy=(0.995, 0.06),
                        xycoords="axes fraction", ha="right",
                        color=INK_SOFT, fontsize=args.fontsize - 1.5)

    axes[-1].set_xlabel(r"$t$  [s]")
    axes[-1].set_xlim(t[0], t[-1])
    axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.02),
                   ncol=1 + len(series), columnspacing=2.0, handlelength=2.8)
    fig.tight_layout(rect=(0, 0, 1, 0.965), pad=0.4)

    stem = "paper_estimates" + ("_cmp" if args.noise_free else "")
    if args.out is not None:
        out = args.out
    elif args.sigma is None:
        out = args.run / "figures" / f"{stem}.pdf"
    else:
        family = args.noise_family or config.data.noise_family
        # The family and the bias go in the name: otherwise two runs at the same
        # sigma but different laws overwrite each other silently.
        parts = [f"sig{sigma:g}"]
        if family != "gaussian":
            parts.append(family)
        if args.bias:
            parts.append(f"bias{args.bias:g}")
        parts.append(f"traj{index}")
        tag = "_".join(parts).replace(".", "p")
        out = args.run / "figures" / f"{stem}_{tag}.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)                                  # vector, for the paper
    fig.savefig(out.with_suffix(".png"), dpi=400)     # raster, for quick looks
    plt.close(fig)

    print(f"run        : {args.run.name}, held-out trajectory {index}")
    print(f"noisy      : sigma = {sigma:g}"
          + ("  (as trained)" if args.sigma is None
             else f"  (re-measured; trained at {config.data.noise_sigma:g})"))
    if law is not None:
        print(f"noise law  : {law!r}")
        print(f"             mean = {law.mean:+.5g}"
              f"  ({100 * law.mean / sigma:+.1f}% of sigma)"
              + ("  -- zero-mean, Proposition 1 applies"
                 if abs(law.mean) < 1e-12 else
                 "  -- NOT zero-mean, so Proposition 1's premise fails and the"
                 " offset propagates through the chain"))
    head = f"{'':>5}" + "".join(f"{plain:>14}" for _, plain, *_ in series)
    print("\nRMSE" + (" (ratio = noisy / noise-free)" if args.noise_free else "")
          + "\n" + head + (f"{'ratio':>9}" if args.noise_free else ""))
    print("-" * (len(head) + (9 if args.noise_free else 0)))
    for j, name in enumerate(names):
        errs = [float(np.sqrt(np.mean((pick(est, j) - truths[j]) ** 2)))
                for est in estimates]
        row = f"{name:>5}" + "".join(f"{e:>14.4e}" for e in errs)
        if args.noise_free:
            row += f"{errs[-1] / errs[0]:>9.2f}"
        print(row)
    print(f"\nwrote {out}\nwrote {out.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
