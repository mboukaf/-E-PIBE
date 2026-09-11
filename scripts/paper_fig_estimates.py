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

``noise-free``
    :math:`y = x_1` exactly.  This is the estimator's ceiling --- whatever error
    remains is approximation error in the decoder and the cell chain, not
    measurement noise.
``sigma = X``
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
noisy estimate.  Reviewers print in greyscale, where the indigo and the crimson
converge to L = 84 and 97 of 255, so the line style rather than the hue is what
separates the three series.

Colour
------
The two identity colours of the paper's own architecture diagram --- indigo for
the unitary estimation cell, crimson for the recursive bank --- so the two
figures read as one family.  Validated as a categorical pair: CVD ΔE 13.9
(protan), normal-vision ΔE 17.6, both above 3:1 against the surface.

The third series shares the crimson rather than taking a third hue.  That is
deliberate and was checked, not assumed: every third step available inside the
diagram's own two families fails the all-pairs checks --- a darker crimson at
normal-vision ΔE 13.4, a lighter one on chroma and contrast, the diagram's navy
at ΔE 7.7 against the indigo.  Hue therefore carries *identity* (truth versus
estimate) and dash carries the *condition* (noise-free versus noisy), which is
also what keeps the figure legible in greyscale.

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
from pibe.data.noise import build_noise_model  # noqa: E402
from pibe.experiment import build_experiment  # noqa: E402
from pibe.training.callbacks import load_checkpoint, resolve_checkpoint  # noqa: E402
from pibe.utils.logging import setup_logging  # noqa: E402
from pibe.utils.seeding import make_generator  # noqa: E402

# The architecture diagram's own two colours, read from its content stream:
# indigo for the estimation cell, crimson for the recursive bank.  Identity is
# fixed per entity -- the truth is always indigo and the estimate always
# crimson, here and in pibe.eval.figures.
TRUTH, ESTIMATE1,ESTIMATE2 = "#2c6ab1", "#ff6c27","#337b3e",
INK, INK_SOFT = "#0b0b0b", "#52514e"

# Dash patterns are load-bearing here, not decoration: they are what separates
# the three series in greyscale and what separates the two same-hue estimates in
# colour.  Kept far apart on purpose -- a long dash against a fine dot.
DASH_CLEAN = (0, (4.5, 2.2))
DASH_NOISY = (0, (1.3, 2.3))

# Where the noise costs little the two estimates coincide, and a dotted curve
# laid straight onto a dashed one of the same hue reads as a single accidental
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


def measure(data, config, experiment, sigma: float | None, seed: int):
    r"""The measurement :math:`y` fed to the bank, at the requested noise level.

    ``sigma = None`` returns the run's own data, i.e. the level it trained at;
    ``sigma = 0`` returns :math:`y = x_1` exactly.  Any other value re-measures
    the *same* trajectories, so two calls differ by :math:`\omega` alone.
    """
    if sigma is None:
        return data.y, config.data.noise_sigma
    if sigma > 0:
        noise = build_noise_model(
            config.data.noise_family, sigma,
            truncation_sigmas=config.data.noise_truncation_sigmas,
        )
        omega = noise.sample(tuple(data.y.shape),
                             generator=make_generator(seed),
                             dtype=experiment.dtype).to(data.y.device)
    else:
        omega = torch.zeros_like(data.y)
    return data.x[..., 0] + omega, sigma


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

    noisy, sigma = measure(data, config, experiment, args.sigma, args.noise_seed)
    with torch.no_grad():
        est_noisy = experiment.bank.estimate(noisy, data.t, experiment.t_coll)
        if args.noise_free:
            clean, _ = measure(data, config, experiment, 0.0, args.noise_seed)
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
    # (legend label, plain label for the console, dash, linewidth, z-order)
    series = []
    if args.noise_free:
        series.append(("noise-free", "noise-free", DASH_CLEAN, 1.35, 3))
        series.append((f"$\\sigma$ = {sigma:g}", f"sigma={sigma:g}",
                       DASH_NOISY, 1.7, 4))
    else:
        series.append(("estimated", "estimated", DASH_CLEAN, 1.4, 4))
    estimates = ([est_clean] if args.noise_free else []) + [est_noisy]

    fig, axes = plt.subplots(3, 1, figsize=(args.width, args.height), sharex=True,
                             gridspec_kw={"hspace": 0.16})
    for j, ax in enumerate(axes):
        ax.plot(t, truths[j], color=TRUTH, lw=1.6, zorder=2,
                label="true" if j == 0 else None)
        for est, (name, _, dash, lw, z) in zip(estimates, series):
            ax.plot(t, pick(est, j), color=ESTIMATE, ls=dash, lw=lw, zorder=z,
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
        tag = f"sig{sigma:g}_traj{index}".replace(".", "p")
        out = args.run / "figures" / f"{stem}_{tag}.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)                                  # vector, for the paper
    fig.savefig(out.with_suffix(".png"), dpi=400)     # raster, for quick looks
    plt.close(fig)

    print(f"run        : {args.run.name}, held-out trajectory {index}")
    print(f"noisy      : sigma = {sigma:g}"
          + ("  (as trained)" if args.sigma is None
             else f"  (re-measured; trained at {config.data.noise_sigma:g})"))
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
