#!/usr/bin/env python3
r"""Paper figure: the unmeasured states and the disturbance, estimated from one
noisy scalar measurement.

Layout
------
Four stacked panels sharing a time axis.  The top strip is the measurement
:math:`y = x_1 + \omega` --- the only thing the estimator sees --- drawn against
the true :math:`x_1`.  Below it, the three quantities that are never measured and
must be inferred: :math:`x_2`, :math:`x_3` and :math:`d(t)`.

The measurement is a separate panel rather than an overlay on purpose.
:math:`y` spans about 1.0 while :math:`x_2` spans about 0.15, so putting them on
one frame would need two y-scales, and a dual-axis plot lets the reader infer a
relationship from a choice of scaling that the data does not support.  Small
multiples say the same thing without that risk, and the shared x-axis keeps the
comparison across time exact.

Print conventions
-----------------
Vector PDF alongside the PNG, since journals rasterize anything else badly.
Serif type at paper size so the figure matches the body text rather than
shouting.  No figure title --- the caption carries it in LaTeX.

Distinguishable without colour: the truth is solid, the estimate dashed, the
measurement a thin solid line.  Reviewers print in greyscale, and the blue and
orange used here converge to similar greys, so the dash pattern rather than the
hue is what separates the two curves that matter.

Colour
------
Slots 1-3 of the validated categorical palette (blue / orange / aqua).  Checked
with the all-pairs validator, as small multiples require: worst CVD ΔE 9.2,
worst normal-vision ΔE 24.0, both clear.  The aqua sits at 2.74:1 against the
surface, below the 3:1 bar, so the measurement carries a direct in-panel label
rather than relying on the legend to identify it.

Usage::

    python scripts/paper_fig_estimates.py --run outputs/big_w10_v2_s1
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
import matplotlib.ticker  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from pibe.config import RunConfig  # noqa: E402
from pibe.data.noise import build_noise_model  # noqa: E402
from pibe.experiment import build_experiment  # noqa: E402
from pibe.training.callbacks import load_checkpoint, resolve_checkpoint  # noqa: E402
from pibe.utils.logging import setup_logging  # noqa: E402
from pibe.utils.seeding import make_generator  # noqa: E402

# Validated categorical slots 1-3.  Identity is fixed per entity and never
# reassigned, so the truth is always blue and the estimate always orange.
TRUTH, ESTIMATE, MEASURED = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_SOFT = "#0b0b0b", "#52514e"


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
        "axes.titlesize": base,
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
        "pdf.fonttype": 42,   # embed as TrueType, so the PDF is editable/searchable
        "ps.fonttype": 42,
    })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("outputs/big_w10_v2_s1"))
    parser.add_argument("--trajectory", type=int, default=0,
                        help="index into the held-out split")
    parser.add_argument("--width", type=float, default=7.0,
                        help="inches; 7.0 is Elsevier full width, 3.4 single column")
    parser.add_argument("--height", type=float, default=6.0)
    parser.add_argument("--fontsize", type=float, default=9.0)
    parser.add_argument("--sans", action="store_true", help="sans-serif instead of serif")
    parser.add_argument("--sigma", type=float, default=None,
                        help="measurement noise std to draw the figure at. The "
                             "same held-out trajectory is kept and only the "
                             "measurement is re-drawn, so the panels isolate the "
                             "effect of the noise. 0 is noise-free. Omit to use "
                             "the level the run was trained at.")
    parser.add_argument("--noise-seed", type=int, default=909)
    parser.add_argument("--zoom", type=float, nargs=2, default=(6.0, 7.5),
                        help="window shown in the measurement inset")
    parser.add_argument("--annotate", action="store_true",
                        help="print each panel's RMSE on the panel")
    parser.add_argument("--out", type=Path, default=None,
                        help="defaults to <run>/figures/paper_estimates.pdf")
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

    # Re-measuring keeps the states, parameters and disturbance of the held-out
    # trajectory fixed and changes only omega, so a figure at a different sigma
    # differs from this one by the noise alone.  The estimate is recomputed from
    # the new measurement -- the model is never retrained, so what the panels
    # show is how the trained estimator behaves on a noisier sensor.
    if args.sigma is None:
        measured = data.y
        sigma = config.data.noise_sigma
    else:
        if args.sigma > 0:
            noise = build_noise_model(
                config.data.noise_family, args.sigma,
                truncation_sigmas=config.data.noise_truncation_sigmas,
            )
            omega = noise.sample(tuple(data.y.shape),
                                 generator=make_generator(args.noise_seed),
                                 dtype=experiment.dtype).to(data.y.device)
        else:
            omega = torch.zeros_like(data.y)
        measured = data.x[..., 0] + omega
        sigma = args.sigma

    with torch.no_grad():
        estimates = experiment.bank.estimate(measured, data.t, experiment.t_coll)

    t = data.t.cpu().numpy()
    x_true = data.x[index].cpu().numpy()
    x_est = estimates.x[index].cpu().numpy()
    y = measured[index].cpu().numpy()
    d_true = data.d[index].cpu().numpy()
    d_est = estimates.d[index].cpu().numpy()

    fig, axes = plt.subplots(
        4, 1, figsize=(args.width, args.height), sharex=True,
        gridspec_kw={"height_ratios": [0.95, 1.0, 1.0, 1.0], "hspace": 0.20},
    )

    # --- the measurement: subordinate, with a zoom that makes it visible ---
    #
    # At the trained noise level sigma is a fraction of a percent of the signal
    # range, so on the full horizon y lies exactly under x_1 and the reader sees
    # one curve.  That is the honest picture and it is worth stating, but a panel
    # whose second series is invisible would advertise something the figure does
    # not show.  The inset resolves it: a short window where the individual
    # samples and the noise on them are separable.
    ax = axes[0]
    ax.plot(t, y, color=MEASURED, lw=0.8, zorder=2)
    ax.plot(t, x_true[:, 0], color=TRUTH, lw=1.5, zorder=3)
    ax.set_ylabel(r"$y,\ x_1$")

    lo, hi = args.zoom
    window = (t >= lo) & (t <= hi)
    inset = ax.inset_axes([0.545, 0.08, 0.33, 0.40])
    inset.plot(t[window], y[window], "-o", color=MEASURED, lw=0.8, ms=2.2,
               zorder=2)
    inset.plot(t[window], x_true[window, 0], color=TRUTH, lw=1.4, zorder=3)
    inset.tick_params(labelsize=args.fontsize - 2.5, length=2, pad=1.5)
    # Three ticks: more than that and the last label runs off the inset.
    inset.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
    inset.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
    inset.grid(alpha=0.5, lw=0.3)
    for side in ("top", "right"):
        inset.spines[side].set_visible(False)
    inset.annotate(f"${lo:g}\\!-\\!{hi:g}$ s", xy=(0.04, 0.80),
                   xycoords="axes fraction", color=INK_SOFT,
                   fontsize=args.fontsize - 2.5)
    # The rectangle alone.  Leader lines from an inset this small cross the
    # curve they are meant to annotate, which costs more than it explains.
    rect, connectors = ax.indicate_inset_zoom(inset, edgecolor=INK_SOFT,
                                              lw=0.5, alpha=0.5)
    for line in connectors:
        line.set_visible(False)
    # Contrast relief for the aqua: name the series in place, not by colour alone.
    # Top-left: the only region of this panel the rising curve leaves empty.
    ax.annotate(r"$y = x_1 + \omega$", xy=(0.015, 0.86),
                xycoords="axes fraction", color=INK_SOFT,
                fontsize=args.fontsize - 1.0)

    # --- the three inferred quantities ---------------------------------
    panels = [
        (axes[1], x_true[:, 1], x_est[:, 1], r"$x_2$"),
        (axes[2], x_true[:, 2], x_est[:, 2], r"$x_3$"),
        (axes[3], d_true, d_est, r"$d$"),
    ]
    for ax, truth, est, label in panels:
        ax.plot(t, truth, color=TRUTH, lw=1.5, zorder=3, label="true")
        ax.plot(t, est, color=ESTIMATE, lw=1.4, ls=(0, (4.5, 2.2)), zorder=4,
                label="estimated")
        ax.set_ylabel(label)
        if args.annotate:
            rmse = float(np.sqrt(np.mean((est - truth) ** 2)))
            ax.annotate(f"RMSE {rmse:.2e}", xy=(0.995, 0.06),
                        xycoords="axes fraction", ha="right",
                        color=INK_SOFT, fontsize=args.fontsize - 1.5)

    axes[-1].set_xlabel(r"$t$  [s]")
    axes[-1].set_xlim(t[0], t[-1])

    handles = [
        plt.Line2D([], [], color=TRUTH, lw=1.5, label="true"),
        plt.Line2D([], [], color=ESTIMATE, lw=1.4, ls=(0, (4.5, 2.2)),
                   label="estimated"),
        plt.Line2D([], [], color=MEASURED, lw=1.3, marker="o", ms=2.6,
                   label=r"measured $y$"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.002), columnspacing=2.2, handlelength=2.6)

    fig.tight_layout(rect=(0, 0, 1, 0.99), pad=0.4)

    if args.out is not None:
        out = args.out
    elif args.sigma is None:
        out = args.run / "figures" / "paper_estimates.pdf"
    else:
        tag = f"sig{sigma:g}_traj{index}".replace(".", "p")
        out = args.run / "figures" / f"paper_estimates_{tag}.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)                                  # vector, for the paper
    fig.savefig(out.with_suffix(".png"), dpi=400)     # raster, for quick looks
    plt.close(fig)

    print(f"run        : {args.run.name}, held-out trajectory {index}")
    print(f"noise      : sigma = {sigma:g}"
          + ("  (as trained)" if args.sigma is None else "  (re-measured; trained at %g)" % config.data.noise_sigma))
    for label, truth, est in (("x2", x_true[:, 1], x_est[:, 1]),
                              ("x3", x_true[:, 2], x_est[:, 2]),
                              ("d ", d_true, d_est)):
        print(f"  {label} RMSE  {np.sqrt(np.mean((est - truth) ** 2)):.4e}")
    print(f"\nwrote {out}\nwrote {out.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
