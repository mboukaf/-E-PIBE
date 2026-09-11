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

Print conventions
-----------------
Vector PDF alongside the PNG, since journals rasterize anything else badly.
Serif type at paper size so the figure matches the body text.  No figure title
--- the caption carries it in LaTeX.

Distinguishable without colour: the truth is solid, the estimate dashed.
Reviewers print in greyscale, and the blue and orange used here converge to
similar greys, so the dash pattern rather than the hue is what separates them.

Colour
------
Slots 1-2 of the validated categorical palette (blue / orange), checked with the
all-pairs validator as small multiples require: CVD ΔE 12.9, normal-vision
ΔE 30.9, both clear of their floors, and both above 3:1 against the surface.

Usage::

    python scripts/paper_fig_estimates.py --run outputs/big_w10_v2_s1
    python scripts/paper_fig_estimates.py --sigma 0.05 --trajectory 124
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
from pibe.data.noise import build_noise_model  # noqa: E402
from pibe.experiment import build_experiment  # noqa: E402
from pibe.training.callbacks import load_checkpoint, resolve_checkpoint  # noqa: E402
from pibe.utils.logging import setup_logging  # noqa: E402
from pibe.utils.seeding import make_generator  # noqa: E402

# Validated categorical slots 1-2.  Identity is fixed per entity: the truth is
# always blue and the estimate always orange, in every figure in this project.
TRUTH, ESTIMATE = "#2a78d6", "#eb6834"
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("outputs/big_w10_v2_s1"))
    parser.add_argument("--trajectory", type=int, default=0,
                        help="index into the held-out split")
    parser.add_argument("--sigma", type=float, default=None,
                        help="measurement noise std to draw the figure at. The "
                             "same held-out trajectory is kept and only the "
                             "measurement is re-drawn, so the panels isolate the "
                             "effect of the noise. 0 is noise-free. Omit to use "
                             "the level the run was trained at.")
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
    panels = [
        (data.x[index, :, 1].cpu().numpy(), estimates.x[index, :, 1].cpu().numpy(),
         r"$x_2$", "x2"),
        (data.x[index, :, 2].cpu().numpy(), estimates.x[index, :, 2].cpu().numpy(),
         r"$x_3$", "x3"),
        (data.d[index].cpu().numpy(), estimates.d[index].cpu().numpy(),
         r"$d$", "d "),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(args.width, args.height), sharex=True,
                             gridspec_kw={"hspace": 0.16})
    for ax, (truth, est, label, _) in zip(axes, panels):
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
    axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2,
                   columnspacing=2.4, handlelength=2.6)
    fig.tight_layout(rect=(0, 0, 1, 0.965), pad=0.4)

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
          + ("  (as trained)" if args.sigma is None
             else f"  (re-measured; trained at {config.data.noise_sigma:g})"))
    for truth, est, _, name in panels:
        print(f"  {name} RMSE  {np.sqrt(np.mean((est - truth) ** 2)):.4e}")
    print(f"\nwrote {out}\nwrote {out.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
