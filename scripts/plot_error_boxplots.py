#!/usr/bin/env python3
r"""Per-trajectory estimation error across the held-out split: clean vs noisy.

The Section 5 table reports one number per quantity --- a root-mean-square over
every test trajectory --- which says where the estimator sits but not how much
it varies from one trajectory to the next.  A single bad trajectory and a
uniformly mediocre fit can produce the same RMS.  This draws the whole
distribution instead: one box per quantity per noise condition, over the
:math:`P_{te}` held-out trajectories.

What one box contains
---------------------
For each held-out trajectory :math:`\ell`, one number per quantity:

.. math::

    e_{x_j}^\ell = \frac{100}{s_{x_j}} \|\hat{\mathbf X}^\ell_j - \mathbf X^\ell_j\|_{N,2},
    \qquad
    e_{\theta_j}^\ell = \frac{100}{s_{\theta_j}} |\hat\theta^\ell_j - \theta^\ell_j|,
    \qquad
    e_d^\ell = \frac{100}{G_q s_a} \|\hat{\mathbf d}^\ell - \mathbf d^\ell\|_{N,2}

i.e. exactly the summands of ``report_errors.py``, kept per trajectory rather
than pooled.  Taking the root mean square of a box's contents recovers that
script's number, so the two views are consistent by construction.

Why a single shared axis is legitimate here
-------------------------------------------
Each quantity is divided by its own reference scale --- the half-width of its
admissible interval for the states and parameters, :math:`G_q s_a` from Eq. (118)
for the disturbance --- so every box is a percentage of the set that quantity is
allowed to live in, and they are directly comparable.  Without that
normalization six quantities of different physical size would need six axes,
which is the one thing a chart must never do.

The comparison is paired
------------------------
Both conditions use the *same* held-out trajectories: the same initial
conditions, the same parameters, the same disturbance coefficients.  Only
:math:`\omega` differs.  So a box pair differs by the measurement noise alone,
and the median ratio printed over each pair is a like-for-like cost of that
noise.

Usage::

    python scripts/plot_error_boxplots.py --run outputs/big_w10_v2_s1 --sigma 0.02
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

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
from report_errors import basis_sup_norm  # noqa: E402

# The same identities as the estimate figure: orange is the estimate from a
# clean measurement, green the estimate from a noisy one.  Validated as a pair
# -- normal-vision dE 31.2, protan dE 8.0.  Protan sits at the floor, so the
# hatch on the noisy boxes is required secondary encoding rather than styling,
# and it is what also carries the distinction in greyscale.
CLEAN, NOISY = "#ff6c27", "#337b3e"
INK, INK_SOFT = "#0b0b0b", "#52514e"


def paper_style(base: float) -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": base,
        "axes.labelsize": base,
        "xtick.labelsize": base,
        "ytick.labelsize": base - 0.5,
        "legend.fontsize": base - 0.5,
        "axes.edgecolor": INK_SOFT, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK_SOFT, "ytick.color": INK_SOFT,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 0.0, "ytick.major.size": 2.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True,
        "grid.color": "#d8d8d4", "grid.linewidth": 0.4,
        "legend.frameon": False,
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "hatch.linewidth": 0.6,
    })


@torch.no_grad()
def errors(experiment, data, y) -> tuple[np.ndarray, list[str]]:
    """Per-trajectory normalized percentage errors, one column per quantity."""
    est = experiment.bank.estimate(y, data.t, experiment.t_coll)

    s_x, s_theta = experiment.system.reference_scales()
    bounds = experiment.coefficient_bounds.cpu()
    s_a = float(torch.linalg.vector_norm(0.5 * (bounds[:, 1] - bounds[:, 0])))
    g_q = basis_sup_norm(experiment.basis, experiment.val_data.t[-1].item())

    # ||.||_{N,2} is the root mean square over samples, Eq. (60).
    state = 100.0 * ((est.x - data.x) ** 2).mean(dim=1).sqrt() / s_x.to(est.x)
    theta = 100.0 * (est.theta - data.theta).abs() / s_theta.to(est.theta)
    dist = 100.0 * ((est.d - data.d) ** 2).mean(dim=1).sqrt() / (g_q * s_a)

    columns = torch.cat([state, theta, dist[:, None]], dim=1).cpu().numpy()
    n = state.shape[1]
    labels = ([f"$x_{{{j+1}}}$" for j in range(n)]
              + [rf"$\theta_{{{j+1}}}$" for j in range(theta.shape[1])]
              + ["$d$"])
    return columns, labels


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("outputs/big_w10_v2_s1"))
    parser.add_argument("--sigma", type=float, default=0.02,
                        help="noise level of the second condition")
    parser.add_argument("--family", default="gaussian")
    parser.add_argument("--noise-seed", type=int, default=909)
    parser.add_argument("--split", choices=("val", "train"), default="val")
    parser.add_argument("--linear", action="store_true",
                        help="linear y axis instead of logarithmic")
    parser.add_argument("--width", type=float, default=7.0)
    parser.add_argument("--height", type=float, default=3.9)
    parser.add_argument("--fontsize", type=float, default=9.0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    setup_logging("WARNING")
    paper_style(args.fontsize)

    config = RunConfig.from_yaml(args.run / "config.resolved.yaml")
    experiment = build_experiment(config)
    load_checkpoint(resolve_checkpoint(args.run), experiment.bank)
    experiment.bank.to(device=experiment.device, dtype=experiment.dtype)
    experiment.bank.eval()

    data = experiment.val_data if args.split == "val" else experiment.train_data

    # Same trajectories, two measurements: y = x_1, and y = x_1 + omega.
    noise = build_noise_model(
        args.family, args.sigma,
        truncation_sigmas=config.data.noise_truncation_sigmas,
    )
    omega = noise.sample(tuple(data.y.shape),
                         generator=make_generator(args.noise_seed),
                         dtype=experiment.dtype).to(data.y.device)
    clean_err, labels = errors(experiment, data, data.x[..., 0])
    noisy_err, _ = errors(experiment, data, data.x[..., 0] + omega)

    fig, ax = plt.subplots(figsize=(args.width, args.height))
    k = len(labels)
    offset, box_width = 0.2, 0.32
    for values, shift, colour, hatch, name in (
        (clean_err, -offset, CLEAN, None, "noise-free"),
        (noisy_err, +offset, NOISY, "///", f"$\\sigma$ = {args.sigma:g}"),
    ):
        bp = ax.boxplot(
            [values[:, j] for j in range(k)],
            positions=np.arange(k) + shift, widths=box_width,
            patch_artist=True, manage_ticks=False,
            medianprops=dict(color="white", lw=1.4),
            whiskerprops=dict(color=colour, lw=0.9),
            capprops=dict(color=colour, lw=0.9),
            # Outliers are the point of showing distributions, so they are drawn
            # -- small and unfilled, so a long tail reads as a tail rather than
            # as a second cloud of data.
            flierprops=dict(marker="o", ms=2.2, markerfacecolor="none",
                            markeredgecolor=colour, markeredgewidth=0.5,
                            alpha=0.55),
        )
        for patch in bp["boxes"]:
            patch.set(facecolor=colour, edgecolor=colour, linewidth=0.9,
                      alpha=0.55, hatch=hatch)
        # A surface-coloured proxy carries the legend so the hatch shows.
        ax.plot([], [], marker="s", ls="none", ms=7, markerfacecolor=colour,
                markeredgecolor=colour, alpha=0.55, label=name)

    if not args.linear:
        ax.set_yscale("log")
    # Headroom for the row of ratio labels, taken after the scale is set so the
    # log case grows by a factor rather than by an increment.
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi * 2.6 if not args.linear else hi + 0.22 * (hi - lo))

    # One direct label per pair: the median ratio is the quantity the figure is
    # for, and printing it beats leaving the reader to eyeball a log axis.  Held
    # at a fixed height in axes coordinates so it cannot collide with a whisker.
    label_y = ax.get_xaxis_transform()
    for j in range(k):
        ratio = np.median(noisy_err[:, j]) / np.median(clean_err[:, j])
        ax.annotate(f"{ratio:.2f}" + r"$\times$", xy=(j, 0.955),
                    xycoords=label_y, ha="center", va="center",
                    fontsize=args.fontsize - 1.0, color=INK_SOFT)
    ax.set_xticks(np.arange(k))
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.65, k - 0.35)
    ax.set_ylabel("error  [% of reference scale]")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2,
              columnspacing=2.4, handletextpad=0.4)
    fig.tight_layout(rect=(0, 0, 1, 0.97), pad=0.4)

    tag = f"sig{args.sigma:g}".replace(".", "p")
    out = args.out or (args.run / "figures" / f"error_boxplots_{tag}.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=400)
    plt.close(fig)

    print(f"{args.run.name}, {args.split} split, {len(data)} trajectories\n"
          f"noise-free vs {args.family} at sigma = {args.sigma:g}\n")
    head = (f"{'':>8}" + "".join(f"{lab:>12}" for lab in
                                 ("median", "q1", "q3", "max", "rms")))
    for name, values in (("noise-free", clean_err), (f"sigma={args.sigma:g}", noisy_err)):
        print(f"{name}   [% of reference scale]")
        print(f"{'':>8}{head[8:]}")
        for j, lab in enumerate(labels):
            v = values[:, j]
            plain = lab.replace("$", "").replace("\\", "").replace("{", "").replace("}", "")
            print(f"{plain:>8}" + "".join(f"{x:>12.4f}" for x in (
                np.median(v), np.percentile(v, 25), np.percentile(v, 75),
                v.max(), np.sqrt((v ** 2).mean()))))
        print()
    print(f"{'ratio':>8}" + "".join(
        f"{np.median(noisy_err[:, j]) / np.median(clean_err[:, j]):>12.3f}"
        for j in range(k)) + "   (medians)")
    print(f"\nwrote {out}\nwrote {out.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
