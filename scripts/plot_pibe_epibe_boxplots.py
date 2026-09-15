#!/usr/bin/env python3
r"""PIBE against EPIBE, per trajectory: noise-free, Gaussian noise, biased sensor.

Three panels --- states, parameters, disturbance --- each with three measurement
conditions side by side, and in each condition one box for PIBE and one for EPIBE
over the held-out trajectories.  A trajectory's state error is the root mean
square of its three coordinate errors, its parameter error that of its two
parameter errors (each coordinate in % of its own reference scale).
``--per-quantity`` draws the six individual panels instead.

The conditions
--------------
``noise-free``
    :math:`y = x_1` exactly, fed to the banks trained on unbiased noisy data.
    What each estimator does with a perfect measurement.
``Gaussian``
    The held-out measurement those banks were trained for,
    :math:`\sigma = 0.05`, zero mean.
``bias``
    :math:`\sigma = 0.05` plus a constant sensor offset, estimated by the banks
    trained on that data.  EPIBE is the amortized-location bank of
    ``Docs/EPIBE.md`` §10, so its estimate subtracts the offset it selected.

All three conditions use the *same* held-out trajectories --- the datasets share
their seed, so states, parameters and disturbance are identical and only the
measurement differs.  Each box pair is therefore paired, and the median ratio
printed above it is a like-for-like comparison.

Errors are the per-trajectory percentages of ``plot_error_boxplots.py``: the
normalized :math:`L^2` error of Eq. (60) divided by each quantity's reference
scale.  Each panel keeps its own logarithmic axis.

Usage::

    python scripts/plot_pibe_epibe_boxplots.py
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
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from pibe.config import RunConfig  # noqa: E402
from pibe.experiment import build_experiment  # noqa: E402
from pibe.training.callbacks import load_checkpoint, resolve_checkpoint  # noqa: E402
from pibe.utils.logging import setup_logging  # noqa: E402
from plot_error_boxplots import INK, INK_SOFT, errors, paper_style  # noqa: E402

# Identity per estimator, validated as a pair on the light surface: normal-vision
# dE 31.2, protan dE 8.0 -- at the floor, so the hatch on the PIBE boxes is the
# required secondary encoding.  The orange is below 3:1 against white; the legend
# and the printed ratios carry it.
PIBE_C, EPIBE_C = "#337b3e", "#ff6c27"


def group(err: np.ndarray, n_x: int, n_theta: int) -> np.ndarray:
    r"""Collapse per-quantity errors into states, parameters and disturbance.

    Each group is the root mean square of its members' percentages, so a group
    value is the typical coordinate error in % of that coordinate's own
    admissible scale; the disturbance column is kept as it is.
    """
    states = np.sqrt((err[:, :n_x] ** 2).mean(axis=1))
    params = np.sqrt((err[:, n_x:n_x + n_theta] ** 2).mean(axis=1))
    return np.stack([states, params, err[:, n_x + n_theta]], axis=1)


def load(run: Path):
    config = RunConfig.from_yaml(run / "config.resolved.yaml")
    experiment = build_experiment(config)
    load_checkpoint(resolve_checkpoint(run), experiment.bank)
    experiment.bank.to(device=experiment.device, dtype=experiment.dtype)
    experiment.bank.eval()
    return experiment


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pibe", type=Path, default=Path("outputs/biasid_pibe_b0"),
                        help="PIBE trained on unbiased noisy data")
    parser.add_argument("--epibe", type=Path, default=Path("outputs/biasid_epibe_b0"),
                        help="EPIBE trained on unbiased noisy data")
    parser.add_argument("--pibe-bias", type=Path, default=Path("outputs/biasid_pibe_b0.5"))
    parser.add_argument("--epibe-bias", type=Path, default=Path("outputs/biasid_epibe_b0.5"))
    parser.add_argument("--per-quantity", action="store_true",
                        help="one panel per state, parameter and d instead of three global panels")
    parser.add_argument("--width", type=float, default=7.0)
    parser.add_argument("--height", type=float, default=None)
    parser.add_argument("--fontsize", type=float, default=9.0)
    parser.add_argument("--out", type=Path,
                        default=None)
    args = parser.parse_args()
    if args.height is None:
        args.height = 4.6 if args.per_quantity else 2.9
    if args.out is None:
        name = "pibe_vs_epibe_boxplots" + ("_per_quantity" if args.per_quantity else "")
        args.out = Path(f"outputs/biasid_epibe_b0.5/figures/{name}.pdf")
    setup_logging("WARNING")
    paper_style(args.fontsize)

    banks = {name: load(path) for name, path in (
        ("pibe", args.pibe), ("epibe", args.epibe),
        ("pibe_bias", args.pibe_bias), ("epibe_bias", args.epibe_bias))}

    clean_data = banks["pibe"].val_data
    bias_data = banks["pibe_bias"].val_data
    bias = banks["pibe_bias"].config.data.noise_bias
    sigma = banks["pibe"].config.data.noise_sigma
    # The comparison is only paired if every condition sees the same trajectories.
    for other in ("epibe", "pibe_bias", "epibe_bias"):
        if not torch.equal(banks[other].val_data.x, clean_data.x):
            raise SystemExit(f"{other}: held-out trajectories differ; the comparison would not be paired")

    conditions = [
        ("noise-free", clean_data.x[..., 0], clean_data, "pibe", "epibe"),
        ("Gaussian", clean_data.y, clean_data, "pibe", "epibe"),
        (f"bias {bias:g}", bias_data.y, bias_data, "pibe_bias", "epibe_bias"),
    ]
    table = []
    labels = None
    for name, y, data, p, e in conditions:
        pibe_err, labels = errors(banks[p], data, y)
        epibe_err, _ = errors(banks[e], data, y)
        table.append((name, pibe_err, epibe_err))

    if not args.per_quantity:
        n_x = banks["pibe"].system.n
        n_th = banks["pibe"].system.theta_dim
        table = [(name, group(pe, n_x, n_th), group(ee, n_x, n_th)) for name, pe, ee in table]
        labels = ["states", "parameters", "disturbance"]

    rows, cols = (2, 3) if args.per_quantity else (1, 3)
    fig, axes = plt.subplots(rows, cols, figsize=(args.width, args.height), squeeze=False)
    offset, width = 0.19, 0.32
    rng = np.random.default_rng(0)
    for j, ax in enumerate(axes.flat):
        for c, (name, pibe_err, epibe_err) in enumerate(table):
            for values, shift, colour, hatch in (
                (pibe_err[:, j], -offset, PIBE_C, "////"),
                (epibe_err[:, j], +offset, EPIBE_C, None),
            ):
                bp = ax.boxplot(
                    [values], positions=[c + shift], widths=width,
                    patch_artist=True, manage_ticks=False, showfliers=False,
                    medianprops=dict(color=INK, lw=1.1),
                    whiskerprops=dict(color=colour, lw=0.9),
                    capprops=dict(color=colour, lw=0.9),
                )
                for patch in bp["boxes"]:
                    patch.set(facecolor=colour, edgecolor=colour, linewidth=0.9,
                              alpha=0.45, hatch=hatch)
                # Thirteen trajectories: show every one, not only the summary.
                jitter = rng.uniform(-0.07, 0.07, size=values.shape)
                ax.plot(c + shift + jitter, values, ls="none", marker="o", ms=2.0,
                        markerfacecolor=colour, markeredgecolor="white",
                        markeredgewidth=0.3, zorder=3)
        ax.set_yscale("log")
        # Plain decimals: 3x10^1 notation is unreadable at this size.
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi * 3.0)
        for c, (_, pibe_err, epibe_err) in enumerate(table):
            ratio = np.median(pibe_err[:, j]) / np.median(epibe_err[:, j])
            text = f"{ratio:.0f}" if ratio >= 10 else f"{ratio:.1f}" if ratio >= 1 else f"{ratio:.2f}"
            ax.annotate(text + r"$\times$", xy=(c, 0.93),
                        xycoords=ax.get_xaxis_transform(), ha="center", va="center",
                        fontsize=args.fontsize - 1.5, color=INK_SOFT)
        ax.set_title(labels[j], fontsize=args.fontsize + 0.5, pad=2)
        ax.set_xticks(range(len(table)))
        bottom_row = j >= (rows - 1) * cols
        ax.set_xticklabels([name for name, *_ in table],
                           fontsize=args.fontsize - 1.0 if bottom_row else 0)
        if not bottom_row:
            ax.tick_params(axis="x", labelbottom=False)
        ax.set_xlim(-0.6, len(table) - 0.4)
        ax.grid(axis="x", visible=False)
        if j % 3 == 0:
            ax.set_ylabel("error  [% of scale]")

    for colour, hatch, name in ((PIBE_C, "////", "PIBE"), (EPIBE_C, None, "EPIBE")):
        axes[0, 0].bar([0], [np.nan], color=colour, alpha=0.45, hatch=hatch,
                       edgecolor=colour, label=name)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center",
               ncol=2, bbox_to_anchor=(0.5, 1.0), columnspacing=2.4, handlelength=1.6)
    fig.text(0.995, 0.985, f"$\\sigma$ = {sigma:g} in the noisy cases\nratio = median PIBE / median EPIBE",
             ha="right", va="top", fontsize=args.fontsize - 1.5, color=INK_SOFT)
    fig.tight_layout(rect=(0, 0, 1, 0.94 if args.per_quantity else 0.88), pad=0.4, h_pad=0.8, w_pad=0.6)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    fig.savefig(args.out.with_suffix(".png"), dpi=400)
    plt.close(fig)

    plain = [lab.replace("$", "").replace("\\", "").replace("{", "").replace("}", "") for lab in labels]
    print(f"held-out trajectories: {len(clean_data)}   (medians, % of reference scale)\n")
    print(f"{'condition':<22}" + "".join(f"{q:>16}" for q in plain))
    for name, pibe_err, epibe_err in table:
        flat = name.replace("\n", ", ").replace("$\\sigma$", "sigma")
        print(f"{flat:<22}" + "".join(
            f"{np.median(pibe_err[:, j]):>7.2f} /{np.median(epibe_err[:, j]):>7.2f}"
            for j in range(len(plain))))
    print("(each cell: PIBE / EPIBE)")
    print(f"\nwrote {args.out}\nwrote {args.out.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
