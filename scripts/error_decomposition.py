#!/usr/bin/env python3
r"""Separate each quantity's intrinsic error from the error it inherits upstream.

Section 4.1 splits every cell's error in two.  Write :math:`\bar\cdot` for what
cell :math:`k` produces when it is fed the *exact* upstream input
:math:`\mathbf{U}^{\star}_{k-1}` --- the true states and parameters of the
trajectory --- and :math:`\breve\cdot` for what it produces in the real cascade,
fed :math:`\breve{\mathbf{U}}_{k-1}` from the cells before it.  Then, Eq. (56),

.. math::

    \epsilon^{int}_{s_k} := \bar x_k - x_k, \qquad
    \epsilon^{prop}_{s_k} := \breve x_k - \bar x_k,

and likewise for the parameters, with Eq. (57) giving

.. math:: \breve x_k - x_k = \epsilon^{prop}_{s_k} + \epsilon^{int}_{s_k}.

The intrinsic part is what this cell gets wrong on its own; the propagated part
is what it inherits.  The total error that :mod:`report_errors` measures is the
sum of the two, so a large total says nothing about *where* to intervene ---
a cell that is itself accurate can still report a poor estimate because its
input was poor, and the fix in that case is upstream.

What the decomposition costs
----------------------------
Each cell is evaluated a second time on an input assembled from ground truth.
That is only possible offline, with the true trajectory in hand: it is a
diagnostic of a trained bank, not something the estimator could compute.

Two structural facts are worth checking in the output rather than assuming:

* :math:`\mathbf{U}^{\star}_1 = \mathbf{Y} = \breve{\mathbf{U}}_1`, so
  :math:`D_1 = 0` and **cell 2 has exactly zero propagated error**.  For
  :math:`n = 3` that means :math:`x_1`, :math:`x_2` and :math:`\theta_1` are
  purely intrinsic, and only :math:`x_3`, :math:`\theta_2` and the disturbance
  inherit anything.
* Errors add as *signed* quantities, so their norms do not:
  :math:`\|\epsilon^{tot}\| \le \|\epsilon^{int}\| + \|\epsilon^{prop}\|`, with
  equality only if the two point the same way.  A total smaller than either part
  means they partly cancel, which is reported rather than hidden.

Normalization matches :mod:`report_errors` and ``plot_error_boxplots`` --- each
quantity as a percentage of its own reference scale --- so the totals here are
the same numbers those produce.

Usage::

    python scripts/error_decomposition.py --run outputs/big_w10_v2_s1
    python scripts/error_decomposition.py --sigma 0.02
"""

from __future__ import annotations

import argparse
import json
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
from pibe.core.bank import Mode  # noqa: E402
from pibe.core.residuals import disturbance_estimate  # noqa: E402
from pibe.data.noise import build_noise_model  # noqa: E402
from pibe.experiment import build_experiment  # noqa: E402
from pibe.training.callbacks import load_checkpoint, resolve_checkpoint  # noqa: E402
from pibe.utils.logging import setup_logging  # noqa: E402
from pibe.utils.seeding import make_generator  # noqa: E402
from report_errors import basis_sup_norm  # noqa: E402

INTRINSIC, PROPAGATED, TOTAL = "#ff6c27", "#337b3e", "#2c6ab1"
INK, INK_SOFT = "#0b0b0b", "#52514e"


def paper_style(base: float) -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": base, "axes.labelsize": base,
        "xtick.labelsize": base, "ytick.labelsize": base - 0.5,
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
        "pdf.fonttype": 42, "ps.fonttype": 42, "hatch.linewidth": 0.6,
    })


@torch.no_grad()
def ideal_outputs(bank, y, x_true, theta_true, t_data, t_coll) -> dict:
    r"""Each cell run on the exact upstream input :math:`\mathbf{U}^\star_{k-1}`.

    The assembly mirrors :meth:`EstimatorBank.assemble_input` block for block ---
    the measurement, then the upstream new coordinates, then the upstream
    parameter heads --- with every estimated block replaced by its ground truth.
    Cells are evaluated independently: nothing here is a cascade, which is the
    whole point.
    """
    outputs = {}
    for k in bank.cell_indices:
        blocks = [y]
        blocks += [x_true[:, :, j - 1] for j in range(2, k)]        # x_2 .. x_{k-1}
        blocks += [theta_true[:, j - 2:j - 1] for j in range(2, k)]  # theta_1 .. theta_{k-2}
        u = torch.cat(blocks, dim=-1)
        expected = bank.cell(k).input_dim
        if u.shape[-1] != expected:
            raise AssertionError(
                f"ideal U_{k - 1} has width {u.shape[-1]}, expected {expected}"
            )
        outputs[k] = bank.cell(k)(u, t_data, t_coll,
                                  need_derivative=False, create_graph=False)
    return outputs


@torch.no_grad()
def decompose(experiment, data, y) -> tuple[dict, list[str]]:
    """Per-trajectory intrinsic, propagated and total error for every quantity."""
    bank = experiment.bank
    n = bank.state_dim
    x_true, theta_true, d_true = data.x, data.theta, data.d

    actual = bank.forward(bank.final_index, y, data.t, experiment.t_coll,
                          mode=Mode.GLOBAL, need_derivative=False)
    ideal = ideal_outputs(bank, y, x_true, theta_true, data.t, experiment.t_coll)

    s_x, s_theta = experiment.system.reference_scales()
    bounds = experiment.coefficient_bounds.cpu()
    s_a = float(torch.linalg.vector_norm(0.5 * (bounds[:, 1] - bounds[:, 0])))
    g_q = basis_sup_norm(experiment.basis, float(data.t[-1]))

    def series(value):
        """||.||_{N,2} per trajectory: the RMS over samples of Eq. (60)."""
        return (value ** 2).mean(dim=1).sqrt()

    rows, labels = {}, []

    # x_1 is cell 2's reconstructed coordinate; x_j (j >= 2) is cell j's new one.
    for j in range(1, n + 1):
        cell = 2 if j == 1 else j
        field = "x_prev_data" if j == 1 else "x_new_data"
        bar = getattr(ideal[cell], field)
        breve = getattr(actual[cell], field)
        truth = x_true[:, :, j - 1]
        scale = 100.0 / float(s_x[j - 1])
        labels.append(f"$x_{{{j}}}$")
        rows[labels[-1]] = dict(
            intrinsic=(scale * series(bar - truth)).cpu().numpy(),
            propagated=(scale * series(breve - bar)).cpu().numpy(),
            total=(scale * series(breve - truth)).cpu().numpy(),
            cell=cell,
        )

    # theta_j is cell (j+1)'s parameter head.
    for j in range(1, n):
        cell = j + 1
        bar, breve = ideal[cell].head[:, 0], actual[cell].head[:, 0]
        truth = theta_true[:, j - 1]
        scale = 100.0 / float(s_theta[j - 1])
        labels.append(rf"$\theta_{{{j}}}$")
        rows[labels[-1]] = dict(
            intrinsic=(scale * (bar - truth).abs()).cpu().numpy(),
            propagated=(scale * (breve - bar).abs()).cpu().numpy(),
            total=(scale * (breve - truth).abs()).cpu().numpy(),
            cell=cell,
        )

    # The disturbance, through the final cell's coefficients.  Note the
    # intrinsic part here also carries the basis truncation eps_{d,q} of
    # Eq. (64): even exact coefficients cannot represent d outside span(Gamma_q).
    final = bank.final_index
    d_bar = disturbance_estimate(experiment.basis, ideal[final].head, data.t)
    d_breve = disturbance_estimate(experiment.basis, actual[final].head, data.t)
    scale = 100.0 / (g_q * s_a)
    labels.append("$d$")
    rows["$d$"] = dict(
        intrinsic=(scale * series(d_bar - d_true)).cpu().numpy(),
        propagated=(scale * series(d_breve - d_bar)).cpu().numpy(),
        total=(scale * series(d_breve - d_true)).cpu().numpy(),
        cell=final,
    )
    return rows, labels


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("outputs/big_w10_v2_s1"))
    parser.add_argument("--sigma", type=float, default=None,
                        help="re-measure at this level; omit for the trained one")
    parser.add_argument("--family", default="gaussian")
    parser.add_argument("--noise-seed", type=int, default=909)
    parser.add_argument("--split", choices=("val", "train"), default="val")
    parser.add_argument("--width", type=float, default=7.0)
    parser.add_argument("--height", type=float, default=3.9)
    parser.add_argument("--fontsize", type=float, default=9.0)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    setup_logging("WARNING")
    paper_style(args.fontsize)

    config = RunConfig.from_yaml(args.run / "config.resolved.yaml")
    experiment = build_experiment(config)
    load_checkpoint(resolve_checkpoint(args.run), experiment.bank)
    experiment.bank.to(device=experiment.device, dtype=experiment.dtype)
    experiment.bank.eval()

    data = experiment.val_data if args.split == "val" else experiment.train_data
    if args.sigma is None:
        y, sigma = data.y, config.data.noise_sigma
    elif args.sigma > 0:
        noise = build_noise_model(
            args.family, args.sigma,
            truncation_sigmas=config.data.noise_truncation_sigmas,
        )
        omega = noise.sample(tuple(data.y.shape),
                             generator=make_generator(args.noise_seed),
                             dtype=experiment.dtype).to(data.y.device)
        y, sigma = data.x[..., 0] + omega, args.sigma
    else:
        y, sigma = data.x[..., 0], 0.0

    rows, labels = decompose(experiment, data, y)

    def rms(v):
        return float(np.sqrt((np.asarray(v) ** 2).mean()))

    print(f"{args.run.name}, {args.split} split, {len(data)} trajectories, "
          f"sigma = {sigma:g}")
    print("errors as a percentage of each quantity's reference scale; "
          "RMS over trajectories\n")
    head = (f"{'':>9}{'cell':>6}{'intrinsic':>12}{'propagated':>12}{'total':>12}"
            f"{'prop/total':>12}{'int+prop':>11}")
    print(head)
    print("-" * len(head))
    summary = {}
    for lab in labels:
        r = rows[lab]
        i, p, t = rms(r["intrinsic"]), rms(r["propagated"]), rms(r["total"])
        plain = (lab.replace("$", "").replace("\\", "")
                 .replace("{", "").replace("}", ""))
        print(f"{plain:>9}{r['cell']:>6}{i:>12.4f}{p:>12.4f}{t:>12.4f}"
              f"{p / t if t else 0.0:>12.3f}{i + p:>11.4f}")
        summary[plain] = dict(cell=r["cell"], intrinsic=i, propagated=p,
                              total=t, prop_share=p / t if t else 0.0)
    print("\n'int+prop' is the triangle-inequality ceiling on 'total'; a total "
          "below it means\nthe two parts partly cancel.  Cell 2 must show "
          "propagated = 0 exactly, since U*_1 = Y.")

    # ---- figure -----------------------------------------------------------
    # A dot plot rather than bars: the values span more than a decade, so the
    # axis has to be logarithmic, and a bar on a log axis encodes nothing --- its
    # length depends on where the axis happens to start.  A marker encodes
    # position, which stays honest under any scale.
    fig, ax = plt.subplots(figsize=(args.width, args.height))
    k = len(labels)
    idx = np.arange(k)
    width = 0.24
    series = ((-width, "intrinsic", INTRINSIC, "o", "intrinsic"),
              (0.0, "propagated", PROPAGATED, "s", "propagated"),
              (+width, "total", TOTAL, "D", "total"))

    # Hairlines between groups, so a triple reads as one quantity's three
    # readings rather than as six loose points.
    for j in range(k - 1):
        ax.axvline(j + 0.5, color="#e4e3df", lw=0.7, zorder=0)

    for shift, key, colour, marker, name in series:
        heights = np.array([rms(rows[lab][key]) for lab in labels])
        drawn = heights > 0
        ax.plot(idx[drawn] + shift, heights[drawn], marker=marker, ls="none",
                ms=8, markerfacecolor=colour, markeredgecolor="white",
                markeredgewidth=1.2, label=name, zorder=3)
        for x, h in zip(idx[~drawn] + shift, heights[~drawn]):
            # Zero has no place on a log axis and is exactly the result worth
            # seeing, so it is written rather than plotted.
            ax.annotate("0", xy=(x, 0.045), xycoords=("data", "axes fraction"),
                        ha="center", va="bottom", fontsize=args.fontsize - 1,
                        color=colour, fontweight="bold")

    # The share inherited from upstream is the number the figure exists to show.
    for j, lab in enumerate(labels):
        r = rows[lab]
        t, pr = rms(r["total"]), rms(r["propagated"])
        ax.annotate(f"{100 * pr / t:.0f}%" if t else "--",
                    xy=(j, 0.955), xycoords=("data", "axes fraction"),
                    ha="center", va="center", fontsize=args.fontsize - 1.0,
                    color=INK_SOFT)
    ax.annotate("inherited", xy=(0.0, 0.955), xycoords=("axes fraction", "axes fraction"),
                xytext=(4, 0), textcoords="offset points", ha="left", va="center",
                fontsize=args.fontsize - 2.0, color="#8d8c88", style="italic")

    ax.set_yscale("log")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo * 0.5, hi * 2.4)
    ax.set_xticks(idx)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.6, k - 0.4)
    ax.set_ylabel("error  [% of reference scale]")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3,
              columnspacing=2.2, handletextpad=0.5)
    fig.tight_layout(rect=(0, 0, 1, 0.955), pad=0.4)

    tag = "trained" if args.sigma is None else f"sig{sigma:g}".replace(".", "p")
    out = args.out or (args.run / "figures" / f"error_decomposition_{tag}.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=400)
    plt.close(fig)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(
            dict(run=args.run.name, split=args.split, sigma=sigma,
                 trajectories=len(data), quantities=summary), indent=2))
        print(f"\nwrote {args.json}")
    print(f"\nwrote {out}\nwrote {out.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
