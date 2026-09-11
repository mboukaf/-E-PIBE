#!/usr/bin/env python3
r"""The scale-normalized percentage errors of Section 5, over the test trajectories.

.. math::

    E_{x_j} &= \frac{100}{s_{x_j}}
        \Bigl(\frac{1}{P_{te}}\sum_{\ell=1}^{P_{te}}
              \|\hat{\mathbf X}^\ell_j - \mathbf X^\ell_j\|^2_{N,2}\Bigr)^{1/2} \\
    E_{\theta_j} &= \frac{100}{s_{\theta_j}}
        \Bigl(\frac{1}{P_{te}}\sum_{\ell=1}^{P_{te}}
              |\hat\theta^\ell_j - \theta^\ell_j|^2\Bigr)^{1/2} \\
    E_d &= \frac{100}{G_q s_a}
        \Bigl(\frac{1}{P_{te}}\sum_{\ell=1}^{P_{te}}
              \|\hat{\mathbf d}^\ell - \mathbf d^\ell\|^2_{N,2}\Bigr)^{1/2}

Three details matter for these to mean what they say, and each differs from what
:mod:`pibe.eval.metrics` reports.

**Root mean square across trajectories, not mean.**  The norm is squared
*inside* the trajectory average and the square root taken after, so a few bad
trajectories weigh more than they would in a mean of per-trajectory norms.
Combined with Eq. (60)'s :math:`\|\cdot\|_{N,2}`, each state and disturbance
figure is the plain RMS error over every sample of every test trajectory.

**The normalization is by reference scale, not by signal magnitude.**  Dividing
by :math:`s_{x_j}` rather than by :math:`\|x_j\|` makes the number a percentage
of the *admissible set*, which is what Corollary 2's componentwise bounds
(114)-(118) are stated against.  A coordinate that happens to be small on a
given trajectory therefore does not inflate its own error figure.

**The disturbance scale is** :math:`G_q s_a`, from Eq. (118):
:math:`\|\hat d - d\|_\infty \le \varepsilon_{d,q} + G_q s_a B_{n+1}` with
:math:`G_q := \sup_t \|\Gamma_q(t)\|_2` of Eq. (73).  So :math:`E_d` is measured
against the largest disturbance the admissible coefficient box can produce,
which is the quantity the bound is expressed in.

On :math:`s_a`
--------------
Section 4.1 asks for positive reference scales :math:`s_y, s_{x_j}, s_{\theta_j},
s_a` to be *fixed*, without prescribing them.  This implementation takes
:math:`s_{x_j}` and :math:`s_{\theta_j}` as the half-widths of the admissible
intervals, so every normalized quantity lives in :math:`[-1, 1]`; for
consistency :math:`s_a` is taken as the 2-norm of the coefficient box's
half-widths, the radius of the smallest ball containing :math:`\mathcal{A}`.
The choice is reported alongside the numbers because it scales :math:`E_d`
directly.

Usage::

    python scripts/report_errors.py --run outputs/big_w10_v2_s1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402

from pibe.config import RunConfig  # noqa: E402
from pibe.data.noise import build_noise_model  # noqa: E402
from pibe.data.simulate import uniform_grid  # noqa: E402
from pibe.experiment import build_experiment  # noqa: E402
from pibe.training.callbacks import load_checkpoint, resolve_checkpoint  # noqa: E402
from pibe.utils.logging import setup_logging  # noqa: E402
from pibe.utils.seeding import make_generator  # noqa: E402


def basis_sup_norm(basis, horizon: float, samples: int = 200_001) -> float:
    r""":math:`G_q = \sup_{t\in[0,T]}\|\Gamma_q(t)\|_2`, Eq. (73).

    Evaluated on a fine grid rather than analytically, so it holds for any basis
    the framework supports rather than only the Fourier one.
    """
    grid = uniform_grid(0.0, horizon, samples, dtype=torch.float64)
    gamma = basis.evaluate(grid.to(basis.t_end if False else grid.dtype))
    return float(torch.linalg.vector_norm(gamma, dim=-1).max())


@torch.no_grad()
def report(run: Path, split: str = "val", sigma: float | None = None,
           family: str = "gaussian", seed: int = 909) -> dict:
    r"""Compute the Section 5 percentages for one run.

    ``sigma = None`` evaluates on the data exactly as the run's own config
    generates it, which is the level the model was trained at.  Passing a
    ``sigma`` keeps the *same* held-out trajectories --- the same states,
    parameters and disturbance coefficients --- and only re-draws the
    measurement, so the comparison isolates the noise.  ``sigma = 0`` is the
    noise-free case, :math:`y = x_1` exactly.
    """
    config = RunConfig.from_yaml(run / "config.resolved.yaml")
    experiment = build_experiment(config)
    load_checkpoint(resolve_checkpoint(run), experiment.bank)
    experiment.bank.to(device=experiment.device, dtype=experiment.dtype)
    experiment.bank.eval()

    data = experiment.val_data if split == "val" else experiment.train_data
    if sigma is None:
        y = data.y
        level = config.data.noise_sigma
        label = f"{config.data.noise_family} (as trained)"
    else:
        # Re-measure the same trajectories: y = x_1 + omega at the given level.
        if sigma > 0:
            noise = build_noise_model(
                family, sigma,
                truncation_sigmas=config.data.noise_truncation_sigmas,
            )
            omega = noise.sample(tuple(data.y.shape), generator=make_generator(seed),
                                 dtype=experiment.dtype).to(data.y.device)
        else:
            omega = torch.zeros_like(data.y)
        y = data.x[..., 0] + omega
        level = sigma
        label = "noise-free" if sigma == 0 else family
    estimates = experiment.bank.estimate(y, data.t, experiment.t_coll)

    s_x, s_theta = experiment.system.reference_scales()
    s_x = s_x.cpu()
    s_theta = s_theta.cpu()
    bounds = experiment.coefficient_bounds.cpu()
    # s_a: radius of the smallest ball containing the coefficient box, matching
    # the half-width convention used for s_x and s_theta.
    half_widths = 0.5 * (bounds[:, 1] - bounds[:, 0])
    s_a = float(torch.linalg.vector_norm(half_widths))
    g_q = basis_sup_norm(experiment.basis, config.data.horizon)

    # ||.||_{N,2} squared is the mean over samples; averaging that over
    # trajectories and taking the root gives the RMS over all samples.
    state_sq = ((estimates.x - data.x) ** 2).mean(dim=1)          # (P, n)
    e_x = 100.0 * state_sq.mean(dim=0).sqrt().cpu() / s_x
    theta_sq = (estimates.theta - data.theta) ** 2                # (P, n-1)
    e_theta = 100.0 * theta_sq.mean(dim=0).sqrt().cpu() / s_theta
    d_sq = ((estimates.d - data.d) ** 2).mean(dim=1)              # (P,)
    e_d = 100.0 * float(d_sq.mean().sqrt()) / (g_q * s_a)

    return {
        "run": run.name,
        "split": split,
        "sigma": level,
        "noise": label,
        "P_te": len(data),
        "N": data.n_samples,
        "s_x": s_x.tolist(),
        "s_theta": s_theta.tolist(),
        "s_a": s_a,
        "G_q": g_q,
        "G_q_s_a": g_q * s_a,
        "E_x": e_x.tolist(),
        "E_theta": e_theta.tolist(),
        "E_d": e_d,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, nargs="+", required=True)
    parser.add_argument("--split", choices=("val", "train"), default="val")
    parser.add_argument("--sigma", type=float, nargs="*", default=None,
                        help="re-measure the same held-out trajectories at these "
                             "noise levels; 0 is noise-free. Omit to use the "
                             "run's own configured noise.")
    parser.add_argument("--family", default="gaussian")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    setup_logging("WARNING")

    if args.sigma is None:
        rows = [report(r, args.split) for r in args.run]
    else:
        rows = [report(r, args.split, sigma=sg, family=args.family)
                for r in args.run for sg in args.sigma]
    first = rows[0]
    print(f"scale-normalized percentage errors, {args.split} split "
          f"(P_te = {first['P_te']}, N = {first['N']})\n")
    print(f"reference scales: s_x = {[round(v, 4) for v in first['s_x']]}, "
          f"s_theta = {[round(v, 4) for v in first['s_theta']]}")
    print(f"                  s_a = {first['s_a']:.6f} (2-norm of the coefficient "
          f"box half-widths)")
    print(f"                  G_q = {first['G_q']:.6f}, "
          f"G_q*s_a = {first['G_q_s_a']:.6f}\n")

    n = len(first["E_x"])
    header = (f"{'noise':>16}{'sigma':>9}"
              + "".join(f"{'E_x' + str(j+1) + ' %':>11}" for j in range(n))
              + "".join(f"{'E_th' + str(j+1) + ' %':>11}" for j in range(n - 1))
              + f"{'E_d %':>10}")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['noise']:>16}{r['sigma']:>9.4g}"
              + "".join(f"{v:>11.4f}" for v in r["E_x"])
              + "".join(f"{v:>11.4f}" for v in r["E_theta"])
              + f"{r['E_d']:>10.4f}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
