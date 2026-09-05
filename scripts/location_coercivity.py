#!/usr/bin/env python3
r"""How strongly does the physics identify a constant offset in :math:`\hat x_1`?

Proposition 1 says the data risk alone cannot separate the signal from an
unknown location, and that the pair is identified when the physics functional
:math:`R_P` has a *unique zero*.  Everything EPIBE does rests on that clause, so
it is worth measuring rather than assuming.

The measurement is direct.  Take the true trajectory, displace the measured
coordinate by a constant,

.. math:: \hat x_1 = x_1 + c,

and then give the rest of the model every chance to absorb it: the triangular
form of Eq. (1) *determines* the remaining states from :math:`\hat x_1`,

.. math:: \hat x_2 = \dot{\hat x}_1 - f_1(\hat x_1, \hat\theta_1), \qquad
          \hat x_3 = \dot{\hat x}_2 - f_2(\hat x_1, \hat x_2, \hat\theta_2),

leaving the last equation as the only real constraint,

.. math:: r(t) = \dot{\hat x}_3 - f_3(\hat x, \hat\theta) - \Gamma_q(t)^\top \hat a .

Minimizing :math:`\|r\|^2` over the admissible :math:`\hat\theta` and
:math:`\hat a` gives the *smallest physics cost the offset can be hidden at*.
Plotted against :math:`c`, that is the profile of :math:`R_P` along the exact
direction the data term cannot see.

Reading it
----------
* A sharp minimum at :math:`c = 0` means the physics does identify the location,
  and the only question is whether :math:`\lambda` is large enough for training
  to feel it.  The curvature says how large.
* A flat profile means the offset is absorbable at no cost --- the unique-zero
  hypothesis fails for this system, and no schedule, weight or initialization
  will recover :math:`\mu_\omega`.

:math:`\hat a` enters :math:`r` linearly, so it is eliminated exactly by least
squares at every step rather than optimized; only the two parameters are
searched.
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

from pibe.basis import build_basis  # noqa: E402
from pibe.config import RunConfig  # noqa: E402
from pibe.data.disturbance import BasisDisturbance  # noqa: E402
from pibe.data.simulate import rk4_integrate, uniform_grid  # noqa: E402
from pibe.systems.registry import build_system  # noqa: E402
from pibe.utils.logging import setup_logging  # noqa: E402


def derivative(values: torch.Tensor, dt: float) -> torch.Tensor:
    """Fourth-order central difference, one-sided at the ends."""
    d = torch.empty_like(values)
    d[..., 2:-2] = (
        values[..., :-4] - 8 * values[..., 1:-3]
        + 8 * values[..., 3:-1] - values[..., 4:]
    ) / (12 * dt)
    d[..., :2] = (values[..., 1:3] - values[..., :2]) / dt
    d[..., -2:] = (values[..., -2:] - values[..., -3:-1]) / dt
    return d


def physics_cost(system, basis, x1_hat, dt, theta, trim: int = 40) -> torch.Tensor:
    r"""Smallest :math:`\|r\|^2` for this :math:`\hat x_1` and :math:`\hat\theta`.

    The states are reconstructed by the triangular form and :math:`\hat a` is
    eliminated by least squares, so this is the physics cost after the model has
    absorbed as much of the displacement as it structurally can.
    """
    n = system.n
    x_hat = [x1_hat]
    for j in range(1, n):
        prev = x_hat[-1]
        stacked = torch.stack(x_hat, dim=-1)
        theta_j = theta[:j].reshape(1, j).expand(stacked.shape[0], j)
        x_hat.append(derivative(prev, dt) - system.f(j, stacked, theta_j))
    x_full = torch.stack(x_hat, dim=-1)

    theta_full = theta.reshape(1, -1).expand(x_full.shape[0], system.theta_dim)
    defect = derivative(x_hat[-1], dt) - system.f_last(x_full, theta_full)

    # a enters linearly: eliminate it exactly.  Trim the ends, where the
    # one-sided differences are least accurate.
    gamma = basis.evaluate(basis_grid)[trim:-trim]      # (M, q)
    target = defect[trim:-trim]                          # (M,)
    solution = torch.linalg.lstsq(gamma, target.unsqueeze(-1)).solution
    residual = target - (gamma @ solution).squeeze(-1)
    return torch.mean(residual**2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path("configs/automatica_n3v2_epibe_cell2.yaml"))
    parser.add_argument("--offsets", type=float, nargs="+", default=None)
    parser.add_argument("--samples", type=int, default=4001)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--grid", type=int, default=41)
    parser.add_argument("--out", type=Path, default=Path("outputs/location_coercivity.json"))
    args = parser.parse_args()
    setup_logging("WARNING")

    config = RunConfig.from_yaml(args.config)
    system = build_system(config.system.name, dtype=torch.float64,
                          **config.system.kwargs)
    basis = build_basis(kind=config.basis.kind, q=config.basis.q, t_start=0.0,
                        t_end=config.data.horizon, dtype=torch.float64,
                        **config.basis.basis_kwargs())

    global basis_grid
    basis_grid = uniform_grid(0.0, config.data.horizon, args.samples,
                              dtype=torch.float64)
    dt = float(basis_grid[1] - basis_grid[0])

    bounds = torch.tensor(config.basis.coefficient_intervals(), dtype=torch.float64)
    coefficients = (0.5 * (bounds[:, 0] + bounds[:, 1])
                    + 0.5 * (bounds[:, 1] - bounds[:, 0])
                    * torch.tensor([0.6, -0.4, 0.5], dtype=torch.float64))
    disturbance = BasisDisturbance(basis, coefficients)
    x0 = system.sample_x0(1, generator=torch.Generator().manual_seed(3))
    truth = rk4_integrate(system, t_grid=basis_grid, x0=x0,
                          theta=system.theta_true.reshape(1, -1),
                          disturbance=disturbance, substeps=8)[0]

    offsets = args.offsets or [round(0.01 * k, 3) for k in range(-10, 11)]
    lo, hi = system.theta_bounds[:, 0], system.theta_bounds[:, 1]

    # The inner problem over theta is small but not convex, and a local method
    # started from the middle of the box gets stuck well above the true optimum
    # (measured: 6.6e-04 where the truth attains 1.9e-18).  A profile is only
    # meaningful if every point is actually minimized, so the two parameters are
    # swept on a grid first and refined from the best cell.
    axes = [torch.linspace(float(lo[j]), float(hi[j]), args.grid, dtype=torch.float64)
            for j in range(system.theta_dim)]
    mesh = torch.cartesian_prod(*axes)

    rows = []
    for c in offsets:
        x1_hat = truth[:, 0] + c
        with torch.no_grad():
            costs = torch.tensor(
                [physics_cost(system, basis, x1_hat, dt, mesh[i])
                 for i in range(mesh.shape[0])],
                dtype=torch.float64,
            )
        best = mesh[int(costs.argmin())].clone()

        raw = torch.logit(((best - lo) / (hi - lo)).clamp(1e-6, 1 - 1e-6))
        raw = raw.detach().requires_grad_(True)
        optimizer = torch.optim.Adam([raw], lr=0.01)
        for _ in range(args.steps):
            optimizer.zero_grad(set_to_none=True)
            theta = lo + (hi - lo) * torch.sigmoid(raw)
            physics_cost(system, basis, x1_hat, dt, theta).backward()
            optimizer.step()
        with torch.no_grad():
            theta = lo + (hi - lo) * torch.sigmoid(raw)
            cost = float(physics_cost(system, basis, x1_hat, dt, theta))
            if cost > float(costs.min()):      # refinement never makes it worse
                theta, cost = best, float(costs.min())
        rows.append({"offset": c, "physics_cost": cost, "theta": theta.tolist()})
        print(f"  c = {c:+.3f}  ->  {cost:.6e}  at theta = "
              f"[{theta[0]:+.4f}, {theta[1]:+.4f}]", flush=True)

    baseline = min(r["physics_cost"] for r in rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2))

    print("physics cost of hiding a constant offset in x_1")
    print("(states reconstructed by the triangular form, theta and a re-fitted)\n")
    print(f"{'offset c':>9} | {'min ||r||^2':>13} | {'vs c=0':>9} | {'theta_hat':>26}")
    print("-" * 70)
    zero = next(r["physics_cost"] for r in rows if abs(r["offset"]) < 1e-12)
    for r in rows:
        print(f"{r['offset']:>9.3f} | {r['physics_cost']:>13.6e} | "
              f"{r['physics_cost']/zero:>8.2f}x | "
              f"[{r['theta'][0]:+.4f}, {r['theta'][1]:+.4f}]")
    print(f"\nminimum at c = {min(rows, key=lambda r: r['physics_cost'])['offset']:+.3f}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
