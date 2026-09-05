#!/usr/bin/env python3
r"""Estimate the sensor offset from the physics, which is where it is identifiable.

The obstruction found in ``Docs/EPIBE.md`` is that the energy data term is
*stationary* along a constant shift of :math:`\hat x_1`: the state estimate and
the learned density slide together at no cost, so no first-order method travels
that direction.  Proposition 1 says identification must therefore come from a
physics functional with a unique zero, and ``scripts/location_coercivity.py``
measures that it does --- the physics cost of hiding an offset :math:`c` grows
cleanly as :math:`c^2`, reaching 6.8e-06 at the true bias.

That signal is real but small: a trained bank's own physics residual sits near
1e-4, fifteen times larger, so the optimizer cannot feel it through the loss.
It is perfectly visible, though, if one stops asking gradient descent to find it
and simply *evaluates* it.

The estimator
-------------
Take any trained bank --- PIBE will do, and PIBE is what a biased sensor gives
you.  Its first decoder is a continuous function of time, so it can be sampled
as finely as the differentiation needs.  For each candidate offset :math:`c`:

1. displace it, :math:`\hat x_1^{(c)} = \hat x^2_1 - c`;
2. rebuild the rest of the chain from the triangular form of Eq. (1), which
   *determines* :math:`\hat x_2,\dots,\hat x_n` from :math:`\hat x_1` and
   :math:`\hat\theta` --- no slack, unlike the soft consistency terms used in
   training, and that slack is precisely what lets the offset hide;
3. minimize the last equation's residual over the admissible
   :math:`\hat\theta` and :math:`\hat a` (the latter enters linearly and is
   eliminated exactly).

The minimizing :math:`c` is the offset.  Because step 2 enforces the chain
exactly, the identifying signal is no longer competing with the reconstruction
slack, and the estimate does not depend on the data term at all --- so it is
immune to the degeneracy that defeats the training objective.

Two-stage use
-------------
``--correct`` writes a de-biased measurement offset into a new config, so the
estimator can be re-run on :math:`y - \hat\mu_\omega`.  That is the practical
form: PIBE to get a trajectory, the physics to get the offset, PIBE again on
the corrected measurement.
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
from pibe.core.bank import Mode  # noqa: E402
from pibe.data.simulate import uniform_grid  # noqa: E402
from pibe.experiment import build_experiment  # noqa: E402
from pibe.training.callbacks import load_checkpoint  # noqa: E402
from pibe.utils.logging import get_logger, setup_logging  # noqa: E402

logger = get_logger("estimate_offset")


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


def chain_residual(system, gamma, x1_hat, dt, theta, trim: int) -> torch.Tensor:
    r"""Last-equation residual after rebuilding the chain exactly from ``x1_hat``.

    ``x1_hat`` has shape ``(P, M)``; the reconstruction and the least-squares
    elimination of :math:`\hat a` are done per trajectory, since each carries its
    own disturbance coefficients.
    """
    x_hat = [x1_hat]
    for j in range(1, system.n):
        stacked = torch.stack(x_hat, dim=-1)                      # (P, M, j)
        theta_j = theta[:j].reshape(1, 1, j).expand(*stacked.shape[:-1], j)
        x_hat.append(derivative(x_hat[-1], dt) - system.f(j, stacked, theta_j))
    x_full = torch.stack(x_hat, dim=-1)                           # (P, M, n)

    theta_full = theta.reshape(1, 1, -1).expand(*x_full.shape[:-1], system.theta_dim)
    defect = derivative(x_hat[-1], dt) - system.f_last(x_full, theta_full)

    target = defect[:, trim:-trim].unsqueeze(-1)                  # (P, M', 1)
    design = gamma[trim:-trim].unsqueeze(0).expand(target.shape[0], -1, -1)
    solution = torch.linalg.lstsq(design, target).solution
    residual = target - design @ solution
    return torch.mean(residual**2)


@torch.no_grad()
def first_state_on(bank, y, t_data, t_fine) -> torch.Tensor:
    r""":math:`\hat x^2_1` sampled on a fine grid.

    The decoder is continuous in ``t``, so the collocation grid can be made as
    fine as three numerical derivatives require --- the ``N = 201`` data grid
    would be far too coarse.
    """
    with torch.enable_grad():
        outputs = bank(target_cell=bank.final_index, y=y, t_data=t_data,
                       t_coll=t_fine, mode=Mode.GLOBAL, create_graph=False)
    return outputs[2].x_prev_coll.detach()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--offsets", type=float, nargs="+", default=None)
    parser.add_argument("--samples", type=int, default=4001)
    parser.add_argument("--grid", type=int, default=31)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--trim", type=int, default=60)
    parser.add_argument("--trajectories", type=int, default=8)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    setup_logging("WARNING")

    config = RunConfig.from_yaml(args.run / "config.resolved.yaml")
    experiment = build_experiment(config)
    load_checkpoint(args.run / "bank.pt", experiment.bank)
    experiment.bank.to(device=experiment.device, dtype=experiment.dtype)
    experiment.bank.eval()
    system = experiment.system
    true_bias = config.data.noise_bias

    data = experiment.val_data
    count = min(args.trajectories, len(data))
    t_fine = uniform_grid(0.0, config.data.horizon, args.samples,
                          dtype=experiment.dtype).to(experiment.device)
    dt = float(t_fine[1] - t_fine[0])
    gamma = experiment.basis.evaluate(t_fine)

    x1 = first_state_on(experiment.bank, data.y[:count], data.t, t_fine)

    lo, hi = system.theta_bounds[:, 0], system.theta_bounds[:, 1]
    axes = [torch.linspace(float(lo[j]), float(hi[j]), args.grid,
                           dtype=experiment.dtype)
            for j in range(system.theta_dim)]
    mesh = torch.cartesian_prod(*axes)

    offsets = args.offsets or [round(0.01 * k, 3) for k in range(-4, 13)]
    rows = []
    for c in offsets:
        shifted = x1 - c
        with torch.no_grad():
            costs = torch.tensor(
                [chain_residual(system, gamma, shifted, dt, mesh[i], args.trim)
                 for i in range(mesh.shape[0])], dtype=experiment.dtype
            )
        best = mesh[int(costs.argmin())]
        raw = torch.logit(((best - lo) / (hi - lo)).clamp(1e-6, 1 - 1e-6))
        raw = raw.detach().requires_grad_(True)
        optimizer = torch.optim.Adam([raw], lr=0.01)
        for _ in range(args.steps):
            optimizer.zero_grad(set_to_none=True)
            theta = lo + (hi - lo) * torch.sigmoid(raw)
            chain_residual(system, gamma, shifted, dt, theta, args.trim).backward()
            optimizer.step()
        with torch.no_grad():
            theta = lo + (hi - lo) * torch.sigmoid(raw)
            cost = float(chain_residual(system, gamma, shifted, dt, theta, args.trim))
            if cost > float(costs.min()):
                theta, cost = best, float(costs.min())
        rows.append({"offset": c, "cost": cost, "theta": theta.tolist()})
        print(f"  c = {c:+.3f} -> {cost:.6e}   theta_hat "
              f"[{theta[0]:+.4f}, {theta[1]:+.4f}]", flush=True)

    best = min(rows, key=lambda r: r["cost"])
    print()
    print(f"estimated sensor offset  mu_hat = {best['offset']:+.4f}")
    print(f"true sensor offset               = {true_bias:+.4f}")
    print(f"theta at the minimum      = {[round(v, 4) for v in best['theta']]}")
    print(f"true theta                = {[round(float(v), 4) for v in data.theta[0]]}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(
            {"rows": rows, "mu_hat": best["offset"], "true_bias": true_bias,
             "theta_at_min": best["theta"]}, indent=2))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
