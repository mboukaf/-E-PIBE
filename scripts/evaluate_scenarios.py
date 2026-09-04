#!/usr/bin/env python3
r"""Evaluate a trained bank on scenarios it was not trained on.

The training split already reports held-out *initial conditions*, but that is
only one axis of generalization. This script exercises several, each isolating
one thing the estimator might or might not cope with:

``fresh_ic``
    Many new initial conditions, same ``theta`` and same disturbance. The
    core generalization test, and the one the paper's train/test split
    measures — run here over far more trajectories for stable statistics.
``noise_realization``
    Identical trajectories, different noise draws. Isolates sensitivity to the
    particular ``omega`` seen, as opposed to the trajectory.
``low_noise`` / ``high_noise``
    A noise level different from training. The estimator was fitted at one
    ``sigma``; this asks whether it degrades gracefully.
``mismatch``
    The basis-mismatch experiment of Eq. (2): ``d = Gamma_q^T a + r_q`` with
    ``r_q(t) = eps * sin(0.15 t^2)``. Part of ``d`` now lies **outside** the
    span of ``Gamma_q``, so exact recovery is impossible and the disturbance
    error is bounded below by ``eps_{d,q}`` however good the estimator is.
``new_a``
    Different disturbance coefficients. Out of training distribution: the bank
    saw exactly one ``a``, so this asks whether the trajectory-conditioned
    encoder *infers* the disturbance from ``y`` or merely memorized it.
``new_theta``
    Different true parameters. Also out of distribution, and the sharper of the
    two questions: the reported ``theta_hat`` is produced by a head reading the
    trajectory code, so a bank that memorized ``theta`` will not track it.

The last two are expected to be the weakest and are reported as diagnostics of
what the architecture actually learned, not as failures of the method — nothing
in Eq. (1) or Algorithm 1 asks for them.

Usage::

    python scripts/evaluate_scenarios.py --run outputs/n3v2_s0
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402

from pibe.config import RunConfig  # noqa: E402
from pibe.data.dataset import generate_dataset  # noqa: E402
from pibe.data.disturbance import BasisDisturbance, ChirpRemainder  # noqa: E402
from pibe.data.noise import NoiseFree, TruncatedGaussianNoise  # noqa: E402
from pibe.eval.metrics import normalized_l2  # noqa: E402
from pibe.experiment import build_experiment  # noqa: E402
from pibe.training.callbacks import load_checkpoint  # noqa: E402
from pibe.utils.logging import get_logger, setup_logging  # noqa: E402
from pibe.utils.seeding import make_generator  # noqa: E402

logger = get_logger("scenarios")


@dataclass
class ScenarioResult:
    """Per-trajectory error statistics for one scenario."""

    name: str
    note: str
    n_trajectories: int
    state_mean: list[float]
    state_p95: list[float]
    theta_err: list[float]
    theta_spread: list[float]
    coeff_err: list[float]
    d_l2_mean: float
    d_l2_p95: float
    d_sup: float
    d_norm: float
    output_l2: float

    def as_dict(self) -> dict:
        return self.__dict__.copy()


@torch.no_grad()
def _stats(estimates, data) -> dict:
    """Per-trajectory errors, so spread across trajectories is visible."""
    state_err = estimates.x - data.x                      # (P, N, n)
    per_traj = normalized_l2(state_err, dim=1)            # (P, n)
    d_err = estimates.d - data.d                          # (P, N)
    d_per_traj = normalized_l2(d_err, dim=1)              # (P,)
    return {
        "state_mean": per_traj.mean(0).tolist(),
        "state_p95": per_traj.quantile(0.95, dim=0).tolist(),
        "theta_err": (estimates.theta - data.theta).abs().mean(0).tolist(),
        "theta_spread": estimates.theta.std(0).tolist(),
        "d_l2_mean": float(d_per_traj.mean()),
        "d_l2_p95": float(d_per_traj.quantile(0.95)),
        "d_sup": float(d_err.abs().max()),
        "d_norm": float(data.d.abs().max()),
        "output_l2": float(normalized_l2(estimates.x[..., 0] - data.y, dim=1).mean()),
    }


def run_scenario(
    experiment, name: str, note: str, n_traj: int, seed: int,
    sigma: float | None = None, remainder: float = 0.0,
    new_coefficients: bool = False, new_theta: bool = False,
) -> ScenarioResult:
    """Generate data under one scenario and score the (already trained) bank."""
    config = experiment.config
    system, basis = experiment.system, experiment.basis
    generator = make_generator(seed)

    # Disturbance: the trained one unless the scenario changes it.
    coefficients = experiment.disturbance.coefficients
    if new_coefficients:
        box = experiment.coefficient_bounds
        u = torch.rand(basis.q, generator=generator, dtype=box.dtype)
        coefficients = box[:, 0] + u * (box[:, 1] - box[:, 0])
    disturbance = BasisDisturbance(
        basis, coefficients,
        remainder=ChirpRemainder(remainder, config.basis.remainder_rate)
        if remainder > 0 else None,
    )

    # Parameters: the trained ones unless the scenario changes them.
    saved_theta = system.theta_true.clone()
    if new_theta:
        lo, hi = system.theta_bounds[:, 0], system.theta_bounds[:, 1]
        u = torch.rand(system.theta_dim, generator=generator, dtype=lo.dtype)
        system.theta_true = 0.25 * (2.0 * u - 1.0).to(lo.device)

    level = config.data.noise_sigma if sigma is None else sigma
    noise = (
        TruncatedGaussianNoise(level, truncation_sigmas=config.data.noise_truncation_sigmas)
        if level > 0 else NoiseFree()
    )
    data = generate_dataset(
        system=system, t_grid=experiment.data.t, n_trajectories=n_traj,
        noise=noise, disturbance=disturbance, generator=generator,
        substeps=config.data.substeps,
    ).to(device=experiment.device, dtype=experiment.dtype)

    estimates = experiment.bank.estimate(data.y, data.t, experiment.t_coll)
    stats = _stats(estimates, data)
    coeff_err = (estimates.a.mean(0) - coefficients.to(estimates.a.device)).abs()

    system.theta_true = saved_theta  # restore
    return ScenarioResult(
        name=name, note=note, n_trajectories=n_traj,
        coeff_err=coeff_err.tolist(), **stats,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--trajectories", type=int, default=200)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    setup_logging("WARNING")
    config = RunConfig.from_yaml(args.run / "config.resolved.yaml")
    experiment = build_experiment(config)
    load_checkpoint(args.run / "bank.pt", experiment.bank)
    experiment.bank.to(device=experiment.device, dtype=experiment.dtype)

    sigma = config.data.noise_sigma
    P = args.trajectories
    scenarios = [
        ("fresh_ic", f"{P} unseen initial conditions, trained sigma={sigma}",
         dict(n_traj=P, seed=101)),
        ("noise_realization", "same setup, a different noise draw",
         dict(n_traj=P, seed=202)),
        ("low_noise", f"sigma {sigma} -> {sigma/5:g}",
         dict(n_traj=P, seed=303, sigma=sigma / 5)),
        ("high_noise", f"sigma {sigma} -> {sigma*5:g}",
         dict(n_traj=P, seed=404, sigma=sigma * 5)),
        ("mismatch_0.03", "r_q = 0.03 sin(0.15 t^2) outside the basis span",
         dict(n_traj=P, seed=505, remainder=0.03)),
        ("new_a", "different disturbance coefficients (out of distribution)",
         dict(n_traj=P, seed=606, new_coefficients=True)),
        ("new_theta", "different true parameters (out of distribution)",
         dict(n_traj=P, seed=707, new_theta=True)),
    ]

    results = []
    for name, note, kwargs in scenarios:
        results.append(run_scenario(experiment, name, note, **kwargs))
        print(f"  ran {name}", flush=True)

    n = experiment.system.n
    print()
    print(f"model: {args.run.name}    system: {experiment.system.name}    "
          f"P = {P} per scenario")
    print(f"true theta = {[round(v, 5) for v in experiment.system.theta_true.tolist()]}")
    print()
    head = f"{'scenario':<18}" + "".join(f"{'x'+str(j+1):>10}" for j in range(n))
    head += "".join(f"{'|th'+str(j+1)+'|':>10}" for j in range(n - 1))
    head += f"{'d L2':>10}{'d sup':>10}{'||d||inf':>10}"
    print(head)
    print("-" * len(head))
    for r in results:
        row = f"{r.name:<18}"
        row += "".join(f"{v:>10.2e}" for v in r.state_mean)
        row += "".join(f"{v:>10.2e}" for v in r.theta_err)
        row += f"{r.d_l2_mean:>10.2e}{r.d_sup:>10.2e}{r.d_norm:>10.3f}"
        print(row)
    print()
    for r in results:
        print(f"  {r.name:<18} {r.note}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps([r.as_dict() for r in results], indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
