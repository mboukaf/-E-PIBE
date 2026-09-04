#!/usr/bin/env python3
r"""Test a trained bank against measurement noise it was not trained on.

The estimator is fitted once, at the noise level in its config, and is then
evaluated on the *same* trajectories corrupted at a range of levels.  This
measures deployed robustness --- what happens when the sensor turns out to be
worse (or better) than it was during training --- as distinct from retraining at
each level, which would instead measure the best accuracy attainable per level.

Only the measurement noise changes between rows.  The states, the parameters and
the disturbance are integrated once and reused, so every difference across the
table is attributable to the noise draw alone rather than to a different set of
initial conditions.  Several independent noise realizations are averaged at each
level, since with a small trajectory count a single draw is itself noisy.

Usage::

    python scripts/noise_sweep.py --run outputs/n3v2_s0
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
from pibe.data.dataset import TrajectoryData, generate_dataset  # noqa: E402
from pibe.data.disturbance import BasisDisturbance  # noqa: E402
from pibe.data.noise import NOISE_FAMILIES, NoiseFree, build_noise_model  # noqa: E402
from pibe.eval.metrics import normalized_l2  # noqa: E402
from pibe.eval.observability import basis_observability  # noqa: E402
from pibe.experiment import build_experiment  # noqa: E402
from pibe.training.callbacks import load_checkpoint  # noqa: E402
from pibe.utils.logging import setup_logging  # noqa: E402
from pibe.utils.seeding import make_generator  # noqa: E402

DEFAULT_LEVELS = [0.0, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]


def clean_trajectories(experiment, count: int, seed: int) -> TrajectoryData:
    """Fresh noise-free trajectories from the experiment's own system.

    Drawn with a generator distinct from the training one, so these are initial
    conditions the bank has not seen.  ``theta`` and the disturbance follow the
    config exactly, including whether they are shared or sampled per trajectory.
    """
    config = experiment.config
    system = experiment.system
    generator = make_generator(seed)
    theta = None
    if config.data.sample_theta_per_trajectory:
        unit = torch.rand(count, system.theta_dim, generator=generator,
                          dtype=experiment.dtype)
        theta = 0.25 * (2.0 * unit - 1.0)

    disturbance = experiment.disturbance
    if config.data.sample_disturbance_per_trajectory:
        # The trained object carries the training set's own coefficient rows,
        # one per trajectory.  Draw fresh ones instead: an unseen ``a`` is both
        # the right count and the honest test of a head that must infer it.
        bounds = experiment.coefficient_bounds.to("cpu")
        unit = torch.rand(count, experiment.basis.q, generator=generator,
                          dtype=experiment.dtype)
        a = bounds[:, 0] + unit * (bounds[:, 1] - bounds[:, 0])
        disturbance = BasisDisturbance(
            experiment.basis, a, remainder=disturbance.remainder
        )

    return generate_dataset(
        system=system,
        t_grid=experiment.data.t.cpu(),
        n_trajectories=count,
        theta=theta,
        noise=NoiseFree(),
        disturbance=disturbance,
        generator=generator,
        substeps=config.data.substeps,
    ).to(device=experiment.device, dtype=experiment.dtype)


@torch.no_grad()
def evaluate(experiment, clean: TrajectoryData, sigma: float, realizations: int,
             seed: int, family: str = "gaussian", bias: float = 0.0) -> dict:
    """Score the bank on ``clean`` re-measured at this noise level."""
    noise = build_noise_model(family, sigma, bias=bias)
    draws = 1 if sigma <= 0 else realizations  # a noise-free row has one outcome
    state, theta_err, coeff_err, d_l2, d_rel, out_l2 = [], [], [], [], [], []
    d_scale = normalized_l2(clean.d, dim=1)

    for r in range(draws):
        generator = make_generator(seed + 1000 * r)
        omega = noise.sample(
            tuple(clean.y.shape), generator=generator, dtype=experiment.dtype
        ).to(clean.y.device)
        y = clean.x[..., 0] + omega

        est = experiment.bank.estimate(y, clean.t, experiment.t_coll)
        state.append(normalized_l2(est.x - clean.x, dim=1).mean(0))
        theta_err.append((est.theta - clean.theta).abs().mean(0))
        if clean.coefficients is not None:
            coeff_err.append((est.a - clean.coefficients).abs().mean(0))
        err = normalized_l2(est.d - clean.d, dim=1)
        d_l2.append(err.mean())
        d_rel.append((err / d_scale).mean())
        out_l2.append(normalized_l2(est.x[..., 0] - y, dim=1).mean())

    stack = lambda xs: torch.stack(xs).mean(0)  # noqa: E731
    return {
        "sigma": sigma,
        "family": family,
        "bias": bias,
        "realizations": draws,
        "state": stack(state).tolist(),
        "theta": stack(theta_err).tolist(),
        "coeff": stack(coeff_err).tolist() if coeff_err else None,
        "d_l2": float(stack(d_l2)),
        "d_rel": float(stack(d_rel)),
        "output_l2": float(stack(out_l2)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--levels", type=float, nargs="+", default=DEFAULT_LEVELS)
    parser.add_argument("--trajectories", type=int, default=64)
    parser.add_argument("--realizations", type=int, default=8,
                        help="independent noise draws averaged per level")
    parser.add_argument("--noise", type=str, default="gaussian",
                        choices=sorted(NOISE_FAMILIES),
                        help="shape of the measurement-noise law")
    parser.add_argument("--bias", type=float, default=0.0,
                        help="constant sensor offset, in units of sigma; "
                             "nonzero breaks Proposition 1's zero-mean premise")
    parser.add_argument("--seed", type=int, default=909)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    setup_logging("WARNING")
    config = RunConfig.from_yaml(args.run / "config.resolved.yaml")
    experiment = build_experiment(config)
    load_checkpoint(args.run / "bank.pt", experiment.bank)
    experiment.bank.to(device=experiment.device, dtype=experiment.dtype)
    experiment.bank.eval()
    trained_at = config.data.noise_sigma

    clean = clean_trajectories(experiment, args.trajectories, args.seed)
    rows = [
        evaluate(experiment, clean, s, args.realizations, args.seed,
                 family=args.noise, bias=args.bias)
        for s in args.levels
    ]

    n = experiment.system.n
    print(f"model    : {args.run}   trained at sigma = {trained_at}")
    print(f"noise    : {args.noise}"
          + (f", biased by {args.bias:+g} sigma" if args.bias else "")
          + "  (calibrated to the stated sigma; the model was trained on "
            "truncated Gaussian)")
    print(f"data     : {args.trajectories} unseen trajectories, identical across "
          f"rows; {args.realizations} noise draws averaged per level")
    print(f"setting  : theta "
          f"{'sampled per trajectory' if config.data.sample_theta_per_trajectory else 'fixed'}"
          f", disturbance "
          f"{'per trajectory' if config.data.sample_disturbance_per_trajectory else 'shared'}")
    print()
    head = f"{'sigma':>8}" + "".join(f"{'x' + str(j + 1):>11}" for j in range(n))
    head += "".join(f"{'|dth' + str(j + 1) + '|':>11}" for j in range(n - 1))
    head += f"{'d L2':>11}{'d rel':>8}"
    print(head)
    print("-" * len(head))
    for r in rows:
        line = f"{r['sigma']:>8.4g}"
        line += "".join(f"{v:>11.3e}" for v in r["state"])
        line += "".join(f"{v:>11.3e}" for v in r["theta"])
        line += f"{r['d_l2']:>11.3e}{100 * r['d_rel']:>7.0f}%"
        print(line + ("  <- trained here" if abs(r["sigma"] - trained_at) < 1e-12 else ""))

    # What the linearized observability analysis predicts, for comparison.
    print()
    print("predicted per-coefficient SNR (eval/observability.py):")
    print(f"{'sigma':>8}" + "".join(f"{'a' + str(j + 1):>9}" for j in range(experiment.basis.q)))
    for s in args.levels:
        if s <= 0:
            continue
        r = basis_observability(
            experiment.system, experiment.basis, experiment.coefficient_bounds,
            noise_sigma=s, n_samples=config.data.n_samples,
        )
        flag = "" if float(r.snr.min()) >= 1.0 else "   <- below the noise floor"
        print(f"{s:>8.4g}" + "".join(f"{v:>9.2f}" for v in r.snr.tolist()) + flag)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
