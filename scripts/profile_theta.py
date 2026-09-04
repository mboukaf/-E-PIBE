#!/usr/bin/env python3
r"""Profile the objective against a clamped parameter vector.

The curvature analysis used elsewhere reconstructs the chain *exactly*: given a
parameter error it sets ``x_k = xdot_{k-1} - f_{k-1}(x_{k-1}, theta+e)`` and
asks what residual survives.  That is an upper bound on identifiability, because
the trained bank does not reconstruct the chain exactly --- its consistency
terms are soft penalties, so it has strictly more freedom to hide a parameter
error than the reconstruction admits.  On the fourth-order system the proxy
predicted large gains that training did not deliver, so a measurement that
includes the network's real freedom is needed.

This script provides it.  For each offset ``e``, every parameter head is
**clamped** to ``theta_true + e`` and everything else --- all decoders, all
encoders, and the final cell's coefficient head --- is trained normally.  The
achieved objective is then the best the model can do while committed to that
parameter value:

    L*(e) = min over all remaining weights of L^{n+1}_Tot

``L*`` is a profile of the actual objective, so its curvature in ``e`` is the
identifiability of ``theta`` *as the estimator experiences it*.

Reading the result:

- ``L*`` rises steeply away from ``e = 0``  -> theta is identifiable; a poor
  estimate means the optimizer is not finding the minimum.
- ``L*`` is flat  -> the objective genuinely cannot distinguish theta, and no
  amount of training or tuning will recover it.

Usage::

    python scripts/profile_theta.py --config configs/automatica_n4.yaml \
        --offsets -0.15 -0.075 0.0 0.075 0.15 --iters 9000
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
from pibe.core.losses import total_loss  # noqa: E402
from pibe.experiment import build_experiment  # noqa: E402
from pibe.training.callbacks import load_checkpoint  # noqa: E402
from pibe.training.phases import apply_phase  # noqa: E402
from pibe.utils.logging import get_logger, setup_logging  # noqa: E402

logger = get_logger("profile_theta")


def clamp_heads(bank, theta_value: torch.Tensor, clamp_index: int) -> list[torch.nn.Parameter]:
    """Clamp **one** parameter head; leave every other weight trainable.

    This is a profile in the statistical sense: one coordinate is fixed and all
    remaining freedom --- including the *other* parameter heads --- is
    re-optimized to compensate.  Clamping all parameters at a common offset
    would instead probe the direction ``(1,1,...,1)``, which is nearly
    orthogonal to the degenerate direction (whose components have opposite
    signs), and would report a steep curve regardless of identifiability.

    ``clamp_index`` is 0-based over ``theta_1..theta_{n-1}``, i.e. cell
    ``clamp_index + 2``.
    """
    trainable: list[torch.nn.Parameter] = []
    target_cell = clamp_index + 2
    for k in bank.cell_indices:
        cell = bank.cell(k)
        if cell.is_final or k != target_cell:
            trainable += list(cell.parameters())
            continue
        value = theta_value[clamp_index].reshape(1, 1)

        def forward(z, _v=value):
            return _v.to(z.dtype).to(z.device).expand(z.shape[0], 1)

        cell.head.forward = forward  # type: ignore[method-assign]
        for p in cell.head.parameters():
            p.requires_grad_(False)
        trainable += list(cell.encoder.parameters()) + list(cell.decoder.parameters())
    return trainable


def run_point(config: RunConfig, offset: float, iters: int, seed: int,
              clamp_index: int, checkpoint: Path | None = None,
              lr: float = 3e-4) -> dict:
    """Fine-tune with theta clamped at ``theta_true + offset``; report the best loss.

    Every point **warm-starts from the same converged checkpoint**.  Training
    each point from scratch instead makes the comparison meaningless: the
    budget needed to converge is far larger than the profile's own scale, so
    the numbers report convergence luck rather than the objective's geometry.
    (Observed directly: from scratch, ``e = 0`` scored *worse* than
    ``e = +-0.12``, which is impossible for a profile.)
    """
    experiment = build_experiment(config)
    bank, data = experiment.bank, experiment.train_data
    if checkpoint is not None:
        load_checkpoint(checkpoint, bank)
        bank.to(device=experiment.device, dtype=experiment.dtype)
    theta = experiment.system.theta_true.clone()
    theta[clamp_index] += offset

    apply_phase(bank, bank.final_index, Mode.GLOBAL)
    trainable = clamp_heads(bank, theta, clamp_index)
    optimizer = torch.optim.Adam(trainable, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iters)
    generator = torch.Generator().manual_seed(seed)

    best = float("inf")
    for step in range(iters):
        index = torch.randperm(len(data), generator=generator)[:16]
        y = data.y[index]
        outputs = bank(bank.final_index, y, data.t, experiment.t_coll, mode=Mode.GLOBAL)
        losses = bank.losses(
            bank.final_index, y, experiment.t_coll, outputs,
            lam=config.training.lam, mode=Mode.GLOBAL,
        )
        objective = total_loss({m: v.local for m, v in losses.items()}, bank.final_index)
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        scheduler.step()
        value = float(objective.detach())
        if step > iters // 2:
            best = min(best, value)
    return {"offset": offset, "best_loss": best, "theta": theta.tolist()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--offsets", type=float, nargs="+",
                        default=[-0.15, -0.075, 0.0, 0.075, 0.15])
    parser.add_argument("--iters", type=int, default=9000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--clamp-index", type=int, default=0,
                        help="0-based index of the theta component to clamp")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="converged bank.pt to warm-start every point from")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    setup_logging("WARNING")
    config = RunConfig.from_yaml(args.config)

    results = []
    for offset in args.offsets:
        point = run_point(config, offset, args.iters, args.seed, args.clamp_index,
                          checkpoint=args.checkpoint, lr=args.lr)
        results.append(point)
        print(f"  theta_{args.clamp_index + 1} offset {offset:+.4f}   "
              f"best L_Tot = {point['best_loss']:.4e}", flush=True)

    baseline = min(r["best_loss"] for r in results if r["offset"] == 0.0)
    print()
    print(f"{'offset':>9} | {'best L_Tot':>12} | {'excess over e=0':>16}")
    print("-" * 45)
    for r in results:
        print(f"{r['offset']:>+9.4f} | {r['best_loss']:>12.4e} | "
              f"{r['best_loss'] - baseline:>+16.4e}")
    widest = max(results, key=lambda r: abs(r["offset"]))
    ratio = widest["best_loss"] / baseline if baseline > 0 else float("inf")
    print()
    print(f"L*(e={widest['offset']:+.3f}) / L*(0) = {ratio:.2f}x")
    print("  >> flat (ratio near 1) means theta is NOT identifiable by this objective")
    print("  >> steep means it is, and a poor estimate is an optimization failure")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
