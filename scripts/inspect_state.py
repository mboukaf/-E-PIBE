#!/usr/bin/env python3
r"""Inspect a resume checkpoint: how far a run got, and how good it already is.

A run that hits its wall clock leaves ``state.pt`` but none of the final
artifacts --- ``bank.pt``, ``metrics_*.json`` and ``figures/`` are written only
after the last cell finishes.  That is a reporting gap, not a data loss:
``state.pt`` holds the weights, the optimizer moments, the schedule position,
the history and both RNG streams, so the run resumes exactly where it stopped.

This reads that file and answers the two questions worth asking before deciding
whether to resume, extend the wall clock, or cut the budget:

**How far did it get?**  Cell index, iteration and phase, expressed as a
fraction of the configured budget, together with the wall-clock rate implied by
the logged history.  That rate is what tells you whether a resubmission will
finish in the remaining time or hit the same wall again.

**Is what it has any good?**  The stored ``state_dict`` is a complete bank, so
the usual metrics can be computed from it directly.

One caveat the output makes explicit: Algorithm 1 trains cells in sequence, so
a run stopped during cell :math:`k` has cells :math:`k+1 \dots n+1` still at
their initialization.  Any metric that depends on them --- the disturbance
above all, since :math:`\hat a` comes from the final cell --- is meaningless
until the run reaches them, and is reported as such rather than silently.

Usage::

    python scripts/inspect_state.py --run outputs/big_w5_s0
    python scripts/inspect_state.py --run outputs/big_w5_s0 --metrics
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402

from pibe.config import RunConfig  # noqa: E402
from pibe.training.callbacks import History, load_training_state  # noqa: E402
from pibe.utils.logging import setup_logging  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--metrics", action="store_true",
                        help="also score the partial weights on the validation split")
    args = parser.parse_args()
    setup_logging("WARNING")

    state_path = args.run / "state.pt"
    if not state_path.exists():
        if (args.run / "bank.pt").exists():
            print(f"{args.run}: no state.pt but bank.pt is present -- this run "
                  f"finished cleanly (the resume file is deleted on success).")
            return 0
        print(f"{args.run}: neither state.pt nor bank.pt; nothing to inspect",
              file=sys.stderr)
        return 1

    state = load_training_state(state_path)
    config = RunConfig.from_yaml(args.run / "config.resolved.yaml")
    training = config.training
    final_cell = config.system.kwargs.get("n", 3) + 1  # overwritten below if built

    cell = int(state["cell_index"])
    iteration = int(state["iteration"])
    mode = state.get("mode", "?")

    # The final cell may carry extra end-to-end iterations.
    budget = training.n_total + (
        training.final_global_iters if cell == final_cell and not training.local_only else 0
    )

    print(f"run            : {args.run}")
    print(f"stopped in     : cell {cell}, iteration {iteration} of {budget} "
          f"({100 * iteration / max(1, budget):.1f}% of this cell), phase '{mode}'")
    print(f"schedule       : n_total={training.n_total}, n_par={training.n_par}, "
          f"final_global_iters={training.final_global_iters}")
    print(f"checkpointing  : every {training.checkpoint_every} iterations")
    print(f"resumes from   : cell {cell}, iteration {iteration + 1}")

    history = History.from_dict(state.get("history", {}))
    done = [entry for entry in history.validation if "seconds" in entry]
    if done:
        print()
        print("cells completed so far:")
        for entry in done:
            print(f"  cell {int(entry['cell'])}: {entry['seconds'] / 3600:.2f} h"
                  + (f", val local {entry['val_local']:.4e}" if "val_local" in entry else ""))
        total = sum(e["seconds"] for e in done) / 3600
        print(f"  ({total:.2f} h of finished cells recorded in the history)")

    # Rate on the current cell, from the logged iterations -- the number that
    # decides whether a resubmission fits in the remaining wall clock.
    current = [r for r in history.iterations if r.cell == cell]
    if len(current) >= 2:
        print()
        print(f"current cell   : {len(current)} logged iterations, "
              f"last objective {current[-1].objective:.4e}")
    remaining = budget - iteration
    print(f"remaining here : {remaining} iterations in cell {cell}"
          + (f", plus cells {cell + 1}..{final_cell}" if cell < final_cell else ""))

    print()
    print("to continue: resubmit the same array task. train_pibe.py finds "
          "state.pt and\nresumes automatically (pass --no-resume to start over).")
    if remaining > 0 and done:
        rate = done[-1]["seconds"] / max(1, training.n_total)
        print(f"at the last cell's rate ({rate:.3f} s/iteration) the rest of this "
              f"cell needs\nabout {remaining * rate / 3600:.1f} h; budget the wall "
              f"clock accordingly.")

    if args.metrics:
        from pibe.eval.metrics import compute_metrics
        from pibe.experiment import build_experiment

        print()
        experiment = build_experiment(config)
        experiment.bank.load_state_dict(state["state_dict"])
        experiment.bank.to(device=experiment.device, dtype=experiment.dtype)
        experiment.bank.eval()
        data = experiment.val_data
        with torch.no_grad():
            estimates = experiment.bank.estimate(data.y, data.t, experiment.t_coll)
        metrics = compute_metrics(estimates, data)
        print("metrics from the partial weights:")
        print(metrics.summary())
        untrained = [k for k in range(cell + 1, experiment.bank.final_index + 1)]
        if untrained:
            print()
            print(f"WARNING: cells {untrained} are still at initialization, so the "
                  f"quantities they\nproduce are meaningless. In particular "
                  f"{'the disturbance estimate is meaningless' if experiment.bank.final_index in untrained else 'downstream states are unreliable'}"
                  f" -- a_hat comes from cell {experiment.bank.final_index}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
