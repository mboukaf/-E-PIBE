# Postmortem: the first big cluster run collapsed

`outputs/big_w10_s0` (checkpoint removed in V1.0; its config and training log are preserved under `outputs/_evidence/`), Margaret, ~24 h wall, killed at the limit having reached
21% of its final cell. The estimates were unusable: state errors of 0.76–0.87
against the ~1e-2 the same system reaches locally.

## Cause

The bounded-output reparameterization saturated to death.

Every state decoder and parameter head maps to its admissible set by
`c + r·tanh(z)`. Nothing bounds `z`, and the gradient multiplier is
`4·exp(−2|z|)`. From the recovered checkpoint:

```
 cell       part |  |z| mean   |z| max | pinned | tanh(z) mean
    2       head |     8.559      9.17 |   100% |      +1.0000
    3    decoder |    18.422     18.66 |   100% |      +1.0000
    3       head |    16.443     16.45 |   100% |      -1.0000
```

At `|z| = 18.4` the multiplier is `4e-16` — below float64 resolution. The unit
is then dead for the rest of the run no matter how wrong it is. The
reparameterization's own docstring warned about this cliff at `|z| ≳ 19`;
nothing enforced it.

The failure propagates. Cell 2's θ head pins at the top of its box, which makes
cell 3's physics inconsistent, which drives cell 3's decoder to its own box edge
and kills it too. Cell 4's physics residual then explodes from 0.16 to 22.4 and
the global phase, which retrains cells 2–4 together, drags what was left:
cell 2's local loss went from 3.3e-02 at the end of its own training to 6.4e-01.
The last 26 000 iterations — 5.5 h — moved the objective not at all
(0.940 → 0.928).

Two things made it worse than it needed to be:

* **The schedule was never affordable.** The measured rate was 0.77 s/iteration;
  the 260k-step schedule needed 55.6 h against a 24 h wall.
* **The failure was silent.** Nothing reported a vanishing gradient, so 16 h of
  compute went into a model that had been dead since hour 8.

## Diagnosis

The whole thing reproduces in four minutes at P=128 with the same architecture
(`configs/automatica_n3v2_repro.yaml`): cell 3's gradient norm falls to 7.5e-07
by iteration 250, and cell 4's physics jumps to 15.6. That reproduction is what
made the rest tractable — 24-hour feedback loops are not a debugging tool.

## Fixes

**`architecture.saturation_limit`** (the fix). A straight-through clamp on `z`:
clamped in the forward pass, identity in the backward. The output is confined to
`c ± r·tanh(A)` — 99.93% of the box at `A = 4` — while the gradient multiplier
is floored at `sech²(4) = 1.3e-03` instead of decaying. On a head at `|z| = 18.4`
the gradient goes from `1.78e-16` to `5.36e-04`, a factor of `3e12`, while the
output moves only `0.400000 → 0.399732`. Adam normalizes per parameter, so a
small but honest gradient still produces full-size steps; `4e-16` is noise.

Same reproduction, changing only that:

| | x₁ | x₂ | x₃ | \|θ₁\| | \|θ₂\| | d |
|---|---|---|---|---|---|---|
| collapsed | 2.266e-01 | 4.125e-01 | 1.080e+00 | 1.650e-01 | 5.039e-01 | 2.027e-01 |
| **fixed** | **1.091e-02** | **5.941e-02** | **6.020e-02** | **3.718e-02** | **3.449e-02** | **7.617e-02** |

Cell 4's objective: 1.56e+01 → 2.04e-04.

**Saturation is now reported.** When a gradient norm drops below 1e-5 the
trainer logs a warning naming the pinned fraction and the remedy. The same
failure will not be silent again.

**`final_global_iters` 200000 → 30000** so the job fits (19.2 h estimated), and
**`lr_global` 5e-4 → 2e-4** so the global phase refines rather than rewrites.

**Checkpointing.** `bank.pt` and `history.json` are now written at every
periodic checkpoint, not only on success, and `load_checkpoint` falls back to
`state.pt`. Previously an interrupted run was fully recoverable but looked
unusable to every evaluation script — which is how the first attempt appeared to
have produced nothing.

## Note on the default

`saturation_limit` defaults to `None`, so every existing config and checkpoint
keeps its exact behaviour; the `_v2` configs set it explicitly. Changing a
default under saved runs has already caused two silent misreadings in this
project and is not worth repeating.
