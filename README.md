# PIBE — Physics-Informed Bank of Estimators

Implementation of the PIBE framework from *Joint Estimation of State and
Parameters for a Class of Nonlinear Disturbed Systems with Unknown Noise*
(Boukaf, Belkhatir, Chadli, Laleg-Kirati), for the Gaussian /
truncated-Gaussian measurement case of Section 3.1.

Jointly estimates the states `x_2..x_n`, the constant parameters
`θ = (θ_1..θ_{n-1})` and the time-varying disturbance `d(t)` of a nonlinear
system in triangular canonical form, from the single noisy output `y = x_1 + ω`.

> **Scope.** This covers PIBE only. EPIBE (Section 3.2, energy-based extension
> for non-Gaussian noise) and the Section 4 error-propagation certificates are
> not implemented yet; the module layout leaves room for both.

## The system class

```
ẋ_j = x_{j+1} + f_j(x_1..x_j, θ_1..θ_j),   j = 1..n-1
ẋ_n = f_n(x, θ) + d(t)
y   = x_1 + ω(t)
d(t) = Γ_q(t)ᵀa + r_q(t)
```

## Systems provided

| name | note |
|---|---|
| `automatica_n4` | The fourth-order benchmark — **the one to use** |
| `sin_chain` | Generic test fixture, parameterized in `n` |

`automatica_n4` ([source](src/pibe/systems/examples/automatica_n4.py),
[config](configs/automatica_n4.yaml)):

```
ẋ_1 = x_2 − 0.35 tanh x_1 + θ_1(0.8 + 0.2 cos x_1)
ẋ_2 = x_3 − 0.30 tanh x_2 + 0.10 sin x_1 + θ_2(0.9 + 0.1 sin x_1 + 0.1 cos x_2)
ẋ_3 = x_4 − 0.25 tanh x_3 + 0.10 sin(x_1+x_2) + θ_3(1 + 0.1 sin(x_1+x_3))
ẋ_4 = −0.576x_1 − 2.736x_2 − 4.76x_3 − 3.6x_4 + 0.15 sin x_1 + 0.08 tanh(x_2x_3)
      + 0.08θ_1 sin x_2 + 0.06θ_2 sin x_3 + 0.05θ_3 tanh x_4 + d(t)
y   = x_1 + ω
```

with `T = 20`, `Ω = 2π/5`, `Γ_4 = [sin Ωt, cos Ωt, sin 2Ωt, cos 2Ωt]ᵀ`,
`θ_i ~ U[-0.25,0.25]`, `a_{1,2} ~ U[-0.15,0.15]`, `a_{3,4} ~ U[-0.12,0.12]`.

Its properties are asserted rather than assumed
([test_automatica_n4.py](tests/test_automatica_n4.py)):

- the vector field matches an **independently transcribed** right-hand side;
- the linear skeleton has poles exactly at `−0.6, −0.8, −1, −1.2`, i.e. char.
  poly `(s+0.6)(s+0.8)(s+1)(s+1.2)`;
- `W_Γ = 10·I_4` to 1e-11 — Eq. (3) with `γ_d = T/2`, perfectly conditioned;
- the sensitivity floors hold over all of `X` (`∂_{θ_1}f_1 ≥ 0.6`,
  `∂_{θ_2}f_2 ≥ 0.7`, `∂_{θ_3}f_3 ≥ 0.9`), giving `γ_2 ≥ 0.36`, `γ_3 ≥ 0.49`,
  `γ_4 ≥ 0.81`;
- trajectories stay inside `X_j` with margin, and a randomised sweep reproduces
  the empirical maxima `≈ (2.65, 0.72, 0.41, 0.75)`.

**Model mismatch.** Set `basis.remainder_amplitude` to `0.03` for the
out-of-class disturbance `r_4(t) = ε sin(0.15t²)`; `0.0` gives the exact-basis
case. A quadratic chirp has unbounded instantaneous frequency, so no fixed
finite basis can represent it at any `q`, and the amplitude *is* the constant
`ε_{d,q}` of Eq. (4).

> **Basis note.** Section 5.1 of the paper describes a **B-spline** basis; this
> system specifies a **Fourier** one. Both are implemented behind
> `DisturbanceBasis` and selected with `basis.kind`.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Run

```bash
python scripts/train_pibe.py --config configs/automatica_n4.yaml
python scripts/train_pibe.py --config configs/automatica_n4.yaml --dry-run   # inspect setup only
python scripts/evaluate.py --checkpoint outputs/automatica_n4/bank.pt
pytest -q                 # full suite
pytest -q -m 'not slow'   # skip the randomised trajectory sweep
```

Override any config entry from the command line:

```bash
python scripts/train_pibe.py --config configs/automatica_n4.yaml \
  --set training.lam=0.5 basis.remainder_amplitude=0.03 --output-dir outputs/mismatch
```

## Adding your own system

The only system-specific surface is one subclass. Nothing else changes:

```python
from pibe.systems.base import TriangularSystem
from pibe.systems.registry import register_system

@register_system("my_system")
class MySystem(TriangularSystem):
    def f(self, j, x, theta):
        """f_j for j = 1..n-1.  x: (..., j) = x_1..x_j;  theta: (..., j)."""
        ...

    def f_last(self, x, theta):
        """f_n.  x: (..., n);  theta: (..., n-1)."""
        ...
```

Construct it with `state_bounds` (the admissible sets `X_j`), `theta_bounds`
(`Θ_j`), `theta_true` and optionally `x0_bounds` (`χ_0`). The same `f` drives
both data generation and the physics residuals — one point of truth. Then set
`system.name` in the config.

Two constraints worth knowing:

- **`state_bounds` must contain the true trajectory.** The decoders are
  reparameterized onto that box, so a trajectory leaving it is unrepresentable.
  `check_admissible` raises at data-generation time rather than letting training
  silently fail to converge.
- **Write `f` with differentiable torch ops.** It is called both to integrate
  the true dynamics and on network outputs carrying an autograd graph.

## Architecture

Cells `k = 2..n+1`, each a trajectory-conditioned PINN `Φ^k = (ℰ_k, 𝒮_k, 𝒬_k)`.
Every cell reconstructs the coordinate below it and, except for the final cell,
estimates one new coordinate:

| cell | `x_prev` | `x_new` | head |
|---|---|---|---|
| `k = 2` | `x̂²₁` (vs. `y`) | `x̂²₂` | `θ̂²₁` |
| `k = 3..n` | `x̂^k_{k-1}` | `x̂^k_k` | `θ̂^k_{k-1}` |
| `k = n+1` | `x̂^{n+1}_n` (auxiliary) | — | `â` |

Reported estimates are `x̂_1 := x̂²₁`, `x̂_k := x̂^k_k`, `θ̂_{k-1} := θ̂^k_{k-1}`
and `d̂ = Γ_qᵀâ`. The final cell's `x̂^{n+1}_n` is auxiliary — *not* a second
estimate of `x_n`.

## Equation map

| Paper | Module |
|---|---|
| System (1), (5) | [systems/base.py](src/pibe/systems/base.py) |
| Disturbance decomposition (2) | [basis/base.py](src/pibe/basis/base.py), [data/disturbance.py](src/pibe/data/disturbance.py) |
| B-spline basis, Def. 1 (136) | [basis/bspline.py](src/pibe/basis/bspline.py) |
| Fourier basis | [basis/fourier.py](src/pibe/basis/fourier.py) |
| Linear independence (3) | `DisturbanceBasis.gram`, `check_linear_independence` |
| Data generation (135) | [data/simulate.py](src/pibe/data/simulate.py), [data/dataset.py](src/pibe/data/dataset.py) |
| Trajectory input (15), (16) | `EstimatorBank.assemble_input` |
| Encoder (17), decoder (18)/(33), head (19)/(34) | [nets/](src/pibe/nets/) |
| Cell (20), (35) | [core/cell.py](src/pibe/core/cell.py) |
| Time derivative (21), (36) | [core/autodiff.py](src/pibe/core/autodiff.py) |
| Upstream arguments (26) | `upstream_states`, `upstream_thetas` |
| Residuals (25), (31), (40) | [core/residuals.py](src/pibe/core/residuals.py) |
| Data / physics / local losses (22)–(24), (28)–(30), (37)–(39) | [core/losses.py](src/pibe/core/losses.py) |
| Global loss (27), (42) | `total_loss`, `global_weights` |
| Disturbance estimate (41) | `disturbance_estimate` |
| Algorithm 1 | [training/trainer.py](src/pibe/training/trainer.py), [training/phases.py](src/pibe/training/phases.py) |
| Normalization (60), (61) | [nets/normalization.py](src/pibe/nets/normalization.py), [eval/metrics.py](src/pibe/eval/metrics.py) |
| Fill distance (87) | `data.simulate.fill_distance` |

## Training schedule

Algorithm 1 runs an outer loop over cells and, inside it, two regimes split at
`N_par`:

- **`i < N_par`, local pre-training.** Cells `2..k-1` frozen and evaluated under
  `no_grad`; `U_{k-1}` built from detached outputs; only `Θ^k` updated against
  `L^k_Loc`.
- **`i ≥ N_par`, end-to-end fine-tuning.** Cells `2..k` re-run with the graph
  intact; all updated against `L^k_Tot = Σ_{m=2}^{k} e^{−(k−m)/k} L^m_Loc`.

Both are covered by tests asserting that no upstream weight moves in the first
regime and that they do move in the second.

Additional knobs beyond Algorithm 1, each off by default:

| option | effect |
|---|---|
| `final_global_iters` | extra end-to-end iterations at cell `n+1` — **the one that matters**, see below |
| `lr_schedule: cosine` | anneal the phase learning rate to zero |
| `local_only` | skip fine-tuning entirely (ablation, see below) |

## Diagnosing a poor fit

Every training run ends with an **oracle comparison** ([eval/oracle.py](src/pibe/eval/oracle.py)):
the true `(x, θ, a)` is scored on the exact objective being minimised, and the
result tells you which of two very different problems you have.

- **Truth scores better** → *optimisation failure*. The right answer is
  representable and preferred, but training didn't reach it. Levers: a longer
  end-to-end phase (`final_global_iters`), higher `lr_global`, lower `n_par`,
  and checking no head is pinned to its box boundary.
- **Truth scores no better** → *loss-design failure*. No amount of training
  helps; this is the Remark 7 degeneracy and needs excitation or extra
  information.

Two reference points the comparison prints: at the truth every physics residual
vanishes (~1e-30, not 0), and cell 2's data term equals the realised noise
variance — a floor no honest estimator beats. A *trained* cell-2 data term
below that floor means the decoder is fitting noise.

## Why identification is hard here (and where it comes from)

At every cell `k ≤ n`, the physics residual

```
ẋ̂_{k-1} = x̂_k + f_{k-1}(·, θ̂_{k-1})
```

is **one equation in two unknowns** — the new state `x̂_k` and the parameter
`θ̂_{k-1}`. That is a one-parameter family the local loss cannot choose within,
and it is exactly the counterexample Remark 7 constructs. The observable
signature is an estimate drifting along the flat direction until it *saturates
at the boundary* of its admissible box, where tanh kills the gradient — a `θ̂`
sitting exactly on an endpoint with spread ~1e-7.

**The only closure in the bank is the final cell.** There the auxiliary state is
pinned by its consistency target, so `â` is the sole free variable in
`r_{n+1} = ẋ̂_n − f_n(x̂,θ̂) − Γ_qᵀâ`, which must hold across the whole horizon —
over-determined, hence identifying. But that cell trains *last*, and its
information reaches cells `2..n` only through `L^{n+1}_Tot`.

Two consequences, both load-bearing:

- **`final_global_iters` deserves most of the budget.** It extends the final
  cell's end-to-end phase, which is where the bank actually closes. Measured
  effect: that phase alone dropped the objective from `1.1257` to `1.28e-3`.
- **Widening `Θ_j` does not fix boundary pinning.** The direction is flat, so
  the estimate runs to whatever the new boundary is. Keeping the truth interior
  (see `THETA_BOUND` vs `THETA_SAMPLING_BOUND`) avoids a *dead gradient*, but
  the degeneracy itself is broken only by the final cell.

### The `local_only` ablation

`training.local_only: true` skips end-to-end fine-tuning entirely — a departure
from Algorithm 1, kept because it isolates the above. The global loss weights
the current cell by 1 and cell 2 by `e^{−(k−2)/k}`, yet cell 2's data term is
the *only* one comparing against a measurement; every other is a consistency
penalty satisfiable by the chain drifting together.

Measured on `automatica_n4` (identical budget, differing only in the phase):

| | global | local-only |
|---|---|---|
| `L_Tot` (trained) | 7.6e-04 | 1.1120 |
| cell 5 `L_Loc` | 4.2e-04 | 1.1118 |
| `x_3` / `x_4` error | 0.16 / 0.31 | 0.51 / 0.55 |

Local-only is worse overall — but cell 5's loss of 1.11 is *honest*: with
upstream frozen, `â` cannot absorb upstream error, so the residual reports it.
Under the global loss the same error is hidden by driving `ẋ̂_4 − f_4 → 0` and
`â → 0`. Useful as a diagnostic, not as a training recipe.

## Notes on the implementation

- **float64 by default.** The physics residuals rest on autograd time
  derivatives; `device: auto` skips Apple MPS when float64 is requested, since
  that backend cannot represent it.
- **`C²` activations are enforced.** Eq. (21) differentiates the decoder in `t`,
  so ReLU-family activations are rejected at construction rather than yielding
  an almost-everywhere derivative.
- **The time grid is materialized per trajectory.** An `expand`-ed grid would
  make the reverse pass accumulate a batch-summed derivative into every row.
- **Validation runs with grad enabled but `create_graph=False`** — the physics
  residual needs the derivative, not a backward graph.

## Layout

```
src/pibe/
├── systems/     TriangularSystem ABC + registry; the only system-specific surface
├── basis/       Γ_q: B-spline (Definition 1) and Fourier, behind one ABC
├── data/        simulation, truncated-Gaussian noise, disturbances, datasets
├── nets/        encoder / decoder / heads, D_{k-1} normalization, box reparameterization
├── core/        cells, bank, residuals, losses, autodiff
├── training/    Algorithm 1, phase schedule, checkpoints
└── eval/        estimation metrics
```
