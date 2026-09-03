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
| `automatica_n3` | Third-order benchmark — **the one to use** ([config](configs/automatica_n3.yaml)) |
| `automatica_n4` | Fourth-order variant ([config](configs/automatica_n4.yaml)) |
| `sin_chain` | Generic test fixture, parameterized in `n` |

### `automatica_n3` — measured results

```
ẋ_1 = x_2 − 0.35 tanh x_1 + θ_1(0.8 + 0.2 cos x_1)
ẋ_2 = x_3 − 0.30 tanh x_2 + 0.10 sin x_1 + θ_2(0.9 + 0.1 sin x_1 + 0.1 cos x_2)
ẋ_3 = −0.648x_1 − 2.34x_2 − 2.7x_3 + 0.15 sin x_1 + 0.08 tanh(x_1x_2)
      + 0.08θ_1 sin x_2 + 0.06θ_2 tanh x_3 + d(t)
y   = x_1 + ω
```
`T = 20`, `Ω = 2π/5`, `Γ_3 = [sin Ωt, cos Ωt, sin 2Ωt]ᵀ`, `W_Γ = 10·I₃`,
poles `−0.6, −0.9, −1.2`, `γ_2 ≥ 0.36`, `γ_3 ≥ 0.49`.

Three seeds at `σ = 0.002`, held-out trajectories:

| quantity | s0 | s1 | s2 | verdict |
|---|---|---|---|---|
| `x_1` (measured) | 2.66e-03 | 2.63e-03 | 3.63e-03 | **reliable** |
| `d` L² | 1.11e-02 | 9.48e-03 | 1.45e-02 | **reliable** (8–12% of ‖d‖∞) |
| `θ_1` | 0.165 | 0.077 | 0.230 | seed-dependent |
| `θ_2` | 0.165 | 0.077 | 0.234 | seed-dependent |
| `x_2` | 0.156 | 0.073 | 0.218 | tracks `θ` |
| `L_Tot` | 2.246e-05 | 2.232e-05 | 3.733e-05 | — |

Disturbance coefficients on the noise-free run recover in observability order:
`a_1` **0.8%**, `a_2` 7.6%, `a_3` 24.1% — `a_3` is the 2Ω component, the least
visible at the output.

**What is solid:** the measured state and the disturbance. `d̂` reproduces
amplitude, shape and phase over all four periods, and is insensitive to which
basin the parameters land in — `â` is pinned by a well-conditioned final
residual, so it does not share their fate.

**What is not:** the state–parameter split. `θ` errors range 0.077–0.234 across
seeds (33–100% relative), and `s0`/`s1` differ 2.1× in `θ` while their
objectives differ by 0.6%. See the degeneracy analysis below.

Compared with `automatica_n4`, the shorter chain improves the disturbance
**4.1×** (one fewer integration between `d` and `y`) and halves the degenerate
family from two dimensions to one — both predicted in advance by
`eval/observability.py` and the null-space argument, respectively.

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

## What limits accuracy on this system — an experimental answer

Roughly thirty training runs were used to isolate what bounds PIBE's accuracy on
`automatica_n4`. The short answer: **it is the Remark 7 degeneracy, and it is
structural.** It is not noise, fitting, capacity, horizon, resolution or schedule.

### The degeneracy, concretely

At every cell `k <= n`, the physics residual

```
ẋ̂_{k-1} = x̂_k + f_{k-1}(·, θ̂_{k-1})
```

is one equation in two unknowns — the new state `x̂_k` and the parameter
`θ̂_{k-1}`. Because `x̂_k` is a *free function*, the residual can be driven to
zero for **any** `θ̂_{k-1}` by setting
`x̂_k = ẋ_{k-1} − f_{k-1}(x_{k-1}, θ_{k-1}+e)`. The compensating shift inherits
the time-variation of `∂f_{k-1}/∂θ_{k-1}` (22% here), and that is measurable:
predicted offset `+0.075` with scatter `4.0e-3`, observed `+0.084` / `1.83e-2`.

The bank's **only** closure is the final cell, where the auxiliary state is
pinned by its consistency target and `â` must fit `r_{n+1}` over the whole
horizon. But the surviving offsets sit in the **null space of that residual's
linear part**:

```
Σ_j COMPANION_j · c_j = +0.0006      (0.045% of the largest term)
```

so the closure is structurally blind to them. `Γ_4` has no DC component, so `â`
could not absorb a constant either — it does not need to.

The visible signature: the bank reconstructs the *shape* of every unmeasured
state to 1–3%, offset by one constant per coordinate.

| | RMSE | mean offset | RMSE after removing it |
|---|---|---|---|
| `x_1` | 1.53e-03 | +0.001 | 1.39e-03 |
| `x_2` | 7.91e-02 | +0.079 | **8.68e-03** |
| `x_3` | 2.92e-01 | −0.291 | **1.88e-02** |
| `x_4` | 3.19e-01 | +0.318 | **3.04e-02** |

### Every lever, tested

| lever | range tested | effect on `θ` |
|---|---|---|
| noise `σ` | 0.01 → 0.002 → **0** | persists at exactly zero noise |
| horizon `T` | 5 / 20 / 40 | trades `d̂` against `θ̂`; fixes neither |
| time harmonics `K` | 8 / 10 / 16 / 40 | none, once `1/k` scaling is applied |
| trajectories `P` | 8 / 16 / 64 / 256 | none |
| capacity | latent 32 → 96, decoder 64³ → 128³ | none (worse) |
| schedule | local-only / Algorithm 1 / joint | none |
| `λ` | 1 / 10 | 10 destabilises |
| final-phase length | 1.5k → 42k iterations | improves the loss, not `θ` |

The cleanest single experiment is `nf_p8` vs `noisefree`, both noise-free:
cutting the amortization burden from 51 trajectories to 6 improved the
objective **7.2×** and `x_1` **2×**, while `θ_3`'s error stayed *identical* to
five digits. **Fit quality and identification are decoupled.**

### The outcome is bimodal, and the loss cannot tell you which basin you are in

Four seeds of the same configuration (sigma = 0.002, T = 20, K = 10):

| seed | `L_Tot` | `theta_1` | `theta_2` | `theta_3` | `x_3` | `x_1` | basin |
|---|---|---|---|---|---|---|---|
| 3 | **7.59e-05** | .064 | .245 | **.420** | .281 | 5.4e-3 | degenerate |
| 2 | 7.79e-05 | .038 | .066 | .086 | .081 | 5.1e-3 | good |
| 0 | 8.46e-05 | .058 | .093 | .093 | .097 | 5.6e-3 | good |
| 1 | 8.98e-05 | .072 | .226 | **.420** | .268 | 5.4e-3 | degenerate |

Two of four seeds converge to a degenerate solution with `theta_3` pinned at the
boundary of `Theta_3`. The measurement fit is *identical* across all four
(`x_1` within 5.1-5.6e-3), and — most importantly — **the seed with the lowest
objective has the worst parameters.** The objective varies 18% while
`theta_3`'s error varies 5x, and the correlation is negative.

The practical consequence: model selection by training loss, validation loss, or
measurement fit will not find the identifiable solution, because none of them
distinguish the basins. Only ground truth does. This is the operational form of
Assumption 4 being a *hypothesis*: there is no post-training quantity in the
framework that verifies it.

### Why this is the expected answer

The paper says so. Remark 7 constructs this family explicitly; **Remark 10**
states that ruling it out "requires a coercivity inequality on the stacked
state–physics operator, or independent information on each new state" — neither
of which the PIBE objective supplies. Assumption 4 is labelled "a post-training
accuracy hypothesis, not a consequence of the Universal Approximation Theorem".

`eval/oracle.py` confirms the objective is not at fault: the true solution
always scores *better*, so the losses and residuals are correct and the target
is representable. The optimizer simply has no gradient along the flat direction.

### What would actually fix it

Not a training change. Either extra information (an additional measured
coordinate; a known initial condition pinning `x(0)`; a constraint on one
intermediate state), or a system whose degenerate direction is not in the null
space of the final residual. A useful check before running anything is
`eval/observability.py`, which reports whether each disturbance coefficient is
even visible at the output — on this system, `a_3, a_4` sit at 0.18σ and 0.15σ
at `σ = 0.01`, i.e. below the noise, which Eq. (3)'s `W_Γ = 10·I₄` does not
reveal.

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
