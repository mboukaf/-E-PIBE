# PIBE — implementation and what it took to make it work

Companion note to *Joint Estimation of State and Parameters for a Class of
Nonlinear Disturbed Systems with Unknown Noise*, covering the PIBE framework
(Section 3.1, Gaussian / truncated-Gaussian and noise-free cases).

This document has two halves:

1. **The architecture as built** — every layer, activation, dimension and
   constraint, with the equation each piece implements.
2. **What had to be added** — five changes, four of them consequences of a
   single mechanism, and the measurements that motivated each.

Nothing below is a design preference. Every departure from a literal reading of
the paper is justified by a measurement reported alongside it.

---

# Part 1 — The architecture

## 1.1 Problem

The system class is Eq. (1): triangular canonical form, single measured output.

```
ẋ_j = x_{j+1} + f_j(x_1..x_j, θ_1..θ_j),   j = 1..n-1
ẋ_n = f_n(x, θ) + d(t)
y   = x_1 + ω(t)
d(t) = Γ_q(t)ᵀ a + r_q(t)
```

Unknown: the states `x_2..x_n`, the constant parameters `θ ∈ Θ ⊂ ℝ^{n-1}`, and
the disturbance coefficients `a ∈ 𝒜 ⊂ ℝ^q`. Known: `y` on a sampled grid, the
nonlinearities `f_j`, the basis `Γ_q`, and the compact admissible sets.

## 1.2 The bank

Cells `k = 2 … n+1`. Each is a trajectory-conditioned PINN
`Φ^k = (ℰ_k, 𝒮_k, 𝒬_k)`, Eq. (20).

| cell | reconstructs (`x_prev`) | estimates (`x_new`) | head |
|---|---|---|---|
| `k = 2` | `x̂²₁` — compared against `y` | `x̂²₂` | `θ̂²₁` |
| `k = 3..n` | `x̂^k_{k-1}` — vs. `x̂^{k-1}_{k-1}` | `x̂^k_k` | `θ̂^k_{k-1}` |
| `k = n+1` | `x̂^{n+1}_n` — *auxiliary* | — | `â ∈ ℝ^q` |

Reported estimates: `x̂_1 := x̂²₁`, `x̂_k := x̂^k_k`, `θ̂_{k-1} := θ̂^k_{k-1}`,
`d̂ = Γ_qᵀâ`. The final cell's `x̂^{n+1}_n` is **not** a second estimate of `x_n`;
it exists only to evaluate the last physics residual.

**Why every cell has two decoder outputs.** The first is the cell's own
reconstruction of the coordinate *below* it, which the consistency data term
anchors; the second is the new coordinate, which is free. Remark 7 turns
entirely on this distinction, and so does §2.2 below.

## 1.3 Layers, exactly as built

Figures are for `automatica_n3v2`: `n = 3`, `N = 201` samples, `q = 3`,
latent `r_k = 32`, horizon `T = 20`.

### Encoder `ℰ_k` — Eq. (17)

Consumes the *entire sampled trajectory*, Eq. (15):

```
U_{k-1} = col( Y, X̂²₂, …, X̂^{k-1}_{k-1}, θ̂²₁, …, θ̂^{k-1}_{k-2} ) ∈ ℝ^{N(k-1)+(k-2)}
```

```
D_{k-1}  fixed diagonal normalization      (Eq. 61, non-trainable buffer)
Linear(d_{k-1} → 128) → Tanh
Linear(128 → 128)     → Tanh
Linear(128 → 32)                            = z_{k-1}
```

| cell | `d_{k-1}` | params |
|---|---|---|
| 2 | 201 | 46,496 |
| 3 | 403 | 72,352 |
| 4 | 605 | 98,208 |

`D_{k-1}` divides every sampled block by `√N · s_{x_j}` and every parameter by
`s_{θ_j}`, where the reference scales are the half-widths of the admissible
intervals. The `1/√N` is what makes the encoder's Lipschitz constant
grid-independent, so a bound uniform in `N` only additionally requires `L_{E_k}`
to stay bounded under grid refinement. Concretely for cell 3 the weight vector
begins `[0.0282, 0.0282, …]` (= `1/(√201 · 2.5)`) and ends `[0.0882, 2.5]`
(= `1/(√201 · 0.8)` and `1/0.4`).

### Decoder `𝒮_k` — Eqs. (18)/(33)

```
t ∈ [0,T]
  → affine map to [-1,1]
  → Fourier lift  [ t, sin(kπt)/k, cos(kπt)/k ]_{k=1..10}      21 dims
  → concat with z (32)                                          53 dims
Linear(53 → 64) → Tanh
Linear(64 → 64) → Tanh
Linear(64 → 64) → Tanh
Linear(64 → out_dim)
  → c + r·tanh(·)   box reparameterization onto 𝒳
```

`out_dim = 2` for cells 2..n, `1` for the final cell. 11,906 params (11,841 for
the final cell).

Boxes actually installed:

| cell | decoder box | head box |
|---|---|---|
| 2 | `[-2.5, 2.5] × [-0.8, 0.8]` | `[-0.4, 0.4]` |
| 3 | `[-0.8, 0.8] × [-0.8, 0.8]` | `[-0.4, 0.4]` |
| 4 | `[-0.8, 0.8]` | `[-0.18,0.18]×[-0.18,0.18]×[-0.12,0.12]` |

### Head `𝒬_k` — Eqs. (19)/(34)

```
Linear(32 → 64) → Tanh
Linear(64 → 64) → Tanh
Linear(64 → out_dim)
  → c + r·tanh(·)   box onto Θ_{k-1}  (or 𝒜 for the final cell)
```

`out_dim = 1` for cells 2..n, `q = 3` for the final cell. ~6,340 params.
**The head never receives `t`**, so `θ̂` is constant over `[0,T]` by
construction rather than by penalty — the unknown parameters of Eq. (1) are
constant, and the architecture enforces exactly that.

**Total: 271,850 parameters** across the three cells.

### Activation choice

`tanh` throughout, and this is a hard requirement, not a preference: Eq. (21)
computes `ẋ̂ = ∂_t 𝒮_k(t, z)` by automatic differentiation, so the decoder must
be `C²` in `t`. ReLU-family activations are **rejected at construction** rather
than silently yielding an almost-everywhere derivative. The box map `c + r tanh`
is likewise smooth, so it does not spoil the regularity.

## 1.4 Losses

Local objective per cell, Eqs. (22)/(28)/(37): `L^k_Loc = L^k_Data + λ L^k_Physics`
with `λ = 1`.

**Data terms** — one rule: cell `k` compares its reconstruction of coordinate
`k-1` against the target supplied from upstream.

```
L²_Data      = (1/N) Σ ( y(t_i) − x̂²₁(t_i) )²                    (23)
L^k_Data     = (1/N) Σ ( x̂^{k-1}_{k-1}(t_i) − x̂^k_{k-1}(t_i) )²  (29)
L^{n+1}_Data = (1/N) Σ | x̂^n_n(t_i) − x̂^{n+1}_n(t_i) |²          (38)
```

Only the first is a measurement fit; the rest are deterministic inter-cell
consistency penalties. No Gaussian propagation through the bank is assumed.

**Physics terms** — mean squared residual on the collocation grid
(`N_r = 401`), Eqs. (24)/(30)/(39), with residuals (25)/(31)/(40):

```
r_2      = ẋ̂²₁ − x̂²₂ − f_1(x̂²₁, θ̂²₁)
r_k      = ẋ̂^k_{k-1} − x̂^k_k − f_{k-1}(ℐ^x_{k-2}, x̂^k_{k-1}, ℐ^θ_{k-2}, θ̂^k_{k-1})
r_{n+1}  = ẋ̂^{n+1}_n − f_n(x̂^{n+1}, θ̂) − Γ_q(t)ᵀ â
```

The upstream arguments follow Eq. (26): coordinate 1 from cell 2's
*reconstruction*, coordinate `j ≥ 2` from cell `j`'s *new* output, and `θ_j`
from cell `j+1`'s head. Note `x_{k-1}` in `r_k` comes from the **current**
cell's reconstruction, not from cell `k-1`.

**Global loss** — Eqs. (27)/(42), used only during end-to-end fine-tuning:

```
L^k_Tot = Σ_{m=2..k} exp(−(k−m)/k) · L^m_Loc
```

## 1.5 Time derivatives

`ẋ̂` is obtained by autograd, with two non-obvious requirements:

- **Per-trajectory time tensors.** The grid is materialized as `(B, M, 1)` with
  independent storage per entry. An `expand`-ed grid shares storage, and the
  reverse pass would accumulate `Σ_b ∂_t x̂[b,m]` into every row — every
  trajectory would get the batch-summed derivative.
- **`create_graph=True`.** The physics loss is a function of `ẋ̂`, so
  `∇_Θ L_Physics` differentiates *through* the derivative. Without the retained
  graph the physics term contributes no gradient at all.

Differentiating `output.sum()` is valid because `output[b,m]` depends on `t`
only through `t[b,m]` — the latent is constant along the time axis.

## 1.6 Training — Algorithm 1

Outer loop over cells; inner loop split at `N_par`:

- **`i < N_par` (local).** Cells `2..k-1` frozen and evaluated under `no_grad`,
  so `U_{k-1}` is built from detached outputs; only `Θ^k` is updated, against
  `L^k_Loc`.
- **`i ≥ N_par` (global).** Cells `2..k` re-run with the graph intact; all
  parameters updated against `L^k_Tot`.

Settings used: `N_tot = 6000`, `N_par = 3000`, plus `final_global_iters =
30000` extra end-to-end iterations on the last cell (§2.3). Adam, `lr_local =
1e-3`, `lr_global = 5e-4`, cosine annealed, gradient clipping at 1.0, batch of
16 trajectories, float64.

---

# Part 2 — What had to be added, and why

Five changes. Four of them follow from **one** mechanism, described first.

## 2.0 The mechanism: Remark 7's degeneracy, made concrete

At every cell `k ≤ n` the physics residual

```
ẋ̂_{k-1} = x̂_k + f_{k-1}(·, θ̂_{k-1})
```

is **one equation in two unknowns** — the new state `x̂_k` and the parameter
`θ̂_{k-1}`. Because `x̂_k` is a *free function*, the residual can be driven to
zero for **any** `θ̂_{k-1}` by taking

```
x̂_k(t) = ẋ_{k-1}(t) − f_{k-1}(x_{k-1}(t), θ_{k-1} + e)
```

The compensating shift is `c_k(t) = −e · ∂f_{k-1}/∂θ_{k-1}(t)`. Measured on the
original system: predicted offset `+0.075` with scatter `4.0e-3`, observed
`+0.084` / `1.83e-2`.

**The bank's only closure is the final cell**, where the auxiliary state is
pinned by its consistency target and `â` must fit `r_{n+1}` across the whole
horizon — over-determined, hence identifying. But the shifts that survive lie
in the **null space of that residual's linear part**. Measured on the
fourth-order system:

```
Σ_j COMPANION_j · c_j = +0.0006      →  0.045% of the largest term
```

So the closure is structurally blind to them. And `Γ_q` has no DC component, so
`â` could not absorb a constant even if required.

The visible signature: the bank recovers the *shape* of every unmeasured state
to 1–3%, offset by one constant per coordinate.

| | RMSE | mean offset | RMSE after removing it |
|---|---|---|---|
| `x_1` | 1.53e-03 | +0.001 | 1.39e-03 |
| `x_2` | 7.91e-02 | +0.079 | **8.68e-03** |
| `x_3` | 2.92e-01 | −0.291 | **1.88e-02** |
| `x_4` | 3.19e-01 | +0.318 | **3.04e-02** |

## 2.1 Bounded output reparameterization — and separating two sets

The paper constrains decoders and heads to `𝒳_j`, `Θ_j`, `𝒜`; we use
`π(z) = c + r·tanh(z)`.

**The addition:** the *sampling* range of `θ` and the *admissible* set `Θ_j`
must be different objects. Taking them equal (both `[-0.25, 0.25]`) put a true
`θ_1 = 0.235` at 94% of the half-width, needing pre-activation
`atanh(0.94) ≈ 1.74`; any overshoot saturates `tanh`, the gradient dies, and
the estimate pins to the boundary permanently. Observed directly: `θ̂_1 =
0.25000` with cross-trajectory spread **exactly 0**.

Fix: `THETA_SAMPLING_BOUND = 0.25`, `THETA_BOUND = 0.4`.

**This does not fix the degeneracy** — the direction is flat, so the estimate
simply runs to whatever the new boundary is (`θ̂_1 = 0.400`, spread 4.8e-07). It
removes a *dead gradient*, nothing more. Worth stating because the two failure
modes look identical in a metrics table and are distinguished only by the
spread being ~1e-7.

## 2.2 Fourier time features, scaled by `1/k`

**Why needed.** The disturbance enters the last equation and is seen only
through `x_1`, so recovering it amounts to differentiating `y` `n` times. An
under-resolved `x̂_1` therefore destroys `d̂` even when its own error looks
small. On the fourth-order system:

```
|G(jΩ)| from d to y            = 0.1728
signature of d in the output   = 0.231 × 0.1728 = 0.0399
x̂₁ fit error                   = 0.0147   ← 37% of it
```

A plain `tanh` MLP in `t` has a spectral bias against exactly the oscillation
the disturbance imprints. Lifting time into harmonics fixed it: `x_1` improved
2.4× and the disturbance error 2.5×, together.

**The `1/k` scaling is essential, and was found by failure.** With
unit-amplitude features, `d/dt sin(kπt) = kπ cos(kπt)`, so harmonic `k`
contributes `k` times as strongly to `ẋ̂` as to `x̂`. At `K = 40` the fastest
features dominate the physics residual and conditioning collapses — measured:
`x_1` got **11× worse** than at `K = 10`. Scaling by `1/k` equalizes each
harmonic's influence on the derivative, which is the quantity the loss sees.
A test measures this: unscaled, harmonic 16's derivative is 16.0× harmonic 1's;
scaled, the ratio is 1.00. Expressiveness is unchanged — the network learns a
correspondingly larger weight.

`K = 10` with scaling is the setting used. Raising it further does **not** help
identification: at `K = 24` (same seed) the objective improves 1.8x to 1.20e-05
— the best of any run — while `θ` gets 1.7x *worse* (8.64e-03 vs 5.03e-03). The
extra resolution buys a better fit to the initial transient and nothing else,
which is the fit/identification decoupling of §2.6 appearing even in the
well-conditioned regime.

## 2.3 `final_global_iters` — budget where the bank actually closes

Since cells `2..n` are each individually degenerate, the *only* constraint is
the final cell, and its information reaches the rest only through
`L^{n+1}_Tot`. Under a uniform budget that phase is starved.

Measured effect of the final cell's global phase alone: the objective fell from
**1.1257 → 1.28e-3** in 1000 iterations. Setting `final_global_iters = 30000`
made this the dominant part of training.

A related ablation, `local_only` (skip fine-tuning entirely), is retained as a
diagnostic. It is *worse* overall (`L_Tot` 1.11 vs 7.6e-4) but **honest**: with
upstream frozen, `â` cannot absorb upstream error, so the residual reports it
instead of hiding it by driving `ẋ̂_n − f_n → 0` and `â → 0`.

## 2.4 `eval/oracle.py` — scoring the true solution

A diagnostic, not a training component, but the single most useful thing added.
It evaluates `L^k_Loc` and `L^{n+1}_Tot` at the exact `(x, θ, a)` and compares
against the trained bank, separating two failure modes that look identical from
the loss curve:

- **truth scores better** → optimisation gap; the levers are schedule and ranges.
- **truth scores no better** → the objective prefers the wrong answer; only
  extra information helps.

Throughout this work the verdict was always the former, which is what justified
continuing to tune rather than concluding the losses were wrong.

Two reference points it prints: at the truth all physics residuals vanish
(~1e-30), and cell 2's data term equals the realized noise variance — a floor no
honest estimator beats. A *trained* cell-2 data term below that floor means the
decoder is fitting noise.

## 2.5 `eval/observability.py` — a check Eq. (3) does not perform

Eq. (3) requires `W_Γ ⪰ γ_d I_q`, a property of the basis *as functions of
time*. It says nothing about whether the plant passes those functions to the
output. On the fourth-order system `W_Γ = 10·I₄` — perfectly conditioned — while:

```
disturbance observability  (sigma = 0.01, N = 201)
  i |    settled |  signature |     SNR |  eff SNR |
  1 |  7.313e-02 |  1.097e-02 |    1.10 |    15.55 |
  2 |  7.508e-02 |  1.126e-02 |    1.13 |    15.97 |
  3 |  1.526e-02 |  1.832e-03 |    0.18 |     2.60 |  <- below the noise floor
  4 |  1.230e-02 |  1.476e-03 |    0.15 |     2.09 |  <- below the noise floor
```

Half the basis was **below the measurement noise**. The module linearizes the
system, drives it with each basis function alone, and reports the trace each
coefficient leaves in `y`. It distinguishes settled from transient response,
because the startup transient is a broadband kick amplified by the chain's DC
gain and flatters the fast components.

This is a well-posedness test that should be run *before* training.

## 2.6 The change that actually resolved it: sensitivity design

Everything above improved the fit. **None of it fixed the parameters.** The
elimination was systematic:

| lever | range tested | effect on `θ` |
|---|---|---|
| noise `σ` | 0.01 → 0.002 → **0** | persists at exactly zero noise |
| horizon `T` | 5 / 20 / 40 | trades `d̂` against `θ̂` |
| harmonics `K` | 8 / 10 / 16 / 40 | none once scaled |
| trajectories `P` | 8 / 16 / 64 / 256 | none |
| capacity | latent 32→96, decoder 64³→128³ | none (worse) |
| schedule | local-only / Alg. 1 / joint | none |
| `λ` | 1 / 10 | 10 destabilises |

The cleanest experiment: noise-free, cutting the amortization burden from 51
trajectories to 6 improved the objective **7.2×** and `x_1` **2×**, while
`θ_3`'s error stayed **identical to five digits**. Fit quality and
identification are decoupled.

Worse, across four seeds of one configuration the run with the **lowest
objective had the worst parameters** — `L_Tot` varied 18% while `θ_3` varied 5×,
correlated negatively. No post-training quantity distinguishes the basins.

### The insight

Since `c_k(t) = −e·∂f_{k-1}/∂θ_{k-1}(t)`, the compensation is *constant* exactly
when the sensitivity is constant — and a constant shift is what the final
residual's companion row annihilates. So the degeneracy is severe precisely
when `∂f/∂θ` is near-constant.

The original design chose `∂f_1/∂θ_1 = 0.8 + 0.2cos x_1`, deliberately bounded
away from zero (correctly avoiding forms like `θ_i x_i` whose sensitivity
vanishes). But its stated floor of 0.6 holds over the admissible *box*;
**on the trajectories the system actually produces it spans only [0.93, 0.98] —
a 5% variation.** The compensation was nearly constant, hence nearly free.

> **`γ_k > 0` (Assumption 6) is necessary but not sufficient.** A sensitivity
> bounded away from zero yet nearly constant satisfies it while leaving the
> state–parameter split practically unidentifiable. What matters additionally is
> the **variation** of `∂f_j/∂θ_j` along realized trajectories, and that
> different parameters key off **different coordinates**.

### The measurement

Reconstruct the chain exactly under a parameter error `e_1`, let `θ_2` and `â`
absorb whatever they can, and record the residual that remains. Its curvature
in `e_1` *is* the identifiability.

| design | `s_1` on-trajectory | residual at `e_1 = 0.10` | gain |
|---|---|---|---|
| `0.8 + 0.2cos x_1`, `s_2(x_1,x_2)` | [0.93, 0.98] | 3.73e-05 | 1.0× |
| `1.3 + 0.7cos x_1`, `s_2` unchanged | [1.45, 1.91] | 7.54e-04 | 20.2× |
| `1.3 + 0.7cos x_1`, `s_2 = 1.2+0.6sin 2x_1` | [1.40, 1.91] | 1.21e-04 | 3.2× |
| **`1.3 + 0.7cos x_1`, `s_2 = 1.2+0.6cos 2x_2`** | [1.35, 1.91] | **9.60e-04** | **25.7×** |

Row 3 vs row 4 is the second half of the lesson: keying `s_2` to `x_1` — the
same coordinate as `s_1` — lets the two compensations re-align and throws away
most of the gain.

### The result

```
ẋ_1 = x_2 − 0.35 tanh x_1 + θ_1(1.3 + 0.7 cos x_1)      ← s_1 keyed to x_1
ẋ_2 = x_3 − 0.30 tanh x_2 + 0.10 sin x_1
                           + θ_2(1.2 + 0.6 cos 2x_2)     ← s_2 keyed to x_2
X_1 = [−2.5, 2.5]     (stronger θ-coupling drives x_1 to ~2.02)
```

`f_3`, the poles, `Γ_3`, `T`, `Ω` and all coefficient ranges unchanged.

Three seeds, held-out trajectories, `σ = 0.002`:

| | rev 1 (best of 3) | rev 2 s0 | rev 2 s1 | rev 2 s2 |
|---|---|---|---|---|
| `θ_1` (true 0.235) | 0.0767 | **0.00503** | 0.0201 | 0.0247 |
| `θ_2` (true 0.104) | 0.0773 | **0.00452** | 0.0179 | 0.0217 |
| `x_2` (unmeasured) | 0.0731 | **0.0110** | 0.0320 | 0.0386 |
| `x_3` (unmeasured) | 0.0649 | **0.0192** | 0.0327 | 0.0393 |
| `d` L² * | 9.48e-03 | 7.11e-03 | 1.06e-02 | 1.28e-02 |
| `γ_2` / `γ_3` | 0.36 / 0.49 | 0.99 / 2.07 | — | — |

\* These disturbance figures were obtained with a *single shared* `a` and are
superseded by §2.7 — with `a` sampled per trajectory they degrade about 13x.
The parameter columns are unaffected in kind and degrade only ~2.5x.

The **worst** new seed beats the **best** old one by 3x on every parameter, and
no seed shows the boundary pin: head spreads are ~2e-2, not ~1e-7. Best seed
reaches 2.1% and 4.4% relative error on `θ`.

The offsets are gone: `x_2` mean offset **+0.003**, `x_3` **−0.010** (were
−0.155 / +0.127). Removing the offset no longer changes the RMSE (1.586e-2 →
1.548e-2), where before it collapsed it 10×. The error structure changed
qualitatively — the residual is now concentrated in the first ~1.5 s, i.e. the
initial transient, not a systematic bias.

### Why: the fourth-order objective does not identify `theta`

The definitive measurement is a **profile of the objective**
(`scripts/profile_theta.py`): clamp `theta_1` at an offset, let every other
weight — including `theta_2`, `theta_3` and `a` — re-optimize to compensate,
and record the best achievable `L_Tot`. Each point warm-starts from the same
converged checkpoint so the points are comparable.

| `theta_1` offset | 3D (`n3v2`) | 4D (`n4` rev1) |
|---|---|---|
| −0.100 | 3.71e-05 | 4.13e-05 ← *minimum* |
| −0.050 | 1.38e-05 | 4.23e-05 |
| **0.000** | **5.39e-06** ← minimum | 4.40e-05 |
| +0.050 | 1.03e-05 | 4.53e-05 |
| +0.100 | 1.90e-05 | 4.55e-05 |
| ratio | **6.89x** | **0.94x** |

**3D is identifiable**: a clean minimum at the true value, rising 6.89x.
**4D is not**: flat to within ±3%, and *monotone* — the objective mildly
**prefers a wrong parameter**. There is nothing for an optimizer to find.

That single fact explains every fourth-order observation:

- **Seed variance dominates.** Three seeds of the identical rev-1 configuration
  give `theta_1` errors 0.058 / 0.318 / 0.034 — a 9.4x spread. With a flat
  profile, where a run lands is chance.
- **Loss does not track accuracy.** `s2` has both the lowest loss and the best
  parameters; `s1` has a *lower* loss than `s0` and far worse parameters.
- **No design change helped**, because none of them were addressing the
  binding constraint.

Good fourth-order results are therefore attainable but not *reproducible*:
seed 2 reaches `theta` errors (0.034, 0.013, 0.0099), the best of any 4D run,
by landing near the truth inside the flat region. Reporting that seed alone
would be misleading.

### Methodological note: three failed diagnostics before a valid one

Worth recording, because each failure mode is easy to repeat.

1. **Curvature proxy** — reconstructs the chain exactly, so it cannot see the
   slack the network's *soft* consistency terms provide. Over-predicted 4D
   gains that training never delivered.
2. **Profile along `(1,1,1)`** — clamping every parameter at a *common* offset
   probes a direction nearly orthogonal to the degenerate one, whose components
   have opposite signs. Reported a steep 55.8x for a system that is in fact
   flat.
3. **Profile trained from scratch** — 4000 iterations per point measures
   convergence luck, not geometry. Caught by the 3D control, where `e = 0`
   scored *worse* than `e = ±0.12`, which is impossible for a profile.

Only the fourth version — one parameter clamped, others free, all points
warm-started from a common checkpoint — produced a profile with its minimum at
the truth on the control. **Running a case whose answer is already known is what
caught errors 2 and 3.**

## 2.7 Correction: the disturbance must vary across trajectories

An earlier version of this work generated **one** disturbance shared by every
trajectory. That is a mis-specification of the problem, not merely a narrow
test setting.

Eq. (2)'s `a` is an *unknown* the estimator has to infer from `y`, and the paper
states that `â^ℓ` "may differ between trajectories" — which is only meaningful
if the true `a` differs too. With a single shared `a`, the coefficient head
`𝒬_{n+1}: z_n → â` can drive the objective down by emitting a **constant**, and
the disturbance-estimation problem is never posed at all.

Sampling `a` independently per trajectory (`sample_disturbance_per_trajectory`,
now the default) changes the result decisively:

| | shared `a` | per-trajectory `a` (s0 / s1) |
|---|---|---|
| **`d` L²** | 7.11e-03 | **9.31e-02 / 1.02e-01** — 13x worse |
| `θ_1` | 5.03e-03 | 1.35e-02 / 1.78e-02 |
| `θ_2` | 4.52e-03 | 1.10e-02 / 1.51e-02 |
| `x_1` | 2.17e-03 | 8.12e-03 / 9.51e-03 |

Relative `d` error is 79% on average, and the head is visibly collapsing toward
a constant rather than inferring:

```
â   spread across trajectories:  0.0630  0.0420  0.0109
true a spread:                   0.1064  0.1152  0.0759
```

`â` varies only 59% / 36% / 14% as much as the truth, worst for `a_3` — which
is also the least observable component (the 2Ω harmonic, see §2.5).

**The 7.11e-03 disturbance figure was substantially memorization and must not be
reported.** The parameter result survives the correction: `θ` degrades only
~2.5x and stays at 5.8% / 10.6% relative error, which is the right outcome,
since `θ` genuinely *is* a single system constant while `a` is not.

`sample_theta_per_trajectory` exists but is off by default: `θ` is a constant of
the system, so varying it asks the bank to identify an *unseen system*, which is
strictly harder than Eq. (1) poses.

### Two defaults changed under existing checkpoints

Twice in this work a default was changed while trained checkpoints existed —
`time_feature_scaling` (§2.2) and `sample_disturbance_per_trajectory` here. In
both cases reloading an old checkpoint silently reinterpreted it: weights
learned against unscaled features evaluated with scaled ones, and weights
trained on a shared `a` scored against per-trajectory data. The first produced
an apparent catastrophic-overfitting result that was purely an artifact.

Every run's `config.resolved.yaml` now records both flags explicitly, so a
checkpoint is always evaluated under the settings it was trained with. The
general lesson: **a config that omits a setting inherits today's default, not
the one in force when it ran.** Resolved configs must be exhaustive.

### It did not transfer to the fourth-order system

The same procedure was applied to `automatica_n4`: measure candidate designs,
pick the highest curvature, retrain. **It made things worse.**

| | `n4` rev1 (best) | `n4v2` m=(1,6,10) s0 | s1 | `n4v3` m=(1,4,6) s0 |
|---|---|---|---|---|
| predicted gain | 1.0x | 27.5x | 27.5x | 9.1x |
| `θ_1` | **0.058** | 0.152 | 0.377 | 0.144 |
| `θ_2` | **0.093** | 0.236 | 0.037 | 0.261 |
| `θ_3` | **0.093** | 0.367 | 0.391 | 0.398 |
| `x_1` | 5.56e-03 | 6.74e-03 | 1.13e-02 | 6.56e-03 |
| `L_Tot` | 8.46e-05 | 1.04e-04 | 8.73e-04 | 8.16e-05 |

Note especially `n4v3`: essentially the **same** objective and the same `x_1`
fit as revision 1, yet `θ` three times worse. So this is not a fittability
penalty from the faster sensitivities — identification degraded at equal fit
quality, the opposite of the prediction.

**The curvature proxy is therefore not reliable for the longer chain.** It
predicted 25.7x for `n = 3` and delivered; it predicted 9.1-27.5x for `n = 4`
and the trained result moved the other way. Candidate explanations, none
verified:

- the fourth-order degenerate family is 2-D rather than 1-D, so there are more
  escape directions than a one-parameter reconstruction models;
- the network's consistency terms are soft, giving slack the exact-chain
  reconstruction does not represent;
- three successive numerical differentiations make the `n = 4` measurement
  noisier than the `n = 3` one.

**Status of the criterion: validated on the third-order system; inapplicable to
the fourth-order one, whose objective does not identify `theta` at all** (the
profile above). The fourth-order failure is therefore *not* evidence against the
criterion — the criterion addresses the shape of a minimum that, at `n = 4`,
does not exist. It should be presented as a design heuristic supported by the
third-order result and the mechanism in §2.0, not as a general principle. The
honest recommendation for the paper is the third-order system, where the
criterion was derived, verified across three seeds, and where the disturbance
is also 4.1x better recovered because of the shorter chain.

---

# Part 3 — Summary of departures from the paper

| # | change | status | justification |
|---|---|---|---|
| 1 | `Θ_j` wider than the `θ` sampling range | implementation detail | removes a dead gradient at the tanh boundary; observed spread ~1e-7 |
| 2 | Fourier time features, `1/k`-scaled | architectural | `x̂_1` resolution gates `d̂`; measured 2.4× / 2.5× |
| 3 | `final_global_iters` | schedule | the final cell is the only closure; 1.13 → 1.28e-3 |
| 4 | oracle + observability diagnostics | tooling | separate optimisation from identifiability failure |
| 5 | **sensitivity design criterion** | **methodological, `n = 3` only** | `γ_k > 0` insufficient; 25.7× measured and confirmed on `n = 3`; **did not transfer to `n = 4`** |

Item 5 is the one that belongs in the paper as a contribution — with its scope
stated honestly. It gives Remark 7 a constructive resolution on the third-order
system, strengthens Assumption 6 into something a designer can act on, and the
mechanism (a constant compensating shift lying in the null space of the final
residual) is exact. But the curvature proxy used to *choose* a design did not
predict the fourth-order outcome, so the criterion is evidenced by one system,
not established in general.

**A caveat on framing.** `automatica_n3v2` was designed *because* it is
identifiable, so it should not be presented as a neutral benchmark. The honest
framing is also the stronger one: the design criterion was derived from the
failure mode, then verified — rather than a system being chosen and happening
to work.

---

# Reproducing

```bash
pip install -e ".[dev]"
python scripts/train_pibe.py --config configs/automatica_n3v2.yaml
python scripts/plot_results.py --run outputs/n3v2
pytest -q                  # 211 tests
pytest -q -m 'not slow'
```

Each run prints the observability table before training and the oracle
comparison after it. Figures land in `outputs/<run>/figures/`.
