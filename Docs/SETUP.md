---
title: Experimental setup — system, values, implementation
tags: [pibe, setup, reference, benchmark]
updated: 2026-09-10
---

# Experimental setup

Single reference for what is being estimated, with what numbers, and how. Every
value here is read from the code rather than transcribed, and the reference run
throughout is **`outputs/big_w10_v2_s1`** — the one that recovered all three
disturbance coefficients.

Companion documents: [[PIBE implementation]] for the layer-by-layer architecture
and its derivations, [[EPIBE]] for the energy-based extension and why it did not
earn its place, [[Cluster postmortem]] and [[SLURM batch jobs]] for the compute
side.

---

## 1. The estimation problem

Systems in the triangular (observability) canonical form of Eq. (1):

$$
\begin{aligned}
\dot x_j &= x_{j+1} + f_j(x_1,\dots,x_j;\ \theta_1,\dots,\theta_j), \quad j = 1,\dots,n-1 \\
\dot x_n &= f_n(x, \theta) + d(t) \\
y &= x_1 + \omega(t)
\end{aligned}
$$

One scalar measurement, of the **first** coordinate only. The disturbance enters
the **last** equation only. Everything else — the other `n−1` states, the `n−1`
constant parameters, and the disturbance — must be inferred from `y`.

The disturbance is decomposed as in Eq. (2):

$$ d(t) = \Gamma_q(t)^\top a + r_q(t) $$

with `Γ_q` a fixed known basis and `a` unknown constants. In every experiment
below `r_q ≡ 0` (`remainder_amplitude: 0.0`), so the disturbance is exactly
representable and `ε_{d,q} = 0`.

---

## 2. The benchmark system: `automatica_n3v2`

Third order, `n = 3`, so two unknown parameters and a bank of three cells.

$$
\begin{aligned}
\dot x_1 &= x_2 - 0.35\tanh x_1 + \theta_1\,(1.3 + 0.7\cos x_1) \\
\dot x_2 &= x_3 - 0.30\tanh x_2 + 0.10\sin x_1 + \theta_2\,(1.2 + 0.6\cos 2x_2) \\
\dot x_3 &= -0.648\,x_1 - 2.34\,x_2 - 2.7\,x_3 + 0.15\sin x_1 + 0.08\tanh(x_1x_2) \\
         &\qquad + 0.08\,\theta_1\sin x_2 + 0.06\,\theta_2\tanh x_3 + d(t) \\
y &= x_1 + \omega
\end{aligned}
$$

The companion coefficients `(0.648, 2.34, 2.7)` place the linear poles at
**−0.6, −0.9, −1.2** (verified: `roots([1, 2.7, 2.34, 0.648])`). Linearised about
the origin at `θ_true`, the eigenvalues are `−0.4975` and `−1.4231 ± 0.7974i` —
the nonlinearity moves them but leaves the system stable.

### Values

| quantity | value |
|---|---|
| state bounds `𝒳_j` | `(−2.5, 2.5)`, `(−0.8, 0.8)`, `(−0.8, 0.8)` |
| initial-condition box `χ₀` | `(−0.8, 0.8)`, `(−0.5, 0.5)`, `(−0.35, 0.35)` |
| parameter bounds `Θ_j` | `±0.4` both |
| θ sampling half-width | `0.25` (`THETA_SAMPLING_BOUND`) |
| `θ_true` (`theta_seed: 0`) | `[0.23502650090327654, 0.10390993219989397]` |
| reference scales `s_x` | `[2.5, 0.8, 0.8]` (half-widths of `𝒳_j`) |
| reference scales `s_θ` | `[0.4, 0.4]` |

> [!note] Why θ is sampled inside a narrower box than it is constrained to
> The heads map to `Θ_j` by `c + r·tanh(z)`. Drawing the truth right up to the
> box edge places it where `tanh` is already saturated and the head's gradient
> has died. The `0.25` vs `0.4` margin is deliberate; see §6.

### Why this revision of the system exists

The state–parameter split in Eq. (1) is degenerate in the way Remark 7
constructs: a parameter error `e` is absorbed by a shift of the next state,
`c_k(t) = −e ∂_{θ_{k-1}} f_{k-1}(t)`. A **constant** shift lies in the null space
of the final residual's companion row, so the degeneracy is severe exactly when
the sensitivity is near-constant along the realised trajectory.

The earlier design used `∂_{θ₁}f₁ = 0.8 + 0.2 cos x₁`, whose stated floor of 0.6
holds over the whole box — but on trajectories the system actually produces it
spans only `[0.93, 0.98]`, a 5% variation. The compensation was nearly constant
and nearly free.

This revision makes the sensitivities vary strongly while *raising* their floors:

| quantity | previous | this revision |
|---|---|---|
| `s₁` on-trajectory | `[0.93, 0.98]` (5%) | `[1.00, 2.00]` (101%) |
| `s₂` on-trajectory | near-constant | `[1.44, 1.80]` (25%) |
| `γ₂` | ≥ 0.36 | ≥ 0.99 |
| `γ₃` | ≥ 0.49 | ≥ 2.07 |
| identifiability | 1× | **25.7×** |

`s₁` is keyed to `x₁` and `s₂` to `x₂` — *different* coordinates, deliberately,
so the two compensations cannot re-align. Keying both to `x₁` recovered most of
the degeneracy: 3.2× instead of 25.7×.

**The generalisable lesson, which Assumption 6 does not imply:** `γ_k > 0` is
necessary but not sufficient. A sensitivity bounded away from zero yet nearly
constant satisfies Assumption 6 while leaving the split practically
unidentifiable. What matters is the *variation* of `∂_{θ_j}f_j` along realised
trajectories, and that different parameters key off different coordinates.

---

## 3. Disturbance basis

Fourier, `q = 3`: `Γ_q(t) = [sin Ωt, cos Ωt, sin 2Ωt]`. **No constant mode** —
checked, because a constant term would be exactly degenerate with a sensor
offset.

| quantity | value |
|---|---|
| `q` | 3 |
| `Ω` | `2π/5 = 1.2566` (`w5` arm) or `2π/10 = 0.6283` (`w10` arm) |
| coefficient box `𝒜` | `±0.18`, `±0.18`, `±0.12` |
| `λ_min(W_Γ)` (Eq. 3) | `10.0` = `T/2`, both frequencies |
| remainder `ε_{d,q}` | `0` |

`λ_min(W_Γ) = T/2` exactly because both horizons span an integer number of
periods (4 and 2), which makes `W_Γ` diagonal.

### Observability: not all coefficients are visible

Eq. (3) asks only that `Γ_q` be linearly independent *as functions of time* — a
property of the basis alone, saying nothing about the plant. But the disturbance
enters the last equation while only the first coordinate is measured, so its
trace in `y` passes through `n` integrations and is attenuated.
`pibe/eval/observability.py` quantifies this per coefficient, **before training**:

| Ω | SNR `a₁` | SNR `a₂` | SNR `a₃` |
|---|---|---|---|
| `2π/5` | 12.56 | 12.61 | **2.05** |
| `2π/10` | 27.60 | 27.33 | **8.37** |

Halving `Ω` raises every SNR ~2.2× at no cost to `λ_min(W_Γ)`. This prediction
transferred quantitatively to the fitted estimator: on the reference run the
regression slope of `â_j` on `a_j` is 0.967 / 0.967 / **0.881** — shrinkage
tracks SNR. And empirically `a₃` was recovered in 2 of 6 low-Ω runs and **0 of 6**
high-Ω runs, so low Ω is necessary and not sufficient.

---

## 4. Measurement noise

Section 2 restricts to compactly supported noise, `‖ω‖_{L∞} ≤ w̄`. The default
is Section 3.1's Gaussian case: zero-mean Gaussian, symmetrically truncated,
sampled by inverse CDF (exact, vectorised, reproducible from the generator
state).

| quantity | value |
|---|---|
| `σ` | `0.002` (reference run) |
| truncation | `3σ`, so `w̄ = 0.006` |
| realised sd | `0.0019719` (truncation reduces it slightly) |
| realised mean | `+2.6e-06` |
| applied to | `y = x₁ + ω` only, i.i.d. per sample and trajectory |

Symmetric truncation preserves the zero mean, so the noise adds no bias to the
quadratic data term — the situation Proposition 1 describes.

Non-Gaussian families are implemented for the robustness studies
(`pibe/data/noise.py`), all calibrated to the requested sd so shape is the only
variable:

| family | kurtosis | skew | bound at σ=0.1 |
|---|---|---|---|
| `gaussian` | 2.84 | 0.00 | 0.300 |
| `uniform` | 1.80 | 0.00 | 0.173 |
| `laplace` | 4.73 | −0.01 | 0.416 |
| `contaminated` (5% at 8×) | 33.9 | −0.25 | 1.194 |
| `skewed` | 4.22 | **+1.28** | 0.369 |
| `+ BiasedNoise` | — | — | + offset |

**Result worth carrying into the paper:** the *shape* of a zero-mean law costs
the estimator nothing — a 19× spread in kurtosis and a full unit of skew moved
the state error by under 3%. Only a nonzero *mean* hurts, and it hurts linearly
(log–log slope 1.004 over three decades). See [[EPIBE]] §3–§5.

---

## 5. Data generation

| quantity | value (reference run) |
|---|---|
| trajectories `P` | 2048 |
| samples `N` | 201, uniform on `[0, 20]` (`h_d = 0.05`) |
| horizon `T` | 20 s |
| integrator | RK4, 8 substeps per data interval |
| collocation `N_r` | 401 (`h_r = 0.025`) |
| train / validation | 1638 / 410, disjoint **whole trajectories** |
| `dtype` | `float64` throughout |

Sampled independently per trajectory: the initial condition (uniform on `χ₀`)
and the disturbance coefficients (uniform on `𝒜`,
`sample_disturbance_per_trajectory: true`). θ is **shared** in the `v2` configs
and sampled per trajectory in `v3`.

> [!warning] Why the disturbance must vary per trajectory
> With one `a` shared by every trajectory, the coefficient head can satisfy the
> objective by emitting a constant, and "infer `a` from `y`" is never posed. An
> early result looked far better than it was for exactly this reason. The
> diagnostic is `corr(â_j, a_j)` across trajectories together with `sd(â_j)`;
> `compare_runs.py` reports both.

Every validation number in this project is on trajectories whose initial
conditions and disturbance coefficients the network never saw.

---

## 6. Estimator architecture

A bank of `n = 3` cells `Φ², Φ³, Φ⁴`, each an encoder + decoder + head.
2,216,458 parameters, `float64`.

```
cell k=2                                                      [635k params]
  encoder  U_1  201  ->  512 -> 512 -> 128        (the measurement y)
  decoder  (t,z) 149 ->  256 -> 256 -> 256 -> 2   -> (x_1, x_2)   boxes ±2.5, ±0.8
  head     z    128  ->  128 -> 128 -> 1          -> theta_1       box  ±0.4

cell k=3                                                      [739k params]
  encoder  U_2  403  ->  512 -> 512 -> 128        (y, x_2 estimate)
  decoder  (t,z) 149 ->  256 -> 256 -> 256 -> 2   -> (x_2, x_3)   boxes ±0.8, ±0.8
  head     z    128  ->  128 -> 128 -> 1          -> theta_2       box  ±0.4

cell k=4  (disturbance cell)                                  [842k params]
  encoder  U_3  605  ->  512 -> 512 -> 128        (y, x_2, x_3, theta_1, theta_2)
  decoder  (t,z) 149 ->  256 -> 256 -> 256 -> 1   -> x_3 auxiliary  box ±0.8
  head     z    128  ->  128 -> 128 -> 3          -> a              boxes ±0.18, ±0.18, ±0.12
```

**Encoder input widths** follow Eq. (15): `d_{k-1} = N(k−1) + (k−2)` = 201, 403,
605. Inputs are non-dimensionalised by the fixed `D_{k-1}` of Eq. (61) — every
sampled block divided by `√N · s_{x_j}`, every parameter block by `s_{θ_j}`.

**The decoder takes time as a Fourier lift**, not a raw scalar: `t → [−1,1]`,
then `[t, sin(kπt), cos(kπt)]` for `k = 1…10`, giving 21 features, concatenated
with the 128-dim latent → 149 inputs. Amplitudes are scaled by `1/k`
(`time_feature_scaling: true`) because harmonic `k` contributes `k×` to the
autograd time derivative in the physics residual. Without it, `K = 40` made `x₁`
**11× worse**; the scaling brought the measured amplification from 16.0× to 1.00×.

**Activations are `tanh` everywhere, and this is required.** The physics residual
differentiates the decoder w.r.t. time by autograd (Eqs. 21/36), so the network
must be `C²`. ReLU would give a discontinuous `ẋ̂`.

**Every output lands in its admissible set by construction**, `c + r·tanh(z)` —
no constraint penalties. With `saturation_limit: 4.0` a straight-through clamp
floors the gradient multiplier at `sech²(4) = 1.3e-03`:

> [!warning] The saturation trap
> Nothing bounds `z`. A head pushed toward its box edge keeps going, and by
> `|z| ≈ 18` the multiplier `4e^{−2|z|}` is `4e-16` — below float64 resolution,
> so the unit is dead permanently. This destroyed a 24-hour cluster run: state
> decoder at `|z| = 18.4`, both θ heads pinned, gradient norms `1e-7`. The clamp
> fixes it (21× better states in a controlled A/B); the trainer now also warns
> whenever a gradient norm falls below `1e-5`. Default is `None` for
> backwards-compatibility — the `_v2`/`_v3` configs set it explicitly.

---

## 7. Losses

Per cell, `L^k_Loc = L^k_Data + λ L^k_Physics`, Eqs. (22)/(28)/(37).

**Data terms** — one rule: cell `k` compares its reconstruction of coordinate
`k−1` against the target from upstream.

$$
\begin{aligned}
L^2_{Data} &= \tfrac1N \textstyle\sum_i (y - \hat x^2_1)^2 && \text{(23)} \\
L^k_{Data} &= \tfrac1N \textstyle\sum_i (\hat x^{k-1}_{k-1} - \hat x^k_{k-1})^2 && \text{(29)} \\
L^{n+1}_{Data} &= \tfrac1N \textstyle\sum_i |\hat x^n_n - \hat x^{n+1}_n|^2 && \text{(38)}
\end{aligned}
$$

Only the first is a genuine measurement fit; the rest are deterministic
inter-cell consistency penalties whose correct value is identically zero.

**Physics terms** — `L^k_{Physics} = (1/N_r) Σ_j |r_k(τ_j)|²`, Eqs. (24)/(30)/(39),
with the time derivative taken by autograd on per-trajectory time tensors
(`create_graph=True`).

**Global loss**, used only during joint fine-tuning:
`L^k_Tot = Σ_{m=2}^{k} e^{−(k−m)/k} L^m_Loc`, Eqs. (27)/(42). Weights for `k=4`
are 0.61, 0.78, 1.0 — the current cell dominates while upstream cells still adapt.

---

## 8. Training schedule

Algorithm 1, per cell, split at `n_par`:

| phase | iterations | what is updated | lr |
|---|---|---|---|
| local | `i < 8000` | `Θ^k_PINN` only; upstream frozen and detached | `1e-3` |
| global | `8000 ≤ i < 20000` | `Θ^m_PINN`, `m = 2…k`, no detachment | `2e-4` |
| final cell extra | `+30000` | as above, closing the identification | `2e-4` |

| quantity | value |
|---|---|
| `λ` | 1.0 |
| batch size | 32 trajectories |
| gradient clip | 1.0 |
| LR schedule | cosine |
| checkpoint every | 2000 iterations |
| seeds | 0, 1, 2 (`seed` and `training.seed` together) |

Total 50 000 iterations for the final cell, 20 000 each for cells 2 and 3 —
90 000 in all, about 19 h at the 0.77 s/iteration measured on Margaret.

> [!tip] Resumability
> `state.pt` (weights + Adam moments + LR-schedule position + history + both RNG
> streams) is written every `checkpoint_every`, temp-file-then-renamed. `bank.pt`
> and `history.json` are written at the same points, so an interrupted run is
> usable by every evaluation script rather than merely recoverable.

---

## 9. Reference results

`outputs/big_w10_v2_s1`, 410 held-out trajectories:

| quantity | value |
|---|---|
| `x₁` | 1.63e-03 |
| `x₂` | 1.56e-02 |
| `x₃` | 2.45e-02 |
| `\|Δθ₁\|`, `\|Δθ₂\|` | 8.76e-03, 7.72e-03 |
| `d` | 7.30e-03 |
| `corr(â, a)` | 0.999, 1.000, 0.997 |
| shrinkage slope | 0.967, 0.967, 0.881 |

### Scale-normalized percentage errors (Section 5)

$$
E_{x_j} = \frac{100}{s_{x_j}}\Bigl(\tfrac{1}{P_{te}}\textstyle\sum_\ell
          \|\hat{\mathbf X}^\ell_j - \mathbf X^\ell_j\|^2_{N,2}\Bigr)^{1/2},
\quad
E_{\theta_j} = \frac{100}{s_{\theta_j}}\Bigl(\tfrac{1}{P_{te}}\textstyle\sum_\ell
          |\hat\theta^\ell_j - \theta^\ell_j|^2\Bigr)^{1/2},
\quad
E_d = \frac{100}{G_q s_a}\Bigl(\tfrac{1}{P_{te}}\textstyle\sum_\ell
          \|\hat{\mathbf d}^\ell - \mathbf d^\ell\|^2_{N,2}\Bigr)^{1/2}
$$

| `E_x₁` | `E_x₂` | `E_x₃` | `E_θ₁` | `E_θ₂` | `E_d` |
|---|---|---|---|---|---|
| **0.0715 %** | **2.1662 %** | **3.4126 %** | **2.6399 %** | **2.3177 %** | **2.0648 %** |

`P_te = 410`, `N = 201`. Constants used:

| quantity | value | source |
|---|---|---|
| `s_x` | `[2.5, 0.8, 0.8]` | half-widths of `𝒳_j` |
| `s_θ` | `[0.4, 0.4]` | half-widths of `Θ_j` |
| `s_a` | `0.281425` | `‖(0.18, 0.18, 0.12)‖₂` — see below |
| `G_q` | `1.414214` | `sup_t ‖Γ_q(t)‖₂ = √2`, Eq. (73) |
| `G_q · s_a` | `0.397995` | the scale in Eq. (118) |

Three points on how these are computed, each differing from what
`pibe/eval/metrics.py` reports:

- **Root mean square across trajectories, not a mean of norms.** The norm is
  squared *inside* the trajectory average, so with Eq. (60)'s `‖·‖_{N,2}` each
  figure is the plain RMS error over every sample of every test trajectory. Bad
  trajectories weigh more than they would in a mean.
- **Normalized by reference scale, not by signal magnitude.** These are
  percentages of the *admissible set*, which is what Corollary 2's componentwise
  bounds (114)–(118) are stated against. A coordinate that happens to be small
  on a trajectory does not inflate its own figure.
- **`E_d` is measured against `G_q s_a`**, from Eq. (118)
  `‖d̂ − d‖_∞ ≤ ε_{d,q} + G_q s_a B_{n+1}` — the largest disturbance the
  admissible coefficient box can produce.

> [!note] The choice of `s_a`
> Section 4.1 asks for `s_y, s_{x_j}, s_{θ_j}, s_a` to be *fixed* without
> prescribing them, and `s_a` is not in the code. Since this implementation
> takes `s_{x_j}` and `s_{θ_j}` as admissible half-widths, `s_a` is taken as the
> 2-norm of the coefficient box's half-widths — the radius of the smallest ball
> containing `𝒜`. It scales `E_d` directly, so it is reported alongside.
> `scripts/report_errors.py --run <dir>` recomputes all of it.

Reading them: `x₁` is recovered to **0.07% of its admissible range**, the two
unmeasured coordinates to 2.2% and 3.4%, both parameters to ~2.5%, and the
disturbance to 2.1% of the largest amplitude the basis box permits. The ordering
`E_x₁ ≪ E_x₂ < E_x₃` is the expected error growth down the chain
(Proposition 4 / Remark 12).

Figures in `outputs/big_w10_v2_s1/figures/`: `states.png`, `disturbance.png`,
`parameters.png`, `errors.png`, plus `trajectories.png` (10 random held-out
trajectories) and `coefficient_errors.png` (error vs true value, KDE).

θ here is **shared** across trajectories, so those θ numbers are not evidence of
parameter inference — a head can satisfy the objective with a constant. The `v3`
runs sample θ per trajectory and give `corr(θ̂, θ) ≈ 0.80 / 0.68`, at the cost of
7× worse `x₂`/`x₃`.

---

## 10. Reproducing

```bash
# reference run
python scripts/train_pibe.py --config configs/automatica_n3v2_big_lowomega_v2.yaml \
    --set seed=1 training.seed=1 --output-dir outputs/big_w10_v2_s1

# analysis
python scripts/compare_runs.py     --runs outputs/big_*_v2_s*
python scripts/plot_trajectories.py --run outputs/big_w10_v2_s1 --count 10
python scripts/coefficient_errors.py --run outputs/big_w10_v2_s1
python scripts/report_errors.py      --run outputs/big_w10_v2_s1   # Section 5 percentages
python scripts/inspect_state.py     --run outputs/big_w10_v2_s1 --metrics

# cluster
scripts/slurm/sync.sh user@host:Automatica
scripts/slurm/submit.sh 0-5           # PIBE_VERSION=v2 for the shared-theta grid
```

Diagnostics worth knowing about: `eval/observability.py` (per-coefficient SNR
before training), `eval/oracle.py` (scores the truth on the objective, so a poor
result is attributed to optimisation or to the loss rather than left ambiguous),
`scripts/location_coercivity.py` and `scripts/profile_theta.py` (identifiability
profiles).
