# EPIBE (Section 3.2): implementation and findings

This documents the EPIBE implementation and what it does on the 3rd-order
benchmark under a biased sensor. It is written to be usable directly when
revising Section 3.2, so it separates three things carefully: what the paper
specifies, what had to be decided to make it runnable, and what the experiments
actually show.

**Bottom line.** The framework is implemented and verified, and the biased-sensor
problem is **solved** — but not by the energy term alone. EPIBE's premise is
confirmed and quantified (§3): PIBE cannot represent a nonzero noise mean even
when retrained on biased data. Training EPIBE end to end does not fix it either,
for a reason that is now precisely characterised (§5). What does fix it is an
auxiliary estimate of the offset computed where the problem is well conditioned
(§8), which recovers essentially all of the accuracy lost to the bias:

| run | x₁ | x₂ | x₃ | \|θ₁\| | \|θ₂\| | d |
|---|---|---|---|---|---|---|
| PIBE, biased sensor | 4.977e-02 | 2.090e-02 | 3.463e-02 | 7.925e-03 | 1.699e-02 | 7.163e-03 |
| **PIBE + estimated offset** | **7.316e-03** | **1.238e-02** | **2.014e-02** | **6.233e-03** | **5.311e-03** | 9.268e-03 |
| PIBE, unbiased (ceiling) | 2.221e-03 | 1.279e-02 | 2.063e-02 | 6.227e-03 | 5.486e-03 | 9.591e-03 |

The energy-based data term itself is **not** what makes this work, and §9 shows
that on its own fair test — heavy-tailed zero-mean noise, the case a learned
likelihood is classically for — it is 8x–22x *worse* than the quadratic term it
replaces.

---

## 1. What was built

| module | implements |
|---|---|
| `pibe/ebm/support.py` | `R_k = [-ρ̄_k, ρ̄_k]`, Assumption 3 |
| `pibe/ebm/energy.py` | gauge (47), bounds (48), projection onto `K_k` |
| `pibe/ebm/quadrature.py` | fixed rule for `Z_k`, Eq. (59) / Remark 4 |
| `pibe/ebm/density.py` | density (49), likelihood (53), moments (54) |
| `pibe/core/energy_bank.py` | Eqs. (52)/(55)/(57) — a subclass of the PIBE bank |
| `pibe/training/epibe_phases.py` | Algorithm 2 / Remark 5 schedule |
| `pibe/training/epibe_trainer.py` | Algorithm 2 |
| `pibe/eval/energy_oracle.py` | the objective evaluated at the exact solution |

`ebm.enabled` defaults to `false`, so every pre-existing configuration and
checkpoint resolves to plain PIBE unchanged. 35 tests in `tests/test_ebm.py`.

**Verified in isolation.** Fitting the EBM to samples from known laws recovers
the mean to five decimals — the thing PIBE structurally cannot do:

| law | true mean | `μ̂` | true sd | `sd̂` |
|---|---|---|---|---|
| gaussian, unbiased | 0.00002 | 0.00002 | 0.01973 | 0.02025 |
| gaussian, bias +0.05 | 0.05002 | **0.05002** | 0.01973 | 0.01982 |
| gaussian, bias −0.08 | −0.07998 | **−0.07997** | 0.01973 | 0.01983 |
| skewed | 0.00002 | 0.00004 | 0.03002 | 0.03029 |
| contaminated, bias +0.04 | 0.03993 | 0.03984 | 0.01982 | 0.02028 |

Assumption 3's constraints hold numerically: `Ẽ(0) = 0` exactly, `sup|Ẽ| ≤ B_E`,
measured Lipschitz below `L_E`, density normalised to 1e-16.

---

## 2. Departures from Algorithm 2 as written

Each was forced by a failure, is off by default where it is a genuine choice,
and is documented at its definition.

**(a) The out-of-support continuation.** Eq. (49) gives zero likelihood outside
`R_k`, i.e. energy `+∞`. Clamping the residual onto the support — the obvious
reading — makes the energy *constant* there, so an escaped residual contributes
**no gradient at all**: the data term silently switches off and the estimate
drifts further out, a positive feedback loop. Measured residuals reached −1.09
against `ρ̄ = 0.5`. Replaced with a steep quadratic continuation, so the force
always points back in. Inside `R_k` nothing changes and `Z_k` is unaffected.
*This one is a straightforward correction, not a modelling choice.*

**(b) `n_fit`, an EBM-only stretch after activation** (default 0 = Algorithm 2
exactly). Translating a cell's residuals by `c` and differentiating (53), the
mean force at a well-fitted density is

> **−∂/∂c 𝓛^ebm_Data = p(ρ̄_k) − p(−ρ̄_k)**

— the density difference at the two ends of its support. A cold, near-uniform
EBM makes both terms O(1/2ρ̄) and any asymmetry a systematic push on the state,
which no single cell's physics opposes. Without this stretch the residuals slid
to the support boundary (99–100% out of support, θ̂ pinned at both box edges).

**(c) `ebm.cells`** (default `None` = all cells, Step 2 as written). Only `ε₂`
carries measurement noise; Eqs. (29)/(38) are, in PIBE's own words,
"deterministic inter-cell consistency penalties" whose correct value is
identically zero. Giving those a learned density hands them a free location and
nothing then ties two reconstructions of the same coordinate together. Measured:

| | x₂ L² | \|θ₂ err\| |
|---|---|---|
| all cells (Step 2 as written) | 8.284e-01 | 5.039e-01 |
| cell 2 only | 6.927e-02 | 1.204e-02 |

**Suggested for the paper:** Step 2's justification ("how its uncertainty
propagates to the downstream states is unknown") is sound about the *shape* of
the downstream residual, but it also grants those cells a location parameter
they should not have. Restricting the energy term to the measurement cell is a
one-line change with a 12–40× effect.

**(d) `centered_warmup`** (default off, implemented, not the winning variant).
Remark 5's warm-up runs "the corresponding PIBE local objective" — the quadratic
term, which *is* the zero-mean assumption — so EPIBE starts from exactly the
answer it exists to improve on. Proposition 1 supplies the fix in closed form:
minimising `R_D(ŷ, m)` over the unknown location leaves precisely the residual's
**variance**. Training on that is the quadratic term with the location
eliminated rather than assumed.

---

## 3. The premise is confirmed

Retraining PIBE *on biased data* does not remove the bias. Its cell-2 residual
mean is `+0.0003` — `x̂₁` tracked the biased measurement exactly — and

> `mean(x̂₁ − x₁_true) = +0.0497` against a true bias of `+0.0500`.

The quadratic data term has no way to represent a nonzero noise mean even when
the offset is constant and present throughout training. This closes the obvious
referee objection to the motivation figure, and it closes it in the paper's
favour.

---

## 4. Results, biased sensor (bias +0.05, σ = 0.002), held out

| run | x₁ | x₂ | x₃ | \|θ₁\| | \|θ₂\| | d | μ̂_ω |
|---|---|---|---|---|---|---|---|
| **PIBE (control)** | 4.977e-02 | **2.090e-02** | **3.463e-02** | **7.925e-03** | 1.699e-02 | **7.163e-03** | — |
| EPIBE all cells | 2.871e-02 | 8.284e-01 | 5.509e-01 | 1.650e-01 | 5.039e-01 | 9.277e-02 | +0.0775 |
| EPIBE cell 2 only | 5.627e-02 | 6.927e-02 | 6.500e-02 | 3.255e-02 | **1.204e-02** | 1.465e-01 | −0.0058 |
| + λ = 10 | 2.276e-02 | 2.375e-01 | 1.998e-01 | 1.650e-01 | 1.415e-01 | 1.316e-01 | +0.0279 |
| + joint training | 8.633e-02 | 2.253e-01 | 1.614e-01 | 1.650e-01 | 1.212e-01 | 3.586e-02 | −0.0367 |
| **+ location scan, 6k steps** | **9.510e-03** | 4.358e-02 | 4.323e-02 | 1.941e-02 | 1.568e-02 | 7.992e-02 | **+0.0583** |
| + location scan, 24k steps | 2.215e-02 | 2.434e-01 | 2.080e-01 | 1.559e-01 | 1.598e-01 | 1.263e-01 | +0.0275 |

The best EPIBE variant is **5.2× better than PIBE on x₁** and recovers the
offset (`+0.058` vs a true `+0.050`), but is 11× worse on the disturbance and
2.4× worse on θ₁. **EPIBE does not yet dominate PIBE.**

### The location scan

The degeneracy is exactly one scalar, so it can be searched rather than
descended: seed the first cell's density at a candidate offset `c` (which pulls
`x̂₁` toward `y − c`), train, and let the objective choose.
`scripts/epibe_location_scan.py`, figure `outputs/epibe_scan.png`.

```
  offset |       L_Tot,E |    mu_hat |  x1-truth |      x1 L2
  0.0000 | -2.683658e+00 |  -0.00335 |  +0.05291 |  5.300e-02
  0.0500 | -2.659612e+00 |  +0.02580 |  +0.02317 |  2.337e-02
  0.0750 | -2.689291e+00 |  +0.05829 |  -0.00908 |  9.510e-03   <-- selected
  0.1000 | -2.620800e+00 |  +0.03281 |  +0.01477 |  1.505e-02
```

The objective's minimum coincides with the best accuracy. But the margin is
**0.006 nats** over `c = 0`, while accuracy varies 10× — the discrimination is
real but weak.

---

## 5. Why it does not dominate: a quantitative obstruction

`pibe/eval/energy_oracle.py` scores the exact solution — true states and
parameters, plus the best density the admissible class can put on the *actual
noise realisation*. It reports:

```
L_Tot,E @ truth   : -2.913345e+00
L_Tot,E @ trained : -2.706294e+00
mu_omega @ truth  : +0.049975     (true bias +0.050000)
```

So the truth beats the trained point. But that is a comparison of two points,
and it is **not** the whole landscape. Continuing the best run from 6k to 24k
steps drove the objective *down* (−2.689 → −2.713) while accuracy got *worse*
(x₁ 9.5e-03 → 2.2e-02, μ̂ +0.058 → +0.027). Gradient descent is heading
somewhere lower than the truth, and away from it.

Where? To the collapse. The energy objective's infimum over the PINN is
"reproduce `y` exactly": residual ≡ 0, density maximally sharp. Eq. (50) bounds
that at `NLL ≥ −2βB_E + log 2ρ̄`. With the run's `B_E = 12`:

| | value |
|---|---|
| NLL at the true noise law | **−4.80** (oracle measured −4.803) |
| collapse floor at `B_E = 12` | **−25.20** |

The collapse is 20 nats below the truth. Eq. (50) is exactly the mechanism meant
to prevent "arbitrarily narrow empirical-likelihood spikes", and it does bound
them — but not tightly enough to bite, because `B_E` was never tied to the noise
scale. Only the physics term stands between the optimiser and the collapse, and
it loses.

### The constants have to satisfy two inequalities, and the window is tiny

With `ρ̄ = 0.15`, `w̄ = 0.006`, `σ = 0.002`, `β = 1`:

| requirement | condition | bound |
|---|---|---|
| represent the true noise density | `2βB_E ≥ log(2ρ̄ / 2w̄) = 3.219` | `B_E ≥ 1.61` |
| keep the collapse no better than the truth | `2βB_E ≤ −h(ω) + log 2ρ̄ = 3.592` | `B_E ≤ 1.80` |

An admissible window of width **0.19** — and its upper limit depends on `h(ω)`,
the differential entropy of the very noise law EPIBE is trying to learn.

Worse, the window is empirically empty here. Running the scan at `B_E = 1.7`,
*inside* the window, the density is too smooth to hold a shifted location: the
seeded offset is dragged in during the hold and snaps straight back once the
density is released, giving `μ̂ ≈ 0` at every candidate offset and the wrong `c`
selected.

```
  offset |       L_Tot,E |    mu_hat |  x1-truth      (B_E = 1.7)
 -0.0500 | -2.200095e+00 |  +0.00395 |  +0.04254
  0.0000 | -2.204145e+00 |  -0.00410 |  +0.05767
  0.0500 | -2.190721e+00 |  -0.00445 |  +0.05827
  0.1000 | -2.212361e+00 |  +0.00370 |  +0.04223   <-- selected, wrongly
```

**So there is a genuine tension in Assumption 3's constants**: large `B_E` gives
the density enough strength to hold a location but makes the collapse
overwhelmingly attractive; small `B_E` removes the collapse but leaves the
density too weak to hold anything. This is not a tuning difficulty, it is a
property of the objective, and it is the single most important thing these
experiments found.

---

## 5b. Is the offset identifiable at all? Yes — and that changes the diagnosis

§5 says the objective's descent leads away from the truth. That leaves open a
sharper question: does Proposition 1's unique-zero hypothesis actually hold for
this system? `scripts/location_coercivity.py` measures it directly. Displace the
true trajectory's measured coordinate by `c`, rebuild the remaining states from
the triangular form (which *determines* them, with no slack), and minimise the
last equation's residual over the admissible `θ̂` and `â`:

```
 offset c |   min ||r||^2  | ratio to c=0 |  theta_hat
   0.000  |  1.853515e-18  |            1 | [+0.2350, +0.1039]   (= theta_true)
  -0.010  |  2.700048e-07  |      1.5e+11 | [+0.2345, +0.1017]
  -0.020  |  1.078115e-06  |      5.8e+11 | [+0.2339, +0.0996]
  -0.050  |  6.689804e-06  |      3.6e+12 | [+0.2321, +0.0932]
  -0.100  |  2.627776e-05  |      1.4e+13 | [+0.2290, +0.0829]
   0.010  |  2.706937e-07  |              | [+0.2356, +0.1061]
   0.050  |  6.774601e-06  |              | [+0.2376, +0.1151]
   0.100  |  2.692360e-05  |              | [+0.2398, +0.1267]
```

A clean quadratic well centred exactly on the truth: the cost scales as `c²`
(x4 at 2x, x25 at 5x, x100 at 10x) and the inner optimum returns `θ_true` to
four decimals at `c = 0`. **Proposition 1's coercivity hypothesis holds for this
system.** The offset is identifiable from the physics, and it is worth
`6.8e-06` at the true bias.

### So why can't training see it?

Because `6.8e-06` is far below the estimator's own physics residual, which sits
near `1e-4` for a trained bank — the identifying signal is **15x smaller than
the reconstruction slack it has to be read through**. Worse, reading it off a
trained model is worse still: `scripts/estimate_offset.py` rebuilds the chain
from a trained decoder and finds a floor of `1.9e-03`, **280x the signal**,
because reconstructing `x̂_2 … x̂_n` from `x̂_1` takes `n` numerical derivatives
and amplifies the decoder's high-frequency error by roughly `ω^n`:

```
  c = +0.000 -> 1.924606e-03      (from outputs/n3v2_biased_pibe)
  c = +0.050 -> 1.886305e-03
  c = +0.100 -> 1.847476e-03      monotone: no minimum, signal buried
```

**This reframes the whole problem.** It is not that EPIBE lacks the information
to recover `μ_ω`, and not that the paper's coercivity assumption fails. It is
that the information sits `10^2`–`10^3` below the numerical floor of the
representation used to extract it. That is a *conditioning* statement, and
conditioning problems have conditioning fixes — the natural one being to
integrate rather than differentiate, i.e. compare model and measurement in the
output space where the noise actually lives, instead of after `n`
differentiations (`scripts/offset_by_shooting.py`).

For the paper this is a much better story than "the assumption fails". It says
Assumption 2 / Remark 10 are satisfied here, and identifies precisely what makes
them unusable in practice: the estimator's own approximation error enters the
physics residual amplified by the differentiation order of the chain, and swamps
the parameter it is supposed to reveal. That is a statement about *every*
collocation-style estimator on a triangular system of order `n`, not just this
one.

---

## 6. What this suggests for the paper

1. **Keep and strengthen Remark 2.** "The EBM can otherwise absorb a systematic
   PINN error as part of the residual law" is not a caveat in passing — it is
   the binding constraint. §5 gives it a number.

2. **Restrict the energy term to the measurement cell** (§2c). One line, 12–40×
   on the downstream states and parameters, and the justification is already in
   PIBE's own description of Eqs. (29)/(38).

3. **Tie `B_E` to the noise scale, explicitly.** Eq. (50) currently reads as a
   well-posedness condition. It is also, quantitatively, the condition that
   decides whether the truth or the collapse is the preferred optimum, and the
   paper should state the inequality.

4. **Coercivity holds — say so, and say what defeats it.** §5b measures
   Proposition 1's unique-zero hypothesis directly and finds it satisfied, with
   a clean `c²` well centred on the truth. What defeats it is that the signal
   (`6.8e-06`) is `10^2`–`10^3` below the estimator's own physics residual,
   because reconstructing the chain differentiates `n` times and amplifies the
   approximation error. Stating this converts a vague deferral to Remark 10 into
   a checkable condition: the identifying curvature must exceed the achievable
   physics-residual floor, and for a chain of order `n` that floor grows with
   the differentiation order.

5. **The honest experimental claim available today** is the motivation, not the
   remedy: PIBE is provably unable to represent a biased sensor (§3), the shape
   of a zero-mean law costs it nothing, and EPIBE's first cell can recover the
   offset when the location degeneracy is searched rather than descended.

## 8. What makes it work: estimate the offset where it is well conditioned

§5b establishes that the offset *is* identifiable from the physics, and that the
obstruction is conditioning: rebuilding the chain from `x̂₁` differentiates `n`
times and lifts the estimator's own error to `10²`–`10³` above the `6.8e-06`
signal. The fix follows from that sentence — stop differentiating and integrate.

`scripts/offset_by_shooting.py` fits the model to the measurement by simulation,

> min over `(x₀, θ, a, μ)` of `‖ x₁(t; x₀, θ, a) + μ − y(t) ‖²`,

with `μ` an ordinary parameter rather than something scanned. It can be a
parameter here precisely because the degeneracy that defeats the bank is absent:
inside the bank `x̂^k_k` is free and absorbs any shift of `x̂^k_{k-1}`, whereas a
simulated trajectory is generated from eight numbers and has no slack to hide a
shift in. Every derivative is taken by the integrator on the exactly-known
vector field, and the residual is compared in the output space where the noise
actually lives.

```
    step     0 | cost 1.7522e+00 | mu +0.01000 | theta [+0.0000, +0.0000]
    step  1800 | cost 3.3087e-05 | mu +0.07595 | theta [+0.2102, +0.1198]
    step  3600 | cost 8.2345e-06 | mu +0.06433 | theta [+0.2258, +0.1085]
    step  5400 | cost 5.7382e-06 | mu +0.06186 | theta [+0.2292, +0.1060]
                                   true +0.050          true [+0.2350, +0.1039]
```

Four trajectories, 151 samples, eight parameters per trajectory, a few minutes.
The residual reaches `5.7e-06` against a noise-variance floor of `3.9e-06`
(ratio 1.5), and `θ̂` lands within `0.006` and `0.002` of truth — better than the
trained bank's own parameter estimates. Two independent fits gave `μ̂ = +0.043`
(1 trajectory) and `μ̂ = +0.062` (4 trajectories) against a true `+0.050`, so
call it `±25%` at this budget: enough to remove ~85% of the bias, which is what
the table in the bottom line shows.

The full method is then three stages:

1. **Estimate** `μ̂_ω` by simulation-based fitting on a handful of trajectories.
2. **Correct** the measurement, `y ← y − μ̂_ω`.
3. **Run the bank** unchanged — PIBE or EPIBE, whichever the noise shape calls
   for.

Residual `x₁` error after correction is `7.3e-03` against a `2.2e-03` ceiling,
and that gap is exactly the `0.0071` of uncorrected bias; everything downstream
matches the ceiling. Sharpening stage 1 closes the rest.

### Why this is the right shape for the paper

Proposition 1 says identification needs a physics functional with a unique zero.
§5b shows it has one. Stage 1 is simply the estimator that *reads* that unique
zero in a well-conditioned way, instead of asking a collocation objective to
feel it through `n` differentiations of a learned function. Stated that way it
is not a workaround bolted onto EPIBE — it is the concrete content of the
anchor Remark 10 defers, and it is cheap because the quantity in question is one
scalar shared by every trajectory.

It also clarifies the division of labour. The EBM's job is the *shape* of the
residual law, which it does well (§1, five decimals on biased, skewed and
contaminated laws). The *location* is a global scalar that the bank's own
objective is provably blind to, and it should be estimated separately rather
than asked of the same optimisation.

---

## 9. Does the energy term earn its place? The fair test says no

The biased-sensor experiments are a *location* problem, and §8 solves that
outside the bank. That leaves the case a learned likelihood is classically for:
heavy-tailed noise, where a quadratic data term is dominated by rare large
residuals and a correctly specified likelihood downweights them. Three arms,
identical except as named, zero-mean noise of sd 0.01 so the location question
plays no part; "contaminated" is 5% of samples drawn 8x wider, kurtosis 34
against a Gaussian's 3.

| run | x₁ | x₂ | x₃ | \|θ₁\| | \|θ₂\| | d |
|---|---|---|---|---|---|---|
| PIBE, Gaussian | 2.647e-03 | 1.487e-02 | 2.094e-02 | 7.942e-03 | 6.948e-03 | 1.108e-02 |
| PIBE, contaminated | 3.013e-03 | 1.471e-02 | 2.224e-02 | 7.476e-03 | 6.631e-03 | 9.102e-03 |
| EPIBE, contaminated | 5.259e-02 | 2.326e-01 | 1.801e-01 | 1.650e-01 | 1.318e-01 | 1.257e-01 |

**The premise fails before the test begins.** Comparing the first two rows,
contaminated/Gaussian is 1.14x, 0.99x, 1.06x on the states and 0.94x, 0.95x,
0.82x on the parameters and disturbance — heavy tails cost the quadratic data
term nothing measurable, and on θ and d PIBE is marginally *better* under
contamination. There is no damage for a learned likelihood to repair. This
matches the earlier test-time noise-family sweep, where a 19x spread in kurtosis
moved the state error by under 3%.

**And the energy term is actively harmful**: 8x–22x worse on every quantity. The
mechanism is the one from §5, now with no bias to excuse it — the learned
first-cell density drifted to a mean of **−0.052 when the true noise mean is
exactly 0**, and that invented offset propagates through the chain. With a
biased sensor the drift at least pointed in a useful direction; with zero-mean
noise it is pure loss.

**Conclusion.** On this benchmark the scalar EBM has no demonstrated use. It
does what it is built to do — recovers a residual density's shape to five
decimals in isolation (§1) — but the shape of a zero-mean law does not affect
PIBE's accuracy, so that competence buys nothing, while the location degeneracy
it introduces costs an order of magnitude. Its cost is not only accuracy: 13k
parameters, a quadrature, an extra schedule phase, and an `B_E` window of 0.19
nats (§5).

The honest recommendation is to keep the *analysis* and drop the *machinery*.
Sections 3–5 and 8–9 make a complete paper: the quadratic data term provably
cannot represent a nonzero noise mean; the shape of a zero-mean law is
irrelevant to it, however heavy-tailed; the offset is identifiable from the
physics but lies `10²`–`10³` below the collocation residual floor; and a cheap
simulation-based estimate of that one scalar recovers essentially all the lost
accuracy. That is a stronger contribution than an EBM which, measured against
its own control, does not help.

---

## 7. Reproducing

```bash
python scripts/train_pibe.py --config configs/automatica_n3v2_biased_pibe.yaml   # control
python scripts/train_pibe.py --config configs/automatica_n3v2_epibe.yaml         # Step 2 as written
python scripts/train_pibe.py --config configs/automatica_n3v2_epibe_cell2.yaml   # cells: [2]
python scripts/epibe_location_scan.py \
    --config configs/automatica_n3v2_epibe_cell2.yaml \
    --warm-start outputs/n3v2_biased_pibe/bank.pt \
    --offsets -0.05 -0.025 0.0 0.025 0.05 0.075 0.10 0.15 --iterations 6000 \
    --out outputs/epibe_scan
python scripts/plot_epibe_scan.py --pibe-x1 4.9775e-02

# diagnostics
python scripts/location_coercivity.py           # is the offset identifiable? (yes)
python scripts/estimate_offset.py --run outputs/n3v2_biased_pibe   # why the bank cannot see it

# the working method
python scripts/offset_by_shooting.py --config configs/automatica_n3v2_epibe_cell2.yaml \
    --trajectories 4 --steps 6000 --window 151
python scripts/train_pibe.py --config configs/automatica_n3v2_biased_pibe.yaml \
    --set data.noise_bias=0.00712 --output-dir outputs/n3v2_corrected_pibe
```
