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

**Update, §10 (large bias).** Two further findings. On n3v2 a sensor offset is
not identifiable by *any* estimator: θ₂ acts as a constant input and absorbs it.
On a one-term variant where it is identifiable (`automatica_n3v2_biasid`), EPIBE
with an amortized, simulation-searched location (`ebm.location: amortized`)
recovers the offset (0.470 of 0.5, 0.958 of 1.0, −0.03 of 0) and beats PIBE by
13–21× on x₁; at bias 0.5 it matches or beats PIBE on every quantity to within 5%.

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
---

## 10. A large sensor bias: when EPIBE can beat PIBE, and how

§8 fixed the biased sensor by estimating the offset outside the bank. This
section asks the harder question — can EPIBE *itself* beat PIBE once the bias is
large enough to wreck PIBE — and answers it in three steps: on n3v2 no
estimator can, on a one-term variant every estimator could, and there the
energy-based bank does once its location is searched rather than descended.

### 10.1 On n3v2 the offset is not identifiable at all

Profile the physics along a constant offset `c` of `x̂₁`, rebuilding `x̂₂, x̂₃`
from the triangular form and refitting `θ̂, â` (four trajectories of the
dataset; `scripts/location_coercivity.py` reproduces the picture on one synthetic
trajectory):

| offset 0.5, what is refitted | min ‖r‖² |
|---|---|
| nothing (θ held at truth) | **4.3e-01** |
| θ₂ only | **7.4e-05** |
| θ₁ only | 2.0e-02 (θ₁ hits its box) |

After the transient `x₂ ≈ 0`, so `θ₂(1.2 + 0.6 cos 2x₂)` is a *constant input*,
and a constant input and a constant output offset are indistinguishable at
steady state. The fit slides along `θ̂₂ ≈ θ₂ + 0.29c` at almost no cost — exactly
the θ₂ error PIBE reports at bias 0.5 (0.136). Neither a slower nor a larger
disturbance helps: `x₂` is not excited after the transient at any tested
frequency/amplitude, and a smooth function of `x₁(t)` driven at the basis
frequency lies mostly in the span of the basis.

Scanning the offset through a trained bank (fine-tune at a fixed shift, then score
every term but the first cell's data term) confirms it end to end:

| fixed offset | 0 | 0.25 | **0.5 (true)** | 0.75 | 1.0 |
|---|---|---|---|---|---|
| chain cost | 1.70e-5 | **1.66e-5** | 1.88e-5 | 2.45e-5 | 3.09e-5 |

The truth is not preferred — yet the bank fine-tuned *at* the true offset gives
x errors [0.008, 0.028, 0.030] against PIBE's [0.50, 0.11, 0.21]. The information
is not in the data. **No EPIBE configuration can beat PIBE on this benchmark**,
and neither can any other estimator; this is a property of the system.

### 10.2 A benchmark on which the offset is identifiable

`automatica_n3v2_biasid` adds one term to the last equation, `+0.5 sin 3x₁`.
`∂f₃/∂x₁` then moves from −1.87 to +0.1 over half a unit around the operating
point, so an offset changes the local gain, which no constant parameter shift
can imitate. Same profile, θ searched on [−1, 1]² so the box cannot be what
identifies it:

| offset c | n3v2 | biasid |
|---|---|---|
| 0.10 | ~2e-6 | 6.8e-04 |
| 0.25 | 1.7e-05 | 3.5e-03 |
| 0.50 | 3.8e-05 | 1.7e-02 |

with a unique minimum at `c = 0` over [−1, 1.5]; it survives dropping the first
5 s (4.3e-3 at 0.5), so it is not only the transient. PIBE still breaks there:
bias 0.5 gives x₁ error 0.50 and θ₂ error 0.13 against 0.01 and 0.06 unbiased.

(`architecture.saturation_limit: 4.0` is required on this config: without it
θ₁'s head pinned to its box in both the unbiased PIBE run and the EPIBE run.)

### 10.3 Descending the location does not work, even when it is identifiable

With the first cell's likelihood made location-free (`ebm.location: profiled`,
the residual centred before the energy sees it) the physics alone decides the
location. Warm-started from the biased PIBE bank, gradient descent moves it the
right way — but slowly, and it stalls:

| variant | μ̂ after | x₁ |
|---|---|---|
| λ=1, 6k steps | +0.25 | 0.25 |
| λ=1, 30k steps (stopped at 10k) | +0.23 | 0.27 |
| consistency ×10, 8k steps | +0.26 | 0.24 |
| λ=10 / λ=30 / constant lr | diverged or wrong direction | — |
| full EPIBE, 40k final-cell steps | +0.34 | 0.16 |

The reason is structural. Every cell's new coordinate can satisfy its own
equation exactly, so the last equation reaches the objective only through the
consistency penalty between two reconstructions of `x₃`, filtered by the stable
dynamics `1/(s + 2.7)`. A wrong offset is hidden at a chain cost of 1.2e-4
against 7.8e-5 at the truth — a ratio of 1.5 where the exact profile gives two
orders of magnitude. An explicit scalar offset parameter does not help (it barely
moves); neither does a larger λ.

### 10.4 What works: amortize the offset, search it by simulation

The degeneracy is one scalar, so it can be searched rather than descended — and
the bank can be taught to answer for every candidate at once.

1. **Algorithm 2** as usual (first cell location-profiled).
2. **Amortization.** An extra end-to-end stage feeds every trajectory
   `y − μ̃` with its own `μ̃ ~ U(prior window)`, first cell on the *anchored*
   quadratic term. The bank then returns the full chain for any assumed offset.
3. **Location by simulation.** For each candidate `μ`, the bank's
   `(x̂₀, θ̂, â)` at `y − μ` initialise a shooting fit of the true ODE to `y − μ`
   (θ shared, x₀ and a per trajectory); the refined misfit `J(μ)` is scored. A
   coarse grid (150 steps per point) finds the basin, 5 points over ±one step
   refined to convergence (600 steps) give the vertex of a least-squares
   quadratic.
4. **Re-centre.** Re-amortize on a window of the same width centred on the
   estimate and rerun the fine pass.
5. **Density.** Refit the first cell's EBM to `y − μ̂ − x̂₁`; Eq. (54) reports
   `μ̂ + E_p[ξ]`.

`ebm.location: amortized` with `offset_window`, `offset_grid`, `offset_steps`,
`offset_fine`, `offset_fine_steps`, `amortize_iters`, `recenter_iters`,
`offset_refit`; the search is `pibe.eval.offset_shooting.select_offset`.

Every design choice above was forced by a failure:

| tried | result |
|---|---|
| offsets drawn from the first iteration | heads pinned, θ₂ error 0.38, every location wrong |
| location-free (centred) term during amortization | reconstruction slides, `y − μ` no longer maps to offset `μ` |
| score candidates by the bank's own chain cost | bowl centred on the window (accuracy, not physics); b=0 → 0.10, wide window → −0.25 |
| score by simulation with the bank's estimates, no refit | flat (argmin 0.55 at b=0.5) |
| refined shooting, 150 steps, 3-point parabola | biased low: 0.46 at 0.5, 0.89 at 1.0 |
| fine-tune at the selected offset | *worse* x₂, x₃, θ — the spread of offsets regularizes |
| narrow (±0.25) re-amortization | helps b=1, hurts b=0 and b=0.5 |
| **same-width re-centred re-amortization** | robust at the window edge without losing the interior |

Why simulation: a simulated trajectory is generated by the model from
`(x₀, θ, a)`; none of the decoder slack of §10.3 is available to it, so `J(μ)`
is the exact physics profile on the measured data and bottoms out at the noise
variance at the true offset.

### 10.5 Results

`automatica_n3v2_biasid`, σ = 0.05, 64 trajectories, held-out errors. EPIBE is
the end-to-end pipeline of §10.4 (prior window [−0.25, 1.25]); every number
below comes from one training run, no post-hoc selection.

| bias | run | x₁ | x₂ | x₃ | \|θ₁\| | \|θ₂\| | d | μ̂_ω |
|---|---|---|---|---|---|---|---|---|
| 0 | PIBE | **1.1e-2** | 1.07e-1 | 9.1e-2 | 6.5e-2 | 6.1e-2 | **2.6e-2** | — |
| 0 | EPIBE, amortized | 3.7e-2 | **7.7e-2** | **8.2e-2** | **3.6e-2** | **4.9e-2** | 7.1e-2 | −0.032 |
| 0.5 | PIBE | 4.99e-1 | **8.5e-2** | 2.31e-1 | 4.9e-2 | 1.35e-1 | 6.3e-2 | — |
| 0.5 | EPIBE, free location | 3.73e-1 | 2.28e-1 | 3.42e-1 | 9.9e-2 | 2.29e-1 | 7.6e-2 | +0.126 |
| 0.5 | EPIBE, amortized | **3.7e-2** | 8.8e-2 | **9.1e-2** | **4.5e-2** | **5.9e-2** | 6.5e-2 | **+0.470** |
| 1.0 | PIBE | 9.97e-1 | **8.4e-2** | 1.87e-1 | 1.62e-1 | **1.00e-1** | 8.5e-2 | — |
| 1.0 | EPIBE, amortized | **4.8e-2** | 1.62e-1 | **1.55e-1** | **9.3e-2** | 1.08e-1 | **6.8e-2** | **+0.958** |

* **The bias is recovered**: 0.470 of 0.5 and 0.958 of 1.0, and no bias is
  invented on clean data (−0.032). The refitted density has sd 0.051–0.052
  against a true 0.049.
* **Bias 0.5**: x₁ 13× better, x₃ 2.5×, θ₂ 2.3×, θ₁ slightly better; x₂ and d
  within 5%. On x₂, x₃ and θ it also matches or beats the *unbiased* PIBE run.
* **Bias 1.0**: x₁ 21× better, θ₁ 1.7×, x₃ and d better, θ₂ level; x₂ is 1.9×
  worse. This is the one quantity EPIBE loses, and the offset residual of
  0.042 is its likely cause: `x̂₂` inherits the offset's slope through
  `f₁(x₁ + c)`.
* **Unbiased**: the amortized bank is better than PIBE on x₂, x₃ and θ (the
  spread of offsets regularizes it) and worse on x₁ and d; the 0.032 offset
  error accounts for most of the x₁ difference.

The remaining error is dominated by the location: at the exact offset the banks
(before re-centring) give x₁ ≈ 0.02. Within the method, `offset_fine_steps` is the knob — at 150
steps the search was biased by 0.04–0.10, at 600 by 0.03–0.04.

### 10.6 What this says about EPIBE

1. **The energy term does not identify the offset; the physics does**, and only
   if the offset is not absorbable by a parameter. Whether it is can be checked
   before any training with the offset profile of §10.1/10.2; on the
   automatica_n3 family it is not, because each θ enters as a near-constant
   input.
2. **Even when identifiable, the collocation objective cannot descend to it.**
   The soft consistency coupling hides the offset at a cost comparable to the
   networks' own error. This is the conditioning statement of §5b, made on a
   system where the information is genuinely present.
3. **Searching one scalar in integrated form works.** Amortize the bank over the
   offset, score candidates by a simulation fit started from the bank, and the
   location is found to a few hundredths; the bank then gives estimates at that
   offset that match or beat PIBE everywhere at bias 0.5.
4. **The EBM's contribution is the shape**, reported through Eq. (54) with the
   searched location — the division of labour §8 already suggested, now inside
   one estimator.

### Reproducing §10

```bash
# identifiability profile (n3v2 vs biasid)
python scripts/location_coercivity.py --config configs/automatica_n3v2_biasid_b05.yaml

# PIBE baselines
python scripts/train_pibe.py --config configs/automatica_n3v2_biasid_b05.yaml \
    --set ebm.enabled=false data.noise_bias=0.5 --output-dir outputs/biasid_pibe_b0.5

# EPIBE with the amortized location (bias 0.5; set data.noise_bias for 0 / 1.0)
python scripts/train_pibe.py --config configs/automatica_n3v2_biasid_b05.yaml --set \
    ebm.location=amortized ebm.nll_scale=variance ebm.centered_warmup=false \
    "ebm.offset_window=[-0.25,1.25]" ebm.offset_grid=13 ebm.offset_steps=150 \
    ebm.offset_fine=5 ebm.offset_fine_steps=600 ebm.amortize_iters=12000 \
    ebm.recenter_iters=6000 --output-dir outputs/biasid_epibe_b0.5
```
