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

> **Test system.** Section 5 of the source PDF is truncated before the
> 6th-order test system is stated, so `sin_chain` is a placeholder chosen to
> exercise the generic code paths. Nothing here reproduces the paper's numbers.

## The system class

```
ẋ_j = x_{j+1} + f_j(x_1..x_j, θ_1..θ_j),   j = 1..n-1
ẋ_n = f_n(x, θ) + d(t)
y   = x_1 + ω(t)
d(t) = Γ_q(t)ᵀa + r_q(t)
```

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Run

```bash
python scripts/train_pibe.py --config configs/train_pibe.yaml
python scripts/train_pibe.py --config configs/train_pibe.yaml --dry-run   # inspect setup only
python scripts/evaluate.py --checkpoint outputs/pibe_run/bank.pt
pytest -q
```

Override any config entry from the command line:

```bash
python scripts/train_pibe.py --config configs/train_pibe.yaml \
  --set training.lam=0.5 system.kwargs.n=6 --output-dir outputs/n6
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
| Disturbance decomposition (2), basis Def. 1 (136) | [basis/bspline.py](src/pibe/basis/bspline.py), [data/disturbance.py](src/pibe/data/disturbance.py) |
| Linear independence (3) | `BSplineBasis.gram`, `check_linear_independence` |
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

## A behaviour to expect, not to debug

On a short run you will likely see `θ̂` sit near the centre of `Θ_j` and `â`
shrunk toward the centre of `A`, while the physics loss is small. This is
**Remark 7**, not a bug: the free second decoder output `x̂^k_k` can absorb a
parameter perturbation exactly, leaving both the data and physics terms at zero
while the parameter is wrong by any admissible constant `c`.

`tests/test_identifiability.py` pins both halves of this down — that the true
parameter *is* recoverable when the states are held at truth (so the model class
is right), and that Remark 7's construction really does drive the residual to
exactly zero at a wrong parameter.

The paper does not claim the local loss resolves this. Identification rests on
Assumption 6 (trajectory excitation, giving the `1/√γ_k` factor in Corollary 1)
and Remark 10's coercivity requirement, and Assumption 4 is stated as "a
post-training accuracy hypothesis, not a consequence of the Universal
Approximation Theorem". Practically, if parameters will not identify, the levers
are richer excitation across trajectories, more trajectories `P`, a larger `λ`,
and a longer fine-tuning phase — not changes to the residuals.

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
├── basis/       B-spline Γ_q (Definition 1)
├── data/        simulation, truncated-Gaussian noise, disturbances, datasets
├── nets/        encoder / decoder / heads, D_{k-1} normalization, box reparameterization
├── core/        cells, bank, residuals, losses, autodiff
├── training/    Algorithm 1, phase schedule, checkpoints
└── eval/        estimation metrics
```
