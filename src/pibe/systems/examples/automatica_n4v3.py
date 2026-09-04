r"""Fourth-order benchmark, revision 3 --- identifiability vs fittability balance.

.. math::

    \dot x_1 &= x_2 - 0.35\tanh(x_1) + \theta_1\bigl(1.3 + 0.7\cos x_1\bigr), \\
    \dot x_2 &= x_3 - 0.30\tanh(x_2) + 0.10\sin x_1
              + \theta_2\bigl(1.2 + 0.6\cos 6x_2\bigr), \\
    \dot x_3 &= x_4 - 0.25\tanh(x_3) + 0.10\sin(x_1 + x_2)
              + \theta_3\bigl(1.15 + 0.55\cos 10x_3\bigr), \\
    \dot x_4 &= -0.576x_1 - 2.736x_2 - 4.76x_3 - 3.6x_4 + 0.15\sin x_1
              + 0.08\tanh(x_2 x_3) \\
             &\quad + 0.08\theta_1\sin x_2 + 0.06\theta_2\sin x_3
              + 0.05\theta_3\tanh x_4 + d(t), \\
    y &= x_1 + \omega.

Identical to :mod:`~pibe.systems.examples.automatica_n4` except for how
:math:`\theta` enters.  The last equation, the poles, the basis, the admissible
sets and every sampling range are unchanged --- design E below needs no wider
box.

The design rule
---------------
Remark 7's degeneracy absorbs a parameter error :math:`e` into a state shift
:math:`c_k(t) = -e\,\partial_{\theta_{k-1}} f_{k-1}(t)`.  A *constant* shift is
exactly what the final residual's companion row annihilates, so the degeneracy
is severe precisely when the sensitivity is near-constant along the realized
trajectories.  Two requirements follow, neither implied by Assumption 6:

1. each :math:`\partial_{\theta_j} f_j` must **vary substantially** on the
   trajectories the system actually produces, and
2. different parameters must key off **different coordinates**, or their
   compensations re-align.

Why the multipliers are moderate

Revision 2 used ``m = (1, 6, 10)``, which the curvature measurement rates at
27.5x.  Trained, it was *worse* than revision 1 on every parameter.  The
measurement assumes the chain is reconstructed exactly; rapidly varying
sensitivities make the system markedly harder to **fit**, and the lost fit
accuracy costs more than the extra curvature buys.  ``m = (1, 4, 6)`` rates at
9.1x with a far smaller fittability penalty.

Frequencies are matched to each coordinate's span
-------------------------------------------------
The second requirement is not enough on its own here.  Keying
:math:`\theta_3` to :math:`x_3` achieves nothing if :math:`x_3` only ranges over
:math:`\pm 0.4`: :math:`\cos(3x_3)` then sweeps 2.4 rad and barely moves.  The
multiplier must be chosen so the argument sweeps of order :math:`2\pi` across
the coordinate's *observed* span.  Measured over 120 random trials:

===========  ==================  =================================
coordinate   observed half-span  multiplier used
===========  ==================  =================================
`x_1`        2.62                1   (sweeps 5.2 rad)
`x_2`        0.68                6   (sweeps 8.2 rad)
`x_3`        0.40                10  (sweeps 8.0 rad)
===========  ==================  =================================

Measured effect
---------------
Identifiability is measured directly: reconstruct the chain exactly under a
parameter error :math:`e_1`, let :math:`\theta_2, \theta_3` and :math:`\hat a`
absorb whatever they can, and record the residual that remains.  Its curvature
in :math:`e_1` *is* the identifiability.

=========================  ========  ========  ========  ======
design                     s1 var    s2 var    s3 var    gain
=========================  ========  ========  ========  ======
revision 1                 29%       **4%**    **1%**    1.0x
m = (1, 2, 3)              106%      41%       33%       5.0x
m = (1, 4, 6)              98%       171%      135%      9.1x
**m = (1, 6, 10)**         96%       182%      183%      **27.5x**
=========================  ========  ========  ========  ======

On the trajectories used for that measurement, revision 1's second and third
sensitivities varied by only 4% and 1% --- effectively constant, hence
effectively unidentifiable, despite satisfying Assumption 6 comfortably.  Over a
broader 32-trajectory sample they reach 19% and 21%, against 200% and 183% for
this revision; the figure depends on how much of the state space the
trajectories explore, which is exactly why the criterion must be evaluated on
*realized* trajectories rather than over the admissible box.

A caution about :math:`\gamma_k`
--------------------------------
This revision *lowers* the excitation constants --- :math:`\gamma_4` falls from
0.81 to 0.36 --- while raising identifiability 27.5-fold.  That is the clearest
statement of the point: :math:`\gamma_k > 0` is necessary but not sufficient,
and a design optimized for :math:`\gamma_k` alone can be strictly worse.
"""

from __future__ import annotations

import torch
from torch import Tensor

from pibe.systems.base import TriangularSystem
from pibe.systems.registry import register_system

COMPANION = (0.576, 2.736, 4.76, 3.6)

# Unchanged from revision 1: design E stays inside these.
STATE_BOUNDS = ((-4.0, 4.0), (-1.2, 1.2), (-0.8, 0.8), (-1.2, 1.2))
X0_BOUNDS = ((-0.8, 0.8), (-0.5, 0.5), (-0.4, 0.4), (-0.3, 0.3))

THETA_SAMPLING_BOUND = 0.25
THETA_BOUND = 0.4

# Multipliers matched to each coordinate's observed span.
FREQUENCIES = (1.0, 4.0, 6.0)

# Floors of the on-trajectory sensitivity ranges, over 300 random trials.
SENSITIVITY_FLOORS = (0.634, 0.600, 0.600)
EXCITATION_FLOORS = (0.402, 0.360, 0.360)


@register_system("automatica_n4v3")
class AutomaticaN4V3System(TriangularSystem):
    r"""Fourth-order benchmark with strongly varying parameter sensitivities."""

    def __init__(
        self,
        theta_true: Tensor | list[float] | None = None,
        theta_seed: int = 0,
        theta_bound: float = THETA_BOUND,
        dtype: torch.dtype = torch.float64,
        **kwargs,
    ) -> None:
        state_bounds = torch.tensor(STATE_BOUNDS, dtype=dtype)
        x0_bounds = torch.tensor(X0_BOUNDS, dtype=dtype)
        theta_bounds = torch.tensor(
            [[-theta_bound, theta_bound]], dtype=dtype
        ).expand(3, 2).clone()

        if theta_true is None:
            generator = torch.Generator().manual_seed(theta_seed)
            unit = torch.rand(3, generator=generator, dtype=dtype)
            theta_true = THETA_SAMPLING_BOUND * (2.0 * unit - 1.0)

        super().__init__(
            n=4,
            state_bounds=state_bounds,
            theta_bounds=theta_bounds,
            theta_true=torch.as_tensor(theta_true, dtype=dtype),
            x0_bounds=x0_bounds,
            dtype=dtype,
            name="automatica_n4v3",
            **kwargs,
        )
        self.companion = torch.tensor(COMPANION, dtype=dtype)

    # ------------------------------------------------------------------
    # nonlinearities
    # ------------------------------------------------------------------

    def f(self, j: int, x: Tensor, theta: Tensor) -> Tensor:
        if not 1 <= j <= 3:
            raise ValueError(f"f_j is defined for 1 <= j <= 3, got j={j}")
        if x.shape[-1] != j or theta.shape[-1] != j:
            raise ValueError(
                f"f_{j} expects {j} states and {j} parameters, got "
                f"{x.shape[-1]} and {theta.shape[-1]}"
            )
        if j == 1:
            x1 = x[..., 0]
            # s_1 keyed to x_1 (span 2.62; multiplier 1 sweeps 5.2 rad).
            return -0.35 * torch.tanh(x1) + theta[..., 0] * (
                1.3 + 0.7 * torch.cos(FREQUENCIES[0] * x1)
            )
        if j == 2:
            x1, x2 = x[..., 0], x[..., 1]
            # s_2 keyed to x_2 (span 0.68; multiplier 6 sweeps 8.2 rad).
            return (
                -0.30 * torch.tanh(x2)
                + 0.10 * torch.sin(x1)
                + theta[..., 1] * (1.2 + 0.6 * torch.cos(FREQUENCIES[1] * x2))
            )
        x1, x2, x3 = x[..., 0], x[..., 1], x[..., 2]
        # s_3 keyed to x_3 (span 0.40; multiplier 10 sweeps 8.0 rad).
        return (
            -0.25 * torch.tanh(x3)
            + 0.10 * torch.sin(x1 + x2)
            + theta[..., 2] * (1.15 + 0.55 * torch.cos(FREQUENCIES[2] * x3))
        )

    def f_last(self, x: Tensor, theta: Tensor) -> Tensor:
        if x.shape[-1] != 4 or theta.shape[-1] != 3:
            raise ValueError(
                f"f_4 expects 4 states and 3 parameters, got "
                f"{x.shape[-1]} and {theta.shape[-1]}"
            )
        companion = self.companion.to(device=x.device, dtype=x.dtype)
        x1, x2, x3, x4 = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
        linear = -(x * companion).sum(dim=-1)
        return (
            linear
            + 0.15 * torch.sin(x1)
            + 0.08 * torch.tanh(x2 * x3)
            + 0.08 * theta[..., 0] * torch.sin(x2)
            + 0.06 * theta[..., 1] * torch.sin(x3)
            + 0.05 * theta[..., 2] * torch.tanh(x4)
        )

    # ------------------------------------------------------------------
    # excitation
    # ------------------------------------------------------------------

    def parameter_sensitivity(self, j: int, x: Tensor) -> Tensor:
        r"""The exact sensitivity :math:`\partial_{\theta_j} f_j`."""
        if j == 1:
            return 1.3 + 0.7 * torch.cos(FREQUENCIES[0] * x[..., 0])
        if j == 2:
            return 1.2 + 0.6 * torch.cos(FREQUENCIES[1] * x[..., 1])
        if j == 3:
            return 1.15 + 0.55 * torch.cos(FREQUENCIES[2] * x[..., 2])
        raise ValueError(f"sensitivity is defined for j = 1, 2, 3, got {j}")

    @staticmethod
    def sensitivity_floors() -> tuple[float, ...]:
        """Floors of the *on-trajectory* sensitivity ranges (300 trials)."""
        return SENSITIVITY_FLOORS

    @staticmethod
    def excitation_floors() -> tuple[float, ...]:
        r"""Conservative :math:`\gamma_k` of Assumption 6, cells ``k = 2, 3, 4``."""
        return EXCITATION_FLOORS

    def to(self, device=None, dtype=None):
        super().to(device=device, dtype=dtype)
        self.companion = self.companion.to(device=device, dtype=dtype)
        return self
