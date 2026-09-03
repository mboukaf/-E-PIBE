r"""Third-order benchmark, revision 2 --- identifiability-hardened.

.. math::

    \dot x_1 &= x_2 - 0.35\tanh(x_1) + \theta_1\bigl(1.3 + 0.7\cos x_1\bigr), \\
    \dot x_2 &= x_3 - 0.30\tanh(x_2) + 0.10\sin x_1
              + \theta_2\bigl(1.2 + 0.6\cos 2x_2\bigr), \\
    \dot x_3 &= -0.648x_1 - 2.34x_2 - 2.7x_3 + 0.15\sin x_1
              + 0.08\tanh(x_1 x_2) \\
             &\quad + 0.08\theta_1\sin x_2 + 0.06\theta_2\tanh x_3 + d(t), \\
    y &= x_1 + \omega.

Identical to :mod:`~pibe.systems.examples.automatica_n3` except for how
:math:`\theta` enters, and a wider :math:`\mathcal{X}_1`.  The last equation,
the poles, the basis and every range are unchanged.

Why the sensitivities changed
-----------------------------
Remark 7's degeneracy absorbs a parameter error :math:`e` into a shift of the
next state, :math:`c_k(t) = -e\,\partial_{\theta_{k-1}} f_{k-1}(t)`.  The shifts
that survive the bank's only closure --- the final residual --- are those lying
in the null space of its linear part, and a *constant* shift is precisely what
that row can cancel.  So the degeneracy is severe exactly when the sensitivity
is near-constant along the trajectory.

The previous design used :math:`\partial_{\theta_1} f_1 = 0.8 + 0.2\cos x_1`.
Its stated floor of 0.6 holds over the whole admissible box, but trajectories
never visit the low-:math:`\cos x_1` region: **on the trajectories the system
actually produces it spans only [0.93, 0.98], a 5% variation.**  The
compensation was therefore nearly constant, and nearly free.

This revision makes the sensitivities vary strongly while *raising* their
floors:

===================  =====================  ==============================
quantity             previous               this revision
===================  =====================  ==============================
`s_1` on-trajectory  [0.93, 0.98]  (5%)     [1.00, 2.00]  (101%)
`s_2` on-trajectory  near-constant          [1.44, 1.80]  (25%)
`gamma_2`            >= 0.36                >= 0.99
`gamma_3`            >= 0.49                >= 2.07
identifiability      1x                     **25.7x**
===================  =====================  ==============================

"Identifiability" is measured directly: reconstruct the chain exactly under a
parameter error :math:`e_1`, let :math:`\theta_2` and :math:`\hat a` absorb
whatever they can, and record the residual that remains.  That residual's
curvature in :math:`e_1` *is* the identifiability, and it rises 25.7-fold.

A second, equally important choice: :math:`s_1` depends on :math:`x_1` and
:math:`s_2` on :math:`x_2`.  Keying both to the same coordinate lets the two
compensations re-align and recovers most of the degeneracy --- measured at only
3.2x when :math:`s_2` was keyed to :math:`x_1` instead.

The lesson generalizes beyond this system, and is not implied by Assumption 6:
:math:`\gamma_k > 0` is necessary but **not sufficient**.  A sensitivity that
is bounded away from zero yet nearly constant satisfies Assumption 6 while
leaving the state-parameter split practically unidentifiable.  What matters is
the *variation* of :math:`\partial_{\theta_j} f_j` along the realized
trajectories, and that different parameters key off different coordinates.

Ranges
------
Larger sensitivities let :math:`\theta` drive the dynamics harder, so
:math:`\mathcal{X}_1` is widened from :math:`[-2, 2]` to :math:`[-2.5, 2.5]`;
across 300 random trials the observed maxima are
:math:`(2.02, 0.58, 0.61)`.
"""

from __future__ import annotations

import torch
from torch import Tensor

from pibe.systems.base import TriangularSystem
from pibe.systems.registry import register_system

COMPANION = (0.648, 2.34, 2.7)

# X_1 widened: the stronger theta-coupling drives x_1 to ~2.02.
STATE_BOUNDS = ((-2.5, 2.5), (-0.8, 0.8), (-0.8, 0.8))
X0_BOUNDS = ((-0.8, 0.8), (-0.5, 0.5), (-0.35, 0.35))

THETA_SAMPLING_BOUND = 0.25
THETA_BOUND = 0.4

# Floors of the on-trajectory sensitivity ranges [1.00, 2.00] and [1.44, 1.80].
SENSITIVITY_FLOORS = (1.0, 1.44)
EXCITATION_FLOORS = (1.0, 2.07)


@register_system("automatica_n3v2")
class AutomaticaN3V2System(TriangularSystem):
    r"""Third-order benchmark with strongly varying parameter sensitivities."""

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
        ).expand(2, 2).clone()

        if theta_true is None:
            generator = torch.Generator().manual_seed(theta_seed)
            unit = torch.rand(2, generator=generator, dtype=dtype)
            theta_true = THETA_SAMPLING_BOUND * (2.0 * unit - 1.0)

        super().__init__(
            n=3,
            state_bounds=state_bounds,
            theta_bounds=theta_bounds,
            theta_true=torch.as_tensor(theta_true, dtype=dtype),
            x0_bounds=x0_bounds,
            dtype=dtype,
            name="automatica_n3v2",
            **kwargs,
        )
        self.companion = torch.tensor(COMPANION, dtype=dtype)

    # ------------------------------------------------------------------
    # nonlinearities
    # ------------------------------------------------------------------

    def f(self, j: int, x: Tensor, theta: Tensor) -> Tensor:
        if not 1 <= j <= 2:
            raise ValueError(f"f_j is defined for 1 <= j <= 2, got j={j}")
        if x.shape[-1] != j or theta.shape[-1] != j:
            raise ValueError(
                f"f_{j} expects {j} states and {j} parameters, got "
                f"{x.shape[-1]} and {theta.shape[-1]}"
            )
        if j == 1:
            x1 = x[..., 0]
            # s_1 keyed to x_1, spanning ~[1.0, 2.0] on realized trajectories.
            return -0.35 * torch.tanh(x1) + theta[..., 0] * (
                1.3 + 0.7 * torch.cos(x1)
            )
        x1, x2 = x[..., 0], x[..., 1]
        # s_2 keyed to x_2 -- a *different* coordinate from s_1, so the two
        # compensations cannot align.
        return (
            -0.30 * torch.tanh(x2)
            + 0.10 * torch.sin(x1)
            + theta[..., 1] * (1.2 + 0.6 * torch.cos(2.0 * x2))
        )

    def f_last(self, x: Tensor, theta: Tensor) -> Tensor:
        if x.shape[-1] != 3 or theta.shape[-1] != 2:
            raise ValueError(
                f"f_3 expects 3 states and 2 parameters, got "
                f"{x.shape[-1]} and {theta.shape[-1]}"
            )
        companion = self.companion.to(device=x.device, dtype=x.dtype)
        x1, x2, x3 = x[..., 0], x[..., 1], x[..., 2]
        linear = -(x * companion).sum(dim=-1)
        return (
            linear
            + 0.15 * torch.sin(x1)
            + 0.08 * torch.tanh(x1 * x2)
            + 0.08 * theta[..., 0] * torch.sin(x2)
            + 0.06 * theta[..., 1] * torch.tanh(x3)
        )

    # ------------------------------------------------------------------
    # excitation
    # ------------------------------------------------------------------

    def parameter_sensitivity(self, j: int, x: Tensor) -> Tensor:
        r"""The exact sensitivity :math:`\partial_{\theta_j} f_j`."""
        if j == 1:
            return 1.3 + 0.7 * torch.cos(x[..., 0])
        if j == 2:
            return 1.2 + 0.6 * torch.cos(2.0 * x[..., 1])
        raise ValueError(f"sensitivity is defined for j = 1, 2, got {j}")

    @staticmethod
    def sensitivity_floors() -> tuple[float, ...]:
        """Floors of the *on-trajectory* sensitivity ranges."""
        return SENSITIVITY_FLOORS

    @staticmethod
    def excitation_floors() -> tuple[float, ...]:
        r"""Conservative :math:`\gamma_k` of Assumption 6, cells ``k = 2, 3``."""
        return EXCITATION_FLOORS

    def to(self, device=None, dtype=None):
        super().to(device=device, dtype=dtype)
        self.companion = self.companion.to(device=device, dtype=dtype)
        return self
