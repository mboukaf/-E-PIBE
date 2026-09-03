r"""The third-order benchmark system.

.. math::

    \dot x_1 &= x_2 - 0.35\tanh(x_1) + \theta_1\bigl(0.8 + 0.2\cos x_1\bigr), \\
    \dot x_2 &= x_3 - 0.30\tanh(x_2) + 0.10\sin x_1
              + \theta_2\bigl(0.9 + 0.1\sin x_1 + 0.1\cos x_2\bigr), \\
    \dot x_3 &= -0.648x_1 - 2.34x_2 - 2.7x_3 + 0.15\sin x_1
              + 0.08\tanh(x_1 x_2) \\
             &\quad + 0.08\theta_1\sin x_2 + 0.06\theta_2\tanh x_3 + d(t), \\
    y &= x_1 + \omega.

Exactly the form of Eq. (1), and it exercises every cell of the bank: cell 2
estimates :math:`(x_2, \theta_1)`, cell 3 estimates :math:`(x_3, \theta_2)`,
and cell 4 reconstructs :math:`x_3` and estimates the disturbance
coefficients.  That :math:`f_2` does not use :math:`\theta_1` is permitted ---
the signature declares what a term *may* depend on.

Why the shorter chain matters
-----------------------------
Two structural properties improve over the fourth-order version, and both bear
directly on what limits accuracy.

**The disturbance is more observable.**  :math:`d` enters the last equation
while only :math:`x_1` is measured, so its trace at the output passes through
:math:`n` integrations.  With poles at :math:`-0.6, -0.9, -1.2` the gain is
:math:`|G(j\Omega)| = 0.267` and :math:`|G(2j\Omega)| = 0.052`, against
:math:`0.173` and :math:`0.019` for the fourth-order chain --- a factor
:math:`1.55` and :math:`2.67`.  See :mod:`pibe.eval.observability`.

**The degenerate family is smaller.**  Remark 7's construction lets a constant
parameter error be absorbed by a constant shift of the next state; the shifts
that survive are those invisible to the final residual, i.e. lying in the null
space of the companion row.  Here that is a single condition
:math:`2.34c_2 + 2.7c_3 = 0` on two offsets --- a **one**-dimensional flat
family, against two dimensions for :math:`n = 4`.

Stability and excitation
------------------------
The linear skeleton has characteristic polynomial

.. math:: (s+0.6)(s+0.9)(s+1.2) = s^3 + 2.7s^2 + 2.34s + 0.648,

which is Hurwitz.  The parameter sensitivities are bounded away from zero,

.. math::

    \partial_{\theta_1} f_1 &= 0.8 + 0.2\cos x_1 \ge 0.6, \\
    \partial_{\theta_2} f_2 &= 0.9 + 0.1\sin x_1 + 0.1\cos x_2 \ge 0.7,

giving :math:`\gamma_2 \ge 0.36` and :math:`\gamma_3 \ge 0.49` in Assumption 6.
Multiplicative forms such as :math:`\theta_i x_i` are deliberately avoided:
their sensitivity vanishes wherever :math:`x_i` crosses zero, which would make
:math:`\gamma_k = 0` and void the conditional certificate of Corollary 1.
"""

from __future__ import annotations

import torch
from torch import Tensor

from pibe.systems.base import TriangularSystem
from pibe.systems.registry import register_system

# Companion coefficients of (s+0.6)(s+0.9)(s+1.2), multiplying x_1..x_3.
COMPANION = (0.648, 2.34, 2.7)

STATE_BOUNDS = ((-2.0, 2.0), (-0.8, 0.8), (-0.8, 0.8))
X0_BOUNDS = ((-0.8, 0.8), (-0.5, 0.5), (-0.35, 0.35))

# Where the truth is drawn from ...
THETA_SAMPLING_BOUND = 0.25
# ... and the estimator's admissible set, deliberately wider.  The parameter
# head is a tanh reparameterization onto Theta_j, so a truth sitting near the
# boundary is reachable only in the saturated limit, where the gradient
# vanishes and the estimate pins to the edge for the rest of training.
THETA_BOUND = 0.4

SENSITIVITY_FLOORS = (0.6, 0.7)
EXCITATION_FLOORS = (0.36, 0.49)


@register_system("automatica_n3")
class AutomaticaN3System(TriangularSystem):
    r"""The third-order triangular benchmark.

    Parameters
    ----------
    theta_true
        The true :math:`(\theta_1, \theta_2)`.  When omitted they are drawn
        once from :math:`\mathcal{U}[-0.25, 0.25]^2` using ``theta_seed`` and
        then held constant, as Eq. (1) requires.
    theta_seed
        Seed for that draw; ignored when ``theta_true`` is given.
    theta_bound
        Half-width of the admissible set :math:`\Theta_j`.
    """

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
            name="automatica_n3",
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
            return -0.35 * torch.tanh(x1) + theta[..., 0] * (
                0.8 + 0.2 * torch.cos(x1)
            )
        x1, x2 = x[..., 0], x[..., 1]
        return (
            -0.30 * torch.tanh(x2)
            + 0.10 * torch.sin(x1)
            + theta[..., 1] * (0.9 + 0.1 * torch.sin(x1) + 0.1 * torch.cos(x2))
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
    # excitation, Assumption 6
    # ------------------------------------------------------------------

    def parameter_sensitivity(self, j: int, x: Tensor) -> Tensor:
        r"""The exact sensitivity :math:`\partial_{\theta_j} f_j`."""
        if j == 1:
            return 0.8 + 0.2 * torch.cos(x[..., 0])
        if j == 2:
            return 0.9 + 0.1 * torch.sin(x[..., 0]) + 0.1 * torch.cos(x[..., 1])
        raise ValueError(f"sensitivity is defined for j = 1, 2, got {j}")

    @staticmethod
    def sensitivity_floors() -> tuple[float, ...]:
        r"""Lower bounds on :math:`\partial_{\theta_j} f_j` over all states."""
        return SENSITIVITY_FLOORS

    @staticmethod
    def excitation_floors() -> tuple[float, ...]:
        r"""Conservative :math:`\gamma_k` of Assumption 6, cells ``k = 2, 3``."""
        return EXCITATION_FLOORS

    # ------------------------------------------------------------------
    # placement
    # ------------------------------------------------------------------

    def to(self, device=None, dtype=None):
        super().to(device=device, dtype=dtype)
        self.companion = self.companion.to(device=device, dtype=dtype)
        return self
