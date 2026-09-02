r"""The fourth-order benchmark system.

.. math::

    \dot x_1 &= x_2 - 0.35\tanh(x_1) + \theta_1\bigl(0.8 + 0.2\cos x_1\bigr), \\
    \dot x_2 &= x_3 - 0.30\tanh(x_2) + 0.10\sin x_1
              + \theta_2\bigl(0.9 + 0.1\sin x_1 + 0.1\cos x_2\bigr), \\
    \dot x_3 &= x_4 - 0.25\tanh(x_3) + 0.10\sin(x_1 + x_2)
              + \theta_3\bigl(1 + 0.1\sin(x_1 + x_3)\bigr), \\
    \dot x_4 &= -0.576x_1 - 2.736x_2 - 4.76x_3 - 3.6x_4
              + 0.15\sin x_1 + 0.08\tanh(x_2 x_3) \\
             &\quad + 0.08\theta_1\sin x_2 + 0.06\theta_2\sin x_3
              + 0.05\theta_3\tanh x_4 + d(t), \\
    y &= x_1 + \omega.

This is exactly the lower-triangular form of Eq. (1): :math:`f_j` depends only
on :math:`x_1,\dots,x_j` and :math:`\theta_1,\dots,\theta_j`, the disturbance
enters the last equation alone, and the output is the first coordinate.

Stable backbone
---------------
The linear skeleton of the last equation is a companion matrix with
characteristic polynomial

.. math:: (s + 0.6)(s + 0.8)(s + 1)(s + 1.2)
          = s^4 + 3.6s^3 + 4.76s^2 + 2.736s + 0.576,

which is where the coefficients :math:`0.576, 2.736, 4.76, 3.6` come from.  All
poles are in the open left half-plane and every nonlinearity is bounded, so
trajectories stay in a compact set.

Excitation
----------
The parameter sensitivities are uniformly bounded away from zero:

.. math::

    \partial_{\theta_1} f_1 &= 0.8 + 0.2\cos x_1 \ge 0.6, \\
    \partial_{\theta_2} f_2 &= 0.9 + 0.1\sin x_1 + 0.1\cos x_2 \ge 0.7, \\
    \partial_{\theta_3} f_3 &= 1 + 0.1\sin(x_1 + x_3) \ge 0.9,

giving the conservative excitation constants of Assumption 6,
:math:`\gamma_2 \ge 0.36`, :math:`\gamma_3 \ge 0.49`, :math:`\gamma_4 \ge
0.81` --- the squares of the sensitivity floors.  This matters directly: the
conditional parameter certificate of Corollary 1 carries a factor
:math:`1/\sqrt{\gamma_k}`, so a sensitivity that could vanish would leave the
parameter unidentifiable (Remark 6).

Admissible sets
---------------
The decoder ranges :math:`\mathcal{X}_1 = [-4,4]`, :math:`\mathcal{X}_2 =
[-1.2,1.2]`, :math:`\mathcal{X}_3 = [-0.8,0.8]`, :math:`\mathcal{X}_4 =
[-1.2,1.2]` leave comfortable margins over the empirical maxima
:math:`\max|x_j| \approx (2.65, 0.72, 0.41, 0.75)` observed across random
trials.  That is a numerical check rather than a proof of invariance, so
:func:`~pibe.data.simulate.check_admissible` still verifies Assumption 1 on
every generated dataset.
"""

from __future__ import annotations

import torch
from torch import Tensor

from pibe.systems.base import TriangularSystem
from pibe.systems.registry import register_system

# Companion coefficients of (s+0.6)(s+0.8)(s+1)(s+1.2), multiplying x_1..x_4.
COMPANION = (0.576, 2.736, 4.76, 3.6)

STATE_BOUNDS = ((-4.0, 4.0), (-1.2, 1.2), (-0.8, 0.8), (-1.2, 1.2))
X0_BOUNDS = ((-0.8, 0.8), (-0.5, 0.5), (-0.4, 0.4), (-0.3, 0.3))
THETA_BOUND = 0.25

# Conservative lower bounds on the parameter sensitivities, and the resulting
# excitation constants gamma_k of Assumption 6.
SENSITIVITY_FLOORS = (0.6, 0.7, 0.9)
EXCITATION_FLOORS = (0.36, 0.49, 0.81)


@register_system("automatica_n4")
class AutomaticaN4System(TriangularSystem):
    r"""The fourth-order triangular benchmark.

    Parameters
    ----------
    theta_true
        The true :math:`(\theta_1, \theta_2, \theta_3)`.  When omitted they are
        drawn once from :math:`\mathcal{U}[-0.25, 0.25]^3` using
        ``theta_seed``, then held constant --- matching the paper's assumption
        that :math:`\theta` is an unknown *constant* vector.
    theta_seed
        Seed for that draw; ignored when ``theta_true`` is given.
    """

    def __init__(
        self,
        theta_true: Tensor | list[float] | None = None,
        theta_seed: int = 0,
        dtype: torch.dtype = torch.float64,
        **kwargs,
    ) -> None:
        state_bounds = torch.tensor(STATE_BOUNDS, dtype=dtype)
        x0_bounds = torch.tensor(X0_BOUNDS, dtype=dtype)
        theta_bounds = torch.tensor(
            [[-THETA_BOUND, THETA_BOUND]], dtype=dtype
        ).expand(3, 2).clone()

        if theta_true is None:
            generator = torch.Generator().manual_seed(theta_seed)
            unit = torch.rand(3, generator=generator, dtype=dtype)
            theta_true = theta_bounds[:, 0] + unit * (
                theta_bounds[:, 1] - theta_bounds[:, 0]
            )

        super().__init__(
            n=4,
            state_bounds=state_bounds,
            theta_bounds=theta_bounds,
            theta_true=torch.as_tensor(theta_true, dtype=dtype),
            x0_bounds=x0_bounds,
            dtype=dtype,
            name="automatica_n4",
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
            return -0.35 * torch.tanh(x1) + theta[..., 0] * (
                0.8 + 0.2 * torch.cos(x1)
            )
        if j == 2:
            x1, x2 = x[..., 0], x[..., 1]
            return (
                -0.30 * torch.tanh(x2)
                + 0.10 * torch.sin(x1)
                + theta[..., 1]
                * (0.9 + 0.1 * torch.sin(x1) + 0.1 * torch.cos(x2))
            )
        x1, x2, x3 = x[..., 0], x[..., 1], x[..., 2]
        return (
            -0.25 * torch.tanh(x3)
            + 0.10 * torch.sin(x1 + x2)
            + theta[..., 2] * (1.0 + 0.1 * torch.sin(x1 + x3))
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
    # excitation, Assumption 6
    # ------------------------------------------------------------------

    def parameter_sensitivity(self, j: int, x: Tensor) -> Tensor:
        r"""The exact sensitivity :math:`\partial_{\theta_j} f_j`.

        Available in closed form because :math:`\theta_j` enters :math:`f_j`
        linearly.  Used to certify the excitation floors rather than to train.
        """
        if j == 1:
            return 0.8 + 0.2 * torch.cos(x[..., 0])
        if j == 2:
            return 0.9 + 0.1 * torch.sin(x[..., 0]) + 0.1 * torch.cos(x[..., 1])
        if j == 3:
            return 1.0 + 0.1 * torch.sin(x[..., 0] + x[..., 2])
        raise ValueError(f"sensitivity is defined for j = 1, 2, 3, got {j}")

    @staticmethod
    def sensitivity_floors() -> tuple[float, ...]:
        r"""Lower bounds on :math:`\partial_{\theta_j} f_j`, valid for all states."""
        return SENSITIVITY_FLOORS

    @staticmethod
    def excitation_floors() -> tuple[float, ...]:
        r"""Conservative :math:`\gamma_k` of Assumption 6, for cells ``k = 2, 3, 4``."""
        return EXCITATION_FLOORS

    # ------------------------------------------------------------------
    # placement
    # ------------------------------------------------------------------

    def to(self, device=None, dtype=None):
        super().to(device=device, dtype=dtype)
        self.companion = self.companion.to(device=device, dtype=dtype)
        return self
