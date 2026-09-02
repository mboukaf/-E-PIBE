r"""A smooth triangular chain, used as a test fixture.

.. warning::

   This is **not** the system of Section 5 of the paper --- the simulation
   section of the source PDF is truncated before the system is stated.  It is a
   placeholder chosen to exercise every generic code path (arbitrary ``n``,
   nonlinearities that genuinely depend on several of their arguments, a
   disturbance entering the last equation only).  Replace it with the paper's
   system by writing another :class:`~pibe.systems.base.TriangularSystem`
   subclass; nothing outside this file needs to change.

The dynamics are

.. math::

    f_j(x_1,\dots,x_j, \theta_1,\dots,\theta_j)
        &= \theta_j \sin(x_j)
         + \kappa \tanh\Bigl(\sum_{i=1}^{j-1} \theta_i x_i\Bigr),
        \qquad j = 1,\dots,n-1, \\
    f_n(x, \theta) &= -\sum_{i=1}^{n} c_i x_i
         + \kappa \tanh\Bigl(\sum_{i=1}^{n-1} \theta_i x_i\Bigr),

where the :math:`c_i` are the coefficients of :math:`(s+1)^n`, so the linear
part is a companion matrix with all poles at :math:`-1`.  The empty sum makes
:math:`f_1 = \theta_1 \sin(x_1)`.  Every nonlinearity is bounded, so
trajectories stay in a compact set and Assumption 1 holds on a generous box.

Every :math:`f_j` depending on its *entire* argument list is deliberate and
load-bearing for the test suite: it means any misordering of the residual's
arguments (Eq. (31)) --- a state taken from the wrong cell, or
:math:`\theta_i` paired with the wrong index --- changes the computed value and
is caught, instead of passing silently as it would for a system whose
:math:`f_j` ignores most of its arguments.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from pibe.systems.base import TriangularSystem
from pibe.systems.registry import register_system


@register_system("sin_chain")
class SinChainSystem(TriangularSystem):
    r"""A stable triangular chain with bounded sinusoidal nonlinearities.

    Parameters
    ----------
    n
        State dimension.
    theta_true
        True parameters, shape ``(n - 1,)``.  Defaults to a spread of values
        inside ``theta_bound``.
    kappa
        Coupling strength of the :math:`\tanh(x_1)` term.
    state_bound, x0_bound, theta_bound
        Half-widths of the symmetric boxes :math:`\mathcal{X}_j`,
        :math:`\chi_0` and :math:`\Theta_j`.
    """

    def __init__(
        self,
        n: int = 3,
        theta_true: Tensor | list[float] | None = None,
        kappa: float = 0.3,
        state_bound: float = 8.0,
        x0_bound: float = 0.5,
        theta_bound: float = 2.0,
        dtype: torch.dtype = torch.float64,
        **kwargs,
    ) -> None:
        if theta_true is None:
            theta_true = [
                0.8 * theta_bound * math.cos(1.7 * (j + 1)) for j in range(n - 1)
            ]
        state_bounds = torch.tensor([[-state_bound, state_bound]], dtype=dtype).expand(
            n, 2
        )
        theta_bounds = torch.tensor([[-theta_bound, theta_bound]], dtype=dtype).expand(
            n - 1, 2
        )
        x0_bounds = torch.tensor([[-x0_bound, x0_bound]], dtype=dtype).expand(n, 2)

        super().__init__(
            n=n,
            state_bounds=state_bounds,
            theta_bounds=theta_bounds,
            theta_true=torch.as_tensor(theta_true, dtype=dtype),
            x0_bounds=x0_bounds,
            dtype=dtype,
            name="sin_chain",
            **kwargs,
        )
        self.kappa = float(kappa)
        # Coefficients of (s + 1)^n, giving a companion matrix with all poles
        # at -1.  c[i] multiplies x_{i+1}.
        self.register_companion_coefficients(dtype)

    def register_companion_coefficients(self, dtype: torch.dtype) -> None:
        """Store the Hurwitz coefficients of the linear part."""
        self.companion = torch.tensor(
            [math.comb(self.n, i) for i in range(self.n)], dtype=dtype
        )

    def f(self, j: int, x: Tensor, theta: Tensor) -> Tensor:
        if not 1 <= j <= self.n - 1:
            raise ValueError(f"f_j is defined for 1 <= j <= {self.n - 1}, got j={j}")
        if x.shape[-1] != j or theta.shape[-1] != j:
            raise ValueError(
                f"f_{j} expects {j} states and {j} parameters, got "
                f"{x.shape[-1]} and {theta.shape[-1]}"
            )
        value = theta[..., j - 1] * torch.sin(x[..., j - 1])
        if j >= 2:
            coupling = (theta[..., : j - 1] * x[..., : j - 1]).sum(dim=-1)
            value = value + self.kappa * torch.tanh(coupling)
        return value

    def f_last(self, x: Tensor, theta: Tensor) -> Tensor:
        if x.shape[-1] != self.n or theta.shape[-1] != self.theta_dim:
            raise ValueError(
                f"f_n expects {self.n} states and {self.theta_dim} parameters, got "
                f"{x.shape[-1]} and {theta.shape[-1]}"
            )
        companion = self.companion.to(device=x.device, dtype=x.dtype)
        linear = -(x * companion).sum(dim=-1)
        coupling = (theta * x[..., : self.theta_dim]).sum(dim=-1)
        return linear + self.kappa * torch.tanh(coupling)

    def to(self, device=None, dtype=None):
        super().to(device=device, dtype=dtype)
        self.companion = self.companion.to(device=device, dtype=dtype)
        return self
