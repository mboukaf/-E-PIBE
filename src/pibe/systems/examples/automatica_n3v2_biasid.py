r"""Third-order benchmark with an identifiable sensor offset.

.. math::

    \dot x_1 &= x_2 - 0.35\tanh(x_1) + \theta_1\bigl(1.3 + 0.7\cos x_1\bigr), \\
    \dot x_2 &= x_3 - 0.30\tanh(x_2) + 0.10\sin x_1
              + \theta_2\bigl(1.2 + 0.6\cos 2x_2\bigr), \\
    \dot x_3 &= -0.648x_1 - 2.34x_2 - 2.7x_3 + 0.15\sin x_1
              + \mathbf{0.5\sin 3x_1} + 0.08\tanh(x_1 x_2) \\
             &\quad + 0.08\theta_1\sin x_2 + 0.06\theta_2\tanh x_3 + d(t), \\
    y &= x_1 + \omega .

Identical to :mod:`~pibe.systems.examples.automatica_n3v2` except for the
single term :math:`0.5\sin 3x_1` in the last equation.

Why n3v2 cannot see a sensor offset
-----------------------------------
Displace the measured coordinate, :math:`\hat x_1 = x_1 + c`, rebuild
:math:`\hat x_2, \hat x_3` from the triangular form and refit
:math:`\hat\theta, \hat a`.  On n3v2 the last residual that remains costs
:math:`3.8\times10^{-5}` at :math:`c = 0.5` --- and :math:`0.43` if
:math:`\theta` is held at its true value.  The difference is :math:`\theta_2`:
after the transient :math:`x_2 \approx 0`, so
:math:`\theta_2(1.2 + 0.6\cos 2x_2)` is a *constant input*, and a constant input
and a constant output offset are indistinguishable at steady state.  The fit
moves along :math:`\hat\theta_2 \approx \theta_2 + 0.29c` at almost no cost,
which is exactly the parameter error PIBE reports under a biased sensor.
Neither a slower nor a larger disturbance helps: the signature of a smooth
function of :math:`x_1(t)` driven at the basis frequency lies mostly in the span
of the basis itself.

What the extra term changes
---------------------------
:math:`\partial f_3/\partial x_1` picks up :math:`1.5\cos 3x_1`, which near the
operating point :math:`x_1 \approx 1.25` moves from :math:`-1.87` to
:math:`+0.1` over half a unit.  An offset therefore changes the *local gain* of
the last equation, which no constant parameter shift can imitate.  The same
profile, with :math:`\theta` searched on :math:`[-1, 1]^2` rather than its
admissible box so the box cannot be what identifies it:

=========  =============  ==============
offset c   n3v2           this system
=========  =============  ==============
0.10       ~2e-6          6.8e-04
0.25       1.7e-05        3.5e-03
0.50       3.8e-05        1.7e-02
=========  =============  ==============

with a unique minimum at :math:`c = 0` over :math:`[-1, 1.5]`.  The
linearization stays Hurwitz on the realized trajectories and every range is
unchanged: the observed state maxima are :math:`(1.40, 0.49, 0.61)`.
"""

from __future__ import annotations

import torch
from torch import Tensor

from pibe.systems.examples.automatica_n3v2 import AutomaticaN3V2System
from pibe.systems.registry import register_system


@register_system("automatica_n3v2_biasid")
class AutomaticaN3V2BiasIdSystem(AutomaticaN3V2System):
    r"""n3v2 with a curvature term in :math:`x_1` that makes a sensor offset observable."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.name = "automatica_n3v2_biasid"

    def f_last(self, x: Tensor, theta: Tensor) -> Tensor:
        return super().f_last(x, theta) + 0.5 * torch.sin(3.0 * x[..., 0])
