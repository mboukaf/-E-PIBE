r"""How visible each disturbance basis function is at the output.

Eq. (3) asks only that :math:`\Gamma_q` be linearly independent *as functions of
time*, i.e. :math:`W_\Gamma \succeq \underline{\gamma}_d I_q`.  That is a
property of the basis alone and says nothing about the plant.  But the
disturbance enters the **last** state equation while only the **first**
coordinate is measured, so its trace in :math:`y` passes through the whole
chain --- :math:`n` integrations --- and is attenuated accordingly.  A basis can
therefore be perfectly conditioned in the sense of (3) and still have
components that leave no measurable trace.

This module quantifies that.  For each basis function :math:`\gamma_i` it
integrates the *linearized* dynamics driven by that function alone,

.. math:: \dot{\delta x} = A\,\delta x + E_d\,\gamma_i(t), \qquad
          \delta y = C\,\delta x,

and reports :math:`\|\delta y\|_{N,2}` per unit coefficient.  Multiplied by the
admissible coefficient magnitude and divided by the measurement-noise standard
deviation, that is the per-sample signal-to-noise ratio of the coefficient; a
further factor :math:`\sqrt{N}` gives what averaging over the grid can buy.

The practical reading:

- SNR well above 1 --- the coefficient is recoverable.
- SNR below 1 --- the coefficient is *below the noise floor* pointwise, and only
  long-horizon averaging can retrieve it.  No estimator, and no amount of
  training, fixes this; it is a property of the plant, the basis and
  :math:`\sigma`.  The levers are a smaller :math:`\sigma`, a lower-frequency
  basis, a longer horizon, or a smaller :math:`q`.

The analysis is a linearization, so it is indicative rather than exact for a
nonlinear system; its purpose is to tell you *before* training whether a
disturbance experiment is well posed.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from pibe.basis.base import DisturbanceBasis
from pibe.systems.base import TriangularSystem


@dataclass
class BasisObservability:
    r"""Per-basis-function observability through the plant.

    Attributes
    ----------
    output_gain
        :math:`\|\delta y\|_{N,2}` per unit coefficient over the whole
        horizon, shape ``(q,)``.  Includes the startup transient.
    settled_gain
        The same over the last ``1 - settle_fraction`` of the horizon.  The
        transient is a broadband kick that the chain's large DC gain amplifies,
        so it flatters the fast components; the settled figure is what actually
        identifies a *periodic* disturbance and is the one to compare across
        harmonics.
    signature
        ``output_gain`` times the admissible coefficient magnitude --- the trace
        the coefficient actually leaves in :math:`y`, shape ``(q,)``.
    snr
        ``signature / sigma``: the per-sample signal-to-noise ratio.
    effective_snr
        ``snr * sqrt(n_samples)``: what averaging over the data grid can buy.
    """

    output_gain: Tensor
    settled_gain: Tensor
    signature: Tensor
    snr: Tensor
    effective_snr: Tensor
    sigma: float
    n_samples: int

    @property
    def weakest(self) -> int:
        """Index of the least observable basis function."""
        return int(self.snr.argmin())

    def summary(self) -> str:
        lines = [
            f"disturbance observability  (sigma = {self.sigma:g}, N = {self.n_samples})",
            f"{'i':>3} | {'gain':>10} | {'settled':>10} | {'signature':>10} | "
            f"{'SNR':>7} | {'eff SNR':>8} |",
            "-" * 76,
        ]
        for i in range(self.snr.numel()):
            flag = "" if float(self.snr[i]) >= 1.0 else "  <- below the noise floor"
            lines.append(
                f"{i + 1:>3} | {float(self.output_gain[i]):>10.3e} | "
                f"{float(self.settled_gain[i]):>10.3e} | "
                f"{float(self.signature[i]):>10.3e} | {float(self.snr[i]):>7.2f} | "
                f"{float(self.effective_snr[i]):>8.2f} |{flag}"
            )
        weak = self.weakest
        lines.append("")
        if float(self.snr[weak]) < 1.0:
            lines.append(
                f"basis function {weak + 1} leaves a trace {float(self.snr[weak]):.2f}x the "
                "noise; it is not\nrecoverable pointwise. Reduce sigma, lower the basis "
                "frequency, lengthen the\nhorizon, or reduce q -- training cannot help."
            )
        else:
            lines.append("every basis function is observable above the noise floor.")
        return "\n".join(lines)


def linearize(
    system: TriangularSystem, x_ref: Tensor | None = None, theta: Tensor | None = None
) -> tuple[Tensor, Tensor, Tensor]:
    r"""Linearize (1) about ``x_ref``, returning :math:`(A, E_d, C)`.

    :math:`A = \partial f/\partial x`, obtained by automatic differentiation of
    the assembled vector field; :math:`E_d = [0,\dots,0,1]^\top` and
    :math:`C = [1,0,\dots,0]` as in Eq. (5).
    """
    theta = system.theta_true if theta is None else theta
    if theta is None:
        raise ValueError("linearization needs parameters; pass theta or set theta_true")
    dtype, device = system.state_bounds.dtype, system.state_bounds.device
    if x_ref is None:
        x_ref = torch.zeros(system.n, dtype=dtype, device=device)

    def field(x: Tensor) -> Tensor:
        return system.vector_field(
            x, theta, torch.zeros((), dtype=x.dtype, device=x.device)
        )

    jacobian = torch.autograd.functional.jacobian(field, x_ref.clone().requires_grad_(True))
    e_d = torch.zeros(system.n, dtype=dtype, device=device)
    e_d[-1] = 1.0
    c = torch.zeros(system.n, dtype=dtype, device=device)
    c[0] = 1.0
    return jacobian.detach(), e_d, c


def basis_observability(
    system: TriangularSystem,
    basis: DisturbanceBasis,
    coefficient_bounds: Tensor,
    noise_sigma: float,
    n_samples: int,
    substeps: int = 8,
    x_ref: Tensor | None = None,
    settle_fraction: float = 0.5,
) -> BasisObservability:
    r"""Compute the per-basis-function observability described above.

    Each basis function is used, one at a time, as the disturbance driving the
    linearized system from rest; the resulting output is compared against the
    measurement noise.
    """
    if noise_sigma <= 0:
        raise ValueError("observability is defined relative to a positive noise level")
    matrix, e_d, c = linearize(system, x_ref=x_ref)
    dtype, device = matrix.dtype, matrix.device

    grid = torch.linspace(
        basis.t_start, basis.t_end, n_samples, dtype=dtype, device=device
    )
    gains = torch.zeros(basis.q, dtype=dtype, device=device)
    settled = torch.zeros(basis.q, dtype=dtype, device=device)
    first_settled = int(settle_fraction * n_samples)

    for i in range(basis.q):
        def field(t: Tensor, x: Tensor, index: int = i) -> Tensor:
            drive = basis.evaluate(t.reshape(1))[0, index]
            return matrix @ x + e_d * drive

        x = torch.zeros(system.n, dtype=dtype, device=device)
        outputs = [torch.dot(c, x)]
        for k in range(n_samples - 1):
            h = (grid[k + 1] - grid[k]) / substeps
            t = grid[k]
            for _ in range(substeps):
                k1 = field(t, x)
                k2 = field(t + 0.5 * h, x + 0.5 * h * k1)
                k3 = field(t + 0.5 * h, x + 0.5 * h * k2)
                k4 = field(t + h, x + h * k3)
                x = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
                t = t + h
            outputs.append(torch.dot(c, x))
        response = torch.stack(outputs)
        gains[i] = torch.sqrt(torch.mean(response**2))
        settled[i] = torch.sqrt(torch.mean(response[first_settled:] ** 2))

    bounds = torch.as_tensor(coefficient_bounds, dtype=dtype, device=device)
    magnitude = 0.5 * (bounds[:, 1] - bounds[:, 0])
    signature = settled * magnitude
    snr = signature / noise_sigma
    return BasisObservability(
        output_gain=gains,
        settled_gain=settled,
        signature=signature,
        snr=snr,
        effective_snr=snr * float(n_samples) ** 0.5,
        sigma=float(noise_sigma),
        n_samples=int(n_samples),
    )
