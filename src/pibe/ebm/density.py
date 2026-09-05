r"""The scalar residual density of Eq. (49) and its negative log-likelihood.

Under Assumption 3 the :math:`k`-th cell models its consistency residual with

.. math::

    p_{k,\zeta_k}(\xi) = \frac{\exp[-\beta \tilde E_{\zeta_k}(\xi)]}
                              {Z_k(\tilde E_{\zeta_k})}\,
                         \mathbb{1}_{\mathcal{R}_k}(\xi),
    \qquad
    Z_k = \int_{\mathcal{R}_k} \exp[-\beta \tilde E_{\zeta_k}(u)]\,du,
    \qquad \text{(49)}

and is fitted by the empirical negative log-likelihood

.. math::

    \mathcal{L}^{ebm,k}_{Data}(\{\varepsilon_k\})
      = \log Z_k(\tilde E_{\zeta_k})
      + \frac{\beta}{N}\sum_{i=1}^N \tilde E_{\zeta_k}(\varepsilon_k(t_i)).
    \qquad \text{(53)}

This replaces the quadratic data term of PIBE, and the replacement is the whole
point of EPIBE.  A squared residual *is* the negative log-likelihood of a
zero-mean Gaussian up to constants, so minimizing it asserts
:math:`\mu_\omega = 0` as a modelling choice rather than inferring it.  Eq. (53)
asserts nothing about the location: the learned density is free to sit off zero,
and Eq. (54) reads that location back out.

What this loss does and does not do
-----------------------------------
It is *not* minimized by driving residuals to zero.  It is minimized by making
the residuals look like samples from the learned density --- which is why
Proposition 1's coercivity role passes entirely to the physics term, and why
Remark 2 warns that "a small EBM negative log-likelihood alone does not prove
that the PINN output is the physical signal: the EBM can otherwise absorb a
systematic PINN error as part of the residual law."  The physics residual is
what pins the signal; the EBM only says what the leftover looks like.

Two properties of (53) are worth keeping in mind when reading the training logs.
It is a log-likelihood, so it is **signed** --- a sharp, well-fitted density
gives a large negative value, and "loss went negative" is success, not a bug.
And because it is bounded below only through Eq. (50), the constants of
Assumption 3 are what stop it running to :math:`-\infty`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from pibe.ebm.energy import EnergyNetwork
from pibe.ebm.quadrature import CompositeGaussLegendre


@dataclass(frozen=True)
class DensityMoments:
    r"""Descriptive moments of a learned residual density, Eq. (54).

    Attributes
    ----------
    mean
        :math:`\hat\mu_\omega := \int_{\mathcal{R}_k} \xi\,p_{k,\Theta}(\xi)\,d\xi`.
        For the first cell this is the estimate of the measurement-noise mean.
        The paper is careful about its status and so is this code: it "is not an
        independently optimized offset and equals the true noise mean only when
        the physical signal is correctly identified and the learned first-cell
        residual density is statistically consistent" --- it is a *moment of a
        fitted density*, reported, never optimized against a target.
    std
        Square root of the second central moment, over the same rule.
    log_partition
        :math:`\log Z_k`, kept because it is the quantity whose finiteness
        Eq. (50) guarantees and a useful health check on the quadrature.
    """

    mean: float
    std: float
    log_partition: float


class ScalarEBM(nn.Module):
    r"""One cell's energy-based residual model: energy, density, likelihood.

    Parameters
    ----------
    radius
        :math:`\bar\rho_k` from Assumption 3.
    beta
        Inverse temperature :math:`\beta > 0`.  It multiplies the energy
        everywhere, so :math:`\beta` and ``energy_scale`` control the density's
        dynamic range only through their product; :math:`\beta` is kept as a
        separate knob because Algorithm 2 lists it as an input.
    hidden, activation, energy_scale, energy_bound, spectral_norm_layers
        Passed to :class:`~pibe.ebm.energy.EnergyNetwork`.
    panels, nodes_per_panel
        The fixed quadrature rule of Remark 4.
    barrier_scale
        Width of the out-of-support barrier, as a fraction of the radius; see
        :meth:`energy_with_barrier`.  Zero restores plain clamping, which is
        available only so the failure it causes can be demonstrated.
    """

    def __init__(
        self,
        radius: float,
        beta: float = 1.0,
        hidden: tuple[int, ...] = (64, 64),
        activation: str = "tanh",
        energy_scale: float = 1.0,
        energy_bound: float | None = None,
        spectral_norm_layers: bool = False,
        panels: int = 64,
        nodes_per_panel: int = 16,
        barrier_scale: float = 0.1,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        if beta <= 0:
            raise ValueError(f"beta must be positive, got {beta}")
        self.beta = float(beta)
        self.radius = float(radius)
        self.barrier_scale = float(barrier_scale)
        self.energy = EnergyNetwork(
            radius=radius,
            hidden=hidden,
            activation=activation,
            energy_scale=energy_scale,
            bound=energy_bound,
            spectral_norm_layers=spectral_norm_layers,
        )
        self.quadrature = CompositeGaussLegendre(
            radius=radius, panels=panels, nodes_per_panel=nodes_per_panel, dtype=dtype
        )
        # The quadrature nodes are built in ``dtype`` while the MLP defaults to
        # float32; align the whole module so the two never meet mismatched.
        self.to(dtype)

    # ------------------------------------------------------------------
    # the density, Eq. (49)
    # ------------------------------------------------------------------

    def log_partition(self) -> Tensor:
        r""":math:`\log Z_k`, Eq. (59), by the fixed rule.

        Differentiating this expression yields the quadrature estimate of
        :math:`\nabla_\zeta \log Z_k = -\beta\,\mathbb{E}_{p}[\nabla_\zeta
        \tilde E]` that Remark 4 requires, so no separate gradient routine is
        needed.
        """
        nodes = self.quadrature.nodes
        return self.quadrature.log_integral(-self.beta * self.energy(nodes))

    def log_density(self, xi: Tensor) -> Tensor:
        r""":math:`\log p_{k,\zeta_k}(\xi)`, ``-inf`` outside the support."""
        inside = xi.abs() <= self.radius
        value = -self.beta * self.energy(xi.clamp(-self.radius, self.radius))
        return torch.where(
            inside, value - self.log_partition(), torch.full_like(xi, -torch.inf)
        )

    # ------------------------------------------------------------------
    # the loss, Eq. (53)
    # ------------------------------------------------------------------

    def energy_with_barrier(self, residual: Tensor) -> Tensor:
        r"""The energy on :math:`\mathcal{R}_k`, continued by a barrier outside it.

        The paper's rule is that "every residual entering (53) is constrained to
        :math:`\mathcal{R}_k`; otherwise its likelihood under (49) is zero" ---
        that is, the energy is :math:`+\infty` there.  An infinite loss cannot be
        optimized, so the outside has to be approximated, and *how* it is
        approximated turns out to matter a great deal.

        Clamping the residual onto the support is the obvious choice and is
        wrong.  It makes the energy **constant** outside, so a residual that
        leaves the support contributes no gradient to the PINN at all; the data
        term silently switches off, the physics term alone does not identify the
        state (Proposition 1's unique-zero hypothesis fails cell by cell), and
        the estimate drifts further out.  That is a positive feedback loop: the
        further out it goes, the less reason it has to come back.

        A steep quadratic continuation instead grows without bound outside, so
        the gradient always points back into the support --- the differentiable
        reading of "likelihood zero".  Inside the support this is exactly the
        energy of Eq. (49) and nothing is changed; the barrier is a training
        device for out-of-class residuals, not part of the density, and it
        contributes nothing to :math:`Z_k`, which is still integrated over
        :math:`\mathcal{R}_k` alone.
        """
        inside = residual.clamp(-self.radius, self.radius)
        energy = self.energy(inside)
        if self.barrier_scale <= 0:
            return energy
        overshoot = (residual - inside).abs()
        width = self.barrier_scale * self.radius
        return energy + self.energy.energy_bound * (overshoot / width) ** 2

    def negative_log_likelihood(self, residual: Tensor) -> Tensor:
        r"""Eq. (53), averaged over every sample and trajectory in the batch.

        Residuals outside :math:`\mathcal{R}_k` are handled by
        :meth:`energy_with_barrier`.  They should not occur --- Assumption 3
        fixes the radius so that they cannot --- so
        :meth:`out_of_support_fraction` is monitored throughout training and a
        nonzero value means the radius was set by hand and set too small.
        """
        return self.log_partition() + self.beta * self.energy_with_barrier(residual).mean()

    @torch.no_grad()
    def out_of_support_fraction(self, residual: Tensor) -> float:
        r"""Fraction of residuals falling outside :math:`\mathcal{R}_k`.

        Should be exactly zero: Assumption 3 fixes the radius "before EBM
        activation using the compact decoder-output and parameter ranges
        together with the known noise-support bound", so a violation means the
        radius was set by hand and set too small.
        """
        if residual.numel() == 0:
            return 0.0
        return float((residual.abs() > self.radius).to(residual.dtype).mean())

    # ------------------------------------------------------------------
    # descriptive moments, Eq. (54)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def moments(self) -> DensityMoments:
        r"""Mean and spread of the learned density, Eq. (54)."""
        nodes = self.quadrature.nodes
        log_partition = self.log_partition()
        density = torch.exp(-self.beta * self.energy(nodes) - log_partition)
        mass = self.quadrature.integrate(density)
        mean = self.quadrature.integrate(nodes * density) / mass
        second = self.quadrature.integrate((nodes - mean) ** 2 * density) / mass
        return DensityMoments(
            mean=float(mean),
            std=float(second.clamp_min(0.0).sqrt()),
            log_partition=float(log_partition),
        )

    @torch.no_grad()
    def normalization_error(self) -> float:
        r"""``|1 - \int p|``, a direct check that the quadrature resolves the density.

        The rule is fixed while the density is learned, so a peak narrower than
        the node spacing would be integrated badly and every downstream quantity
        --- :math:`\log Z_k`, the likelihood, the mean --- would be wrong
        together.  This catches that in one number.
        """
        nodes = self.quadrature.nodes
        density = torch.exp(-self.beta * self.energy(nodes) - self.log_partition())
        return float((self.quadrature.integrate(density) - 1.0).abs())

    @torch.no_grad()
    def project(self, weight_bound: float | None) -> None:
        """Projected parameter update onto :math:`\\mathcal{K}_k` (Algorithm 2)."""
        self.energy.project(weight_bound)

    def extra_repr(self) -> str:  # pragma: no cover - trivial
        return f"beta={self.beta:g}, radius={self.radius:g}"
