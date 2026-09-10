r"""The constrained scalar energy of Assumption 3.

Assumption 3 asks for a *well-posed* EBM class, not merely a network.  For each
cell :math:`k` it fixes a compact residual support :math:`\mathcal{R}_k =
[-\bar\rho_k, \bar\rho_k]`, defines the gauge-normalized energy

.. math:: \tilde E_{\zeta_k}(\xi) := E_{\zeta_k}(\xi) - E_{\zeta_k}(0),
   \qquad \text{(47)}

and requires a compact parameter set :math:`\mathcal{K}_k` on which

.. math:: \|\tilde E_{\zeta_k}\|_{L^\infty(\mathcal{R}_k)} \le B_{E,k},
          \qquad \mathrm{Lip}_{\mathcal{R}_k}(\tilde E_{\zeta_k}) \le L_{E,k},
   \qquad \text{(48)}

to be implemented "by bounded output parameterizations, spectral normalization,
and projected parameter updates".  All three appear below.

Why the constraints are load-bearing
------------------------------------
They are not regularization in the usual sense.  Eq. (50) turns the energy bound
into two-sided bounds on :math:`Z_k` and on the density itself, and it is that
lower bound on the density which "prevents arbitrarily narrow empirical-likelihood
spikes".  Without it the joint PINN--EBM problem has an obvious cheat: drive the
learned density towards a delta at whatever residual the PINN currently
produces, sending the likelihood to infinity while learning nothing.  The bound
makes the EBM subproblem bounded below, which is what lets Assumption 3 claim a
minimum is attained.

The expressiveness trade-off
----------------------------
The bounds also cap what the class can represent, and the cap is quantitative.
Since :math:`\tilde E` varies by at most :math:`\mathrm{Lip} \cdot 2\bar\rho`
across the support, the density's dynamic range obeys

.. math:: \frac{\max p}{\min p} \le e^{2\beta B_{E,k}},
          \qquad \text{and locally } \le e^{\beta L_{E,k} \delta}
          \text{ over a width } \delta .

A residual density concentrated on a width :math:`\delta` of a support of width
:math:`2\bar\rho` therefore needs roughly :math:`2\beta B_{E,k} \gtrsim
\log(2\bar\rho/\delta)` and a Lipschitz constant of order
:math:`\log(2\bar\rho/\delta)/\delta`.  Two consequences worth stating plainly,
because they decide whether a run works:

* a conservative :math:`\bar\rho_k` is not free --- the looser the support, the
  larger the constants the same density needs, so the support radius and the
  energy bound have to be chosen together;
* spectral normalization pinned at :math:`1` is far too tight here.  It caps
  :math:`L_{E,k}` at :math:`A/\bar\rho_k`, which forbids any density feature
  narrower than roughly :math:`\bar\rho_k/(\beta A)`; with a support chosen
  conservatively that is wider than the residual law itself, and the fitted
  density saturates against the constraint rather than converging.  What
  Assumption 3 actually requires is a *compact* :math:`\mathcal{K}_k` and a
  *finite* :math:`L_{E,k}`, which the weight-box projection of
  :meth:`EnergyNetwork.project` supplies on its own.  Spectral normalization
  therefore stays available but off by default, and
  :attr:`EnergyNetwork.lipschitz_constant` reports whichever constant is in
  force.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn.utils.parametrizations import spectral_norm

from pibe.nets.mlp import MLP


class EnergyNetwork(nn.Module):
    r"""A scalar energy :math:`\tilde E_\zeta` on :math:`[-\bar\rho, \bar\rho]`.

    The forward pass composes four steps, each of which implements one clause of
    Assumption 3:

    1. **Normalize** the input, :math:`u = \xi / \bar\rho \in [-1, 1]`, so the
       network's own Lipschitz constant is measured in support-relative units
       and the architecture does not have to be retuned per problem scale.
    2. **Evaluate** a spectrally normalized MLP :math:`g`, whose Lipschitz
       constant in :math:`u` is at most 1 (every activation in
       :mod:`pibe.nets.mlp` is 1-Lipschitz and every weight has unit spectral
       norm).
    3. **Gauge-normalize**, Eq. (47): subtract :math:`g(0)`.  This fixes the
       additive freedom in the energy, which the density (49) quotients out
       anyway, so the parameterization is identifiable without changing what it
       represents.
    4. **Bound**, Eq. (48): pass through :math:`B_E \tanh(\cdot / B_E)`, which
       is smooth, 1-Lipschitz, leaves the gauge intact (:math:`\tanh 0 = 0`) and
       enforces :math:`\|\tilde E\|_\infty \le B_E` by construction rather than
       by penalty.

    Parameters
    ----------
    radius
        :math:`\bar\rho`, the support half-width.
    hidden
        Hidden widths of the MLP.
    activation
        Name of a ``C^2`` activation from :mod:`pibe.nets.mlp`.
    energy_scale
        A multiplier :math:`A` on the normalized network, applied before the
        bound.  Under spectral normalization it *is* the Lipschitz budget,
        :math:`L_E = A/\bar\rho`; otherwise it is a plain gain and 1 is the
        natural value.
    bound
        :math:`B_E`, the guaranteed sup-norm of :math:`\tilde E`.  This is the
        constant that matters for expressiveness: the density's dynamic range is
        :math:`e^{2\beta B_E}`, which must exceed :math:`2\bar\rho/\delta`
        for a residual law of width :math:`\delta`.
    symmetric
        Constrain the energy to be even in :math:`\xi`, so the density is
        symmetric and :math:`\hat\mu_\omega = 0` exactly.  Removes the
        translation degeneracy at the cost of Eq. (54); see :meth:`forward`.
    spectral_norm_layers
        Whether to spectrally normalize the linear layers.  Off by default, and
        the default is the point: pinning every spectral norm to 1 caps the
        Lipschitz constant at :math:`A/\bar\rho`, which forbids any density
        feature narrower than about :math:`\bar\rho/(\beta A)` --- for a
        conservatively chosen support that is far wider than the actual residual
        law, and the EBM saturates against the constraint instead of fitting.
        Assumption 3 asks only that :math:`\mathcal{K}_k` be compact and
        :math:`L_{E,k}` *finite*, which the weight-box projection of
        :meth:`project` supplies on its own; :attr:`lipschitz_constant` then
        reports the value in force rather than imposing one.
    """

    def __init__(
        self,
        radius: float,
        hidden: tuple[int, ...] = (64, 64),
        activation: str = "tanh",
        energy_scale: float = 1.0,
        bound: float | None = None,
        spectral_norm_layers: bool = False,
        symmetric: bool = False,
    ) -> None:
        super().__init__()
        if radius <= 0:
            raise ValueError(f"the support radius must be positive, got {radius}")
        if energy_scale <= 0:
            raise ValueError(f"energy_scale must be positive, got {energy_scale}")
        self.radius = float(radius)
        self.energy_scale = float(energy_scale)
        self.bound = float(12.0 if bound is None else bound)
        if self.bound <= 0:
            raise ValueError(f"the energy bound must be positive, got {self.bound}")
        self.spectral_norm_layers = bool(spectral_norm_layers)
        self.symmetric = bool(symmetric)

        self.mlp = MLP(
            in_dim=1, out_dim=1, hidden=hidden, activation=activation, bias=True
        )
        if self.spectral_norm_layers:
            for index, module in enumerate(self.mlp.net):
                if isinstance(module, nn.Linear):
                    self.mlp.net[index] = spectral_norm(module)

    # ------------------------------------------------------------------
    # Assumption 3 constants
    # ------------------------------------------------------------------

    @property
    def energy_bound(self) -> float:
        r""":math:`B_{E,k}`, the guaranteed bound on :math:`\|\tilde E\|_\infty`."""
        return self.bound

    @property
    def lipschitz_constant(self) -> float:
        r""":math:`L_{E,k}`, the Lipschitz constant of :math:`\tilde E` on the support.

        With spectral normalization this is the *guaranteed* value
        :math:`A/\bar\rho`: the normalized network is 1-Lipschitz in
        :math:`u = \xi/\bar\rho`, the scale contributes :math:`A`, and the outer
        ``tanh`` is 1-Lipschitz.  Without it, the MLP's own spectral product is
        used instead, which is an upper bound measured from the current weights
        rather than a constraint.
        """
        if self.spectral_norm_layers:
            return self.energy_scale / self.radius
        return self.energy_scale * self.mlp.lipschitz_upper_bound() / self.radius

    @property
    def log_density_range(self) -> float:
        r"""How much :math:`\log p` may vary across the support, :math:`2 B_E`.

        Multiplied by :math:`\beta`, this is the log of the largest density
        ratio the class can express --- the quantity to compare against
        :math:`\log(2\bar\rho/\delta)` when choosing constants for a residual of
        width :math:`\delta`.
        """
        return 2.0 * self.bound

    # ------------------------------------------------------------------
    # evaluation
    # ------------------------------------------------------------------

    def forward(self, xi: Tensor) -> Tensor:
        r"""Evaluate :math:`\tilde E_\zeta(\xi)`, shape preserved.

        Parameters
        ----------
        xi
            Residual values of any shape.  Points outside
            :math:`[-\bar\rho, \bar\rho]` are *not* clamped here --- the caller
            decides what an out-of-support residual means (see
            :meth:`~pibe.ebm.density.ScalarEBM.negative_log_likelihood`).
        """
        shape = xi.shape
        u = (xi / self.radius).reshape(-1, 1)
        if self.symmetric:
            # Average the network with its reflection, making the energy an even
            # function of xi.  The density (49) is then symmetric and its mean is
            # exactly zero -- which removes the translation degeneracy that has
            # defeated every unconstrained run: the pair (state estimate,
            # density) can no longer slide together, because the density cannot
            # move.  All freedom over the *shape* is retained, so the class still
            # spans peaked, flat and heavy-tailed laws.
            #
            # No generality is lost where it is used: Section 2 and Proposition 1
            # already assume symmetric zero-mean measurement noise.  It does
            # forfeit Eq. (54) -- a symmetric density cannot report a nonzero
            # mu_omega -- so this is for the symmetric-noise case only, and the
            # biased-sensor problem needs the unconstrained class (or the
            # simulation-based estimate of scripts/offset_by_shooting.py).
            raw = 0.5 * (self.mlp(u) + self.mlp(-u)).reshape(shape)
        else:
            raw = self.mlp(u).reshape(shape)

        # Eq. (47): the gauge is fixed by the network's own value at zero, so it
        # moves with the parameters and stays exact rather than being tracked.
        zero = torch.zeros(1, 1, dtype=xi.dtype, device=xi.device)
        offset = self.mlp(zero).reshape(())

        centred = self.energy_scale * (raw - offset)
        # Eq. (48): bounded output parameterization.  tanh(0) = 0 keeps (47).
        return self.bound * torch.tanh(centred / self.bound)

    @torch.no_grad()
    def project(self, weight_bound: float | None) -> None:
        r"""Projected parameter update: clamp weights into a box (Algorithm 2).

        Lines 17 and 23 of Algorithm 2 constrain every EBM update to
        :math:`\mathcal{K}_k`.  The bounded output and the spectral norm already
        pin the two constants of (48); this makes the parameter set literally
        compact, which is what the existence argument under Eq. (50) uses.  It
        is a no-op when ``weight_bound`` is ``None``.
        """
        if weight_bound is None:
            return
        for parameter in self.parameters():
            parameter.clamp_(-weight_bound, weight_bound)

    def extra_repr(self) -> str:  # pragma: no cover - trivial
        return (
            f"radius={self.radius:g}, energy_scale={self.energy_scale:g}, "
            f"B_E={self.energy_bound:g}, L_E={self.lipschitz_constant:.3g}, "
            f"spectral_norm={self.spectral_norm_layers}"
        )
