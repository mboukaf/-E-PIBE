r"""Continuous-time state decoders :math:`\mathcal{S}_k`.

For intermediate cells, Eq. (18):

.. math::

    \mathcal{S}_k : [0,T] \times \mathbb{R}^{r_k}
        \longrightarrow \mathcal{X}_{k-1} \times \mathcal{X}_k, \qquad
    (t, \mathbf{z}^\ell_{k-1}) \longmapsto
        \begin{bmatrix} \hat x^{k,\ell}_{k-1}(t) \\ \hat x^{k,\ell}_k(t) \end{bmatrix}.

The **two** outputs are the point of the construction: the first is the cell's
own reconstruction of the *upstream* coordinate --- compared against the
upstream estimate by the consistency data term (29) --- and the second is the
new coordinate this cell estimates.  Remark 7 turns on precisely this
distinction: the local loss anchors the reconstructed coordinate
:math:`x_{k-1}`, not the free new output :math:`x_k`.

For the final cell, Eq. (33) yields a single auxiliary output
:math:`\hat x^{n+1,\ell}_n(t)`, used only to evaluate the last physics residual
and the disturbance-coefficient estimate --- "it is not a second reported
estimate of :math:`x_n`".

The decoder is evaluated at both the data grid and the collocation grid, and
its time derivative (21)/(36) is taken by automatic differentiation with
:math:`\mathbf{z}` held fixed.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn

from pibe.nets.mlp import MLP
from pibe.nets.normalization import normalize_time
from pibe.nets.reparam import make_output_map


class FourierTimeFeatures(nn.Module):
    r"""Lift normalized time into ``[t, sin(k pi t), cos(k pi t)]_{k=1..K}``.

    A plain MLP in :math:`t` has a well-known spectral bias: it fits smooth,
    low-frequency shapes long before oscillatory ones.  That matters here
    beyond fit quality.  The disturbance reaches the measured output only
    through the full chain, so recovering it amounts to differentiating
    :math:`y` :math:`n` times; an under-resolved :math:`\hat x_1` therefore
    destroys :math:`\hat d` even when its own error looks small.  Giving the
    decoder an explicit oscillatory basis removes that bottleneck.

    Every feature is :math:`C^\infty`, so Eq. (21)'s requirement that the
    decoder be twice continuously differentiable in :math:`t` still holds, and
    the harmonics are of the *horizon*, so a disturbance at :math:`\Omega` with
    an integer number of periods over :math:`[0,T]` is represented exactly.

    Amplitude scaling
    -----------------
    Harmonics are scaled by :math:`1/k`.  This matters because the physics
    residual differentiates the decoder: since
    :math:`\frac{d}{dt}\sin(k\pi t) = k\pi\cos(k\pi t)`, unit-amplitude
    features make harmonic :math:`k` contribute :math:`k` times as strongly to
    :math:`\dot{\hat x}` as to :math:`\hat x`.  With a large ``K`` the highest
    harmonics then dominate the residual and its conditioning collapses --- in
    practice training degrades sharply somewhere above ``K ~ 10``.  Scaling by
    :math:`1/k` equalizes each harmonic's influence on the *derivative*, which
    is the quantity the loss actually sees, and costs nothing in
    expressiveness: the network simply learns a correspondingly larger weight.

    Parameters
    ----------
    n_features
        Number of harmonics ``K``.  ``0`` disables the lift entirely.
    scale_by_harmonic
        Apply the :math:`1/k` amplitude scaling described above.
    """

    def __init__(self, n_features: int, scale_by_harmonic: bool = True,
        saturation_limit: float | None = None,
    ) -> None:
        super().__init__()
        if n_features < 0:
            raise ValueError(f"n_features must be non-negative, got {n_features}")
        self.n_features = int(n_features)
        self.scale_by_harmonic = bool(scale_by_harmonic)
        if self.n_features:
            # Stored as exact integers, with pi applied in the working dtype at
            # call time.  Baking pi into a buffer built at the default dtype
            # would round the harmonics to float32 and survive an upcast to
            # float64 as a ~1e-7 error in every phase.
            self.register_buffer(
                "harmonics",
                torch.arange(1, self.n_features + 1, dtype=torch.float64),
            )

    @property
    def out_dim(self) -> int:
        """Width of the lifted time input."""
        return 1 + 2 * self.n_features

    def forward(self, t_scaled: Tensor) -> Tensor:
        """``t_scaled`` in ``[-1, 1]``, shape ``(..., 1)`` -> ``(..., out_dim)``."""
        if not self.n_features:
            return t_scaled
        harmonics = self.harmonics.to(t_scaled.dtype)
        phase = t_scaled * (math.pi * harmonics)
        amplitude = (1.0 / harmonics) if self.scale_by_harmonic else 1.0
        return torch.cat(
            [t_scaled, amplitude * torch.sin(phase), amplitude * torch.cos(phase)],
            dim=-1,
        )

    def extra_repr(self) -> str:  # pragma: no cover - trivial
        return (f"n_features={self.n_features}, out_dim={self.out_dim}, "
                f"scale_by_harmonic={self.scale_by_harmonic}")


class StateDecoder(nn.Module):
    r"""The map :math:`(t, \mathbf{z}) \mapsto` constrained state outputs.

    Parameters
    ----------
    latent_dim
        :math:`r_k`.
    output_bounds
        Admissible interval per output coordinate, shape ``(out_dim, 2)``.
        For an intermediate cell this is
        ``[X_{k-1}; X_k]``; for the final cell, ``[X_n]``.
    t_start, t_end
        Horizon endpoints, used to map ``t`` into ``[-1, 1]``.
    hidden, activation
        Passed to :class:`~pibe.nets.mlp.MLP`; the activation must be ``C^2``.
    """

    def __init__(
        self,
        latent_dim: int,
        output_bounds: Tensor,
        t_start: float = 0.0,
        t_end: float = 1.0,
        hidden: Sequence[int] = (64, 64, 64),
        activation: str = "tanh",
        time_fourier_features: int = 0,
        time_feature_scaling: bool = True,
        saturation_limit: float | None = None,
    ) -> None:
        super().__init__()
        output_bounds = torch.as_tensor(output_bounds)
        if output_bounds.ndim != 2 or output_bounds.shape[1] != 2:
            raise ValueError(
                f"output_bounds must have shape (out_dim, 2), got {tuple(output_bounds.shape)}"
            )
        self.latent_dim = int(latent_dim)
        self.out_dim = int(output_bounds.shape[0])
        self.t_start = float(t_start)
        self.t_end = float(t_end)

        self.time_features = FourierTimeFeatures(
            time_fourier_features, scale_by_harmonic=time_feature_scaling
        )
        self.net = MLP(
            in_dim=self.time_features.out_dim + self.latent_dim,
            out_dim=self.out_dim,
            hidden=hidden,
            activation=activation,
        )
        self.output_map = make_output_map(
            output_bounds, self.out_dim, saturation_limit=saturation_limit
        )

    def forward(self, t: Tensor, z: Tensor) -> Tensor:
        """Evaluate the decoder on a shared time grid.

        Parameters
        ----------
        t
            Times, shape ``(B, M, 1)``.  Must be a distinct tensor element per
            ``(batch, time)`` entry when its derivative is required; see
            :func:`~pibe.core.autodiff.make_time_input`.
        z
            Latent codes, shape ``(B, r_k)``, constant along the time axis.

        Returns
        -------
        Tensor
            Shape ``(B, M, out_dim)``, inside the admissible box.
        """
        if t.ndim != 3 or t.shape[-1] != 1:
            raise ValueError(f"t must have shape (B, M, 1), got {tuple(t.shape)}")
        if z.ndim != 2 or z.shape[-1] != self.latent_dim:
            raise ValueError(
                f"z must have shape (B, {self.latent_dim}), got {tuple(z.shape)}"
            )
        if t.shape[0] != z.shape[0]:
            raise ValueError(
                f"batch mismatch between t ({t.shape[0]}) and z ({z.shape[0]})"
            )

        t_scaled = self.time_features(normalize_time(t, self.t_start, self.t_end))
        z_expanded = z.unsqueeze(1).expand(-1, t.shape[1], -1)
        features = torch.cat([t_scaled, z_expanded], dim=-1)
        return self.output_map(self.net(features))

    def extra_repr(self) -> str:  # pragma: no cover - trivial
        return (
            f"r_k={self.latent_dim}, out_dim={self.out_dim}, "
            f"horizon=[{self.t_start:g}, {self.t_end:g}]"
        )
