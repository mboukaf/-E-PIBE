r"""Trajectory encoders :math:`\mathcal{E}_k`.

Eq. (17) maps the whole trajectory-level input to a latent code,

.. math:: \mathbf{z}^\ell_{k-1} = \mathcal{E}_k(\mathbf{U}^\ell_{k-1};
          \Theta_{E_k}) \in \mathbb{R}^{r_k},

where :math:`\mathbf{U}^\ell_{k-1}` of Eq. (15) stacks the sampled measurement
vector, every upstream sampled state estimate and every upstream parameter
estimate.  Because the code is produced once per trajectory and then held
fixed while the decoder sweeps :math:`t`, the resulting estimator is
*finite-horizon and not causal in time*: the paper notes that a causal variant
would replace :math:`\mathbf{U}^\ell_{k-1}` by a past-only sliding window.

The fixed normalization :math:`\mathsf{D}_{k-1}` of Eq. (61) is applied first,
as an input preprocessing layer.
"""

from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor, nn

from pibe.nets.mlp import MLP
from pibe.nets.normalization import TrajectoryInputNormalizer, trajectory_input_dim


class TrajectoryEncoder(nn.Module):
    r"""The map :math:`\mathcal{E}_k : \mathbb{R}^{d_{k-1}} \to \mathbb{R}^{r_k}`.

    Parameters
    ----------
    cell_index
        ``k``, with ``2 <= k <= n + 1``.
    n_samples
        ``N``.
    latent_dim
        :math:`r_k`.
    state_scales, theta_scales
        Reference scales for :math:`\mathsf{D}_{k-1}`; see
        :class:`~pibe.nets.normalization.TrajectoryInputNormalizer`.
    hidden, activation
        Passed to :class:`~pibe.nets.mlp.MLP`.
    """

    def __init__(
        self,
        cell_index: int,
        n_samples: int,
        latent_dim: int,
        state_scales: Tensor,
        theta_scales: Tensor,
        hidden: Sequence[int] = (128, 128),
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        self.cell_index = int(cell_index)
        self.latent_dim = int(latent_dim)
        self.input_dim = trajectory_input_dim(cell_index, n_samples)

        self.normalizer = TrajectoryInputNormalizer(
            cell_index=cell_index,
            n_samples=n_samples,
            state_scales=state_scales,
            theta_scales=theta_scales,
        )
        self.net = MLP(
            in_dim=self.input_dim,
            out_dim=self.latent_dim,
            hidden=hidden,
            activation=activation,
        )

    def forward(self, u: Tensor) -> Tensor:
        """Encode ``u`` of shape ``(B, d_{k-1})`` into ``(B, r_k)``."""
        if u.ndim != 2:
            raise ValueError(
                f"trajectory input must be 2-D (B, d), got shape {tuple(u.shape)}"
            )
        return self.net(self.normalizer(u))

    def extra_repr(self) -> str:  # pragma: no cover - trivial
        return f"cell_index={self.cell_index}, in_dim={self.input_dim}, r_k={self.latent_dim}"
