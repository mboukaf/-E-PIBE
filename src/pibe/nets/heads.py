r"""Time-independent parameter and coefficient heads.

Intermediate cells carry the parameter head of Eq. (19),

.. math:: \mathcal{Q}_k : \mathbb{R}^{r_k} \to \Theta_{k-1}, \qquad
          \mathbf{z}^\ell_{k-1} \mapsto \hat\theta^{k,\ell}_{k-1},

and the final cell the coefficient head of Eq. (34),

.. math:: \mathcal{Q}_{n+1} : \mathbb{R}^{r_{n+1}} \to \mathcal{A}, \qquad
          \mathbf{z}^\ell_n \mapsto \hat a^\ell.

Both take the latent code only.  "Since the parameter head does not receive
:math:`t`, :math:`\hat\theta^{k,\ell}_{k-1}` is constant over :math:`[0,T]` by
construction" --- the unknown parameters of (1) are constant, and the
architecture enforces that exactly rather than penalizing drift.  The estimate
still varies across trajectories :math:`\ell`.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from pibe.nets.mlp import MLP
from pibe.nets.reparam import make_output_map


class ParameterHead(nn.Module):
    r"""The map :math:`\mathcal{Q}_k`, constrained to a box.

    Parameters
    ----------
    latent_dim
        :math:`r_k`.
    output_bounds
        Admissible box, shape ``(out_dim, 2)``: a single row
        :math:`\Theta_{k-1}` for an intermediate cell, ``q`` rows
        :math:`\mathcal{A}` for the final one.
    hidden, activation
        Passed to :class:`~pibe.nets.mlp.MLP`.
    """

    def __init__(
        self,
        latent_dim: int,
        output_bounds: Tensor,
        hidden: Sequence[int] = (64, 64),
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        output_bounds = torch.as_tensor(output_bounds)
        if output_bounds.ndim != 2 or output_bounds.shape[1] != 2:
            raise ValueError(
                f"output_bounds must have shape (out_dim, 2), got {tuple(output_bounds.shape)}"
            )
        self.latent_dim = int(latent_dim)
        self.out_dim = int(output_bounds.shape[0])

        self.net = MLP(
            in_dim=self.latent_dim,
            out_dim=self.out_dim,
            hidden=hidden,
            activation=activation,
        )
        self.output_map = make_output_map(output_bounds, self.out_dim)

    def forward(self, z: Tensor) -> Tensor:
        """Map latent codes ``(B, r_k)`` to constrained outputs ``(B, out_dim)``."""
        if z.ndim != 2 or z.shape[-1] != self.latent_dim:
            raise ValueError(
                f"z must have shape (B, {self.latent_dim}), got {tuple(z.shape)}"
            )
        return self.output_map(self.net(z))

    def extra_repr(self) -> str:  # pragma: no cover - trivial
        return f"r_k={self.latent_dim}, out_dim={self.out_dim}"
