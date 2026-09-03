r"""A single PIBE estimation cell.

The complete trajectory-conditioned PINN of Eq. (20) is

.. math::

    \Phi^k : [0,T] \times \mathcal{U}_{k-1}
        \longrightarrow (\mathcal{X}_{k-1} \times \mathcal{X}_k) \times \Theta_{k-1},
    \quad (t, \mathbf{U}^\ell_{k-1}) \longmapsto
        \bigl(\mathcal{S}_k(t, \mathbf{z}^\ell_{k-1}),\
              \mathcal{Q}_k(\mathbf{z}^\ell_{k-1})\bigr),

with parameter block :math:`\Theta^k_{PINN} = (\Theta_{E_k}, \Theta_{S_k},
\Theta_{Q_k})`, and the final cell is Eq. (35).

Naming
------
Every cell reconstructs the coordinate *below* it and, except for the final
cell, estimates one new coordinate:

======================  ==================================  =========================
cell                    ``x_prev``                          ``x_new``
======================  ==================================  =========================
``k = 2``               :math:`\hat x^2_1` (vs. :math:`y`)  :math:`\hat x^2_2`
``k = 3..n``            :math:`\hat x^k_{k-1}`              :math:`\hat x^k_k`
``k = n+1`` (final)     :math:`\hat x^{n+1}_n` (auxiliary)  --- (none)
======================  ==================================  =========================

This is uniform: cell ``k`` always compares its ``x_prev`` against the target
supplied from upstream (the measurement for ``k = 2``, otherwise cell
``k-1``'s ``x_new``), and the time derivative appearing in every physics
residual --- Eqs. (21) and (36) alike --- is always that of ``x_prev``.

By the convention of Section 3.1, the *reported* estimates are
:math:`\hat x_k := \hat x^k_k` and :math:`\hat\theta_{k-1} :=
\hat\theta^k_{k-1}`.  The final cell's :math:`\hat x^{n+1}_n` is auxiliary:
"it is not a second reported estimate of :math:`x_n`".
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from pibe.config import ArchitectureConfig
from pibe.core.autodiff import make_time_input, time_derivative
from pibe.nets.decoder import StateDecoder
from pibe.nets.encoder import TrajectoryEncoder
from pibe.nets.heads import ParameterHead


@dataclass(frozen=True)
class CellOutput:
    r"""Everything cell ``k`` produces for one minibatch of trajectories.

    Attributes
    ----------
    latent
        :math:`\mathbf{z}^\ell_{k-1}`, shape ``(B, r_k)``.
    x_prev_data, x_prev_coll
        The reconstructed upstream coordinate on the data and collocation
        grids, shapes ``(B, N)`` and ``(B, N_r)``.
    x_prev_dot_coll
        :math:`\partial_t` of ``x_prev`` on the collocation grid, shape
        ``(B, N_r)``; ``None`` when the derivative was not requested.
    x_new_data, x_new_coll
        The newly estimated coordinate, shapes ``(B, N)`` and ``(B, N_r)``;
        ``None`` for the final cell.
    head
        :math:`\hat\theta^k_{k-1}` of shape ``(B, 1)``, or :math:`\hat a^\ell`
        of shape ``(B, q)`` for the final cell.  Constant in ``t`` by
        construction.
    """

    latent: Tensor
    x_prev_data: Tensor
    x_prev_coll: Tensor
    x_prev_dot_coll: Tensor | None
    x_new_data: Tensor | None
    x_new_coll: Tensor | None
    head: Tensor

    @property
    def is_final(self) -> bool:
        return self.x_new_data is None

    def detached(self) -> CellOutput:
        """A copy with every tensor detached from the autograd graph."""

        def _detach(value: Tensor | None) -> Tensor | None:
            return None if value is None else value.detach()

        return CellOutput(
            latent=self.latent.detach(),
            x_prev_data=self.x_prev_data.detach(),
            x_prev_coll=self.x_prev_coll.detach(),
            x_prev_dot_coll=_detach(self.x_prev_dot_coll),
            x_new_data=_detach(self.x_new_data),
            x_new_coll=_detach(self.x_new_coll),
            head=self.head.detach(),
        )


class EstimatorCell(nn.Module):
    r"""One cell :math:`\Phi^k` of the bank.

    Parameters
    ----------
    cell_index
        ``k``, with ``2 <= k <= n + 1``.  ``k = n + 1`` builds the final cell.
    state_dim
        ``n``.
    n_samples
        ``N``; fixes the encoder input width :math:`d_{k-1}`.
    state_bounds
        The admissible sets :math:`\mathcal{X}_j`, shape ``(n, 2)``.
    theta_bounds
        The admissible sets :math:`\Theta_j`, shape ``(n - 1, 2)``.
    state_scales, theta_scales
        Reference scales for the input normalization (61).
    coefficient_bounds
        The box :math:`\mathcal{A}`, shape ``(q, 2)``.  Required for, and used
        only by, the final cell.
    t_start, t_end
        Horizon endpoints.
    architecture
        Network widths and activation.
    """

    def __init__(
        self,
        cell_index: int,
        state_dim: int,
        n_samples: int,
        state_bounds: Tensor,
        theta_bounds: Tensor,
        state_scales: Tensor,
        theta_scales: Tensor,
        coefficient_bounds: Tensor | None = None,
        t_start: float = 0.0,
        t_end: float = 1.0,
        architecture: ArchitectureConfig | None = None,
    ) -> None:
        super().__init__()
        if not 2 <= cell_index <= state_dim + 1:
            raise ValueError(
                f"cell index must satisfy 2 <= k <= n + 1 = {state_dim + 1}, "
                f"got {cell_index}"
            )
        architecture = architecture or ArchitectureConfig()
        self.cell_index = int(cell_index)
        self.state_dim = int(state_dim)
        self.is_final = self.cell_index == self.state_dim + 1

        # Decoder outputs: coordinates (k-1, k) for an intermediate cell,
        # coordinate n alone for the final one.  Rows are 0-based.
        if self.is_final:
            decoder_bounds = state_bounds[[self.state_dim - 1]]
        else:
            decoder_bounds = state_bounds[[self.cell_index - 2, self.cell_index - 1]]

        # Head: theta_{k-1} for an intermediate cell, the coefficient vector a
        # for the final one.
        if self.is_final:
            if coefficient_bounds is None:
                raise ValueError("the final cell requires coefficient_bounds (the set A)")
            head_bounds = coefficient_bounds
        else:
            head_bounds = theta_bounds[[self.cell_index - 2]]

        self.encoder = TrajectoryEncoder(
            cell_index=self.cell_index,
            n_samples=n_samples,
            latent_dim=architecture.latent_dim,
            state_scales=state_scales,
            theta_scales=theta_scales,
            hidden=architecture.encoder_hidden,
            activation=architecture.activation,
        )
        self.decoder = StateDecoder(
            latent_dim=architecture.latent_dim,
            output_bounds=decoder_bounds,
            t_start=t_start,
            t_end=t_end,
            hidden=architecture.decoder_hidden,
            activation=architecture.activation,
            time_fourier_features=architecture.time_fourier_features,
            time_feature_scaling=architecture.time_feature_scaling,
        )
        self.head = ParameterHead(
            latent_dim=architecture.latent_dim,
            output_bounds=head_bounds,
            hidden=architecture.head_hidden,
            activation=architecture.activation,
        )

    @property
    def input_dim(self) -> int:
        r""":math:`d_{k-1} = N(k-1) + (k-2)`."""
        return self.encoder.input_dim

    @property
    def head_dim(self) -> int:
        """1 for an intermediate cell, ``q`` for the final one."""
        return self.head.out_dim

    def forward(
        self,
        u: Tensor,
        t_data: Tensor,
        t_coll: Tensor,
        need_derivative: bool = True,
        create_graph: bool = True,
    ) -> CellOutput:
        r"""Evaluate the cell on both grids.

        Parameters
        ----------
        u
            :math:`\mathbf{U}_{k-1}`, shape ``(B, d_{k-1})``.
        t_data, t_coll
            Shared data and collocation grids, shapes ``(N,)`` and ``(N_r,)``.
        need_derivative
            Compute :math:`\partial_t` of ``x_prev`` on the collocation grid.
            Only the cell whose local loss is being evaluated needs it; frozen
            upstream cells contribute state values to the residual arguments
            but no derivative, and skipping it there avoids building an unused
            graph.
        create_graph
            Retain the derivative's graph so it is differentiable with respect
            to the parameters.  Required whenever the physics loss is
            backpropagated.
        """
        batch_size = u.shape[0]
        latent = self.encoder(u)
        head = self.head(latent)

        # The data grid is never differentiated: the data terms (23)/(29)/(38)
        # compare values only.
        t_data_input = make_time_input(t_data, batch_size, requires_grad=False)
        out_data = self.decoder(t_data_input, latent)

        t_coll_input = make_time_input(
            t_coll, batch_size, requires_grad=need_derivative
        )
        out_coll = self.decoder(t_coll_input, latent)

        x_prev_data = out_data[..., 0]
        x_prev_coll = out_coll[..., 0]
        if self.is_final:
            x_new_data = None
            x_new_coll = None
        else:
            x_new_data = out_data[..., 1]
            x_new_coll = out_coll[..., 1]

        x_prev_dot_coll = (
            time_derivative(x_prev_coll, t_coll_input, create_graph=create_graph)
            if need_derivative
            else None
        )

        return CellOutput(
            latent=latent,
            x_prev_data=x_prev_data,
            x_prev_coll=x_prev_coll,
            x_prev_dot_coll=x_prev_dot_coll,
            x_new_data=x_new_data,
            x_new_coll=x_new_coll,
            head=head,
        )

    def set_trainable(self, trainable: bool) -> None:
        """Freeze or unfreeze every parameter of this cell (Algorithm 1, lines 6, 11)."""
        for parameter in self.parameters():
            parameter.requires_grad_(trainable)

    def extra_repr(self) -> str:  # pragma: no cover - trivial
        kind = "final" if self.is_final else "state"
        return f"k={self.cell_index}, kind={kind}, d_in={self.input_dim}"


@torch.no_grad()
def count_parameters(module: nn.Module) -> int:
    """Total number of parameters in ``module``."""
    return sum(p.numel() for p in module.parameters())
