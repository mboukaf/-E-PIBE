r"""Fixed non-dimensionalizing normalization of the trajectory input.

Section 4.1 equips the trajectory input :math:`\mathbf{U}_{k-1}` of Eq. (15)
with the norm (61)

.. math::

    \|\mathbf{U}_{k-1}\|_{\mathscr{U}_{k-1}}^2
      = \frac{\|\mathbf{Y}\|_{N,2}^2}{s_y^2}
      + \sum_{j=2}^{k-1} \frac{\|\mathbf{X}_j\|_{N,2}^2}{s_{x_j}^2}
      + \sum_{j=1}^{k-2} \frac{|\theta_j|^2}{s_{\theta_j}^2},

where :math:`\|v\|_{N,2} = (\frac{1}{N}\sum_i |v_i|^2)^{1/2}` is the normalized
discrete :math:`L^2` norm of Eq. (60).  Equivalently
:math:`\|\mathbf{U}_{k-1}\|_{\mathscr{U}_{k-1}} = \|\mathsf{D}_{k-1}
\mathbf{U}_{k-1}\|_2`, where :math:`\mathsf{D}_{k-1}` divides every sampled
trajectory block by :math:`\sqrt{N}` and its reference scale, and every
parameter block by its reference scale.

The paper notes this fixed normalization "can be implemented as an input
preprocessing layer, or absorbed into the first encoder layer"; we take the
former.  The :math:`1/\sqrt{N}` factor is what removes the spurious
grid-size dependence, so that a claim uniform in :math:`N` only additionally
requires the encoder constants :math:`L_{E_k}` to stay bounded as the grid is
refined.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


def trajectory_input_dim(cell_index: int, n_samples: int) -> int:
    r"""The dimension :math:`d_{k-1} = N(k-1) + (k-2)` of Eq. (16).

    Parameters
    ----------
    cell_index
        The cell index ``k``, with ``2 <= k <= n + 1``.
    n_samples
        ``N``.

    Notes
    -----
    ``k = 2`` gives ``N``, since :math:`\mathbf{U}_1 = \mathbf{Y}`; the final
    cell ``k = n + 1`` gives ``N n + (n - 1)``.
    """
    if cell_index < 2:
        raise ValueError(f"cell index must be at least 2, got {cell_index}")
    return n_samples * (cell_index - 1) + (cell_index - 2)


class TrajectoryInputNormalizer(nn.Module):
    r"""The diagonal map :math:`\mathsf{D}_{k-1}` applied to :math:`\mathbf{U}_{k-1}`.

    Parameters
    ----------
    cell_index
        ``k``.
    n_samples
        ``N``.
    state_scales
        :math:`(s_{x_1},\dots,s_{x_n})`, shape ``(n,)``.  Per the paper's
        convention :math:`s_y := s_{x_1}`, so the measurement block reuses the
        first entry.
    theta_scales
        :math:`(s_{\theta_1},\dots,s_{\theta_{n-1}})`, shape ``(n - 1,)``.
    """

    def __init__(
        self,
        cell_index: int,
        n_samples: int,
        state_scales: Tensor,
        theta_scales: Tensor,
    ) -> None:
        super().__init__()
        self.cell_index = int(cell_index)
        self.n_samples = int(n_samples)
        self.input_dim = trajectory_input_dim(cell_index, n_samples)

        k, n_state = self.cell_index, state_scales.numel()
        if k - 1 > n_state:
            raise ValueError(
                f"cell {k} needs {k - 1} state scales but only {n_state} were given"
            )
        if k - 2 > theta_scales.numel():
            raise ValueError(
                f"cell {k} needs {k - 2} parameter scales but only "
                f"{theta_scales.numel()} were given"
            )
        if bool((state_scales <= 0).any()) or bool((theta_scales <= 0).any()):
            raise ValueError("reference scales must be strictly positive")

        root_n = float(self.n_samples) ** 0.5
        blocks = [
            # Y block: s_y = s_{x_1}.  Then X_2, ..., X_{k-1}.
            torch.full((self.n_samples,), 1.0 / (root_n * float(state_scales[j])))
            for j in range(k - 1)
        ]
        # Parameter blocks theta_1, ..., theta_{k-2}, one scalar each.
        blocks.extend(
            torch.full((1,), 1.0 / float(theta_scales[j])) for j in range(k - 2)
        )
        weight = torch.cat(blocks) if blocks else torch.empty(0)
        assert weight.numel() == self.input_dim, (weight.numel(), self.input_dim)
        self.register_buffer("weight", weight.to(state_scales.dtype))

    def forward(self, u: Tensor) -> Tensor:
        """Apply the normalization to ``u`` of shape ``(..., d_{k-1})``."""
        if u.shape[-1] != self.input_dim:
            raise ValueError(
                f"cell {self.cell_index} expects trajectory input of width "
                f"{self.input_dim}, got {u.shape[-1]}"
            )
        return u * self.weight

    def extra_repr(self) -> str:  # pragma: no cover - trivial
        return f"cell_index={self.cell_index}, N={self.n_samples}, dim={self.input_dim}"


def normalize_time(t: Tensor, t_start: float, t_end: float) -> Tensor:
    """Map ``[t_start, t_end]`` affinely onto ``[-1, 1]`` for the decoder input.

    Kept separate from :class:`TrajectoryInputNormalizer`: the time argument of
    :math:`\\mathcal{S}_k` is not part of :math:`\\mathbf{U}_{k-1}` and carries
    no reference scale of its own.  The map is affine in ``t``, so it only
    rescales :math:`\\partial_t` by the constant ``2 / (t_end - t_start)``,
    which autograd tracks exactly.
    """
    span = t_end - t_start
    if span <= 0:
        raise ValueError(f"require t_start < t_end, got ({t_start}, {t_end})")
    return 2.0 * (t - t_start) / span - 1.0
