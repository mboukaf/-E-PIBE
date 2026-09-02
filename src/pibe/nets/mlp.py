r"""Multi-layer perceptrons with twice-differentiable activations.

Eq. (21) computes the state derivative

.. math:: \dot{\hat x}^{k,\ell}_{k-1}(t) = \frac{\partial}{\partial t}
          \mathcal{S}_k(t, \mathbf{z}^\ell_{k-1})

by automatic differentiation, and Section 3.1 requires the state decoder to
"use twice continuously differentiable activation functions".  Piecewise-linear
activations (ReLU and relatives) are therefore *rejected at construction time*
rather than silently producing an almost-everywhere-defined derivative and a
physics residual with no second-order regularity.

Lemma 1 bounds the network Lipschitz constant by
:math:`\Lambda = \prod_{j=1}^L \alpha_j \|A^{(j)}\|`; the activation constants
:math:`\alpha_j` below are all 1.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

# Activations that are C^2 on all of R, as required by Eq. (21).
_C2_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "softplus": nn.Softplus,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "elu": nn.ELU,  # C^1 only at 0 for alpha != 1; C^inf elsewhere
}

_REJECTED = {"relu", "leaky_relu", "prelu", "rrelu", "hardtanh", "relu6"}


def make_activation(name: str) -> nn.Module:
    """Instantiate a twice-differentiable activation by name."""
    key = name.lower()
    if key in _REJECTED:
        raise ValueError(
            f"activation {name!r} is not twice continuously differentiable; "
            f"Eq. (21) differentiates the decoder in t, so choose one of "
            f"{sorted(_C2_ACTIVATIONS)}"
        )
    if key not in _C2_ACTIVATIONS:
        raise ValueError(
            f"unknown activation {name!r}, expected one of {sorted(_C2_ACTIVATIONS)}"
        )
    return _C2_ACTIVATIONS[key]()


class MLP(nn.Module):
    """A plain fully connected network.

    Parameters
    ----------
    in_dim, out_dim
        Input and output widths.
    hidden
        Hidden layer widths.  Empty means a single affine map.
    activation
        Name of a ``C^2`` activation; see :func:`make_activation`.
    bias
        Whether affine layers carry a bias.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: Sequence[int] = (64, 64),
        activation: str = "tanh",
        bias: bool = True,
    ) -> None:
        super().__init__()
        if in_dim < 1 or out_dim < 1:
            raise ValueError(f"invalid widths: in_dim={in_dim}, out_dim={out_dim}")
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.activation_name = activation

        widths = [self.in_dim, *hidden, self.out_dim]
        layers: list[nn.Module] = []
        for index in range(len(widths) - 1):
            layers.append(nn.Linear(widths[index], widths[index + 1], bias=bias))
            if index < len(widths) - 2:
                layers.append(make_activation(activation))
        self.net = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Xavier initialization with the gain matched to the activation."""
        try:
            gain = nn.init.calculate_gain(self.activation_name)
        except ValueError:
            gain = 1.0
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=gain)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

    @torch.no_grad()
    def lipschitz_upper_bound(self) -> float:
        r"""The bound :math:`\prod_j \alpha_j \|A^{(j)}\|_2` of Lemma 1.

        The activation constants :math:`\alpha_j` are 1 for every activation in
        :data:`_C2_ACTIVATIONS`, so this is the product of the spectral norms
        of the weight matrices.  It is an upper bound, generally loose, and is
        reported for the Section 4 constants rather than used in training.
        """
        bound = 1.0
        for module in self.net:
            if isinstance(module, nn.Linear):
                bound *= float(torch.linalg.matrix_norm(module.weight, ord=2))
        return bound
