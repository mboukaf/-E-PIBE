r"""Network building blocks: the normalization (61), bounded outputs, activations."""

from __future__ import annotations

import pytest
import torch

from pibe.nets.mlp import MLP, make_activation
from pibe.nets.normalization import (
    TrajectoryInputNormalizer,
    normalize_time,
    trajectory_input_dim,
)
from pibe.nets.reparam import BoxReparameterization, make_output_map


# ----------------------------------------------------------------------
# input normalization, Eq. (61)
# ----------------------------------------------------------------------


@pytest.mark.parametrize("cell_index", [2, 3, 5])
def test_normalizer_reproduces_the_u_norm(cell_index: int) -> None:
    r""":math:`\|\mathsf{D}_{k-1}\mathbf{U}\|_2` equals the norm of Eq. (61).

    That norm is
    :math:`\|\mathbf{Y}\|_{N,2}^2/s_y^2 + \sum_j \|\mathbf{X}_j\|_{N,2}^2/s_{x_j}^2
    + \sum_j |\theta_j|^2/s_{\theta_j}^2` with
    :math:`\|v\|_{N,2}^2 = \frac{1}{N}\sum_i v_i^2`.
    """
    torch.manual_seed(0)
    n_samples, n_state = 16, 6
    state_scales = torch.linspace(0.5, 3.0, n_state, dtype=torch.float64)
    theta_scales = torch.linspace(0.25, 2.0, n_state - 1, dtype=torch.float64)

    normalizer = TrajectoryInputNormalizer(
        cell_index, n_samples, state_scales, theta_scales
    )
    state_blocks = [
        torch.randn(1, n_samples, dtype=torch.float64) for _ in range(cell_index - 1)
    ]
    theta_blocks = [
        torch.randn(1, 1, dtype=torch.float64) for _ in range(cell_index - 2)
    ]
    u = torch.cat(state_blocks + theta_blocks, dim=-1)

    normalized = normalizer(u)
    computed = float((normalized**2).sum())

    expected = sum(
        float((block**2).mean()) / float(state_scales[j]) ** 2
        for j, block in enumerate(state_blocks)
    ) + sum(
        float(block[0, 0]) ** 2 / float(theta_scales[j]) ** 2
        for j, block in enumerate(theta_blocks)
    )
    assert computed == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize(("cell_index", "n_samples"), [(2, 10), (4, 7), (6, 32)])
def test_trajectory_input_dim(cell_index: int, n_samples: int) -> None:
    assert trajectory_input_dim(cell_index, n_samples) == n_samples * (
        cell_index - 1
    ) + (cell_index - 2)


def test_normalizer_rejects_wrong_width() -> None:
    scales = torch.ones(4, dtype=torch.float64)
    normalizer = TrajectoryInputNormalizer(3, 8, scales, scales[:3])
    with pytest.raises(ValueError, match="expects trajectory input of width"):
        normalizer(torch.zeros(1, 5, dtype=torch.float64))


def test_normalize_time_maps_horizon_to_unit_interval() -> None:
    t = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
    scaled = normalize_time(t, 0.0, 2.0)
    assert torch.allclose(scaled, torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64))


# ----------------------------------------------------------------------
# bounded output reparameterization
# ----------------------------------------------------------------------


def test_box_reparameterization_never_leaves_the_box() -> None:
    """Outputs stay in the admissible set even for absurd pre-activations.

    The map is onto the *open* interval mathematically, but ``tanh`` saturates
    to exactly ``+-1`` in floating point, so the numerical image is the closed
    box.  The property that matters --- the admissible set is never exited ---
    holds either way.
    """
    bounds = torch.tensor([[-1.0, 3.0], [0.0, 0.5]], dtype=torch.float64)
    reparam = BoxReparameterization(bounds)
    z = 1e3 * torch.randn(500, 2, dtype=torch.float64)
    out = reparam(z)
    assert bool((out >= bounds[:, 0]).all()) and bool((out <= bounds[:, 1]).all())


def test_box_reparameterization_is_interior_for_moderate_inputs() -> None:
    """In the regime training actually operates in, the image is interior."""
    bounds = torch.tensor([[-1.0, 3.0], [0.0, 0.5]], dtype=torch.float64)
    reparam = BoxReparameterization(bounds)
    out = reparam(torch.randn(500, 2, dtype=torch.float64))
    assert bool((out > bounds[:, 0]).all()) and bool((out < bounds[:, 1]).all())


def test_box_reparameterization_centres_at_zero() -> None:
    bounds = torch.tensor([[-2.0, 6.0]], dtype=torch.float64)
    reparam = BoxReparameterization(bounds)
    centre = reparam(torch.zeros(1, 1, dtype=torch.float64))
    assert float(centre) == pytest.approx(2.0)


def test_box_reparameterization_is_smooth() -> None:
    """Needed so the decoder stays ``C^2`` in ``t`` (Eq. 21)."""
    bounds = torch.tensor([[-1.0, 1.0]], dtype=torch.float64)
    reparam = BoxReparameterization(bounds)
    z = torch.zeros(1, 1, dtype=torch.float64, requires_grad=True)
    first = torch.autograd.grad(reparam(z).sum(), z, create_graph=True)[0]
    second = torch.autograd.grad(first.sum(), z)[0]
    assert torch.isfinite(second).all()


def test_make_output_map_without_bounds_is_identity() -> None:
    identity = make_output_map(None, 3)
    z = torch.randn(4, 3)
    assert torch.equal(identity(z), z)


# ----------------------------------------------------------------------
# activations
# ----------------------------------------------------------------------


@pytest.mark.parametrize("name", ["relu", "leaky_relu", "relu6"])
def test_rejects_non_c2_activations(name: str) -> None:
    """Eq. (21) differentiates the decoder in ``t``; kinks are not acceptable."""
    with pytest.raises(ValueError, match="twice continuously differentiable"):
        make_activation(name)


@pytest.mark.parametrize("name", ["tanh", "gelu", "silu", "softplus"])
def test_accepts_c2_activations(name: str) -> None:
    assert make_activation(name) is not None
    MLP(in_dim=2, out_dim=1, hidden=(4,), activation=name)


def test_mlp_rejects_unknown_activation() -> None:
    with pytest.raises(ValueError, match="unknown activation"):
        MLP(in_dim=1, out_dim=1, activation="not_an_activation")


def test_mlp_lipschitz_bound_is_positive() -> None:
    """Lemma 1's product of spectral norms."""
    mlp = MLP(in_dim=3, out_dim=2, hidden=(8, 8))
    assert mlp.lipschitz_upper_bound() > 0.0


def test_mlp_without_hidden_layers_is_affine() -> None:
    mlp = MLP(in_dim=3, out_dim=2, hidden=())
    assert mlp(torch.zeros(5, 3)).shape == (5, 2)
