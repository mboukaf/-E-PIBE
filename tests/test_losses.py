"""Loss terms and the weighted global objective of Eqs. (27)/(42)."""

from __future__ import annotations

import math

import pytest
import torch

from pibe.core.losses import (
    data_loss,
    global_weights,
    local_loss,
    physics_loss,
    total_loss,
)


def test_data_loss_is_mean_squared_error() -> None:
    prediction = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    target = torch.tensor([[1.0, 0.0], [0.0, 4.0]], dtype=torch.float64)
    # squared errors 0, 4, 9, 0 -> mean 13/4
    assert float(data_loss(prediction, target)) == pytest.approx(13.0 / 4.0)


def test_data_loss_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        data_loss(torch.zeros(2, 3), torch.zeros(2, 4))


def test_physics_loss_is_mean_squared_residual() -> None:
    residual = torch.tensor([[1.0, -2.0, 3.0]], dtype=torch.float64)
    assert float(physics_loss(residual)) == pytest.approx(14.0 / 3.0)


def test_local_loss_weights_the_physics_term() -> None:
    data = torch.tensor(2.0, dtype=torch.float64)
    physics = torch.tensor(5.0, dtype=torch.float64)
    result = local_loss(data, physics, lam=0.25)
    assert float(result.local) == pytest.approx(2.0 + 0.25 * 5.0)
    assert float(result.data) == 2.0
    assert float(result.physics) == 5.0


def test_local_loss_requires_positive_lambda() -> None:
    with pytest.raises(ValueError, match="lambda"):
        local_loss(torch.tensor(1.0), torch.tensor(1.0), lam=0.0)


@pytest.mark.parametrize("cell_index", [2, 3, 5, 7])
def test_global_weights_match_equation_27(cell_index: int) -> None:
    r""":math:`w_m = e^{-(k-m)/k}` for :math:`m = 2,\dots,k`."""
    weights = global_weights(cell_index)
    assert sorted(weights) == list(range(2, cell_index + 1))
    for m, weight in weights.items():
        assert weight == pytest.approx(math.exp(-(cell_index - m) / cell_index))


@pytest.mark.parametrize("cell_index", [2, 4, 6])
def test_current_cell_has_unit_weight(cell_index: int) -> None:
    """The cell being trained is never down-weighted."""
    assert global_weights(cell_index)[cell_index] == pytest.approx(1.0)


def test_weights_decay_towards_the_head_of_the_chain() -> None:
    """Upstream cells are progressively down-weighted."""
    weights = global_weights(6)
    ordered = [weights[m] for m in range(2, 7)]
    assert all(a < b for a, b in zip(ordered, ordered[1:])), (
        "weights must increase with m, i.e. decay towards the head of the chain"
    )


def test_total_loss_is_the_weighted_sum() -> None:
    losses = {
        m: torch.tensor(float(m), dtype=torch.float64) for m in range(2, 5)
    }
    weights = global_weights(4)
    expected = sum(weights[m] * float(m) for m in range(2, 5))
    assert float(total_loss(losses, 4)) == pytest.approx(expected)


def test_total_loss_reports_missing_cells() -> None:
    with pytest.raises(ValueError, match="missing"):
        total_loss({2: torch.tensor(1.0)}, cell_index=4)


def test_total_loss_is_differentiable() -> None:
    parameter = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    losses = {m: parameter**2 for m in range(2, 5)}
    total_loss(losses, 4).backward()
    weight_sum = sum(global_weights(4).values())
    assert float(parameter.grad) == pytest.approx(2.0 * 2.0 * weight_sum)
