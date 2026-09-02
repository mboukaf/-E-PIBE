r"""Bank wiring: trajectory-input assembly and the gradient policy.

The gradient tests are the executable form of Algorithm 1's lines 5-15.  In the
local phase, "freeze :math:`\{\Theta^m_{PINN}\}_{m=2}^{k-1}`" and "update only
:math:`\Theta^k_{PINN}`" mean that no upstream tensor may receive a gradient;
in the fine-tuning phase, evaluating the chain "without detaching the upstream
outputs" means every upstream tensor must.
"""

from __future__ import annotations

import pytest
import torch

from pibe.basis.bspline import BSplineBasis
from pibe.config import ArchitectureConfig
from pibe.core.bank import EstimatorBank, Mode
from pibe.core.losses import total_loss
from pibe.nets.normalization import trajectory_input_dim
from pibe.systems.examples.sin_chain import SinChainSystem

N_SAMPLES = 24
N_COLL = 16
HORIZON = 2.0


def build_bank(n: int = 4, q: int = 5) -> tuple[EstimatorBank, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    system = SinChainSystem(n=n)
    basis = BSplineBasis(q=q, degree=3, t_start=0.0, t_end=HORIZON)
    bounds = torch.tensor([[-4.0, 4.0]], dtype=torch.float64).expand(q, 2).clone()
    bank = EstimatorBank(
        system=system,
        basis=basis,
        n_samples=N_SAMPLES,
        coefficient_bounds=bounds,
        architecture=ArchitectureConfig(
            latent_dim=6, encoder_hidden=(16,), decoder_hidden=(16, 16), head_hidden=(16,)
        ),
    ).to(torch.float64)
    t_data = torch.linspace(0.0, HORIZON, N_SAMPLES, dtype=torch.float64)
    t_coll = torch.linspace(0.0, HORIZON, N_COLL, dtype=torch.float64)
    return bank, t_data, t_coll


# ----------------------------------------------------------------------
# trajectory input, Eq. (15)-(16)
# ----------------------------------------------------------------------


@pytest.mark.parametrize("n", [2, 3, 5])
def test_input_dimensions_match_equation_16(n: int) -> None:
    r""":math:`d_{k-1} = N(k-1) + (k-2)`, with :math:`\mathbf{U}_1 = \mathbf{Y}`."""
    bank, _, _ = build_bank(n=n)
    for k in bank.cell_indices:
        expected = N_SAMPLES * (k - 1) + (k - 2)
        assert bank.cell(k).input_dim == expected == trajectory_input_dim(k, N_SAMPLES)
    assert bank.cell(2).input_dim == N_SAMPLES, "U_1 must be exactly Y"


def test_assembled_input_matches_manual_concatenation() -> None:
    """Eq. (15)'s block order is reproduced exactly."""
    bank, t_data, t_coll = build_bank(n=4)
    y = torch.randn(3, N_SAMPLES, dtype=torch.float64)
    outputs = bank(bank.final_index, y, t_data, t_coll, need_derivative=False)

    for k in bank.cell_indices:
        assembled = bank.assemble_input(k, y, outputs)
        blocks = [y]
        blocks += [outputs[j].x_new_data for j in range(2, k)]
        blocks += [outputs[j].head for j in range(2, k)]
        assert torch.equal(assembled, torch.cat(blocks, dim=-1)), (
            f"U_{k - 1} does not follow the block order of Eq. (15)"
        )


def test_final_cell_has_no_new_coordinate() -> None:
    """Cell ``n+1`` produces only the auxiliary reconstruction of Eq. (33)."""
    bank, t_data, t_coll = build_bank(n=3)
    y = torch.randn(2, N_SAMPLES, dtype=torch.float64)
    outputs = bank(bank.final_index, y, t_data, t_coll, need_derivative=False)
    assert outputs[bank.final_index].x_new_data is None
    assert outputs[bank.final_index].head.shape[-1] == bank.basis.q
    for k in range(2, bank.final_index):
        assert outputs[k].head.shape[-1] == 1


# ----------------------------------------------------------------------
# gradient policy, Algorithm 1 lines 5-15
# ----------------------------------------------------------------------


def _grad_totals(bank: EstimatorBank) -> dict[int, float]:
    return {
        k: sum(
            float(p.grad.abs().sum())
            for p in bank.cell_parameters(k)
            if p.grad is not None
        )
        for k in bank.cell_indices
    }


def test_local_phase_updates_only_the_current_cell() -> None:
    """Lines 6 and 9: upstream cells receive no gradient at all."""
    bank, t_data, t_coll = build_bank(n=4)
    target = 4
    y = torch.randn(3, N_SAMPLES, dtype=torch.float64)

    bank.zero_grad(set_to_none=True)
    outputs = bank(target, y, t_data, t_coll, mode=Mode.LOCAL)
    losses = bank.losses(target, y, t_coll, outputs, lam=1.0, mode=Mode.LOCAL)
    assert set(losses) == {target}, "the local phase scores only the current cell"
    losses[target].local.backward()

    totals = _grad_totals(bank)
    assert totals[target] > 0.0, "the current cell received no gradient"
    for k in bank.cell_indices:
        if k != target:
            assert totals[k] == 0.0, (
                f"cell {k} received gradient during local pre-training of cell {target}"
            )


def test_global_phase_reaches_every_upstream_cell() -> None:
    """Lines 11-15: the chain stays differentiable end to end."""
    bank, t_data, t_coll = build_bank(n=4)
    target = 4
    y = torch.randn(3, N_SAMPLES, dtype=torch.float64)

    bank.zero_grad(set_to_none=True)
    outputs = bank(target, y, t_data, t_coll, mode=Mode.GLOBAL)
    losses = bank.losses(target, y, t_coll, outputs, lam=1.0, mode=Mode.GLOBAL)
    assert set(losses) == set(range(2, target + 1))
    total_loss({m: loss.local for m, loss in losses.items()}, target).backward()

    totals = _grad_totals(bank)
    for k in range(2, target + 1):
        assert totals[k] > 0.0, f"cell {k} received no gradient during fine-tuning"
    for k in range(target + 1, bank.final_index + 1):
        assert totals[k] == 0.0, f"downstream cell {k} should not be trained yet"


def test_local_upstream_outputs_are_detached() -> None:
    """Line 7: upstream outputs entering ``U_{k-1}`` carry no autograd history."""
    bank, t_data, t_coll = build_bank(n=4)
    y = torch.randn(2, N_SAMPLES, dtype=torch.float64)
    outputs = bank(4, y, t_data, t_coll, mode=Mode.LOCAL)
    for k in (2, 3):
        assert not outputs[k].x_new_data.requires_grad
        assert not outputs[k].head.requires_grad
    assert outputs[4].x_prev_data.requires_grad, "the target cell must stay differentiable"


# ----------------------------------------------------------------------
# reporting convention
# ----------------------------------------------------------------------


def test_estimate_follows_reporting_convention() -> None:
    r"""Reported states are :math:`(\hat x^2_1, \hat x^2_2, \dots, \hat x^n_n)`.

    The final cell's :math:`\hat x^{n+1}_n` is auxiliary and must not replace
    :math:`\hat x^n_n` in the reported trajectory.
    """
    bank, t_data, t_coll = build_bank(n=4)
    y = torch.randn(3, N_SAMPLES, dtype=torch.float64)
    estimates = bank.estimate(y, t_data, t_coll)

    assert estimates.x.shape == (3, N_SAMPLES, 4)
    assert estimates.theta.shape == (3, 3)
    assert estimates.a.shape == (3, bank.basis.q)
    assert estimates.d.shape == (3, N_SAMPLES)

    outputs = bank(bank.final_index, y, t_data, t_coll, need_derivative=False)
    assert torch.allclose(estimates.x[..., 0], outputs[2].x_prev_data)
    for j in range(2, 5):
        assert torch.allclose(estimates.x[..., j - 1], outputs[j].x_new_data)
    assert torch.allclose(estimates.x_aux, outputs[bank.final_index].x_prev_data)
    assert not torch.allclose(estimates.x[..., 3], estimates.x_aux), (
        "the auxiliary final-cell trajectory was reported as x_n"
    )


def test_estimates_respect_admissible_boxes() -> None:
    """Decoder and head outputs land inside the admissible sets by construction."""
    bank, t_data, t_coll = build_bank(n=3)
    y = 50.0 * torch.randn(4, N_SAMPLES, dtype=torch.float64)  # wildly out of range
    estimates = bank.estimate(y, t_data, t_coll)

    lo = bank.system.state_bounds[:, 0]
    hi = bank.system.state_bounds[:, 1]
    assert bool((estimates.x >= lo).all()) and bool((estimates.x <= hi).all())

    theta_lo = bank.system.theta_bounds[:, 0]
    theta_hi = bank.system.theta_bounds[:, 1]
    assert bool((estimates.theta >= theta_lo).all())
    assert bool((estimates.theta <= theta_hi).all())
