"""The trigonometric disturbance basis."""

from __future__ import annotations

import math

import pytest
import torch

from pibe.basis import FourierBasis, build_basis

OMEGA = 2.0 * math.pi / 5.0
HORIZON = 20.0


def make(q: int = 4, omega: float = OMEGA, t_end: float = HORIZON) -> FourierBasis:
    return FourierBasis(q=q, omega=omega, t_start=0.0, t_end=t_end)


def test_component_order_matches_the_specification() -> None:
    r""":math:`\Gamma_4 = [\sin\Omega t, \cos\Omega t, \sin 2\Omega t, \cos 2\Omega t]^\top`."""
    basis = make()
    t = torch.linspace(0.0, HORIZON, 37, dtype=torch.float64)
    expected = torch.stack(
        [
            torch.sin(OMEGA * t),
            torch.cos(OMEGA * t),
            torch.sin(2 * OMEGA * t),
            torch.cos(2 * OMEGA * t),
        ],
        dim=-1,
    )
    assert torch.allclose(basis.evaluate(t), expected, atol=1e-14)


def test_gram_is_exactly_half_the_horizon_times_identity() -> None:
    r"""Four whole periods give :math:`W_\Gamma = (T/2) I_q = 10 I_4`.

    This is the perfectly conditioned case: Eq. (3) holds with
    :math:`\underline{\gamma}_d = T/2`.
    """
    basis = make()
    gram = basis.gram()
    assert torch.allclose(
        gram, 10.0 * torch.eye(4, dtype=torch.float64), atol=1e-11
    ), f"max deviation {float((gram - 10 * torch.eye(4, dtype=torch.float64)).abs().max()):.3e}"
    assert basis.min_gram_eigenvalue() == pytest.approx(10.0, abs=1e-9)


def test_quadrature_agrees_with_the_analytic_gram() -> None:
    basis = make()
    assert torch.allclose(basis.gram(), basis.analytic_gram(), atol=1e-11)


def test_analytic_gram_is_unavailable_off_resonance() -> None:
    """An incommensurate horizon has no diagonal closed form."""
    basis = make(t_end=17.3)
    assert basis.analytic_gram() is None
    # Still linearly independent, just worse conditioned than T/2.
    gamma_d = basis.min_gram_eigenvalue()
    assert 0.0 < gamma_d < 0.5 * 17.3


@pytest.mark.parametrize("q", [2, 4, 6, 8])
def test_derivative_matches_central_differences(q: int) -> None:
    basis = make(q=q)
    t = torch.linspace(0.5, HORIZON - 0.5, 61, dtype=torch.float64)
    step = 1e-6
    numeric = (basis.evaluate(t + step) - basis.evaluate(t - step)) / (2.0 * step)
    assert torch.allclose(basis.derivative(t), numeric, atol=1e-7)


def test_rejects_odd_q() -> None:
    """Sine/cosine pairs require an even number of basis functions."""
    with pytest.raises(ValueError, match="even q"):
        FourierBasis(q=5)


def test_default_omega_is_one_period_over_the_horizon() -> None:
    basis = FourierBasis(q=4, t_start=0.0, t_end=8.0)
    assert basis.omega == pytest.approx(2.0 * math.pi / 8.0)
    assert basis.periods == pytest.approx(1.0)


def test_build_basis_dispatches_and_validates() -> None:
    fourier = build_basis("fourier", q=4, t_start=0.0, t_end=HORIZON, omega=OMEGA)
    assert isinstance(fourier, FourierBasis)

    # Options belonging to the other kind must be rejected, not ignored.
    with pytest.raises(ValueError, match="do not apply to the fourier basis"):
        build_basis("fourier", q=4, t_start=0.0, t_end=HORIZON, degree=3)
    with pytest.raises(ValueError, match="do not apply to the bspline basis"):
        build_basis("bspline", q=6, t_start=0.0, t_end=HORIZON, omega=1.0)
    with pytest.raises(ValueError, match="unknown basis kind"):
        build_basis("chebyshev", q=4, t_start=0.0, t_end=HORIZON)


def test_dtype_and_device_move() -> None:
    basis = make().to(dtype=torch.float32)
    assert basis.dtype is torch.float32
    assert basis.evaluate(torch.linspace(0, 1, 5)).dtype is torch.float32
