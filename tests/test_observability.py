r"""Per-basis-function observability of the disturbance through the plant."""

from __future__ import annotations

import math

import pytest
import torch

from pibe.basis import FourierBasis
from pibe.eval.observability import basis_observability, linearize
from pibe.systems.examples.automatica_n4 import COMPANION, AutomaticaN4System

OMEGA = 2.0 * math.pi / 5.0
BOUNDS = torch.tensor(
    [[-0.15, 0.15], [-0.15, 0.15], [-0.12, 0.12], [-0.12, 0.12]], dtype=torch.float64
)


def test_linearization_recovers_the_companion_structure() -> None:
    r"""At the origin, ``A`` is the companion matrix of the linear skeleton."""
    system = AutomaticaN4System()
    matrix, e_d, c = linearize(system)

    # Chain structure: xdot_j = x_{j+1} + ..., so A[j, j+1] picks up the 1.
    for j in range(3):
        assert float(matrix[j, j + 1]) == pytest.approx(1.0, abs=1e-9)
    # The last row carries -COMPANION plus the derivatives of the smooth terms,
    # which at the origin contribute only 0.06*theta_2*cos(0) to column 3.
    expected = -COMPANION[2] + 0.06 * float(system.theta_true[1])
    assert float(matrix[3, 2]) == pytest.approx(expected, abs=1e-9)
    assert torch.equal(e_d, torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64))
    assert torch.equal(c, torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64))


def test_second_harmonic_is_far_less_observable() -> None:
    r"""The plant attenuates :math:`2\Omega` much more than :math:`\Omega`.

    Four poles between 0.6 and 1.2 make the chain strongly low-pass, so the
    higher-frequency half of the basis leaves a much smaller trace in ``y``
    even though Eq. (3) is perfectly satisfied.
    """
    system = AutomaticaN4System()
    basis = FourierBasis(q=4, omega=OMEGA, t_start=0.0, t_end=20.0)
    result = basis_observability(
        system, basis, BOUNDS, noise_sigma=0.01, n_samples=201
    )

    # Compared on the settled response: the startup transient is a broadband
    # kick amplified by the chain's DC gain (1/0.576), which flatters the fast
    # components and would mask the attenuation being measured here.
    first = result.settled_gain[:2].mean()
    second = result.settled_gain[2:].mean()
    assert float(first / second) > 4.0, (
        f"expected strong attenuation of the second harmonic, got {float(first/second):.2f}x"
    )
    # The weakest function is one of the second-harmonic pair.
    assert result.weakest in (2, 3)


def test_low_noise_lifts_every_coefficient_above_the_floor() -> None:
    """Reducing sigma is what makes the weak half of the basis recoverable."""
    system = AutomaticaN4System()
    basis = FourierBasis(q=4, omega=OMEGA, t_start=0.0, t_end=20.0)

    loud = basis_observability(system, basis, BOUNDS, noise_sigma=0.01, n_samples=201)
    quiet = basis_observability(system, basis, BOUNDS, noise_sigma=0.002, n_samples=201)

    # At sigma = 0.01 the second harmonic is far below the floor pointwise.
    assert float(loud.snr.min()) < 0.25
    # Reducing sigma 5x brings it to the edge pointwise ...
    assert float(quiet.snr.min()) > 0.7
    # ... and comfortably recoverable once the grid is averaged over, which is
    # the regime that actually matters for fitting q coefficients.
    assert float(loud.effective_snr.min()) < 3.0
    assert float(quiet.effective_snr.min()) > 10.0
    # SNR scales exactly inversely with sigma; the gains are unchanged.
    assert torch.allclose(quiet.snr, loud.snr * 5.0, rtol=1e-9)
    assert torch.allclose(quiet.output_gain, loud.output_gain, rtol=1e-9)


def test_effective_snr_includes_the_averaging_factor() -> None:
    system = AutomaticaN4System()
    basis = FourierBasis(q=4, omega=OMEGA, t_start=0.0, t_end=20.0)
    result = basis_observability(system, basis, BOUNDS, noise_sigma=0.01, n_samples=201)
    assert torch.allclose(
        result.effective_snr, result.snr * math.sqrt(201), rtol=1e-12
    )


def test_summary_flags_the_unobservable_component() -> None:
    system = AutomaticaN4System()
    basis = FourierBasis(q=4, omega=OMEGA, t_start=0.0, t_end=20.0)
    text = basis_observability(
        system, basis, BOUNDS, noise_sigma=0.01, n_samples=201
    ).summary()
    assert "below the noise floor" in text
    assert "training cannot help" in text


def test_requires_a_positive_noise_level() -> None:
    system = AutomaticaN4System()
    basis = FourierBasis(q=4, omega=OMEGA, t_start=0.0, t_end=20.0)
    with pytest.raises(ValueError, match="positive noise level"):
        basis_observability(system, basis, BOUNDS, noise_sigma=0.0, n_samples=51)
