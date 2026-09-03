r"""Fourier time features on the decoder input."""

from __future__ import annotations

import pytest
import torch

from pibe.core.autodiff import make_time_input, time_derivative
from pibe.nets.decoder import FourierTimeFeatures, StateDecoder

BOUNDS = torch.tensor([[-3.0, 3.0], [-3.0, 3.0]], dtype=torch.float64)


def test_disabled_features_are_the_identity() -> None:
    lift = FourierTimeFeatures(0)
    t = torch.linspace(-1, 1, 9, dtype=torch.float64).reshape(-1, 1)
    assert lift.out_dim == 1
    assert torch.equal(lift(t), t)


@pytest.mark.parametrize("k", [1, 4, 8])
def test_feature_values_and_width(k: int) -> None:
    """Raw (unscaled) feature values and layout."""
    lift = FourierTimeFeatures(k, scale_by_harmonic=False).to(torch.float64)
    t = torch.linspace(-1, 1, 11, dtype=torch.float64).reshape(-1, 1)
    out = lift(t)
    assert lift.out_dim == 1 + 2 * k
    assert out.shape == (11, 1 + 2 * k)
    assert torch.allclose(out[:, 0:1], t)
    for h in range(1, k + 1):
        assert torch.allclose(out[:, h], torch.sin(torch.pi * h * t[:, 0]), atol=1e-12)
        assert torch.allclose(out[:, k + h], torch.cos(torch.pi * h * t[:, 0]), atol=1e-12)


def test_features_keep_the_decoder_twice_differentiable() -> None:
    r"""Eq. (21) needs ``C^2`` in ``t``; sines and cosines are ``C^inf``."""
    torch.manual_seed(0)
    decoder = StateDecoder(
        latent_dim=4, output_bounds=BOUNDS, t_start=0.0, t_end=2.0,
        hidden=(16, 16), time_fourier_features=6,
    ).to(torch.float64)
    z = torch.randn(2, 4, dtype=torch.float64)
    grid = torch.linspace(0.1, 1.9, 13, dtype=torch.float64)

    t = make_time_input(grid, 2)
    first = time_derivative(decoder(t, z)[..., 0], t, create_graph=True)
    second = torch.autograd.grad(first.sum(), t, create_graph=False)[0]
    assert torch.isfinite(second).all()


def test_derivative_still_matches_finite_differences() -> None:
    """The lift must not break the autograd time derivative."""
    torch.manual_seed(1)
    decoder = StateDecoder(
        latent_dim=4, output_bounds=BOUNDS, t_start=0.0, t_end=2.0,
        hidden=(16, 16), time_fourier_features=8,
    ).to(torch.float64)
    z = torch.randn(3, 4, dtype=torch.float64)
    grid = torch.linspace(0.2, 1.8, 15, dtype=torch.float64)

    t = make_time_input(grid, 3)
    analytic = time_derivative(decoder(t, z)[..., 0], t, create_graph=False)
    step = 1e-6
    with torch.no_grad():
        plus = decoder(make_time_input(grid + step, 3, requires_grad=False), z)
        minus = decoder(make_time_input(grid - step, 3, requires_grad=False), z)
    numeric = (plus[..., 0] - minus[..., 0]) / (2.0 * step)
    assert torch.allclose(analytic, numeric, atol=1e-6)


def test_features_help_fit_an_oscillatory_target() -> None:
    r"""The point of the lift: resolve several periods over the horizon.

    A plain tanh MLP in ``t`` has a spectral bias against exactly the
    oscillation the disturbance imprints on the states, and that fit error is
    what destroys the disturbance estimate downstream.
    """
    horizon, periods = 20.0, 4
    grid = torch.linspace(0.0, horizon, 201, dtype=torch.float64)
    target = torch.sin(2 * torch.pi * periods * grid / horizon).unsqueeze(0)

    def fit(n_features: int) -> float:
        torch.manual_seed(0)
        decoder = StateDecoder(
            latent_dim=2, output_bounds=BOUNDS, t_start=0.0, t_end=horizon,
            hidden=(64, 64), time_fourier_features=n_features,
        ).to(torch.float64)
        z = torch.zeros(1, 2, dtype=torch.float64)
        opt = torch.optim.Adam(decoder.parameters(), lr=3e-3)
        t = make_time_input(grid, 1, requires_grad=False)
        for _ in range(600):
            opt.zero_grad(set_to_none=True)
            loss = ((decoder(t, z)[..., 0] - target) ** 2).mean()
            loss.backward()
            opt.step()
        return float(loss.detach())

    plain, lifted = fit(0), fit(8)
    assert lifted < plain / 10, (
        f"Fourier features should dominate on an oscillatory target: "
        f"plain={plain:.3e}, lifted={lifted:.3e}"
    )


def test_harmonic_scaling_equalises_derivative_influence() -> None:
    r"""Scaling by ``1/k`` makes every harmonic contribute equally to ``d/dt``.

    Unscaled, harmonic ``k``'s derivative grows like ``k``, so a large lift lets
    the fastest features dominate the physics residual and its conditioning
    collapses.  This is measured, not assumed: it is why ``K = 40`` trains far
    worse than ``K = 10`` without the scaling.
    """
    grid = torch.linspace(-1, 1, 401, dtype=torch.float64).reshape(-1, 1)
    step = 1e-6

    def derivative_scale(scaled: bool) -> torch.Tensor:
        lift = FourierTimeFeatures(16, scale_by_harmonic=scaled).to(torch.float64)
        d = (lift(grid + step) - lift(grid - step)) / (2 * step)
        # RMS derivative magnitude of each sine feature (columns 1..K).
        return d[:, 1:17].pow(2).mean(0).sqrt()

    unscaled = derivative_scale(False)
    scaled = derivative_scale(True)
    # Unscaled: the 16th harmonic's derivative is ~16x the first.
    assert float(unscaled[-1] / unscaled[0]) == pytest.approx(16.0, rel=0.05)
    # Scaled: every harmonic contributes the same derivative magnitude.
    assert float(scaled.max() / scaled.min()) == pytest.approx(1.0, abs=0.02)


def test_scaling_preserves_representable_functions() -> None:
    """The lift still spans the same space; only the parameterization changes."""
    grid = torch.linspace(-1, 1, 51, dtype=torch.float64).reshape(-1, 1)
    scaled = FourierTimeFeatures(6, scale_by_harmonic=True).to(torch.float64)(grid)
    plain = FourierTimeFeatures(6, scale_by_harmonic=False).to(torch.float64)(grid)
    ks = torch.arange(1, 7, dtype=torch.float64)
    assert torch.allclose(scaled[:, 1:7] * ks, plain[:, 1:7], atol=1e-12)
    assert torch.allclose(scaled[:, 7:13] * ks, plain[:, 7:13], atol=1e-12)
