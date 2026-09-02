r"""The autograd time derivative of Eqs. (21) and (36).

Two independent failure modes are checked.

**Value.**  The derivative must match central differences of the decoder.

**Aliasing.**  :func:`~pibe.core.autodiff.make_time_input` must give every
``(trajectory, time)`` pair its own tensor element.  Had the grid been
``expand``-ed without a copy, the reverse pass would write
:math:`\sum_b \partial_t \hat x[b,m]` into every batch row, so all trajectories
would share one derivative field.  The test below detects exactly that by
comparing per-trajectory derivatives against per-trajectory finite differences
for latents chosen to give visibly different slopes.
"""

from __future__ import annotations

import torch

from pibe.core.autodiff import make_time_input, time_derivative
from pibe.nets.decoder import StateDecoder


def build_decoder(latent_dim: int = 4, seed: int = 0) -> StateDecoder:
    torch.manual_seed(seed)
    return StateDecoder(
        latent_dim=latent_dim,
        output_bounds=torch.tensor([[-3.0, 3.0], [-3.0, 3.0]], dtype=torch.float64),
        t_start=0.0,
        t_end=2.0,
        hidden=(16, 16),
    ).to(torch.float64)


def test_time_derivative_matches_central_differences() -> None:
    """The autograd derivative agrees with a finite-difference estimate."""
    decoder = build_decoder()
    batch, latent_dim = 5, decoder.latent_dim
    torch.manual_seed(1)
    z = torch.randn(batch, latent_dim, dtype=torch.float64)
    grid = torch.linspace(0.2, 1.8, 17, dtype=torch.float64)

    t = make_time_input(grid, batch)
    values = decoder(t, z)[..., 0]
    analytic = time_derivative(values, t, create_graph=False)

    step = 1e-6
    with torch.no_grad():
        plus = decoder(make_time_input(grid + step, batch, requires_grad=False), z)
        minus = decoder(make_time_input(grid - step, batch, requires_grad=False), z)
    numeric = (plus[..., 0] - minus[..., 0]) / (2.0 * step)

    assert torch.allclose(analytic, numeric, atol=1e-7), (
        f"max discrepancy {float((analytic - numeric).abs().max()):.3e}"
    )


def test_derivative_is_per_trajectory() -> None:
    """Each trajectory gets its own derivative field, not a batch-summed one."""
    decoder = build_decoder()
    latent_dim = decoder.latent_dim
    # Deliberately distinct latents so the derivative fields differ.
    z = torch.stack(
        [
            torch.full((latent_dim,), -2.0, dtype=torch.float64),
            torch.full((latent_dim,), 2.0, dtype=torch.float64),
            torch.zeros(latent_dim, dtype=torch.float64),
        ]
    )
    grid = torch.linspace(0.1, 1.9, 11, dtype=torch.float64)

    t = make_time_input(grid, z.shape[0])
    batched = time_derivative(decoder(t, z)[..., 0], t, create_graph=False)

    # The derivative fields must not be identical across the batch, otherwise
    # the test cannot distinguish correct behaviour from batch summation.
    assert not torch.allclose(batched[0], batched[1]), (
        "chosen latents give indistinguishable derivatives; test is vacuous"
    )

    # Each row must equal the derivative computed for that trajectory alone.
    for index in range(z.shape[0]):
        single_t = make_time_input(grid, 1)
        single = time_derivative(
            decoder(single_t, z[index : index + 1])[..., 0],
            single_t,
            create_graph=False,
        )
        assert torch.allclose(batched[index], single[0], atol=1e-12), (
            f"trajectory {index}'s derivative depends on the rest of the batch; "
            "the time grid is aliased across the batch dimension"
        )


def test_derivative_is_differentiable_wrt_parameters() -> None:
    """``create_graph=True`` lets the physics loss reach the network weights.

    Without the retained graph the physics term would contribute no gradient,
    and the whole physics-informed half of the objective would be inert.
    """
    decoder = build_decoder()
    z = torch.randn(3, decoder.latent_dim, dtype=torch.float64)
    grid = torch.linspace(0.0, 2.0, 9, dtype=torch.float64)

    t = make_time_input(grid, 3)
    derivative = time_derivative(decoder(t, z)[..., 0], t, create_graph=True)
    (derivative**2).mean().backward()

    grads = [p.grad for p in decoder.parameters() if p.grad is not None]
    assert grads, "no parameter received a gradient from the time derivative"
    total = sum(float(g.abs().sum()) for g in grads)
    assert total > 0.0, "the physics-side gradient vanished identically"


def test_make_time_input_shape_and_leafness() -> None:
    """The constructed time input is a differentiable leaf of the right shape."""
    grid = torch.linspace(0.0, 1.0, 7, dtype=torch.float64)
    t = make_time_input(grid, 4)
    assert t.shape == (4, 7, 1)
    assert t.requires_grad and t.is_leaf
    assert torch.allclose(t[0, :, 0], grid)

    plain = make_time_input(grid, 4, requires_grad=False)
    assert not plain.requires_grad
