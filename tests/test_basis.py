"""B-spline basis of Definition 1."""

from __future__ import annotations

import pytest
import torch

from pibe.basis.bspline import BSplineBasis

CASES = [(6, 3), (8, 2), (5, 4), (12, 3)]
STYLES = ["clamped", "uniform"]


def make(q: int, degree: int, style: str = "clamped") -> BSplineBasis:
    return BSplineBasis(q=q, degree=degree, t_start=0.0, t_end=2.5, knot_style=style)


@pytest.mark.parametrize("style", STYLES)
@pytest.mark.parametrize(("q", "degree"), CASES)
def test_knot_vector_length(q: int, degree: int, style: str) -> None:
    """The knot vector is ``{xi_0, ..., xi_{q+m}}``, i.e. ``q + m + 1`` entries."""
    basis = make(q, degree, style)
    assert basis.knots.numel() == q + degree + 1
    assert bool((basis.knots[1:] >= basis.knots[:-1]).all()), "knots must be nondecreasing"


@pytest.mark.parametrize("style", STYLES)
@pytest.mark.parametrize(("q", "degree"), CASES)
def test_partition_of_unity_and_nonnegativity(q: int, degree: int, style: str) -> None:
    """The ``q`` basis functions are nonnegative and sum to one on the horizon."""
    basis = make(q, degree, style)
    t = torch.linspace(0.0, 2.5, 401, dtype=torch.float64)
    values = basis.evaluate(t)
    assert values.shape == (401, q)
    assert bool((values >= -1e-14).all())
    assert torch.allclose(values.sum(-1), torch.ones_like(t), atol=1e-12)


@pytest.mark.parametrize("style", STYLES)
@pytest.mark.parametrize(("q", "degree"), [(6, 3), (5, 4), (12, 3)])
def test_derivative_matches_central_differences(q: int, degree: int, style: str) -> None:
    """``derivative`` agrees with finite differences away from the knots.

    Degree 2 is excluded: its second derivative jumps at interior knots, so a
    central difference straddling one is inaccurate for reasons unrelated to
    the implementation.
    """
    basis = make(q, degree, style)
    t = torch.linspace(0.05, 2.45, 97, dtype=torch.float64)
    step = 1e-6
    numeric = (basis.evaluate(t + step) - basis.evaluate(t - step)) / (2.0 * step)
    assert torch.allclose(basis.derivative(t), numeric, atol=1e-7)


@pytest.mark.parametrize("style", STYLES)
@pytest.mark.parametrize(("q", "degree"), CASES)
def test_gram_is_positive_definite(q: int, degree: int, style: str) -> None:
    """Eq. (3): the basis functions are linearly independent on ``[0, T]``."""
    basis = make(q, degree, style)
    gram = basis.gram()
    assert gram.shape == (q, q)
    assert torch.allclose(gram, gram.T, atol=1e-14), "the Gram matrix must be symmetric"
    assert basis.min_gram_eigenvalue() > 0.0
    assert basis.check_linear_independence() > 0.0


def test_right_endpoint_is_continuous_from_the_left() -> None:
    """Definition 1: "values at ``t = T`` are defined by continuous extension"."""
    basis = make(6, 3)
    at_end = basis.evaluate(torch.tensor([2.5], dtype=torch.float64))
    just_before = basis.evaluate(torch.tensor([2.5 - 1e-12], dtype=torch.float64))
    assert torch.allclose(at_end, just_before, atol=1e-10)
    assert torch.allclose(at_end.sum(), torch.tensor(1.0, dtype=torch.float64))


def test_gram_quadrature_is_exact() -> None:
    """More quadrature nodes must not change the Gram matrix.

    Each basis function is a degree-``m`` polynomial on every knot span, so
    ``m + 1`` Gauss-Legendre nodes integrate the degree-``2m`` product exactly.
    """
    basis = make(8, 3)
    assert torch.allclose(basis.gram(), basis.gram(nodes_per_span=12), atol=1e-13)


def test_rejects_degree_below_two() -> None:
    """Eq. (2) requires ``Gamma_q`` in ``C^1``, hence ``m >= 2``."""
    with pytest.raises(ValueError, match="degree"):
        BSplineBasis(q=6, degree=1)


def test_rejects_too_few_basis_functions() -> None:
    with pytest.raises(ValueError, match="q >= degree"):
        BSplineBasis(q=3, degree=3)
