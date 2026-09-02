r"""Basis functions for the structured disturbance component :math:`d_q = \Gamma_q^\top a`."""

from __future__ import annotations

import torch

from pibe.basis.base import DisturbanceBasis
from pibe.basis.bspline import BSplineBasis
from pibe.basis.fourier import FourierBasis

__all__ = ["DisturbanceBasis", "BSplineBasis", "FourierBasis", "build_basis"]

_KINDS = {"bspline": BSplineBasis, "fourier": FourierBasis}


def build_basis(
    kind: str,
    q: int,
    t_start: float,
    t_end: float,
    dtype: torch.dtype = torch.float64,
    **kwargs,
) -> DisturbanceBasis:
    """Construct a basis by name.

    ``kind="bspline"`` accepts ``degree`` and ``knot_style``;
    ``kind="fourier"`` accepts ``omega``.  Options belonging to the other kind
    are rejected rather than silently ignored, so a stale config entry surfaces
    as an error.
    """
    key = kind.lower()
    if key not in _KINDS:
        raise ValueError(f"unknown basis kind {kind!r}, expected one of {sorted(_KINDS)}")

    allowed = {
        "bspline": {"degree", "knot_style"},
        "fourier": {"omega"},
    }[key]
    unexpected = {name for name, value in kwargs.items() if value is not None} - allowed
    if unexpected:
        raise ValueError(
            f"option(s) {sorted(unexpected)} do not apply to the {key} basis; "
            f"it accepts {sorted(allowed)}"
        )
    options = {name: kwargs[name] for name in allowed if kwargs.get(name) is not None}
    return _KINDS[key](q=q, t_start=t_start, t_end=t_end, dtype=dtype, **options)
