"""Device and dtype resolution."""

from __future__ import annotations

import torch

_DTYPES = {
    "float32": torch.float32,
    "float64": torch.float64,
}


def resolve_device(spec: str = "auto", dtype: torch.dtype | None = None) -> torch.device:
    """Resolve a device specification.

    ``"auto"`` prefers CUDA, then Apple MPS, then CPU.  When ``dtype`` is given
    it is taken into account: MPS is skipped for float64, since the backend
    cannot represent it and silently falling back to float32 would degrade the
    autograd time derivatives the physics residuals are built on.  An
    *explicit* ``"mps"`` is honoured and left for
    :func:`check_dtype_support` to reject, so a deliberate choice fails loudly
    rather than being quietly overridden.
    """
    if spec != "auto":
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and dtype is not torch.float64:
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(spec: str = "float64") -> torch.dtype:
    """Resolve a floating point dtype by name."""
    try:
        return _DTYPES[spec]
    except KeyError:
        raise ValueError(
            f"unknown dtype {spec!r}, expected one of {sorted(_DTYPES)}"
        ) from None


def check_dtype_support(device: torch.device, dtype: torch.dtype) -> None:
    """Raise if ``dtype`` cannot be used on ``device``.

    Apple MPS has no float64 support; silently downcasting would corrupt the
    time derivatives the physics residuals depend on, so we fail loudly.
    """
    if device.type == "mps" and dtype == torch.float64:
        raise RuntimeError(
            "float64 is not supported on the MPS backend; "
            "use device='cpu' or dtype='float32'"
        )
