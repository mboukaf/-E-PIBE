"""Device and dtype resolution."""

from __future__ import annotations

import torch

_DTYPES = {
    "float32": torch.float32,
    "float64": torch.float64,
}


def resolve_device(spec: str = "auto") -> torch.device:
    """Resolve a device specification.

    ``"auto"`` prefers CUDA, then Apple MPS, then CPU.
    """
    if spec != "auto":
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
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
