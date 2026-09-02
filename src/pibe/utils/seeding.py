"""Reproducible random number generation."""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and torch global RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_generator(seed: int, device: torch.device | str = "cpu") -> torch.Generator:
    """Return a dedicated :class:`torch.Generator`.

    Preferred over the global RNG wherever draws must be reproducible
    independently of unrelated calls (data generation, minibatch sampling).
    """
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator
