"""Name-based lookup of :class:`~pibe.systems.base.TriangularSystem` subclasses.

Lets a config file name a system without the core package importing any
concrete example.  Register with the decorator::

    @register_system("my_system")
    class MySystem(TriangularSystem):
        ...
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from pibe.systems.base import TriangularSystem

_REGISTRY: dict[str, type[TriangularSystem]] = {}

T = TypeVar("T", bound=type[TriangularSystem])


def register_system(name: str) -> Callable[[T], T]:
    """Class decorator registering a system under ``name``."""

    def decorator(cls: T) -> T:
        key = name.lower()
        if key in _REGISTRY and _REGISTRY[key] is not cls:
            raise ValueError(f"system name {name!r} is already registered")
        _REGISTRY[key] = cls
        return cls

    return decorator


def build_system(name: str, **kwargs: Any) -> TriangularSystem:
    """Instantiate a registered system.

    Concrete systems live in :mod:`pibe.systems.examples`; importing that
    package is what populates the registry, so it is imported lazily here to
    keep the core free of example dependencies.
    """
    import pibe.systems.examples  # noqa: F401  (import for side effect)

    key = name.lower()
    if key not in _REGISTRY:
        raise KeyError(
            f"unknown system {name!r}; registered systems are {sorted(_REGISTRY)}"
        )
    return _REGISTRY[key](**kwargs)


def available_systems() -> list[str]:
    """Names of all registered systems."""
    import pibe.systems.examples  # noqa: F401

    return sorted(_REGISTRY)
