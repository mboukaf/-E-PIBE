"""Concrete example systems.

Importing this package registers every example with
:mod:`pibe.systems.registry`.  The core package never imports it except
lazily, from :func:`~pibe.systems.registry.build_system`.
"""

from pibe.systems.examples.automatica_n4 import AutomaticaN4System
from pibe.systems.examples.sin_chain import SinChainSystem

__all__ = ["AutomaticaN4System", "SinChainSystem"]
