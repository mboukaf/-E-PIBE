r"""Energy-based residual models for EPIBE, Section 3.2.

PIBE's quadratic data term is the negative log-likelihood of a zero-mean
Gaussian up to constants, so it *asserts* a noise location rather than inferring
one.  Proposition 1 makes the consequence precise: the data risk alone cannot
separate the physical signal from an unknown location shift, and the paper's own
summary is that "non-Gaussianity alone therefore does not create bias.  However,
an unknown nonzero noise mean shifts that conditional mean, and a quadratic data
term alone cannot separate the physical signal from an unknown location shift."

EPIBE replaces that term with the negative log-likelihood of a *learned* scalar
residual density, one per cell, "to model residual shapes without prescribing a
named parametric family".  This package is that density:

:mod:`~pibe.ebm.support`
    :math:`\mathcal{R}_k = [-\bar\rho_k, \bar\rho_k]` from Assumption 3.
:mod:`~pibe.ebm.energy`
    The gauge-normalized, bounded, Lipschitz-controlled energy (47)-(48).
:mod:`~pibe.ebm.quadrature`
    The fixed one-dimensional rule for :math:`Z_k`, Eq. (59) / Remark 4.
:mod:`~pibe.ebm.density`
    The density (49), its likelihood (53), and its moments (54).

The cells, the schedule and the bank that use them live in
:mod:`pibe.core.energy_bank` and :mod:`pibe.training.epibe_trainer`.
"""

from pibe.ebm.density import DensityMoments, ScalarEBM
from pibe.ebm.energy import EnergyNetwork
from pibe.ebm.quadrature import CompositeGaussLegendre
from pibe.ebm.support import residual_radii, resolve_radii

__all__ = [
    "CompositeGaussLegendre",
    "DensityMoments",
    "EnergyNetwork",
    "ScalarEBM",
    "residual_radii",
    "resolve_radii",
]
