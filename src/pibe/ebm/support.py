r"""Residual supports :math:`\mathcal{R}_k = [-\bar\rho_k, \bar\rho_k]`, Assumption 3.

Assumption 3 is specific about where the radius comes from, and about where it
may *not* come from::

    The radius is fixed before EBM activation using the compact
    decoder-output and parameter ranges defined above together with the known
    noise-support bound; the warm-up residual range may be used only as a
    diagnostic.

So the radius is an a priori quantity, derived from the admissible sets the
estimator is already constrained to, not measured from a partially trained
model.  Reading it off the warm-up residuals would make the support depend on
the very fit whose likelihood it defines.

The bounds
----------
Cell :math:`k` compares its reconstruction of coordinate :math:`k-1` against the
target supplied from upstream, so each residual is a difference of two
quantities with known ranges:

===================  ==========================================  ==========================================
cell                 residual                                    :math:`\bar\rho_k`
===================  ==========================================  ==========================================
:math:`k=2`          :math:`\varepsilon_2 = y - \hat x^2_1`      :math:`(\overline{x}_1 - \underline{x}_1) + \bar w`
:math:`3 \le k\le n` :math:`\hat x^{k-1}_{k-1}-\hat x^k_{k-1}`   :math:`\overline{x}_{k-1} - \underline{x}_{k-1}`
:math:`k=n+1`        :math:`\hat x^n_n - \hat x^{n+1}_n`         :math:`\overline{x}_n - \underline{x}_n`
===================  ==========================================  ==========================================

The first row is the one that carries the measurement: :math:`y = x_1 + \omega`
with :math:`x_1` in its admissible interval and :math:`\|\omega\|_\infty \le
\bar w`, against a decoder output reparameterized into the same interval.  The
downstream rows are differences of two decoder outputs for the same coordinate,
which is why Assumption 3 says they "also account for admissible upstream
discrepancies" --- taking the full box width covers exactly that.

On looseness
------------
These bounds are guaranteed but generous: at :math:`\sigma = 0.002` the true
first-cell residual is of order :math:`0.006` while the box width is
:math:`5`, a ratio near :math:`10^3`.  Assumption 3's radius is the honest
default and is what :func:`residual_radii` returns, but a density concentrated
on :math:`0.1\%` of its support needs large Assumption 3 constants to represent
(see :mod:`pibe.ebm.energy`), so a run may legitimately narrow the support by
hand.  Doing so is a modelling choice that must still contain every realized
residual, which is why
:meth:`~pibe.ebm.density.ScalarEBM.out_of_support_fraction` is monitored
throughout training rather than assumed.
"""

from __future__ import annotations

from pibe.systems.base import TriangularSystem


def residual_radii(
    system: TriangularSystem, noise_bound: float = 0.0
) -> dict[int, float]:
    r"""A priori :math:`\bar\rho_k` for every cell :math:`k = 2,\dots,n+1`.

    Parameters
    ----------
    system
        Supplies the admissible state intervals :math:`\mathcal{X}_j`, which are
        also the decoder output ranges by construction.
    noise_bound
        :math:`\bar w`, the compact support of the measurement noise.  Enters
        the first cell only, the measurement appearing nowhere else.

    Returns
    -------
    dict[int, float]
        Radius per cell index.
    """
    if noise_bound < 0:
        raise ValueError(f"the noise bound must be non-negative, got {noise_bound}")
    bounds = system.state_bounds
    widths = [float(bounds[j, 1] - bounds[j, 0]) for j in range(system.n)]

    radii = {2: widths[0] + float(noise_bound)}
    for k in range(3, system.n + 1):
        radii[k] = widths[k - 2]
    # The final cell reconstructs x_n a second time; its residual compares the
    # two reconstructions of that same coordinate.
    radii[system.n + 1] = widths[system.n - 1]
    return radii


def resolve_radii(
    system: TriangularSystem,
    noise_bound: float = 0.0,
    override: float | list[float] | None = None,
) -> dict[int, float]:
    """The radii a run will actually use, honouring an explicit override.

    ``override`` may be a single radius for every cell or one per cell in index
    order ``2..n+1``.  ``None`` keeps the a priori bounds of
    :func:`residual_radii`.
    """
    a_priori = residual_radii(system, noise_bound)
    if override is None:
        return a_priori
    indices = sorted(a_priori)
    if isinstance(override, (int, float)):
        values = [float(override)] * len(indices)
    else:
        values = [float(v) for v in override]
        if len(values) != len(indices):
            raise ValueError(
                f"expected {len(indices)} radii for cells {indices}, got {len(values)}"
            )
    if any(v <= 0 for v in values):
        raise ValueError("every residual radius must be positive")
    return dict(zip(indices, values))
