r"""The Physics-Informed Bank of Estimators.

A bank is "a sequence of estimators cascaded together, each one taking as input
the concatenated outputs of the previous ones" (Section 2).  Cells are indexed
``k = 2, ..., n+1``: cells ``2..n`` estimate one state coordinate and one
parameter each, and cell ``n+1`` estimates the disturbance coefficients.

Trajectory-level input, Eq. (15)
--------------------------------
.. math::

    \mathbf{U}^\ell_{k-1} := \operatorname{col}\bigl(
        \mathbf{Y}^\ell, \widehat{\mathbf{X}}^{2,\ell}_2, \dots,
        \widehat{\mathbf{X}}^{k-1,\ell}_{k-1},
        \hat\theta^{2,\ell}_1, \dots, \hat\theta^{k-1,\ell}_{k-2}\bigr),

with empty sequences omitted, so that :math:`\mathbf{U}^\ell_1 =
\mathbf{Y}^\ell`.  All blocks are sampled on the data grid, Eqs. (13)-(14).

Gradient policy
---------------
Algorithm 1 runs two regimes, and the difference between them is entirely a
matter of what carries a gradient:

``LOCAL`` (``i < N_par``)
    Cells ``2..k-1`` are frozen and evaluated under ``no_grad``, so
    :math:`\mathbf{U}_{k-1}` is built "from detached upstream outputs" and only
    :math:`\Theta^k_{PINN}` is updated, against :math:`\mathcal{L}^k_{Loc}`.

``GLOBAL`` (``i >= N_par``)
    Cells :math:`\{\Phi^m\}_{m=2}^k` are evaluated "sequentially without
    detaching the upstream outputs" and every parameter block is updated
    against :math:`\mathcal{L}^k_{Tot}`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from collections.abc import Iterator

import torch
from torch import Tensor, nn

from pibe.basis.base import DisturbanceBasis
from pibe.config import ArchitectureConfig
from pibe.core.cell import CellOutput, EstimatorCell
from pibe.core.losses import CellLoss, data_loss, local_loss, physics_loss
from pibe.core.residuals import (
    disturbance_estimate,
    final_residual,
    state_residual,
    upstream_states,
)
from pibe.systems.base import TriangularSystem


class Mode(Enum):
    """Which of Algorithm 1's two regimes a forward pass is running under."""

    LOCAL = "local"
    GLOBAL = "global"


@dataclass(frozen=True)
class Estimates:
    r"""The reported estimates of the trained bank.

    Attributes
    ----------
    x
        :math:`\hat{\boldsymbol x} = (\hat x^2_1, \hat x^2_2, \dots, \hat x^n_n)`
        on the data grid, shape ``(B, N, n)``.  Per Section 3.1 the reported
        estimates are :math:`\hat x_1 := \hat x^2_1` and
        :math:`\hat x_k := \hat x^k_k`; the final cell's
        :math:`\hat x^{n+1}_n` is *not* included, being auxiliary.
    theta
        :math:`\hat\theta = (\hat\theta^2_1, \dots, \hat\theta^n_{n-1})`,
        shape ``(B, n - 1)``.
    a
        :math:`\hat a`, shape ``(B, q)``.
    d
        :math:`\hat d(t) = \Gamma_q(t)^\top \hat a` on the data grid, Eq. (41),
        shape ``(B, N)``.
    x_aux
        The auxiliary final-cell trajectory :math:`\hat x^{n+1}_n`, shape
        ``(B, N)``, kept for diagnostics.
    """

    x: Tensor
    theta: Tensor
    a: Tensor
    d: Tensor
    x_aux: Tensor


class EstimatorBank(nn.Module):
    r"""The cascade :math:`\{\Phi^k\}_{k=2}^{n+1}`.

    Parameters
    ----------
    system
        Supplies ``n``, the nonlinearities :math:`f_j`, and the admissible sets.
    basis
        The disturbance basis :math:`\Gamma_q`.
    n_samples
        ``N``; fixes every encoder's input width.
    coefficient_bounds
        The box :math:`\mathcal{A}`, shape ``(q, 2)``.
    architecture
        Network widths, shared by all cells.
    """

    def __init__(
        self,
        system: TriangularSystem,
        basis: DisturbanceBasis,
        n_samples: int,
        coefficient_bounds: Tensor,
        architecture: ArchitectureConfig | None = None,
    ) -> None:
        super().__init__()
        self.system = system
        self.basis = basis
        self.n_samples = int(n_samples)
        self.architecture = architecture or ArchitectureConfig()

        coefficient_bounds = torch.as_tensor(coefficient_bounds)
        if coefficient_bounds.shape != (basis.q, 2):
            raise ValueError(
                f"coefficient_bounds must have shape ({basis.q}, 2), "
                f"got {tuple(coefficient_bounds.shape)}"
            )

        state_scales, theta_scales = system.reference_scales()
        self.cells = nn.ModuleDict(
            {
                str(k): EstimatorCell(
                    cell_index=k,
                    state_dim=system.n,
                    n_samples=self.n_samples,
                    state_bounds=system.state_bounds,
                    theta_bounds=system.theta_bounds,
                    state_scales=state_scales,
                    theta_scales=theta_scales,
                    coefficient_bounds=coefficient_bounds,
                    t_start=basis.t_start,
                    t_end=basis.t_end,
                    architecture=self.architecture,
                )
                for k in range(2, system.n + 2)
            }
        )

    # ------------------------------------------------------------------
    # structure
    # ------------------------------------------------------------------

    @property
    def state_dim(self) -> int:
        """``n``."""
        return self.system.n

    @property
    def final_index(self) -> int:
        """``n + 1``, the index of the disturbance cell."""
        return self.system.n + 1

    @property
    def cell_indices(self) -> range:
        """``range(2, n + 2)``."""
        return range(2, self.final_index + 1)

    def cell(self, index: int) -> EstimatorCell:
        """The cell :math:`\\Phi^k`."""
        if index not in self.cell_indices:
            raise KeyError(
                f"cell index {index} out of range; expected 2..{self.final_index}"
            )
        return self.cells[str(index)]  # type: ignore[return-value]

    def cell_parameters(self, index: int) -> Iterator[nn.Parameter]:
        """Parameters of a single cell."""
        return self.cell(index).parameters()

    def parameters_upto(self, index: int) -> Iterator[nn.Parameter]:
        r"""Parameters of cells ``2..index``, i.e. the block :math:`\mathfrak{W}_{2:k}`."""
        for k in range(2, index + 1):
            yield from self.cell(k).parameters()

    def set_trainable_upto(self, index: int, trainable: bool) -> None:
        """Freeze or unfreeze cells ``2..index``."""
        for k in range(2, index + 1):
            self.cell(k).set_trainable(trainable)

    # ------------------------------------------------------------------
    # trajectory input, Eq. (15)
    # ------------------------------------------------------------------

    def assemble_input(
        self, cell_index: int, y: Tensor, outputs: dict[int, CellOutput]
    ) -> Tensor:
        r"""Build :math:`\mathbf{U}_{k-1}` from the measurement and upstream outputs.

        Parameters
        ----------
        cell_index
            ``k``.
        y
            Sampled measurements :math:`\mathbf{Y}`, shape ``(B, N)``.
        outputs
            Outputs of cells ``2..k-1``.

        Returns
        -------
        Tensor
            Shape ``(B, N(k-1) + (k-2))``.
        """
        if y.ndim != 2 or y.shape[1] != self.n_samples:
            raise ValueError(
                f"measurements must have shape (B, {self.n_samples}), "
                f"got {tuple(y.shape)}"
            )
        blocks: list[Tensor] = [y]
        # Sampled upstream state estimates X_2, ..., X_{k-1}.
        for j in range(2, cell_index):
            state = outputs[j].x_new_data
            if state is None:  # pragma: no cover - the final cell is never upstream
                raise ValueError(f"cell {j} produced no new coordinate")
            blocks.append(state)
        # Upstream parameter estimates theta_1, ..., theta_{k-2}.
        for j in range(2, cell_index):
            blocks.append(outputs[j].head)

        u = torch.cat(blocks, dim=-1)
        expected = self.cell(cell_index).input_dim
        if u.shape[-1] != expected:
            raise AssertionError(
                f"assembled U_{cell_index - 1} has width {u.shape[-1]}, expected {expected}"
            )
        return u

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(
        self,
        target_cell: int,
        y: Tensor,
        t_data: Tensor,
        t_coll: Tensor,
        mode: Mode = Mode.GLOBAL,
        need_derivative: bool = True,
        create_graph: bool = True,
    ) -> dict[int, CellOutput]:
        """Run cells ``2..target_cell`` sequentially.

        Under :attr:`Mode.LOCAL` the upstream cells are evaluated without
        gradients and their outputs detached; under :attr:`Mode.GLOBAL` the
        whole chain stays differentiable.

        Parameters
        ----------
        need_derivative
            Compute :math:`\\partial_t` of each scored cell's reconstruction.
            Required for any physics residual; set ``False`` for pure state /
            parameter readout, which is what lets :meth:`estimate` run under
            ``no_grad``.
        create_graph
            Retain the derivative's graph.  Set ``False`` to evaluate losses
            without building a backward graph --- note this still requires
            grad *mode* to be enabled, so validation must not be wrapped in
            ``torch.no_grad()``.

        Returns
        -------
        dict[int, CellOutput]
            Outputs keyed by cell index.
        """
        if target_cell not in self.cell_indices:
            raise KeyError(
                f"target cell {target_cell} out of range; expected 2..{self.final_index}"
            )

        outputs: dict[int, CellOutput] = {}
        for k in range(2, target_cell + 1):
            u = self.assemble_input(k, y, outputs)
            cell = self.cell(k)
            upstream_and_local = mode is Mode.LOCAL and k != target_cell
            if upstream_and_local:
                # Frozen upstream cell: values only, no graph, fully detached.
                with torch.no_grad():
                    output = cell(
                        u, t_data, t_coll,
                        need_derivative=False, create_graph=False,
                    )
                outputs[k] = output.detached()
            else:
                outputs[k] = cell(
                    u, t_data, t_coll,
                    need_derivative=need_derivative,
                    create_graph=create_graph and need_derivative,
                )
        return outputs

    # ------------------------------------------------------------------
    # losses
    # ------------------------------------------------------------------

    def cell_loss(
        self,
        cell_index: int,
        y: Tensor,
        t_coll: Tensor,
        outputs: dict[int, CellOutput],
        lam: float,
    ) -> CellLoss:
        r"""The local objective :math:`\mathcal{L}^k_{Loc}` of one cell.

        Eqs. (22), (28) and (37), with the data term selected by the uniform
        rule: cell ``k``'s reconstruction of coordinate ``k-1`` is compared
        against the measurement when ``k = 2`` and against cell ``k-1``'s new
        coordinate otherwise.
        """
        output = outputs[cell_index]
        if cell_index == 2:
            target = y
        else:
            target = outputs[cell_index - 1].x_new_data
            if target is None:  # pragma: no cover - unreachable by construction
                raise ValueError(f"cell {cell_index - 1} produced no new coordinate")

        data = data_loss(output.x_prev_data, target)

        if cell_index == self.final_index:
            residual = final_residual(
                self.system, self.basis, output, outputs, t_coll
            )
        else:
            residual = state_residual(self.system, cell_index, output, outputs)
        return local_loss(data, physics_loss(residual), lam)

    def losses(
        self,
        target_cell: int,
        y: Tensor,
        t_coll: Tensor,
        outputs: dict[int, CellOutput],
        lam: float,
        mode: Mode = Mode.GLOBAL,
    ) -> dict[int, CellLoss]:
        """Local losses for the cells the current regime scores.

        :attr:`Mode.LOCAL` scores only the target cell; :attr:`Mode.GLOBAL`
        scores every cell ``2..target_cell``, as the global loss (27) requires.
        """
        indices = (
            [target_cell] if mode is Mode.LOCAL else list(range(2, target_cell + 1))
        )
        return {
            k: self.cell_loss(k, y, t_coll, outputs, lam) for k in indices
        }

    # ------------------------------------------------------------------
    # inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def estimate(self, y: Tensor, t_data: Tensor, t_coll: Tensor) -> Estimates:
        r"""Run the full bank and collect the reported estimates.

        The state estimate assembles :math:`\hat x^2_1` and
        :math:`\{\hat x^j_j\}_{j=2}^n` per the reporting convention of
        Section 3.1, and the disturbance follows Eq. (41).
        """
        outputs = self.forward(
            self.final_index,
            y,
            t_data,
            t_coll,
            mode=Mode.GLOBAL,
            need_derivative=False,
        )
        states = upstream_states(outputs, self.state_dim, grid="data")
        theta = torch.cat(
            [outputs[k].head for k in range(2, self.state_dim + 1)], dim=-1
        )
        a_hat = outputs[self.final_index].head
        return Estimates(
            x=torch.stack(states, dim=-1),
            theta=theta,
            a=a_hat,
            d=disturbance_estimate(self.basis, a_hat, t_data),
            x_aux=outputs[self.final_index].x_prev_data,
        )

    def extra_repr(self) -> str:  # pragma: no cover - trivial
        return (
            f"n={self.state_dim}, cells=2..{self.final_index}, "
            f"N={self.n_samples}, q={self.basis.q}"
        )
