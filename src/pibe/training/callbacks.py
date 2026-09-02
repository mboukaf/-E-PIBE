"""Training history and checkpointing."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn


@dataclass
class IterationRecord:
    """One optimizer step's diagnostics."""

    cell: int
    iteration: int
    mode: str
    objective: float
    data: float
    physics: float
    local: float
    grad_norm: float


@dataclass
class History:
    """Records every logged iteration and the per-cell validation summaries."""

    iterations: list[IterationRecord] = field(default_factory=list)
    validation: list[dict[str, Any]] = field(default_factory=list)

    def log_iteration(self, record: IterationRecord) -> None:
        self.iterations.append(record)

    def log_validation(self, entry: dict[str, Any]) -> None:
        self.validation.append(entry)

    def for_cell(self, cell: int) -> list[IterationRecord]:
        return [record for record in self.iterations if record.cell == cell]

    def to_dict(self) -> dict[str, Any]:
        return {
            "iterations": [asdict(record) for record in self.iterations],
            "validation": self.validation,
        }

    def save(self, path: Path | str) -> None:
        """Write the history as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as handle:
            json.dump(self.to_dict(), handle, indent=2)


def save_checkpoint(
    path: Path | str,
    bank: nn.Module,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save the bank's state dict alongside arbitrary metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": bank.state_dict(), "metadata": metadata or {}}, path
    )


def load_checkpoint(path: Path | str, bank: nn.Module) -> dict[str, Any]:
    """Restore a bank's parameters in place; returns the stored metadata."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    bank.load_state_dict(payload["state_dict"])
    return payload.get("metadata", {})
