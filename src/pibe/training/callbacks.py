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

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "History":
        """Rebuild a history from :meth:`to_dict` (used when resuming)."""
        return cls(
            iterations=[IterationRecord(**r) for r in payload.get("iterations", [])],
            validation=list(payload.get("validation", [])),
        )

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


def save_training_state(
    path: Path | str,
    bank: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any,
    cell_index: int,
    iteration: int,
    mode: str,
    history: "History",
    generator: torch.Generator | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Write everything needed to resume training exactly where it stopped.

    Cluster jobs are preempted and time-limited, so a long run must be able to
    continue rather than restart.  Saving only the weights is not enough: Adam's
    moments and the LR schedule's position materially affect the trajectory, and
    the minibatch RNG must continue rather than replay.

    Written to a temporary file and renamed, so a job killed mid-write leaves
    the previous checkpoint intact rather than a truncated one.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": bank.state_dict(),
        "optimizer": None if optimizer is None else optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "cell_index": int(cell_index),
        "iteration": int(iteration),
        "mode": mode,
        "history": history.to_dict(),
        "generator": None if generator is None else generator.get_state(),
        "torch_rng": torch.get_rng_state(),
        "metadata": metadata or {},
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_training_state(path: Path | str) -> dict[str, Any] | None:
    """Read a resume checkpoint, or ``None`` if there is none to read."""
    path = Path(path)
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu", weights_only=False)
