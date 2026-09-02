"""Algorithm 1: phase schedule, freezing, and an end-to-end training run."""

from __future__ import annotations

import pytest
import torch

from pibe.config import RunConfig, TrainingConfig
from pibe.core.bank import Mode
from pibe.experiment import build_experiment
from pibe.training.phases import apply_phase, mode_for_iteration


# ----------------------------------------------------------------------
# schedule
# ----------------------------------------------------------------------


def test_mode_switches_at_the_cutoff() -> None:
    """``i < N_par`` is local pre-training, the rest is fine-tuning."""
    n_par = 5
    assert [mode_for_iteration(i, n_par) for i in range(4)] == [Mode.LOCAL] * 4
    assert mode_for_iteration(4, n_par) is Mode.LOCAL
    assert mode_for_iteration(5, n_par) is Mode.GLOBAL
    assert mode_for_iteration(9, n_par) is Mode.GLOBAL


def test_training_config_enforces_the_cutoff_inequality() -> None:
    """Algorithm 1's input requires ``0 < N_par < N_tot``."""
    with pytest.raises(ValueError, match="0 < n_par < n_total"):
        TrainingConfig(n_total=100, n_par=100)
    with pytest.raises(ValueError, match="0 < n_par < n_total"):
        TrainingConfig(n_total=100, n_par=0)


# ----------------------------------------------------------------------
# freezing
# ----------------------------------------------------------------------


def small_experiment(**overrides):
    payload = {
        "system": {"name": "sin_chain", "kwargs": {"n": 3}},
        "data": {
            "n_trajectories": 8,
            "n_samples": 24,
            "horizon": 3.0,
            "noise_sigma": 0.01,
            "seed": 1,
        },
        "basis": {"q": 5, "degree": 3},
        "architecture": {
            "latent_dim": 8,
            "encoder_hidden": [32],
            "decoder_hidden": [32, 32],
            "head_hidden": [32],
        },
        "training": {
            "n_total": 8,
            "n_par": 6,
            "n_collocation": 20,
            "batch_size": 4,
            "log_every": 0,
        },
        "device": "cpu",
    }
    for section, values in overrides.items():
        payload[section] = {**payload.get(section, {}), **values}
    return build_experiment(RunConfig.from_dict(payload))


def test_apply_phase_local_freezes_upstream() -> None:
    """Line 6: only the current cell is trainable during pre-training."""
    experiment = small_experiment()
    bank = experiment.bank
    active = apply_phase(bank, target_cell=3, mode=Mode.LOCAL)

    assert all(not p.requires_grad for p in bank.cell_parameters(2))
    assert all(p.requires_grad for p in bank.cell_parameters(3))
    assert all(not p.requires_grad for p in bank.cell_parameters(4))
    assert len(active) == len(list(bank.cell_parameters(3)))


def test_apply_phase_global_unfreezes_the_prefix() -> None:
    """Line 11: cells ``2..k`` become trainable, later cells stay frozen."""
    experiment = small_experiment()
    bank = experiment.bank
    active = apply_phase(bank, target_cell=3, mode=Mode.GLOBAL)

    assert all(p.requires_grad for p in bank.cell_parameters(2))
    assert all(p.requires_grad for p in bank.cell_parameters(3))
    assert all(not p.requires_grad for p in bank.cell_parameters(4))
    assert len(active) == len(list(bank.parameters_upto(3)))


def test_local_phase_leaves_upstream_weights_untouched() -> None:
    """A full local step must not move a single upstream parameter."""
    experiment = small_experiment()
    bank = experiment.bank
    trainer = experiment.make_trainer()

    before = [p.detach().clone() for p in bank.cell_parameters(2)]
    active = apply_phase(bank, target_cell=3, mode=Mode.LOCAL)
    optimizer = torch.optim.Adam(active, lr=1e-2)
    trainer.step(3, Mode.LOCAL, optimizer, active)

    after = list(bank.cell_parameters(2))
    for old, new in zip(before, after):
        assert torch.equal(old, new), "an upstream weight changed during pre-training"


def test_global_phase_moves_upstream_weights() -> None:
    """Fine-tuning must actually reach the upstream cells."""
    experiment = small_experiment()
    bank = experiment.bank
    trainer = experiment.make_trainer()

    before = [p.detach().clone() for p in bank.cell_parameters(2)]
    active = apply_phase(bank, target_cell=3, mode=Mode.GLOBAL)
    optimizer = torch.optim.Adam(active, lr=1e-2)
    trainer.step(3, Mode.GLOBAL, optimizer, active)

    after = list(bank.cell_parameters(2))
    assert any(not torch.equal(old, new) for old, new in zip(before, after)), (
        "no upstream weight moved during end-to-end fine-tuning"
    )


# ----------------------------------------------------------------------
# end to end
# ----------------------------------------------------------------------


def test_training_reduces_the_first_cell_objective() -> None:
    """The measurement fit of cell 2 must improve over a short run."""
    experiment = small_experiment(
        training={"n_total": 220, "n_par": 200, "log_every": 10, "lr_local": 3e-3}
    )
    trainer = experiment.make_trainer()
    trainer.train_cell(2)

    records = trainer.history.for_cell(2)
    assert len(records) >= 5
    start = records[0].local
    end = min(record.local for record in records[-3:])
    assert end < 0.5 * start, (
        f"cell 2's local loss barely moved: {start:.3e} -> {end:.3e}"
    )


def test_full_run_produces_finite_estimates() -> None:
    """Every cell trains and the bank yields usable estimates and ``d_hat``."""
    experiment = small_experiment(
        training={"n_total": 12, "n_par": 8, "log_every": 0}
    )
    trainer = experiment.make_trainer()
    trainer.train()

    estimates = experiment.bank.estimate(
        experiment.val_data.y, experiment.val_data.t, experiment.t_coll
    )
    for name, tensor in [
        ("x", estimates.x),
        ("theta", estimates.theta),
        ("a", estimates.a),
        ("d", estimates.d),
    ]:
        assert torch.isfinite(tensor).all(), f"{name} contains non-finite values"

    assert estimates.d.shape == experiment.val_data.y.shape
    assert len(trainer.history.validation) == len(list(experiment.bank.cell_indices))


def test_validation_does_not_leak_gradients() -> None:
    """``evaluate`` returns detached scalars despite needing grad mode."""
    experiment = small_experiment()
    trainer = experiment.make_trainer()
    loss = trainer.evaluate(2, experiment.val_data)
    assert not loss.local.requires_grad
    assert torch.isfinite(loss.local)
