r"""Resuming a preempted run must continue it, not restart it."""

from __future__ import annotations

import torch

from pibe.config import RunConfig
from pibe.experiment import build_experiment
from pibe.training.callbacks import load_training_state
from pibe.training.trainer import PIBETrainer
from pibe.utils.seeding import make_generator, seed_everything


def small(**training):
    payload = {
        "system": {"name": "sin_chain", "kwargs": {"n": 3}},
        "data": {"n_trajectories": 8, "n_samples": 24, "horizon": 3.0,
                 "noise_sigma": 0.01, "seed": 1},
        "basis": {"q": 5, "degree": 3},
        "architecture": {"latent_dim": 8, "encoder_hidden": [16],
                         "decoder_hidden": [16, 16], "head_hidden": [16]},
        "training": {"n_total": 12, "n_par": 6, "n_collocation": 16,
                     "batch_size": 4, "log_every": 0, **training},
        "device": "cpu", "seed": 0,
    }
    return build_experiment(RunConfig.from_dict(payload))


def make_trainer(experiment, path=None):
    return PIBETrainer(
        bank=experiment.bank,
        train_data=experiment.train_data,
        t_coll=experiment.t_coll,
        config=experiment.config.training,
        val_data=experiment.val_data,
        generator=make_generator(experiment.config.training.seed),
        checkpoint_path=path,
    )


def test_checkpoint_records_position_and_state(tmp_path) -> None:
    path = tmp_path / "state.pt"
    experiment = small(checkpoint_every=4)
    make_trainer(experiment, path).train_cell(2)

    state = load_training_state(path)
    assert state is not None
    assert state["cell_index"] == 2
    # Last write is the largest multiple of 4 that is <= n_total.
    assert state["iteration"] == 11
    for key in ("state_dict", "optimizer", "history", "generator", "torch_rng"):
        assert state[key] is not None, f"{key} missing from the checkpoint"


def test_resume_reproduces_uninterrupted_training(tmp_path) -> None:
    """Interrupting and resuming must land on the same weights as running through.

    This is the property that makes preemption safe.  It requires restoring the
    optimizer moments, the schedule position and both RNG streams --- weights
    alone would silently diverge.
    """
    seed_everything(0)
    uninterrupted = small(checkpoint_every=0)
    make_trainer(uninterrupted).train_cell(2)
    reference = [p.detach().clone() for p in uninterrupted.bank.cell_parameters(2)]

    # Same run, stopped after 7 iterations and resumed from the checkpoint.
    seed_everything(0)
    path = tmp_path / "state.pt"
    first = small(checkpoint_every=7)
    trainer = make_trainer(first, path)
    trainer.train_cell(2)          # writes at iteration 6 and 13 -> only 6 lands
    state = load_training_state(path)
    assert state["iteration"] == 6

    seed_everything(0)
    second = small(checkpoint_every=0)
    resumed = make_trainer(second, path)
    start_cell, start_iteration = resumed.resume()
    assert (start_cell, start_iteration) == (2, 7)
    resumed.train_cell(2, start_iteration=start_iteration)

    for a, b in zip(reference, resumed.bank.cell_parameters(2)):
        assert torch.allclose(a, b, atol=1e-9), (
            "resumed weights differ from uninterrupted training; "
            "optimizer state, schedule position or RNG was not restored"
        )


def test_resume_without_a_checkpoint_starts_from_scratch(tmp_path) -> None:
    experiment = small()
    trainer = make_trainer(experiment, tmp_path / "absent.pt")
    assert trainer.resume() == (2, 0)


def test_resume_skips_completed_cells(tmp_path) -> None:
    """A checkpoint taken during cell 3 must not retrain cell 2."""
    path = tmp_path / "state.pt"
    experiment = small(checkpoint_every=4)
    trainer = make_trainer(experiment, path)
    trainer.train_cell(2)
    trainer.train_cell(3)

    fresh = make_trainer(small(checkpoint_every=4), path)
    cell, iteration = fresh.resume()
    assert cell == 3, "resume must continue at the cell that was in progress"
    assert iteration == 12


def test_history_survives_a_resume(tmp_path) -> None:
    path = tmp_path / "state.pt"
    experiment = small(checkpoint_every=4, log_every=2)
    make_trainer(experiment, path).train_cell(2)
    logged = len(load_training_state(path)["history"]["iterations"])
    assert logged > 0

    fresh = make_trainer(small(checkpoint_every=4, log_every=2), path)
    fresh.resume()
    assert len(fresh.history.iterations) == logged
