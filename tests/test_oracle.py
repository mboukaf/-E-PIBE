r"""The oracle comparison: scoring the true solution on the training objective."""

from __future__ import annotations

import pytest
import torch

from pibe.config import RunConfig
from pibe.eval.oracle import compare_to_oracle, oracle_losses
from pibe.experiment import build_experiment


def small_experiment(noise_sigma: float = 0.01):
    return build_experiment(
        RunConfig.from_dict(
            {
                "system": {"name": "sin_chain", "kwargs": {"n": 3}},
                "data": {
                    "n_trajectories": 8,
                    "n_samples": 24,
                    "horizon": 3.0,
                    "noise_sigma": noise_sigma,
                    "seed": 1,
                },
                "basis": {"q": 5, "degree": 3},
                "architecture": {
                    "latent_dim": 8,
                    "encoder_hidden": [16],
                    "decoder_hidden": [16, 16],
                    "head_hidden": [16],
                },
                "training": {
                    "n_total": 4,
                    "n_par": 2,
                    "n_collocation": 20,
                    "batch_size": 4,
                    "log_every": 0,
                },
                "device": "cpu",
            }
        )
    )


def test_physics_terms_vanish_at_the_truth() -> None:
    """Every residual is identically zero on the exact solution."""
    experiment = small_experiment()
    losses = oracle_losses(
        experiment.system,
        experiment.basis,
        experiment.disturbance,
        experiment.train_data,
        experiment.t_coll,
        lam=experiment.config.training.lam,
    )
    assert set(losses) == set(experiment.bank.cell_indices)
    for k, loss in losses.items():
        assert float(loss.physics) < 1e-16, (
            f"cell {k}: physics loss {float(loss.physics):.3e} should vanish at the truth"
        )


def test_consistency_terms_vanish_at_the_truth() -> None:
    """Downstream data terms compare the truth against itself."""
    experiment = small_experiment()
    losses = oracle_losses(
        experiment.system,
        experiment.basis,
        experiment.disturbance,
        experiment.train_data,
        experiment.t_coll,
        lam=experiment.config.training.lam,
    )
    for k, loss in losses.items():
        if k > 2:
            assert float(loss.data) < 1e-20, (
                f"cell {k}'s consistency term should be exactly zero at the truth"
            )


def test_first_cell_data_term_equals_the_noise_variance() -> None:
    r"""At the truth, :math:`\hat x_1 = x_1` and :math:`y = x_1 + \omega`.

    So cell 2's data term is the empirical variance of the realized noise ---
    a floor no honest estimator can beat.
    """
    experiment = small_experiment(noise_sigma=0.05)
    losses = oracle_losses(
        experiment.system,
        experiment.basis,
        experiment.disturbance,
        experiment.train_data,
        experiment.t_coll,
        lam=experiment.config.training.lam,
    )
    realized = experiment.train_data.y - experiment.train_data.x[..., 0]
    assert float(losses[2].data) == pytest.approx(
        float((realized**2).mean()), rel=1e-9
    )
    # ... and it tracks the truncated law's variance.
    assert float(losses[2].data) == pytest.approx(
        experiment.noise.variance, rel=0.3
    )


def test_noise_free_case_has_a_zero_oracle_loss() -> None:
    """With no measurement noise the exact solution attains a zero objective."""
    experiment = small_experiment(noise_sigma=0.0)
    losses = oracle_losses(
        experiment.system,
        experiment.basis,
        experiment.disturbance,
        experiment.train_data,
        experiment.t_coll,
        lam=experiment.config.training.lam,
    )
    for k, loss in losses.items():
        assert float(loss.local) < 1e-16, f"cell {k} local loss {float(loss.local):.3e}"


def test_untrained_bank_is_diagnosed_as_an_optimization_failure() -> None:
    """A randomly initialized bank must score worse than the truth."""
    experiment = small_experiment()
    comparison = compare_to_oracle(
        bank=experiment.bank,
        disturbance=experiment.disturbance,
        data=experiment.train_data,
        t_coll=experiment.t_coll,
        lam=experiment.config.training.lam,
        noise_floor=experiment.noise.variance,
    )
    assert comparison.truth_scores_better
    assert "optimization failure" in comparison.verdict
    assert comparison.oracle_total < comparison.trained_total
    assert "L_Tot @ truth" in comparison.summary()
