"""Configuration parsing, validation and round-tripping."""

from __future__ import annotations

import pytest

from pibe.config import (
    ArchitectureConfig,
    BasisConfig,
    DataConfig,
    RunConfig,
    TrainingConfig,
)


def test_defaults_are_valid() -> None:
    config = RunConfig()
    assert config.system.name
    assert 0 < config.training.n_par < config.training.n_total


def test_nested_dictionaries_build_nested_dataclasses() -> None:
    config = RunConfig.from_dict(
        {
            "system": {"name": "sin_chain", "kwargs": {"n": 4, "kappa": 0.1}},
            "data": {"n_samples": 64, "horizon": 7.0},
            "training": {"n_total": 10, "n_par": 4},
        }
    )
    assert isinstance(config.data, DataConfig)
    assert isinstance(config.training, TrainingConfig)
    assert isinstance(config.basis, BasisConfig)
    assert isinstance(config.architecture, ArchitectureConfig)
    assert config.system.kwargs == {"n": 4, "kappa": 0.1}
    assert config.data.horizon == 7.0


def test_unknown_keys_are_rejected() -> None:
    """A typo in a config file must fail loudly rather than be ignored."""
    with pytest.raises(ValueError, match="unknown key"):
        RunConfig.from_dict({"data": {"n_sample": 64}})
    with pytest.raises(ValueError, match="unknown key"):
        RunConfig.from_dict({"nonexistent_section": {}})


def test_yaml_round_trip(tmp_path) -> None:
    config = RunConfig.from_dict(
        {
            "system": {"name": "sin_chain", "kwargs": {"n": 3}},
            "training": {"n_total": 50, "n_par": 30, "lam": 2.5},
        }
    )
    path = tmp_path / "run.yaml"
    config.to_yaml(path)
    restored = RunConfig.from_yaml(path)
    assert restored.to_dict() == config.to_dict()
    assert restored.training.lam == 2.5


def test_architecture_normalizes_hidden_widths_to_tuples() -> None:
    """YAML gives lists; the dataclass must store hashable tuples."""
    architecture = ArchitectureConfig(encoder_hidden=[8, 8], decoder_hidden=[4])
    assert architecture.encoder_hidden == (8, 8)
    assert architecture.decoder_hidden == (4,)


@pytest.mark.parametrize(
    ("section", "payload", "message"),
    [
        (BasisConfig, {"coeff_lo": 1.0, "coeff_hi": 0.0}, "coeff_lo < coeff_hi"),
        (DataConfig, {"n_samples": 1}, "n_samples"),
        (DataConfig, {"horizon": -1.0}, "horizon"),
        (DataConfig, {"noise_sigma": -0.5}, "noise_sigma"),
        (TrainingConfig, {"lam": 0.0}, "lambda"),
        (TrainingConfig, {"n_collocation": 1}, "n_collocation"),
    ],
)
def test_invalid_values_are_rejected(section, payload, message) -> None:
    with pytest.raises(ValueError, match=message):
        section(**payload)
