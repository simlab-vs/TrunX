"""Checks for loading parameter priors, in particular literature bound overrides."""

from pathlib import Path

import polars as pl
import pytest

from trunx.gp3.bayesiancalibrations.load_files import load_priors_from_file


@pytest.fixture
def param_bound_file(tmp_path: Path) -> str:
    """A minimal param_bound parquet file with two calibrated parameters."""
    file_path = tmp_path / "param_bound.parquet"
    pl.DataFrame(
        {
            "param_name": ["Tmax", "alphaCx"],
            "min": [25.0, 0.02],
            "max": [40.0, 0.09],
            "default": [36.0, 0.07],
        }
    ).write_parquet(file_path)
    return str(file_path)


def test_bound_overrides_replace_the_loaded_range(param_bound_file: str) -> None:
    priors = load_priors_from_file(param_bound_file, bound_overrides={"Tmax": (25.0, 45.0)})

    assert priors["Tmax"] == (25.0, 45.0)
    assert priors["alphaCx"] == (0.02, 0.09)


def test_bound_overrides_for_parameters_not_loaded_are_ignored(param_bound_file: str) -> None:
    priors = load_priors_from_file(
        param_bound_file, param_names=["alphaCx"], bound_overrides={"Tmax": (25.0, 45.0)}
    )

    assert set(priors) == {"alphaCx"}


def test_no_overrides_leaves_the_loaded_range_unchanged(param_bound_file: str) -> None:
    priors = load_priors_from_file(param_bound_file)

    assert priors["Tmax"] == (25.0, 40.0)
