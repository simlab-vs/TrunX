"""Checks for loading parameter priors, in particular literature bound overrides."""

from pathlib import Path

import pandas as pd
import polars as pl
import pytest

from trunx.gp3.bayesiancalibrations.load_files import (
    literature_bound_overrides,
    load_priors_from_file,
)


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


@pytest.fixture
def literature_dir(tmp_path: Path) -> str:
    """A minimal literature dir with per-species Tmax and MaxAge bounds.

    Tmax matches for every species (as in the real Forrester table); MaxAge
    is only defined for Picea abies and Fagus sylvatica, and the two disagree
    (as in the real Trotsiuk table).
    """
    literature_dir = tmp_path / "literature"
    literature_dir.mkdir()

    pl.DataFrame(
        {
            "parameter": ["Tmax", "Tmax", "Tmax"],
            "species": ["Picea abies", "Fagus sylvatica", "Pinus sylvestris"],
            "min": [30.0, 30.0, 30.0],
            "max": [45.0, 45.0, 45.0],
        }
    ).write_parquet(literature_dir / "literature_params_forrester_forrester.parquet")

    pl.DataFrame(
        {
            "parameter": ["MaxAge", "MaxAge"],
            "species": ["Picea abies", "Fagus sylvatica"],
            "min": [200.0, 200.0],
            "max": [500.0, 400.0],
        }
    ).write_parquet(literature_dir / "literature_params_trotsiuk.parquet")

    return str(literature_dir)


def _species_file(tmp_path: Path, species_names: list[str]) -> str:
    """A minimal Excel file with a `species` sheet, for `literature_bound_overrides`."""
    file_path = tmp_path / "plot_data.xlsx"
    pd.DataFrame({"species": species_names}).to_excel(file_path, sheet_name="species", index=False)
    return str(file_path)


def test_literature_bound_overrides_for_a_species_with_both_bounds(
    tmp_path: Path, literature_dir: str
) -> None:
    file_path = _species_file(tmp_path, ["Fagus sylvatica"])

    overrides = literature_bound_overrides(file_path, literature_dir=literature_dir)

    assert overrides == {"Tmax": (30.0, 45.0), "MaxAge": (200.0, 400.0)}


def test_literature_bound_overrides_skips_a_species_missing_from_the_table(
    tmp_path: Path, literature_dir: str
) -> None:
    file_path = _species_file(tmp_path, ["Pinus sylvestris"])

    overrides = literature_bound_overrides(file_path, literature_dir=literature_dir)

    assert overrides == {"Tmax": (30.0, 45.0)}


def test_literature_bound_overrides_skips_disagreeing_species(
    tmp_path: Path, literature_dir: str
) -> None:
    file_path = _species_file(tmp_path, ["Picea abies", "Fagus sylvatica"])

    overrides = literature_bound_overrides(file_path, literature_dir=literature_dir)

    assert overrides == {"Tmax": (30.0, 45.0)}
