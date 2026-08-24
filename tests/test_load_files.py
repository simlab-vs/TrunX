"""Checks for loading parameter priors, in particular literature bound overrides."""

from pathlib import Path

import pandas as pd
import polars as pl
import pytest

from trunx.gp3.bayesiancalibrations.load_files import (
    literature_bound_overrides,
    load_param_defaults_from_file,
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


def _param_bound_xlsx(tmp_path: Path, n_species_columns: int = 1) -> str:
    """A minimal Excel file with `param_bound`, `error_param`, and `parameters`.

    `param_bound` carries only bounds (no `default`): `Tmax`/`alphaCx` are
    free (both min and max set), `wSx1000` is fixed (neither set). Physiology
    defaults live only in `parameters`, the single source of truth for a
    parameter's runtime/seed value.
    """
    file_path = tmp_path / "plot_data.xlsx"
    param_bound = pd.DataFrame(
        {
            "param_name": ["Tmax", "alphaCx", "wSx1000"],
            "min": [25.0, 0.02, None],
            "max": [40.0, 0.09, None],
        }
    )
    error_param = pd.DataFrame(
        {"param_name": ["err_WS"], "default": [0.5], "min": [0.05], "max": [2.0]}
    )
    parameters = pd.DataFrame({"parameter": ["Tmax", "alphaCx", "wSx1000"]})
    for i in range(n_species_columns):
        parameters[f"species_{i}"] = [32.0, 0.055, 300.0]

    with pd.ExcelWriter(file_path) as writer:
        param_bound.to_excel(writer, sheet_name="param_bound", index=False)
        error_param.to_excel(writer, sheet_name="error_param", index=False)
        parameters.to_excel(writer, sheet_name="parameters", index=False)
    return str(file_path)


def test_defaults_come_from_the_parameters_sheet_for_physiology_params(
    tmp_path: Path,
) -> None:
    file_path = _param_bound_xlsx(tmp_path)

    defaults = load_param_defaults_from_file(file_path, ["Tmax", "alphaCx", "wSx1000", "err_WS"])

    assert defaults == {"Tmax": 32.0, "alphaCx": 0.055, "wSx1000": 300.0, "err_WS": 0.5}


def test_priors_are_unaffected_by_the_parameters_sheet_split(tmp_path: Path) -> None:
    file_path = _param_bound_xlsx(tmp_path)

    priors = load_priors_from_file(file_path)

    assert priors == {"Tmax": (25.0, 40.0), "alphaCx": (0.02, 0.09), "err_WS": (0.05, 2.0)}


def test_a_multi_species_parameters_sheet_raises(tmp_path: Path) -> None:
    file_path = _param_bound_xlsx(tmp_path, n_species_columns=2)

    with pytest.raises(ValueError, match="single-species"):
        load_priors_from_file(file_path)
