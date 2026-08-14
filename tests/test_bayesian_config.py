"""Checks for shared Bayesian calibration configuration."""

from pathlib import Path

import polars as pl
import pytest

from trunx.gp3.bayesiancalibrations.bayesian_config import DIAGNOSTIC_ONLY_ERROR_NAMES
from trunx.gp3.bayesiancalibrations.load_files import load_priors_from_file


def test_diagnostic_only_error_names_cover_dbh_ba_height() -> None:
    assert frozenset({"err_DBH", "err_BA", "err_Height"}) == DIAGNOSTIC_ONLY_ERROR_NAMES


@pytest.fixture
def param_bound_file(tmp_path: Path) -> str:
    """A param_bound file mixing a diagnostic-only and a calibrated error prior."""
    file_path = tmp_path / "param_bound.parquet"
    pl.DataFrame(
        {
            "param_name": ["alphaCx", "err_WS", "err_DBH", "err_BA", "err_Height"],
            "min": [0.02, 0.3, 0.1, 0.1, 0.1],
            "max": [0.09, 30.0, 15.0, 30.0, 10.0],
            "default": [0.07, 5.0, 3.0, 6.0, 10.0],
        }
    ).write_parquet(file_path)
    return str(file_path)


def test_excluding_diagnostic_only_errors_from_loaded_priors(param_bound_file: str) -> None:
    """Mirrors the pop pattern used in run_map_analysis/run_pymc_analysis/run_bayesian_for_plot."""
    priors = load_priors_from_file(param_bound_file)
    for error_name in DIAGNOSTIC_ONLY_ERROR_NAMES:
        priors.pop(error_name, None)

    assert set(priors) == {"alphaCx", "err_WS"}
