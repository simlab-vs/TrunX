"""Estimate stand age from DBH observations using power-law regression.

Note: This module is based on the data received from Sophia Etzold. We
need to determine when this age was determined, i.e. as of which year, because
the age of the trees appears to be constant over time with the current data.

"""

import logging
import os

import numpy as np
import polars as pl

from trunx.config import clean_data_folder

logger = logging.getLogger(__name__)

_MIN_SAMPLES = 10


def _aggregate_training_data(icp_df: pl.DataFrame) -> pl.DataFrame:
    """Aggregate ICP data to one row per (plot_id, specie) with mean DBH and age."""
    return (
        icp_df.filter(pl.col("soph_avg_age").is_not_null())
        .group_by(["tree_id", "specie"])
        .agg(
            pl.col("diameter_end").mean().alias("mean_dbh"),
            pl.col("soph_avg_age").mean().alias("age"),
        )
        .drop_nulls()
    )


def fit_models(icp_df: pl.DataFrame) -> dict[str, tuple[float, float]]:
    """Fit per-species power-law regression models (age = a * DBH^b).

    Uses log-log ordinary least squares: log(age) = log(a) + b * log(DBH).
    """
    train_df = _aggregate_training_data(icp_df)
    models: dict[str, tuple[float, float]] = {}

    for specie in sorted(train_df["specie"].unique().to_list()):
        subset = train_df.filter(pl.col("specie") == specie)
        if subset.height < _MIN_SAMPLES:
            logger.warning("Skipping '%s': only %d training samples", specie, subset.height)
            continue

        dbh = subset["mean_dbh"].to_numpy()
        age = subset["age"].to_numpy()

        coeffs = np.polyfit(np.log(dbh), np.log(age), 1)
        b, log_a = float(coeffs[0]), float(coeffs[1])
        a = float(np.exp(log_a))

        models[specie] = (a, b)
        logger.info(
            "Fitted model for '%s': age = %.4f * DBH^%.4f (n=%d)", specie, a, b, subset.height
        )

    return models


def predict_age_from_dbh(dbh_values: np.ndarray, a: float, b: float) -> float:
    """Predict stand age from DBH values using power-law coefficients."""
    mean_dbh = float(np.mean(dbh_values))
    return float(a * mean_dbh**b)


def estimate_plot_age(
    plot_id: str,
    icp_df: pl.DataFrame | None = None,
    models: dict[str, tuple[float, float]] | None = None,
) -> dict[str, float]:
    """Estimate stand age for a given plot from its DBH observations."""
    if icp_df is None:
        icp_df = pl.read_parquet(os.path.join(clean_data_folder, "icp_level2_cleaned.parquet"))

    if models is None:
        models = fit_models(icp_df)

    plot_df = icp_df.filter(pl.col("plot_id") == plot_id)
    if plot_df.is_empty():
        raise ValueError(f"No data found for plot_id '{plot_id}'")

    estimates: dict[str, float] = {}
    for specie in sorted(plot_df["specie"].unique().to_list()):
        if specie not in models:
            logger.warning("No model available for species '%s', skipping", specie)
            continue

        dbh = plot_df.filter(pl.col("specie") == specie)["diameter_end"].drop_nulls().to_numpy()
        if dbh.size == 0:
            continue

        a, b = models[specie]
        age = predict_age_from_dbh(dbh, a, b)
        estimates[specie] = age
        logger.info("Plot '%s', %s: estimated age = %.1f years", plot_id, specie, age)

    return estimates


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    icp = pl.read_parquet(os.path.join(clean_data_folder, "icp_level2_cleaned.parquet"))
    fitted_models = fit_models(icp)

    plot_id = "50.0013"
    ages = estimate_plot_age(plot_id, icp_df=icp, models=fitted_models)
    print(f"\nEstimated ages for plot {plot_id}:")
    for sp, age in ages.items():
        print(f"  {sp}: {age:.1f} years")
