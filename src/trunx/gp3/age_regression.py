"""Estimate stand age from mean DBH using power-law regression.

The models are fitted on the ICP plots with a stand age from Etzold et al. (referenced
to 2000 and shifted to each survey year in `prepare_icp_tree_data`).
"""

import logging
import os

import numpy as np
import polars as pl

from trunx.config import clean_data_folder

logger = logging.getLogger(__name__)

_MIN_SAMPLES = 10


def _aggregate_training_data(plot_df: pl.DataFrame) -> pl.DataFrame:
    """Select one row per (plot_id, specie, survey_year) with mean DBH and stand age.

    The fit is on stands, not trees: trees of one stand share its age whatever their DBH.
    """
    return plot_df.select(
        "plot_id",
        "specie",
        "survey_year",
        pl.col("dbh_mean").alias("mean_dbh"),
        pl.col("stand_age").alias("age"),
    ).filter((pl.col("mean_dbh") > 0) & (pl.col("age") > 0))


def fit_models(plot_df: pl.DataFrame | None = None) -> dict[str, tuple[float, float]]:
    """Fit per-species power-law regression models (age = a * DBH^b) on stands.

    Uses log-log ordinary least squares: log(age) = log(a) + b * log(DBH), with DBH the
    stand's mean DBH.

    Parameters
    ----------
    plot_df : pl.DataFrame | None
        ICP plot-level data with `plot_id`, `specie`, `survey_year`, `dbh_mean` and
        `stand_age` columns; read from `icp_plot_data.parquet` when not given.

    Returns
    -------
    dict[str, tuple[float, float]]
        Coefficients (a, b) per species with at least `_MIN_SAMPLES` stands.
    """
    if plot_df is None:
        plot_df = pl.read_parquet(os.path.join(clean_data_folder, "icp_plot_data.parquet"))
    train_df = _aggregate_training_data(plot_df)
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
    plot_df: pl.DataFrame | None = None,
    models: dict[str, tuple[float, float]] | None = None,
) -> dict[str, float]:
    """Estimate the stand age of each species of a plot at its first survey.

    Parameters
    ----------
    plot_id : str
        ICP plot id, e.g. "04.1605".
    plot_df : pl.DataFrame | None
        ICP plot-level data; read from `icp_plot_data.parquet` when not given.
    models : dict[str, tuple[float, float]] | None
        Models from `fit_models`; fitted on `plot_df` when not given.

    Returns
    -------
    dict[str, float]
        Estimated age per species with a model.
    """
    if plot_df is None:
        plot_df = pl.read_parquet(os.path.join(clean_data_folder, "icp_plot_data.parquet"))

    if models is None:
        models = fit_models(plot_df)

    first_surveys = (
        plot_df.filter(pl.col("plot_id") == plot_id)
        .sort("survey_year")
        .group_by("specie", maintain_order=True)
        .first()
        .drop_nulls("dbh_mean")
    )
    if first_surveys.is_empty():
        raise ValueError(f"No data found for plot_id '{plot_id}'")

    estimates: dict[str, float] = {}
    for specie, dbh_mean in first_surveys.select("specie", "dbh_mean").iter_rows():
        if specie not in models:
            logger.warning("No model available for species '%s', skipping", specie)
            continue

        a, b = models[specie]
        age = predict_age_from_dbh(np.array([dbh_mean]), a, b)
        estimates[specie] = age
        logger.info("Plot '%s', %s: estimated age = %.1f years", plot_id, specie, age)

    return estimates


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    plots = pl.read_parquet(os.path.join(clean_data_folder, "icp_plot_data.parquet"))
    fitted_models = fit_models(plots)

    plot_id = "04.1605"
    ages = estimate_plot_age(plot_id, plot_df=plots, models=fitted_models)
    print(f"\nEstimated ages for plot {plot_id}:")
    for sp, age in ages.items():
        print(f"  {sp}: {age:.1f} years")
