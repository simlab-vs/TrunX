"""Prepare monthly deposition inputs for the 3PG model."""

import datetime
import os
import warnings

import jax.numpy as jnp
import polars as pl

from trunx.config import clean_data_folder
from trunx.gp3.model_inputs import DepositionData


def load_monthly_deposition(
    deposition_path: str | None = None,
) -> pl.DataFrame:
    """Load processed monthly deposition data.

    Parameters
    ----------
    deposition_path : str | None, default None
        Path to monthly deposition parquet. Defaults to
        ``data/clean/icp_monthly_deposition.parquet``.

    Returns
    -------
    pl.DataFrame
        Monthly deposition table.
    """
    if deposition_path is None:
        deposition_path = str(os.path.join(clean_data_folder, "icp_monthly_deposition.parquet"))
    return pl.read_parquet(deposition_path)


def dep_range(deposition: pl.DataFrame) -> None:
    """Check whether deposition values are within plausible ranges.

    Parameters
    ----------
    deposition : pl.DataFrame
        Plot-level deposition table with ``dep_n_tot`` and ``dep_s_so4``.
    """
    if deposition.filter(pl.col("dep_n_tot") < 0).height > 0:
        raise ValueError("dep_n_tot has negative values.")

    if deposition.filter(pl.col("dep_n_tot") > 1000).height > 0:
        print("Warning: dep_n_tot outside plausible range (0–1000 kg/ha/month).")

    if deposition.filter(pl.col("dep_s_so4") < 0).height > 0:
        raise ValueError("dep_s_so4 has negative values.")

    if deposition.filter(pl.col("dep_s_so4") > 1000).height > 0:
        print("Warning: dep_s_so4 outside plausible range (0–1000 kg/ha/month).")


def _ensure_deposition_columns(deposition: pl.DataFrame) -> pl.DataFrame:
    """Ensure deposition columns exist and fill missing values with zeros."""
    missing = [col for col in ("dep_n_tot", "dep_s_so4") if col not in deposition.columns]
    if missing:
        warnings.warn(
            "Deposition columns missing for 3PG: "
            f"{missing}; running without deposition effects using zero values.",
            UserWarning,
            stacklevel=2,
        )

    for column in ("dep_n_tot", "dep_s_so4"):
        if column not in deposition.columns:
            deposition = deposition.with_columns(pl.lit(0.0).alias(column))
        else:
            deposition = deposition.with_columns(
                pl.when(pl.col(column).is_null())
                .then(pl.lit(0.0))
                .otherwise(pl.col(column))
                .alias(column)
            )

    return deposition


def prepare_deposition(
    deposition: pl.DataFrame,
    from_: str = "2000-01",
    to: str = "2020-12",
) -> DepositionData:
    """Prepare deposition table for 3-PG simulation.

    Parameters
    ----------
    deposition : pl.DataFrame
        Monthly deposition table with at least ``year`` and ``month``
        columns; ``dep_n_tot`` and ``dep_s_so4`` are optional and will be
        zero-filled when missing.
    from_ : str
        Simulation start month formatted as ``YYYY-MM``.
    to : str
        Simulation end month formatted as ``YYYY-MM``.

    Returns
    -------
    DepositionData
        Deposition arrays aligned to the requested monthly period.
    """
    required = ["year", "month"]
    missing = [c for c in required if c not in deposition.columns]
    if missing:
        raise ValueError(f"Deposition table missing required columns: {missing}")

    from_date = datetime.date.fromisoformat(f"{from_}-01")
    to_date = datetime.date.fromisoformat(f"{to}-01")

    if from_date >= to_date:
        raise ValueError("The start date is later than or equal to the end date")

    deposition = deposition.with_columns(pl.date(pl.col("year"), pl.col("month"), 1).alias("date"))

    plot_df = deposition.filter((pl.col("date") >= from_date) & (pl.col("date") <= to_date)).sort(
        "year", "month"
    )

    if plot_df.is_empty():
        raise ValueError(f"No deposition data found in [{from_}, {to}]")

    has_from = plot_df.filter(pl.col("date") == from_date).height > 0
    has_to = plot_df.filter(pl.col("date") == to_date).height > 0
    if not has_from or not has_to:
        raise ValueError("Requested time period is outside of provided dates in deposition table.")

    plot_df = _ensure_deposition_columns(plot_df)
    plot_df = plot_df.select("year", "month", "dep_n_tot", "dep_s_so4")

    dep_range(plot_df)

    return DepositionData(
        dep_n_tot=jnp.asarray(plot_df["dep_n_tot"].to_numpy(), dtype=float),
        dep_s_so4=jnp.asarray(plot_df["dep_s_so4"].to_numpy(), dtype=float),
    )


def get_deposition_df(
    plot_id: str,
    from_: str,
    to: str,
    deposition: pl.DataFrame | None = None,
    deposition_path: str | None = None,
) -> pl.DataFrame:
    """Return the filtered monthly deposition table as a Polars DataFrame.

    Parameters
    ----------
    plot_id : str
        Plot identifier (format ``CC.PPPP``).
    from_ : str
        Start month formatted as ``YYYY-MM``.
    to : str
        End month formatted as ``YYYY-MM``.
    deposition : pl.DataFrame | None, default None
        Pre-loaded monthly deposition table. Loaded from parquet if omitted.
    deposition_path : str | None, default None
        Optional custom path to the monthly deposition parquet.

    Returns
    -------
    pl.DataFrame
        Rows for ``plot_id`` within ``[from_, to]`` with columns
        ``year``, ``month``, ``dep_n_tot``, and ``dep_s_so4``.
    """
    if deposition is None:
        deposition = load_monthly_deposition(deposition_path=deposition_path)

    from_date = datetime.date.fromisoformat(f"{from_}-01")
    to_date = datetime.date.fromisoformat(f"{to}-01")
    if from_date > to_date:
        raise ValueError("The start date is later than the end date")

    plot_df = (
        deposition.filter(pl.col("plot_id") == plot_id)
        .with_columns(pl.date(pl.col("year"), pl.col("month"), 1).alias("_date"))
        .filter((pl.col("_date") >= from_date) & (pl.col("_date") <= to_date))
        .drop("_date")
        .sort("year", "month")
    )

    if plot_df.is_empty():
        raise ValueError(f"No deposition data found for plot_id {plot_id} in [{from_}, {to}]")

    plot_df = _ensure_deposition_columns(plot_df)

    return plot_df.select(
        pl.col("year").cast(pl.Int64),
        pl.col("month").cast(pl.Int64),
        pl.col("dep_n_tot"),
        pl.col("dep_s_so4"),
    )


if __name__ == "__main__":
    deposition_df = load_monthly_deposition()
    plot_id = "04.1402"

    deposition_df = (
        deposition_df.filter(pl.col("plot_id") == plot_id)
        .sort("year", "month")
        .select("dep_n_tot", "dep_s_so4", "month", "year")
    )
    print(deposition_df.head())
