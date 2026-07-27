"""Create data input for 3PG model using ICP data."""

import logging
import os

import jax.numpy as jnp
import pandas as pd
import polars as pl

from trunx.config import clean_data_folder, threepg_data_folder
from trunx.datasets.ICP_weather_data import prepare_icp_weather_data
from trunx.gp3.age_regression import fit_models, predict_age_from_dbh
from trunx.gp3.allometrics import add_allometric_columns, load_forrester_eq3
from trunx.gp3.weather_processing import (
    create_weather_input,
    fill_weather_with_era5,
    weather_summary,
)

logger = logging.getLogger(__name__)

# Forrester et al. (2017) eq. 3 coefficients for all species with complete data.
# Loaded once at import time to avoid repeated Excel reads.
_FORRESTER_COEFFS = load_forrester_eq3()


def dms_to_decimal(dms):
    """Convert DMS packed as ±DDMMSS or ±DDDMMSS to decimal degrees."""
    sign = -1 if str(dms).startswith("-") else 1
    dms = abs(int(dms))

    degrees = dms // 10000
    minutes = (dms % 10000) // 100
    seconds = dms % 100

    val = sign * (degrees + minutes / 60 + seconds / 3600)
    return val


def create_input_params(icp_df):
    """Create parameter data input for 3PG model."""
    input_params = {}
    params_df = pl.read_excel(
        os.path.join(threepg_data_folder, "data.input.xlsx"), sheet_name="parameters"
    )
    input_params["parameter"] = params_df["parameter"].to_list()
    for specie in icp_df["specie"].unique().to_list():
        if specie in params_df.schema:
            input_params[specie] = params_df[specie]

    input_params_df = pl.DataFrame(input_params)

    if len(input_params_df.schema) < 2:
        raise ValueError("No parameter values found for species in this location")

    return input_params_df


def create_species_data(
    icp_df: pl.DataFrame,
    models: dict[str, tuple[float, float]] | None = None,
) -> tuple[pl.DataFrame, int]:
    """Create species data input for 3PG model.

    Parameters
    ----------
    icp_df : pl.DataFrame
        ICP tree-level data for the target plot(s).
    models : dict[str, tuple[float, float]] | None
        Per-species power-law age-vs-DBH models from
        :func:`trunx.gp3.age_regression.fit_models`, used to estimate the
        planted date when direct age observations are unavailable. If None,
        fits a model from `icp_df` alone, which is unreliable for a single
        plot with few DBH observations — pass a model fit across many plots
        when calling this for one plot at a time.
    """
    default_species_df = (
        pl.read_excel(os.path.join(threepg_data_folder, "data.input.xlsx"), sheet_name="species")
        .filter(pl.col("species").is_in(icp_df.select("specie").unique().to_series().to_list()))
        .select("species", "fertility")
    )

    if models is None:
        models = fit_models(icp_df)

    species_df = (
        _allometric_observations(icp_df)
        .sort("specie", "date")
        .with_columns(pl.col("date").dt.year().alias("year"))
        .filter(pl.col("specie").is_in(default_species_df.select("species").to_series()))
    )

    common_start = (
        species_df.group_by("specie")
        .agg([pl.col("year").min().alias("min_year")])
        .select([pl.col("min_year").max().alias("start_year")])
        .sort(["start_year"])
    )

    start_year = common_start["start_year"].item()

    species_df = species_df.filter(pl.col("year") == start_year)
    species_df = species_df.with_columns(planted=pl.lit(f"{start_year}-01"), fertility=pl.lit(0.5))
    for specie in species_df["specie"].unique().to_list():
        _raw_age = (
            icp_df.filter((pl.col("date").dt.year() == start_year) & (pl.col("specie") == specie))
            .select(pl.col("soph_avg_age").cast(pl.Float64).mean())
            .item()
        )
        direct_age: float | None = float(_raw_age) if _raw_age is not None else None

        if direct_age is not None:
            planting_year = int(start_year - round(direct_age))
            species_df = species_df.with_columns(
                pl.when(pl.col("specie") == specie)
                .then(pl.lit(f"{planting_year}-01"))
                .otherwise(pl.col("planted"))
                .alias("planted")
            )
            print(
                "Direct age for '%s': %d (age %.1f yr at %d)",
                specie,
                planting_year,
                direct_age,
                start_year,
            )
            continue
        if specie not in models:
            print("No age model for '%s', keeping template planted date", specie)
            continue
        dbh = icp_df.filter(
            (pl.col("date").dt.year() == start_year) & (pl.col("specie") == specie)
        )["dbh_cm"].to_numpy()
        a, b = models[specie]
        estimated_age = predict_age_from_dbh(dbh, a, b)
        planting_year = int(start_year - round(estimated_age))
        species_df = species_df.with_columns(
            pl.when(pl.col("specie") == specie)
            .then(pl.lit(f"{planting_year}-01"))
            .otherwise(pl.col("planted"))
            .alias("planted")
        )
        print(
            "Estimated planting year for '%s': %d (age %.1f yr at %d)",
            specie,
            planting_year,
            estimated_age,
            start_year,
        )

    species_df = species_df.rename(
        {"specie": "species", "WS": "biom_stem", "WF": "biom_foliage", "WR": "biom_root"}
    ).select(
        "species", "planted", "fertility", "stems_n", "biom_stem", "biom_root", "biom_foliage"
    )
    return species_df, start_year


def create_site_data(icp_df, weather_df):
    """Create site data input for 3PG model."""
    site_df = pl.read_excel(
        os.path.join(threepg_data_folder, "data.input.xlsx"), sheet_name="site"
    )

    latitude_value = icp_df["Lat"][0]
    altitude_value = icp_df["plot_altitude"][0]

    # Add as constant columns
    site_df = site_df.with_columns(
        [pl.lit(latitude_value).alias("latitude"), pl.lit(altitude_value).alias("altitude")]
    )

    min_year = weather_df["year"].min()
    min_month = weather_df.filter(pl.col("year") == min_year).select("month").min().item()
    site_df = site_df.with_columns([pl.lit(f"{min_year}-{min_month:02d}").alias("from")])

    max_year = weather_df["year"].max()
    max_month = weather_df.filter(pl.col("year") == max_year).select("month").max().item()
    site_df = site_df.with_columns([pl.lit(f"{max_year}-{max_month:02d}").alias("to")])

    return site_df


def _allometric_observations(icp_df: pl.DataFrame) -> pl.DataFrame:
    """Aggregate per-tree Forrester biomass to survey-date observations.

    Parameters
    ----------
    icp_df : pl.DataFrame
        Tree-level ICP data with ``dbh_cm`` (cm) and ``specie`` columns.

    Returns
    -------
    pl.DataFrame
        One row per (specie, month_year) with columns ``WS``, ``WF``, ``WR``
        (t ha⁻¹) and ``LAI`` (m² m⁻²).  Rows whose species has no Forrester
        equation 3 coefficients carry null values.

    """
    return (
        add_allometric_columns(
            icp_df,
            _FORRESTER_COEFFS,
            dbh_col="dbh_cm",
            species_col="specie",
        )
        .group_by(["specie", "date"])
        .agg(
            (pl.col("allo_sb_kg").sum() / 1000.0).alias("WS"),
            (pl.col("allo_fb_kg").sum() / 1000.0).alias("WF"),
            (pl.col("allo_rb_kg").sum() / 1000.0).alias("WR"),
            (pl.col("allo_la_m2").sum() / 10000.0).alias("LAI"),
            (pl.len().alias("num_trees") / pl.col("plot_size_ha").mean()).alias("stems_n"),
        )
        .with_columns(
            pl.col("date").dt.strftime("%m-%Y").alias("month_year"),
        )
    )


def create_observation_data(
    plot_id: str,
    icp_df: pl.DataFrame,
    start_year: int,
) -> pl.DataFrame:
    """Create observed data input for 3PG model.

    Parameters
    ----------
    plot_id : str
        Plot identifier used to filter GPP data.
    icp_df : pl.DataFrame
        ICP tree-level data filtered to the target plot and species.
    start_year : int
        First year to include in the output.
    """
    if icp_df.is_empty():
        logger.warning("No ICP data for plot_id %s — returning empty observations", plot_id)
        return pl.DataFrame()

    dbh_df = (
        icp_df.group_by(["specie", "date"])
        .agg(pl.col("dbh_cm").mean().alias("DBH"))
        .sort("date")
        .with_columns(
            pl.col("date").dt.year().cast(pl.Int64).alias("year"),
            pl.col("date").dt.month().cast(pl.Int64).alias("month"),
            pl.col("date").dt.strftime("%m-%Y").alias("month_year"),
        )
    )

    gpp_raw = pl.read_csv(os.path.join(clean_data_folder, "GOSIF_GPP_icp.csv"))
    gpp_plot = gpp_raw.filter(pl.col("plot_id") == float(plot_id))

    if gpp_plot.is_empty():
        logger.warning("No GPP data for plot_id %s — GPP column will be null", plot_id)
        specie_gpp = pl.DataFrame()
    else:
        gpp_plot = (
            gpp_plot.with_columns(pl.datetime(pl.col("year"), pl.col("month"), 1).alias("Date"))
            .sort(["year", "month"])
            .with_columns(pl.col("Date").dt.month_end())
            .with_columns(pl.col("Date").dt.strftime("%m-%Y").alias("month_year"))
            .drop(["plot_id"])
            .filter(pl.col("Date").dt.year() >= start_year)
        )

        species = dbh_df["specie"].unique().to_list()
        if not species:
            logger.warning("No species in DBH data for plot_id %s", plot_id)
            specie_gpp = pl.DataFrame()
        else:
            specie_gpp = pl.concat(
                [gpp_plot.with_columns(pl.lit(sp).alias("specie")) for sp in species]
            ).drop_nulls(subset=["GPP"])

    biom_df = _allometric_observations(icp_df)

    if specie_gpp.is_empty():
        # No GPP: build from DBH side only
        observed_data = (
            dbh_df.join(biom_df, on=["specie", "month_year"], how="left")
            .with_columns(pl.lit(None).cast(pl.Float64).alias("GPP"))
            .with_columns(pl.date(pl.col("year"), pl.col("month"), 1).dt.month_end().alias("Date"))
        )
    else:
        # Full join so DBH-only rows (census dates) and GPP-only rows are kept.
        # Both sides carry month/year — coalesce to avoid nulls for one-sided rows.
        observed_data = (
            specie_gpp.join(dbh_df, on=["specie", "month_year"], how="full", coalesce=True)
            .with_columns(
                pl.coalesce("year", "year_right").alias("year"),
                pl.coalesce("month", "month_right").alias("month"),
            )
            .drop(["year_right", "month_right"], strict=False)
            .join(biom_df, on=["specie", "month_year"], how="left")
            .with_columns(pl.date(pl.col("year"), pl.col("month"), 1).dt.month_end().alias("Date"))
        )

    observed_data = (
        observed_data.select(
            ["specie", "month", "year", "Date", "GPP", "DBH", "WS", "WF", "WR", "LAI"]
        )
        .drop_nulls(subset=["specie"])
        .sort("Date")
    )

    return observed_data


def create_input_data(input_data_file, plot_id):
    """Create input data for 3PG model."""
    # ICP weather data
    # raw_file_path = os.path.join(icp_raw_data_folder, "595_mm_20260227091917/mm_mem.csv")
    # processor = prepare_icp_weather_data(raw_file_path)
    # df = processor.clean_data()

    df = pl.read_parquet(os.path.join(clean_data_folder, "ICP_weather_data.parquet"))

    # ICP data — load full dataset first for cross-plot age model fitting
    icp_full = pl.read_parquet(os.path.join(clean_data_folder, "icp_tree_data.parquet"))
    icp_df = icp_full.filter(pl.col("plot_id") == plot_id)

    icp_df = icp_df.filter(
        pl.col("specie").is_in(
            list(
                [
                    "Picea abies",
                    "Pinus sylvestris",
                    "Fagus sylvatica",
                    "Quercus robur",
                    "Quercus petraea",
                ]
            )
        )
    )
    print("Unique dates in ICP data:", icp_df.select("date").unique().to_series().to_list())

    icp_df = icp_df.with_columns(
        pl.col("plot_latitude").map_elements(dms_to_decimal, return_dtype=pl.Float64).alias("Lat"),
        pl.col("plot_longitude")
        .map_elements(dms_to_decimal, return_dtype=pl.Float64)
        .alias("Lon"),
    )

    logging.info("Pre-processed ICP data for plot_id: %s", plot_id)
    # Weather data
    miss_months, weather_df = create_weather_input(df, plot_id=plot_id)
    logging.info("Pre-processed weather data for plot_id: %s", plot_id)

    # Species data
    input_species_df, start_year = create_species_data(icp_df)
    logging.info("Created species data for plot_id: %s", plot_id)

    # Parameter data
    input_params_df = create_input_params(icp_df)
    logging.info("Created parameter data for plot_id: %s", plot_id)

    miss_months, weather_df = fill_weather_with_era5(weather_df, plot_id, start_year)
    logger.info("Filled weather data from start year %d for plot_id: %s", start_year, plot_id)
    # Site data
    input_site_df = create_site_data(icp_df, weather_df)
    logger.info("Created site data for plot_id: %s", plot_id)

    icp_df = icp_df.filter(pl.col("specie").is_in(input_species_df["species"].to_list()))

    observed_data = create_observation_data(plot_id, icp_df, start_year)

    if len(miss_months) == 0:
        with pd.ExcelWriter(input_data_file, engine="openpyxl") as writer:
            weather_df.to_pandas().to_excel(writer, sheet_name="climate", index=False)
            input_params_df.to_pandas().to_excel(writer, sheet_name="parameters", index=False)
            input_species_df.to_pandas().to_excel(writer, sheet_name="species", index=False)
            input_site_df.to_pandas().to_excel(writer, sheet_name="site", index=False)
            pd.DataFrame().to_excel(writer, sheet_name="thinning", index=False)
            pd.DataFrame().to_excel(writer, sheet_name="sizeDist", index=False)
            observed_data.to_pandas().to_excel(writer, sheet_name="observed", index=False)
    else:
        print("The weather data is not complete and need processing to fill missing data.")
        summary_wdf = weather_summary(weather_df)
        print(summary_wdf)

    return miss_months, observed_data.to_pandas()


if __name__ == "__main__":
    plot_id = "59.0004"
    file_path = os.path.join(threepg_data_folder, "S_weather_data.xlsx")
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Deleted: {file_path}")
    miss_months, observed_data = create_input_data(file_path, plot_id)

    print(observed_data.head())
