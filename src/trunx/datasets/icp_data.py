"""Prepare ICP Forests Level II plot-level data for 3PG modelling."""

import glob
import logging
import math
import os
from io import StringIO

import pandas as pd
import polars as pl
import requests

from trunx.config import clean_data_folder, data_folder
from trunx.gp3.allometrics import CoefficientsDict, add_allometric_columns, load_forrester_eq3

logger = logging.getLogger(__name__)

_ICP_FOLDER = str(os.path.join(data_folder, "raw/ICP"))

SPECIES_TARGET: list[str] = [
    "Picea abies",
    "Pinus sylvestris",
    "Fagus sylvatica",
    # "Quercus petraea",
    # "Quercus robur",
]

_FORRESTER_EQ3: CoefficientsDict = load_forrester_eq3(species=SPECIES_TARGET)
_AGE_REFERENCE_YEAR = 2000


def _find_csv(pattern: str) -> str:
    """Find the most recent CSV file matching a glob pattern."""
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file found matching: {pattern}")
    # Take the last match, which should be the most recent one based on the naming convention
    return matches[-1]


def _make_plot_id(df: pl.DataFrame) -> pl.DataFrame:
    """Add ``plot_id`` column from ``code_country`` and ``code_plot``."""
    return df.with_columns(
        (
            pl.col("code_country").cast(pl.Utf8).str.zfill(2)
            + "."
            + pl.col("code_plot").cast(pl.Utf8).str.zfill(4)
        ).alias("plot_id")
    )


def _load_dictionaries() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Fetch species and country lookup tables from ICP-Forests website."""
    species_html = requests.get(
        "https://icp-forests.org/documentation/Dictionaries/d_tree_spec.html"
    ).text
    species_df = pl.from_pandas(pd.read_html(StringIO(species_html))[0]).rename(
        {"CODE": "code_tree_species", "DESCRIPTION": "species"}
    )

    country_html = requests.get(
        "https://icp-forests.org/documentation/Dictionaries/d_country.html"
    ).text
    country_df = pl.from_pandas(pd.read_html(StringIO(country_html))[0]).rename(
        {"CODE": "code_country", "LIB_COUNTRY": "country"}
    )
    return species_df, country_df


def _load_plots() -> pl.DataFrame:
    """Load site-level plot info from the most recent ``si_plt.csv``."""
    path = _find_csv(os.path.join(_ICP_FOLDER, "595_si_*/si_plt.csv"))
    return (
        pl.read_csv(path, separator=";")
        .pipe(_make_plot_id)
        .rename(
            {
                "latitude": "lat",
                "longitude": "lon",
                "code_altitude": "plot_altitude",
                "plot_size": "plot_size_ha",
            }
        )
        .select("plot_id", "lat", "lon", "plot_altitude", "plot_size_ha")
    )


def _load_trees(
    species_df: pl.DataFrame,
    country_df: pl.DataFrame,
) -> pl.DataFrame:
    """Load alive tree measurements from ``gr_ipm.csv`` for target species."""
    path = _find_csv(os.path.join(_ICP_FOLDER, "595_gr_*/gr_ipm.csv"))
    return (
        pl.read_csv(path, separator=";", ignore_errors=True)
        .with_columns(pl.col("date_assessment").str.to_datetime().alias("date"))
        .pipe(_make_plot_id)
        .join(species_df.select(["code_tree_species", "species"]), on="code_tree_species")
        .join(country_df.select(["code_country", "country"]), on="code_country")
        .filter(pl.col("species").is_in(SPECIES_TARGET))
        .drop_nulls(subset="diameter")
        .filter(pl.col("diameter").gt(0))
        .filter(
            pl.col("code_diameter_qc").cast(pl.Int64, strict=False).is_null()
            | ~pl.col("code_diameter_qc").cast(pl.Int64, strict=False).gt(2)
        )
        .filter(
            pl.col("code_diameter").cast(pl.Int64, strict=False).is_null()
            | ~pl.col("code_diameter").cast(pl.Int64, strict=False).is_in([7])
        )
        .filter(
            pl.col("code_removal").cast(pl.Int64, strict=False).is_null()
            | ~pl.col("code_removal").cast(pl.Int64, strict=False).gt(10)
        )
        .with_columns(
            pl.col("diameter").alias("dbh_cm"),
            pl.col("height").alias("height_m"),
        )
        .select("plot_id", "species", "survey_year", "date", "dbh_cm", "height_m")
    )


def _filter_single_species(trees: pl.DataFrame) -> pl.DataFrame:
    """Keep only (plot, census year) observations with a single target species."""
    single_species = (
        trees.group_by("plot_id", "survey_year")
        .agg(pl.col("species").n_unique().alias("n_species"))
        .filter(pl.col("n_species") == 1)
        .select("plot_id", "survey_year")
    )
    return trees.join(single_species, on=["plot_id", "survey_year"], how="inner")


def _aggregate_per_plot(trees: pl.DataFrame, plots: pl.DataFrame) -> pl.DataFrame:
    """Compute allometric quantities and aggregate to plot-level per-ha values."""
    trees = add_allometric_columns(trees, _FORRESTER_EQ3)

    per_plot = (
        trees.sort("date")
        .group_by("plot_id", "species", "survey_year")
        .agg(
            pl.first("date"),
            pl.len().alias("n_count"),
            pl.col("dbh_cm").mean(),
            pl.col("height_m").mean(),
            pl.col("allo_sb_kg").sum().alias("plot_sb_kg"),
            pl.col("allo_fb_kg").sum().alias("plot_fb_kg"),
            pl.col("allo_rb_kg").sum().alias("plot_rb_kg"),
            pl.col("allo_la_m2").sum().alias("plot_la_m2"),
            (math.pi * pl.col("dbh_cm").pow(2) / 40000.0).sum().alias("plot_ba_m2"),
        )
    )

    return (
        per_plot.join(plots, on="plot_id", how="inner")
        .with_columns(
            (pl.col("n_count") / pl.col("plot_size_ha")).alias("n_stems"),
            (pl.col("plot_sb_kg") / pl.col("plot_size_ha") / 1000.0).alias("biom_stem"),
            (pl.col("plot_fb_kg") / pl.col("plot_size_ha") / 1000.0).alias("biom_foliage"),
            (pl.col("plot_rb_kg") / pl.col("plot_size_ha") / 1000.0).alias("biom_root"),
            (pl.col("plot_la_m2") / (pl.col("plot_size_ha") * 10000.0)).alias("lai"),
            (pl.col("plot_ba_m2") / pl.col("plot_size_ha")).alias("basal_area"),
            pl.col("date").dt.year().alias("year"),
        )
        .select(
            "species",
            "plot_id",
            "date",
            "year",
            "lat",
            "lon",
            "plot_altitude",
            "n_stems",
            "dbh_cm",
            "height_m",
            "biom_stem",
            "biom_foliage",
            "biom_root",
            "lai",
            "basal_area",
        )
        .sort(["species", "plot_id", "date"])
    )


def _load_plot_meta() -> pl.DataFrame:
    """Load plot metadata from Etzold and average stand age per plot.

    Note: This age is as of the year 2000
    """
    path = os.path.join(_ICP_FOLDER, "icpf/01_raw/ICP-Forests-Plots_Meta.csv")
    return (
        pl.read_csv(path)
        .with_columns(plot_id=pl.col("plotid").cast(pl.Utf8).replace("NA", None))
        .drop_nulls(subset="plot_id")
        .with_columns(
            plot_id=pl.col("plot_id").str.slice(0, 2) + "." + pl.col("plot_id").str.slice(2),
            yr_first=pl.col("yr_first").replace("NA", None).cast(pl.Int32),
            yr_last=pl.col("yr_last").replace("NA", None).cast(pl.Int32),
            age=pl.col("age").replace("NA", None).cast(pl.Float32),
        )
        .drop_nulls(subset=["yr_first", "yr_last"])
        .group_by("plot_id")
        .agg(pl.mean("age").alias("avg_age"))
    )


def prepare_icp_data(output_path: str | None = None) -> pl.DataFrame:
    """Load, clean, and aggregate ICP Level II data for 3PG calibration.

    Parameters
    ----------
    output_path : str | None
        Parquet path to write the result. Defaults to
        ``clean_data_folder/icp_cleaned.parquet``.

    Returns
    -------
    pl.DataFrame
        Columns: ``species``, ``plot_id``, ``date``, ``year``,
        ``lat``, ``lon``, ``plot_altitude``,
        ``n_stems`` (ha⁻¹), ``dbh_cm``, ``height_m``,
        ``biom_stem``, ``biom_foliage``, ``biom_root`` (all t ha⁻¹),
        ``lai`` (m² m⁻²), ``basal_area`` (m² ha⁻¹),
        ``age`` (years, estimated stand age at census year).
    """
    if output_path is None:
        output_path = str(os.path.join(clean_data_folder, "icp_cleaned.parquet"))

    species_df, country_df = _load_dictionaries()
    logger.info("Loaded species and country dictionaries")

    plots = _load_plots()
    logger.info("Loaded %d plots", plots.height)

    trees = _load_trees(species_df, country_df)
    logger.info(
        "Loaded %d tree records across %d species",
        trees.height,
        trees["species"].n_unique(),
    )

    trees = _filter_single_species(trees)
    logger.info(
        "After single-species filter: %d records across %d plots",
        trees.height,
        trees["plot_id"].n_unique(),
    )

    result = _aggregate_per_plot(trees, plots)
    logger.info(
        "Aggregated to %d plot-year observations across %d plots",
        result.height,
        result["plot_id"].n_unique(),
    )

    plot_meta = _load_plot_meta()
    result = (
        result.join(plot_meta, on="plot_id", how="left")
        .with_columns((pl.col("avg_age") + (pl.col("year") - _AGE_REFERENCE_YEAR)).alias("age"))
        .drop("avg_age")
    )
    logger.info("Joined plot metadata (%d plots with age data)", plot_meta.height)

    result.write_parquet(output_path)
    logger.info("Saved to %s", output_path)
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    df = prepare_icp_data()
    print(df.head())
    print("\nPer-species plot counts:")
    print(df.group_by("species").agg(pl.col("plot_id").n_unique().alias("n_plots")))
