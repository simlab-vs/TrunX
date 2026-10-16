"""Prepare ICP Forests Level II data."""

import glob
import logging
import math
import os
from io import StringIO

import pandas as pd
import polars as pl
import polars.selectors as cs
import requests

from trunx.config import clean_data_folder, data_folder
from trunx.gp3.allometrics import (
    BIOMASS_COLS,
    CoefficientsDict,
    add_allometric_columns,
    aggregate_per_plot,
    computed_flags,
    dms_to_decimal,
    load_forrester_eq3,
    measured_flag,
    scale_to_hectare,
)

logger = logging.getLogger(__name__)

_ICP_FOLDER = str(os.path.join(data_folder, "raw/ICP"))

SPECIES_TARGET: list[str] = [
    "Picea abies",
    "Pinus sylvestris",
    "Fagus sylvatica",
    "Quercus robur",
    "Quercus petraea",
]

_FORRESTER_EQ3: CoefficientsDict = load_forrester_eq3()
_AGE_REFERENCE_YEAR = 2000
_COUNTRIES_EXCLUDE: list[str] = ["Belgium", "Spain", "Serbia"]

_DEP_NAMES: list[str] = [
    "ph",
    "cond",
    "k",
    "ca",
    "mg",
    "na",
    "n_nh4",
    "cl",
    "n_no3",
    "s_so4",
    "alk",
    "n_tot",
    "doc",
    "al",
    "mn",
    "fe",
    "p_po4",
    "cu",
    "zn",
    "hg",
    "pb",
    "co",
    "mo",
    "ni",
    "cd",
    "s_tot",
    "c_tot",
    "n_org",
    "p_tot",
    "cr",
    "n_no2",
    "hco3",
    "don",
    "n_no3_plus_n_no2",
]

_DEP_NON_CONC: list[str] = ["dep_alk", "dep_ph", "dep_cond"]

# Deposition plausibility limits, beyond which values are treated as unit or
# placeholder errors.
_DEP_MAX_QUANTITY_MM = 1000.0
_DEP_MAX_NS_CONC_MG_L = 100.0
_DEP_MAX_NTOT_DIN_RATIO = 3.0
_DEP_MAX_DON_MG_L = 5.0
# More distinct records than this for one sampler and period means the reported
# dates are broken (e.g. a whole year of samples sharing one period's dates).
_DEP_MAX_RECORDS_PER_SAMPLER_PERIOD = 3

_SOIL_NAMES: list[str] = [
    "ph",
    "cond",
    "k",
    "ca",
    "mg",
    "n_no3",
    "s_so4",
    "alk",
    "al",
    "doc",
    "na",
    "n_nh4",
    "cl",
    "n_tot",
    "fe",
    "mn",
    "al_labile",
    "p",
    "cr",
    "ni",
    "zn",
    "cu",
    "pb",
    "cd",
    "si",
    "n_no2",
    "n_no3_plus_n_no2",
]


def _find_csv(pattern: str) -> str:
    """Find the most recent CSV file matching a glob pattern."""
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file found matching: {pattern}")
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


def _make_tree_id(df: pl.DataFrame) -> pl.DataFrame:
    """Add ``tree_id`` column from ``code_country``, ``code_plot``, and ``tree_number``."""
    return df.with_columns(
        (
            pl.col("code_country").cast(pl.Utf8).str.zfill(2)
            + "."
            + pl.col("code_plot").cast(pl.Utf8).str.zfill(4)
            + "."
            + pl.col("tree_number").cast(pl.Utf8).str.zfill(5)
        ).alias("tree_id")
    )


def _load_dictionaries() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Fetch species and country lookup tables from ICP-Forests website."""
    species_html = requests.get(
        "https://icp-forests.org/documentation/Dictionaries/d_tree_spec.html"
    ).text
    species_df = pl.from_pandas(pd.read_html(StringIO(species_html))[0]).rename(
        {"CODE": "code_tree_species", "DESCRIPTION": "specie"}
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
    df_plots_raw = pl.read_csv(path, separator=";")

    df_plots_raw = df_plots_raw.with_columns(
        (
            pl.col("code_country").cast(pl.Utf8).str.zfill(2)
            + "."
            + pl.col("code_plot").cast(pl.Utf8).str.zfill(4)
        ).alias("plot_id")
    ).rename(
        {
            "latitude": "plot_latitude",
            "longitude": "plot_longitude",
            "slope": "plot_slope",
            "code_orientation": "plot_orientation",
            "code_altitude": "plot_altitude",
            "plot_size": "plot_size_ha",
        }
    )
    return df_plots_raw


def _load_trees(
    species_df: pl.DataFrame,
    country_df: pl.DataFrame,
) -> pl.DataFrame:
    """Load cleaned tree measurements from ``gr_ipm.csv`` for all species."""
    path = _find_csv(os.path.join(_ICP_FOLDER, "595_gr_*/gr_ipm.csv"))

    df = (
        pl.read_csv(path, separator=";", ignore_errors=True)
        .with_columns(pl.col("date_assessment").str.to_datetime().alias("date"))
        .with_columns(
            pl.when(pl.col("date").is_null())
            .then(pl.date(pl.col("survey_year"), 7, 1).cast(pl.Datetime))
            .otherwise(pl.col("date"))
            .alias("date")
        )
        .pipe(_make_plot_id)
        .pipe(_make_tree_id)
        .join(species_df.select(["code_tree_species", "specie"]), on="code_tree_species")
        .join(country_df.select(["code_country", "country"]), on="code_country")
        .filter(~pl.col("country").is_in(_COUNTRIES_EXCLUDE))
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
        # Some assessments are submitted under two adjacent survey_year campaigns
        # Keep one row per tree × date, preferring the survey_year matching the date.
        .sort("tree_id", "date", "survey_year")
        .unique(subset=["tree_id", "date"], keep="last")
        .rename({"diameter": "dbh_cm"})
        .select(
            "survey_year",
            "tree_id",
            "plot_id",
            "date",
            "code_country",
            "country",
            "code_tree_species",
            "specie",
            "code_plot",
            "tree_number",
            "dbh_cm",
            "height",
        )
    )

    df = df.with_columns(ba_tree=math.pi * (pl.col("dbh_cm") / 200.0) ** 2)
    return df


def _filter_single_species(trees: pl.DataFrame) -> pl.DataFrame:
    """Keep only (plot, survey year) observations with a single target species."""
    single_species = (
        trees.group_by("plot_id", "survey_year")
        .agg(pl.col("specie").n_unique().alias("n_species"))
        .filter(pl.col("n_species") == 1)
        .select("plot_id", "survey_year")
    )
    return trees.join(single_species, on=["plot_id", "survey_year"], how="inner")


def _altitude_m() -> pl.Expr:
    """Altitude in metres, from the coded 50 m class `plot_altitude` where `altitude_m` is missing.

    Class `c` covers `(c - 1) * 50` to `c * 50` m, so its midpoint is used.
    """
    return pl.coalesce(pl.col("altitude_m"), pl.col("plot_altitude") * 50.0 - 25.0).alias(
        "altitude"
    )


def _aggregate_per_plot(trees: pl.DataFrame, plots: pl.DataFrame) -> pl.DataFrame:
    """Aggregate tree-level data to plot-level per-ha values."""
    per_plot = aggregate_per_plot(
        trees.sort("date"),
        group_by=["plot_id", "specie", "date"],
        extra_aggs=[pl.col("height").mean(), pl.col("soph_avg_age").mean().alias("stand_age")],
    )

    return (
        scale_to_hectare(per_plot.join(plots, on="plot_id", how="inner"), pl.col("plot_size_ha"))
        .rename({"n_trees": "n_stems"})
        .with_columns(
            pl.col("plot_latitude")
            .map_elements(dms_to_decimal, return_dtype=pl.Float64)
            .alias("lat"),
            pl.col("plot_longitude")
            .map_elements(dms_to_decimal, return_dtype=pl.Float64)
            .alias("lon"),
            _altitude_m(),
            (pl.col("plot_size_ha") * 10000.0).alias("area_m2"),
            pl.col("date").dt.year().alias("year"),
        )
        .sort(["specie", "plot_id", "date"])
    )


def _load_crown(trees: pl.DataFrame) -> pl.DataFrame:
    """Load crown data matched to census dates via a 5-year backward window.

    Parameters
    ----------
    trees : pl.DataFrame
        Tree census data with ``tree_id`` and ``date`` columns.
    """
    path = _find_csv(os.path.join(_ICP_FOLDER, "595_cc_*/cc_trc.csv"))

    crown_raw = (
        pl.read_csv(path, separator=";")
        .with_columns(pl.col("date_survey").str.to_datetime().alias("date"))
        .pipe(_make_tree_id)
        .filter(
            pl.col("code_defoliation").cast(pl.Int64, strict=False).is_not_null()
            & pl.col("code_defoliation").cast(pl.Int64, strict=False).ge(0)
        )
        .with_columns(defoliation=pl.col("code_defoliation").cast(pl.Int32))
    )

    census_ref = (
        trees.select("tree_id", "date", pl.col("date").alias("census_date"))
        .unique()
        .sort(["tree_id", "date"])
    )

    return (
        crown_raw.sort("date")
        .join_asof(
            census_ref,
            by="tree_id",
            on="date",
            strategy="forward",
        )
        .drop_nulls(subset="census_date")
        .filter(
            pl.col("date").is_between(
                pl.col("census_date") - pl.duration(days=int(365.25 * 5)),
                pl.col("census_date"),
            )
        )
        .group_by("tree_id", "census_date")
        .agg(
            pl.len().alias("num_defoliation_obs"),
            pl.mean("defoliation").alias("defoliation_mean"),
            pl.min("defoliation").alias("defoliation_min"),
            pl.max("defoliation").alias("defoliation_max"),
            pl.median("defoliation").alias("defoliation_median"),
            pl.last("defoliation").alias("defoliation_last"),
            pl.col("code_social_class").min().alias("social_class_min"),
            pl.col("code_social_class").max().alias("social_class_max"),
            pl.col("code_social_class").mode().first().alias("social_class_mode"),
            pl.col("code_social_class").last().alias("social_class_last"),
            pl.col("code_social_class").eq(1).any().alias("was_dominant"),
            pl.col("code_social_class").eq(2).any().alias("was_codominant"),
            pl.col("code_social_class").eq(3).any().alias("was_subdominant"),
            pl.col("code_social_class").eq(4).any().alias("was_suppressed"),
            pl.col("code_social_class").eq(5).any().alias("was_dying"),
        )
        .filter(pl.col("defoliation_max").lt(100))
        .filter(pl.col("num_defoliation_obs").gt(1))
        .rename({"census_date": "date"})
    )


def _load_deposition_periods() -> tuple[pl.DataFrame, list[str], list[str]]:
    """Load cleaned throughfall deposition as one plot-level record per sampling period.

    Concentrations are converted to fluxes (kg/ha) and records from multiple
    samplers of the same plot and period are averaged.

    Returns
    -------
    tuple[pl.DataFrame, list[str], list[str]]
        Period records, flux deposition columns, and non-flux deposition columns.
    """
    path = _find_csv(os.path.join(_ICP_FOLDER, "595_dp_*/dp_dem.csv"))
    countries = pl.read_csv(
        os.path.join(os.path.dirname(path), "adds/dictionaries/d_country.csv"), separator=";"
    )
    excluded_codes = countries.filter(pl.col("lib_country").is_in(_COUNTRIES_EXCLUDE))["code"]
    src_renames = {
        "n_total": "n_tot",
        "c_total": "c_tot",
        "s_total": "s_tot",
        "p_total": "p_tot",
        "conductivity": "cond",
        "alkalinity": "alk",
    }
    dep_rename = {col: f"dep_{col}" for col in _DEP_NAMES}

    header = pl.read_csv(path, separator=";", n_rows=0).columns
    active_src = {k: v for k, v in src_renames.items() if k in header}
    post_src = (set(header) - set(active_src)) | set(active_src.values())
    active_dep = {k: v for k, v in dep_rename.items() if k in post_src}

    df = (
        pl.read_csv(path, separator=";")
        .pipe(_make_plot_id)
        .rename(active_src)
        .rename(active_dep)
        .filter(
            pl.col("date_start").is_not_null()
            & pl.col("date_end").is_not_null()
            & (pl.col("code_sampler") == 1)
            & ~pl.col("code_country").is_in(excluded_codes.implode())
        )
    )

    if "code_vsampling" in df.columns:
        df = df.filter(~pl.col("code_vsampling").is_in([2, 3, 4, 7, 9]))

    df = df.filter(~pl.col("code_sampler").eq(8))

    dep_cols = [c for c in dep_rename.values() if c in df.columns]
    non_conc = [c for c in _DEP_NON_CONC if c in dep_cols]
    flux_cols = [c for c in dep_cols if c not in non_conc]

    df = df.with_columns([pl.col(c).cast(pl.Float64, strict=False) for c in dep_cols])

    if flux_cols:
        df = df.with_columns(
            pl.when(cs.by_name(*flux_cols).ne(-1.0)).then(cs.by_name(*flux_cols)).otherwise(None)
        )

    if dep_cols:
        df = df.with_columns(cs.by_name(*dep_cols).fill_nan(None))

    ns_conc = ["dep_n_tot", "dep_n_nh4", "dep_n_no3", "dep_s_so4"]
    din = pl.col("dep_n_nh4") + pl.col("dep_n_no3")
    df = df.with_columns(
        pl.when(cs.by_name(*ns_conc).le(_DEP_MAX_NS_CONC_MG_L)).then(cs.by_name(*ns_conc)),
        quantity=pl.when(pl.col("quantity").is_between(0, _DEP_MAX_QUANTITY_MM)).then(
            pl.col("quantity")
        ),
    ).with_columns(
        dep_n_tot=pl.when(
            (pl.col("dep_n_tot") > _DEP_MAX_NTOT_DIN_RATIO * din)
            & (pl.col("dep_n_tot") - din > _DEP_MAX_DON_MG_L)
        )
        .then(None)
        .otherwise(pl.col("dep_n_tot"))
    )

    df = df.with_columns(
        dep_n_tot=pl.when(pl.col("dep_n_tot").is_null())
        .then(din + pl.col("dep_n_org").fill_null(0))
        .otherwise(pl.col("dep_n_tot"))
    )

    if flux_cols:
        df = df.with_columns(cs.by_name(*flux_cols) * pl.col("quantity") / 100)

    df = df.with_columns(
        pl.col("date_start", "date_end").str.slice(0, 10).str.to_date(strict=False)
    ).drop_nulls(subset=["date_start", "date_end"])

    # Drop resubmitted duplicates, then sampler-periods with broken dates
    period = ["plot_id", "date_start", "date_end"]
    df = df.unique(subset=[*period, "sampler_id", *dep_cols, "quantity"]).filter(
        pl.len().over(*period, "sampler_id") <= _DEP_MAX_RECORDS_PER_SAMPLER_PERIOD
    )

    df = df.group_by(period).agg(
        pl.col("survey_year").first(), cs.by_name(*dep_cols, "quantity").mean()
    )
    return df, flux_cols, non_conc


def _sum_or_null(expr: pl.Expr) -> pl.Expr:
    """Sum an expression in an aggregation, returning null when all values are null."""
    return pl.when(expr.is_not_null().any()).then(expr.sum())


def _load_deposition(trees: pl.DataFrame) -> pl.DataFrame:
    """Load deposition and aggregate over a ±5-year window around each census date.

    Parameters
    ----------
    trees : pl.DataFrame
        Tree census data with ``tree_id``, ``plot_id``, and ``date`` columns.
    """
    df, flux_cols, non_conc = _load_deposition_periods()

    # Annual aggregation per plot
    annual_agg: list[pl.Expr] = [pl.len().alias("num_deposition_obs")]
    annual_agg.extend(_sum_or_null(pl.col(c)).alias(c) for c in flux_cols)
    if non_conc:
        annual_agg.append(cs.by_name(*non_conc).mean())
    annual_agg.append(_sum_or_null(pl.col("quantity")).alias("yearly_precip"))

    df_annual = df.group_by("plot_id", "survey_year").agg(annual_agg)

    # Integrate over ±5 year window around each census date (mean annual values)
    census_ref = trees.select("plot_id", "tree_id", "date").unique()

    window_agg: list[pl.Expr] = [pl.sum("num_deposition_obs").alias("num_deposition_obs")]
    if flux_cols:
        window_agg.append(cs.by_name(*[c for c in flux_cols if c in df_annual.columns]).mean())
    if non_conc:
        window_agg.append(cs.by_name(*[c for c in non_conc if c in df_annual.columns]).mean())
    if "yearly_precip" in df_annual.columns:
        window_agg.append(pl.mean("yearly_precip").alias("yearly_precip"))

    return (
        df_annual.join(census_ref, on="plot_id", how="inner")
        .with_columns(
            period_start_year=pl.col("date").dt.year() - 5,
            period_end_year=pl.col("date").dt.year() + 5,
        )
        .filter(
            pl.col("survey_year").is_between(
                pl.col("period_start_year"), pl.col("period_end_year")
            )
        )
        .group_by("tree_id", "date")
        .agg(window_agg)
    )


def _load_soil(trees: pl.DataFrame) -> pl.DataFrame:
    """Load soil solutions and aggregate over a ±5-year window around each census date.

    Parameters
    ----------
    trees : pl.DataFrame
        Tree census data with ``tree_id``, ``plot_id``, and ``date`` columns.
    """
    path = _find_csv(os.path.join(_ICP_FOLDER, "595_ss_*/ss_ssm.csv"))
    src_renames = {"conductivity": "cond", "alkalinity": "alk", "n_total": "n_tot"}
    soil_rename = {col: f"ss_{col}" for col in _SOIL_NAMES}

    header = pl.read_csv(path, separator=";", n_rows=0).columns
    active_src = {k: v for k, v in src_renames.items() if k in header}
    post_src = (set(header) - set(active_src)) | set(active_src.values())
    active_soil = {k: v for k, v in soil_rename.items() if k in post_src}

    df = (
        pl.read_csv(path, separator=";").pipe(_make_plot_id).rename(active_src).rename(active_soil)
    )

    ss_cols = [c for c in df.columns if c.startswith("ss_")]
    df = df.filter(
        pl.col("sample_vol").cast(pl.Float64, strict=False).is_null()
        | pl.col("sample_vol").cast(pl.Float64, strict=False).gt(0)
    ).with_columns([pl.col(c).cast(pl.Float64, strict=False) for c in ss_cols])

    if ss_cols:
        df = df.with_columns(
            pl.when(cs.by_name(*ss_cols).is_between(0.0001, 10000))
            .then(cs.by_name(*ss_cols))
            .otherwise(None)
        )

    # Annual aggregation per plot
    annual_agg: list[pl.Expr] = [pl.len().alias("num_soil_obs")]
    if ss_cols:
        annual_agg.append(cs.by_name(*ss_cols).mean())

    df_annual = df.group_by("plot_id", "survey_year").agg(annual_agg)

    # Integrate over ±5 year window around each census date
    census_ref = trees.select("plot_id", "tree_id", "date").unique()

    window_agg: list[pl.Expr] = [pl.sum("num_soil_obs").alias("num_soil_obs")]
    if ss_cols:
        window_agg.append(cs.by_name(*[c for c in ss_cols if c in df_annual.columns]).mean())

    return (
        df_annual.join(census_ref, on="plot_id", how="inner")
        .with_columns(
            period_start_year=pl.col("date").dt.year() - 5,
            period_end_year=pl.col("date").dt.year() + 5,
        )
        .filter(
            pl.col("survey_year").is_between(
                pl.col("period_start_year"), pl.col("period_end_year")
            )
        )
        .group_by("tree_id", "date")
        .agg(window_agg)
    )


def _load_plot_meta() -> pl.DataFrame:
    """Load plot metadata from Etzold et al.; age is referenced to year 2000."""
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
            sdi=pl.col("sdi").replace("NA", None).cast(pl.Float32),
            temp=pl.col("temp").replace("NA", None).cast(pl.Float32),
            precip=pl.col("precip").replace("NA", None).cast(pl.Float32),
        )
        .drop_nulls(subset=["yr_first", "yr_last"])
        .group_by("plot_id")
        .agg(
            pl.mean("age").alias("soph_avg_age"),
            pl.mean("sdi").alias("soph_avg_sdi"),
            pl.mean("temp").alias("soph_avg_temp"),
            pl.mean("precip").alias("soph_avg_precip"),
        )
    )


def prepare_icp_tree_data(output_path: str | None = None) -> pl.DataFrame:
    """Load and clean ICP Level II data at tree × census level.

    Adds `lat`/`lon` (decimal degrees, converted from the packed DMS
    `plot_latitude`/`plot_longitude`), `year`/`month` (from `date`, which
    falls back to 1 July of `survey_year` for assessments missing a
    recorded date), `n_stems` (always 1, one physical tree per row),
    `area_m2` (`plot_size_ha * 10000`), `altitude` (same value as
    `altitude_m`, in metres — not `plot_altitude`, which is a coded
    altitude class) and per-tree `biom_stem`, `biom_foliage`, `biom_root`
    (kg tree⁻¹), `la_m2` (leaf area, m² tree⁻¹) and `basal_area` (m²
    tree⁻¹, same value as `ba_tree`, kept for existing consumers).
    """
    if output_path is None:
        output_path = str(os.path.join(clean_data_folder, "icp_tree_data.parquet"))

    species_df, country_df = _load_dictionaries()
    logger.info("Loaded species and country dictionaries")

    plots = _load_plots()
    logger.info("Loaded %d plots", plots.height)

    trees = _load_trees(species_df, country_df)
    logger.info("Loaded %d tree records", trees.height)
    trees = trees.join(plots, on="plot_id", how="left")
    logger.info("Joined tree records with plot metadata: %d rows", trees.height)
    # Missing plot sizes are null or coded as -1 in si_plt.
    trees = trees.filter(pl.col("plot_size_ha").gt(0))
    logger.info("After dropping trees without plot size: %d rows", trees.height)

    crown = _load_crown(trees)
    logger.info("Loaded crown conditions: %d tree×census rows", crown.height)
    trees = trees.join(crown, on=["tree_id", "date"], how="left")

    deposition = _load_deposition(trees)
    logger.info("Loaded deposition: %d tree×census rows", deposition.height)
    trees = trees.join(deposition, on=["tree_id", "date"], how="left")

    soil = _load_soil(trees)
    logger.info("Loaded soil solutions: %d tree×census rows", soil.height)
    trees = trees.join(soil, on=["tree_id", "date"], how="left")

    plot_meta = _load_plot_meta()
    logger.info("Loaded plot metadata: %d plots with age data", plot_meta.height)
    trees = trees.join(plot_meta, on="plot_id", how="left").with_columns(
        (pl.col("soph_avg_age") + (pl.col("survey_year") - _AGE_REFERENCE_YEAR)).alias(
            "soph_avg_age"
        )
    )

    trees = (
        trees.pipe(add_allometric_columns, _FORRESTER_EQ3, dbh_col="dbh_cm", species_col="specie")
        .rename(
            {
                "allo_sb_kg": "biom_stem",
                "allo_fb_kg": "biom_foliage",
                "allo_rb_kg": "biom_root",
                "allo_la_m2": "la_m2",
            }
        )
        .with_columns(
            pl.col("plot_latitude")
            .map_elements(dms_to_decimal, return_dtype=pl.Float64)
            .alias("lat"),
            pl.col("plot_longitude")
            .map_elements(dms_to_decimal, return_dtype=pl.Float64)
            .alias("lon"),
            pl.col("ba_tree").alias("basal_area"),
            pl.lit(1.0).alias("n_stems"),
            (pl.col("plot_size_ha") * 10000.0).alias("area_m2"),
            _altitude_m(),
            pl.col("date").dt.year().alias("year"),
            pl.col("date").dt.month().alias("month"),
            # ICP heights are field measurements; biomass is always Forrester allometry.
            measured_flag("height", pl.lit(True)),
            *computed_flags(BIOMASS_COLS),
        )
    )

    trees = trees.sort(["specie", "tree_id", "date"])

    trees.write_parquet(output_path)
    logger.info("Saved %d rows to %s", trees.height, output_path)

    return trees


def _fill_with_measured(plot_df: pl.DataFrame, inventory: pl.DataFrame) -> pl.DataFrame:
    """Prefer ``gr_inv`` values over tree aggregates and add ``flag_*`` columns.

    Each ``flag_<metric>`` is "measured" when the value comes from
    ``gr_inv``, "computed" when it is aggregated from ``gr_ipm`` trees, and
    null when neither is available. ICP has no measured LAI or biomass.
    """
    measured_cols: dict[str, str | None] = {
        "height": "height_basal_area_tree",
        "dbh_qmd": "diameter_basal_area_tree",
        "basal_area": "basal_area",
        "lai": None,
        **dict.fromkeys(BIOMASS_COLS),
    }
    measured = inventory.select(
        "plot_id",
        "specie",
        "date",
        *(pl.col(src).alias(f"{dst}_measured") for dst, src in measured_cols.items() if src),
    )
    df = plot_df.with_columns(pl.col("date").cast(pl.Date).alias("_date")).join(
        measured.rename({"date": "_date"}), on=["plot_id", "specie", "_date"], how="left"
    )
    fills: list[pl.Expr] = []
    for metric, src in measured_cols.items():
        value = pl.col(f"{metric}_measured") if src else pl.lit(None, dtype=pl.Float64)
        fills += [
            pl.coalesce(value, pl.col(metric)).alias(metric),
            pl.when(value.is_not_null())
            .then(pl.lit("measured"))
            .when(pl.col(metric).is_not_null())
            .then(pl.lit("computed"))
            .alias(f"flag_{metric}"),
        ]
    return df.with_columns(fills).drop("_date", cs.ends_with("_measured"))


def prepare_icp_plot_data(
    output_path: str | None = None, filter_single_species: bool = False
) -> pl.DataFrame:
    """Aggregate ICP Level II tree data to plot level for 3PG calibration.

    Height, QMD and basal area reported in ``gr_inv`` (see
    `prepare_icp_inventory_data`) replace the tree aggregates where
    available; ``flag_height``, ``flag_dbh_qmd``, ``flag_basal_area``,
    ``flag_lai`` and ``flag_biom_*`` record whether each value is
    "measured" or "computed".

    Parameters
    ----------
    output_path : str | None
        Parquet path to write the result. Defaults to
        `clean_data_folder/icp_plot_data.parquet`.
    filter_single_species : bool
        Keep only single-species (plot, survey year) observations (see
        `_filter_single_species`). When False, mixed plots give one row per species.

    Returns
    -------
    pl.DataFrame
        One row per (plot_id, specie, date).
    """
    if output_path is None:
        output_path = str(os.path.join(clean_data_folder, "icp_plot_data.parquet"))

    trees = pl.read_parquet(os.path.join(clean_data_folder, "icp_tree_data.parquet"))
    inventory = pl.read_parquet(os.path.join(clean_data_folder, "icp_inventory_data.parquet"))
    plots = _load_plots()
    if filter_single_species:
        trees = _filter_single_species(trees)
        logger.info("After single-species filter: %d records", trees.height)

    result = _aggregate_per_plot(trees, plots).pipe(_fill_with_measured, inventory)
    logger.info(
        "Aggregated to %d plot×year observations across %d plots",
        result.height,
        result["plot_id"].n_unique(),
    )

    result.write_parquet(output_path)
    logger.info("Saved %d rows to %s", result.height, output_path)

    return result


def prepare_icp_inventory_data(output_path: str | None = None) -> pl.DataFrame:
    """Prepare the basal-area mean tree reported in ``gr_inv.csv`` and derive basal area.

    ``trees_remain`` is reported per plot by some countries and per hectare
    by others, so stem density is instead counted from the ``gr_ipm`` trees
    in ``icp_tree_data.parquet`` (divided by the ``si_plt`` plot size).
    ``trees_remain`` is only used as a relative weight to combine several
    growth subplots of one plot. Because ``diameter_basal_area_tree`` is the
    quadratic mean diameter, basal area is ``n_stems * pi * (d / 200)**2``.

    Parameters
    ----------
    output_path : str | None
        Parquet path to write the result. Defaults to
        `clean_data_folder/icp_inventory_data.parquet`.

    Returns
    -------
    pl.DataFrame
        One row per (plot_id, specie, date) with `diameter_basal_area_tree`
        (cm), `height_basal_area_tree` (m), `n_stems` (ha⁻¹, from gr_ipm)
        and `basal_area` (m² ha⁻¹).
    """
    if output_path is None:
        output_path = str(os.path.join(clean_data_folder, "icp_inventory_data.parquet"))

    species_df, _ = _load_dictionaries()
    keys = ["plot_id", "specie", "date"]

    stem_density = (
        pl.read_parquet(os.path.join(clean_data_folder, "icp_tree_data.parquet"))
        .filter(pl.col("plot_size_ha").gt(0))
        .with_columns(pl.col("date").cast(pl.Date))
        .group_by(keys)
        .agg((pl.len() / pl.col("plot_size_ha").first()).alias("n_stems"))
    )

    weight = pl.col("trees_remain")
    inventory = (
        pl.read_csv(
            _find_csv(os.path.join(_ICP_FOLDER, "595_gr_*/gr_inv.csv")),
            separator=";",
            infer_schema_length=0,
        )
        .with_columns(
            pl.col("code_tree_species").cast(pl.Int64),
            pl.col("date_sampling").str.to_date().alias("date"),
            pl.col("trees_remain", "diameter_basal_area_tree", "height_basal_area_tree").cast(
                pl.Float64, strict=False
            ),
        )
        .filter(pl.col("trees_remain").gt(0) & pl.col("diameter_basal_area_tree").gt(0))
        .pipe(_make_plot_id)
        .join(species_df.select("code_tree_species", "specie"), on="code_tree_species")
        .group_by(keys)
        .agg(
            ((weight * pl.col("diameter_basal_area_tree").pow(2)).sum() / weight.sum())
            .sqrt()
            .alias("diameter_basal_area_tree"),
            (
                (weight * pl.col("height_basal_area_tree")).sum()
                / weight.filter(pl.col("height_basal_area_tree").is_not_null()).sum()
            )
            .fill_nan(None)
            .alias("height_basal_area_tree"),
        )
        .join(stem_density, on=keys, how="inner")
        .with_columns(
            basal_area=pl.col("n_stems")
            * math.pi
            * (pl.col("diameter_basal_area_tree") / 200.0) ** 2
        )
        .sort(keys)
    )

    inventory.write_parquet(output_path)
    logger.info("Saved %d rows to %s", inventory.height, output_path)
    return inventory


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    tree_df = prepare_icp_tree_data()
    print("Tree-level data:")
    print(tree_df.head())

    inventory_df = prepare_icp_inventory_data()
    print("Inventory data:")
    print(inventory_df.head())

    plot_id = "50.0013"
    print(tree_df.filter(pl.col("plot_id") == plot_id))

    plot_df = prepare_icp_plot_data()
    print("\nPlot-level data:")
    print(plot_df.head())
    print("\nPer-species plot counts:")
    print(plot_df.group_by("specie").agg(pl.col("plot_id").n_unique().alias("n_plots")))
