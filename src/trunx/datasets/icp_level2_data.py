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
from trunx.gp3.allometrics import CoefficientsDict, add_allometric_columns, load_forrester_eq3

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
_COUNTRIES_EXCLUDE: list[str] = ["Belgium", "Spain"]

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


def _aggregate_per_plot(trees: pl.DataFrame, plots: pl.DataFrame) -> pl.DataFrame:
    """Compute allometrics and aggregate tree-level data to plot-level per-ha values."""
    trees = add_allometric_columns(trees, _FORRESTER_EQ3, dbh_col="dbh_cm", species_col="specie")

    logger.info("Computed allometric quantities for %d trees", trees.height)

    per_plot = (
        trees.sort("date")
        .group_by("plot_id", "specie", "date")
        .agg(
            # pl.first("date"),
            pl.len().alias("n_count"),
            pl.col("dbh_cm").mean(),
            pl.col("height").mean(),
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
        )
        .with_columns(
            basal_area=pl.lit(math.pi) * (pl.col("dbh_cm") / 200.0).pow(2) * pl.col("n_stems"),
        )
        .drop("n_count", "plot_sb_kg", "plot_fb_kg", "plot_rb_kg", "plot_la_m2", "plot_ba_m2")
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


def _load_deposition(trees: pl.DataFrame) -> pl.DataFrame:
    """Load deposition and aggregate over a ±5-year window around each census date.

    Parameters
    ----------
    trees : pl.DataFrame
        Tree census data with ``tree_id``, ``plot_id``, and ``date`` columns.
    """
    path = _find_csv(os.path.join(_ICP_FOLDER, "595_dp_*/dp_dem.csv"))
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

    df = df.with_columns(
        dep_n_tot=pl.when(pl.col("dep_n_tot").is_null())
        .then(pl.col("dep_n_nh4") + pl.col("dep_n_no3") + pl.col("dep_n_org").fill_null(0))
        .otherwise(pl.col("dep_n_tot"))
    )

    if flux_cols:
        df = df.with_columns(cs.by_name(*flux_cols) * pl.col("quantity") / 100)

    # Annual aggregation per plot
    annual_agg: list[pl.Expr] = [pl.len().alias("num_deposition_obs")]
    if flux_cols:
        annual_agg.append(cs.by_name(*flux_cols).sum())
    if non_conc:
        annual_agg.append(cs.by_name(*non_conc).mean())
    if "quantity" in df.columns:
        annual_agg.append(pl.col("quantity").sum().alias("yearly_precip"))

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
    """Load and clean ICP Level II data at tree × census level."""
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

    trees = trees.sort(["specie", "tree_id", "date"])

    trees.write_parquet(output_path)
    logger.info("Saved %d rows to %s", trees.height, output_path)

    return trees


def prepare_icp_plot_data(output_path: str | None = None) -> pl.DataFrame:
    """Aggregate ICP Level II tree data to plot level for 3PG calibration."""
    trees = pl.read_parquet(os.path.join(clean_data_folder, "icp_tree_data.parquet"))
    plots = _load_plots()
    trees = _filter_single_species(trees)
    logger.info("After single-species filter: %d records", trees.height)

    result = _aggregate_per_plot(trees, plots)
    logger.info(
        "Aggregated to %d plot×year observations across %d plots",
        result.height,
        result["plot_id"].n_unique(),
    )
    logger.info("Saved %d rows to %s", result.height, output_path)

    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    tree_df = prepare_icp_tree_data()
    print("Tree-level data:")
    print(tree_df.head())

    plot_id = "50.0013"
    print(tree_df.filter(pl.col("plot_id") == plot_id))

    plot_df = prepare_icp_plot_data()
    print("\nPlot-level data:")
    print(plot_df.head())
    print("\nPer-species plot counts:")
    print(plot_df.group_by("specie").agg(pl.col("plot_id").n_unique().alias("n_plots")))
