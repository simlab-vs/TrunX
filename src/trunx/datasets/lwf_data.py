"""Prepare Swiss LWF plot-level observation data for 3PG modelling.

Data limitations
----------------
Missing: plot area and altitude are not in the raw tree-level file. Both
are sourced from https://lwf.wsl.ch/en/flaechen/, which publishes one area
and one altitude range per site (not per `parcelname` subplot); the site
value is applied to every subplot of that site. Lantsch has no published
plot area ("n.d." on its page), so its per-hectare columns are null.

Unit conventions
-----------------
- `dbh_corrected`/`height_corrected` (gap-filled) columns are used for DBH
  and height rather than the raw measurements.
- All output biomass columns are in t ha⁻¹ (divided by 1 000).
- Basal area (`basal_area`) is in m² ha⁻¹.
- Leaf area index (`lai`) is (dimensionless).
- `altitude` is the midpoint of the site's published altitude range (m).
"""

import logging
import math
import os

import polars as pl

from trunx.config import clean_data_folder, data_folder
from trunx.gp3.allometrics import (
    CoefficientsDict,
    add_allometric_columns,
    aggregate_per_plot,
    load_forrester_eq3,
    scale_to_hectare,
)

logger = logging.getLogger(__name__)

_LWF_FOLDER = str(os.path.join(data_folder, "SwissData/LWF"))

SPECIES_LIST = ["Picea abies", "Fagus sylvatica"]

_FORRESTER_EQ3: CoefficientsDict = load_forrester_eq3(species=SPECIES_LIST)

# Plot area (ha) and altitude range midpoint (m), per site, from
# https://lwf.wsl.ch/en/flaechen/<site>/. Lantsch's plot area is "n.d." on
# its page.
_SITE_AREA_HA: dict[str, float | None] = {
    "Alptal": 0.6,
    "Beatenberg": 2.0,
    "Bettlachstock": 1.28,
    "Celerina": 2.0,
    "Chironico": 2.0,
    "Davos": 0.6,
    "Isone": 2.0,
    "Jussy": 1.99,
    "Laegeren": 1.34,
    "Lantsch": None,
    "Lausanne": 2.0,
    "Lens": 2.0,
    "Nationalpark": 2.0,
    "Neunkirch": 2.0,
    "Novaggio": 1.5,
    "Othmarsingen": 1.0,
    "Schänis": 2.0,
    "Visp": 2.0,
    "Vordemwald": 2.0,
}

_SITE_ALTITUDE_M: dict[str, float] = {
    "Alptal": (1149 + 1170) / 2,
    "Beatenberg": (1490 + 1532) / 2,
    "Bettlachstock": (1101 + 1196) / 2,
    "Celerina": (1846 + 1896) / 2,
    "Chironico": (1342 + 1387) / 2,
    "Davos": (1635 + 1665) / 2,
    "Isone": (1181 + 1259) / 2,
    "Jussy": (496 + 506) / 2,
    "Laegeren": (643 + 718) / 2,
    "Lantsch": (1458 + 1490) / 2,
    "Lausanne": (800 + 814) / 2,
    "Lens": (1033 + 1093) / 2,
    "Nationalpark": (1890 + 1907) / 2,
    "Neunkirch": (554 + 609) / 2,
    "Novaggio": (902 + 997) / 2,
    "Othmarsingen": (467 + 500) / 2,
    "Schänis": (693 + 773) / 2,
    "Visp": (657 + 733) / 2,
    "Vordemwald": (473 + 487) / 2,
}


def _site_info() -> pl.DataFrame:
    """Build a per-site lookup of plot area and altitude.

    Returns
    -------
    pl.DataFrame
        Columns: `sitename`, `area_m2`, `altitude`.
    """
    return pl.DataFrame(
        {
            "sitename": list(_SITE_AREA_HA),
            "area_m2": [
                area_ha * 10000.0 if area_ha is not None else None
                for area_ha in _SITE_AREA_HA.values()
            ],
            "altitude": list(_SITE_ALTITUDE_M.values()),
        }
    )


def _load_trees() -> pl.DataFrame:
    """Load and clean the LWF tree-level CSV.

    Returns
    -------
    pl.DataFrame
        One row per tree x year, filtered to target species.
    """
    return (
        pl.read_csv(
            os.path.join(_LWF_FOLDER, "Givi_LWFdata_Jun26_flagged.csv"), null_values=["NA"]
        )
        .filter(pl.col("tree_species").is_in(SPECIES_LIST))
        .drop("height")
        .rename(
            {
                "banr": "tree_id",
                "tree_species": "specie",
                "dbh_corrected": "dbh_cm",
                "height_corrected": "height",
                "tree_status": "status",
                "date_observation": "date",
            }
        )
        .with_columns(
            pl.concat_str(["sitename", "parcelname"], separator="_").alias("plot_id"),
            pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"),
        )
        .select(
            [
                "plot_id",
                "sitename",
                "tree_id",
                "year",
                "date",
                "specie",
                "status",
                "dbh_cm",
                "height",
                "lat",
                "lon",
            ]
        )
    )


def prepare_lwf_tree_data(output_path: str | None = None) -> pl.DataFrame:
    """Load and clean LWF tree-level data.

    Parameters
    ----------
    output_path : str | None
        Parquet path to write the result. Defaults to
        `clean_data_folder/lwf_tree_data.parquet`.

    Returns
    -------
    pl.DataFrame
        One row per tree x year, joined with per-site area and altitude,
        for the target species. Not filtered to single-species plots — see
        `prepare_lwf_data` for that. Columns include `plot_id`, `tree_id`,
        `specie`, `lat`, `lon`, `altitude`, `date`, `year`, `month`,
        `dbh_cm`, `height`, `n_stems` (always 1, one physical tree per
        row), `biom_stem`, `biom_foliage`, `biom_root` (kg tree⁻¹), `la_m2`
        (leaf area, m² tree⁻¹) and `basal_area` (m² tree⁻¹).
    """
    if output_path is None:
        output_path = str(os.path.join(clean_data_folder, "lwf_tree_data.parquet"))

    trees = (
        _load_trees()
        .join(_site_info(), on="sitename", how="left")
        .pipe(add_allometric_columns, _FORRESTER_EQ3, species_col="specie")
        .rename(
            {
                "allo_sb_kg": "biom_stem",
                "allo_fb_kg": "biom_foliage",
                "allo_rb_kg": "biom_root",
                "allo_la_m2": "la_m2",
            }
        )
        .with_columns(
            (math.pi * pl.col("dbh_cm").pow(2) / 40000.0).alias("basal_area"),
            pl.lit(1.0).alias("n_stems"),
            pl.col("date").dt.month().alias("month"),
        )
    )

    trees.write_parquet(output_path)
    logger.info("Saved %d rows to %s", trees.height, output_path)
    return trees


def _filter_single_species(trees: pl.DataFrame) -> pl.DataFrame:
    """Keep only (plot, year) measurements where alive trees are a single species.

    Parameters
    ----------
    trees : pl.DataFrame

    Returns
    -------
    pl.DataFrame
        Filtered tree-level data.
    """
    valid = (
        trees.filter(pl.col("status") == "alive")
        .group_by(["plot_id", "year"])
        .agg(pl.col("specie").n_unique().alias("n_species"))
        .filter(pl.col("n_species") == 1)
        .select(["plot_id", "year"])
    )
    return trees.join(valid, on=["plot_id", "year"], how="inner")


def _aggregate_per_plot(trees: pl.DataFrame) -> pl.DataFrame:
    """Aggregate alive trees to per-hectare plot-level quantities.

    Per-hectare conversion uses the per-site plot area from
    `_SITE_AREA_HA`, applied to every subplot of that site.

    Parameters
    ----------
    trees : pl.DataFrame
        Filtered tree-level data with allometric columns, joined with
        per-site area and altitude (see `prepare_lwf_tree_data`).

    Returns
    -------
    pl.DataFrame
        One row per (plot, year) with per-hectare stand quantities.
    """
    alive = trees.filter(pl.col("status") == "alive")

    per_plot = aggregate_per_plot(
        alive,
        group_by=["specie", "plot_id", "year"],
        extra_aggs=[
            pl.col("height").mean(),
            pl.col("lat").mean(),
            pl.col("lon").mean(),
            pl.col("date").min(),
            pl.col("altitude").first(),
            pl.col("area_m2").first(),
        ],
    )

    return (
        scale_to_hectare(per_plot, pl.col("area_m2") / 10000.0)
        .rename({"n_trees": "n_stems"})
        .with_columns(
            pl.col("n_stems").cast(pl.Int64),
            pl.lit("alive").alias("status"),
        )
        .drop(["area_m2"])
        .select(
            [
                "specie",
                "plot_id",
                "year",
                "date",
                "status",
                "lon",
                "lat",
                "altitude",
                "biom_stem",
                "biom_root",
                "biom_foliage",
                "lai",
                "basal_area",
                "n_stems",
                "dbh_cm",
                "mean_dbh",
                "height",
            ]
        )
    )


def prepare_lwf_data(output_path: str | None = None) -> pl.DataFrame:
    """Load, clean, and aggregate LWF data for 3PG calibration.

    Parameters
    ----------
    output_path : str | None
        Parquet path to write the result. Defaults to
        `clean_data_folder/lwf_cleaned.parquet`.

    Returns
    -------
    pl.DataFrame
        Columns: `specie`, `plot_id`, `year`, `date`, `status`, `lon`, `lat`,
        `altitude`, `biom_stem`, `biom_root`, `biom_foliage` (t ha⁻¹),
        `lai` (m² m⁻²), `basal_area` (m² ha⁻¹), `n_stems` (ha⁻¹), `dbh_cm`
        (quadratic mean diameter), `mean_dbh` (arithmetic mean), `height`.
    """
    if output_path is None:
        output_path = str(os.path.join(clean_data_folder, "lwf_cleaned.parquet"))

    trees = prepare_lwf_tree_data()
    logger.info(
        "Loaded %d tree x year rows across %d plots", trees.height, trees["plot_id"].n_unique()
    )

    trees = _filter_single_species(trees)
    logger.info("After single-species filter: %d plots remain", trees["plot_id"].n_unique())

    result = _aggregate_per_plot(trees).sort(["specie", "plot_id", "date"])
    logger.info(
        "Aggregated to %d plot x year observations across %d plots",
        result.height,
        result["plot_id"].n_unique(),
    )

    result.write_parquet(output_path)
    logger.info("Saved to %s", output_path)
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    df = prepare_lwf_data()
    print(df.head())
    print(df.schema)
    print("\nPer-species plot counts:")
    print(df.group_by("specie").agg(pl.col("plot_id").n_unique().alias("n_plots")))
