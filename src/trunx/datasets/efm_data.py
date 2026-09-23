"""Prepare EFM plot-level observation data for 3PG modelling."""

import logging
import math
import os

import geopandas as gpd
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

_EFM_FOLDER = str(os.path.join(data_folder, "SwissData/EFM"))

SPECIES_LIST = [
    "Abies alba",
    "Acer",
    "Acer campestre",
    "Acer platanoides",
    "Acer pseudoplatanus",
    "Alnus",
    "Alnus glutinosa",
    "Alnus incana",
    "Betula",
    "Betula pendula",
    "Carpinus betulus",
    "Castanea sativa",
    "Corylus avellana",
    "Fagus sylvatica",
    "Frangula alnus",
    "Fraxinus excelsior",
    "Ilex aquifolium",
    "Juglans regia",
    "Juniperus communis",
    "Larix decidua",
    "Larix kaempferi (Lamb.) Carrière",
    "Malus sylvestris",
    "Other broadleaves",
    "Picea abies",
    "Picea sitchensis",
    "Pinus cembra",
    "Pinus mugo Turra subsp. mugo",
    "Pinus nigra",
    "Pinus strobus",
    "Pinus sylvestris",
    "Populus",
    "Populus tremula",
    "Prunus avium",
    "Pseudotsuga menziesii",
    "Pyrus pyraster",
    "Quercus",
    "Quercus pubescens",
    "Quercus robur",
    "Quercus rubra",
    "Robinia pseudoacacia",
    "Salix",
    "Salix alba",
    "Salix caprea",
    "Salix viminalis",
    "Sorbus",
    "Sorbus aria",
    "Sorbus aucuparia",
    "Sorbus torminalis",
    "Taxus baccata",
    "Tilia",
    "Tilia cordata",
    "Tilia platyphyllos",
    "Ulmus",
    "Ulmus glabra",
]

# Species without a complete set of Forrester et al. (2017) Table A.5
# coefficients (e.g. genus-only entries like "Quercus") get null biomass —
# load every species with complete coefficients rather than requiring all
# of SPECIES_LIST to have them.
_FORRESTER_EQ3: CoefficientsDict = load_forrester_eq3()


def _load_geo(folder: str) -> pl.DataFrame:
    """Load plot metadata from the GeoPackage `plot_point` layer."""
    gdf = gpd.read_file(os.path.join(folder, "efm_geo_data.gpkg"), layer="plot_point").to_crs(
        "EPSG:4326"
    )
    gdf = gdf[gdf.geometry.notna()]
    return pl.DataFrame(
        {
            "plot_id": gdf["plot"].astype(int).tolist(),
            "lon": gdf.geometry.x.tolist(),
            "lat": gdf.geometry.y.tolist(),
            "altitude": gdf["elevation"].tolist(),
            "area_m2": gdf["area"].astype(float).tolist(),
            "stand_establishment": gdf["stand_establishment"].tolist(),
        }
    )


def _load_trees(folder: str) -> pl.DataFrame:
    """Load and clean the EFM tree-level CSV."""
    return (
        pl.read_csv(os.path.join(folder, "efm_tree_data.csv"), null_values=["NA"])
        .filter(pl.col("species").is_in(SPECIES_LIST))
        .drop("height")
        .rename(
            {
                "diameter": "dbh_cm",
                "height_calc": "height",
                "species": "specie",
                "plot": "plot_id",
                "TreeNo": "tree_id",
            }
        )
        .select(["plot_id", "year", "tree_id", "specie", "status", "dbh_cm", "height"])
    )


def prepare_efm_tree_data(output_path: str | None = None) -> pl.DataFrame:
    """Load and clean EFM tree-level data.

    Parameters
    ----------
    output_path : str | None
        Parquet path to write the result. Defaults to
        `clean_data_folder/efm_tree_data.parquet`.

    Returns
    -------
    pl.DataFrame
        One row per tree x year, joined with plot coordinates and area,
        for the target species. Not filtered to single-species plots — see
        `prepare_efm_data` for that. Columns include `plot_id`, `tree_id`,
        `specie`, `lat`, `lon`, `altitude`, `date` (synthetic, 1 July of
        `year`, since the source only records measurement year), `year`,
        `month` (always 7, from the synthetic date), `dbh_cm`, `height`,
        `n_stems` (always 1, one physical tree per row), `biom_stem`,
        `biom_foliage`, `biom_root` (kg tree⁻¹), `la_m2` (leaf area, m²
        tree⁻¹) and `basal_area` (m² tree⁻¹).
    """
    if output_path is None:
        output_path = str(os.path.join(clean_data_folder, "efm_tree_data.parquet"))

    trees = (
        _load_trees(_EFM_FOLDER)
        .join(_load_geo(_EFM_FOLDER), on="plot_id", how="inner")
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
            (pl.col("year").cast(pl.String) + pl.lit("-07-01"))
            .str.strptime(pl.Date, "%Y-%m-%d")
            .alias("date"),
        )
        .with_columns(pl.col("date").dt.month().alias("month"))
    )

    trees.write_parquet(output_path)
    logger.info("Saved %d rows to %s", trees.height, output_path)
    return trees


def _filter_single_species(trees: pl.DataFrame) -> pl.DataFrame:
    """Keep only (plot, year) measurements where alive trees are a single species."""
    # 1-alive, 2-thinned, 3-dead, 4-missing
    valid = (
        trees.filter(pl.col("status") == 1)
        .group_by(["plot_id", "year"])
        .agg(pl.col("specie").n_unique().alias("n_species"))
        .filter(pl.col("n_species") == 1)
        .select(["plot_id", "year"])
    )
    return trees.join(valid, on=["plot_id", "year"], how="inner")


def _filter_ingrowth_measurements(trees: pl.DataFrame) -> pl.DataFrame:
    """Remove (plot, year) pairs that contain ingrowth trees."""
    plot_first = trees.group_by("plot_id").agg(pl.col("year").min().alias("plot_first_year"))
    tree_first = trees.group_by(["plot_id", "tree_id"]).agg(
        pl.col("year").min().alias("tree_first_year")
    )
    ingrowth_years = (
        tree_first.join(plot_first, on="plot_id")
        .filter(pl.col("tree_first_year") > pl.col("plot_first_year"))
        .rename({"tree_first_year": "year"})
        .select(["plot_id", "year"])
        .unique()
    )
    return trees.join(ingrowth_years, on=["plot_id", "year"], how="anti")


def _filter_min_measurements(trees: pl.DataFrame, min_n: int = 2) -> pl.DataFrame:
    """Keep only plots with at least `min_n` distinct measurement years."""
    valid_plots = (
        trees.select(["plot_id", "year"])
        .unique()
        .group_by("plot_id")
        .agg(pl.len().alias("n_years"))
        .filter(pl.col("n_years") >= min_n)
        .select("plot_id")
    )
    return trees.join(valid_plots, on="plot_id", how="inner")


def _aggregate_alive(trees: pl.DataFrame) -> pl.DataFrame:
    """Aggregate alive trees to per-hectare stand quantities."""
    alive = trees.filter(pl.col("status") == 1)

    per_plot = aggregate_per_plot(
        alive,
        group_by=["specie", "plot_id", "year"],
        extra_aggs=[
            pl.col("height").mean(),
            pl.col("lon").first(),
            pl.col("lat").first(),
            pl.col("altitude").first(),
            pl.col("area_m2").first(),
        ],
    )

    return (
        scale_to_hectare(per_plot, pl.col("area_m2") / 10000.0)
        .rename({"n_trees": "n_stems"})
        .with_columns(
            pl.col("n_stems").cast(pl.Int64),
            (pl.col("year").cast(pl.String) + pl.lit("-07-01"))
            .str.strptime(pl.Date, "%Y-%m-%d")
            .alias("date"),
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
                "dbh_qmd",
                "dbh_mean",
                "dbh_std",
                "height",
            ]
        )
    )


def prepare_efm_data(output_path: str | None = None) -> pl.DataFrame:
    """Load, clean, and aggregate EFM data for 3PG calibration."""
    if output_path is None:
        output_path = str(os.path.join(clean_data_folder, "efm_cleaned.parquet"))

    trees = prepare_efm_tree_data()
    logger.info(
        "Loaded %d tree x year rows across %d plots",
        trees.height,
        trees["plot_id"].n_unique(),
    )

    trees = _filter_single_species(trees)
    logger.info("After single-species filter: %d plots remain", trees["plot_id"].n_unique())

    trees = _filter_ingrowth_measurements(trees)
    logger.info(
        "After ingrowth filter: %d (plot, year) measurements remain",
        trees.select(["plot_id", "year"]).unique().height,
    )

    trees = _filter_min_measurements(trees, min_n=2)
    logger.info("After ≥2-measurement filter: %d plots remain", trees["plot_id"].n_unique())

    d_obs = _aggregate_alive(trees)
    logger.info(
        "Aggregated to %d alive-tree observations across %d plots",
        d_obs.height,
        d_obs["plot_id"].n_unique(),
    )

    result = d_obs.sort(["specie", "plot_id", "date"])

    result.write_parquet(output_path)
    logger.info("Saved to %s", output_path)
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    df = prepare_efm_data()
    print(df.head())
    print(df.schema)
    print("\nPer-species plot counts:")
    print(df.group_by("specie").agg(pl.col("plot_id").n_unique().alias("n_plots")))
