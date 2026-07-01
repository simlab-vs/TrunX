"""Prepare EFM plot-level observation data for 3PG modelling."""

import logging
import math
import os

import geopandas as gpd
import polars as pl

from trunx.config import clean_data_folder, data_folder
from trunx.gp3.allometrics import CoefficientsDict, add_allometric_columns, load_forrester_eq3

logger = logging.getLogger(__name__)

_EFM_FOLDER = str(os.path.join(data_folder, "SwissData/EFM"))

SPECIES_LIST = ["Picea abies", "Fagus sylvatica"]

_FORRESTER_EQ3: CoefficientsDict = load_forrester_eq3(species=SPECIES_LIST)


def _load_geo(folder: str) -> pl.DataFrame:
    """Load plot metadata from the GeoPackage `plot_point` layer."""
    gdf = gpd.read_file(os.path.join(folder, "efm_geo_data.gpkg"), layer="plot_point").to_crs(
        "EPSG:4326"
    )
    gdf = gdf[gdf.geometry.notna()]
    return pl.DataFrame(
        {
            "plot": gdf["plot"].astype(int).tolist(),
            "lon": gdf.geometry.x.tolist(),
            "lat": gdf.geometry.y.tolist(),
            "elevation": gdf["elevation"].tolist(),
            "area_m2": gdf["area"].astype(float).tolist(),
            "stand_establishment": gdf["stand_establishment"].tolist(),
        }
    )


def _load_trees(folder: str) -> pl.DataFrame:
    """Load and clean the EFM tree-level CSV."""
    return (
        pl.read_csv(os.path.join(folder, "efm_tree_data.csv"), null_values=["NA"])
        .filter(pl.col("species").is_in(SPECIES_LIST))
        .rename({"diameter": "dbh_cm", "height_calc": "height_m"})
        .select(["plot", "year", "TreeNo", "species", "status", "dbh_cm", "height_m"])
    )


def _filter_single_species(trees: pl.DataFrame) -> pl.DataFrame:
    """Keep only (plot, year) measurements where alive trees are a single species."""
    # 1-alive, 2-thinned, 3-dead, 4-missing
    valid = (
        trees.filter(pl.col("status") == 1)
        .group_by(["plot", "year"])
        .agg(pl.col("species").n_unique().alias("n_species"))
        .filter(pl.col("n_species") == 1)
        .select(["plot", "year"])
    )
    return trees.join(valid, on=["plot", "year"], how="inner")


def _filter_ingrowth_measurements(trees: pl.DataFrame) -> pl.DataFrame:
    """Remove (plot, year) pairs that contain ingrowth trees."""
    plot_first = trees.group_by("plot").agg(pl.col("year").min().alias("plot_first_year"))
    tree_first = trees.group_by(["plot", "TreeNo"]).agg(
        pl.col("year").min().alias("tree_first_year")
    )
    ingrowth_years = (
        tree_first.join(plot_first, on="plot")
        .filter(pl.col("tree_first_year") > pl.col("plot_first_year"))
        .rename({"tree_first_year": "year"})
        .select(["plot", "year"])
        .unique()
    )
    return trees.join(ingrowth_years, on=["plot", "year"], how="anti")


def _filter_min_measurements(trees: pl.DataFrame, min_n: int = 2) -> pl.DataFrame:
    """Keep only plots with at least `min_n` distinct measurement years."""
    valid_plots = (
        trees.select(["plot", "year"])
        .unique()
        .group_by("plot")
        .agg(pl.len().alias("n_years"))
        .filter(pl.col("n_years") >= min_n)
        .select("plot")
    )
    return trees.join(valid_plots, on="plot", how="inner")


def _aggregate_alive(trees: pl.DataFrame, geo: pl.DataFrame) -> pl.DataFrame:
    """Aggregate alive trees to per-hectare stand quantities."""
    alive = trees.filter(pl.col("status") == 1)

    per_plot = alive.group_by(["species", "plot", "year"]).agg(
        pl.col("allo_sb_kg").sum().alias("plot_sb_kg"),
        pl.col("allo_rb_kg").sum().alias("plot_rb_kg"),
        pl.col("allo_fb_kg").sum().alias("plot_fb_kg"),
        pl.col("allo_la_m2").sum().alias("plot_la_m2"),
        (math.pi * pl.col("dbh_cm").pow(2) / 40000.0).sum().alias("plot_ba_m2"),
        pl.len().alias("plot_n_trees"),
        pl.col("dbh_cm").mean().alias("dbh_cm"),
        pl.col("height_m").mean().alias("height_m"),
    )

    return (
        per_plot.join(geo, on="plot", how="inner")
        .with_columns(
            # Convert per-plot quantities to per-hectare
            (pl.col("plot_sb_kg") * 10000.0 / pl.col("area_m2") / 1000.0).alias("biom_stem"),
            (pl.col("plot_rb_kg") * 10000.0 / pl.col("area_m2") / 1000.0).alias("biom_root"),
            (pl.col("plot_fb_kg") * 10000.0 / pl.col("area_m2") / 1000.0).alias("biom_foliage"),
            (pl.col("plot_la_m2") / pl.col("area_m2")).alias("lai"),
            (pl.col("plot_ba_m2") * 10000.0 / pl.col("area_m2")).alias("basal_area"),
            (pl.col("plot_n_trees") * 10000.0 / pl.col("area_m2")).cast(pl.Int64).alias("n_trees"),
            (pl.col("year").cast(pl.String) + pl.lit("-07-01"))
            .str.strptime(pl.Date, "%Y-%m-%d")
            .alias("date"),
            pl.lit("alive").alias("status"),
        )
        .drop(
            [
                "plot_sb_kg",
                "plot_rb_kg",
                "plot_fb_kg",
                "plot_la_m2",
                "plot_ba_m2",
                "plot_n_trees",
                "area_m2",
                "stand_establishment",
            ]
        )
        .select(
            [
                "species",
                "plot",
                "year",
                "date",
                "status",
                "lon",
                "lat",
                "elevation",
                "biom_stem",
                "biom_root",
                "biom_foliage",
                "lai",
                "basal_area",
                "n_trees",
                "dbh_cm",
                "height_m",
            ]
        )
    )


def prepare_efm_data(output_path: str | None = None) -> pl.DataFrame:
    """Load, clean, and aggregate EFM data for 3PG calibration."""
    if output_path is None:
        output_path = str(os.path.join(clean_data_folder, "efm_cleaned.parquet"))

    geo = _load_geo(_EFM_FOLDER)
    logger.info("Loaded geo data for %d plots", geo.height)

    trees = _load_trees(_EFM_FOLDER)
    logger.info(
        "Loaded %d tree x year rows across %d plots",
        trees.height,
        trees["plot"].n_unique(),
    )

    trees = _filter_single_species(trees)
    logger.info("After single-species filter: %d plots remain", trees["plot"].n_unique())

    trees = _filter_ingrowth_measurements(trees)
    logger.info(
        "After ingrowth filter: %d (plot, year) measurements remain",
        trees.select(["plot", "year"]).unique().height,
    )

    trees = _filter_min_measurements(trees, min_n=2)
    logger.info("After ≥2-measurement filter: %d plots remain", trees["plot"].n_unique())

    trees = add_allometric_columns(trees, _FORRESTER_EQ3, species_col="species")

    d_obs = _aggregate_alive(trees, geo)
    logger.info(
        "Aggregated to %d alive-tree observations across %d plots",
        d_obs.height,
        d_obs["plot"].n_unique(),
    )

    result = d_obs.sort(["species", "plot", "date"])

    result.write_parquet(output_path)
    logger.info("Saved to %s", output_path)
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    df = prepare_efm_data()
    print(df.head())
    print(df.schema)
    print("\nPer-species plot counts:")
    print(df.group_by("species").agg(pl.col("plot").n_unique().alias("n_plots")))
