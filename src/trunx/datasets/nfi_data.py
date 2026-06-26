"""Prepare Swiss NFI plot-level observation data for 3PG modelling.

Data limitations
----------------
Missing: Tree height, info about even or uneven aged.

Unit conventions
----------------
- `D13` (DBH) in the source CSV is in **cm**.
- `RPSTZ` is the per-hectare expansion factor.
- All output biomass columns are in t ha⁻¹ (divided by 1 000).
- Basal area (`ba`) is in m² ha⁻¹.
- Leaf area index (`la`) is (dimensionless).

"""

import logging
import math
import os

import polars as pl
from pyproj import Transformer

from trunx.config import clean_data_folder, data_folder
from trunx.gp3.allometrics import CoefficientsDict, add_allometric_columns, load_forrester_eq3

logger = logging.getLogger(__name__)

_NFI_RAW_FOLDER = str(os.path.join(data_folder, "SwissData/NFI"))
_INVENTORY_STEP = 100

# Refer metadata to get the codes for the species.
SPECIES_CODES: dict[int, str] = {
    10: "Picea abies",
    50: "Fagus sylvatica",
}

_FORRESTER_EQ3: CoefficientsDict = load_forrester_eq3(species=list(SPECIES_CODES.values()))


def _load_plots() -> pl.DataFrame:
    """Load and clean the NFI plot-level CSV.

    Returns
    -------
    pl.DataFrame
        Columns: `plot_id`, `inv_nr`, `lat`, `lon`, `elevation`,
        ``date``.
    """
    df = pl.read_csv(os.path.join(_NFI_RAW_FOLDER, "Givi_plot_level.csv")).rename(
        {"CLNR": "plot_id", "INVNR": "inv_nr", "Z25": "elevation"}
    )

    transformer = Transformer.from_crs("EPSG:21781", "EPSG:4326", always_xy=True)
    lon_arr, lat_arr = transformer.transform(df["X"].to_numpy(), df["Y"].to_numpy())

    return df.with_columns(
        [
            pl.Series("lat", lat_arr),
            pl.Series("lon", lon_arr),
            pl.col("DATUMF").str.strptime(pl.Date, "%d/%m/%Y").alias("date"),
        ]
    ).select(["plot_id", "inv_nr", "lat", "lon", "elevation", "date"])


def _load_trees() -> pl.DataFrame:
    """Load and clean the NFI tree-level CSV.

    Returns
    -------
    pl.DataFrame
        One row per tree × inventory, filtered to target species.
    """
    return (
        pl.read_csv(os.path.join(_NFI_RAW_FOLDER, "Givi_tree_level.csv"))
        .rename(
            {
                "CLNR": "plot_id",
                "BANR": "tree_id",
                "INVNR": "inv_nr",
                "BARTLFI": "species_code",
                "RPSTZ": "tree_rep_fact",
                "D13": "dbh_cm",
                "VMRDBIOM": "biom_stem_kg",
                "ASTDHBIOM": "biom_branch_kg",
                "REISIGBIOM": "biom_brushwood_kg",
                "WURZELN": "biom_root_kg",
                "NADELN": "biom_foliage_kg",
            }
        )
        .filter(pl.col("species_code").is_in(list(SPECIES_CODES.keys())))
        .with_columns(pl.col("species_code").replace_strict(SPECIES_CODES).alias("species"))
        .drop(["BIOMASSE", "species_code"])
    )


def _filter_single_species_plots(trees: pl.DataFrame) -> pl.DataFrame:
    """Keep only single-species plots that are consistently one species.

    A plot is kept if every inventory observation contains exactly one species
    and that species is the same across all inventories for the plot.

    Parameters
    ----------
    trees : pl.DataFrame

    Returns
    -------
    pl.DataFrame
        Filtered tree-level data with an added ``species`` column at plot level.
    """
    species_per_inv = (
        trees.group_by(["plot_id", "inv_nr"])
        .agg(
            pl.col("species").n_unique().alias("n_species"),
            pl.col("species").first().alias("species"),
        )
        .filter(pl.col("n_species") == 1)
    )

    single_species_plots = (
        species_per_inv.group_by("plot_id")
        .agg(
            pl.col("species").n_unique().alias("n_distinct_species"),
            pl.col("species").first().alias("plot_species"),
        )
        .filter(pl.col("n_distinct_species") == 1)
        .select(["plot_id", "plot_species"])
    )

    return trees.join(single_species_plots, on="plot_id", how="inner")


def _filter_consecutive_inventories(df: pl.DataFrame) -> pl.DataFrame:
    """Keep only plot x inventory rows that belong to consecutive inventory runs.

    Consecutive means successive Swiss NFI cycles (inventory number step = 100).
    Runs with fewer than two inventories are dropped.

    Parameters
    ----------
    df : pl.DataFrame
        Data with `plot_id` and `inv_nr` columns.

    Returns
    -------
    pl.DataFrame
        Input rows restricted to valid consecutive groups.
    """
    grouped = (
        df.select(["plot_id", "inv_nr"])
        .unique()
        .sort(["plot_id", "inv_nr"])
        .with_columns(
            (pl.col("inv_nr") - pl.col("inv_nr").shift(1).over("plot_id")).alias("inv_diff")
        )
        .with_columns(
            (pl.col("inv_diff").is_null() | (pl.col("inv_diff") != _INVENTORY_STEP))
            .cast(pl.Int32)
            .alias("new_group")
        )
        .with_columns(pl.col("new_group").cum_sum().over("plot_id").alias("consec_group"))
    )

    valid_groups = (
        grouped.group_by(["plot_id", "consec_group"])
        .agg(pl.len().alias("n_inv"))
        .filter(pl.col("n_inv") == 5)
        .select(["plot_id", "consec_group"])
    )

    keep = grouped.join(valid_groups, on=["plot_id", "consec_group"]).select(["plot_id", "inv_nr"])
    return df.join(keep, on=["plot_id", "inv_nr"], how="inner")


def _aggregate_per_plot(trees: pl.DataFrame, plots: pl.DataFrame) -> pl.DataFrame:
    """Aggregate tree-level data to plot-level per-hectare quantities.

    Biomass components are computed using allometric eqns from Forrester et al. (2017).
    Leaf area index is ``sum(la_m2_per_tree * rep_fact) / 10000``.

    Parameters
    ----------
    trees : pl.DataFrame
        Filtered tree-level data with allometric columns.
    plots : pl.DataFrame
        Plot-level data with coordinates and dates.

    Returns
    -------
    pl.DataFrame
        One row per (plot_id, inv_nr) with per-hectare stand quantities.
    """
    per_ha = trees.group_by(["plot_id", "inv_nr", "plot_species"]).agg(
        pl.col("tree_rep_fact").sum().alias("n_stems"),
        ((pl.col("dbh_cm") * pl.col("tree_rep_fact")).sum() / pl.col("tree_rep_fact").sum()).alias(
            "dbh_cm"
        ),
        (pl.col("allo_sb_kg") * pl.col("tree_rep_fact")).sum().alias("biom_stem"),
        (pl.col("allo_fb_kg") * pl.col("tree_rep_fact")).sum().alias("biom_foliage"),
        (pl.col("allo_rb_kg") * pl.col("tree_rep_fact")).sum().alias("biom_root"),
        (pl.col("allo_la_m2") * pl.col("tree_rep_fact")).sum().alias("la_m2_ha"),
        (math.pi * pl.col("dbh_cm").pow(2) / 40000.0 * pl.col("tree_rep_fact"))
        .sum()
        .alias("basal_area"),
    )

    return (
        per_ha.join(plots, on=["plot_id", "inv_nr"], how="inner")
        .rename({"plot_species": "species"})
        .with_columns(
            (pl.col("biom_stem") / 1000.0).alias("biom_stem"),
            (pl.col("biom_foliage") / 1000.0).alias("biom_foliage"),
            (pl.col("biom_root") / 1000.0).alias("biom_root"),
            (pl.col("la_m2_ha") / 10000.0).alias("lai"),
            pl.col("date").dt.year().alias("year"),
        )
        .drop("la_m2_ha")
        .sort(["plot_id", "date"])
    )


def prepare_nfi_data(output_path: str | None = None) -> pl.DataFrame:
    """Load, clean, and aggregate Swiss NFI data for 3PG calibration.

    Parameters
    ----------
    output_path : str | None
        Parquet path to write the result.  Defaults to
        `clean_data_folder/nfi_cleaned.parquet`.

    Returns
    -------
    pl.DataFrame
        Columns: `plot_id`, `species`, `date`, `year`, `lat`,
        `lon`, `elevation`, `n_stems` (ha⁻¹), `dbh_cm`,
        `biom_stem`, `biom_foliage`, `biom_root` (all t ha⁻¹),
        `lai` (m² m⁻²), `basal_area` (m² ha⁻¹).
    """
    if output_path is None:
        output_path = str(os.path.join(clean_data_folder, "nfi_cleaned.parquet"))

    plots = _load_plots()
    logger.info("Loaded %d plot x inventory rows", plots.height)

    trees = _load_trees()
    logger.info(
        "Loaded %d tree x inventory rows across %d species",
        trees.height,
        trees["species"].n_unique(),
    )

    trees = _filter_single_species_plots(trees)
    logger.info(
        "After single-species filter: %d plot x inventory x tree rows, %d plots",
        trees.height,
        trees["plot_id"].n_unique(),
    )

    trees = _filter_consecutive_inventories(trees)
    logger.info(
        "After consecutive-inventory filter: %d plots remain",
        trees["plot_id"].n_unique(),
    )

    trees = add_allometric_columns(trees, _FORRESTER_EQ3)

    result = _aggregate_per_plot(trees, plots)
    logger.info(
        "Aggregated to %d plot x inventory observations across %d plots",
        result.height,
        result["plot_id"].n_unique(),
    )

    result.write_parquet(output_path)
    logger.info("Saved to %s", output_path)
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    df = prepare_nfi_data()
    print(df.head())
    print("\nPer-species plot counts:")
    print(df.group_by("species").agg(pl.col("plot_id").n_unique().alias("n_plots")))
