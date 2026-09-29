"""Build standardized tree-level and plot-level tables across all datasets."""

import logging
import os

import polars as pl

from trunx.config import clean_data_folder
from trunx.datasets.efm_data import prepare_efm_data, prepare_efm_tree_data
from trunx.datasets.icp_level2_data import prepare_icp_plot_data, prepare_icp_tree_data
from trunx.datasets.lwf_data import prepare_lwf_data, prepare_lwf_tree_data
from trunx.datasets.nfi_data import prepare_nfi_data, prepare_nfi_tree_data
from trunx.gp3.allometrics import BIOMASS_COLS, PLOT_FLAG_METRICS

logger = logging.getLogger(__name__)

TREE_FLAG_COLUMNS = [f"flag_{c}" for c in ["height", *BIOMASS_COLS]]
PLOT_FLAG_COLUMNS = [f"flag_{c}" for c in PLOT_FLAG_METRICS]

TREE_COLUMNS = [
    "basal_area",
    "biom_foliage",
    "biom_root",
    "biom_stem",
    "date",
    "dbh_cm",
    "altitude",
    "height",
    "la_m2",
    "lat",
    "lon",
    "plot_id",
    "specie",
    "tree_id",
    *TREE_FLAG_COLUMNS,
]

PLOT_COLUMNS = [
    "altitude",
    "area_m2",
    "basal_area",
    "biom_foliage",
    "biom_root",
    "biom_stem",
    "date",
    "dbh_qmd",
    "height",
    "lai",
    "lat",
    "lon",
    "dbh_mean",
    "dbh_std",
    "n_stems",
    "plot_id",
    "specie",
    "stand_age",
    *PLOT_FLAG_COLUMNS,
]


def get_tree_level_tables() -> dict[str, pl.DataFrame]:
    """Prepare each dataset's tree-level table, restricted to `TREE_COLUMNS`.

    Returns
    -------
    dict[str, pl.DataFrame]
        Mapping of dataset name to its tree-level table.
    """
    return {
        "NFI": prepare_nfi_tree_data().select(TREE_COLUMNS),
        "EFM": prepare_efm_tree_data().select(TREE_COLUMNS),
        "LWF": prepare_lwf_tree_data().select(TREE_COLUMNS),
        "ICP": prepare_icp_tree_data().select(TREE_COLUMNS),
    }


def get_combined_tree_data(output_path: str | None = None) -> pl.DataFrame:
    """Stack every dataset's tree-level table into one, with a `source` column.

    Casts `plot_id`/`tree_id`/`flag_*` to string and `dbh_cm`/`altitude`/`height`/
    `date` to matching types across sources, since e.g. NFI/EFM use
    numeric plot and tree ids while LWF/ICP use strings.

    Parameters
    ----------
    output_path : str | None
        Parquet path to write the result. Defaults to
        `clean_data_folder/trunx_tree_level_data.parquet`.

    Returns
    -------
    pl.DataFrame
        `TREE_COLUMNS` plus `source` (one of "NFI", "EFM", "LWF", "ICP").
    """
    if output_path is None:
        output_path = str(os.path.join(clean_data_folder, "trunx_tree_level_data.parquet"))

    combined = pl.concat(
        [
            df.with_columns(
                pl.col("plot_id").cast(pl.Utf8),
                pl.col("tree_id").cast(pl.Utf8),
                pl.col("dbh_cm").cast(pl.Float64),
                pl.col("altitude").cast(pl.Float64),
                pl.col("height").cast(pl.Float64),
                pl.col("date").cast(pl.Date),
                pl.col(TREE_FLAG_COLUMNS).cast(pl.Utf8),
                pl.lit(name).alias("source"),
            )
            for name, df in get_tree_level_tables().items()
        ],
        how="vertical",
    )

    combined.write_parquet(output_path)
    logger.info("Saved %d rows to %s", combined.height, output_path)
    return combined


def _standardize_plot_table(name: str, df: pl.DataFrame) -> pl.DataFrame:
    """Cast one dataset's plot-level table to `PLOT_COLUMNS`, tagged with `source`."""
    # Only some datasets report height and stand age (years, ICP only)
    for col in ("height", "stand_age"):
        if col not in df.columns:
            df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias(col))
    return df.with_columns(
        pl.col("plot_id").cast(pl.Utf8),
        pl.col("altitude").cast(pl.Float64),
        pl.col("area_m2").cast(pl.Float64),
        # "year" is not in all datasets, so we need to handle it carefully
        # pl.col("year").cast(pl.Int64),
        pl.col("n_stems").cast(pl.Float64),
        pl.col("date").cast(pl.Date),
        pl.col("lat").cast(pl.Float64),
        pl.col("lon").cast(pl.Float64),
        pl.col("height").cast(pl.Float64),
        pl.col("stand_age").cast(pl.Float64),
        pl.col(PLOT_FLAG_COLUMNS).cast(pl.Utf8),
        pl.lit(name).alias("source"),
    ).select([*PLOT_COLUMNS, "source"])


def get_combined_plot_data(
    output_path: str | None = None, filter_single_species: bool = False
) -> pl.DataFrame:
    """Combine each dataset's already-prepared plot-level table into one.

    Uses each dataset's real `prepare_*_data`/`prepare_icp_plot_data`
    output — scaled to per-hectare quantities, one row per species — not a
    re-aggregation from tree-level data.

    Parameters
    ----------
    output_path : str | None
        Parquet path to write the result. Defaults to
        `clean_data_folder/trunx_plot_level_data.parquet`.
    filter_single_species : bool
        Forwarded to each dataset's `prepare_*` function.

    Returns
    -------
    pl.DataFrame
        `PLOT_COLUMNS` plus `source` (one of "NFI", "EFM", "LWF", "ICP").
    """
    if output_path is None:
        output_path = str(os.path.join(clean_data_folder, "trunx_plot_level_data.parquet"))

    tables = {
        "NFI": prepare_nfi_data(filter_single_species=filter_single_species),
        "EFM": prepare_efm_data(filter_single_species=filter_single_species),
        "LWF": prepare_lwf_data(filter_single_species=filter_single_species),
        "ICP": prepare_icp_plot_data(filter_single_species=filter_single_species),
    }
    combined = pl.concat(
        [_standardize_plot_table(name, df) for name, df in tables.items()],
        how="vertical",
    )

    combined.write_parquet(output_path)
    logger.info("Saved %d rows to %s", combined.height, output_path)
    return combined


if __name__ == "__main__":
    for name, df in get_tree_level_tables().items():
        print(f"\n=== {name} tree-level ({df.height} rows) ===")
        print(df.head())

    combined_trees = get_combined_tree_data()
    print(f"\n=== Combined tree-level ({combined_trees.height} rows) ===")
    print(combined_trees.group_by("source").len().sort("source"))
    combined_trees.head()

    combined_plots = get_combined_plot_data()
    print(f"\n=== Combined plot-level ({combined_plots.height} rows) ===")
    print(combined_plots.group_by("source").len().sort("source"))
    print(combined_plots.head())
