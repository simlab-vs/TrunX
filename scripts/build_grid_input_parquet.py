"""Build a nested parquet table from the flattened grid_input CSV export."""

import os
import warnings

import polars as pl
import rdata

from trunx.config import project_root, threepg_data_folder

CLIMATE_COLS = [
    "tmp_min",
    "tmp_max",
    "tmp_ave",
    "prcp",
    "srad",
    "frost_days",
    "vpd_day",
    "co2",
    "d13catm",
    "n_days",
    "month",
]
SITE_COLS = ["latitude", "altitude", "soil_class", "asw_i", "asw_min", "asw_max", "from", "to"]
SPECIES_COLS = [
    "species",
    "planted",
    "fertility",
    "stems_n",
    "biom_stem",
    "biom_root",
    "biom_foliage",
]
THINNING_COLS = ["species", "age", "stems_n", "stem", "root", "foliage"]


def _nest_by_grid(
    df: pl.DataFrame, cols: list[str], name: str, sort_by: str | None = None
) -> pl.DataFrame:
    """Group rows by grid_id, packing `cols` into a list-of-struct column named `name`."""
    if sort_by is not None:
        df = df.sort(["grid_id", sort_by])
    return df.group_by("grid_id", maintain_order=True).agg(pl.struct(cols).alias(name))


def _broadcast_row(df: pl.DataFrame, cols: list[str], name: str) -> pl.DataFrame:
    """Wrap a single shared template row as a one-row, list-of-struct frame for a cross join."""
    return df.select(cols).to_struct(name).implode().to_frame()


def build_grid_input_parquet(input_file: str, output_path: str) -> pl.DataFrame:
    """Build one nested-parquet table from the flattened grid_input CSV export.

    Parameters
    ----------
    input_file : str
        Path to the input RData file containing the grid data.
    output_path : str
        Destination parquet file path.

    Returns
    -------
    pl.DataFrame
        The combined nested table that was written to `output_path`.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        converted = rdata.conversion.convert(rdata.parser.parse_file(input_file))

    print("Available objects in grid_input.rda:", list(converted.keys()))

    coord = pl.DataFrame(converted["coord.grid"])
    site_grid = converted["site.grid"]
    climate_grid = converted["climate.grid"]
    species = pl.DataFrame(converted["species.grid"])

    # Site data
    sites = []
    for id, site in zip(site_grid["grid_id"], site_grid["site"], strict=True):
        site_df = pl.DataFrame(site)
        site_df = site_df.with_columns(grid_id=pl.lit(id))
        sites.append(site_df)
    sites = pl.concat(sites)

    # Climate data
    climates = []
    for grid_id, climate in zip(climate_grid["grid_id"], climate_grid["forc"], strict=True):
        climate_df = pl.DataFrame(climate)
        climate_df = climate_df.with_columns(
            grid_id=pl.lit(grid_id),
            month=pl.arange(1, climate_df.height + 1),
            n_days=pl.Series([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]),
        )

        climates.append(climate_df)

    climates = pl.concat(climates)

    # Only keep grid cells that have both site and climate data.
    grid_ids = (
        sites.select("grid_id").unique().join(climates.select("grid_id").unique(), on="grid_id")
    )

    site_nested = _nest_by_grid(sites, SITE_COLS, "site")
    climate_nested = _nest_by_grid(climates, CLIMATE_COLS, "climate", sort_by="month")

    species = species.with_columns(
        pl.when(pl.col("species") == "piab")
        .then(pl.lit("Picea abies"))
        .otherwise(pl.col("species"))
        .alias("species")
    )
    species_row = _broadcast_row(species, SPECIES_COLS, "species")

    result = (
        grid_ids.join(coord, on="grid_id")
        .join(climate_nested, on="grid_id")
        .join(site_nested, on="grid_id")
        .join(species_row, how="cross")
        .select("grid_id", "x", "y", "climate", "site", "species")
    )
    result = result.unique(subset="grid_id").sort("grid_id")
    result.write_parquet(output_path)
    return result


if __name__ == "__main__":
    input_file = os.path.join(
        project_root, "models/r3PG/vignettes_build/vignette_data/grid_input.rda"
    )

    output_file = os.path.join(threepg_data_folder, "grid_input.parquet")

    df = build_grid_input_parquet(input_file, output_file)
