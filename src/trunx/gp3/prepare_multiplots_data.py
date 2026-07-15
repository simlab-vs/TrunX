"""Prepare per-species parquet files of ICP plot data for Bayesian parameter estimation.

Only single-species plots are included. One parquet file is written per species,
named ``icp_plot_data_{Species_name}.parquet``. Each row corresponds to one plot
with nested list-of-struct columns.

Parameters are excluded; pass them separately to the estimation pipeline.

Notes
-----
- The initial state for the 3PG model is based on the first DBH
observed data available in each plot.

TODO:
- Add option to include multi-species plots, with per-species columns for observations.

"""

import logging
import os
from pathlib import Path

import polars as pl

from trunx.config import clean_data_folder, threepg_data_folder
from trunx.gp3.age_regression import fit_models
from trunx.gp3.create_data_inputs import (
    create_observation_data,
    create_site_data,
    create_species_data,
    dms_to_decimal,
)
from trunx.gp3.weather_processing import create_weather_input, fill_weather_with_era5

logger = logging.getLogger(__name__)

SUPPORTED_SPECIES: list[str] = [
    "Picea abies",
    "Pinus sylvestris",
    "Fagus sylvatica",
    "Quercus robur",
    "Quercus petraea",
]

# Columns kept per section.
SECTION_COLS: dict[str, list[str]] = {
    "climate": ["year", "month", "tmp_ave", "tmp_min", "tmp_max", "frost_days", "prcp", "srad"],
    "site": ["latitude", "altitude", "soil_class", "asw_i", "asw_min", "asw_max", "from", "to"],
    "species": [
        "species",
        "planted",
        "fertility",
        "stems_n",
        "biom_stem",
        "biom_root",
        "biom_foliage",
    ],
    "observed": ["specie", "month", "year", "Date", "GPP", "DBH", "WS", "WF", "WR", "LAI"],
}


def _prepare_icp_df(icp_raw: pl.DataFrame, plot_id: str) -> pl.DataFrame:
    """Filter ICP level2 data for one plot and add decimal coordinates.

    Parameters
    ----------
    icp_raw : pl.DataFrame
        Full ICP level2 dataset.
    plot_id : str
        Target plot identifier.

    Returns
    -------
    pl.DataFrame
        Filtered rows for ``plot_id`` with ``Lat`` and ``Lon`` columns added.
    """
    return (
        icp_raw.filter(pl.col("plot_id") == plot_id)
        .filter(pl.col("specie").is_in(SUPPORTED_SPECIES))
        .with_columns(
            pl.col("plot_latitude")
            .map_elements(dms_to_decimal, return_dtype=pl.Float64)
            .alias("Lat"),
            pl.col("plot_longitude")
            .map_elements(dms_to_decimal, return_dtype=pl.Float64)
            .alias("Lon"),
        )
    )


def _build_plot_row(
    plot_id: str,
    weather_raw: pl.DataFrame,
    icp_raw: pl.DataFrame,
    models: dict[str, tuple[float, float]] | None = None,
) -> pl.DataFrame | None:
    """Build a single-row DataFrame for one plot with nested section columns.

    Parameters
    ----------
    plot_id : str
        Plot identifier.
    weather_raw : pl.DataFrame
        Full ICP weather dataset.
    icp_raw : pl.DataFrame
        Full ICP level2 dataset.
    models : dict[str, tuple[float, float]] | None
        Per-species power-law age models from
        :func:`trunx.gp3.age_regression.fit_models`. Passed through to
        :func:`update_species_data` to estimate the planted date from DBH.

    Returns
    -------
    pl.DataFrame | None
        One-row DataFrame with columns ``plot_id``, ``climate``, ``site``,
        ``species``, ``observed`` as list-of-struct. ``None`` if data is
        insufficient.
    """
    icp_df = _prepare_icp_df(icp_raw, plot_id)
    if icp_df.is_empty():
        logger.warning("plot_id %s: no ICP data — skipping", plot_id)
        return None

    _, weather_df = create_weather_input(weather_raw, plot_id)

    species_df, start_year = create_species_data(icp_df, models=models)
    if species_df.is_empty():
        logger.warning("plot_id %s: no species data — skipping", plot_id)
        return None

    _, weather_df = fill_weather_with_era5(weather_df, plot_id, start_year)

    icp_filtered = icp_df.filter(pl.col("specie").is_in(species_df["species"].to_list()))

    observed_df = create_observation_data(plot_id, icp_filtered, start_year)
    site_df = create_site_data(icp_df, weather_df, observed_df)

    def _to_nested(df: pl.DataFrame, section: str) -> pl.Series:
        cols = [c for c in SECTION_COLS[section] if c in df.columns]
        return df.select(cols).to_struct(name=section).implode()

    return pl.DataFrame(
        {
            "plot_id": [plot_id],
            "climate": _to_nested(weather_df, "climate"),
            "site": _to_nested(site_df, "site"),
            "species": _to_nested(species_df, "species"),
            "observed": _to_nested(observed_df, "observed"),
        }
    )


def _get_single_species_plots(icp_raw: pl.DataFrame) -> dict[str, list[str]]:
    """Return single-species plot IDs grouped by species name.

    Parameters
    ----------
    icp_raw : pl.DataFrame
        Full ICP level2 dataset.

    Returns
    -------
    dict[str, list[str]]
        Mapping of species name to sorted list of plot IDs that contain
        exactly that one supported species.
    """
    single_species_df = (
        icp_raw.filter(pl.col("specie").is_in(SUPPORTED_SPECIES))
        .group_by("plot_id")
        .agg(
            pl.col("specie").n_unique().alias("n_species"),
            pl.col("specie").first().alias("species"),
        )
        .filter(pl.col("n_species") == 1)
        .drop("n_species")
        .sort("plot_id")
    )

    logger.info(
        "Found %d single-species plots across %d species",
        single_species_df.height,
        single_species_df["species"].n_unique(),
    )

    plots_by_species: dict[str, list[str]] = {}
    for row in single_species_df.iter_rows(named=True):
        plots_by_species.setdefault(row["species"], []).append(row["plot_id"])
    return plots_by_species


def prepare_data_bayesian_opt(output_dir: Path | str) -> None:
    """Collect single-species ICP plot data and save one parquet file per species.

    Each parquet contains one row per plot with nested list-of-struct columns for
    climate, site, species, and observed data. Physics parameters are not included;
    pass them separately to the parameter estimation pipeline.

    Parameters
    ----------
    output_dir : Path | str
        Directory where per-species parquet files are written.
        Files are named ``icp_plot_data_{Species_name}.parquet``.
    """
    output_dir = Path(output_dir)

    weather_raw = pl.read_parquet(os.path.join(clean_data_folder, "ICP_weather_data.parquet"))
    # icp_raw = pl.read_parquet(os.path.join(clean_data_folder, "icp_level2_cleaned.parquet"))
    icp_raw = pl.read_parquet(os.path.join(clean_data_folder, "icp_tree_data.parquet"))
    icp_raw = icp_raw.filter(pl.col("specie").is_in(SUPPORTED_SPECIES))
    age_models = fit_models(icp_raw)

    plots_by_species = _get_single_species_plots(icp_raw)
    total = sum(len(ids) for ids in plots_by_species.values())
    by_species: dict[str, list[pl.DataFrame]] = {}
    counter = 0
    for species_name, plot_ids in plots_by_species.items():
        for plot_id in plot_ids:
            counter += 1
            try:
                row = _build_plot_row(plot_id, weather_raw, icp_raw, models=age_models)
                if row is not None:
                    by_species.setdefault(species_name, []).append(row)
                    logger.info(
                        "(%d/%d) processed plot_id %s [%s]",
                        counter,
                        total,
                        plot_id,
                        species_name,
                    )
            except Exception:
                logger.exception("(%d/%d) skipping plot_id %s", counter, total, plot_id)

    if not by_species:
        raise RuntimeError("No plot data could be collected — check ICP data sources")

    for species_name, rows in by_species.items():
        filename = f"icp_plot_data_{species_name.replace(' ', '_')}.parquet"
        pl.concat(rows, how="vertical").write_parquet(output_dir / filename)
        logger.info("Wrote %d plots for '%s' to %s", len(rows), species_name, filename)


def load_section(df: pl.DataFrame, plot_id: str, section: str) -> pl.DataFrame:
    """Load one section for one plot as a flat DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        Per-species parquet DataFrame produced by ``prepare_data_bayesian_opt``.
    plot_id : str
        Plot identifier.
    section : str
        One of ``"climate"``, ``"site"``, ``"species"``, ``"observed"``.

    Returns
    -------
    pl.DataFrame
        Flat DataFrame with only that section's columns.
    """
    return df.filter(pl.col("plot_id") == plot_id).select(section).explode(section).unnest(section)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    prepare_data_bayesian_opt(threepg_data_folder)

    # Example: read climate data for one plot from the Picea abies file
    df = pl.read_parquet(os.path.join(threepg_data_folder, "icp_plot_data_Picea_abies.parquet"))
    pid = "50.0018"
    print(load_section(df, pid, "climate").head())
    print(load_section(df, pid, "site"))
    obv = load_section(df, pid, "observed")
    print(obv.drop_nulls(subset=["DBH"]))
    print(load_section(df, pid, "species"))
