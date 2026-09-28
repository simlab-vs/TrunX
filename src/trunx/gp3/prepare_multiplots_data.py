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

import datetime
import logging
import os
from pathlib import Path

import pandas as pd
import polars as pl

from trunx.config import clean_data_folder, threepg_data_folder
from trunx.gp3.age_regression import fit_models
from trunx.gp3.allometrics import dms_to_decimal
from trunx.gp3.create_data_inputs import (
    create_observation_data,
    create_site_data,
    create_species_data,
)
from trunx.gp3.prepare_climate import prepare_climate, trim_to_window
from trunx.gp3.prepare_site import prepare_site
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

    if species_df.height != 1:
        # This dataset is single-species-per-plot by design (see module docstring); a
        # second row here means create_species_data picked up more than one census date
        # within the same start_year (e.g. two surveys in one calendar year) rather than
        # a genuinely different species. Caught here rather than left for
        # load_observations_from_section to reject at load time.
        logger.warning(
            "plot_id %s: species table has %d rows (expected exactly 1) — skipping",
            plot_id,
            species_df.height,
        )
        return None

    _, weather_df = fill_weather_with_era5(weather_df, plot_id, start_year)

    icp_filtered = icp_df.filter(pl.col("specie").is_in(species_df["species"].to_list()))

    observed_df = create_observation_data(plot_id, icp_filtered, start_year)
    site_df = create_site_data(icp_df, weather_df, observed_df)

    try:
        prepare_site(site_df)
    except ValueError as exc:
        logger.warning("plot_id %s: invalid site data (%s) — skipping", plot_id, exc)
        return None

    # Save exactly the window the model will simulate — the same one `prepare_climate`
    # trims to at load time — so an observation's month index (assigned by row position
    # against this same section) can't drift out of alignment with it. See
    # `load_files.load_observations_from_section`.
    site_row = site_df.row(0, named=True)
    weather_df = trim_to_window(
        weather_df,
        datetime.date.fromisoformat(site_row["from"] + "-01"),
        datetime.date.fromisoformat(site_row["to"] + "-01"),
    )

    try:
        prepare_climate(weather_df, site_row["from"], site_row["to"])
    except ValueError as exc:
        logger.warning("plot_id %s: invalid climate data (%s) — skipping", plot_id, exc)
        return None

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


def prepare_multiplot_param_bounds(
    output_dir: Path | str,
    literature_sources: tuple[str, ...] = ("Forrester", "Trotsiuk"),
    species_names: tuple[str, ...] = ("Picea abies", "Pinus sylvestris", "Fagus sylvatica"),
) -> None:
    """Build one params_bounds parquet per (species, literature_source) pair.

    Parameters
    ----------
    output_dir : Path | str
        Directory to write the parquet files to. Files are named
        ``params_bounds_{literature_source}_{Species_name}.parquet``.
    literature_sources : tuple[str, ...]
        Keys into `pymc_icp_plots._LITERATURE_SOURCES`.
    species_names : tuple[str, ...]
        Species to build files for. Defaults to the three species this project's
        multiplot pipeline actually calibrates (see `bayesian_config.species_plot_ids`);
        Trotsiuk's literature table only covers Picea abies and Fagus sylvatica.
    """
    from trunx.gp3.bayesiancalibrations.pymc_icp_plots import _load_species_param_bound

    output_dir = Path(output_dir)

    error_bound = pd.read_excel(
        os.path.join(threepg_data_folder, "solling_data.xlsx"), sheet_name="error_param"
    )
    error_bound[["default", "min", "max"]] = error_bound[["default", "min", "max"]].astype(
        "float64"
    )

    for literature_source in literature_sources:
        for species_name in species_names:
            try:
                param_bound = _load_species_param_bound(
                    species_name, literature_source=literature_source
                )
            except ValueError as exc:
                logger.warning(
                    "No %s bounds for '%s' — skipping (%s)", literature_source, species_name, exc
                )
                continue

            param_bound = pd.concat([param_bound, error_bound], ignore_index=True)

            species_slug = species_name.replace(" ", "_")
            filename = f"params_bounds_{literature_source}_{species_slug}.parquet"
            pl.from_pandas(param_bound).write_parquet(output_dir / filename)
            logger.info(
                "Wrote %d parameters for '%s' (%s) to %s",
                len(param_bound),
                species_name,
                literature_source,
                filename,
            )


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

    prepare_multiplot_param_bounds(threepg_data_folder)

    # Example: read climate data for one plot from the Picea abies file
    df = pl.read_parquet(os.path.join(threepg_data_folder, "icp_plot_data_Picea_abies.parquet"))
    pid = "50.0018"
    print(load_section(df, pid, "climate").head())
    print(load_section(df, pid, "site"))
    obv = load_section(df, pid, "observed")
    print(obv.drop_nulls(subset=["DBH"]))
    print(load_section(df, pid, "species"))
