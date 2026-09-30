"""Prepare per-species parquet files of ICP plot data for Bayesian parameter estimation.

Plot inputs are built by `create_combined_inputs.build_plot_input_sheets`, the same
builder as the 3PG input files. Only plots whose input has a single supported species
are included. One parquet file is written per species, named
``icp_plot_data_{Species_name}.parquet``. Each row corresponds to one plot with nested
list-of-struct columns.

Parameters are excluded; pass them separately to the estimation pipeline.
"""

import logging
import os
from pathlib import Path

import pandas as pd
import polars as pl

from trunx.config import clean_data_folder, threepg_data_folder
from trunx.gp3.age_regression import fit_models
from trunx.gp3.create_combined_inputs import _load_species_param_bound, build_plot_input_sheets
from trunx.gp3.prepare_climate import prepare_climate
from trunx.gp3.prepare_deposition import load_monthly_deposition
from trunx.gp3.prepare_site import prepare_site

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
    "observed": ["specie", "month", "year", "GPP", "DBH", "WS", "WF", "WR", "LAI"],
}


def _build_plot_row(
    plot_id: str,
    plot_data: pl.DataFrame,
    weather_data: pl.DataFrame,
    age_models: dict[str, tuple[float, float]],
    deposition: pl.DataFrame,
    gpp: pl.DataFrame,
) -> tuple[str, pl.DataFrame] | None:
    """Build a single-row DataFrame for one ICP plot with nested section columns.

    Parameters
    ----------
    plot_id : str
        ICP plot identifier.
    plot_data : pl.DataFrame
        Combined plot-level table.
    weather_data : pl.DataFrame
        Combined monthly weather.
    age_models : dict[str, tuple[float, float]]
        Per-species age-vs-DBH models from `trunx.gp3.age_regression.fit_models`.
    deposition : pl.DataFrame
        Monthly ICP deposition.
    gpp : pl.DataFrame
        Monthly GOSIF GPP of ICP plots.

    Returns
    -------
    tuple[str, pl.DataFrame] | None
        Species name and one-row DataFrame with columns ``plot_id``, ``climate``,
        ``site``, ``species``, ``observed`` as list-of-struct. ``None`` if the plot
        is not single-species or its input is invalid.
    """
    sheets = {
        name: pl.from_pandas(sheet)
        for name, sheet in build_plot_input_sheets(
            plot_id,
            "ICP",
            plot_data,
            weather_data,
            age_models=age_models,
            deposition=deposition,
            gpp=gpp,
        ).items()
        if name in SECTION_COLS
    }
    species = sheets["species"]["species"].to_list()
    if len(species) != 1 or species[0] not in SUPPORTED_SPECIES:
        logger.info(
            "plot_id %s: species %s not a single supported one — skipping", plot_id, species
        )
        return None

    try:
        prepare_site(sheets["site"])
        site_row = sheets["site"].row(0, named=True)
        prepare_climate(sheets["climate"], site_row["from"], site_row["to"])
    except ValueError as exc:
        logger.warning("plot_id %s: invalid input (%s) — skipping", plot_id, exc)
        return None

    row = pl.DataFrame(
        {"plot_id": [plot_id]}
        | {
            name: sheet.select(SECTION_COLS[name]).to_struct(name=name).implode()
            for name, sheet in sheets.items()
        }
    )
    return species[0], row


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
    icp = pl.col("source") == "ICP"
    plot_data = pl.read_parquet(
        os.path.join(clean_data_folder, "trunx_plot_level_data.parquet")
    ).filter(icp)
    weather_data = pl.read_parquet(
        os.path.join(clean_data_folder, "trunx_plot_weather.parquet")
    ).filter(icp)
    age_models = fit_models()
    deposition = load_monthly_deposition()
    gpp = pl.read_csv(os.path.join(clean_data_folder, "GOSIF_GPP_icp.csv"))

    plot_ids = (
        plot_data.filter(pl.col("specie").is_in(SUPPORTED_SPECIES))["plot_id"]
        .unique()
        .sort()
        .to_list()
    )
    by_species: dict[str, list[pl.DataFrame]] = {}
    for counter, plot_id in enumerate(plot_ids, start=1):
        try:
            result = _build_plot_row(plot_id, plot_data, weather_data, age_models, deposition, gpp)
        except ValueError as exc:
            logger.warning("(%d/%d) skipping plot_id %s: %s", counter, len(plot_ids), plot_id, exc)
            continue
        if result is not None:
            species_name, row = result
            by_species.setdefault(species_name, []).append(row)
            logger.info("(%d/%d) processed plot_id %s", counter, len(plot_ids), plot_id)

    if not by_species:
        raise RuntimeError("No plot data could be collected — check the combined tables")

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
        Keys into `create_combined_inputs._LITERATURE_SOURCES`.
    species_names : tuple[str, ...]
        Species to build files for. Defaults to the three species this project's
        multiplot pipeline actually calibrates (see `bayesian_config.species_plot_ids`);
        Trotsiuk's literature table only covers Picea abies and Fagus sylvatica.
    """
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
