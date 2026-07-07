"""
Functions to load plot data and parameter priors from parquet files.

The parquet format is produced by ``prepare_multiplots_data.py`` and contains
nested columns for site, climate, species, and observed data. This module
flattens the nested structure and extracts the relevant data for each plot,
while also loading shared parameter priors from a separate parquet file.

TODO:
- Add functionalities to load from Excel files with separate sheets for parameters,
priors, and plot data to unifying the codes.

"""

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
import pandas as pd
import polars as pl

from trunx.config import SPECIES_INDICES
from trunx.gp3.helper_function import is_dormant
from trunx.gp3.model_inputs import ClimateData, Params, SiteData, SpeciesData, State
from trunx.gp3.prepare_climate import prepare_climate
from trunx.gp3.prepare_site import prepare_site
from trunx.gp3.prepare_species import prepare_species


class PlotData(NamedTuple):
    """All data required to run and evaluate the 3PG model for one forest plot.

    Each plot has its own site, climate, initial state, and observations, but
    shares estimated physiology parameters (Params) with other plots of the
    same species.

    Parameters
    ----------
    plot_id : str
        Plot identifier.
    initial_state : State
        Initial biomass and age state for this plot.
    climate : ClimateData
        Monthly climate time series for this plot.
    site : SiteData
        Site characteristics (latitude, soil class, ASW, etc.).
    species : SpeciesData
        Species composition, fertility rating, and planting info.
    n_species : int
        Number of species in this plot.
    observations : dict[str, tuple[jnp.ndarray, jnp.ndarray]]
        Measured variables. Keys are output names (e.g. "DBH", "WS");
        values are (obs_times, obs_values) where obs_times are integer
        month indices into the simulated time series.
    """

    plot_id: str
    initial_state: State
    climate: ClimateData
    site: SiteData
    species: SpeciesData
    n_species: int
    observations: dict[str, tuple[jnp.ndarray, jnp.ndarray]]


def load_plot_ids_from_file(plot_file: str) -> list[str]:
    """Load all plot identifiers from a parquet plot file."""
    if not plot_file.lower().endswith(".parquet"):
        raise ValueError(f"Expected parquet file for plot data, got: {plot_file}")

    plot_df = pl.read_parquet(plot_file).select("plot_id").unique().sort("plot_id")
    plot_ids = plot_df["plot_id"].to_list()
    if len(plot_ids) == 0:
        raise ValueError(f"No plot_id values found in plot file: {plot_file}")
    return plot_ids


def load_priors_from_file(
    file_path: str,
    param_names: list[str] | None = None,
) -> dict[str, tuple[float, float]]:
    """Load parameter priors from a parquet file."""
    if not file_path.lower().endswith(".parquet"):
        raise ValueError(f"Expected parquet file for priors, got: {file_path}")

    param_bounds_df = pl.read_parquet(file_path)
    priors = {}

    if param_names is None:
        bounded_df = param_bounds_df.filter(
            pl.col("min").is_not_null() & pl.col("max").is_not_null()
        )
        param_names = bounded_df["param_name"].to_list()
        if len(param_names) == 0:
            raise ValueError(
                "No optimizable parameters found in param_bound with both min and max values"
            )

    for param_name in param_names:
        row = param_bounds_df.filter(pl.col("param_name") == param_name)
        if len(row) == 0:
            raise ValueError(f"Parameter {param_name} not found in param_bound sheet")

        min_val = row["min"][0]
        max_val = row["max"][0]
        if min_val is None or max_val is None:
            raise ValueError(
                f"Parameter {param_name} must have both min and max in param_bound to be optimized"
            )

        priors[param_name] = (float(min_val), float(max_val))

    return priors


def _load_section_from_parquet(plot_df: pl.DataFrame, plot_id: str, section: str) -> pl.DataFrame:
    """Load one flattened section for one plot from nested parquet rows."""
    return (
        plot_df.filter(pl.col("plot_id") == plot_id)
        .select(section)
        .explode(section)
        .unnest(section)
    )


def load_observations_from_section(
    observed_df: pl.DataFrame,
    climate_df: pl.DataFrame,
    species_names: list[str],
) -> dict[str, tuple[jnp.ndarray, jnp.ndarray]]:
    """Load observations from a flattened parquet ``observed`` section.

    The parquet format produced by ``prepare_multiplots_data.py`` stores
    observations with ``year`` and ``month`` columns (no explicit ``idx``).
    This function maps each observation row to the climate time index using
    ``(year, month)`` and returns per-variable arrays shaped
    ``(n_obs, n_species)``.

    Parameters
    ----------
    observed_df : pl.DataFrame
        Flattened observed section for one plot.
    climate_df : pl.DataFrame
        Flattened climate section for one plot.
    species_names : list[str]
        Species names in model order.

    Returns
    -------
    dict[str, tuple[jnp.ndarray, jnp.ndarray]]
        Variable names → ``(obs_times, obs_values)``.
    """
    if observed_df.is_empty():
        return {}

    # prepare_multiplots_data.py currently writes single-species parquet files.
    if len(species_names) != 1:
        raise ValueError(
            "Parquet observed-section loader currently supports single-species plots only"
        )

    if not {"year", "month"}.issubset(observed_df.columns):
        raise ValueError("Observed section must contain year and month columns")

    if not {"year", "month"}.issubset(climate_df.columns):
        raise ValueError("Climate section must contain year and month columns")

    climate_index_df = climate_df.select(
        pl.int_range(0, climate_df.height).alias("idx"),
        pl.col("year").cast(pl.Int32),
        pl.col("month").cast(pl.Int32),
    )

    obs_with_idx = (
        observed_df.with_columns(
            pl.col("year").cast(pl.Int32),
            pl.col("month").cast(pl.Int32),
        )
        .join(climate_index_df, on=["year", "month"], how="inner")
        .sort("idx")
    )

    if "specie" in obs_with_idx.columns:
        obs_with_idx = obs_with_idx.filter(pl.col("specie").is_in(species_names))

    excluded_cols = {"idx", "month", "year", "Date", "date", "specie"}
    value_cols = [c for c in obs_with_idx.columns if c not in excluded_cols]

    observations: dict[str, tuple[jnp.ndarray, jnp.ndarray]] = {}
    for var_name in value_cols:
        var_df = obs_with_idx.filter(pl.col(var_name).is_not_null())
        if var_df.is_empty():
            continue

        idx = jnp.asarray(var_df["idx"].to_numpy(), dtype=jnp.int32)
        values = var_df[var_name].to_numpy().astype(np.float32)[:, None]
        observations[var_name] = (idx, jnp.asarray(values, dtype=jnp.float32))

    return observations


def load_params_from_file(params_file: str, species_names: list[str]) -> Params:
    """Load the shared species parameter vector from a dedicated params file.

    Parameters
    ----------
    params_file : str
        Parquet file containing a ``parameter`` column with one column per species.
    species_names : list[str]
        Species column names to extract (must match columns in the sheet).

    Returns
    -------
    Params
        Full physics parameter set for use as ``fixed_params`` in the model.
    """
    if not params_file.lower().endswith(".parquet"):
        raise ValueError(f"Expected parquet file for parameters, got: {params_file}")

    params_df = pl.read_parquet(params_file)
    param_names = params_df["param_name"].to_list()
    values_matrix = params_df["default"].to_numpy()
    params_dict = {name: jnp.asarray(values_matrix[i]) for i, name in enumerate(param_names)}
    return Params(**params_dict)


def load_plot_data(plot_file: str, plot_id: str, params_file: str) -> tuple[PlotData, Params]:
    """Load one plot's data from plot_file; load shared parameters from params_file.

    This separates plot-specific parquet sections from species physiology
    parameters shared across plots.

    Parameters
    ----------
    plot_file : str
        Parquet file produced by ``prepare_multiplots_data.py`` with nested
        ``site``, ``climate``, ``species``, ``observed`` columns.
    plot_id : str
                Plot identifier.
    params_file : str
        Parquet file with ``parameters`` and ``param_bound`` style columns.
        Shared by all plots of the same species.

    Returns
    -------
    PlotData
        Plot data ready for multi-plot inference.
    Params
        Fixed (non-estimated) parameters loaded from params_file.
    """
    if not plot_file.lower().endswith(".parquet"):
        raise ValueError(f"Expected parquet file for plot data, got: {plot_file}")

    plot_df = pl.read_parquet(plot_file)
    site_df = _load_section_from_parquet(plot_df, plot_id, "site")
    climate_df = _load_section_from_parquet(plot_df, plot_id, "climate")
    species_df = _load_section_from_parquet(plot_df, plot_id, "species")
    observed_df = _load_section_from_parquet(plot_df, plot_id, "observed")

    site_data, site_start, site_end = prepare_site(site_df)
    climate = prepare_climate(climate_df, str(site_start), str(site_end))
    species_data = prepare_species(species_df)
    index_to_species = {value: key for key, value in SPECIES_INDICES.items()}
    species_names = [index_to_species[int(idx)] for idx in species_data.specie.tolist()]

    n_species = len(species_data.specie)

    # Shared parameters from species file
    fixed_params = load_params_from_file(params_file, species_names)

    # Build initial state — same logic as prepare_data in PG3_model_impl.py
    start_month = site_data.month_i
    start_dormant = is_dormant(start_month, fixed_params.leafgrow, fixed_params.leaffall)
    initial_WF = jnp.where(start_dormant, jnp.asarray(0.0), species_data.WF)
    initial_WF_debt = jnp.where(start_dormant, species_data.WF, jnp.asarray(0.0))

    asw_min = jnp.where(
        site_data.ASW_min > site_data.ASW_max, site_data.ASW_max, site_data.ASW_min
    )
    initial_ASW = jnp.clip(site_data.ASW, asw_min, site_data.ASW_max)

    climate_year = int(site_data.year_i[0])
    climate_month = int(site_data.month_i[0])
    age_months = (climate_year - species_data.year_p) * 12 + (climate_month - species_data.month_p)

    initial_state = State(
        WF=initial_WF,
        WR=species_data.WR,
        WS=species_data.WS,
        N=species_data.N,
        ASW=jnp.full(n_species, initial_ASW, dtype=jnp.float32),
        age=age_months,
        WF_debt=initial_WF_debt,
        prev_month=jnp.full(n_species, 12 if start_month == 1 else start_month - 1),
    )

    observations = load_observations_from_section(
        observed_df=observed_df,
        climate_df=climate_df,
        species_names=species_names,
    )

    plot = PlotData(
        plot_id=plot_id,
        initial_state=initial_state,
        climate=climate,
        site=site_data,
        species=species_data,
        n_species=n_species,
        observations=observations,
    )
    return plot, fixed_params
