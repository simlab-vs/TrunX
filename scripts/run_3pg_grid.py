"""Run 3PG for every grid cell using posterior-calibrated parameters."""

import calendar
import datetime
import os
import time
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl

from trunx.config import SPECIES_INDICES, results_data_folder, threepg_data_folder
from trunx.gp3.bayesiancalibrations.jax_bayesian_param_est_multiplots import (
    PackedClimateData,
    PackedPlotBatch,
    PackedSiteData,
    PackedSpeciesData,
    run_packed_plots_forward,
)
from trunx.gp3.bayesiancalibrations.save_load_results import load_inference_data
from trunx.gp3.helper_function import is_dormant
from trunx.gp3.model_inputs import Params, State
from trunx.gp3.PG3_model_impl import prepare_data

OUTPUT_VARS = [
    "WS",
    "WF",
    "WR",
    "DBH",
    "Height",
    "BA",
]


def sample_posterior_params(
    inference_data_path: str, fixed_params: Params, num_predictions: int = 500
) -> Params:
    """Randomly draw `num_predictions` posterior parameter sets."""
    posterior = cast(Any, load_inference_data(inference_data_path)).posterior
    calibrated = {name for name in posterior.data_vars if name in Params._fields}

    n_total = int(posterior.sizes["chain"] * posterior.sizes["draw"])
    n_pick = min(num_predictions, n_total)

    chain_indices, draw_indices = (
        np.random.randint(0, len(posterior.chain), size=n_pick),
        np.random.randint(0, len(posterior.draw), size=n_pick),
    )

    fields = {}
    for name in Params._fields:
        if name in calibrated:
            values = posterior[name].values[chain_indices, draw_indices]
            fields[name] = jnp.asarray(values, dtype=jnp.float32).reshape(n_pick, 1)
        else:
            base = getattr(fixed_params, name)
            fields[name] = jnp.broadcast_to(base, (n_pick, *base.shape))

    return Params(**fields)


def _calendar_sequence(
    from_str: str, to_str: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int]:
    """Build the year/month/n_days sequence between two "YYYY-MM" dates, inclusive."""
    from_date = datetime.date.fromisoformat(from_str + "-01")
    to_date = datetime.date.fromisoformat(to_str + "-01")
    n_years = to_date.year - from_date.year + 1

    year_seq = np.repeat(np.arange(from_date.year, to_date.year + 1), 12)
    month_seq = np.tile(np.arange(1, 13), n_years)

    start = from_date.month - 1
    end = len(month_seq) - (12 - to_date.month)
    year_seq, month_seq = year_seq[start:end], month_seq[start:end]

    n_days_seq = np.array(
        [calendar.monthrange(int(y), int(m))[1] for y, m in zip(year_seq, month_seq, strict=True)]
    )
    return year_seq, month_seq, n_days_seq, n_years, start, end


def build_initial_state(
    site: PackedSiteData, species: PackedSpeciesData, params: Params, year_i: int, month_i: int
) -> State:
    """Build the initial 3PG state for a packed batch of grid cells."""
    start_dormant = is_dormant(jnp.asarray(month_i), params.leafgrow, params.leaffall)
    initial_WF = jnp.where(start_dormant, 0.0, species.WF)
    initial_WF_debt = jnp.where(start_dormant, species.WF, 0.0)

    asw_min = jnp.where(site.ASW_min > site.ASW_max, site.ASW_max, site.ASW_min)
    initial_ASW = jnp.clip(site.ASW, asw_min, site.ASW_max)

    age_months = (year_i - species.year_p) * 12 + (month_i - species.month_p)

    return State(
        WF=initial_WF,
        WR=species.WR,
        WS=species.WS,
        N=species.N,
        ASW=initial_ASW,
        age=age_months,
        WF_debt=initial_WF_debt,
        prev_month=jnp.full_like(species.year_p, 12 if month_i == 1 else month_i - 1),
    )


def load_and_pack_grid(
    grid_input_path: str, fixed_params: Params
) -> tuple[pl.DataFrame, PackedPlotBatch]:
    """Load every grid cell into one `PackedPlotBatch`, treating each cell as a "plot".

    Parameters
    ----------
    grid_input_path : str
        Nested parquet produced by `build_grid_input_parquet.py`.
    fixed_params : Params
        Baseline physiology parameters, used to determine the initial dormancy
        state (leaf phenology is not resampled per grid cell).

    Returns
    -------
    tuple
        `grid_id`/`x`/`y` coordinate table and the packed batch, both in the
        same row order.
    """
    grid = pl.read_parquet(grid_input_path)
    n_grid = grid.height
    coords = grid.select("grid_id", "x", "y")

    site_flat = grid.select("grid_id", "site").explode("site").unnest("site")
    if site_flat.height != n_grid:
        raise ValueError("Expected exactly one site row per grid cell")

    from_to = site_flat.select("from", "to").unique()
    if from_to.height != 1:
        raise ValueError("All grid cells must share the same simulation period (from/to)")

    from_str, to_str = from_to.row(0)
    _, month_seq, n_days_seq, n_years, start, end = _calendar_sequence(from_str, to_str)
    n_months = len(month_seq)
    year_i, month_i = (int(part) for part in from_str.split("-"))

    climate_flat = grid.select("grid_id", "climate").explode("climate").unnest("climate")
    climate_lengths = climate_flat.group_by("grid_id", maintain_order=True).len()["len"].unique()
    if climate_lengths.len() != 1 or climate_lengths[0] != 12:
        raise ValueError("Expected a 12-row monthly climatology per grid cell")

    def _clim_tiled(col: str) -> jnp.ndarray:
        values = climate_flat[col].to_numpy().reshape(n_grid, 12).astype(np.float32)
        return jnp.asarray(np.tile(values, n_years)[:, start:end])

    climate = PackedClimateData(
        T_avg=_clim_tiled("tmp_ave"),
        T_max=_clim_tiled("tmp_max"),
        VPD=_clim_tiled("vpd_day"),
        precip=_clim_tiled("prcp"),
        solar_rad=_clim_tiled("srad"),
        frost_days=_clim_tiled("frost_days"),
        n_days=_clim_tiled("n_days"),
        co2=_clim_tiled("co2"),
        d13catm=_clim_tiled("d13catm"),
        month=_clim_tiled("month"),
        lengths=jnp.full((n_grid,), n_months, dtype=jnp.int32),
    )

    site = PackedSiteData(
        latitude=jnp.asarray(site_flat["latitude"].to_numpy(), dtype=jnp.float32).reshape(-1, 1),
        altitude=jnp.asarray(site_flat["altitude"].to_numpy(), dtype=jnp.float32).reshape(-1, 1),
        soil_class=jnp.asarray(site_flat["soil_class"].to_numpy(), dtype=jnp.int32).reshape(-1, 1),
        ASW=jnp.asarray(site_flat["asw_i"].to_numpy(), dtype=jnp.float32).reshape(-1, 1),
        ASW_max=jnp.asarray(site_flat["asw_max"].to_numpy(), dtype=jnp.float32).reshape(-1, 1),
        ASW_min=jnp.asarray(site_flat["asw_min"].to_numpy(), dtype=jnp.float32).reshape(-1, 1),
        year_i=jnp.full((n_grid, 1), year_i, dtype=jnp.int32),
        month_i=jnp.full((n_grid, 1), month_i, dtype=jnp.int32),
    )

    species_flat = grid.select("grid_id", "species").explode("species").unnest("species")
    species_counts = species_flat.group_by("grid_id", maintain_order=True).len()["len"].unique()
    if species_counts.len() != 1:
        raise ValueError("All grid cells must have the same number of species")
    if int(species_counts[0]) != 1:
        raise NotImplementedError("Grid batching currently supports exactly one species per cell")

    species_names = [name for name in species_flat["species"].to_list()]
    unknown_species = sorted({name for name in species_names if name not in SPECIES_INDICES})
    if unknown_species:
        raise ValueError(f"Unknown species code(s) in grid input: {unknown_species}")
    specie_idx = np.asarray([SPECIES_INDICES[name] for name in species_names], dtype=np.int32)

    planted_parts = [str(planted).split("-") for planted in species_flat["planted"].to_list()]
    planted_year = np.asarray([int(year) for year, _ in planted_parts], dtype=np.int32)
    planted_month = np.asarray([int(month) for _, month in planted_parts], dtype=np.int32)

    species = PackedSpeciesData(
        specie=jnp.asarray(specie_idx).reshape(-1, 1),
        FR=jnp.asarray(species_flat["fertility"].to_numpy(), dtype=jnp.float32).reshape(-1, 1),
        WF=jnp.asarray(species_flat["biom_foliage"].to_numpy(), dtype=jnp.float32).reshape(-1, 1),
        WR=jnp.asarray(species_flat["biom_root"].to_numpy(), dtype=jnp.float32).reshape(-1, 1),
        WS=jnp.asarray(species_flat["biom_stem"].to_numpy(), dtype=jnp.float32).reshape(-1, 1),
        N=jnp.asarray(species_flat["stems_n"].to_numpy(), dtype=jnp.float32).reshape(-1, 1),
        year_p=jnp.asarray(planted_year).reshape(-1, 1),
        month_p=jnp.asarray(planted_month).reshape(-1, 1),
    )

    initial_state = build_initial_state(site, species, fixed_params, year_i, month_i)

    packed_grid = PackedPlotBatch(
        plot_ids=tuple(str(grid_id) for grid_id in coords["grid_id"].to_list()),
        initial_state=initial_state,
        climate=climate,
        site=site,
        species=species,
        observations={},
        species_names=tuple(dict.fromkeys(species_names)),
        n_plots=n_grid,
        n_species=1,
        max_months=n_months,
    )
    return coords, packed_grid


def run_3pg_grid(
    grid_input_path: str = os.path.join(threepg_data_folder, "grid_input.parquet"),
    inference_data_path: str = os.path.join(
        results_data_folder, "results/pymc_inference_results", "inference_data.nc"
    ),
    physiology_file_path: str = os.path.join(threepg_data_folder, "solling_data.xlsx"),
    num_predictions: int = 5,
    output_path: str = os.path.join(results_data_folder, "grid_3pg_outputs.parquet"),
) -> pl.DataFrame:
    """Run 3PG for every grid cell using posterior-mean calibrated parameters.

    Parameters
    ----------
    grid_input_path : str
        Nested parquet produced by `build_grid_input_parquet.py`.
    inference_data_path : str
        Saved posterior (`inference_data.nc`) produced by `pymc_param_est.py`.
    physiology_file_path : str
        Excel workbook with the baseline physiology parameters that were held
        fixed during calibration (its calibrated fields are overridden by the
        posterior mean).
    output_path : str
        Destination parquet file path.

    Returns
    -------
    pl.DataFrame
        One row per grid cell with `grid_id`, `x`, `y`, and a monthly
        `outputs` list-of-struct column (fields from `OUTPUT_VARS`).
    """
    _, _, fixed_params, _, _, _, _ = prepare_data(physiology_file_path)
    coords, packed_grid = load_and_pack_grid(grid_input_path, fixed_params)

    params_batch = sample_posterior_params(inference_data_path, fixed_params, num_predictions)

    print(
        f"Running 3PG for {packed_grid.n_plots} grid cells x {num_predictions} posterior draws..."
    )
    outputs = jax.vmap(lambda params: run_packed_plots_forward(packed_grid, params))(params_batch)

    print("Finishing up and writing parquet...")

    n_months = packed_grid.max_months
    grid_ids = coords["grid_id"].to_numpy()

    n_grids = len(grid_ids)
    all_flat = []
    for var in OUTPUT_VARS:
        stacked = np.asarray(outputs[var])[..., 0].reshape(num_predictions, -1)
        all_flat.append(stacked)

    print("Finished with stacking")
    grid_ids_repeated = np.repeat(grid_ids, n_months)
    all_data = {
        "grid_id": np.tile(grid_ids_repeated, num_predictions),
        "param_idx": np.repeat(np.arange(num_predictions), n_grids * n_months),
    }

    for var, arr in zip(OUTPUT_VARS, all_flat, strict=True):
        all_data[var] = arr.reshape(-1)

    print("Creating dataframe...")
    # Create DataFrame
    all_flat_df = pl.DataFrame(all_data)

    all_flat_df.write_parquet(output_path)

    print(f"Wrote {output_path}: {all_flat_df.height} grid cells")
    return all_flat_df


if __name__ == "__main__":
    start_time = time.perf_counter()

    run_3pg_grid(num_predictions=200)

    elapsed_time = time.perf_counter() - start_time
    print(f"Total runtime: {elapsed_time:.2f} seconds")
