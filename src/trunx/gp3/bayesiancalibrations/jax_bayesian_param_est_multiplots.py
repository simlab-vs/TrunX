"""
JAX-parallel HMC parameter estimation across multiple forest plots.

Implementation work flow:
1. Load and pack all plot data into one batch with padding for variable-length sequences.
2. Define a NumPyro model that samples shared parameters and runs all plots in parallel
using JAX vmap.
3. Run MCMC inference to estimate the shared parameters across all plots.

This file contains the main analysis function `run_multi_plot_analysis` which can be called
with a list of plot files and a shared parameters file. It also includes helper functions
for packing the data and defining the model.

TODO:
- Add plots for visualizations.

"""

import os
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

from trunx.config import threepg_data_folder
from trunx.gp3.bayesiancalibrations.load_files import (
    PlotData,
    load_plot_data,
    load_plot_ids_from_file,
    load_priors_from_file,
)
from trunx.gp3.model_inputs import ClimateData, Params, SiteData, SpeciesData, State
from trunx.gp3.run_3pg import run_3pg

cpu_count = os.cpu_count() or 1
numpyro.set_host_device_count(max(1, cpu_count))

_DUMMY_MONTH = np.datetime64("2000-01")


class PackedClimateData(NamedTuple):
    """Climate arrays padded to a common sequence length."""

    T_avg: jnp.ndarray  # shape:(n_plots, max_months)
    T_max: jnp.ndarray  # shape:(n_plots, max_months)
    VPD: jnp.ndarray  # shape:(n_plots, max_months)
    precip: jnp.ndarray  # shape:(n_plots, max_months)
    solar_rad: jnp.ndarray  # shape:(n_plots, max_months)
    frost_days: jnp.ndarray  # shape:(n_plots, max_months)
    n_days: jnp.ndarray  # shape:(n_plots, max_months)
    co2: jnp.ndarray  # shape:(n_plots, max_months)
    d13catm: jnp.ndarray  # shape:(n_plots, max_months)
    month: jnp.ndarray  # shape:(n_plots, max_months)
    lengths: jnp.ndarray  # shape:(n_plots,)


class PackedSiteData(NamedTuple):
    """Site scalars stacked across all plots."""

    latitude: jnp.ndarray  # shape:(n_plots,)
    altitude: jnp.ndarray  # shape:(n_plots,)
    soil_class: jnp.ndarray  # shape:(n_plots,)
    ASW: jnp.ndarray  # shape:(n_plots,)
    ASW_max: jnp.ndarray  # shape:(n_plots,)
    ASW_min: jnp.ndarray  # shape:(n_plots,)
    year_i: jnp.ndarray  # shape:(n_plots,)
    month_i: jnp.ndarray  # shape:(n_plots,)


class PackedSpeciesData(NamedTuple):
    """Species arrays stacked across all plots."""

    specie: jnp.ndarray  # shape:(n_plots, n_species)
    FR: jnp.ndarray  # shape:(n_plots, n_species)
    WF: jnp.ndarray  # shape:(n_plots, n_species)
    WR: jnp.ndarray  # shape:(n_plots, n_species)
    WS: jnp.ndarray  # shape:(n_plots, n_species)
    N: jnp.ndarray  # shape:(n_plots, n_species)
    year_p: jnp.ndarray  # shape:(n_plots,)
    month_p: jnp.ndarray  # shape:(n_plots,)


class PackedObservationData(NamedTuple):
    """Padded observation arrays for one variable across plots."""

    times: jnp.ndarray  # shape:(n_plots, max_observations)
    values: jnp.ndarray  # shape:(n_plots, max_observations, n_species)
    mask: jnp.ndarray  # shape:(n_plots, max_observations)


class PackedPlotBatch(NamedTuple):
    """Single padded batch of all plots for JAX-parallel analysis."""

    plot_ids: tuple[str, ...]
    initial_state: State
    climate: PackedClimateData
    site: PackedSiteData
    species: PackedSpeciesData
    observations: dict[str, PackedObservationData]
    species_names: tuple[str, ...]
    # species_planted: tuple[object, ...]
    n_plots: int
    n_species: int
    max_months: int


def _stack_state(states: list[State]) -> State:
    """Stack per-plot initial states into one batched state tree."""
    return State(
        WF=jnp.stack([state.WF for state in states], axis=0),
        WR=jnp.stack([state.WR for state in states], axis=0),
        WS=jnp.stack([state.WS for state in states], axis=0),
        N=jnp.stack([state.N for state in states], axis=0),
        ASW=jnp.stack([state.ASW for state in states], axis=0),
        age=jnp.stack([state.age for state in states], axis=0),
        WF_debt=jnp.stack([state.WF_debt for state in states], axis=0),
        prev_month=jnp.stack([state.prev_month for state in states], axis=0),
    )


def _pad_1d(src: jnp.ndarray, target_len: int, dtype: jnp.dtype) -> jnp.ndarray:
    """Pad one climate vector to target length using edge-value padding."""
    pad_width = target_len - src.shape[0]
    out = jnp.pad(src, (0, pad_width), mode="edge").astype(dtype)
    return out


def _pack_observations(
    plots: list[PlotData],
    n_species: int,
) -> dict[str, PackedObservationData]:
    """Pad observations per variable for vectorized likelihood evaluation."""
    observations: dict[str, PackedObservationData] = {}
    all_var_names = sorted({var_name for plot in plots for var_name in plot.observations})

    for var_name in all_var_names:
        max_obs_len = max(
            plot.observations[var_name][0].shape[0] if var_name in plot.observations else 0
            for plot in plots
        )
        if max_obs_len == 0:
            continue

        obs_times = np.zeros((len(plots), max_obs_len), dtype=jnp.int32)
        obs_values = np.zeros((len(plots), max_obs_len, n_species), dtype=jnp.float32)
        obs_mask = np.zeros((len(plots), max_obs_len), dtype=bool)

        for plot_idx, plot in enumerate(plots):
            if var_name not in plot.observations:
                continue

            plot_times, plot_values = plot.observations[var_name]
            plot_times_np = np.asarray(plot_times, dtype=jnp.int32)
            plot_values_np = np.asarray(plot_values, dtype=jnp.float32)
            if plot_values_np.ndim == 1:
                plot_values_np = plot_values_np[:, None]

            if plot_values_np.shape[1] != n_species:
                raise ValueError(
                    f"Observation {var_name} in plot {plot.plot_id} has "
                    f"{plot_values_np.shape[1]} species columns, expected {n_species}."
                )

            obs_len = int(plot_times_np.shape[0])
            obs_times[plot_idx, :obs_len] = plot_times_np
            obs_values[plot_idx, :obs_len, :] = plot_values_np
            obs_mask[plot_idx, :obs_len] = True

        observations[var_name] = PackedObservationData(
            times=jnp.asarray(obs_times),
            values=jnp.asarray(obs_values),
            mask=jnp.asarray(obs_mask),
        )

    return observations


def _pack_plots_with_padding(plots: list[PlotData]) -> PackedPlotBatch:
    """Pack all plots into one padded batch for plot-level vmap parallelism."""
    if len(plots) == 0:
        raise ValueError("plots cannot be empty")

    n_species = plots[0].n_species
    for plot in plots[1:]:
        if plot.n_species != n_species:
            raise ValueError("All plots must have the same n_species for batching")

    lengths = jnp.asarray([int(plot.climate.month.shape[0]) for plot in plots], dtype=jnp.int32)
    max_months = int(lengths.max())

    climate_T_avg = np.zeros((len(plots), max_months), dtype=jnp.float32)
    climate_T_max = np.zeros((len(plots), max_months), dtype=jnp.float32)
    climate_VPD = np.zeros((len(plots), max_months), dtype=jnp.float32)
    climate_precip = np.zeros((len(plots), max_months), dtype=jnp.float32)
    climate_solar = np.zeros((len(plots), max_months), dtype=jnp.float32)
    climate_frost = np.zeros((len(plots), max_months), dtype=jnp.float32)
    climate_days = np.zeros((len(plots), max_months), dtype=jnp.float32)
    climate_co2 = np.zeros((len(plots), max_months), dtype=jnp.float32)
    climate_d13c = np.zeros((len(plots), max_months), dtype=jnp.float32)
    climate_month = np.zeros((len(plots), max_months), dtype=jnp.int32)

    for idx, plot in enumerate(plots):
        climate_T_avg[idx] = _pad_1d(plot.climate.T_avg, max_months, jnp.float32)
        climate_T_max[idx] = _pad_1d(plot.climate.T_max, max_months, jnp.float32)
        climate_VPD[idx] = _pad_1d(plot.climate.VPD, max_months, jnp.float32)
        climate_precip[idx] = _pad_1d(plot.climate.precip, max_months, jnp.float32)
        climate_solar[idx] = _pad_1d(plot.climate.solar_rad, max_months, jnp.float32)
        climate_frost[idx] = _pad_1d(plot.climate.frost_days, max_months, jnp.float32)
        climate_days[idx] = _pad_1d(plot.climate.n_days, max_months, jnp.float32)
        climate_co2[idx] = _pad_1d(plot.climate.co2, max_months, jnp.float32)
        climate_d13c[idx] = _pad_1d(plot.climate.d13catm, max_months, jnp.float32)
        climate_month[idx] = _pad_1d(plot.climate.month, max_months, jnp.int32)

    climate = PackedClimateData(
        T_avg=jnp.asarray(climate_T_avg),
        T_max=jnp.asarray(climate_T_max),
        VPD=jnp.asarray(climate_VPD),
        precip=jnp.asarray(climate_precip),
        solar_rad=jnp.asarray(climate_solar),
        frost_days=jnp.asarray(climate_frost),
        n_days=jnp.asarray(climate_days),
        co2=jnp.asarray(climate_co2),
        d13catm=jnp.asarray(climate_d13c),
        month=jnp.asarray(climate_month),
        lengths=jnp.asarray(lengths),
    )

    site = PackedSiteData(
        latitude=jnp.asarray([plot.site.latitude for plot in plots], dtype=jnp.float32),
        altitude=jnp.asarray([plot.site.altitude for plot in plots], dtype=jnp.float32),
        soil_class=jnp.asarray([plot.site.soil_class for plot in plots], dtype=jnp.int32),
        ASW=jnp.asarray([plot.site.ASW for plot in plots], dtype=jnp.float32),
        ASW_max=jnp.asarray([plot.site.ASW_max for plot in plots], dtype=jnp.float32),
        ASW_min=jnp.asarray([plot.site.ASW_min for plot in plots], dtype=jnp.float32),
        year_i=jnp.asarray([plot.site.year_i for plot in plots], dtype=jnp.int32),
        month_i=jnp.asarray([plot.site.month_i for plot in plots], dtype=jnp.int32),
    )

    species = PackedSpeciesData(
        FR=jnp.stack([plot.species.FR for plot in plots], axis=0),
        WF=jnp.stack([plot.species.WF for plot in plots], axis=0),
        WR=jnp.stack([plot.species.WR for plot in plots], axis=0),
        WS=jnp.stack([plot.species.WS for plot in plots], axis=0),
        N=jnp.stack([plot.species.N for plot in plots], axis=0),
        year_p=jnp.stack([plot.species.year_p for plot in plots], axis=0),
        month_p=jnp.stack([plot.species.month_p for plot in plots], axis=0),
        specie=jnp.stack([plot.species.specie for plot in plots], axis=0),
    )

    observations = _pack_observations(plots, n_species)

    return PackedPlotBatch(
        plot_ids=tuple(plot.plot_id for plot in plots),
        initial_state=_stack_state([plot.initial_state for plot in plots]),
        climate=climate,
        site=site,
        species=species,
        observations=observations,
        species_names=tuple(plots[0].species.specie),
        # species_planted=tuple(plots[0].species.planted),
        n_plots=len(plots),
        n_species=n_species,
        max_months=max_months,
    )


def _select_single_plot(
    batch: PackedPlotBatch,
    idx: jax.Array,
) -> tuple[State, ClimateData, SiteData, SpeciesData]:
    """Extract one plot from the packed batch using JAX indexing."""
    state = jax.tree.map(lambda x: x[idx], batch.initial_state)

    climate = ClimateData(
        T_avg=batch.climate.T_avg[idx],
        T_max=batch.climate.T_max[idx],
        VPD=batch.climate.VPD[idx],
        precip=batch.climate.precip[idx],
        solar_rad=batch.climate.solar_rad[idx],
        frost_days=batch.climate.frost_days[idx],
        n_days=batch.climate.n_days[idx],
        co2=batch.climate.co2[idx],
        d13catm=batch.climate.d13catm[idx],
        month=batch.climate.month[idx],
        # start_month=_DUMMY_MONTH,
    )

    site = SiteData(
        latitude=batch.site.latitude[idx],
        altitude=batch.site.altitude[idx],
        soil_class=batch.site.soil_class[idx],
        ASW=batch.site.ASW[idx],
        ASW_max=batch.site.ASW_max[idx],
        ASW_min=batch.site.ASW_min[idx],
        year_i=batch.site.year_i[idx],
        month_i=batch.site.month_i[idx],
        # site_start=_DUMMY_MONTH,
        # site_end=_DUMMY_MONTH,
    )

    species = SpeciesData(
        specie=batch.species.specie[idx],
        FR=batch.species.FR[idx],
        WF=batch.species.WF[idx],
        WR=batch.species.WR[idx],
        WS=batch.species.WS[idx],
        N=batch.species.N[idx],
        # planted=list(batch.species_planted),
        year_p=batch.species.year_p[idx],
        month_p=batch.species.month_p[idx],
    )

    return state, climate, site, species


def _run_packed_plots_forward(batch: PackedPlotBatch, params: Params) -> dict[str, jnp.ndarray]:
    """Run all plots in parallel using one vmapped forward pass."""

    def run_one_plot(idx: jax.Array) -> dict[str, jnp.ndarray]:
        state, climate, site, species = _select_single_plot(batch, idx)
        _, outputs = run_3pg(state, climate, params, site, species, batch.n_species)
        return outputs

    return jax.vmap(run_one_plot)(jnp.arange(batch.n_plots, dtype=jnp.int32))


def multi_plot_model(
    packed_plots: PackedPlotBatch,
    fixed_params: Params,
    priors: dict[str, tuple[float, float]] | None = None,
) -> None:
    """Bayesian model with shared parameters and plot-level JAX parallelism."""
    if priors is None:
        raise ValueError("Priors must be provided for plot analysis")

    sampled_params = {
        name: jnp.asarray(numpyro.sample(name, dist.Uniform(lo, hi)))
        for name, (lo, hi) in priors.items()
    }
    params = fixed_params._replace(**sampled_params)

    # Pack plots and run model in parallel across plots using JAX vmap.
    outputs = _run_packed_plots_forward(packed_plots, params)

    for var_name, obs in packed_plots.observations.items():
        if var_name not in outputs:
            continue

        pred_values = jnp.take_along_axis(
            outputs[var_name],
            obs.times[..., None],
            axis=1,
        )

        sigma = jnp.asarray(
            numpyro.sample(
                f"sigma_{var_name}",
                dist.HalfNormal(jnp.ones((packed_plots.n_plots,), dtype=jnp.float32)),
            ),
            dtype=jnp.float32,
        )

        numpyro.sample(
            f"obs_{var_name}",
            dist.StudentT(df=3.0, loc=pred_values, scale=sigma[:, None, None]).mask(
                obs.mask[..., None]
            ),
            obs=obs.values,
        )


def run_multi_plot_analysis(
    params_file: str,
    plot_files: list[tuple[str, str]],
    param_names: list[str] | None = None,
    num_warmup: int = 1000,
    num_samples: int = 1000,
    num_chains: int = 4,
    seed: int = 42,
) -> tuple[MCMC, dict]:
    """Run shared-parameter HMC inference across many plots."""
    priors = load_priors_from_file(params_file, param_names)
    if len(plot_files) == 0:
        raise ValueError("plot_files must contain at least one (file_path, plot_id) entry")

    plots: list[PlotData] = []
    fixed_params = None
    for file_path, plot_id in plot_files:
        plot, fp = load_plot_data(file_path, plot_id, params_file)
        plots.append(plot)
        if fixed_params is None:
            fixed_params = fp

    species_ref = tuple(plots[0].species.specie)
    for plot in plots[1:]:
        if tuple(plot.species.specie) != species_ref:
            got_species = tuple(plot.species.specie)
            raise ValueError(
                "All plots in one multi-plot run must share the same species set and order. "
                f"Expected {species_ref}, got {got_species} for plot {plot.plot_id}."
            )

    empty_obs_plots = [plot.plot_id for plot in plots if len(plot.observations) == 0]
    if empty_obs_plots:
        raise ValueError(
            "All plots must contain observations for fitting. "
            f"Plots without observations: {empty_obs_plots}"
        )

    packed_plots = _pack_plots_with_padding(plots)

    print(f"Loaded {len(plots)} plots")
    print(
        "Climate sequence lengths summary: "
        f"min={int(packed_plots.climate.lengths.min())}, "
        f"max={int(packed_plots.climate.lengths.max())}, "
        f"padded_to={packed_plots.max_months}"
    )

    rng_key = random.PRNGKey(seed)
    _, subkey = random.split(rng_key)

    kernel = NUTS(
        multi_plot_model,
        target_accept_prob=0.9,
        max_tree_depth=10,
    )
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method="parallel",
        progress_bar=True,
    )
    mcmc.run(subkey, packed_plots, fixed_params, priors)

    samples = mcmc.get_samples()

    mcmc.print_summary()

    return mcmc, samples


def run_multi_plot_analysis_for_file(
    plot_file: str,
    params_file: str,
    param_names: list[str] | None = None,
    num_warmup: int = 200,
    num_samples: int = 200,
    num_chains: int = 2,
    seed: int = 42,
    max_plots: int | None = None,
) -> tuple[MCMC, dict]:
    """Run shared-parameter inference across all plots in one parquet file."""
    plot_ids = load_plot_ids_from_file(plot_file)
    if max_plots is not None:
        plot_ids = plot_ids[:max_plots]

    plot_files = [(plot_file, plot_id) for plot_id in plot_ids]
    print(f"Running shared-parameter optimization across {len(plot_ids)} plots")

    return run_multi_plot_analysis(
        params_file=params_file,
        plot_files=plot_files,
        param_names=param_names,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        seed=seed,
    )


if __name__ == "__main__":
    species = "Picea_abies"
    run_multi_plot_analysis_for_file(
        plot_file=os.path.join(threepg_data_folder, f"icp_plot_data_{species}.parquet"),
        params_file=os.path.join(threepg_data_folder, "params_bounds.parquet"),
        param_names=None,
        max_plots=4,  # Limit to 4 plots for quick testing; remove or increase for full analysis.
        num_warmup=20,
        num_samples=20,
        num_chains=4,
    )
