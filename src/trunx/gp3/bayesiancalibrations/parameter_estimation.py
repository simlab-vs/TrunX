"""HMC parameter estimation for 3PG model using DBH observations."""

import os
import time
from typing import Any

import arviz as az
import jax
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
import polars as pl
from jax import jit, tree_util, vmap
from numpyro.handlers import substitute
from numpyro.infer import MCMC, NUTS, init_to_uniform, init_to_value

from trunx.config import data_folder, results_data_folder, threepg_data_folder
from trunx.gp3.bayesiancalibrations.load_files import (
    load_observations_from_file,
    load_param_defaults_from_file,
    load_priors_from_file,
    load_top_sensitive_params,
)
from trunx.gp3.bayesiancalibrations.save_load_results import save_predictions
from trunx.gp3.model_inputs import State
from trunx.gp3.PG3_model_impl import prepare_data
from trunx.gp3.run_3pg import run_3pg as run_3pg_orig

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"

os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
jax.config.update("jax_enable_x64", False)
# jax.config.update('jax_log_compiles', True)

az.rcParams["plot.backend"] = "matplotlib"

numpyro.set_host_device_count(4)  # Set number of chains to run in parallel

print("JIT-compiling run_3pg...")
run_3pg = jax.jit(
    run_3pg_orig,
)
print("JIT-compiled run_3pg")


def model(
    climate,
    site,
    species,
    n_species: int,
    fixed_params,
    priors: dict[str, tuple[float, float]],
    observations: dict[str, tuple[jnp.ndarray, jnp.ndarray]] | None = None,
    initial_state: State | None = None,
):
    """
    Bayesian model for 3PG parameter estimation using multiple observations.

    Parameters
    ----------
    climate
        Climate data for the simulation
    site
        Site parameters
    species
        Species parameters
    n_species : int
        Number of species
    fixed_params
        Fixed parameters that won't be estimated
    priors : dict[str, tuple[float, float]]
        Dictionary mapping parameter names to (min, max) tuples for priors.
        Includes both physiology parameters and sigma/error parameters
        (e.g. `err_DBH`), typically loaded together via `load_priors_from_file`.
    observations : dict[str, tuple[jnp.ndarray, jnp.ndarray]] | None
        Dictionary mapping variable names to (obs_times, obs_values) tuples.
        Variables: DBH, Height, BA, N, WS, WF, WR
    initial_state : State | None
        Initial state for simulation

    """
    assert fixed_params is not None

    # Sample from priors
    samples = {}
    for param_name, (lower, upper) in priors.items():
        samples[param_name] = numpyro.sample(param_name, dist.Uniform(lower, upper))

    param_updates = {
        name: value for name, value in samples.items() if name in fixed_params._fields
    }
    params = fixed_params._replace(**param_updates)

    # Run model simulation
    _, outputs = run_3pg(initial_state, climate, params, site, species)

    # Observation likelihoods for each variable.
    if observations is not None:
        for var_name, (obs_times, obs_values) in observations.items():
            sigma_name = f"err_{var_name}"

            if var_name not in outputs or sigma_name not in samples:
                continue

            # Predictions and observations must line up element-for-element;
            # a mismatch would broadcast into an (n_obs, n_obs) outer product
            # that silently scores every prediction against every observation.
            pred_values = outputs[var_name][obs_times].reshape(-1)
            obs_flat = jnp.asarray(obs_values).reshape(-1)
            assert pred_values.shape == obs_flat.shape, (
                f"Likelihood shape mismatch for {var_name}: "
                f"predictions {pred_values.shape} vs observations {obs_flat.shape}"
            )
            numpyro.sample(
                f"obs_{var_name}",
                # dist.StudentT(df=4, loc=pred_values, scale=samples[sigma_name]),
                dist.Normal(loc=pred_values, scale=samples[sigma_name]),
                obs=obs_values
            )


def run_hmc_inference(
    initial_state: State,
    climate,
    site,
    species,
    n_species: int,
    observations: dict[str, tuple[jnp.ndarray, jnp.ndarray]],
    fixed_params,
    priors: dict[str, tuple[float, float]] | None = None,
    num_warmup: int = 1000,
    num_samples: int = 1000,
    num_chains: int = 4,
    seed: int = 42,
    thinning: int = 1,
    adaptive_warmup: bool = True,
    adapt_step_size: bool = True,
    adapt_mass_matrix: bool = True,
    target_accept_prob: float = 0.9,
    max_tree_depth: int = 10,
    param_defaults: dict[str, float] | None = None,
) -> tuple[MCMC, dict]:
    """
    Run HMC inference using NumPyro's NUTS sampler.

    Parameters
    ----------
    observations : dict[str, tuple[jnp.ndarray, jnp.ndarray]]
        Dictionary mapping variable names to (obs_times, obs_values) tuples.
    priors : dict[str, tuple[float, float]] | None
        Dictionary mapping parameter names to (min, max) tuples for priors.
    adaptive_warmup : bool
        If True, use adaptation during warmup. If False, disables adaptation.
    adapt_step_size : bool
        If True, adapt step size during warmup when adaptive_warmup is enabled.
    adapt_mass_matrix : bool
        If True, adapt mass matrix during warmup when adaptive_warmup is enabled.
    param_defaults : dict[str, float] | None
        Starting value for each calibrated parameter, used to seed every
        chain at the same point instead of a random prior draw — matching
        the R reference's `createUniformPrior(min, max, best)`. If None,
        NumPyro falls back to its default init strategy (a random prior draw).

    Returns
    -------
    tuple[MCMC, dict]
        - mcmc: The MCMC object containing samples
        - samples: Dictionary with posterior samples
    """
    # Set up random key
    rng_key = random.PRNGKey(seed)
    rng_key, subkey = random.split(rng_key)

    # Model arguments
    model_args = (
        climate,
        site,
        species,
        n_species,
        fixed_params,
        priors,
        observations,
        initial_state,
    )

    # Create the MCMC object with NUTS
    use_step_size_adaptation = adaptive_warmup and adapt_step_size
    use_mass_matrix_adaptation = adaptive_warmup and adapt_mass_matrix

    init_strategy = (
        init_to_value(values=dict(param_defaults))
        if param_defaults is not None
        else init_to_uniform
    )

    kernel = NUTS(
        model,
        adapt_step_size=use_step_size_adaptation,
        adapt_mass_matrix=use_mass_matrix_adaptation,
        target_accept_prob=target_accept_prob,
        max_tree_depth=max_tree_depth,
        init_strategy=init_strategy,
    )

    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        thinning=thinning,
        chain_method="parallel",
        progress_bar=True,
    )

    # Run MCMC
    mcmc.run(subkey, *model_args)

    # Get samples
    samples = mcmc.get_samples()

    return mcmc, samples


def predict_with_uncertainty(
    mcmc: MCMC,
    climate,
    site,
    species,
    n_species: int,
    initial_state: State,
    fixed_params,
    observations: dict[str, tuple[jnp.ndarray, jnp.ndarray]],
    priors: dict[str, tuple[float, float]],
    n_predictions: int = 50,
    seed: int = 42,
    include_obs_error: bool = False,
) -> dict[str, tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    """
    Generate predictions with uncertainty using posterior samples.

    Parameters
    ----------
    observations : dict[str, tuple[jnp.ndarray, jnp.ndarray]]
        Dictionary mapping variable names to (obs_times, obs_values) tuples.
    priors : dict[str, tuple[float, float]]
        Dictionary of parameter priors (used to get parameter names).

    Returns
    -------
    dict[str, tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]
        Dictionary mapping variable names to (mean_pred, lower_pred, upper_pred).
    """
    rng_key = random.PRNGKey(seed)
    samples = mcmc.get_samples()

    param_names = list(priors.keys())
    # Sigma/error parameters (e.g. `err_DBH`) are not fields of `fixed_params`.
    physiology_names = [name for name in param_names if name in fixed_params._fields]

    # Randomly select n_predictions samples
    n_total_samples = len(samples[param_names[0]])
    indices = random.choice(
        rng_key, n_total_samples, shape=(min(n_predictions, n_total_samples),), replace=False
    )

    param_sets = {name: samples[name][indices] for name in physiology_names}

    # Convert to list of arrays for vmap
    param_values = [param_sets[name] for name in physiology_names]

    # Define the model function
    def run_model(*params):
        """Run 3PG model with parameters as separate arguments."""
        param_dict = dict(zip(physiology_names, params, strict=True))
        params_obj = fixed_params._replace(**param_dict)
        _, outputs = run_3pg(initial_state, climate, params_obj, site, species)
        return outputs

    # Vectorize over the first dimension (samples)
    batched_run = vmap(run_model, in_axes=(0,) * len(physiology_names))

    # Run all simulations
    all_outputs = batched_run(*param_values)

    # Get variable names from the first output
    first_outputs = tree_util.tree_map(lambda x: x[0], all_outputs)

    predictions: dict[str, tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]] = {}
    for var_name in observations:
        if var_name not in first_outputs:
            continue

        var_series = all_outputs[var_name]
        if n_species == 1:
            var_series = var_series[..., 0]
        mean_pred = jnp.mean(var_series, axis=0)
        lower_pred = jnp.percentile(var_series, 2.5, axis=0)
        upper_pred = jnp.percentile(var_series, 97.5, axis=0)
        predictions[var_name] = (mean_pred, lower_pred, upper_pred)

    return predictions


def plot_results(
    inf_data: az.InferenceData,
    params: list[str] | None,
    observations: dict[str, tuple[jnp.ndarray, jnp.ndarray]] | None = None,
    predictions: dict[str, tuple[Any, Any, Any]] | None = None,
    climate=None,
    output_dir: str | None = None,
):
    """
    Plot trace, posterior, and prediction-uncertainty figures.

    Parameters
    ----------
    inf_data : az.InferenceData
        Inference data produced by `az.from_numpyro`.
    params : list[str] | None
        Parameter names to plot. If None, plots all posterior variables.
    predictions : dict[str, tuple[Any, Any, Any]] | None
        Prediction uncertainty bands (mean, lower, upper), as returned by
        `predict_with_uncertainty` (jnp arrays) or `load_predictions` (np arrays).
        Plotted alongside `observations` when both are given.
    climate
        Climate data, needed to determine the prediction time axis when
        `predictions` is given.
    output_dir : str | None
        If given, save each figure as a PNG in this directory.
    """
    if params is None:
        params = [str(name) for name in inf_data["posterior"].data_vars]

    # Trace plots
    az.plot_trace(inf_data, var_names=params)
    if output_dir is not None:
        plt.gcf().savefig(os.path.join(output_dir, "trace_plots.png"))

    # Posterior plots
    az.plot_posterior(inf_data, var_names=params)
    if output_dir is not None:
        plt.gcf().savefig(os.path.join(output_dir, "posterior_plots.png"))

    # Summary diagnostics
    summary = az.summary(inf_data, var_names=params)
    print(summary)

    if predictions is not None and observations is not None:
        assert climate is not None, "climate is required to plot predictions"

        # Determine number of months
        n_months = len(climate.month) if hasattr(climate, "month") else len(climate.T_avg)
        time_months = np.arange(n_months)

        for var_name, (mean_pred, lower_pred, upper_pred) in predictions.items():
            fig, ax = plt.subplots(figsize=(12, 6))

            mean_pred_np = np.asarray(mean_pred)
            lower_pred_np = np.asarray(lower_pred)
            upper_pred_np = np.asarray(upper_pred)

            ax.fill_between(
                time_months,
                lower_pred_np,
                upper_pred_np,
                alpha=0.3,
                color="tab:blue",
                label="95% CI",
            )
            ax.plot(
                time_months,
                mean_pred_np,
                color="tab:blue",
                linewidth=2,
                label="Mean Prediction",
            )

            obs_times_var, obs_values_var = observations[var_name]
            ax.scatter(
                np.asarray(obs_times_var),
                np.asarray(obs_values_var),
                color="red",
                s=50,
                zorder=5,
                label="Observations",
                edgecolors="black",
                linewidths=1.5,
            )

            ax.set_xlabel("Time (months)", fontsize=12)
            ax.set_ylabel(var_name, fontsize=12)
            ax.set_title(f"3PG Model Predictions with Uncertainty ({var_name})", fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            if output_dir is not None:
                fig.savefig(os.path.join(output_dir, f"prediction_{var_name}.png"))

    plt.show()


def run_full_analysis(
    initial_state: State,
    climate,
    site,
    species,
    output_dir: str,
    n_species: int,
    observations: dict[str, tuple[jnp.ndarray, jnp.ndarray]],
    fixed_params,
    priors: dict[str, tuple[float, float]],
    num_warmup: int = 1000,
    num_samples: int = 1000,
    num_chains: int = 4,
    seed: int = 42,
    show_plots: bool = True,
    predict_with_uncert: bool = False,
    param_defaults: dict[str, float] | None = None,
) -> tuple[MCMC, dict]:
    """
    Run complete HMC analysis with diagnostics and plotting.

    Parameters
    ----------
    observations : dict[str, tuple[jnp.ndarray, jnp.ndarray]]
        Dictionary mapping variable names to (obs_times, obs_values) tuples.
    priors : dict[str, tuple[float, float]] | None
        Dictionary mapping parameter names to (min, max) tuples for priors.
    param_defaults : dict[str, float] | None
        Starting value for each calibrated parameter, seeding every chain at
        the same point instead of a random prior draw. See `run_hmc_inference`.

    Returns
    -------
    tuple[MCMC, dict]
        - mcmc: MCMC object with samples
        - samples: Dictionary of posterior samples
    """
    param_names = list(priors.keys())

    print("Running HMC inference for 3PG model (multi-variable)")
    print(f"Number of species: {n_species}")
    print(f"Number of observation variables: {len(observations)}")
    print(f"Warmup samples: {num_warmup}")
    print(f"Posterior samples: {num_samples}")
    print(f"Number of chains: {num_chains}")
    print(f"Parameters to estimate: {param_names}")

    # Run HMC inference
    mcmc, samples = run_hmc_inference(
        initial_state=initial_state,
        climate=climate,
        site=site,
        species=species,
        n_species=n_species,
        observations=observations,
        fixed_params=fixed_params,
        priors=priors,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        seed=seed,
        param_defaults=param_defaults,
    )

    # Print summary
    print("Convergence Diagnostics")
    print("R-hat values (should be <= 1.0):")
    mcmc.print_summary()

    predictions = None
    if predict_with_uncert:
        predictions = predict_with_uncertainty(
            mcmc,
            climate,
            site,
            species,
            n_species,
            initial_state,
            fixed_params,
            observations=observations,
            priors=priors,
            n_predictions=min(500, len(samples[param_names[0]])) if param_names else 50,
        )

    print("Saving results...")
    inf_data = az.from_numpyro(mcmc)
    file_path = os.path.join(output_dir, "numpyro_inference_data.nc")
    inf_data.to_netcdf(file_path)
    if predictions is not None:
        save_predictions(predictions, output_dir)

    if show_plots:
        print("Generating plots...")
        plot_results(
            inf_data=inf_data,
            params=param_names,
            observations=observations,
            predictions=predictions,
            climate=climate,
            output_dir=output_dir,
        )

    return mcmc, samples


def run_hmc_analysis(
    file_path: str = os.path.join(threepg_data_folder, "solling_data.xlsx"),
    param_names: list[str] | None = None,
    predict_with_uncert: bool = False,
    show_plots: bool = False,
):
    """
    Run HMC implementation.

    Parameters
    ----------
    file_path : str
        Path to Excel file with input data and parameter bounds
    param_names : list[str] | None
        List of parameter names to estimate. If None, uses default set.
        Parameters must exist in the param_bound sheet of the file.
    predict_with_uncert : bool
        Whether to generate predictions with uncertainty quantification
    """
    initial_state, climate, fixed_params, site_data, species_data, n_species, species_names = (
        prepare_data(file_path)
    )

    priors = load_priors_from_file(file_path, param_names=param_names)
    param_defaults = load_param_defaults_from_file(file_path, list(priors.keys()))
    print(f"Loaded priors for parameters: {list(priors.keys())}")

    # Load all observations from file
    observations = load_observations_from_file(file_path, site_data=site_data)
    print(f"Loaded observations for variables: {list(observations.keys())}")

    skipped = [name for name in observations if f"err_{name}" not in priors]
    if skipped:
        print(f"Skipping observations with no matching sigma prior in error_param: {skipped}")

    print(f"Fixed parameters: {fixed_params}")

    # Run analysis
    mcmc, samples = run_full_analysis(
        initial_state=initial_state,
        climate=climate,
        site=site_data,
        species=species_data,
        n_species=n_species,
        observations=observations,
        fixed_params=fixed_params,
        priors=priors,
        num_warmup=100,
        num_samples=100,
        num_chains=4,
        output_dir=os.path.join(data_folder, "hmc_results"),
        show_plots=show_plots,
        predict_with_uncert=predict_with_uncert,
        param_defaults=param_defaults,
    )

    # Print parameter summaries
    print("\nParameter Summary:")
    for param in priors:
        if param in samples:
            mean_val = jnp.mean(samples[param])
            std_val = jnp.std(samples[param])
            print(f"  {param}: {mean_val:.4f} ± {std_val:.4f}")


if __name__ == "__main__":
    start_time = time.perf_counter()

    # Use solling_data.xlsx by default

    file_path = os.path.join(threepg_data_folder, "full_solling_data.xlsx")

    # Restrict calibration to the most sensitive physiology parameters.
    morris_results_path = os.path.join(
        results_data_folder, "morris_analysis_results_jax", "morris_all_components.csv"
    )
    error_names = [name for name in load_priors_from_file(file_path) if name.startswith("err_")]
    top_params = load_top_sensitive_params(morris_results_path, n_top=5)
    r_20_params = [
        "pFS20",
        "aWS",
        "nWS",
        "pRn",
        "Tmin",
        "Topt",
        "Tmax",
        "fN0",
        "fNn",
        "MaxAge",
        "rAge",
        "gammaN1",
        "thinPower",
        "mS",
        "alphaCx",
        "rhoMin",
        "rhoMax",
        "aH",
        "nHB",
        "nHC",
    ]

    # param_names = top_params + error_names
    param_names = r_20_params + error_names

    run_hmc_analysis(
        file_path=file_path,
        param_names=param_names,
        predict_with_uncert=True,
        show_plots=True,
    )

    elapsed_time = time.perf_counter() - start_time
    print(f"Total runtime: {elapsed_time:.2f} seconds")
