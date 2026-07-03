"""HMC parameter estimation for 3PG model using DBH observations."""

import os

import arviz as az
import jax
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
import polars as pl
from numpyro.handlers import substitute
from numpyro.infer import MCMC, NUTS

from trunx.config import data_folder, results_data_folder, threepg_data_folder
from trunx.gp3.model_inputs import State
from trunx.gp3.PG3_model_impl import prepare_data
from trunx.gp3.run_3pg import run_3pg

az.rcParams["plot.backend"] = "matplotlib"

numpyro.set_host_device_count(4)  # Set number of chains to run in parallel


def load_priors_from_file(
    file_path: str,
    param_names: list[str] | None = None,
    default_lower_factor: float = 0.8,
    default_upper_factor: float = 1.2,
) -> dict[str, tuple[float, float]]:
    """
    Load parameter priors from the param_bound and error_param sheets in Excel file.

    Parameters
    ----------
    file_path : str
        Path to Excel file containing param_bound and error_param sheets
    param_names : list[str] | None
        List of parameter names to load. If None, loads all available parameters.
    default_lower_factor : float
        Factor to apply to default value if min/max not specified (lower bound)
    default_upper_factor : float
        Factor to apply to default value if min/max not specified (upper bound)

    Returns
    -------
    dict[str, tuple[float, float]]
        Dictionary mapping parameter names (including sigma parameters such as
        `err_DBH`) to (min, max) tuples
    """
    param_bounds_df = pl.concat(
        [
            pl.read_excel(file_path, sheet_name="param_bound"),
            pl.read_excel(file_path, sheet_name="error_param"),
        ],
        how="vertical_relaxed",
    )
    priors = {}

    if param_names is None:
        param_names = param_bounds_df.filter(
            pl.col("min").is_not_null() & pl.col("max").is_not_null()
        )["param_name"].to_list()

        if len(param_names) == 0:
            raise ValueError(
                "No parameters with specified min/max bounds found in param_bound sheet"
            )

    for param_name in param_names:
        row = param_bounds_df.filter(pl.col("param_name") == param_name)

        if len(row) == 0:
            raise ValueError(f"Parameter {param_name} not found in param_bound sheet")

        min_val = row["min"][0]
        max_val = row["max"][0]
        default_val = row["default"][0]

        # Use default-based bounds if min/max not specified
        if min_val is None or max_val is None:
            if default_val is None:
                raise ValueError(f"Parameter {param_name} has no bounds and no default value")
            min_val = default_val * default_lower_factor
            max_val = default_val * default_upper_factor

        priors[param_name] = (float(min_val), float(max_val))

    return priors


def load_top_sensitive_params(morris_results_path: str, n_top: int = 20) -> list[str]:
    """
    Load the N most sensitive physiology parameter names from Morris results.

    Following the r3PG reference vignette, calibration is restricted to the
    most sensitive parameters (identified via Morris screening on the total
    log-likelihood) rather than every bounded parameter.

    Parameters
    ----------
    morris_results_path : str
        Path to a Morris results CSV with `Component`, `Parameter`, `mu_star`
        columns (as produced by the Morris sensitivity analysis scripts).
    n_top : int
        Number of top physiology parameters to keep. Error parameters are
        excluded here since they are added separately as sigma priors.

    Returns
    -------
    list[str]
        Parameter names ranked by `mu_star` on the `total` component, descending.
    """
    df = pl.read_csv(morris_results_path)
    ranked = df.filter(
        (pl.col("Component") == "total") & ~pl.col("Parameter").str.starts_with("err_")
    ).sort("mu_star", descending=True)
    return ranked["Parameter"].head(n_top).to_list()


def load_observations_from_file(
    file_path: str,
) -> dict[str, tuple[jnp.ndarray, jnp.ndarray]]:
    """
    Load all observations from observed sheet in Excel file.

    Parameters
    ----------
    file_path : str
        Path to Excel file containing observed sheet

    Returns
    -------
    dict[str, tuple[jnp.ndarray, jnp.ndarray]]
        Dictionary mapping variable names to (obs_times, obs_values) tuples.
        Variables included: DBH, Height, BA, N, WS, WF, WR
    """
    obs_df = pl.read_excel(file_path, sheet_name="observed")

    obs_times = jnp.asarray(obs_df["idx"].to_numpy(), dtype=jnp.int32)

    observations = {}
    excluded_cols = {"idx", "month", "year", "date"}
    var_names = [col for col in obs_df.columns if col not in excluded_cols]

    for var_name in var_names:
        values_np = obs_df[var_name].to_numpy()
        valid_mask = ~np.isnan(values_np)
        if not np.any(valid_mask):
            continue

        var_obs_times = obs_times[valid_mask]
        obs_values = jnp.asarray(values_np[valid_mask], dtype=jnp.float32)
        observations[var_name] = (var_obs_times, obs_values)

    return observations


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
    _, outputs = run_3pg(initial_state, climate, params, site, species, n_species)

    # Observation likelihoods for each variable.
    if observations is not None:
        for var_name, (obs_times, obs_values) in observations.items():
            if var_name not in outputs:
                continue
            sigma_name = f"err_{var_name}"
            if sigma_name not in samples:
                continue
            pred_values = outputs[var_name][obs_times]
            numpyro.sample(
                f"obs_{var_name}",
                # dist.StudentT(df=3, loc=pred_values, scale=samples[sigma_name]),
                dist.Normal(loc=pred_values, scale=samples[sigma_name]),
                obs=obs_values,
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

    kernel = NUTS(
        model,
        adapt_step_size=use_step_size_adaptation,
        adapt_mass_matrix=use_mass_matrix_adaptation,
        target_accept_prob=target_accept_prob,
        max_tree_depth=max_tree_depth,
    )

    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        thinning=thinning,
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

    # Randomly select n_predictions samples
    n_total_samples = len(samples[param_names[0]])
    indices = random.choice(
        rng_key, n_total_samples, shape=(min(n_predictions, n_total_samples),), replace=False
    )

    # Run model for each selected sample
    all_outputs = []

    for idx in indices:
        # Extract physiology parameters for this sample (sigma/error parameters
        # such as `err_DBH` are not fields of `fixed_params`)
        param_dict = {
            name: samples[name][idx]
            for name in param_names
            if name in samples and name in fixed_params._fields
        }
        params = fixed_params._replace(**param_dict)

        # Run simulation
        _, outputs = run_3pg(initial_state, climate, params, site, species, n_species)
        all_outputs.append(outputs)

    predictions: dict[str, tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]] = {}
    for var_name in observations:
        if var_name not in all_outputs[0]:
            continue

        var_series = jnp.stack([output[var_name] for output in all_outputs], axis=0)
        if n_species == 1:
            var_series = var_series[..., 0]
        mean_pred = jnp.mean(var_series, axis=0)
        lower_pred = jnp.percentile(var_series, 2.5, axis=0)
        upper_pred = jnp.percentile(var_series, 97.5, axis=0)
        predictions[var_name] = (mean_pred, lower_pred, upper_pred)

    return predictions


def save_inference_data(
    inf_data: az.InferenceData, output_dir: str, filename: str = "inference_data.nc"
) -> str:
    """
    Save arviz InferenceData (posterior samples, diagnostics) to a NetCDF file.

    Parameters
    ----------
    inf_data : az.InferenceData
        Inference data produced by ``az.from_numpyro``.
    output_dir : str
        Directory to save the file in (created if missing).
    filename : str
        Name of the NetCDF file.

    Returns
    -------
    str
        Full path to the saved file.
    """
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, filename)
    inf_data.to_netcdf(file_path)
    return file_path


def load_inference_data(file_path: str) -> az.InferenceData:
    """
    Load arviz InferenceData previously saved with `save_inference_data`.

    Parameters
    ----------
    file_path : str
        Path to the NetCDF file.

    Returns
    -------
    az.InferenceData
        Loaded inference data.
    """
    return az.from_netcdf(file_path)


def save_predictions(
    predictions: dict[str, tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]],
    output_dir: str,
    filename: str = "predictions.npz",
) -> str:
    """
    Save prediction uncertainty bands to a compressed .npz file.

    Parameters
    ----------
    predictions : dict[str, tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]
        Dictionary mapping variable names to (mean_pred, lower_pred, upper_pred),
        as returned by `predict_with_uncertainty`.
    output_dir : str
        Directory to save the file in (created if missing).
    filename : str
        Name of the .npz file.

    Returns
    -------
    str
        Full path to the saved file.
    """
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, filename)
    arrays: dict[str, np.ndarray] = {}
    for var_name, (mean_pred, lower_pred, upper_pred) in predictions.items():
        arrays[f"{var_name}_mean"] = np.asarray(mean_pred)
        arrays[f"{var_name}_lower"] = np.asarray(lower_pred)
        arrays[f"{var_name}_upper"] = np.asarray(upper_pred)
    np.savez(file_path, **arrays)  # ty: ignore[invalid-argument-type]  # pyright: ignore[reportArgumentType]
    return file_path


def load_predictions(file_path: str) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Load prediction uncertainty bands previously saved with `save_predictions`.

    Parameters
    ----------
    file_path : str
        Path to the .npz file.

    Returns
    -------
    dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]
        Dictionary mapping variable names to (mean_pred, lower_pred, upper_pred).
    """
    data = np.load(file_path)
    var_names = sorted({key.rsplit("_", 1)[0] for key in data.files})
    return {
        var_name: (data[f"{var_name}_mean"], data[f"{var_name}_lower"], data[f"{var_name}_upper"])
        for var_name in var_names
    }


def save_results(
    mcmc: MCMC,
    output_dir: str,
    predictions: dict[str, tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]] | None = None,
) -> az.InferenceData:
    """
    Save inference data and, if available, prediction uncertainty bands to disk.

    Parameters
    ----------
    mcmc : MCMC
        Fitted MCMC object.
    output_dir : str
        Directory to save results in.
    predictions : dict[str, tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]] | None
        Prediction uncertainty bands from `predict_with_uncertainty`, if computed.

    Returns
    -------
    az.InferenceData
        The inference data, so callers can reuse it for plotting without recomputing.
    """
    inf_data = az.from_numpyro(mcmc)
    save_inference_data(inf_data, output_dir)
    if predictions is not None:
        save_predictions(predictions, output_dir)
    return inf_data


def plot_results(
    inf_data: az.InferenceData,
    params: list[str] | None,
    observations: dict[str, tuple[jnp.ndarray, jnp.ndarray]] | None = None,
    predictions: dict[str, tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]] | None = None,
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
    predictions : dict[str, tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]] | None
        Prediction uncertainty bands from `predict_with_uncertainty`. Plotted
        alongside `observations` when both are given.
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
) -> tuple[MCMC, dict]:
    """
    Run complete HMC analysis with diagnostics and plotting.

    Parameters
    ----------
    observations : dict[str, tuple[jnp.ndarray, jnp.ndarray]]
        Dictionary mapping variable names to (obs_times, obs_values) tuples.
    priors : dict[str, tuple[float, float]] | None
        Dictionary mapping parameter names to (min, max) tuples for priors.

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
    inf_data = save_results(mcmc, output_dir, predictions)

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
    print(f"Loaded priors for parameters: {list(priors.keys())}")

    # Load all observations from file
    observations = load_observations_from_file(file_path)
    print(f"Loaded observations for variables: {list(observations.keys())}")

    skipped = [name for name in observations if f"err_{name}" not in priors]
    if skipped:
        print(f"Skipping observations with no matching sigma prior in error_param: {skipped}")

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
        num_warmup=1000,
        num_samples=1000,
        num_chains=4,
        output_dir=os.path.join(data_folder, "hmc_results"),
        show_plots=True,
        predict_with_uncert=predict_with_uncert,
    )

    # Print parameter summaries
    print("\nParameter Summary:")
    for param in priors:
        if param in samples:
            mean_val = jnp.mean(samples[param])
            std_val = jnp.std(samples[param])
            print(f"  {param}: {mean_val:.4f} ± {std_val:.4f}")


if __name__ == "__main__":
    # Use solling_data.xlsx by default

    file_path = os.path.join(threepg_data_folder, "solling_data.xlsx")

    # Restrict calibration to the most sensitive physiology parameters.
    morris_results_path = os.path.join(
        results_data_folder, "morris_analysis_results_jax", "morris_all_components.csv"
    )
    error_names = [name for name in load_priors_from_file(file_path) if name.startswith("err_")]
    top_params = load_top_sensitive_params(morris_results_path, n_top=20)
    param_names = top_params + error_names

    run_hmc_analysis(
        file_path=file_path,
        param_names=param_names,
        predict_with_uncert=True,
    )
