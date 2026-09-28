"""HMC parameter estimation for 3PG model using DBH observations."""

import argparse
import gc
import os
import time
from typing import Any

import arviz as az
import jax
import jax.numpy as jnp
import jax.random as random
import numpyro
import numpyro.distributions as dist
import polars as pl
from numpyro.distributions.transforms import AffineTransform, ComposeTransform, SigmoidTransform
from numpyro.infer import HMC, MCMC, NUTS, init_to_uniform, init_to_value

from trunx.config import data_folder, results_data_folder, threepg_data_folder
from trunx.gp3.bayesiancalibrations.bayesian_config import DIAGNOSTIC_ONLY_ERROR_NAMES, FIT_PARAMS
from trunx.gp3.bayesiancalibrations.calibration_utils import (
    clip_defaults_to_priors,
    plot_inference_results,
    predict_from_parameter_draws,
)
from trunx.gp3.bayesiancalibrations.load_files import (
    literature_bound_overrides,
    load_observations_from_file,
    load_param_defaults_from_file,
    load_priors_from_file,
    load_top_sensitive_params,
)
from trunx.gp3.bayesiancalibrations.save_load_results import save_predictions
from trunx.gp3.model_inputs import State
from trunx.gp3.PG3_model_impl import prepare_data
from trunx.gp3.run_3pg import run_3pg

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"

os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
# Each host device (one per parallel chain) must do its own compute
# single-threaded, otherwise every device spawns its own intra-op thread
# pool and chains oversubscribe the SLURM-allocated CPUs, burning time on
# context switches instead of running the chains truly in parallel.
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
jax.config.update("jax_enable_x64", True)
# jax.config.update('jax_log_compiles', True)

az.rcParams["plot.backend"] = "matplotlib"

_sched_getaffinity = getattr(os, "sched_getaffinity", None)
available_cpus = len(_sched_getaffinity(0)) if _sched_getaffinity else (os.cpu_count() or 1)
numpyro.set_host_device_count(available_cpus)


def model(
    climate,
    site,
    species,
    n_species: int,
    fixed_params,
    priors: dict[str, tuple[float, float]],
    initial_state: State,
    param_defaults: dict[str, float] | None = None,
    observations: dict[str, tuple[jnp.ndarray, jnp.ndarray]] | None = None,
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
    initial_state : State
        Initial state for simulation
    observations : dict[str, tuple[jnp.ndarray, jnp.ndarray]] | None
        Dictionary mapping variable names to (obs_times, obs_values) tuples.
        Variables: DBH, Height, BA, N, WS, WF, WR

    """
    assert fixed_params is not None

    # Sample from priors

    samples = {}
    for param_name, (lower, upper) in priors.items():
        samples[param_name] = numpyro.sample(param_name, dist.Uniform(lower, upper))
        # # Transform to unconstrained space
        # transform = ComposeTransform(
        #     [SigmoidTransform(), AffineTransform(loc=lower, scale=upper - lower)]
        # )

        # # Use Gaussian prior in unconstrained space
        # if param_defaults and param_name in param_defaults:
        #     p = (param_defaults[param_name] - lower) / (upper - lower)
        #     p = jnp.clip(p, 1e-6, 1 - 1e-6)
        #     mu = jnp.log(p / (1 - p))
        #     sigma = (upper - lower) / 6
        # else:
        #     mu = 0.0
        #     sigma = (upper - lower) / 6

        # samples[param_name] = numpyro.sample(
        #     param_name, dist.TransformedDistribution(dist.Normal(mu, sigma), transform)
        # )

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

            # numpyro.sample(
            #     f"obs_{var_name}",
            #     # dist.StudentT(df=4, loc=pred_values, scale=samples[sigma_name]),
            #     dist.Normal(loc=pred_values, scale=samples[sigma_name]),
            #     obs=obs_flat,
            # )
            log_prob = dist.Normal(
                loc=pred_values,
                scale=samples[sigma_name],
            ).log_prob(obs_flat)
            numpyro.factor(f"obs_{var_name}", jnp.sum(log_prob))


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
    target_accept_prob: float = 0.95,
    max_tree_depth: int = 10,
    param_defaults: dict[str, float] | None = None,
    chain_method: str = "parallel",
    jit_model_args: bool = True,
    progress_bar: bool = False,
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
    progress_bar : bool
        Whether NumPyro prints a per-sample progress bar. Disabling this
        removes real per-step overhead, especially with `chain_method="vectorized"`.

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
        initial_state,
        param_defaults,
        observations,
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
        # max_tree_depth=max_tree_depth,
        init_strategy=init_strategy,
    )

    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        thinning=thinning,
        chain_method=chain_method,
        jit_model_args=jit_model_args,
        progress_bar=progress_bar,
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

    param_sets = {name: samples[name][indices] for name in param_names if name in samples}

    result = predict_from_parameter_draws(
        parameter_draws=param_sets,
        param_names=param_names,
        initial_state=initial_state,
        climate=climate,
        site=site,
        species=species,
        fixed_params=fixed_params,
        observations=observations,
        n_species=n_species,
    )

    del samples  # Free memory
    gc.collect()
    return result


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
    plot_inference_results(
        inf_data=inf_data,
        params=params,
        observations=observations,
        predictions=predictions,
        climate=climate,
        output_dir=output_dir,
    )


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
    num_warmup: int = 100,
    num_samples: int = 100,
    num_chains: int = 4,
    seed: int = 42,
    show_plots: bool = True,
    predict_with_uncert: bool = False,
    param_defaults: dict[str, float] | None = None,
    chain_method: str = "parallel",
    progress_bar: bool = False,
    jit_model_args: bool = True,
    max_tree_depth: int = 10,
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
        chain_method=chain_method,
        jit_model_args=jit_model_args,
        max_tree_depth=max_tree_depth,
        progress_bar=progress_bar,
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
    os.makedirs(output_dir, exist_ok=True)
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
    chain_method: str = "parallel",
    progress_bar: bool = False,
    jit_model_args: bool = True,
    max_tree_depth: int = 10,
    num_chains: int = 4,
    num_warmup: int = 100,
    num_samples: int = 100,
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
    input_data = prepare_data(file_path)

    priors = load_priors_from_file(
        file_path, param_names, bound_overrides=literature_bound_overrides(file_path)
    )
    for error_name in DIAGNOSTIC_ONLY_ERROR_NAMES:
        priors.pop(error_name, None)
    param_defaults = load_param_defaults_from_file(file_path, list(priors.keys()))
    param_defaults = clip_defaults_to_priors(param_defaults, priors)
    print(f"Loaded priors for parameters: {list(priors.keys())}")

    # Load all observations from file
    observations = load_observations_from_file(file_path, site_data=input_data.site)
    print(f"Loaded observations for variables: {list(observations.keys())}")

    skipped = [name for name in observations if f"err_{name}" not in priors]
    if skipped:
        print(f"Skipping observations with no matching sigma prior in error_param: {skipped}")

    # Run analysis
    mcmc, samples = run_full_analysis(
        initial_state=input_data.initial_state,
        climate=input_data.climate,
        site=input_data.site,
        species=input_data.species,
        n_species=input_data.n_species,
        observations=observations,
        fixed_params=input_data.params,
        priors=priors,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        output_dir=os.path.join(results_data_folder, "numpyro_bayesian_results"),
        show_plots=show_plots,
        predict_with_uncert=predict_with_uncert,
        param_defaults=param_defaults,
        chain_method=chain_method,
        progress_bar=progress_bar,
        jit_model_args=jit_model_args,
        max_tree_depth=max_tree_depth,
    )

    # Print parameter summaries
    print("\nParameter Summary:")
    for param in priors:
        if param in samples:
            mean_val = jnp.mean(samples[param])
            std_val = jnp.std(samples[param])
            print(f"  {param}: {mean_val:.4f} ± {std_val:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--file-path",
        default=os.path.join(threepg_data_folder, "solling_data.xlsx"),
        help="Excel file with input data and parameter bounds (default: %(default)s)",
    )
    parser.add_argument(
        "--param-names",
        nargs="+",
        default=None,
        metavar="NAME",
        help="Parameter names to estimate. Defaults to FIT_PARAMS plus every "
        "err_* sigma found in --file-path.",
    )
    parser.add_argument("--num-chains", type=int, default=4)
    parser.add_argument("--num-warmup", type=int, default=100)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--max-tree-depth", type=int, default=10)
    parser.add_argument(
        "--chain-method",
        default="parallel",
        choices=["parallel", "sequential", "vectorized"],
        help="NumPyro MCMC chain_method (default: %(default)s)",
    )
    parser.add_argument(
        "--predict-with-uncert",
        action="store_true",
        default=True,
        help="Generate predictions with uncertainty quantification (default: enabled)",
    )
    parser.add_argument(
        "--no-predict-with-uncert",
        dest="predict_with_uncert",
        action="store_false",
        help="Disable prediction-with-uncertainty generation",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        default=True,
        help="Generate diagnostic/prediction plots (default: enabled)",
    )
    parser.add_argument(
        "--no-show-plots", dest="show_plots", action="store_false", help="Disable plotting"
    )
    parser.add_argument(
        "--progress-bar",
        action="store_true",
        default=True,
        help="Show NumPyro's per-sample progress bar (default: enabled)",
    )
    parser.add_argument(
        "--no-progress-bar", dest="progress_bar", action="store_false", help="Hide progress bar"
    )
    args = parser.parse_args()

    start_time = time.perf_counter()

    param_names = args.param_names
    if param_names is None:
        error_names = [
            name for name in load_priors_from_file(args.file_path) if name.startswith("err_")
        ]
        param_names = FIT_PARAMS + error_names

    run_hmc_analysis(
        file_path=args.file_path,
        param_names=param_names,
        predict_with_uncert=args.predict_with_uncert,
        show_plots=args.show_plots,
        chain_method=args.chain_method,
        progress_bar=args.progress_bar,
        num_chains=args.num_chains,
        max_tree_depth=args.max_tree_depth,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
    )

    elapsed_time = time.perf_counter() - start_time
    print(f"Total runtime: {elapsed_time:.2f} seconds")
