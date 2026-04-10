"""HMC parameter estimation for 3PG model using DBH observations."""

import json
import os

import arviz as az
import jax
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.handlers import substitute
from numpyro.infer import MCMC, NUTS

from trunx.gp3.model_inputs import State
from trunx.gp3.PG3_model_impl import prepare_data
from trunx.gp3.run_3pg import run_3pg

import polars as pl

az.rcParams["plot.backend"] = "matplotlib"


def model(
    climate,
    site,
    species,
    n_species: int,
    fixed_params,
    obs_DBH: jnp.ndarray | None = None,
    obs_times: jnp.ndarray | None = None,
    initial_state: State | None = None,
):
    """
    Bayesian model for 3PG parameter estimation using DBH observations.

    Parameters
    ----------
    - climate: Climate data for the simulation
    - site: Site parameters
    - species: Species parameters
    - n_species: Number of species
    - obs_DBH: Observed DBH (diameter at breast height)
    - obs_times: Time indices for observations
    - initial_state: Initial state for simulation
    - fixed_params: Fixed parameters that won't be estimated
    """
    assert fixed_params is not None
    # Priors for parameters
    # alphaCx = numpyro.sample("alphaCx", dist.LogNormal(jnp.log(0.05), 0.5))
    # CoeffCond = numpyro.sample("CoeffCond", dist.LogNormal(jnp.log(0.05), 0.5))
    # Y = numpyro.sample("Y", dist.Normal(0.47, 0.05))

    # gammaF0 = numpyro.sample("gammaF0", dist.LogNormal(jnp.log(0.001), 0.3))
    # gammaF1 = numpyro.sample("gammaF1", dist.LogNormal(jnp.log(0.02), 0.3))
    # tgammaF = numpyro.sample("tgammaF", dist.LogNormal(jnp.log(60.0), 0.1))

    # tRho = numpyro.sample("tRho", dist.LogNormal(jnp.log(1.0), 0.02))

    alphaCx = numpyro.sample("alphaCx", dist.Uniform(0.020, 0.090))
    CoeffCond = numpyro.sample("CoeffCond", dist.Uniform(0.0001, 0.070))
    Y = numpyro.sample("Y", dist.Uniform(0.440, 0.510))
    gammaF0 = numpyro.sample("gammaF0", dist.Uniform(0.0001, 0.003))
    gammaF1 = numpyro.sample("gammaF1", dist.Uniform(0.0001, 0.040))
    tgammaF = numpyro.sample("tgammaF", dist.Uniform(12.0, 150.0))
    tRho = numpyro.sample("tRho", dist.Uniform(0.0, 150.0)) 
        
    # Update parameters
    params = fixed_params._replace(
        alphaCx=alphaCx,
        CoeffCond=CoeffCond,
        Y=Y,
        gammaF0=gammaF0,
        # gammaF1=gammaF1,
        tgammaF=tgammaF,
        tRho=tRho,
    )

    # Run model simulation
    _, outputs = run_3pg(initial_state, climate, params, site, species, n_species)
    pred_DBH = outputs["DBH"][obs_times] if obs_times is not None else outputs["DBH"]

    # Observation likelihood
    sigma_DBH = numpyro.sample("sigma_DBH", dist.HalfNormal(1.0))
    numpyro.sample("obs_DBH", dist.StudentT(df=3, loc=pred_DBH, scale=sigma_DBH), obs=obs_DBH)


def run_hmc_inference(
    initial_state: State,
    climate,
    site,
    species,
    n_species: int,
    obs_DBH: jnp.ndarray,
    obs_times: jnp.ndarray,
    fixed_params,
    num_warmup: int = 1000,
    num_samples: int = 1000,
    num_chains: int = 4,
    seed: int = 42,
    thinning: int = 1,
    adapt_step_size: bool = True,
    target_accept_prob: float = 0.9,
    max_tree_depth: int = 10,
) -> tuple[MCMC, dict]:
    """
    Run HMC inference using NumPyro's NUTS sampler.

    Returns
    -------
    - mcmc: The MCMC object containing samples
    - diagnostics: Dictionary with convergence diagnostics
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
        obs_DBH,
        obs_times,
        initial_state,
    )

    # Create the MCMC object with NUTS
    kernel = NUTS(
        model,
        adapt_step_size=adapt_step_size,
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
    n_predictions: int = 50,
    seed: int = 42,
    include_obs_error: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray] | None]:
    """
    Generate predictions with uncertainty using posterior samples.

    Returns
    -------
    - mean_pred: Mean predictions across all time steps
    - lower_pred: 2.5th percentile predictions
    - upper_pred: 97.5th percentile predictions
    - pred_intervals: Prediction intervals including observation error (if include_obs_error)
    """
    rng_key = random.PRNGKey(seed)
    samples = mcmc.get_samples()

    # Randomly select n_predictions samples
    n_total_samples = len(samples["Y"])
    indices = random.choice(
        rng_key, n_total_samples, shape=(min(n_predictions, n_total_samples),), replace=False
    )

    # Run model for each selected sample
    all_DBH = []
    all_sigma = []

    for idx in indices:
        # Extract parameters for this sample
        params = fixed_params._replace(
            alphaCx=samples["alphaCx"][idx],
            CoeffCond=samples["CoeffCond"][idx],
            Y=samples["Y"][idx],
            gammaF0=samples["gammaF0"][idx],
            gammaF1=samples["gammaF1"][idx],
            tgammaF=samples["tgammaF"][idx],
            tRho=samples.get("tRho", fixed_params.tRho)[idx]
            if "tRho" in samples
            else fixed_params.tRho,
        )

        # Run simulation
        _, outputs = run_3pg(initial_state, climate, params, site, species, n_species)
        all_DBH.append(outputs["DBH"])

        if include_obs_error and "sigma_DBH" in samples:
            all_sigma.append(samples["sigma_DBH"][idx])

    # Stack predictions
    all_DBH = jnp.stack(all_DBH, axis=0)

    # Compute statistics
    mean_pred = jnp.mean(all_DBH, axis=0)
    lower_pred = jnp.percentile(all_DBH, 2.5, axis=0)
    upper_pred = jnp.percentile(all_DBH, 97.5, axis=0)

    if include_obs_error and all_sigma:
        all_sigma = jnp.array(all_sigma)
        mean_sigma = jnp.mean(all_sigma)

        lower_pred_with_error = lower_pred - 1.96 * mean_sigma
        upper_pred_with_error = upper_pred + 1.96 * mean_sigma

        return mean_pred, lower_pred, upper_pred, (lower_pred_with_error, upper_pred_with_error)

    return mean_pred, lower_pred, upper_pred, None


def plot_results(
    mcmc: MCMC,
    climate,
    site,
    species,
    n_species: int,
    initial_state: State,
    fixed_params,
    params,
    obs_times: jnp.ndarray,
    obs_DBH: jnp.ndarray,
    save_path: str | None = None,
    show_plots: bool = True,
    predict_with_uncert: bool = False,
):
    """Plot MCMC results including trace plots, posterior distributions, and predictions."""
    inf_data = az.from_numpyro(mcmc)

    if params is None:
        params = list(inf_data.posterior.data_vars)

    # Trace plots
    az.plot_trace(inf_data, var_names=params)

    # Posterior plots
    az.plot_posterior(inf_data, var_names=params)

    # Summary diagnostics
    summary = az.summary(inf_data, var_names=params)
    print(summary)

    if predict_with_uncert:
        # Get predictions
        mean_pred, lower_pred, upper_pred, pred_intervals = predict_with_uncertainty(
            mcmc,
            climate,
            site,
            species,
            n_species,
            initial_state,
            fixed_params,
            n_predictions=min(500, len(mcmc.get_samples()["Y"])),
        )

        if pred_intervals is not None:
            lower_err, upper_err = map(np.asarray, pred_intervals)

        # Determine number of months
        n_months = len(climate.month) if hasattr(climate, "month") else len(climate.T_avg)
        time_months = np.arange(n_months)

        def ensure_species_first(arr):
            arr = np.asarray(arr)
            if arr.ndim == 1:
                return arr[np.newaxis, :]
            elif arr.shape[0] == n_months:
                return arr.T
            return arr

        mean_pred = ensure_species_first(mean_pred)
        lower_pred = ensure_species_first(lower_pred)
        upper_pred = ensure_species_first(upper_pred)
        if pred_intervals is not None:
            lower_err = ensure_species_first(lower_err)
            upper_err = ensure_species_first(upper_err)

        n_species_plot = mean_pred.shape[0]

        # Plot predictions with intervals
        fig, ax = plt.subplots(figsize=(12, 6))
        cmap = plt.get_cmap("tab10")
        colors = [cmap(i) for i in range(10)]
        for s in range(n_species_plot):
            color = colors[s]
            ax.fill_between(
                time_months,
                lower_pred[s],
                upper_pred[s],
                alpha=0.3,
                color=color,
                label=f"95% CI Species {s + 1}" if n_species_plot > 1 else "95% CI",
            )
            if pred_intervals is not None:
                ax.fill_between(
                    time_months,
                    lower_err[s],
                    upper_err[s],
                    alpha=0.2,
                    color=color,
                    label=f"95% CI + Obs Error Species {s + 1}"
                    if n_species_plot > 1
                    else "95% CI + Obs Error",
                )
            ax.plot(
                time_months,
                mean_pred[s],
                color=color,
                linewidth=2,
                label=f"Mean Prediction Species {s + 1}"
                if n_species_plot > 1
                else "Mean Prediction",
            )

            _, outputs = run_3pg(initial_state, climate, fixed_params, site, species, n_species)
            ax.plot(time_months, outputs["DBH"], color="black", label="3PG Prediction")

        # Plot observations
        if obs_times is not None and obs_DBH is not None:
            obs_times_np = np.asarray(obs_times)
            obs_DBH_np = np.asarray(obs_DBH)
            ax.scatter(
                obs_times_np,
                obs_DBH_np,
                color="red",
                s=50,
                zorder=5,
                label="Observations",
                edgecolors="black",
                linewidths=1.5,
            )

        ax.set_xlabel("Time (months)", fontsize=12)
        ax.set_ylabel("DBH (cm)", fontsize=12)
        ax.set_title("3PG Model Predictions with Uncertainty (DBH)", fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Residual plots per species
        if obs_times is not None and len(obs_times) > 0:
            fig, ax = plt.subplots(figsize=(10, 5))
            cmap = plt.get_cmap("tab10")
            colors = [cmap(i) for i in range(10)]
            for s in range(n_species_plot):
                pred_at_obs = np.interp(obs_times_np, time_months, mean_pred[s])
                residuals = obs_DBH_np - pred_at_obs
                color = colors[s]
                ax.scatter(
                    pred_at_obs,
                    residuals,
                    alpha=0.6,
                    color=color,
                    label=f"Species {s + 1}" if n_species_plot > 1 else "Residuals",
                )

            ax.axhline(y=0, color="red", linestyle="--", alpha=0.5)
            ax.set_xlabel("Predicted DBH (cm)", fontsize=12)
            ax.set_ylabel("Residuals (cm)", fontsize=12)
            ax.set_title("Residual Plot", fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
    plt.show()


def run_full_analysis(
    initial_state: State,
    climate,
    site,
    species,
    n_species: int,
    obs_DBH: jnp.ndarray,
    obs_times: jnp.ndarray,
    fixed_params,
    num_warmup: int = 1000,
    num_samples: int = 1000,
    num_chains: int = 4,
    output_dir: str = "./hmc_results",
    seed: int = 42,
    show_plots: bool = True,
    predict_with_uncert: bool = False,
) -> tuple[MCMC, dict]:
    """
    Run complete HMC analysis with diagnostics and plotting.

    Returns
    -------
    - mcmc: MCMC object with samples
    - samples: Dictionary of posterior samples
    - diagnostics: Convergence diagnostics
    """
    print("Running HMC inference for 3PG model (DBH)")
    print(f"Number of species: {n_species}")
    print(f"Number of observations: {len(obs_times)}")
    print(f"Warmup samples: {num_warmup}")
    print(f"Posterior samples: {num_samples}")
    print(f"Number of chains: {num_chains}")

    # Run HMC inference
    mcmc, samples = run_hmc_inference(
        initial_state=initial_state,
        climate=climate,
        site=site,
        species=species,
        n_species=n_species,
        obs_DBH=obs_DBH,
        obs_times=obs_times,
        fixed_params=fixed_params,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        seed=seed,
    )

    # Print summary
    print("Convergence Diagnostics")
    print("R-hat values (should be <= 1.0):")
    mcmc.print_summary()

    # Plot results
    print("Generating plots...")

    plot_results(
        mcmc=mcmc,
        climate=climate,
        site=site,
        species=species,
        n_species=n_species,
        initial_state=initial_state,
        fixed_params=fixed_params,
        obs_times=obs_times,
        obs_DBH=obs_DBH,
        params=["alphaCx", "CoeffCond", "Y", "gammaF0", "gammaF1", "tgammaF"],
        # save_path=f"{output_dir}/results",
        show_plots=show_plots,
        predict_with_uncert=predict_with_uncert,
    )

    return mcmc, samples


def run_hmc_analysis(file_path: str, predict_with_uncert: bool = False):
    """Run HMC implementation."""
    initial_state, climate, fixed_params, site_data, species_data, n_species = prepare_data(
        file_path
    )

    # Dummy DBH observations
    obs_times = jnp.array([12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132])
    obs_DBH = jnp.array([14, 14.8, 15.2, 15.9, 15.8, 16.1, 17.3, 17.8, 18.5, 18.8, 19.2])

    # obs_times = jnp.asarray(pl.read_excel(file_path, sheet_name="observed")["idx"])
    # obs_DBH = jnp.asarray(pl.read_excel(file_path, sheet_name="observed")["dbh"])

    # Run analysis
    mcmc, samples = run_full_analysis(
        initial_state=initial_state,
        climate=climate,
        site=site_data,
        species=species_data,
        n_species=n_species,
        obs_DBH=obs_DBH,
        obs_times=obs_times,
        fixed_params=fixed_params,
        num_warmup=200,
        num_samples=200,
        num_chains=2,
        output_dir="./hmc_results",
        show_plots=True,
        predict_with_uncert=predict_with_uncert,
    )

    # Print parameter summaries
    print("\nParameter Summary:")
    for param in ["alphaCx", "CoeffCond", "Y", "gammaF0", "gammaF1", "tgammaF", "tRho"]:
        if param in samples:
            mean_val = jnp.mean(samples[param])
            std_val = jnp.std(samples[param])
            print(f"  {param}: {mean_val:.4f} ± {std_val:.4f}")


if __name__ == "__main__":
    file_path = "./data/data_sspecies_nothinning.xlsx"
    # file_path = "./data/solling_data.xlsx"
    run_hmc_analysis(file_path, predict_with_uncert=True)
    