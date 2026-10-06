"""Bayesian calibration of the learnable nutrition modifier with PyMC.

Uses the same plots, modifier inputs (clipped and standardised deposition) and evaluation as
the gradient-descent training in `nn_nutrition_modifier.py`, but samples a posterior of the
modifier weights, and optionally of physiological parameters, instead of a point estimate.
Follows `bayesiancalibrations/pymc_param_est_multiplots.py`: the 3PG log-likelihood of all
plots is a JAX function of one flat parameter vector, wrapped as a PyTensor Op with its JAX
gradient, and is sampled with NUTS or DEMetropolisZ in checkpointed chunks.

Priors: Normal(0, `modifier_prior_scale`) on each `poly_nm` weight (the modifier is centred on
1, no effect), Uniform over the `param_bound` sheet's bounds for `config.fit_phys_params`, and
Uniform over the `error_param` sheet's bounds for one `err_{var}` per target variable, shared
across plots. Likelihood: Normal, as in `bayesiancalibrations/pymc_param_est.py`.
"""

import os
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import arviz as az
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pymc as pm
import pytensor.tensor as pt
from jax.scipy.stats import norm

from trunx.config import images_folder, results_data_folder
from trunx.gp3.bayesiancalibrations.bayesian_config import species_plot_ids
from trunx.gp3.bayesiancalibrations.calibration_utils import clip_defaults_to_priors
from trunx.gp3.bayesiancalibrations.load_files import (
    load_param_defaults_from_file,
    load_priors_from_file,
)
from trunx.gp3.bayesiancalibrations.pymc_icp_plots import prepare_plot_input
from trunx.gp3.bayesiancalibrations.pymc_param_est_multiplots import (
    JaxLogLikeOp,
    sample_with_checkpoints,
)
from trunx.gp3.bayesiancalibrations.save_load_results import save_results
from trunx.gp3.extended_helper import poly_nm
from trunx.gp3.gradient_descent import apply_fitted_params
from trunx.gp3.model_inputs import ExtendedParams, InputData
from trunx.gp3.nn_nutrition_modifier import (
    NutritionModifierConfig,
    NutritionModifierFitResult,
    _months_to_dates,
    _plot_variables,
    build_observation_data,
    build_predicted_series,
    compute_rmse,
    input_scaling,
    phys_param_bounds,
    phys_param_defaults,
    summarize_rmse,
)
from trunx.gp3.plots_nutrition_modifier import (
    plot_deposition_density,
    plot_deposition_over_time,
    plot_learned_modifier_vs_deposition,
    plot_modifier_effect_over_time,
    plot_modifier_effect_vs_deposition,
    plot_modifier_response_surface,
    plot_observed_vs_predicted,
    plot_posterior_predictions,
)
from trunx.gp3.prepare_data import prepare_data
from trunx.gp3.run_3pg import run_3pg

# One plot's model inputs, simulation month of each observation and observed values
PlotObservations = tuple[InputData, jnp.ndarray, dict[str, jnp.ndarray]]


def load_plot_observations(config: NutritionModifierConfig) -> list[PlotObservations]:
    """Load each plot's model inputs and unscaled observations of `config.target_vars`.

    Parameters
    ----------
    config : NutritionModifierConfig
        Plots (`config.file_paths`) and target variables.

    Returns
    -------
    list[PlotObservations]
        One entry per plot, in `config.file_paths` order.
    """
    plots = []
    for file_path in config.file_paths:
        input_data = prepare_data(file_path)
        obs_indices, obs_values, _ = build_observation_data(
            file_path, input_data.site, config.target_vars, standardize_targets=False
        )
        plots.append((input_data, obs_indices, obs_values))
    return plots


def predictions_at_observations(
    plots: list[PlotObservations],
    extended_params: ExtendedParams,
    phys_params: dict[str, Any],
    config: NutritionModifierConfig,
) -> list[dict[str, tuple[jnp.ndarray, jnp.ndarray]]]:
    """Run 3PG with the modifier on every plot and pair its predictions with the observations.

    Parameters
    ----------
    plots : list[PlotObservations]
        Plots, see `load_plot_observations`.
    extended_params : ExtendedParams
        Modifier weights and input standardisation.
    phys_params : dict[str, Any]
        Values of the calibrated physiological parameters for `config.species_index`.
    config : NutritionModifierConfig
        Modifier inputs, target variables and species.

    Returns
    -------
    list[dict[str, tuple[jnp.ndarray, jnp.ndarray]]]
        Per plot, the (predicted, observed) values of each target variable at the
        observation times; unobserved values are NaN.
    """
    predictions = []
    for input_data, obs_indices, obs_values in plots:
        params = apply_fitted_params(
            input_data.params, list(phys_params), phys_params, config.species_index
        )
        _, outputs = run_3pg(
            input_data.initial_state,
            input_data.climate,
            params,
            input_data.site,
            input_data.species,
            input_data.deposition,
            extended_params,
            config.modifier_fn,
            config.input_vars,
        )
        plot_predictions = {}
        for var_name in config.target_vars:
            predicted = outputs[var_name][obs_indices]
            if predicted.ndim == 2:
                predicted = predicted[:, config.species_index]
            plot_predictions[var_name] = (predicted, obs_values[var_name])
        predictions.append(plot_predictions)
    return predictions


def posterior_mean_fit(
    idata: az.InferenceData, input_preparation: ExtendedParams, phys_names: list[str]
) -> NutritionModifierFitResult:
    """Posterior means as a fit result, for the evaluation and plots of `nn_nutrition_modifier`.

    Parameters
    ----------
    idata : az.InferenceData
        Posterior draws of `nutrition_modifier_pymc_model`.
    input_preparation : ExtendedParams
        Standardisation of the modifier inputs used in the calibration.
    phys_names : list[str]
        Calibrated physiological parameters.

    Returns
    -------
    NutritionModifierFitResult
        Posterior-mean modifier weights and physiological parameters, without histories.
    """
    posterior = cast(Any, idata).posterior
    return NutritionModifierFitResult(
        fitted_modifier_params=jnp.asarray(
            posterior["modifier_params"].mean(("chain", "draw")).values
        ),
        fitted_phys_params={name: float(posterior[name].mean()) for name in phys_names},
        loss_history=[],
        param_history=[],
        input_mean=jnp.asarray(input_preparation.input_mean),
        input_std=jnp.asarray(input_preparation.input_std),
    )


def calibration_setup(
    config: NutritionModifierConfig,
) -> tuple[ExtendedParams, dict[str, tuple[float, float]], dict[str, Any]]:
    """Input standardisation, priors and chains' starting values of a modifier calibration.

    Parameters
    ----------
    config : NutritionModifierConfig
        Plots, targets and modifier inputs; only `poly_nm` is supported, since MLP weights
        have many equivalent solutions.

    Returns
    -------
    tuple[ExtendedParams, dict[str, tuple[float, float]], dict[str, Any]]
        Standardisation of the modifier inputs (without weights); Uniform prior bounds of
        `config.fit_phys_params` and of one `err_{var}` per target variable; and starting
        values: no modifier effect, the first plot's physiological parameters and the
        default noise scales, strictly inside the priors.
    """
    if config.modifier_fn is not poly_nm:
        raise ValueError("Only poly_nm is supported: MLP weights have many equivalent solutions")

    input_mean, input_std = input_scaling(config)
    input_preparation = ExtendedParams(
        modifier_params=None, input_mean=input_mean, input_std=input_std
    )
    phys_bounds = phys_param_bounds(config)
    error_names = [f"err_{var}" for var in config.target_vars]
    priors = {**phys_bounds, **load_priors_from_file(config.file_paths[0], error_names)}

    phys_defaults = {
        name: float(value)
        for name, value in phys_param_defaults(config, list(phys_bounds)).items()
    }
    error_defaults = load_param_defaults_from_file(config.file_paths[0], error_names)
    param_defaults: dict[str, Any] = {
        **clip_defaults_to_priors({**phys_defaults, **error_defaults}, priors),
        "modifier_params": jnp.zeros((2,) * len(config.input_vars)),
    }
    return input_preparation, priors, param_defaults


def predict_posterior_bands(
    config: NutritionModifierConfig,
    idata: az.InferenceData,
    fit_result: NutritionModifierFitResult,
    file_path: str,
    num_predictions: int = 500,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Posterior predictive mean and 95% interval of one plot, and its default run.

    `num_predictions` random posterior draws (of the modifier weights and the calibrated
    physiological parameters) are run through 3PG, as in
    `pymc_param_est.predict_with_uncertainity`; the interval spans their 2.5th to 97.5th
    percentiles.

    Parameters
    ----------
    config : NutritionModifierConfig
        Modifier inputs, plot variables and species.
    idata : az.InferenceData
        Posterior draws.
    fit_result : NutritionModifierFitResult
        Posterior means; gives the input standardisation and the calibrated
        physiological parameters.
    file_path : str
        Input file of the plot to predict.
    num_predictions : int
        Number of posterior draws to run.
    seed : int
        Seed of the draw selection.

    Returns
    -------
    dict[str, np.ndarray]
        `"dates"` (one per simulated month) plus, for each plot variable,
        `"mean_<var>"`, `"lower_<var>"` and `"upper_<var>"` over the draws and
        `"default_<var>"` (default physiological parameters, no nutrition modifier).
    """
    posterior = cast(Any, idata).posterior
    n_draws = posterior.sizes["draw"]
    picked = np.random.default_rng(seed).choice(
        posterior.sizes["chain"] * n_draws,
        size=min(num_predictions, posterior.sizes["chain"] * n_draws),
        replace=False,
    )
    chains, draws = np.divmod(picked, n_draws)
    weights = jnp.asarray(posterior["modifier_params"].values[chains, draws])
    phys_draws = {
        name: jnp.asarray(posterior[name].values[chains, draws])
        for name in fit_result.fitted_phys_params
    }

    input_data = prepare_data(file_path)
    plot_variables = _plot_variables(config)

    def run_draw(modifier_params: jnp.ndarray, phys_params: dict[str, Any]) -> dict[str, Any]:
        """Plot variables of one posterior draw."""
        params = apply_fitted_params(
            input_data.params, list(phys_params), phys_params, config.species_index
        )
        extended_params = ExtendedParams(
            modifier_params=modifier_params,
            input_mean=fit_result.input_mean,
            input_std=fit_result.input_std,
        )
        _, outputs = run_3pg(
            input_data.initial_state,
            input_data.climate,
            params,
            input_data.site,
            input_data.species,
            input_data.deposition,
            extended_params,
            config.modifier_fn,
            config.input_vars,
        )
        return {name: outputs[name] for name in plot_variables}

    draw_outputs = jax.vmap(run_draw)(weights, phys_draws)
    _, default_outputs = run_3pg(
        input_data.initial_state,
        input_data.climate,
        input_data.params,
        input_data.site,
        input_data.species,
    )

    def species_series(values: Any) -> np.ndarray:
        """Select the calibrated species from a per-species output."""
        values = np.asarray(values)
        return values[..., config.species_index] if values.ndim >= 2 else values

    series: dict[str, np.ndarray] = {}
    for name in plot_variables:
        samples = np.asarray(draw_outputs[name])
        if samples.ndim == 3:
            samples = samples[..., config.species_index]
        series[f"mean_{name}"] = samples.mean(axis=0)
        series[f"lower_{name}"] = np.percentile(samples, 2.5, axis=0)
        series[f"upper_{name}"] = np.percentile(samples, 97.5, axis=0)
        series[f"default_{name}"] = species_series(default_outputs[name])

    n_months = len(series[f"mean_{plot_variables[0]}"])
    start_year = int(np.asarray(input_data.site.year_i).reshape(-1)[0])
    start_month = int(np.asarray(input_data.site.month_i).reshape(-1)[0])
    series["dates"] = _months_to_dates(start_year, start_month, n_months)
    return series


def evaluate_posterior_mean(
    config: NutritionModifierConfig,
    idata: az.InferenceData,
    fit_result: NutritionModifierFitResult,
    plot_ids: list[str],
    test_file_paths: list[str],
    test_plot_ids: list[str],
) -> pl.DataFrame:
    """Save the trace and the posterior-mean fit's figures, and summarise its train/test RMSE.

    Figures go to `config.image_dir`: the trace; observed vs predicted with the posterior
    means and the deposition density for the training (`config.file_paths`) and test
    plots; and, on the training plots, the modifier's effect on `alpha_c` and
    `f_nutri_classic_learnable` over time and against deposition, the deposition over
    time and the modifier's response surface.

    Parameters
    ----------
    config : NutritionModifierConfig
        Calibration configuration; its `file_paths` are the training plots.
    idata : az.InferenceData
        Posterior draws.
    fit_result : NutritionModifierFitResult
        Posterior means, see `posterior_mean_fit`.
    plot_ids : list[str]
        Identifiers of the training plots, in `config.file_paths` order.
    test_file_paths, test_plot_ids : list[str]
        Input files and identifiers of the held-out plots.

    Returns
    -------
    pl.DataFrame
        RMSE summary per split and variable, see `summarize_rmse`.
    """
    image_dir = Path(config.image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    az.plot_trace(idata)
    plt.savefig(image_dir / "trace.png", dpi=200, bbox_inches="tight")

    test_config = replace(config, file_paths=test_file_paths)
    predicted_series, rmse_tables = [], []
    for split, split_config, split_ids in (
        ("train", config, plot_ids),
        ("test", test_config, test_plot_ids),
    ):
        split_series = [
            build_predicted_series(config, file_path, fit_result)
            for file_path in split_config.file_paths
        ]
        rmse_tables += [
            compute_rmse(config, file_path, series).with_columns(pl.lit(split).alias("split"))
            for file_path, series in zip(split_config.file_paths, split_series, strict=True)
        ]
        plot_observed_vs_predicted(
            split_config,
            split_ids,
            split_series,
            save_path=str(image_dir / f"observed_vs_predicted_{split}.png"),
            show=False,
        )
        plot_deposition_density(
            split_config,
            split_ids,
            save_path=str(image_dir / f"deposition_density_{split}.png"),
            show=False,
        )
        if split == "train":
            predicted_series = split_series
    for var_name in ("alpha_c", "f_nutri_classic_learnable"):
        plot_modifier_effect_over_time(
            config,
            plot_ids,
            predicted_series,
            var_name=var_name,
            save_path=str(image_dir / f"{var_name}.png"),
            show=False,
        )
    plot_modifier_effect_vs_deposition(
        config,
        predicted_series,
        plot_ids,
        var_name="alpha_c",
        save_path=str(image_dir / "alpha_c_deposition.png"),
        show=False,
    )
    plot_learned_modifier_vs_deposition(
        config,
        predicted_series,
        plot_ids,
        save_path=str(image_dir / "f_nutri_classic_learnable_deposition.png"),
        show=False,
    )
    plot_deposition_over_time(
        config,
        predicted_series,
        plot_ids,
        save_path=str(image_dir / "deposition_over_time.png"),
        show=False,
    )
    plot_modifier_response_surface(
        config,
        plot_ids,
        fit_result,
        save_path=str(image_dir / "modifier_response_surface.png"),
        show=False,
    )
    return summarize_rmse(pl.concat(rmse_tables))


def make_loglikelihood(
    plots: list[PlotObservations],
    priors: dict[str, tuple[float, float]],
    input_preparation: ExtendedParams,
    config: NutritionModifierConfig,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Build the Normal log-likelihood of all plots as a function of a flat parameter vector.

    The vector holds the flattened modifier weights, then one value per `priors` entry, in
    order.

    Parameters
    ----------
    plots : list[PlotObservations]
        Plots to calibrate, see `load_plot_observations`.
    priors : dict[str, tuple[float, float]]
        Calibrated physiological parameters and `err_*` noise scales.
    input_preparation : ExtendedParams
        Standardisation of the modifier inputs.
    config : NutritionModifierConfig
        Modifier inputs, target variables and species.

    Returns
    -------
    Callable[[jnp.ndarray], jnp.ndarray]
        JAX-differentiable log-likelihood.
    """
    weights_shape = (2,) * len(config.input_vars)
    n_weights = int(np.prod(weights_shape))

    def loglikelihood(param_values: jnp.ndarray) -> jnp.ndarray:
        """Log-likelihood of the observed target variables of all plots."""
        values = dict(zip(priors, param_values[n_weights:], strict=True))
        phys_params = {name: v for name, v in values.items() if not name.startswith("err_")}
        extended_params = input_preparation._replace(
            modifier_params=param_values[:n_weights].reshape(weights_shape)
        )
        log_likelihood = jnp.array(0.0)
        for plot_predictions in predictions_at_observations(
            plots, extended_params, phys_params, config
        ):
            for var_name, (predicted, observed) in plot_predictions.items():
                is_observed = ~jnp.isnan(observed)
                # Unobserved rows are masked out; 0 keeps NaN out of the gradients
                log_probs = norm.logpdf(
                    predicted,
                    loc=jnp.where(is_observed, observed, 0.0),
                    scale=values[f"err_{var_name}"],
                )
                log_likelihood = log_likelihood + jnp.sum(jnp.where(is_observed, log_probs, 0.0))
        return log_likelihood

    return loglikelihood


def nutrition_modifier_pymc_model(
    plots: list[PlotObservations],
    priors: dict[str, tuple[float, float]],
    input_preparation: ExtendedParams,
    modifier_prior_scale: float,
    config: NutritionModifierConfig,
) -> pm.Model:
    """PyMC model of the modifier weights and physiological parameters across plots.

    Parameters
    ----------
    plots : list[PlotObservations]
        Plots to calibrate, see `load_plot_observations`.
    priors : dict[str, tuple[float, float]]
        Uniform prior bounds of the physiological parameters and the `err_*` noise scales.
    input_preparation : ExtendedParams
        Standardisation of the modifier inputs.
    modifier_prior_scale : float
        Standard deviation of the Normal prior of each modifier weight.
    config : NutritionModifierConfig
        Modifier inputs, target variables and species.

    Returns
    -------
    pm.Model
        Model with a `modifier_params` Normal prior, one Uniform prior per `priors` entry
        and the 3PG log-likelihood as a potential.
    """
    loglike_op = JaxLogLikeOp(make_loglikelihood(plots, priors, input_preparation, config))
    with pm.Model() as model:
        modifier_params = pm.Normal(
            "modifier_params",
            mu=0.0,
            sigma=modifier_prior_scale,
            shape=(2,) * len(config.input_vars),
        )
        param_vars = [pm.Uniform(name, lower=lo, upper=hi) for name, (lo, hi) in priors.items()]
        param_vector = pt.concatenate([modifier_params.flatten(), pt.stack(param_vars)])
        pm.Potential("likelihood", cast(Any, loglike_op(param_vector)))
    return model


def calibrate_nutrition_modifier_pymc(
    config: NutritionModifierConfig,
    num_warmup: int = 1000,
    num_samples: int = 1000,
    chains: int = 4,
    cores: int | None = None,
    step_method: str = "nuts",
    modifier_prior_scale: float = 0.5,
    output_dir: str | None = None,
    checkpoint_every: int = 500,
    resume_tune: int = 200,
) -> tuple[az.InferenceData, NutritionModifierFitResult]:
    """Sample the posterior of the nutrition modifier (and `config.fit_phys_params`) with PyMC.

    Parameters
    ----------
    config : NutritionModifierConfig
        Plots, targets and modifier inputs, see `calibration_setup`.
    num_warmup, num_samples, chains, cores, step_method, resume_tune
        Sampling settings, see `pymc_param_est_multiplots.sample_with_checkpoints`.
    modifier_prior_scale : float
        Standard deviation of the Normal prior of each modifier weight, on standardised inputs.
    output_dir : str | None
        Directory for checkpoints and the final `inference_data.nc`. Re-running with the same
        `output_dir` resumes an interrupted run. If None, nothing is saved.
    checkpoint_every : int
        Post-tuning draws per chain between checkpoints.

    Returns
    -------
    tuple[az.InferenceData, NutritionModifierFitResult]
        Posterior draws, and the posterior means as a fit result.
    """
    input_preparation, priors, param_defaults = calibration_setup(config)
    model = nutrition_modifier_pymc_model(
        load_plot_observations(config), priors, input_preparation, modifier_prior_scale, config
    )
    idata = sample_with_checkpoints(
        model,
        num_warmup=num_warmup,
        num_samples=num_samples,
        chains=chains,
        cores=cores,
        step_method=step_method,
        initvals={name: np.asarray(value) for name, value in param_defaults.items()},
        checkpoint_dir=output_dir,
        checkpoint_every=checkpoint_every,
        resume_tune=resume_tune,
    )
    if output_dir is not None:
        save_results(mcmc=idata, output_dir=output_dir)
    phys_names = [name for name in priors if not name.startswith("err_")]
    return idata, posterior_mean_fit(idata, input_preparation, phys_names)


if __name__ == "__main__":
    # Calibrate on these plots; the remaining plots of the species are held out for testing
    plot_ids = ["04.1402", "04.1403", "14.0017", "59.0008"]
    test_plot_ids = [p for p in species_plot_ids["Picea abies"] if p not in plot_ids]
    literature_source = "Trotsiuk"
    file_paths = [prepare_plot_input(p, literature_source=literature_source) for p in plot_ids]
    test_file_paths = [
        prepare_plot_input(p, literature_source=literature_source) for p in test_plot_ids
    ]

    config = NutritionModifierConfig(
        file_paths=file_paths,
        target_vars=["DBH"],
        plot_variables=["BA", "DBH", "Height", "WS", "WF", "WR"],
        input_vars=("N", "S"),
        standardize_inputs=True,
        fit_phys_params=[],
        modifier_fn=poly_nm,
        image_dir=str(images_folder / "pymc_nutrition_modifier"),
    )
    output_dir = os.path.join(results_data_folder, "pymc_nutrition_modifier")

    idata, fit_result = calibrate_nutrition_modifier_pymc(
        config,
        num_warmup=1000,
        num_samples=1000,
        chains=4,
        step_method="nuts",
        output_dir=output_dir,
        checkpoint_every=250,
    )
    print(az.summary(idata))
    rmse_summary = evaluate_posterior_mean(
        config, idata, fit_result, plot_ids, test_file_paths, test_plot_ids
    )
    with pl.Config(tbl_rows=-1, tbl_cols=-1, float_precision=3):
        print(rmse_summary)

    # 95% posterior prediction interval of the training and test plots
    for split, split_file_paths, split_ids in (
        ("train", file_paths, plot_ids),
        ("test", test_file_paths, test_plot_ids),
    ):
        plot_posterior_predictions(
            replace(config, file_paths=split_file_paths),
            split_ids,
            [
                predict_posterior_bands(config, idata, fit_result, file_path)
                for file_path in split_file_paths
            ],
            save_path=os.path.join(config.image_dir, f"posterior_predictions_{split}.png"),
            show=False,
        )
    print(f"Saved results to {output_dir} and figures to {config.image_dir}")
