"""Shared utilities for Bayesian calibration workflows."""

import os
from typing import Any

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
from jax import numpy as jnp
from jax import tree_util, vmap

from trunx.gp3.model_inputs import ClimateData, State
from trunx.gp3.run_3pg import run_3pg


def predict_from_parameter_draws(
    parameter_draws: dict[str, Any],
    param_names: list[str],
    initial_state: State,
    climate: Any,
    site: Any,
    species: Any,
    fixed_params: Any,
    observations: dict[str, tuple[jnp.ndarray, jnp.ndarray]],
    n_species: int,
) -> dict[str, tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    """Run 3PG with sampled parameters and compute posterior prediction intervals."""
    physiology_names = [name for name in param_names if name in fixed_params._fields]
    if not physiology_names:
        return {}

    param_values = [jnp.asarray(parameter_draws[name]) for name in physiology_names]

    def run_model(*params: jnp.ndarray) -> dict[str, Any]:
        """Run 3PG model with parameters as separate arguments."""
        param_dict = dict(zip(physiology_names, params, strict=True))
        params_obj = fixed_params._replace(**param_dict)
        _, outputs = run_3pg(initial_state, climate, params_obj, site, species)
        return outputs

    batched_run = vmap(run_model, in_axes=(0,) * len(physiology_names))
    all_outputs = batched_run(*param_values)
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


def plot_inference_results(
    inf_data: az.InferenceData,
    params: list[str] | None,
    observations: dict[str, tuple[jnp.ndarray, jnp.ndarray]] | None = None,
    predictions: dict[str, tuple[Any, Any, Any]] | None = None,
    climate: ClimateData | None = None,
    output_dir: str | None = None,
) -> None:
    """Plot trace/posterior diagnostics and optional prediction intervals."""
    if params is None:
        params = [str(name) for name in inf_data["posterior"].data_vars]

    az.plot_trace(inf_data, var_names=params)
    if output_dir is not None:
        plt.gcf().savefig(os.path.join(output_dir, "trace_plots.png"))

    az.plot_posterior(inf_data, var_names=params)
    if output_dir is not None:
        plt.gcf().savefig(os.path.join(output_dir, "posterior_plots.png"))

    summary = az.summary(inf_data, var_names=params)
    print(summary)

    if predictions is not None and observations is not None:
        assert climate is not None, "climate is required to plot predictions"

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
