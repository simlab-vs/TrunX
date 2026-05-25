"""Gradient descent optimization for 3PG model parameters."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd
from jax import grad, jit, value_and_grad
from tqdm import tqdm

from trunx.gp3.model_inputs import Params, SiteData, SpeciesData, State
from trunx.gp3.PG3_model_impl import prepare_data, run_threepg_main
from trunx.gp3.run_3pg import run_3pg


@dataclass
class GradientDescentFitResult:
    """COntainer for results of gradient descent fitting."""

    fitted_params: dict[str, float]
    loss_history: list[float]
    param_history: list[dict[str, float]]


@dataclass
class GradientDescentConfig:
    """Configuration for gradient descent optimization."""

    target_vars: list[str]  # List of target variable names to fit
    fit_params: list[str]  # List of parameter names to optimize
    file_path: str = "./data/solling_data.xlsx"
    observed_sheet: str = "observed"
    species_index: int = 0  # Index of the species to fit parameters for
    optimizer_name: str = "adam"
    learning_rate: float = 1e-3
    n_steps: int = 500
    print_every: int = 50
    standardize_targets = (True,)
    global_clip_norm: float = 1.0  # For gradient clipping
    output_dir: str = "./data/gradient_descent_results"
    image_dir: str = "./images/gradient_descent_results"


def build_observation_indices(observed_data: pd.DataFrame):
    """Build observation indices from the observed data table."""
    print("Building observation indices...")
    if "idx" in observed_data.columns:
        idx_values = observed_data["idx"].to_numpy(dtype=np.int32)
        return jnp.asarray(idx_values, dtype=jnp.int32)
    else:
        print("Warning: 'idx' column not found in observed data.")
        return jnp.array([], dtype=jnp.int32)


def make_loss_function(
    inital_state: State,
    climate: Any,
    base_params: Params,
    site_data: SiteData,
    species_data: SpeciesData,
    n_species: int,
    fit_params: list[str],
    obs_indices: jnp.ndarray,
    obs_scales: dict[str, jnp.ndarray],
    obs_values: dict[str, jnp.ndarray],
    species_index: int,
    target_vars: list[str],
):
    """Create a loss function for gradient descent optimization."""
    n_obs = len(obs_indices)

    def loss_function(params_values: jnp.ndarray) -> jnp.ndarray:
        # Update params with the current parameter vector
        params_dict = {field: getattr(base_params, field) for field in base_params._fields}

        # Update the parameters with new fit_params values
        for idx, param_name in enumerate(fit_params):
            params_array = jnp.asarray(params_dict[param_name])
            if hasattr(params_array, "ndim") and params_array.ndim > 0:
                params_array = params_array.at[species_index].set(params_values[idx])
            else:
                params_array = params_values[idx]
            params_dict[param_name] = params_array

        # Run the 3PG model with the updated parameters
        _, pg3_outputs = run_3pg(
            initial_state=inital_state,
            climate=climate,
            params=Params(**params_dict),
            site=site_data,
            species=species_data,
            n_species=n_species,
        )

        total_squared_error = jnp.asarray(0.0, dtype=jnp.float32)
        for var_name in target_vars:
            pg3_predictions = pg3_outputs[var_name][obs_indices]
            if pg3_predictions.ndim == 2:
                pg3_predictions = pg3_predictions[:, species_index]

            observed_values = obs_values[var_name]
            scale = obs_scales[var_name]
            mask = ~(jnp.isnan(observed_values) | jnp.isnan(pg3_predictions))
            residuals = (pg3_predictions - observed_values) / scale
            squared = jnp.where(mask, residuals**2, 0.0)
            total_squared_error += jnp.sum(squared)

        return total_squared_error / jnp.asarray(n_obs, dtype=jnp.float32)

    return loss_function


def build_optimizer(config: GradientDescentConfig) -> optax.GradientTransformation:
    """Build an Optax optimizer based on the configuration."""
    if config.optimizer_name == "adam":
        base_optimizer = optax.adam(learning_rate=config.learning_rate)
    elif config.optimizer_name == "sgd":
        base_optimizer = optax.sgd(learning_rate=config.learning_rate)
    else:
        raise ValueError("optimizer_name must be either 'adam' or 'sgd'")

    return optax.chain(
        optax.clip_by_global_norm(config.global_clip_norm),  # Gradient clipping
        base_optimizer,
    )


def fit_with_gradient_descent(config: GradientDescentConfig):
    """Fit 3PG model parameter using gradient descent."""
    observed_data = pd.read_excel(config.file_path, sheet_name=config.observed_sheet)

    # Prepare data and initial state
    initial_state, climate, params, site_data, species_data, n_species, species_names = (
        prepare_data(config.file_path)
    )

    obs_indices = build_observation_indices(observed_data)

    obs_values: dict[str, jnp.ndarray] = {}
    obs_scales: dict[str, jnp.ndarray] = {}
    for var_name in config.target_vars:
        if var_name not in observed_data.columns:
            raise KeyError(f"Target variable '{var_name}' is not in observed data sheet")
        observed_np = observed_data[var_name].to_numpy(dtype=float)
        obs_values[var_name] = jnp.asarray(observed_np, dtype=jnp.float32)

        if config.standardize_targets:
            scale = float(np.nanstd(observed_np))
            obs_scales[var_name] = jnp.asarray(max(scale, 1e-6), dtype=jnp.float32)
        else:
            obs_scales[var_name] = jnp.asarray(1.0, dtype=jnp.float32)

    initial_values = []
    for name in config.fit_params:
        param_array = getattr(params, name)
        param_array = jnp.asarray(param_array)
        if param_array.ndim > 0:
            initial_values.append(float(param_array[config.species_index]))
        else:
            initial_values.append(float(param_array))

    param_values = jnp.asarray(initial_values, dtype=jnp.float32)

    loss_function = make_loss_function(
        inital_state=initial_state,
        climate=climate,
        base_params=params,
        site_data=site_data,
        species_data=species_data,
        n_species=n_species,
        fit_params=config.fit_params,
        obs_indices=obs_indices,
        obs_values=obs_values,
        obs_scales=obs_scales,
        species_index=config.species_index,
        target_vars=config.target_vars,
    )

    value_and_grad_fn = jax.jit(jax.value_and_grad(loss_function))
    optimizer = build_optimizer(config)
    opt_state = optimizer.init(param_values)

    loss_history: list[float] = []
    param_history: list[dict[str, float]] = []
    for step in range(config.n_steps):
        loss_value, grads = value_and_grad_fn(param_values)
        updates, opt_state = optimizer.update(grads, opt_state)
        param_values = optax.apply_updates(param_values, updates)

        loss_history.append(float(loss_value))
        params_values_np = np.array(param_values)
        step_params = {
            param_name: float(params_values_np[idx])
            for idx, param_name in enumerate(config.fit_params)
        }
        param_history.append(step_params)

        if step % config.print_every == 0 or step == config.n_steps - 1:
            params_text = ", ".join(f"{k}={v:.6f}" for k, v in step_params.items())
            print(f"step={step:04d} loss={float(loss_value):.6f} params: {params_text}")

    param_values_np = np.asarray(param_values)
    fitted_params = {
        param_name: float(param_values_np[i]) for i, param_name in enumerate(config.fit_params)
    }

    return GradientDescentFitResult(
        fitted_params=fitted_params,
        loss_history=loss_history,
        param_history=param_history,
    )


def apply_fitted_params(
    base_params: Params,
    fit_params: list[str],
    fitted_values: dict[str, float],
    species_index: int,
) -> Params:
    """Build a new Params object with fitted values.

    Parameters
    ----------
    base_params : Params
        Original parameters from data input.
    fit_params : list[str]
        Parameter names to overwrite.
    fitted_values : dict[str, float]
        Fitted values by parameter name.
    species_index : int
        Species index to update.

    Returns
    -------
    Params
        Updated parameters object.
    """
    params_dict = {field: getattr(base_params, field) for field in base_params._fields}
    for param_name in fit_params:
        param_array = jnp.asarray(params_dict[param_name])
        updated_value = jnp.asarray(fitted_values[param_name], dtype=jnp.float32)
        if param_array.ndim > 0:
            param_array = param_array.at[species_index].set(updated_value)
        else:
            param_array = updated_value
        params_dict[param_name] = param_array

    return Params(**params_dict)


def build_predicted_dataframe(
    config: GradientDescentConfig,
    fitted_params: dict[str, float],
) -> pd.DataFrame:
    """Create observed-vs-predicted table at observation times.

    Parameters
    ----------
    config : GradientDescentConfig
        Fitting configuration.
    fitted_params : dict[str, float]
        Fitted parameter values.

    Returns
    -------
    pd.DataFrame
        Comparison table for each target variable.
    """
    initial_state, climate, params, site_data, species_data, n_species, _ = prepare_data(
        config.file_path
    )

    fitted_model_params = apply_fitted_params(
        base_params=params,
        fit_params=config.fit_params,
        fitted_values=fitted_params,
        species_index=config.species_index,
    )

    _, outputs_fitted = run_3pg(
        initial_state=initial_state,
        climate=climate,
        params=fitted_model_params,
        site=site_data,
        species=species_data,
        n_species=n_species,
    )

    _, outputs_default = run_3pg(
        initial_state=initial_state,
        climate=climate,
        params=params,
        site=site_data,
        species=species_data,
        n_species=n_species,
    )

    df_data = pd.DataFrame()
    for var_name in ["DBH", "LAI", "GPP", "WS", "WF", "WR"]:
        pred_fitted = outputs_fitted[var_name]
        if pred_fitted.ndim == 2:
            pred_fitted = pred_fitted[:, config.species_index]

        pred_default = outputs_default[var_name]
        if pred_default.ndim == 2:
            pred_default = pred_default[:, config.species_index]

        df_data[f"pred_fitted_{var_name}"] = np.asarray(pred_fitted)
        df_data[f"pred_default_{var_name}"] = np.asarray(pred_default)
        df_data[f"pred_{var_name}"] = np.asarray(pred_fitted)

    start_date = pd.Timestamp(year=int(site_data.year_i), month=int(site_data.month_i), day=1)
    df_data["date"] = pd.date_range(start=start_date, periods=len(df_data), freq="ME")
    return pd.DataFrame(df_data)


def plot_observed_vs_predicted(
    comparison_df: pd.DataFrame,
    target_vars: list[str],
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """Plot observed, fitted, and default predictions over observation time."""
    n_vars = len(target_vars)
    observed_data = pd.read_excel(config.file_path, sheet_name=config.observed_sheet)
    x_col = "date"
    comparison_df["date"] = pd.to_datetime(comparison_df["date"])

    observed_dates = None
    if {"year", "month"}.issubset(observed_data.columns):
        observed_dates = pd.to_datetime(
            observed_data[["year", "month"]].assign(day=1).astype(int)
        ) + pd.offsets.MonthEnd(0)
    observed_data["date"] = observed_dates

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    if n_vars == 1:
        axes = [axes]
    axes = np.ravel(axes).tolist()

    for ax, var_name in zip(axes, ["DBH", "LAI", "GPP", "WS", "WF", "WR"], strict=True):
        pred_fitted_col = f"pred_fitted_{var_name}"
        pred_default_col = f"pred_default_{var_name}"

        valid_fitted = ~comparison_df[pred_fitted_col].isna()
        valid_default = ~comparison_df[pred_default_col].isna()

        ax.plot(
            comparison_df.loc[valid_fitted, x_col],
            comparison_df.loc[valid_fitted, pred_fitted_col],
            label=" Gradient descent fit",
            linewidth=2,
            color="tab:blue",
        )
        ax.plot(
            comparison_df.loc[valid_default, x_col],
            comparison_df.loc[valid_default, pred_default_col],
            label=" Default parameters",
            linewidth=2,
            color="tab:green",
            linestyle="--",
        )

        if var_name in observed_data.columns:
            ax.scatter(
                observed_data["date"],
                observed_data[var_name],
                label=f"Observed {var_name}",
                color="tab:red",
                s=50,
                zorder=5,
            )

        if x_col == "date":
            locator = mdates.AutoDateLocator()
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

        ax.set_ylabel(var_name)
        ax.grid(alpha=0.3)
        ax.legend()

    axes[-1].set_xlabel("Date")
    fig.suptitle(f"Observed vs Predicted Trajectories, [{', '.join(config.fit_params)}]")
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()


def plot_loss_over_iterations(
    loss_history: list[float], save_path: str | None = None, show: bool = True
) -> None:
    """Plot optimization loss over iterations.

    Parameters
    ----------
    loss_history : list[float]
        Loss value at each iteration.
    save_path : str | None
        Optional output image path.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(np.arange(len(loss_history)), loss_history, color="tab:blue", linewidth=2)
    ax.set_xlabel("Iteration t")
    ax.set_ylabel("Loss")
    ax.set_title(f"Gradient Descent Loss Trajectory: {', '.join(config.fit_params)}")
    ax.grid(alpha=0.3)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()


def save_loss_history(loss_history: list[float], save_path: str) -> None:
    """Save loss trajectory to CSV.

    Parameters
    ----------
    loss_history : list[float]
        Loss value at each iteration
    save_path : str
        Output CSV path
    """
    loss_df = pd.DataFrame({"iteration": np.arange(len(loss_history)), "loss": loss_history})
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    loss_df.to_csv(save_path, index=False)


if __name__ == "__main__":
    fit_params = ["alphaCx", "Y", "CoeffCond", "aWS", "nWS", "Tmin", "rAge"]

    config = GradientDescentConfig(
        file_path="./data/solling_data.xlsx",
        observed_sheet="observed",
        fit_params=fit_params,
        target_vars=["DBH", "WS", "WF", "WR"],
        n_steps=500,
        optimizer_name="adam",
        learning_rate=1e-3,
        global_clip_norm=1.0,
    )

    # _, _ = run_threepg_main(
    #     file_path=config.file_path,
    #     observed_data = None,
    #     plot_output=True,
    #     r_comparison=True,
    #     show_plots=False,
    # )
    fit_results = fit_with_gradient_descent(config)

    print("\nFitted parameters:")
    for name, value in fit_results.fitted_params.items():
        print(f"  {name}: {value:.6f}")

    output_dir = Path(config.output_dir)
    image_dir = Path(config.image_dir)
    plot_loss_over_iterations(
        fit_results.loss_history,
        save_path=str(image_dir / "Gradient_descent_loss.png"),
        show=False,
    )
    save_loss_history(
        fit_results.loss_history,
        save_path=str(output_dir / "Gradient_descent_loss_history.csv"),
    )

    comparison_df = build_predicted_dataframe(
        config=config,
        fitted_params=fit_results.fitted_params,
    )
    comparison_df.to_csv(output_dir / "Gradient_descent_observed_vs_predicted.csv", index=False)
    plot_observed_vs_predicted(
        comparison_df,
        target_vars=config.target_vars,
        save_path=str(image_dir / "Gradient_descent_observed_vs_predicted.png"),
        show=False,
    )

    plt.show()
