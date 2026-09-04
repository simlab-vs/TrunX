"""Learnable componenet of the 3PG model: a nutrition modifier optimized using gradient descent."""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import optax
import polars as pl

from trunx.config import images_folder
from trunx.gp3.bayesiancalibrations.pymc_icp_plots import prepare_plot_input
from trunx.gp3.extended_helper import INPUT_VARIABLES, poly_nm
from trunx.gp3.model_inputs import ExtendedParams, InputData, SiteData
from trunx.gp3.prepare_data import prepare_data
from trunx.gp3.run_3pg import run_3pg

_METRIC_LABELS = {
    "DBH": "DBH (cm)",
    "WS": "Stem Biomass (t DM ha⁻¹)",
    "WF": "Foliage Biomass (t DM ha⁻¹)",
    "WR": "Root Biomass (t DM ha⁻¹)",
    "Height": "Height (m)",
    "BA": "Basal Area (m² ha⁻¹)",
}


@dataclass
class NutritionModifierConfig:
    """Configuration for the nutrition modifier."""

    file_path: str  # Path to the input data file
    target_vars: list[str]  # List of target variables to optimize against

    observed_sheet: str = "observed"  # Sheet name for observed data
    species_index: int = 0  # Index of the species to optimize for
    # Which of ("N", "S", "T_avg") the modifier is built over
    input_vars: tuple[str, ...] = INPUT_VARIABLES
    optimizer_name: str = "adam"  # Optimizer name: 'adam' or 'sgd'
    learning_rate: float = 1e-3  # Learning rate for the optimizer
    global_clip_norm: float = 1.0  # Global norm for gradient clipping
    num_epochs: int = 1000  # Number of training epochs
    standardize_targets: bool = True  # Whether to standardize target variables
    image_dir: str = field(default_factory=lambda: str(images_folder / "nn_nutrition_modifier"))


@dataclass
class NutritionModifierFitResult:
    """Container for nutrition modifier training results."""

    fitted_modifier_params: Any
    loss_history: list[float]
    param_history: list[Any]


def init_modifier_params(input_vars: tuple[str, ...] = INPUT_VARIABLES) -> jnp.ndarray:
    """Build a neutral (all-zero) starting point for `poly_nm`.

    Parameters
    ----------
    input_vars : tuple[str, ...]
        Which of `("N", "S", "T_avg")` the modifier is built over, and in what
        order — must match what's passed to `run_3pg`/`train_nutrition_modifier`.
    """
    return jnp.zeros(tuple(2 for _ in input_vars))


def make_loss_function(
    input_data: InputData,
    target_vars: list[str],
    obs_indices: jnp.ndarray,
    obs_values: dict[str, jnp.ndarray],
    obs_scales: dict[str, jnp.ndarray],
    modifier_fn: Callable[[Any, jnp.ndarray, tuple[str, ...]], jnp.ndarray] = poly_nm,
    input_vars: tuple[str, ...] = INPUT_VARIABLES,
    species_index: int = 0,
):
    """Create a loss function for the 3PG model with a nutrition modifier."""
    n_obs = len(obs_indices)

    def loss_function(modifier_params):
        extended_params = ExtendedParams(modifier_params=modifier_params)
        _, pg3_outputs = run_3pg(
            input_data.initial_state,
            input_data.climate,
            input_data.params,
            input_data.site,
            input_data.species,
            input_data.deposition,
            extended_params,
            modifier_fn,
            input_vars,
        )

        variable_weights = {
            "BA": 1.0,
            "DBH": 1.0,
            "Height": 1.0,
            "WF": 1.0,
            "WS": 1.0,
            "WR": 1.0,
        }
        total_squared_error = jnp.asarray(0.0, dtype=jnp.float32)
        for var_name in target_vars:
            pg3_predictions = pg3_outputs[var_name][jnp.asarray(obs_indices)]
            if pg3_predictions.ndim == 2:
                pg3_predictions = pg3_predictions[:, species_index].reshape(-1)

            observed_values = obs_values[var_name].reshape(-1)
            scale = obs_scales[var_name]
            mask = ~(jnp.isnan(observed_values) | jnp.isnan(pg3_predictions))
            residuals = (pg3_predictions - observed_values) / scale
            squared = jnp.where(mask, residuals**2, 0.0)
            weight = variable_weights.get(var_name, 1.0)
            total_squared_error += weight * jnp.sum(squared)

        return total_squared_error / jnp.asarray(n_obs, dtype=jnp.float32)

    return loss_function


def build_optimizer(config: NutritionModifierConfig) -> optax.GradientTransformation:
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


def build_observation_indices(observed_data: pl.DataFrame, site_data: SiteData) -> jnp.ndarray:
    """Build observation month indices relative to the simulation start month."""
    if not {"year", "month"}.issubset(observed_data.columns):
        raise ValueError("Observed sheet must contain year and month columns")

    start_year = int(np.asarray(site_data.year_i).reshape(-1)[0])
    start_month = int(np.asarray(site_data.month_i).reshape(-1)[0])

    year = observed_data["year"].cast(pl.Int32).to_numpy()
    month = observed_data["month"].cast(pl.Int32).to_numpy()
    idx_values = (year - start_year) * 12 + (month - start_month)
    return jnp.asarray(idx_values, dtype=jnp.int32)


def build_observation_data(
    file_path: str,
    site_data: SiteData,
    target_vars: list[str],
    standardize_targets: bool = True,
):
    """Build observed values and scales for the target variables."""
    observed_data = pl.read_excel(file_path, sheet_name="observed")
    obs_indices = build_observation_indices(observed_data, site_data)
    obs_values: dict[str, jnp.ndarray] = {}
    obs_scales: dict[str, jnp.ndarray] = {}
    for var_name in target_vars:
        if var_name not in observed_data.columns:
            raise KeyError(f"Target variable '{var_name}' is not in observed data sheet")
        observed_np = observed_data[var_name].cast(pl.Float64).to_numpy()
        obs_values[var_name] = jnp.asarray(observed_np, dtype=jnp.float32)

        scale = float(np.nanstd(observed_np)) if standardize_targets else 1.0
        obs_scales[var_name] = jnp.asarray(max(scale, 1e-6), dtype=jnp.float32)

    return obs_indices, obs_values, obs_scales


def train_nutrition_modifier(
    config: NutritionModifierConfig,
    initial_modifier_params: Any,
    modifier_fn: Callable[[Any, jnp.ndarray, tuple[str, ...]], jnp.ndarray] = poly_nm,
) -> NutritionModifierFitResult:
    """Train the nutrition modifier using gradient descent."""
    input_data = prepare_data(config.file_path)
    obs_indices, obs_values, obs_scales = build_observation_data(
        config.file_path,
        input_data.site,
        config.target_vars,
        standardize_targets=config.standardize_targets,
    )

    modifier_params = jax.tree_util.tree_map(
        lambda leaf: jnp.asarray(leaf, dtype=jnp.float32), initial_modifier_params
    )
    # Create the loss function
    loss_function = make_loss_function(
        input_data=input_data,
        target_vars=config.target_vars,
        obs_indices=obs_indices,
        obs_values=obs_values,
        obs_scales=obs_scales,
        modifier_fn=modifier_fn,
        input_vars=config.input_vars,
        species_index=config.species_index,
    )

    # Create an optimizer
    optimizer = build_optimizer(config)
    opt_state = optimizer.init(modifier_params)

    @jax.jit
    def update(params, opt_state):
        loss, grads = jax.value_and_grad(loss_function)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    loss_history: list[float] = []
    param_history: list[Any] = []
    for epoch in range(config.num_epochs):
        modifier_params, opt_state, loss = update(modifier_params, opt_state)
        loss_history.append(float(loss))
        param_history.append(jax.tree_util.tree_map(np.asarray, modifier_params))
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")

    print(f"Final Loss: {loss_history[-1]}", f"Final modifier params: {modifier_params}")
    return NutritionModifierFitResult(
        fitted_modifier_params=modifier_params,
        loss_history=loss_history,
        param_history=param_history,
    )


def build_predicted_series(
    config: NutritionModifierConfig,
    fitted_modifier_params: Any,
    modifier_fn: Callable[[Any, jnp.ndarray, tuple[str, ...]], jnp.ndarray] = poly_nm,
) -> dict[str, np.ndarray]:
    """Simulate 3PG with and without the fitted nutrition modifier, for plotting.

    Returns
    -------
    dict[str, np.ndarray]
        `"dates"` (one per simulated month) plus `"pred_fitted_<var>"` and
        `"pred_default_<var>"` for each of `config.target_vars`.
    """
    input_data = prepare_data(config.file_path)
    extended_params = ExtendedParams(modifier_params=fitted_modifier_params)

    _, outputs_fitted = run_3pg(
        input_data.initial_state,
        input_data.climate,
        input_data.params,
        input_data.site,
        input_data.species,
        input_data.deposition,
        extended_params,
        modifier_fn,
        config.input_vars,
    )
    _, outputs_default = run_3pg(
        input_data.initial_state,
        input_data.climate,
        input_data.params,
        input_data.site,
        input_data.species,
    )

    series: dict[str, np.ndarray] = {}
    for var_name in config.target_vars:
        for label, outputs in (("fitted", outputs_fitted), ("default", outputs_default)):
            predictions = outputs[var_name]
            if predictions.ndim == 2:
                predictions = predictions[:, config.species_index]
            series[f"pred_{label}_{var_name}"] = np.asarray(predictions)

    n_months = len(series[f"pred_fitted_{config.target_vars[0]}"])
    start_year = int(np.asarray(input_data.site.year_i).reshape(-1)[0])
    start_month = int(np.asarray(input_data.site.month_i).reshape(-1)[0])
    series["dates"] = _months_to_dates(start_year, start_month, n_months)
    return series


def _months_to_dates(start_year: int, start_month: int, n_months: int) -> np.ndarray:
    """Build a monthly `datetime64` axis starting at (`start_year`, `start_month`)."""
    month_index = start_year * 12 + (start_month - 1) + np.arange(n_months)
    years, months = np.divmod(month_index, 12)
    return np.array(
        [
            np.datetime64(f"{year}-{month + 1:02d}", "M")
            for year, month in zip(years, months, strict=True)
        ]
    )


def plot_loss_over_iterations(
    loss_history: list[float], save_path: str | None = None, show: bool = True
) -> None:
    """Plot training loss over epochs."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(np.arange(len(loss_history)), loss_history, color="tab:blue", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Nutrition Modifier Loss Trajectory")
    ax.grid(alpha=0.3)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()


def _named_traces(param_history: list[Any]) -> list[tuple[str, np.ndarray]]:
    """Flatten a pytree parameter history into (label, values-over-epochs) traces."""
    leaves_by_step = [jax.tree_util.tree_flatten_with_path(step)[0] for step in param_history]
    n_leaves = len(leaves_by_step[0])

    traces: list[tuple[str, np.ndarray]] = []
    for leaf_idx in range(n_leaves):
        path = leaves_by_step[0][leaf_idx][0]
        base_name = jax.tree_util.keystr(path).lstrip(".") or "poly_params"
        values = np.stack([np.asarray(step[leaf_idx][1]) for step in leaves_by_step])
        leaf_shape = values.shape[1:]
        flat_values = values.reshape(values.shape[0], -1)
        for flat_idx in range(flat_values.shape[1]):
            if flat_values.shape[1] == 1:
                label = base_name
            else:
                index = [int(i) for i in np.unravel_index(flat_idx, leaf_shape)]
                label = f"{base_name}{index}"
            traces.append((label, flat_values[:, flat_idx]))

    return traces


def plot_param_history_over_iterations(
    param_history: list[Any], save_path: str | None = None, show: bool = True
) -> None:
    """Plot each nutrition-modifier parameter's value over training epochs."""
    if not param_history:
        return

    traces = _named_traces(param_history)
    n_traces = len(traces)

    n_cols = min(3, n_traces)
    n_rows = int(np.ceil(n_traces / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows), layout="constrained"
    )
    axes_list = np.ravel(np.atleast_1d(axes)).tolist()

    iterations = np.arange(len(param_history))
    for ax, (label, values) in zip(axes_list[:n_traces], traces, strict=True):
        ax.plot(iterations, values, color="tab:blue", linewidth=2)
        ax.set_title(label)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
        ax.grid(alpha=0.3)

    for ax in axes_list[n_traces:]:
        ax.set_visible(False)

    fig.suptitle("Nutrition Modifier Parameter Trajectories")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()


def plot_observed_vs_predicted(
    config: NutritionModifierConfig,
    predicted_series: dict[str, np.ndarray],
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """Plot observed data against predictions with and without the fitted nutrition modifier."""
    observed_data = pl.read_excel(config.file_path, sheet_name=config.observed_sheet)
    observed_years = observed_data["year"].cast(pl.Int32).to_numpy()
    observed_months = observed_data["month"].cast(pl.Int32).to_numpy()
    observed_dates = np.array(
        [
            np.datetime64(f"{year}-{month:02d}", "M")
            for year, month in zip(observed_years, observed_months, strict=True)
        ]
    )

    target_vars = config.target_vars
    n_cols = min(3, len(target_vars))
    n_rows = int(np.ceil(len(target_vars) / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), layout="constrained"
    )
    axes_list = np.ravel(np.atleast_1d(axes)).tolist()

    dates = predicted_series["dates"]
    for ax, var_name in zip(axes_list, target_vars, strict=False):
        ax.plot(
            dates,
            predicted_series[f"pred_fitted_{var_name}"],
            label="With nutrition modifier",
            color="tab:blue",
            linewidth=2,
        )
        ax.plot(
            dates,
            predicted_series[f"pred_default_{var_name}"],
            label="Without nutrition modifier",
            color="tab:green",
            linewidth=2,
            linestyle="--",
        )

        if var_name in observed_data.columns:
            observed_values = observed_data[var_name].cast(pl.Float64).to_numpy()
            mask = ~np.isnan(observed_values)
            ax.scatter(
                observed_dates[mask],
                observed_values[mask],
                label="Observed",
                color="tab:red",
                s=50,
                zorder=5,
            )

        locator = mdates.AutoDateLocator()
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        ax.set_ylabel(_METRIC_LABELS.get(var_name, var_name))
        ax.set_title(var_name)
        ax.grid(alpha=0.3)
        ax.legend(loc="best")

    for ax in axes_list[len(target_vars) :]:
        ax.set_visible(False)

    fig.suptitle("Observed vs Predicted, With/Without Nutrition Modifier")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()


if __name__ == "__main__":
    # file_path = os.path.join(threepg_data_folder, "S_weather_data.xlsx")
    plot_id = "04.1402"
    literature_source = "Forrester"
    file_path = prepare_plot_input(plot_id, literature_source=literature_source)

    # Which of ("N", "S", "T_avg") to build the modifier over
    input_vars = ("N", "S")
    config = NutritionModifierConfig(
        file_path=file_path,
        target_vars=["BA", "DBH", "Height", "WS", "WF", "WR"],
        input_vars=input_vars,
        optimizer_name="adam",
        learning_rate=1e-3,
        num_epochs=5000,
    )

    initial_modifier_params = init_modifier_params(input_vars)
    fit_result = train_nutrition_modifier(config, initial_modifier_params=initial_modifier_params)

    image_dir = Path(config.image_dir)
    plot_loss_over_iterations(
        fit_result.loss_history,
        save_path=str(image_dir / "loss.png"),
        show=False,
    )
    plot_param_history_over_iterations(
        fit_result.param_history,
        save_path=str(image_dir / "param_history.png"),
        show=False,
    )
    predicted_series = build_predicted_series(config, fit_result.fitted_modifier_params)
    plot_observed_vs_predicted(
        config,
        predicted_series,
        save_path=str(image_dir / "observed_vs_predicted.png"),
        show=False,
    )

    plt.show()
