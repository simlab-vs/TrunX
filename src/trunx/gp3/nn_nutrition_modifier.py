"""Learnable componenet of the 3PG model: a nutrition modifier optimized using gradient descent."""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import optax
import polars as pl

from trunx.config import images_folder
from trunx.gp3.bayesiancalibrations.bayesian_config import species_plot_ids
from trunx.gp3.bayesiancalibrations.pymc_icp_plots import prepare_plot_input
from trunx.gp3.extended_helper import INPUT_VARIABLES, init_mlp_modifier_params, mlp_nm, poly_nm
from trunx.gp3.gradient_descent import apply_fitted_params, load_param_bounds
from trunx.gp3.model_inputs import ExtendedParams, InputData, SiteData
from trunx.gp3.prepare_data import prepare_data
from trunx.gp3.run_3pg import run_3pg
from trunx.gp3.training_utils import (
    build_observation_indices,
    build_optimizer,
    plot_loss_over_iterations,
    plot_traces_grid,
    weighted_squared_error,
)

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

    file_paths: list[str]  # Input data files, one per plot, calibrated jointly
    target_vars: list[str]  # List of target variables to optimize against

    observed_sheet: str = "observed"  # Sheet name for observed data
    fit_phys_params: list[str] | None = None
    param_bounds_sheet: str = "param_bound"
    species_index: int = 0  # Index of the species to optimize for
    # Which of ("N", "S", "T_avg") the modifier is built over
    input_vars: tuple[str, ...] = INPUT_VARIABLES
    optimizer_name: str = "adam"  # Optimizer name: 'adam' or 'sgd'
    learning_rate: float = 1e-3  # Learning rate for the optimizer
    global_clip_norm: float = 1.0  # Global norm for gradient clipping
    num_epochs: int = 1000  # Number of training epochs
    standardize_targets: bool = True  # Whether to standardize target variables
    image_dir: str = field(default_factory=lambda: str(images_folder / "nn_nutrition_modifier"))
    modifier_fn: Callable[[Any, jnp.ndarray, tuple[str, ...]], jnp.ndarray] = poly_nm


@dataclass
class NutritionModifierFitResult:
    """Container for nutrition modifier training results."""

    fitted_modifier_params: Any
    fitted_phys_params: dict[str, float]
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
) -> Callable[[dict[str, Any]], jnp.ndarray]:
    """Create a loss jointly over physiological and nutrition-modifier parameters.

    The returned loss takes a pytree `{"phys_params": {name: value}, "modifier_params": ...}`;
    `phys_params` overrides the matching `input_data.params` fields for `species_index`.
    """
    n_obs = len(obs_indices)
    variable_weights = {
        "BA": 1.0,
        "DBH": 1.0,
        "Height": 1.0,
        "WF": 1.0,
        "WS": 1.0,
        "WR": 1.0,
    }

    def loss_function(trainable: dict[str, Any]) -> jnp.ndarray:
        phys_params = trainable["phys_params"]
        params = apply_fitted_params(
            input_data.params, list(phys_params), phys_params, species_index
        )
        extended_params = ExtendedParams(modifier_params=trainable["modifier_params"])
        _, pg3_outputs = run_3pg(
            input_data.initial_state,
            input_data.climate,
            params,
            input_data.site,
            input_data.species,
            input_data.deposition,
            extended_params,
            modifier_fn,
            input_vars,
        )
        total_squared_error = weighted_squared_error(
            pg3_outputs,
            target_vars,
            obs_indices,
            obs_values,
            obs_scales,
            species_index,
            variable_weights,
        )
        return total_squared_error / jnp.asarray(n_obs, dtype=jnp.float32)

    return loss_function


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
) -> NutritionModifierFitResult:
    """Jointly train `config.fit_phys_params` and the nutrition modifier over all plots.

    The loss is the mean of the per-plot losses, so every plot weighs equally; the
    physiological parameters start from the first plot's values.
    `initial_modifier_params` must match `config.modifier_fn`'s own parameter pytree —
    e.g. `init_modifier_params` for `poly_nm`, `init_mlp_modifier_params` for `mlp_nm` —
    since they aren't interchangeable.
    """
    plot_losses = []
    for file_path in config.file_paths:
        input_data = prepare_data(file_path)
        obs_indices, obs_values, obs_scales = build_observation_data(
            file_path,
            input_data.site,
            config.target_vars,
            standardize_targets=config.standardize_targets,
        )
        plot_losses.append(
            make_loss_function(
                input_data=input_data,
                target_vars=config.target_vars,
                obs_indices=obs_indices,
                obs_values=obs_values,
                obs_scales=obs_scales,
                modifier_fn=config.modifier_fn,
                input_vars=config.input_vars,
                species_index=config.species_index,
            )
        )

    def loss_function(trainable: dict[str, Any]) -> jnp.ndarray:
        """Mean of the per-plot losses."""
        return jnp.mean(jnp.stack([plot_loss(trainable) for plot_loss in plot_losses]))

    first_file_path = config.file_paths[0]
    first_params = prepare_data(first_file_path).params
    sheet_bounds = {
        name: bounds
        for name, bounds in load_param_bounds(first_file_path, config.param_bounds_sheet).items()
        if not np.isnan(bounds).any()
    }
    fit_phys_params = (
        list(sheet_bounds) if config.fit_phys_params is None else config.fit_phys_params
    )
    missing = [name for name in fit_phys_params if name not in sheet_bounds]
    if missing:
        raise ValueError(f"No min/max in '{config.param_bounds_sheet}' for: {missing}")
    lower = {name: sheet_bounds[name][0] for name in fit_phys_params}
    upper = {name: sheet_bounds[name][1] for name in fit_phys_params}

    def clip_to_bounds(phys_params: dict[str, jnp.ndarray]) -> dict[str, jnp.ndarray]:
        """Clip each physiological parameter to its [min, max] bounds."""
        return {
            name: jnp.clip(value, lower[name], upper[name]) for name, value in phys_params.items()
        }

    phys_params = {
        name: jnp.atleast_1d(jnp.asarray(getattr(first_params, name)))[config.species_index]
        for name in fit_phys_params
    }
    trainable = jax.tree_util.tree_map(
        lambda leaf: jnp.asarray(leaf, dtype=jnp.float32),
        {"phys_params": clip_to_bounds(phys_params), "modifier_params": initial_modifier_params},
    )
    # Create an optimizer
    optimizer = build_optimizer(
        config.optimizer_name, config.learning_rate, config.global_clip_norm
    )
    opt_state = optimizer.init(trainable)

    @jax.jit
    def update(params, opt_state):
        loss, grads = jax.value_and_grad(loss_function)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = cast(dict[str, Any], optax.apply_updates(params, updates))
        params["phys_params"] = clip_to_bounds(params["phys_params"])
        return params, opt_state, loss

    loss_history: list[float] = []
    param_history: list[Any] = []
    for epoch in range(config.num_epochs):
        trainable, opt_state, loss = update(trainable, opt_state)
        loss_history.append(float(loss))
        param_history.append(jax.tree_util.tree_map(np.asarray, trainable))
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")

    fitted_phys_params = {name: float(v) for name, v in trainable["phys_params"].items()}
    print(f"Final Loss: {loss_history[-1]}", f"Final phys params: {fitted_phys_params}")
    return NutritionModifierFitResult(
        fitted_modifier_params=trainable["modifier_params"],
        fitted_phys_params=fitted_phys_params,
        loss_history=loss_history,
        param_history=param_history,
    )


def build_predicted_series(
    config: NutritionModifierConfig,
    file_path: str,
    fitted_modifier_params: Any,
    fitted_phys_params: dict[str, float],
) -> dict[str, np.ndarray]:
    """Simulate one plot (`file_path`) with fitted and default parameters, for plotting.

    Uses `config.modifier_fn` — must match whatever `fitted_modifier_params`
    was fitted with (see `train_nutrition_modifier`).

    Returns
    -------
    dict[str, np.ndarray]
        `"dates"` (one per simulated month) plus `"pred_fitted_<var>"` and
        `"pred_default_<var>"` for each of `config.target_vars`.
    """
    input_data = prepare_data(file_path)
    extended_params = ExtendedParams(modifier_params=fitted_modifier_params)
    fitted_params = apply_fitted_params(
        input_data.params, list(fitted_phys_params), fitted_phys_params, config.species_index
    )

    _, outputs_fitted = run_3pg(
        input_data.initial_state,
        input_data.climate,
        fitted_params,
        input_data.site,
        input_data.species,
        input_data.deposition,
        extended_params,
        config.modifier_fn,
        config.input_vars,
    )
    _, outputs_default = run_3pg(
        input_data.initial_state,
        input_data.climate,
        fitted_params,
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


def compute_rmse(
    config: NutritionModifierConfig,
    file_path: str,
    predicted_series: dict[str, np.ndarray],
) -> pl.DataFrame:
    """RMSE per target variable of one plot, with and without the fitted nutrition modifier.

    Returns
    -------
    pl.DataFrame
        One row per target variable: `plot`, `variable`, `n_obs`, `rmse_with_nm`
        and `rmse_without_nm`.
    """
    observed_data = pl.read_excel(file_path, sheet_name=config.observed_sheet)
    observed_dates = np.array(
        [
            np.datetime64(f"{year}-{month:02d}", "M")
            for year, month in zip(observed_data["year"], observed_data["month"], strict=True)
        ]
    )
    obs_indices = (observed_dates - predicted_series["dates"][0]).astype(int)

    rows = []
    for var_name in config.target_vars:
        observed_values = observed_data[var_name].cast(pl.Float64).to_numpy()
        mask = ~np.isnan(observed_values)
        row: dict[str, Any] = {
            "plot": Path(file_path).stem,
            "variable": var_name,
            "n_obs": int(mask.sum()),
        }
        for label, column in (("with_nm", "fitted"), ("without_nm", "default")):
            predictions = predicted_series[f"pred_{column}_{var_name}"][obs_indices[mask]]
            row[f"rmse_{label}"] = float(
                np.sqrt(np.mean((predictions - observed_values[mask]) ** 2))
            )
        rows.append(row)
    return pl.DataFrame(rows)


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

    plot_traces_grid(
        _named_traces(param_history),
        suptitle="Nutrition Modifier Parameter Trajectories",
        xlabel="Epoch",
        save_path=save_path,
        show=show,
    )


def plot_observed_vs_predicted(
    config: NutritionModifierConfig,
    file_path: str,
    predicted_series: dict[str, np.ndarray],
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """Plot one plot's observed data against predictions with and without the fitted modifier."""
    observed_data = pl.read_excel(file_path, sheet_name=config.observed_sheet)
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

    fig.suptitle(f"Observed vs Predicted, With/Without Nutrition Modifier: {Path(file_path).stem}")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()


if __name__ == "__main__":
    plot_ids = species_plot_ids["Picea abies"]
    literature_source = "Trotsiuk"
    file_paths = [
        prepare_plot_input(plot_id, literature_source=literature_source) for plot_id in plot_ids
    ]

    # Which of ("N", "S", "T_avg") to build the modifier over
    input_vars = ("N", "S")

    # Select which learnable nutrition-modifier function to fit.
    modifier_fn = poly_nm
    if modifier_fn is poly_nm:
        initial_modifier_params = init_modifier_params(input_vars)
    elif modifier_fn is mlp_nm:
        initial_modifier_params = init_mlp_modifier_params(
            jax.random.PRNGKey(0), input_vars, hidden_sizes=(3, 2)
        )
    else:
        raise ValueError(f"No initializer wired up for modifier_fn={modifier_fn!r}")

    config = NutritionModifierConfig(
        file_paths=file_paths,
        target_vars=["BA", "DBH", "Height", "WS", "WF", "WR"],
        input_vars=input_vars,
        optimizer_name="adam",
        learning_rate=1e-3,
        num_epochs=2000,
        modifier_fn=modifier_fn,
    )

    fit_result = train_nutrition_modifier(config, initial_modifier_params=initial_modifier_params)

    image_dir = Path(config.image_dir)
    plot_loss_over_iterations(
        fit_result.loss_history,
        title="Nutrition Modifier Loss Trajectory",
        xlabel="Epoch",
        save_path=str(image_dir / "loss.png"),
        show=False,
    )
    plot_param_history_over_iterations(
        fit_result.param_history,
        save_path=str(image_dir / "param_history.png"),
        show=False,
    )
    rmse_tables = []
    for plot_id, file_path in zip(plot_ids, file_paths, strict=True):
        predicted_series = build_predicted_series(
            config, file_path, fit_result.fitted_modifier_params, fit_result.fitted_phys_params
        )
        rmse_tables.append(compute_rmse(config, file_path, predicted_series))
        plot_observed_vs_predicted(
            config,
            file_path,
            predicted_series,
            save_path=str(image_dir / f"observed_vs_predicted_{plot_id}.png"),
            show=False,
        )

    with pl.Config(tbl_rows=-1, float_precision=3):
        print(pl.concat(rmse_tables))

    plt.show()
