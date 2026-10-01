"""Learnable componenet of the 3PG model: a nutrition modifier optimized using gradient descent."""

from collections.abc import Callable
from dataclasses import dataclass, field, replace
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
    count_observed_rows,
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
    "alpha_c": "alpha_c (mol C mol⁻¹ PAR)",
    "f_nutri_classic_learnable": "fN × learned modifier (–)",
}
# 40 per-plot colors (enough for the largest species group, 21 plots): tab20's dark
# shades first, then its light ones, so neighbouring plots never get the same hue
_PLOT_COLORS = [
    plt.colormaps[name](i)
    for name, indices in (
        ("tab20", range(0, 20, 2)),
        ("tab20", range(1, 20, 2)),
        ("tab20b", range(20)),
    )
    for i in indices
]


@dataclass
class NutritionModifierConfig:
    """Configuration for the nutrition modifier."""

    file_paths: list[str]  # Input data files, one per plot, calibrated jointly
    target_vars: list[str]  # List of target variables to optimize against

    # Variables plotted against observations; defaults to `target_vars`
    plot_variables: list[str] | None = None
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
    n_obs = count_observed_rows(obs_values, target_vars)
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
    """Build observed values and scales for the target variables.

    Only the target columns are kept, and rows without any target observation (e.g.
    GPP-only months) are dropped.
    """
    observed_data = pl.read_excel(file_path, sheet_name="observed")
    missing = [var for var in target_vars if var not in observed_data.columns]
    if missing:
        raise KeyError(f"Target variables {missing} are not in observed data sheet")
    observed_data = observed_data.select("year", "month", *target_vars).filter(
        pl.any_horizontal(pl.col(target_vars).is_not_null())
    )
    obs_indices = build_observation_indices(observed_data, site_data)
    obs_values: dict[str, jnp.ndarray] = {}
    obs_scales: dict[str, jnp.ndarray] = {}
    for var_name in target_vars:
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


def _run_labels(config: NutritionModifierConfig) -> dict[str, str]:
    """Legend label of each distinct run of `build_predicted_series`, keyed by run name.

    Without `config.fit_phys_params` the `"phys"` run uses the default parameters, so it
    is the same as `"default"`, which is then left out.
    """
    phys = "Fitted" if config.fit_phys_params else "Default"
    labels = {
        "fitted": f"{phys} physiological parameters, with nutrition modifier",
        "phys": f"{phys} physiological parameters, without nutrition modifier",
    }
    if config.fit_phys_params:
        labels["default"] = "Default physiological parameters, without nutrition modifier"
    return labels


def _plot_variables(config: NutritionModifierConfig) -> list[str]:
    """Variables plotted against observations: `config.plot_variables`, else the targets."""
    return config.plot_variables or config.target_vars


_RUN_STYLES: dict[str, dict[str, str]] = {
    "fitted": {"color": "tab:blue", "linestyle": "-"},
    "phys": {"color": "tab:orange", "linestyle": "-."},
    "default": {"color": "tab:green", "linestyle": "--"},
}


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
        `"dates"` (one per simulated month) plus, for each of `config.target_vars` and
        the plot variables, `"pred_fitted_<var>"` (fitted phys params + modifier),
        `"pred_phys_<var>"` (fitted phys params only) and `"pred_default_<var>"`
        (default params).
        `alpha_c` and `f_nutri_classic_learnable` are always included, to show the
        modifier's effect on them, plus `"learned_modifier"` (the modifier's monthly value).
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
    _, outputs_phys = run_3pg(
        input_data.initial_state,
        input_data.climate,
        fitted_params,
        input_data.site,
        input_data.species,
    )
    _, outputs_default = run_3pg(
        input_data.initial_state,
        input_data.climate,
        input_data.params,
        input_data.site,
        input_data.species,
    )

    series: dict[str, np.ndarray] = {}
    for var_name in dict.fromkeys(
        [*config.target_vars, *_plot_variables(config), "alpha_c", "f_nutri_classic_learnable"]
    ):
        for label, outputs in (
            ("fitted", outputs_fitted),
            ("phys", outputs_phys),
            ("default", outputs_default),
        ):
            predictions = outputs[var_name]
            if predictions.ndim == 2:
                predictions = predictions[:, config.species_index]
            series[f"pred_{label}_{var_name}"] = np.asarray(predictions)
    # fN is the same in both runs, so this ratio is exactly the learned modifier
    series["learned_modifier"] = (
        series["pred_fitted_f_nutri_classic_learnable"]
        / series["pred_phys_f_nutri_classic_learnable"]
    )

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
    """RMSE per plot variable of one plot, for each series of `build_predicted_series`.

    Returns
    -------
    pl.DataFrame
        One row per plot variable (see `_plot_variables`): `plot`, `variable`, `target`
        (whether it is one of `config.target_vars`), `n_obs`, and `rmse_<run>` for each
        distinct run of `_run_labels`: `rmse_fitted` (with the modifier), `rmse_phys`
        (without it) and, when phys params are fitted, `rmse_default`.
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
    for var_name in _plot_variables(config):
        observed_values = observed_data[var_name].cast(pl.Float64).to_numpy()
        mask = ~np.isnan(observed_values)
        row: dict[str, Any] = {
            "plot": Path(file_path).stem,
            "variable": var_name,
            "target": var_name in config.target_vars,
            "n_obs": int(mask.sum()),
        }
        for run in _run_labels(config):
            predictions = predicted_series[f"pred_{run}_{var_name}"][obs_indices[mask]]
            row[f"rmse_{run}"] = float(
                np.sqrt(np.mean((predictions - observed_values[mask]) ** 2))
            )
        rows.append(row)
    return pl.DataFrame(rows)


def summarize_rmse(rmse_table: pl.DataFrame) -> pl.DataFrame:
    """Mean RMSE per split and variable with and without the modifier, and its improvement.

    Parameters
    ----------
    rmse_table : pl.DataFrame
        Concatenated `compute_rmse` tables with an added `split` column (e.g. "train",
        "test").

    Returns
    -------
    pl.DataFrame
        One row per split and variable: `plots`, mean `rmse_fitted` (with the modifier),
        mean `rmse_phys` (without it) and `improvement %` (relative RMSE reduction by the
        modifier; negative if it got worse).
    """
    return (
        rmse_table.group_by("split", "variable", "target", maintain_order=True)
        .agg(pl.len().alias("plots"), pl.col("rmse_fitted").mean(), pl.col("rmse_phys").mean())
        .with_columns(
            (100 * (pl.col("rmse_phys") - pl.col("rmse_fitted")) / pl.col("rmse_phys"))
            .round(1)
            .alias("improvement %")
        )
    )


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


def _format_date_axis(ax: Any) -> None:
    """Apply concise, rotated date ticks and a light grid to `ax`."""
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.tick_params(axis="x", labelrotation=45)
    ax.grid(alpha=0.3)


def _load_deposition(file_paths: list[str]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Monthly N and S deposition of each plot, in `file_paths` order."""
    n_values, s_values = [], []
    for file_path in file_paths:
        deposition = prepare_data(file_path).deposition
        if deposition is None:
            raise ValueError(f"No deposition data in {file_path}")
        n_values.append(np.asarray(deposition.dep_n_tot))
        s_values.append(np.asarray(deposition.dep_s_so4))
    return n_values, s_values


def _deposition_range(values: list[np.ndarray]) -> tuple[float, float]:
    """1st-99th percentile of the pooled `values`, padded by 20% and floored at 0."""
    low, high = np.nanpercentile(np.concatenate(values), [1, 99])
    pad = 0.2 * (high - low)
    return max(float(low - pad), 0.0), float(high + pad)


def plot_deposition_over_time(
    config: NutritionModifierConfig,
    predicted_series: list[dict[str, np.ndarray]],
    plot_ids: list[str],
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """Plot each plot's monthly N and S deposition over its simulation period."""
    n_values, s_values = _load_deposition(config.file_paths)
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), layout="constrained", sharex=True)

    for ax, values, label in (
        (axes[0], n_values, "N deposition (kg ha⁻¹ month⁻¹)"),
        (axes[1], s_values, "S deposition (kg ha⁻¹ month⁻¹)"),
    ):
        for idx, (plot_id, deposition, series) in enumerate(
            zip(plot_ids, values, predicted_series, strict=True)
        ):
            ax.plot(
                series["dates"],
                deposition,
                color=_PLOT_COLORS[idx % len(_PLOT_COLORS)],
                linewidth=0.8,
                label=plot_id,
            )
        ax.set_ylabel(label)
        _format_date_axis(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside right upper", fontsize=7, title="Plot")
    fig.suptitle("Monthly deposition per plot")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()


def plot_modifier_response_surface(
    config: NutritionModifierConfig,
    plot_ids: list[str],
    fitted_modifier_params: Any,
    n_grid: int = 100,
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """Plot the fitted modifier over an (N, S) deposition grid, with each plot's monthly inputs.

    The grid covers the 1st-99th percentile of the pooled deposition values, padded
    by 20% to show how the modifier behaves just beyond the data; the few extreme
    months outside it are not drawn.
    """
    if set(config.input_vars) != {"N", "S"}:
        raise ValueError(f"Needs a modifier over exactly N and S, got {config.input_vars}")

    n_values, s_values = _load_deposition(config.file_paths)
    n_grid_values, s_grid_values = np.meshgrid(
        np.linspace(*_deposition_range(n_values), n_grid),
        np.linspace(*_deposition_range(s_values), n_grid),
    )
    channels = {"N": n_grid_values, "S": s_grid_values}
    inputs = jnp.stack([channels[name] for name in config.input_vars], axis=-1)
    modifier = np.asarray(config.modifier_fn(fitted_modifier_params, inputs, config.input_vars))

    # Diverging colors centered on 1 (no effect), symmetric around it
    spread = max(float(np.abs(modifier - 1.0).max()), 1e-6)
    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    surface = ax.contourf(
        n_grid_values,
        s_grid_values,
        modifier,
        levels=20,
        cmap="PuOr",
        vmin=1.0 - spread,
        vmax=1.0 + spread,
    )
    contours = ax.contour(
        n_grid_values, s_grid_values, modifier, levels=10, colors="black", linewidths=0.4
    )
    ax.clabel(contours, fontsize=7)
    # The "no effect" line only exists if the modifier crosses 1 on the grid
    if modifier.min() < 1.0 < modifier.max():
        ax.contour(
            n_grid_values, s_grid_values, modifier, levels=[1.0], colors="black", linewidths=1.5
        )
    fig.colorbar(surface, ax=ax, label="Nutrition modifier (1 = no effect)")

    for idx, (plot_id, n, s) in enumerate(zip(plot_ids, n_values, s_values, strict=True)):
        ax.scatter(
            n,
            s,
            s=12,
            color=_PLOT_COLORS[idx % len(_PLOT_COLORS)],
            edgecolor="black",
            linewidth=0.3,
            label=plot_id,
        )
    ax.set_xlim(n_grid_values.min(), n_grid_values.max())
    ax.set_ylim(s_grid_values.min(), s_grid_values.max())

    ax.set_xlabel("N deposition (kg ha⁻¹ month⁻¹)")
    ax.set_ylabel("S deposition (kg ha⁻¹ month⁻¹)")
    ax.set_title("Fitted nutrition modifier response surface")
    fig.legend(loc="outside right upper", fontsize=7, title="Plot")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()


def _binned_median(
    x: np.ndarray, y: np.ndarray, n_bins: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """Median of `x` and `y` within `n_bins` equal-count bins of `x`, for a trend line."""
    order = np.argsort(x)
    bins = [b for b in np.array_split(order, min(n_bins, len(x))) if len(b)]
    return np.array([np.median(x[b]) for b in bins]), np.array([np.median(y[b]) for b in bins])


def _scatter_by_plot(
    ax: Any, plot_ids: list[str], x_values: list[np.ndarray], y_values: list[np.ndarray]
) -> None:
    """Scatter each plot's (x, y) in its own color, with a line through binned medians.

    A dashed black line shows the binned medians of all plots pooled together.
    """
    for idx, (plot_id, x, y) in enumerate(zip(plot_ids, x_values, y_values, strict=True)):
        color = _PLOT_COLORS[idx % len(_PLOT_COLORS)]
        ax.scatter(x, y, s=8, alpha=0.4, color=color, label=plot_id)
        ax.plot(*_binned_median(x, y), color=color, linewidth=1.8)
    ax.plot(
        *_binned_median(np.concatenate(x_values), np.concatenate(y_values), n_bins=20),
        color="black",
        linewidth=2.0,
        linestyle="--",
        label="All plots",
    )
    ax.grid(alpha=0.3)


def plot_learned_modifier_vs_deposition(
    config: NutritionModifierConfig,
    predicted_series: list[dict[str, np.ndarray]],
    plot_ids: list[str],
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """Plot each plot's monthly f_learned and f_nutri_classic_learnable against N and S.

    Rows are the learned modifier (f_learned, 1 = no effect) and its product with the
    classic fN (f_nutri_classic_learnable), columns the deposition variables. One color
    per plot, with a per-plot line through binned medians.
    """
    n_values, s_values = _load_deposition(config.file_paths)
    fig, axes = plt.subplots(
        2, 2, figsize=(13, 9), layout="constrained", sharex="col", sharey=True
    )

    for row, (key, y_label) in enumerate(
        (
            ("learned_modifier", "f_learned (1 = no effect)"),
            ("pred_fitted_f_nutri_classic_learnable", _METRIC_LABELS["f_nutri_classic_learnable"]),
        )
    ):
        y_values = [series[key] for series in predicted_series]
        for col, (values, dep_label) in enumerate(
            (
                (n_values, "N deposition (kg ha⁻¹ month⁻¹)"),
                (s_values, "S deposition (kg ha⁻¹ month⁻¹)"),
            )
        ):
            ax = axes[row, col]
            _scatter_by_plot(ax, plot_ids, values, y_values)
            ax.axhline(1.0, color="black", linewidth=0.8, linestyle=":")
            ax.set_xlim(*_deposition_range(values))
            if row == 0:
                ax.set_title(dep_label.split(" (")[0])
            else:
                ax.set_xlabel(dep_label)
            if col == 0:
                ax.set_ylabel(y_label)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside right upper", fontsize=7, title="Plot")
    fig.suptitle(
        "Nutrition modifiers against deposition (solid: per-plot binned median, dashed: all plots)"
    )

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()


def plot_modifier_effect_vs_deposition(
    config: NutritionModifierConfig,
    predicted_series: list[dict[str, np.ndarray]],
    plot_ids: list[str],
    var_name: str = "alpha_c",
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """Plot each plot's monthly `var_name`, with and without the modifier, against N and S.

    Rows are the runs with and without the modifier, columns the deposition variables,
    all on a shared y-axis. One color per plot, with a per-plot line through binned
    medians. Months with alpha_c = 0 without the modifier (no photosynthesis, e.g.
    winter) are left out.
    """
    n_values, s_values = _load_deposition(config.file_paths)
    fig, axes = plt.subplots(
        2, 2, figsize=(13, 9), layout="constrained", sharex="col", sharey=True
    )

    run_labels = _run_labels(config)
    for row, (run_label, key) in enumerate(
        ((run_labels["fitted"], "fitted"), (run_labels["phys"], "phys"))
    ):
        for col, (values, dep_label) in enumerate(
            (
                (n_values, "N deposition (kg ha⁻¹ month⁻¹)"),
                (s_values, "S deposition (kg ha⁻¹ month⁻¹)"),
            )
        ):
            ax = axes[row, col]
            active = [series["pred_phys_alpha_c"] > 0 for series in predicted_series]
            _scatter_by_plot(
                ax,
                plot_ids,
                [dep[mask] for dep, mask in zip(values, active, strict=True)],
                [
                    series[f"pred_{key}_{var_name}"][mask]
                    for series, mask in zip(predicted_series, active, strict=True)
                ],
            )
            ax.set_xlim(*_deposition_range(values))
            if row == 0:
                ax.set_title(dep_label.split(" (")[0])
            else:
                ax.set_xlabel(dep_label)
            if col == 0:
                ax.set_ylabel(f"{run_label}\n{_METRIC_LABELS[var_name]}")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside right upper", fontsize=7, title="Plot")
    fig.suptitle(
        f"Monthly {var_name} against deposition (solid: per-plot binned median, dashed: all plots)"
    )

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()


def plot_modifier_effect_over_time(
    config: NutritionModifierConfig,
    plot_ids: list[str],
    predicted_series: list[dict[str, np.ndarray]],
    var_name: str = "alpha_c",
    n_cols: int = 3,
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """Plot each plot's monthly `var_name` with and without the modifier, and their ratio below.

    The ratio is fitted with / without modifier; it is undefined where the run without
    the modifier is 0.
    """
    run_labels = _run_labels(config)
    n_block_rows = int(np.ceil(len(plot_ids) / n_cols))
    fig = plt.figure(figsize=(5 * n_cols, 4 * n_block_rows), layout="constrained")
    grid = fig.add_gridspec(2 * n_block_rows, n_cols, height_ratios=[3, 1] * n_block_rows)

    for idx, (plot_id, series) in enumerate(zip(plot_ids, predicted_series, strict=True)):
        block_row, col = divmod(idx, n_cols)
        ax = fig.add_subplot(grid[2 * block_row, col])
        ratio_ax = fig.add_subplot(grid[2 * block_row + 1, col], sharex=ax)

        dates = series["dates"]
        fitted = series[f"pred_fitted_{var_name}"]
        phys = series[f"pred_phys_{var_name}"]
        for values, run in ((fitted, "fitted"), (phys, "phys")):
            ax.plot(
                dates,
                values,
                color=_RUN_STYLES[run]["color"],
                linewidth=1.0,
                label=run_labels[run],
            )

        ratio = np.divide(fitted, phys, out=np.full_like(fitted, np.nan), where=phys > 0)
        ratio_ax.plot(dates, ratio, color="black", linewidth=1.0, label="Ratio (modifier effect)")
        ratio_ax.axhline(1.0, color="black", linewidth=0.8, linestyle=":")

        ax.set_title(plot_id)
        ax.set_ylabel(_METRIC_LABELS[var_name])
        ax.tick_params(axis="x", labelbottom=False)
        ax.grid(alpha=0.3)
        ratio_ax.set_ylabel("ratio")
        _format_date_axis(ratio_ax)

    # fig.axes alternates value and ratio axes, one pair per plot
    handles, labels = fig.axes[0].get_legend_handles_labels()
    ratio_handles, ratio_labels = fig.axes[1].get_legend_handles_labels()
    fig.legend(
        handles + ratio_handles,
        labels + ratio_labels,
        loc="outside upper center",
        ncols=3,
    )

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()


def plot_observed_vs_predicted(
    config: NutritionModifierConfig,
    plot_ids: list[str],
    predicted_series: list[dict[str, np.ndarray]],
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """Plot every plot's observed data against its `build_predicted_series` in one figure.

    Rows are plots (`config.file_paths`, labelled by `plot_ids`), columns are
    the plot variables (`config.plot_variables`, else `config.target_vars`).
    """
    plot_variables = _plot_variables(config)
    run_labels = _run_labels(config)
    n_rows, n_cols = len(plot_ids), len(plot_variables)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.5 * n_cols, 2.5 * n_rows),
        layout="constrained",
        squeeze=False,
    )

    for row, (plot_id, file_path, series) in enumerate(
        zip(plot_ids, config.file_paths, predicted_series, strict=True)
    ):
        observed_data = pl.read_excel(file_path, sheet_name=config.observed_sheet)
        observed_dates = np.array(
            [
                np.datetime64(f"{year}-{month:02d}", "M")
                for year, month in zip(observed_data["year"], observed_data["month"], strict=True)
            ]
        )
        dates = series["dates"]
        for col, var_name in enumerate(plot_variables):
            ax = axes[row, col]
            for run, label in run_labels.items():
                ax.plot(
                    dates,
                    series[f"pred_{run}_{var_name}"],
                    label=label,
                    linewidth=1.5,
                    **_RUN_STYLES[run],
                )

            if var_name in observed_data.columns:
                observed_values = observed_data[var_name].cast(pl.Float64).to_numpy()
                mask = ~np.isnan(observed_values)
                is_target = var_name in config.target_vars
                ax.scatter(
                    observed_dates[mask],
                    observed_values[mask],
                    label="Observed (target variable)" if is_target else "Observed (not fitted)",
                    color="tab:red" if is_target else "tab:gray",
                    s=20,
                    zorder=5,
                )

            _format_date_axis(ax)
            if row == 0:
                ax.set_title(_METRIC_LABELS.get(var_name, var_name))
            if col == 0:
                ax.set_ylabel(plot_id)

    # Target and non-target observations sit in different columns, so gather the
    # legend entries of every column, keeping the first handle per label
    legend_entries: dict[str, Any] = {}
    for ax in axes[0]:
        for handle, label in zip(*ax.get_legend_handles_labels(), strict=True):
            legend_entries.setdefault(label, handle)
    fig.legend(
        list(legend_entries.values()),
        list(legend_entries),
        loc="outside upper center",
        ncols=2,
    )

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()


if __name__ == "__main__":
    # Train on these plots; the remaining plots of the species are held out for testing
    plot_ids = ["04.1402", "04.1403", "14.0017", "59.0008"]
    test_plot_ids = [p for p in species_plot_ids["Picea abies"] if p not in plot_ids]
    literature_source = "Trotsiuk"
    file_paths = [
        prepare_plot_input(plot_id, literature_source=literature_source) for plot_id in plot_ids
    ]
    test_file_paths = [
        prepare_plot_input(plot_id, literature_source=literature_source)
        for plot_id in test_plot_ids
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
        target_vars=["DBH"],
        plot_variables=["BA", "DBH", "Height", "WS", "WF", "WR"],
        input_vars=input_vars,
        fit_phys_params=[],
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
    test_config = replace(config, file_paths=test_file_paths)
    predicted_series, rmse_tables = [], []
    for split, split_config, split_ids in (
        ("train", config, plot_ids),
        ("test", test_config, test_plot_ids),
    ):
        split_series = [
            build_predicted_series(
                config, file_path, fit_result.fitted_modifier_params, fit_result.fitted_phys_params
            )
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
        fit_result.fitted_modifier_params,
        save_path=str(image_dir / "modifier_response_surface.png"),
        show=False,
    )

    rmse_table = pl.concat(rmse_tables)
    with pl.Config(tbl_rows=-1, tbl_cols=-1, float_precision=3):
        print(rmse_table)
        print(summarize_rmse(rmse_table))

    plt.show()
