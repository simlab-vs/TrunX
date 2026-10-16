"""Learnable componenet of the 3PG model: a nutrition modifier optimized using gradient descent."""

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import polars as pl

from trunx.config import images_folder
from trunx.gp3.bayesiancalibrations.bayesian_config import species_plot_ids
from trunx.gp3.bayesiancalibrations.pymc_icp_plots import prepare_plot_input
from trunx.gp3.extended_helper import (
    INPUT_VARIABLES,
    init_mlp_modifier_params,
    mlp_nm,
    poly_nm,
    prepare_modifier_inputs,
)
from trunx.gp3.gradient_descent import apply_fitted_params, load_param_bounds
from trunx.gp3.model_inputs import ExtendedParams, InputData, SiteData
from trunx.gp3.prepare_data import prepare_data
from trunx.gp3.run_3pg import run_3pg
from trunx.gp3.training_utils import (
    build_observation_indices,
    build_optimizer,
    count_observed_rows,
    plot_loss_over_iterations,
    weighted_squared_error,
)


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
    standardize_inputs: bool = True  # Whether to standardize the modifier inputs
    image_dir: str = field(default_factory=lambda: str(images_folder / "nn_nutrition_modifier"))
    modifier_fn: Callable[[Any, jnp.ndarray, tuple[str, ...]], jnp.ndarray] = poly_nm


@dataclass
class NutritionModifierFitResult:
    """Container for nutrition modifier training results."""

    fitted_modifier_params: Any
    fitted_phys_params: dict[str, float]
    loss_history: list[float]
    param_history: list[Any]
    input_mean: jnp.ndarray  # Training-plot mean of each modifier input
    input_std: jnp.ndarray  # Training-plot standard deviation of each modifier input


def input_statistics(
    file_paths: list[str], input_vars: tuple[str, ...]
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Mean and standard deviation of each modifier input, pooled over the plots' months.

    Computed after deposition is clipped to its limits (see `prepare_modifier_inputs`),
    so the statistics describe the inputs the modifier actually sees.

    Parameters
    ----------
    file_paths : list[str]
        Input data files of the plots to pool.
    input_vars : tuple[str, ...]
        Modifier inputs, any of `("N", "S", "T_avg")`.

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray]
        Mean and standard deviation, one entry per `input_vars` item.
    """
    pooled: list[np.ndarray] = []
    for file_path in file_paths:
        input_data = prepare_data(file_path)
        channels = {"T_avg": input_data.climate.T_avg}
        if input_data.deposition is not None:
            channels["N"] = input_data.deposition.dep_n_tot
            channels["S"] = input_data.deposition.dep_s_so4
        pooled.append(np.asarray(prepare_modifier_inputs(channels, input_vars)))
    values = np.concatenate(pooled, axis=0)
    std = np.maximum(np.nanstd(values, axis=0), 1e-6)
    return jnp.asarray(np.nanmean(values, axis=0)), jnp.asarray(std)


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
    input_mean: jnp.ndarray,
    input_std: jnp.ndarray,
    modifier_fn: Callable[[Any, jnp.ndarray, tuple[str, ...]], jnp.ndarray] = poly_nm,
    input_vars: tuple[str, ...] = INPUT_VARIABLES,
    species_index: int = 0,
) -> Callable[[dict[str, Any]], jnp.ndarray]:
    """Create a loss jointly over physiological and nutrition-modifier parameters.

    The returned loss takes a pytree `{"phys_params": {name: value}, "modifier_params": ...}`;
    `phys_params` overrides the matching `input_data.params` fields for `species_index`.
    The modifier inputs are standardised with `input_mean` and `input_std`.
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
        extended_params = ExtendedParams(
            modifier_params=trainable["modifier_params"],
            input_mean=input_mean,
            input_std=input_std,
        )
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
    physiological parameters start from the first plot's values. With
    `config.standardize_inputs`, the modifier inputs are standardised with their mean
    and standard deviation over these plots.
    `initial_modifier_params` must match `config.modifier_fn`'s own parameter pytree —
    e.g. `init_modifier_params` for `poly_nm`, `init_mlp_modifier_params` for `mlp_nm` —
    since they aren't interchangeable.
    """
    if config.standardize_inputs:
        input_mean, input_std = input_statistics(config.file_paths, config.input_vars)
    else:
        input_mean = jnp.zeros(len(config.input_vars))
        input_std = jnp.ones(len(config.input_vars))
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
                input_mean=input_mean,
                input_std=input_std,
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
        input_mean=input_mean,
        input_std=input_std,
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


def build_predicted_series(
    config: NutritionModifierConfig,
    file_path: str,
    fit_result: NutritionModifierFitResult,
) -> dict[str, np.ndarray]:
    """Simulate one plot (`file_path`) with fitted and default parameters, for plotting.

    Uses `config.modifier_fn` — must match whatever `fit_result` was fitted with
    (see `train_nutrition_modifier`) — and the training plots' input standardisation.

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
    extended_params = ExtendedParams(
        modifier_params=fit_result.fitted_modifier_params,
        input_mean=fit_result.input_mean,
        input_std=fit_result.input_std,
    )
    fitted_phys_params = fit_result.fitted_phys_params
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


if __name__ == "__main__":
    from trunx.gp3.plots_nutrition_modifier import (
        plot_deposition_density,
        plot_deposition_over_time,
        plot_learned_modifier_vs_deposition,
        plot_modifier_effect_over_time,
        plot_modifier_effect_vs_deposition,
        plot_modifier_response_surface,
        plot_observed_vs_predicted,
        plot_param_history_over_iterations,
    )

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
        standardize_inputs=True,
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

    rmse_table = pl.concat(rmse_tables)
    with pl.Config(tbl_rows=-1, tbl_cols=-1, float_precision=3):
        print(rmse_table)
        print(summarize_rmse(rmse_table))

    plt.show()
