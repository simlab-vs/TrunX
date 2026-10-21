"""Learnable componenet of the 3PG model: a nutrition modifier optimized using gradient descent."""

import shutil
from collections.abc import Callable
from dataclasses import dataclass, field, fields, replace
from datetime import datetime
from functools import cache
from pathlib import Path
from typing import Any, Literal, cast

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import polars as pl

from trunx.config import clean_data_folder, images_folder, results_data_folder
from trunx.gp3.bayesiancalibrations.bayesian_config import species_plot_ids
from trunx.gp3.bayesiancalibrations.pymc_icp_plots import prepare_plot_input
from trunx.gp3.deposition_coverage import (
    DEPOSITION_RANGE_EDGES,
    MIN_MEASURED_SHARE,
    deposition_coverage,
    selected_plot_ids,
)
from trunx.gp3.extended_helper import (
    CURVATURE_INIT,
    DEPOSITION_CHANNELS,
    INPUT_VARIABLES,
    bivariate_quadratic_nm,
    constant_nm,
    init_mlp_modifier_params,
    liebig_nm,
    mlp_nm,
    poly_nm,
    prepare_modifier_inputs,
    product_nm,
)
from trunx.gp3.gradient_descent import apply_fitted_params, load_param_bounds
from trunx.gp3.model_inputs import ExtendedParams, InputData, SiteData, State
from trunx.gp3.prepare_data import prepare_data
from trunx.gp3.prepare_deposition import load_monthly_deposition
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
    # Whether `file_paths` hold each plot's calibrated parameters (see `prepare_plot_input`)
    calibrated_params: bool = False

    # Variables plotted against observations; defaults to `target_vars`
    plot_variables: list[str] | None = None
    observed_sheet: str = "observed"  # Sheet name for observed data
    fit_phys_params: list[str] | None = None
    param_bounds_sheet: str = "param_bound"
    species_index: int = 0  # Index of the species to optimize for
    # Which of "T_avg" and the `DEPOSITION_CHANNELS` keys the modifier is built over
    input_vars: tuple[str, ...] = INPUT_VARIABLES
    optimizer_name: str = "adam"  # Optimizer name: 'adam' or 'sgd'
    learning_rate: float = 1e-3  # Learning rate for the optimizer
    global_clip_norm: float = 1.0  # Global norm for gradient clipping
    num_epochs: int = 1000  # Number of training epochs
    modifier_l2: float = 0.0  # L2 penalty on the modifier weights (0: none)
    # Stop early when the loss improves by less than `loss_tol` over `loss_window` epochs
    loss_tol: float = 0.0  # 0: never stop early
    loss_window: int = 1000
    standardize_targets: bool = True  # Whether to standardize target variables
    # Start from the biomass pools derived from the observed DBH (see
    # `matched_initial_state`), instead of the observed biomass
    matched_start: bool = False
    # Fixed (mean, std) of the modifier inputs; None: those of all ICP plots (`input_statistics`)
    input_stats: tuple[jnp.ndarray, jnp.ndarray] | None = None
    image_dir: str = field(
        default_factory=lambda: str(images_folder / "nutrition_modifier/constant_modifier")
    )
    modifier_fn: Callable[[Any, jnp.ndarray, tuple[str, ...]], jnp.ndarray] = poly_nm


@dataclass
class NutritionModifierFitResult:
    """Container for nutrition modifier training results."""

    fitted_modifier_params: Any
    fitted_phys_params: dict[str, float]
    loss_history: list[float]
    param_history: list[Any]
    input_mean: jnp.ndarray  # Mean each modifier input is z-scored with
    input_std: jnp.ndarray  # Standard deviation each modifier input is z-scored with


@cache
def input_statistics(input_vars: tuple[str, ...]) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Mean and standard deviation of each modifier input over all ICP plot-months.

    Deposition comes from `icp_monthly_deposition.parquet` and `"T_avg"` from the ICP
    plots' `tmp_ave` in `trunx_plot_weather.parquet`, without any plot selection.
    Computed after deposition is clipped to its limits (see `prepare_modifier_inputs`),
    so the statistics describe the inputs the modifier actually sees.

    Parameters
    ----------
    input_vars : tuple[str, ...]
        Modifier inputs, any of `"T_avg"` and the `DEPOSITION_CHANNELS` keys.

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray]
        Mean and standard deviation, one entry per `input_vars` item.
    """
    deposition = load_monthly_deposition()
    weather = (
        pl.scan_parquet(clean_data_folder / "trunx_plot_weather.parquet")
        .filter(pl.col("source") == "ICP")
        .select("tmp_ave")
        .collect()
    )
    channels = {"T_avg": jnp.asarray(weather["tmp_ave"].to_numpy())}
    for name, column in DEPOSITION_CHANNELS.items():
        channels[name] = jnp.asarray(deposition[column].to_numpy())

    # Each input has its own plot-months, so each is pooled on its own
    means, stds = [], []
    for name in input_vars:
        values = np.asarray(prepare_modifier_inputs({name: channels[name]}, (name,)))
        means.append(np.nanmean(values))
        stds.append(max(np.nanstd(values), 1e-6))
    return jnp.asarray(means), jnp.asarray(stds)


def init_modifier_params(input_vars: tuple[str, ...] = INPUT_VARIABLES) -> jnp.ndarray:
    """Build a neutral (all-zero) starting point for `poly_nm`.

    Quadratic for a single input, bilinear cross-term otherwise.

    Parameters
    ----------
    input_vars : tuple[str, ...]
        Which of `"T_avg"` and the `DEPOSITION_CHANNELS` keys the modifier is built
        over, and in what order — must match what's passed to
        `run_3pg`/`train_nutrition_modifier`.
    """
    if len(input_vars) == 1:
        return jnp.zeros(3)
    return jnp.zeros(tuple(2 for _ in input_vars))


def neutral_modifier_params(
    modifier_fn: Callable[[Any, jnp.ndarray, tuple[str, ...]], jnp.ndarray],
    input_vars: tuple[str, ...],
) -> Any:
    """Neutral starting weights of `modifier_fn`, one of the `extended_helper` modifiers.

    Parameters
    ----------
    modifier_fn : Callable
        Modifier function the weights are for.
    input_vars : tuple[str, ...]
        Inputs of the modifier, see `init_modifier_params`.
    """
    if modifier_fn is constant_nm:
        return jnp.zeros(1)
    if modifier_fn is poly_nm:
        return init_modifier_params(input_vars)
    if modifier_fn in (product_nm, liebig_nm):
        return jnp.zeros((len(input_vars), 3)).at[:, 2].set(CURVATURE_INIT)
    if modifier_fn is bivariate_quadratic_nm:
        return jnp.zeros(6).at[3:5].set(CURVATURE_INIT)
    if modifier_fn is mlp_nm:
        return init_mlp_modifier_params(jax.random.PRNGKey(0), input_vars, hidden_sizes=(4,))
    raise ValueError(f"No initializer wired up for modifier_fn={modifier_fn!r}")


def initial_dbh(file_path: str, site_data: SiteData, observed_sheet: str = "observed") -> float:
    """Observed DBH of a plot at the start of its simulation.

    Parameters
    ----------
    file_path : str
        Input data file of the plot.
    site_data : SiteData
        Site of the plot, giving the start year and month.
    observed_sheet : str
        Sheet of the observations.

    Returns
    -------
    float
        DBH (cm) observed in the simulation's first month.
    """
    start_year = int(np.asarray(site_data.year_i).reshape(-1)[0])
    start_month = int(np.asarray(site_data.month_i).reshape(-1)[0])
    dbh = (
        pl.read_excel(file_path, sheet_name=observed_sheet)
        .filter(pl.col("year") == start_year, pl.col("month") == start_month)
        .get_column("DBH")
        .drop_nulls()
    )
    if dbh.is_empty():
        raise ValueError(
            f"No DBH observed at the start ({start_year}-{start_month}) of {file_path}"
        )
    return float(dbh[0])


def biomass_from_dbh(
    params: Any, dbh: Any, n_stems: Any, species_index: int = 0
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Stem, foliage and root biomass of a stand from its DBH, with the allometry of `params`.

    Stem biomass inverts `compute_dbh` (WS = N * aWS * DBH^nWS / 1000), foliage is WS
    times the foliage:stem ratio at `dbh` (from `pFS2` and `pFS20`), and roots are
    (WS + WF) times the root fraction halfway between `pRn` and `pRx`.

    Parameters
    ----------
    params : Any
        Physiological parameters, giving `aWS`, `nWS`, `pFS2`, `pFS20`, `pRn` and `pRx`.
    dbh : Any
        DBH (cm), a value or an array.
    n_stems : Any
        Stems per hectare, matching `dbh`.
    species_index : int
        Species whose parameters are used.

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
        Stem, foliage and root biomass (t ha⁻¹), shaped like `dbh`.
    """
    s = species_index
    # Stem biomass per tree (kg) from DBH, then per stand (t ha⁻¹)
    ws = params.aWS[s] * dbh ** params.nWS[s] * n_stems / 1000.0
    # Foliage:stem ratio as a power law of DBH through pFS2 at 2 cm and pFS20 at 20 cm
    pfs_power = jnp.log(params.pFS20[s] / (params.pFS2[s] + 1e-8)) / jnp.log(10.0)
    pfs_const = params.pFS2[s] / 2.0**pfs_power
    wf = ws * pfs_const * dbh**pfs_power
    # Root fraction halfway between its min and max, i.e. a moderate growing-condition
    # modifier of 0.5
    p_rs = params.pRn[s] + (params.pRx[s] - params.pRn[s]) * 0.5
    wr = (ws + wf) * p_rs
    return ws, wf, wr


def matched_initial_state(
    initial_state: State, params: Any, dbh: float, species_index: int = 0
) -> State:
    """Set the initial biomass pools from `dbh` with the allometry of `params`.

    For `species_index` only, see `biomass_from_dbh`. In a dormant start month the
    foliage goes to `WF_debt`, as in `prepare_data`.

    Parameters
    ----------
    initial_state : State
        Initial state to adjust.
    params : Any
        Physiological parameters, see `biomass_from_dbh`.
    dbh : float
        Observed DBH (cm) at the start, see `initial_dbh`.
    species_index : int
        Species whose biomass is set.

    Returns
    -------
    State
        `initial_state` with the matched stem, foliage and root biomass.
    """
    s = species_index
    ws, wf, wr = biomass_from_dbh(params, dbh, initial_state.N[s], s)

    # A dormant start keeps its foliage in WF_debt (WF is 0), see `prepare_data`
    dormant = initial_state.WF_debt[s] > 0
    return initial_state._replace(
        WS=initial_state.WS.at[s].set(ws),
        WF=initial_state.WF.at[s].set(jnp.where(dormant, 0.0, wf)),
        WF_debt=initial_state.WF_debt.at[s].set(jnp.where(dormant, wf, 0.0)),
        WR=initial_state.WR.at[s].set(wr),
    )


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
    start_dbh: float | None = None,
) -> Callable[[dict[str, Any]], jnp.ndarray]:
    """Create a loss jointly over physiological and nutrition-modifier parameters.

    The returned loss takes a pytree `{"phys_params": {name: value}, "modifier_params": ...}`;
    `phys_params` overrides the matching `input_data.params` fields for `species_index`.
    The modifier inputs are standardised with `input_mean` and `input_std`. With a
    `start_dbh`, the run starts from `matched_initial_state`.
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
        initial_state = input_data.initial_state
        if start_dbh is not None:
            initial_state = matched_initial_state(initial_state, params, start_dbh, species_index)
        _, pg3_outputs = run_3pg(
            initial_state,
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
    observed_sheet: str = "observed",
):
    """Build observed values and scales for the target variables.

    Only the target columns are kept, and rows without any target observation (e.g.
    GPP-only months) are dropped.
    """
    observed_data = pl.read_excel(file_path, sheet_name=observed_sheet)
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


def input_scaling(config: NutritionModifierConfig) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Mean and standard deviation the modifier inputs are z-scored with.

    Parameters
    ----------
    config : NutritionModifierConfig
        The statistics are `config.input_stats` if given, else those of all ICP plots
        (see `input_statistics`).

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray]
        Mean and standard deviation, one entry per `config.input_vars` item.
    """
    if config.input_stats is not None:
        return config.input_stats
    return input_statistics(tuple(config.input_vars))


def phys_param_bounds(config: NutritionModifierConfig) -> dict[str, tuple[float, float]]:
    """(min, max) bounds of the physiological parameters to fit, from the first plot.

    Parameters
    ----------
    config : NutritionModifierConfig
        `config.fit_phys_params` names the parameters; None means every parameter with a
        min and max in the `config.param_bounds_sheet` sheet.

    Returns
    -------
    dict[str, tuple[float, float]]
        Bounds of each parameter to fit.
    """
    sheet_bounds = {
        name: (float(bounds[0]), float(bounds[1]))
        for name, bounds in load_param_bounds(
            config.file_paths[0], config.param_bounds_sheet
        ).items()
        if not np.isnan(bounds).any()
    }
    names = list(sheet_bounds) if config.fit_phys_params is None else config.fit_phys_params
    missing = [name for name in names if name not in sheet_bounds]
    if missing:
        raise ValueError(f"No min/max in '{config.param_bounds_sheet}' for: {missing}")
    return {name: sheet_bounds[name] for name in names}


def phys_param_defaults(config: NutritionModifierConfig, names: list[str]) -> dict[str, Any]:
    """Values of the physiological parameters `names` in the first plot's input file.

    Parameters
    ----------
    config : NutritionModifierConfig
        Gives the first plot (`config.file_paths[0]`) and `config.species_index`.
    names : list[str]
        Physiological parameter names.

    Returns
    -------
    dict[str, Any]
        Value of each parameter for the species.
    """
    params = prepare_data(config.file_paths[0]).params
    return {
        name: jnp.atleast_1d(jnp.asarray(getattr(params, name)))[config.species_index]
        for name in names
    }


def train_nutrition_modifier(
    config: NutritionModifierConfig,
    initial_modifier_params: Any,
) -> NutritionModifierFitResult:
    """Jointly train `config.fit_phys_params` and the nutrition modifier over all plots.

    The loss is the mean of the per-plot losses, so every plot weighs equally, plus
    `config.modifier_l2` times the sum of the squared modifier weights; the
    physiological parameters start from the first plot's values. The modifier inputs
    are z-scored with their mean and standard deviation over all ICP plots (see
    `input_scaling`).
    `initial_modifier_params` must match `config.modifier_fn`'s own parameter pytree —
    e.g. `init_modifier_params` for `poly_nm`, `init_mlp_modifier_params` for `mlp_nm` —
    since they aren't interchangeable.
    """
    input_mean, input_std = input_scaling(config)
    plot_losses = []
    for file_path in config.file_paths:
        input_data = prepare_data(file_path)
        obs_indices, obs_values, obs_scales = build_observation_data(
            file_path,
            input_data.site,
            config.target_vars,
            standardize_targets=config.standardize_targets,
            observed_sheet=config.observed_sheet,
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
                start_dbh=(
                    initial_dbh(file_path, input_data.site, config.observed_sheet)
                    if config.matched_start
                    else None
                ),
            )
        )

    def loss_function(trainable: dict[str, Any]) -> jnp.ndarray:
        """Mean of the per-plot losses plus the L2 penalty on the modifier weights."""
        data_loss = jnp.mean(jnp.stack([plot_loss(trainable) for plot_loss in plot_losses]))
        penalty = sum(
            jnp.sum(leaf**2) for leaf in jax.tree_util.tree_leaves(trainable["modifier_params"])
        )
        return data_loss + config.modifier_l2 * penalty

    bounds = phys_param_bounds(config)

    def clip_to_bounds(phys_params: dict[str, jnp.ndarray]) -> dict[str, jnp.ndarray]:
        """Clip each physiological parameter to its [min, max] bounds."""
        return {name: jnp.clip(value, *bounds[name]) for name, value in phys_params.items()}

    phys_params = phys_param_defaults(config, list(bounds))
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
        if (
            config.loss_tol > 0
            and epoch >= config.loss_window
            and loss_history[-config.loss_window - 1] - loss_history[-1] < config.loss_tol
        ):
            print(
                f"Stopped at epoch {epoch}: loss improved < {config.loss_tol} "
                f"over {config.loss_window} epochs"
            )
            break

    fitted_phys_params = {name: float(v) for name, v in trainable["phys_params"].items()}
    print(f"Final Loss: {loss_history[-1]}", f"Final phys params: {fitted_phys_params}")
    print(
        "Fitted modifier weights:",
        jax.tree_util.tree_map(np.asarray, trainable["modifier_params"]),
    )
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
    is the same as `"default"`, which is then left out. `"start_ref"` is the `"phys"` run
    from the other initial state than `config.matched_start` sets.
    """
    phys = "Fitted" if config.fit_phys_params else "Default"
    start, other_start = ("matched-DBH", "observed-biomass")
    if not config.matched_start:
        start, other_start = other_start, start
    labels = {
        "fitted": f"{phys} physiological parameters, with nutrition modifier, {start} start",
        "phys": f"{phys} physiological parameters, without nutrition modifier, {start} start",
    }
    if config.fit_phys_params:
        labels["default"] = (
            f"Default physiological parameters, without nutrition modifier, {start} start"
        )
    labels["start_ref"] = (
        f"{phys} physiological parameters, without nutrition modifier, {other_start} start"
    )
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
    (see `train_nutrition_modifier`) — and its input standardisation.

    Returns
    -------
    dict[str, np.ndarray]
        `"dates"` (one per simulated month) plus, for each of `config.target_vars` and
        the plot variables, `"pred_fitted_<var>"` (fitted phys params + modifier),
        `"pred_phys_<var>"` (fitted phys params only), `"pred_default_<var>"`
        (default params) and `"pred_start_ref_<var>"` (fitted phys params only, from the
        other initial state than `config.matched_start` sets).
        `alpha_c`, `f_nutri_classic_learnable` and `NPP` are always included, to show
        the modifier's effect on them, plus `"learned_modifier"` (the modifier's monthly
        value).
        `"dbh_derived_WS"`, `"dbh_derived_WF"` and `"dbh_derived_WR"` are the biomass
        derived from each observed DBH and stem number (see `biomass_from_dbh`), at
        `"dbh_derived_dates"`.
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
    start_dbh = initial_dbh(file_path, input_data.site, config.observed_sheet)

    def initial_state(params: Any, matched: bool) -> State:
        """Observed-biomass initial state, or the DBH-matched one for `params`."""
        if not matched:
            return input_data.initial_state
        return matched_initial_state(
            input_data.initial_state, params, start_dbh, config.species_index
        )

    _, outputs_fitted = run_3pg(
        initial_state(fitted_params, config.matched_start),
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
        initial_state(fitted_params, config.matched_start),
        input_data.climate,
        fitted_params,
        input_data.site,
        input_data.species,
    )
    _, outputs_default = run_3pg(
        initial_state(input_data.params, config.matched_start),
        input_data.climate,
        input_data.params,
        input_data.site,
        input_data.species,
    )
    _, outputs_start_ref = run_3pg(
        initial_state(fitted_params, not config.matched_start),
        input_data.climate,
        fitted_params,
        input_data.site,
        input_data.species,
    )

    series: dict[str, np.ndarray] = {}
    for var_name in dict.fromkeys(
        [
            *config.target_vars,
            *_plot_variables(config),
            "alpha_c",
            "f_nutri_classic_learnable",
            "NPP",
        ]
    ):
        for label, outputs in (
            ("fitted", outputs_fitted),
            ("phys", outputs_phys),
            ("default", outputs_default),
            ("start_ref", outputs_start_ref),
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

    # Biomass derived from each census with both DBH and stem numbers observed
    censuses = pl.read_excel(file_path, sheet_name=config.observed_sheet).filter(
        pl.col("DBH").is_not_null(), pl.col("N").is_not_null()
    )
    derived = biomass_from_dbh(
        fitted_params,
        censuses["DBH"].to_numpy(),
        censuses["N"].to_numpy(),
        config.species_index,
    )
    series["dbh_derived_dates"] = np.array(
        [
            np.datetime64(f"{year}-{month:02d}", "M")
            for year, month in zip(censuses["year"], censuses["month"], strict=True)
        ]
    )
    for var_name, values in zip(("WS", "WF", "WR"), derived, strict=True):
        series[f"dbh_derived_{var_name}"] = np.asarray(values)
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


def config_summary_lines(config: NutritionModifierConfig, skip: tuple[str, ...] = ()) -> list[str]:
    """One indented `name: value` line per setting of `config`, for a run summary.

    Parameters
    ----------
    config : NutritionModifierConfig
        Training configuration.
    skip : tuple[str, ...]
        Settings to leave out, besides `file_paths` and `input_stats`.

    Returns
    -------
    list[str]
        Lines starting with the modifier function's name.
    """
    left_out = ("file_paths", "input_stats", "modifier_fn", *skip)
    return [
        f"  modifier_fn: {getattr(config.modifier_fn, '__name__', config.modifier_fn)}",
        *(
            f"  {f.name}: {getattr(config, f.name)}"
            for f in fields(config)
            if f.name not in left_out
        ),
    ]


def write_run_summary(
    save_path: Path,
    config: NutritionModifierConfig,
    fit_result: NutritionModifierFitResult,
    species: str,
    literature_source: str,
    plot_ids: list[str],
    test_plot_ids: list[str],
    rmse_table: pl.DataFrame,
) -> None:
    """Write the settings, fitted values and RMSE of a nutrition modifier run to a text file.

    Parameters
    ----------
    save_path : Path
        Text file to write.
    config : NutritionModifierConfig
        Training configuration of the run.
    fit_result : NutritionModifierFitResult
        Result of `train_nutrition_modifier`.
    species : str
        Species of the plots.
    literature_source : str
        Literature source of the 3PG parameters of the plot inputs.
    plot_ids, test_plot_ids : list[str]
        Training and test plot ids.
    rmse_table : pl.DataFrame
        Concatenated `compute_rmse` tables with a `split` column.
    """
    modifier_params = jax.tree_util.tree_map(
        lambda leaf: np.asarray(leaf).tolist(), fit_result.fitted_modifier_params
    )
    lines = [
        "Nutrition modifier run",
        f"Date: {datetime.now():%Y-%m-%d %H:%M}",
        "",
        "Data",
        f"  species: {species}",
        f"  literature_source: {literature_source}",
        f"  train plots ({len(plot_ids)}): {', '.join(plot_ids)}",
        f"  test plots ({len(test_plot_ids)}): {', '.join(test_plot_ids)}",
        "",
        "Settings",
        *config_summary_lines(config),
        f"  input mean: {np.asarray(fit_result.input_mean).tolist()}",
        f"  input std: {np.asarray(fit_result.input_std).tolist()}",
        "",
        "Results",
        f"  epochs run: {len(fit_result.loss_history)}",
        f"  final loss: {fit_result.loss_history[-1]:.6f}",
        f"  fitted modifier params: {modifier_params}",
        f"  fitted phys params: {fit_result.fitted_phys_params}",
        "",
    ]
    with pl.Config(tbl_rows=-1, tbl_cols=-1, float_precision=3, tbl_width_chars=200):
        lines += [
            "RMSE per plot",
            str(rmse_table),
            "",
            "Mean RMSE",
            str(summarize_rmse(rmse_table)),
        ]
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text("\n".join(lines) + "\n")


def split_plot_ids(
    species: str, plot_ids: list[str], test_fraction: float = 0.2, seed: int = 0
) -> tuple[list[str], list[str]]:
    """Randomly split `plot_ids` passing `deposition_coverage` into train and test.

    The split is done within each N deposition range (`DEPOSITION_RANGE_EDGES`), keeping
    at least one plot of every range in the training set.

    Parameters
    ----------
    species : str
        Dominant species of the plots, e.g. "Picea abies".
    plot_ids : list[str]
        Plots to split, e.g. those kept by `poorly_fitted_without_modifier`.
    test_fraction : float
        Share of each range's plots held out for testing, rounded up.
    seed : int
        Random seed of the split.

    Returns
    -------
    tuple[list[str], list[str]]
        Sorted train and test plot ids.
    """
    coverage = deposition_coverage(
        {species: plot_ids}, variables=("dep_n_tot", "dep_s_so4")
    ).filter(pl.col("% measured") >= MIN_MEASURED_SHARE)
    ranges = coverage.with_columns(
        pl.col("mean dep_n_tot").cut(list(DEPOSITION_RANGE_EDGES)).alias("range")
    ).partition_by("range")

    rng = np.random.default_rng(seed)
    train_ids, test_ids = [], []
    for plots in ranges:
        shuffled = rng.permutation(plots["plot_id"].to_list()).tolist()
        # Hold out at most all but one plot of the range
        n_test = min(int(np.ceil(test_fraction * len(shuffled))), len(shuffled) - 1)
        test_ids += shuffled[:n_test]
        train_ids += shuffled[n_test:]
    return sorted(train_ids), sorted(test_ids)


def with_target_observations(
    config: NutritionModifierConfig, plot_ids: list[str], file_paths: list[str]
) -> tuple[list[str], list[str]]:
    """Keep the plots with at least one observation of any of `config.target_vars`.

    Parameters
    ----------
    config : NutritionModifierConfig
        Gives the target variables and the observed sheet.
    plot_ids : list[str]
        Plot ids, in `file_paths` order.
    file_paths : list[str]
        Input data files of the plots.

    Returns
    -------
    tuple[list[str], list[str]]
        Plot ids and input files of the kept plots, in the same order.
    """
    kept = []
    for plot_id, file_path in zip(plot_ids, file_paths, strict=True):
        observed = pl.read_excel(file_path, sheet_name=config.observed_sheet)
        n_values = sum(
            observed[var].cast(pl.Float64).fill_nan(None).count()
            for var in config.target_vars
            if var in observed.columns
        )
        if n_values:
            kept.append((plot_id, file_path))
        else:
            print(f"Leaving out plot {plot_id}: no observation of {config.target_vars}")
    return [plot_id for plot_id, _ in kept], [file_path for _, file_path in kept]


# Plots whose DBH the run without the modifier already fits within these thresholds are
# left out, as the modifier has nothing to correct there. Per misfit criterion:
# "rmse": RMSE (cm), "relative_rmse": RMSE / mean observed DBH, "bias": |mean residual| (cm)
MIN_DBH_MISFIT = {"rmse": 0.5, "relative_rmse": 0.02, "bias": 0.2}


def poorly_fitted_without_modifier(
    config: NutritionModifierConfig,
    plot_ids: list[str],
    file_paths: list[str],
    criterion: Literal["rmse", "relative_rmse", "bias"] = "rmse",
    min_misfit: float | None = None,
) -> tuple[list[str], list[str]]:
    """Keep the plots whose DBH misfit without the modifier is at least `min_misfit`.

    Parameters
    ----------
    config : NutritionModifierConfig
        Gives the observed sheet and phys parameter settings.
    plot_ids : list[str]
        Plot ids, in `file_paths` order.
    file_paths : list[str]
        Input data files of the plots.
    criterion : {"rmse", "relative_rmse", "bias"}
        DBH misfit: "rmse": RMSE (cm),
        "relative_rmse": RMSE over the mean observed DBH,
        "bias": |mean (predicted - observed)| (cm).
    min_misfit : float | None
        Smallest misfit of a kept plot; None: `MIN_DBH_MISFIT[criterion]`.

    Returns
    -------
    tuple[list[str], list[str]]
        Plot ids and input files of the kept plots, in the same order.
    """
    if min_misfit is None:
        min_misfit = MIN_DBH_MISFIT[criterion]
    # Only the run without the modifier (`pred_phys_DBH`) is used, so any modifier will do;
    # constant_nm with a zero weight works whatever `config.modifier_fn` is
    config = replace(config, modifier_fn=constant_nm)
    input_mean, input_std = input_scaling(config)
    untrained = NutritionModifierFitResult(
        fitted_modifier_params=jnp.zeros(1),
        fitted_phys_params={},
        loss_history=[],
        param_history=[],
        input_mean=input_mean,
        input_std=input_std,
    )
    kept = []
    for plot_id, file_path in zip(plot_ids, file_paths, strict=True):
        series = build_predicted_series(config, file_path, untrained)
        # DBH residuals without the modifier at the observed months, as in `compute_rmse`
        observed_data = pl.read_excel(file_path, sheet_name=config.observed_sheet)
        observed_dates = np.array(
            [
                np.datetime64(f"{year}-{month:02d}", "M")
                for year, month in zip(observed_data["year"], observed_data["month"], strict=True)
            ]
        )
        obs_indices = (observed_dates - series["dates"][0]).astype(int)
        observed = observed_data["DBH"].cast(pl.Float64).to_numpy()
        mask = ~np.isnan(observed)
        residuals = series["pred_phys_DBH"][obs_indices[mask]] - observed[mask]

        rmse = float(np.sqrt(np.mean(residuals**2)))
        misfit = {
            "rmse": rmse,
            "relative_rmse": rmse / float(np.mean(observed[mask])),
            "bias": abs(float(np.mean(residuals))),
        }[criterion]
        if misfit >= min_misfit:
            kept.append((plot_id, file_path))
        else:
            print(
                f"Leaving out plot {plot_id}: DBH {criterion} without modifier {misfit:.2f} "
                f"< {min_misfit}"
            )
        jax.clear_caches()
    return [plot_id for plot_id, _ in kept], [file_path for _, file_path in kept]


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
        plot_learned_modifier_in_deposition_space,
        plot_learned_modifier_vs_deposition,
        plot_modifier_effect_over_time,
        plot_modifier_effect_vs_deposition,
        plot_modifier_pass_through,
        plot_modifier_response_curve,
        plot_modifier_response_surface,
        plot_observed_vs_predicted,
        plot_param_history_over_iterations,
    )

    # Train on these plots; the remaining plots of the species are held out for testing
    # plot_ids = ["04.1402", "04.1403", "14.0017", "59.0008"]
    # test_plot_ids = [p for p in species_plot_ids["Picea abies"] if p not in plot_ids]

    species = "Picea abies"
    literature_source = "Forrester"
    calibrated_params = False  # Use each plot's parameters from the calibration sweep

    all_plot_ids = selected_plot_ids()[species]
    all_file_paths = [
        prepare_plot_input(
            plot_id, literature_source=literature_source, calibrated=calibrated_params
        )
        for plot_id in all_plot_ids
    ]

    # "T_avg" or `DEPOSITION_CHANNELS` keys: "N", "S" (monthly), "N_avg", "S_avg" (12-month
    # means), "N_sum", "S_sum" (12-month sums)
    input_vars = ("N", "S")

    # Select which learnable nutrition-modifier function to fit.
    # constant_nm ignores its inputs, but the input pipeline needs one, e.g. ("N",)
    modifier_fn = product_nm
    modifier_l2 = 0.2

    config = NutritionModifierConfig(
        file_paths=all_file_paths,
        target_vars=[
            "DBH",
        ],
        calibrated_params=calibrated_params,
        plot_variables=["BA", "DBH", "WS", "WF", "WR", "N"],
        input_vars=input_vars,
        observed_sheet="observed",
        matched_start=True,
        fit_phys_params=[],
        optimizer_name="adam",
        learning_rate=1e-3,
        num_epochs=5000,
        modifier_l2=modifier_l2,
        loss_tol=1e-4,
        modifier_fn=modifier_fn,
        # One folder per literature source, inputs, modifier and L2 penalty, as in the LOOCV
        # runs
        image_dir=str(
            results_data_folder
            / "nutrition_modifier"
            / literature_source
            / "_".join(sorted(input_vars))  # Same folder for ("N", "S") and ("S", "N")
            / "tests"
            / modifier_fn.__name__
            / f"l2={modifier_l2}"
            / "images"
        ),
    )
    # Leave out plots without any target observation (e.g. no GOSIF GPP), and plots the
    # run without the modifier already fits well, before splitting into train and test
    all_plot_ids, all_file_paths = with_target_observations(config, all_plot_ids, all_file_paths)
    all_plot_ids, all_file_paths = poorly_fitted_without_modifier(
        config, all_plot_ids, all_file_paths, criterion="rmse"
    )

    plot_ids, test_plot_ids = split_plot_ids(species, all_plot_ids)
    file_path_of = dict(zip(all_plot_ids, all_file_paths, strict=True))
    file_paths = [file_path_of[plot_id] for plot_id in plot_ids]
    test_file_paths = [file_path_of[plot_id] for plot_id in test_plot_ids]
    config = replace(config, file_paths=file_paths)

    # Start from an empty image folder, so no figure of an earlier run is left behind
    image_dir = Path(config.image_dir)
    shutil.rmtree(image_dir, ignore_errors=True)
    image_dir.mkdir(parents=True)

    fit_result = train_nutrition_modifier(
        config, initial_modifier_params=neutral_modifier_params(modifier_fn, input_vars)
    )

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
    for var_name in ("alpha_c", "f_nutri_classic_learnable", "NPP"):
        plot_modifier_effect_over_time(
            config,
            plot_ids,
            predicted_series,
            var_name=var_name,
            save_path=str(image_dir / f"{var_name}.png"),
            show=False,
        )
    plot_modifier_pass_through(
        plot_ids,
        predicted_series,
        save_path=str(image_dir / "modifier_pass_through.png"),
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
    plot_learned_modifier_in_deposition_space(
        config,
        predicted_series,
        plot_ids,
        save_path=str(image_dir / "learned_modifier_n_vs_s.png"),
        show=False,
    )
    plot_deposition_over_time(
        config,
        predicted_series,
        plot_ids,
        save_path=str(image_dir / "deposition_over_time.png"),
        show=False,
    )
    if len(config.input_vars) == 1:
        plot_modifier_response_curve(
            config,
            plot_ids,
            [fit_result],
            save_path=str(image_dir / "modifier_response_curve.png"),
            show=False,
        )
    else:
        surfaces = [(False, "response_surface")]
        # The score before tanh only exists for poly_nm's 1 + tanh squash
        if config.modifier_fn is poly_nm:
            surfaces.append((True, "score_before_tanh"))
        for pre_tanh, name in surfaces:
            plot_modifier_response_surface(
                config,
                plot_ids,
                [fit_result],
                save_path=str(image_dir / f"modifier_{name}.png"),
                show=False,
                pre_tanh=pre_tanh,
            )

    rmse_table = pl.concat(rmse_tables)
    with pl.Config(tbl_rows=-1, tbl_cols=-1, float_precision=3):
        print(rmse_table)
        print(summarize_rmse(rmse_table))
    # In the run folder, next to `images/`, as in the LOOCV runs
    run_summary_path = image_dir.parent / "run_summary.txt"
    write_run_summary(
        run_summary_path,
        config,
        fit_result,
        species,
        literature_source,
        plot_ids,
        test_plot_ids,
        rmse_table,
    )
    print(f"Saved {run_summary_path}")

    plt.show()
