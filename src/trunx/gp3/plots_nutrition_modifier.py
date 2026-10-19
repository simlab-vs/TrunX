"""Plots of the learnable nutrition modifier's training and fitted behaviour."""

from pathlib import Path
from typing import Any, Literal

import jax
import jax.numpy as jnp
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from trunx.gp3.extended_helper import DEPOSITION_CHANNELS, prepare_modifier_inputs
from trunx.gp3.nn_nutrition_modifier import (
    NutritionModifierConfig,
    NutritionModifierFitResult,
    _plot_variables,
    _run_labels,
)
from trunx.gp3.prepare_data import prepare_data
from trunx.gp3.training_utils import plot_traces_grid

_METRIC_LABELS = {
    "DBH": "DBH (cm)",
    "WS": "Stem Biomass (t DM ha⁻¹)",
    "WF": "Foliage Biomass (t DM ha⁻¹)",
    "WR": "Root Biomass (t DM ha⁻¹)",
    "Height": "Height (m)",
    "BA": "Basal Area (m² ha⁻¹)",
    "alpha_c": "alpha_c (mol C mol⁻¹ PAR)",
    "f_nutri_classic_learnable": "fN × learned modifier (–)",
    "NPP": "NPP (t DM ha⁻¹ month⁻¹)",
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

# (N, S) deposition input pairs: axis label suffix and unit of their channels
_DEPOSITION_PAIRS: dict[tuple[str, str], tuple[str, str]] = {
    ("N", "S"): ("", "kg ha⁻¹ month⁻¹"),
    ("N_avg", "S_avg"): (", 12-month mean", "kg ha⁻¹ month⁻¹"),
    ("N_sum", "S_sum"): (", 12-month sum", "kg ha⁻¹ per 12 months"),
}

_RUN_STYLES: dict[str, dict[str, str]] = {
    "fitted": {"color": "tab:blue", "linestyle": "-"},
    "phys": {"color": "tab:orange", "linestyle": "-."},
    "default": {"color": "tab:green", "linestyle": "--"},
    "start_ref": {"color": "tab:purple", "linestyle": ":"},
}


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


def _load_deposition(
    file_paths: list[str], names: tuple[str, ...] = ("N", "S")
) -> list[list[np.ndarray]]:
    """Load the modifier inputs of each plot, in `file_paths` order, one list per name.

    `names` are `"T_avg"` or `DEPOSITION_CHANNELS` keys to load, monthly ("N", "S") by
    default.
    """
    values: list[list[np.ndarray]] = [[] for _ in names]
    for file_path in file_paths:
        input_data = prepare_data(file_path)
        for name_values, name in zip(values, names, strict=True):
            if name == "T_avg":
                name_values.append(np.asarray(input_data.climate.T_avg))
                continue
            if input_data.deposition is None:
                raise ValueError(f"No deposition data in {file_path}")
            name_values.append(
                np.asarray(getattr(input_data.deposition, DEPOSITION_CHANNELS[name]))
            )
    return values


def _input_label(name: str) -> str:
    """Axis label of a modifier input, `"T_avg"` or a `DEPOSITION_CHANNELS` key."""
    if name == "T_avg":
        return "Mean temperature (°C)"
    label, unit = next(info for pair, info in _DEPOSITION_PAIRS.items() if name in pair)
    return f"{name[0]} deposition{label} ({unit})"


def _modifier_deposition(config: NutritionModifierConfig) -> list[tuple[list[np.ndarray], str]]:
    """Deposition inputs of the modifier, each plot's values with an axis label.

    One item per `DEPOSITION_CHANNELS` key in `config.input_vars` (e.g. only N for a
    modifier over N alone); monthly N and S when the modifier has no deposition input.
    """
    names = tuple(name for name in config.input_vars if name in DEPOSITION_CHANNELS)
    names = names or ("N", "S")
    labels = [_input_label(name) for name in names]
    return list(zip(_load_deposition(config.file_paths, names), labels, strict=True))


def _deposition_range(values: list[np.ndarray], floor_at_zero: bool = True) -> tuple[float, float]:
    """1st-99th percentile of the pooled `values`, padded by 20% and floored at 0.

    `floor_at_zero=False` keeps negative values, e.g. for temperature.
    """
    low, high = np.nanpercentile(np.concatenate(values), [1, 99])
    pad = 0.2 * (high - low)
    low = float(low - pad)
    return max(low, 0.0) if floor_at_zero else low, float(high + pad)


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


def _gaussian_kde(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Gaussian kernel density of `values` on `grid`, with Silverman's bandwidth."""
    values = values[~np.isnan(values)]
    bandwidth = 1.06 * np.std(values) * len(values) ** (-1 / 5)
    z = (grid[:, None] - values[None, :]) / bandwidth
    return np.exp(-0.5 * z**2).sum(axis=1) / (len(values) * bandwidth * np.sqrt(2 * np.pi))


def plot_deposition_density(
    config: NutritionModifierConfig,
    plot_ids: list[str],
    kind: Literal["hist", "kde"] = "hist",
    n_bins: int = 30,
    n_grid: int = 200,
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """Plot each plot's density of monthly N and S deposition, as a histogram or a KDE.

    `n_bins` sets the histogram bins and `n_grid` the KDE evaluation points.
    """
    n_values, s_values = _load_deposition(config.file_paths)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), layout="constrained")

    for ax, values, label in (
        (axes[0], n_values, "N deposition (kg ha⁻¹ month⁻¹)"),
        (axes[1], s_values, "S deposition (kg ha⁻¹ month⁻¹)"),
    ):
        value_range = _deposition_range(values)
        grid = np.linspace(*value_range, n_grid)
        for idx, (plot_id, deposition) in enumerate(zip(plot_ids, values, strict=True)):
            color = _PLOT_COLORS[idx % len(_PLOT_COLORS)]
            deposition = deposition[~np.isnan(deposition)]
            if not deposition.size:
                continue
            if kind == "hist":
                ax.hist(
                    deposition,
                    bins=n_bins,
                    range=value_range,
                    density=True,
                    histtype="step",
                    color=color,
                    linewidth=1.5,
                    label=plot_id,
                )
            else:
                ax.plot(
                    grid,
                    _gaussian_kde(deposition, grid),
                    color=color,
                    linewidth=1.5,
                    label=plot_id,
                )
        ax.set_xlabel(label)
        ax.set_ylabel("Density")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside right upper", fontsize=7, title="Plot")
    fig.suptitle("Monthly deposition distribution per plot")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()


def _input_grid(
    config: NutritionModifierConfig, n_grid: int
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray]:
    """Each plot's values of the modifier's two inputs, and a grid over their padded range.

    The grid covers the 1st-99th percentile of the pooled values, padded by 20%, with
    `config.input_vars[0]` along x and `config.input_vars[1]` along y.
    """
    if len(config.input_vars) != 2:
        raise ValueError(f"Needs a modifier over exactly two inputs, got {config.input_vars}")
    x_name, y_name = config.input_vars
    x_values, y_values = _load_deposition(config.file_paths, config.input_vars)
    x_grid_values, y_grid_values = np.meshgrid(
        np.linspace(
            *_deposition_range(x_values, floor_at_zero=x_name in DEPOSITION_CHANNELS), n_grid
        ),
        np.linspace(
            *_deposition_range(y_values, floor_at_zero=y_name in DEPOSITION_CHANNELS), n_grid
        ),
    )
    return x_values, y_values, x_grid_values, y_grid_values


def _modifier_over_draws(
    config: NutritionModifierConfig,
    weight_draws: jnp.ndarray,
    channels: dict[str, Any],
    input_mean: jnp.ndarray,
    input_std: jnp.ndarray,
) -> np.ndarray:
    """f_learned of each weight draw at the input values in `channels`, draws first."""
    prepared = prepare_modifier_inputs(channels, config.input_vars, input_mean, input_std)
    return np.asarray(
        jax.vmap(
            lambda weights, x: config.modifier_fn(weights, x, config.input_vars),
            in_axes=(0, None),
        )(weight_draws, prepared)
    )


def plot_modifier_response_surface(
    config: NutritionModifierConfig,
    plot_ids: list[str],
    fit_results: list[NutritionModifierFitResult],
    n_grid: int = 100,
    save_path: str | None = None,
    show: bool = True,
    pre_tanh: bool = False,
) -> None:
    """Plot the fitted modifier over a grid of its two inputs, with each plot's monthly inputs.

    The axes are `config.input_vars` in order (e.g. N and S deposition, or N_avg and
    T_avg). The grid covers the 1st-99th percentile of the pooled input values, padded
    by 20% to show how the modifier behaves just beyond the data; the few extreme
    months outside it are not drawn. With several `fit_results` (e.g. cross-validation
    folds), their mean modifier is drawn, next to its standard deviation across fits.
    With `pre_tanh`, the score before the tanh squash is drawn instead, centered on 0 (no
    effect); only for `poly_nm`, whose modifier is 1 + tanh of its polynomial.
    """
    x_name, y_name = config.input_vars
    x_values, y_values, x_grid_values, y_grid_values = _input_grid(config, n_grid)
    channels = {x_name: jnp.asarray(x_grid_values), y_name: jnp.asarray(y_grid_values)}
    modifiers = np.stack(
        [
            np.asarray(
                config.modifier_fn(
                    fit_result.fitted_modifier_params,
                    prepare_modifier_inputs(
                        channels, config.input_vars, fit_result.input_mean, fit_result.input_std
                    ),
                    config.input_vars,
                )
            )
            for fit_result in fit_results
        ]
    )
    no_effect, value_label = 1.0, "Nutrition modifier (1 = no effect)"
    title = "modifier response surface"
    if pre_tanh:
        # poly_nm is 1 + tanh(score) (see `extended_helper._squash`); invert it,
        # clipped so a saturated modifier gives a large finite score
        modifiers = np.arctanh(np.clip(modifiers - 1.0, -1.0 + 1e-6, 1.0 - 1e-6))
        no_effect, value_label = 0.0, "Score before tanh (0 = no effect)"
        title = "modifier score before tanh"
    modifier = modifiers.mean(axis=0)

    # Diverging colors centered on no effect, symmetric around it
    spread = max(float(np.abs(modifier - no_effect).max()), 1e-6)
    n_panels = 1 if len(fit_results) == 1 else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(8 * n_panels, 6), layout="constrained")
    ax = axes if n_panels == 1 else axes[0]
    surface = ax.contourf(
        x_grid_values,
        y_grid_values,
        modifier,
        levels=20,
        cmap="PuOr",
        vmin=no_effect - spread,
        vmax=no_effect + spread,
    )
    contours = ax.contour(
        x_grid_values, y_grid_values, modifier, levels=10, colors="black", linewidths=0.4
    )
    ax.clabel(contours, fontsize=7)
    # The "no effect" line only exists if the modifier crosses it on the grid
    if modifier.min() < no_effect < modifier.max():
        ax.contour(
            x_grid_values,
            y_grid_values,
            modifier,
            levels=[no_effect],
            colors="black",
            linewidths=1.5,
        )
    fig.colorbar(surface, ax=ax, label=value_label)

    for idx, (plot_id, x, y) in enumerate(zip(plot_ids, x_values, y_values, strict=True)):
        ax.scatter(
            x,
            y,
            s=12,
            color=_PLOT_COLORS[idx % len(_PLOT_COLORS)],
            edgecolor="black",
            linewidth=0.3,
            label=plot_id,
        )
    ax.set_xlim(x_grid_values.min(), x_grid_values.max())
    ax.set_ylim(y_grid_values.min(), y_grid_values.max())

    ax.set_xlabel(_input_label(x_name))
    ax.set_ylabel(_input_label(y_name))
    ax.set_title(f"Fitted nutrition {title}")
    fig.legend(loc="outside right upper", fontsize=7, title="Plot")

    if n_panels == 2:
        std_ax = axes[1]
        std_surface = std_ax.contourf(
            x_grid_values, y_grid_values, modifiers.std(axis=0), levels=20, cmap="Greys"
        )
        fig.colorbar(std_surface, ax=std_ax, label="Standard deviation across fits")
        std_ax.set_xlabel(_input_label(x_name))
        std_ax.set_ylabel(_input_label(y_name))
        std_ax.set_title(f"Spread across {len(fit_results)} fits")
        ax.set_title(f"Mean nutrition {title} ({len(fit_results)} fits)")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()


def plot_modifier_response_curve(
    config: NutritionModifierConfig,
    plot_ids: list[str],
    fit_results: list[NutritionModifierFitResult],
    n_grid: int = 200,
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """Plot the fitted modifier of a single deposition input, with each plot's monthly inputs.

    The curve covers the 1st-99th percentile of the pooled deposition values, padded by
    20%. With several `fit_results` (e.g. cross-validation folds), their mean is drawn
    with the min-max range across fits; each plot's monthly inputs are a rug below it.
    """
    (name,) = config.input_vars
    values = []
    for file_path in config.file_paths:
        deposition = prepare_data(file_path).deposition
        if deposition is None:
            raise ValueError(f"No deposition data in {file_path}")
        values.append(np.asarray(getattr(deposition, DEPOSITION_CHANNELS[name])))
    grid = np.linspace(*_deposition_range(values), n_grid)
    modifiers = np.stack(
        [
            np.asarray(
                config.modifier_fn(
                    fit_result.fitted_modifier_params,
                    prepare_modifier_inputs(
                        {name: jnp.asarray(grid)},
                        config.input_vars,
                        fit_result.input_mean,
                        fit_result.input_std,
                    ),
                    config.input_vars,
                )
            )
            for fit_result in fit_results
        ]
    )

    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    if len(fit_results) > 1:
        ax.fill_between(
            grid,
            modifiers.min(axis=0),
            modifiers.max(axis=0),
            color="tab:blue",
            alpha=0.25,
            label=f"Range across {len(fit_results)} fits",
        )
    ax.plot(grid, modifiers.mean(axis=0), color="tab:blue", linewidth=2.0, label="Mean modifier")
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle=":")
    # One rug row per plot, below the modifier's (0, 2) range
    for idx, (plot_id, deposition) in enumerate(zip(plot_ids, values, strict=True)):
        ax.scatter(
            deposition,
            np.full_like(deposition, -0.05 - 0.04 * idx),
            s=6,
            marker="|",
            color=_PLOT_COLORS[idx % len(_PLOT_COLORS)],
            label=plot_id,
        )
    ax.set_xlim(grid[0], grid[-1])
    ax.set_ylim(-0.1 - 0.04 * len(plot_ids), 2.05)
    ax.set_xlabel(f"{name} deposition (kg ha⁻¹ month⁻¹)")
    ax.set_ylabel("Nutrition modifier (1 = no effect)")
    ax.set_title("Fitted nutrition modifier response (rugs: monthly inputs per plot)")
    ax.grid(alpha=0.3)
    fig.legend(loc="outside right upper", fontsize=7)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()


def plot_modifier_posterior_curves(
    config: NutritionModifierConfig,
    plot_ids: list[str],
    weight_draws: jnp.ndarray,
    input_mean: jnp.ndarray,
    input_std: jnp.ndarray,
    other_percentiles: tuple[float, ...] = (10, 50, 90),
    n_grid: int = 200,
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """Plot f_learned against each modifier input, with its 95% interval over weight draws.

    One panel per input in `config.input_vars`: that input spans the 1st-99th percentile
    of the pooled monthly values, padded by 20%, while the others are held at each of
    `other_percentiles` of their pooled values, one band per level. Each band spans the
    2.5th to 97.5th percentiles of f_learned over the draws; each plot's monthly inputs
    are a rug below it.

    Parameters
    ----------
    config : NutritionModifierConfig
        Modifier function and inputs; its `file_paths` give the plotted inputs.
    plot_ids : list[str]
        Plot labels, in `config.file_paths` order.
    weight_draws : jnp.ndarray
        Posterior draws of the modifier weights, draws along the first axis.
    input_mean, input_std : jnp.ndarray
        Standardisation of the modifier inputs.
    other_percentiles : tuple[float, ...]
        Percentiles of the pooled values at which the other inputs are held.
    n_grid : int
        Number of points along each input.
    save_path : str | None
        File to save the figure to.
    show : bool
        Whether to show the figure.
    """
    names = config.input_vars
    inputs = _load_deposition(config.file_paths, names)
    pooled = {name: np.concatenate(v) for name, v in zip(names, inputs, strict=True)}
    fig, axes = plt.subplots(
        1, len(names), figsize=(4 + 5 * len(names), 5.5), layout="constrained", squeeze=False
    )
    # A single-input modifier has no other input to vary, so one level suffices
    levels = other_percentiles if len(names) > 1 else (50,)
    level_colors = plt.colormaps["viridis"](np.linspace(0.1, 0.85, len(levels)))

    for ax, values, name in zip(axes[0], inputs, names, strict=True):
        grid = np.linspace(
            *_deposition_range(values, floor_at_zero=name in DEPOSITION_CHANNELS), n_grid
        )
        level_handles = []
        for percentile, color in zip(levels, level_colors, strict=True):
            held = {other: float(np.nanpercentile(pooled[other], percentile)) for other in names}
            channels = {other: jnp.full(n_grid, value) for other, value in held.items()}
            channels[name] = jnp.asarray(grid)
            modifiers = _modifier_over_draws(config, weight_draws, channels, input_mean, input_std)
            ax.fill_between(
                grid,
                np.percentile(modifiers, 2.5, axis=0),
                np.percentile(modifiers, 97.5, axis=0),
                color=color,
                alpha=0.2,
            )
            others = ", ".join(f"{o} = {v:.2f}" for o, v in held.items() if o != name)
            (line,) = ax.plot(
                grid,
                modifiers.mean(axis=0),
                color=color,
                linewidth=2.0,
                label=f"p{percentile:g}: {others}" if others else "Mean",
            )
            level_handles.append(line)
        ax.axhline(1.0, color="black", linewidth=0.8, linestyle=":")
        # One rug row per plot, below the modifier's (0, 2) range
        for idx, (plot_id, plot_values) in enumerate(zip(plot_ids, values, strict=True)):
            ax.scatter(
                plot_values,
                np.full_like(plot_values, -0.05 - 0.04 * idx),
                s=6,
                marker="|",
                color=_PLOT_COLORS[idx % len(_PLOT_COLORS)],
                label=plot_id,
            )
        ax.set_xlim(grid[0], grid[-1])
        ax.set_ylim(-0.1 - 0.04 * len(plot_ids), 2.05)
        ax.set_xlabel(_input_label(name))
        ax.legend(handles=level_handles, fontsize=7, title="Other inputs held at")
        ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("f_learned (1 = no effect)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    n_levels = len(levels)
    fig.legend(
        handles[n_levels:], labels[n_levels:], loc="outside right upper", fontsize=7, title="Plot"
    )
    fig.suptitle(
        f"Posterior f_learned: mean and 95% interval over {len(weight_draws)} weight draws "
        "(rugs: monthly inputs per plot)"
    )

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()


def plot_modifier_posterior_at_observations(
    config: NutritionModifierConfig,
    plot_ids: list[str],
    weight_draws: jnp.ndarray,
    input_mean: jnp.ndarray,
    input_std: jnp.ndarray,
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """Plot f_learned at each plot's monthly inputs, with its 95% interval over weight draws.

    One row per plot and one column per input in `config.input_vars`, against which each
    month's posterior mean is drawn with an error bar from the 2.5th to the 97.5th
    percentile over the draws. Only input combinations that occur in the data are shown.

    Parameters
    ----------
    config : NutritionModifierConfig
        Modifier function and inputs; its `file_paths` give the plotted inputs.
    plot_ids : list[str]
        Plot labels, in `config.file_paths` order.
    weight_draws : jnp.ndarray
        Posterior draws of the modifier weights, draws along the first axis.
    input_mean, input_std : jnp.ndarray
        Standardisation of the modifier inputs.
    save_path : str | None
        File to save the figure to.
    show : bool
        Whether to show the figure.
    """
    names = config.input_vars
    inputs = _load_deposition(config.file_paths, names)
    bands = []
    for idx in range(len(plot_ids)):
        plot_inputs = {
            name: jnp.asarray(values[idx]) for name, values in zip(names, inputs, strict=True)
        }
        modifiers = _modifier_over_draws(config, weight_draws, plot_inputs, input_mean, input_std)
        lower, upper = np.percentile(modifiers, [2.5, 97.5], axis=0)
        bands.append((modifiers.mean(axis=0), lower, upper))
    _posterior_errorbars_by_plot(
        config,
        plot_ids,
        bands,
        y_label="f_learned",
        no_effect=1.0,
        title=(
            f"Posterior f_learned (1 = no effect) at observed monthly inputs: mean and 95% "
            f"interval over {len(weight_draws)} weight draws"
        ),
        ylim=(0.0, 2.0),
        save_path=save_path,
        show=show,
    )


def plot_posterior_residual_vs_deposition(
    config: NutritionModifierConfig,
    plot_ids: list[str],
    posterior_series: list[dict[str, np.ndarray]],
    var_name: str = "alpha_c",
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """Plot each plot's monthly `var_name` with minus without the modifier, with its 95% interval.

    One row per plot and one column per input in `config.input_vars`, against which each
    month's posterior mean residual is drawn with an error bar from the 2.5th to the
    97.5th percentile over the draws. Months without photosynthesis are left out.

    Parameters
    ----------
    config : NutritionModifierConfig
        Modifier inputs; its `file_paths` give the plotted inputs.
    plot_ids : list[str]
        Plot labels, in `config.file_paths` order.
    posterior_series : list[dict[str, np.ndarray]]
        One `pymc_nutrition_modifier.predict_posterior_bands` result per plot.
    var_name : str
        `"alpha_c"` or `"NPP"`.
    save_path : str | None
        File to save the figure to.
    show : bool
        Whether to show the figure.
    """
    bands = [
        (
            series[f"mean_residual_{var_name}"],
            series[f"lower_residual_{var_name}"],
            series[f"upper_residual_{var_name}"],
        )
        for series in posterior_series
    ]
    _posterior_errorbars_by_plot(
        config,
        plot_ids,
        bands,
        y_label=f"Δ {_METRIC_LABELS[var_name]}",
        no_effect=0.0,
        title=(
            f"Posterior monthly {var_name} with − without modifier at observed monthly inputs: "
            "mean and 95% interval over the draws"
        ),
        save_path=save_path,
        show=show,
    )


def _posterior_errorbars_by_plot(
    config: NutritionModifierConfig,
    plot_ids: list[str],
    bands: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    y_label: str,
    no_effect: float,
    title: str,
    ylim: tuple[float, float] | None = None,
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """Draw each plot's monthly posterior means with 95% error bars against each input.

    One row per plot and one column per input in `config.input_vars`. `bands` holds each
    plot's monthly (mean, lower, upper), in `plot_ids` order; NaN months are skipped. A
    dotted line marks `no_effect`.
    """
    names = config.input_vars
    inputs = _load_deposition(config.file_paths, names)
    fig, axes = plt.subplots(
        len(plot_ids),
        len(names),
        figsize=(1 + 5 * len(names), 1 + 2.5 * len(plot_ids)),
        layout="constrained",
        sharex="col",
        sharey=True,
        squeeze=False,
    )

    for idx, (row, plot_id, (mean, lower, upper)) in enumerate(
        zip(axes, plot_ids, bands, strict=True)
    ):
        for ax, values in zip(row, inputs, strict=True):
            ax.errorbar(
                values[idx],
                mean,
                yerr=[mean - lower, upper - mean],
                fmt="o",
                markersize=2,
                elinewidth=0.5,
                alpha=0.4,
                color=_PLOT_COLORS[idx % len(_PLOT_COLORS)],
            )
            ax.axhline(no_effect, color="black", linewidth=0.8, linestyle=":")
            ax.grid(alpha=0.3)
        row[0].set_ylabel(f"{plot_id}\n{y_label}")

    for ax, values, name in zip(axes[-1], inputs, names, strict=True):
        ax.set_xlim(*_deposition_range(values, floor_at_zero=name in DEPOSITION_CHANNELS))
        ax.set_xlabel(_input_label(name))
    if ylim is not None:
        axes[0, 0].set_ylim(*ylim)
    fig.suptitle(title)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()


def plot_modifier_interval_width_surface(
    config: NutritionModifierConfig,
    plot_ids: list[str],
    weight_draws: jnp.ndarray,
    input_mean: jnp.ndarray,
    input_std: jnp.ndarray,
    n_grid: int = 100,
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """Plot the width of f_learned's 95% interval over weight draws, on a grid of its two inputs.

    The width is the 97.5th minus the 2.5th percentile over the draws; the grid is that of
    `plot_modifier_response_surface`, with each plot's monthly inputs on top.

    Parameters
    ----------
    config : NutritionModifierConfig
        Modifier function and its two inputs; its `file_paths` give the plotted inputs.
    plot_ids : list[str]
        Plot labels, in `config.file_paths` order.
    weight_draws : jnp.ndarray
        Posterior draws of the modifier weights, draws along the first axis.
    input_mean, input_std : jnp.ndarray
        Standardisation of the modifier inputs.
    n_grid : int
        Number of points along each input.
    save_path : str | None
        File to save the figure to.
    show : bool
        Whether to show the figure.
    """
    x_name, y_name = config.input_vars
    x_values, y_values, x_grid_values, y_grid_values = _input_grid(config, n_grid)
    channels = {x_name: jnp.asarray(x_grid_values), y_name: jnp.asarray(y_grid_values)}
    modifiers = _modifier_over_draws(config, weight_draws, channels, input_mean, input_std)
    lower, upper = np.percentile(modifiers, [2.5, 97.5], axis=0)

    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    surface = ax.contourf(x_grid_values, y_grid_values, upper - lower, levels=20, cmap="Greys")
    fig.colorbar(surface, ax=ax, label="Width of the 95% interval of f_learned")
    for idx, (plot_id, x, y) in enumerate(zip(plot_ids, x_values, y_values, strict=True)):
        ax.scatter(
            x,
            y,
            s=12,
            color=_PLOT_COLORS[idx % len(_PLOT_COLORS)],
            edgecolor="black",
            linewidth=0.3,
            label=plot_id,
        )
    ax.set_xlim(x_grid_values.min(), x_grid_values.max())
    ax.set_ylim(y_grid_values.min(), y_grid_values.max())
    ax.set_xlabel(_input_label(x_name))
    ax.set_ylabel(_input_label(y_name))
    ax.set_title(f"Posterior uncertainty of f_learned ({len(weight_draws)} weight draws)")
    fig.legend(loc="outside right upper", fontsize=7, title="Plot")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()


def _binned_median(
    x: np.ndarray, y: np.ndarray, n_bins: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """Median of `x` and `y` within `n_bins` equal-count bins of `x`, for a trend line.

    Pairs with a missing value are left out.
    """
    keep = ~(np.isnan(x) | np.isnan(y))
    x, y = x[keep], y[keep]
    if not x.size:
        return x, y
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


def plot_modifier_pass_through(
    plot_ids: list[str],
    predicted_series: list[dict[str, np.ndarray]],
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """Plot alpha_c and NPP with and without the modifier against the learned modifier.

    Each panel shows the monthly values of all plots pooled, one color per run, with a
    line through their binned medians: the gap between the two lines is the modifier's
    effect. Months without photosynthesis in the run without the modifier are left out.

    Parameters
    ----------
    plot_ids : list[str]
        Plot labels, in `predicted_series` order.
    predicted_series : list[dict[str, np.ndarray]]
        One `build_predicted_series` result per plot.
    save_path : str | None
        File to save the figure to.
    show : bool
        Whether to show the figure.
    """
    var_names = ("alpha_c", "NPP")
    runs = {"fitted": "With nutrition modifier", "phys": "Without nutrition modifier"}
    fig, axes = plt.subplots(
        1, len(var_names), figsize=(6 * len(var_names), 5.5), layout="constrained"
    )
    # Growing months of every plot, pooled
    active = [series["pred_phys_alpha_c"] > 0 for series in predicted_series]
    modifier = np.concatenate(
        [
            series["learned_modifier"][mask]
            for series, mask in zip(predicted_series, active, strict=True)
        ]
    )
    for ax, var_name in zip(axes, var_names, strict=True):
        for run, label in runs.items():
            values = np.concatenate(
                [
                    series[f"pred_{run}_{var_name}"][mask]
                    for series, mask in zip(predicted_series, active, strict=True)
                ]
            )
            color = _RUN_STYLES[run]["color"]
            ax.scatter(modifier, values, s=6, alpha=0.2, color=color)
            ax.plot(
                *_binned_median(modifier, values, n_bins=20),
                linewidth=2.0,
                label=label,
                **_RUN_STYLES[run],
            )
        ax.set_xlabel("Learned modifier (–)")
        ax.set_ylabel(_METRIC_LABELS[var_name])
        ax.set_title(f"{var_name} with and without the modifier ({len(plot_ids)} plots)")
        ax.grid(alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncols=2)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()


def plot_learned_modifier_vs_deposition(
    config: NutritionModifierConfig,
    predicted_series: list[dict[str, np.ndarray]],
    plot_ids: list[str],
    save_path: str | None = None,
    show: bool = True,
    title_note: str = "",
) -> None:
    """Plot each plot's monthly f_learned and f_nutri_classic_learnable against each input.

    Rows are the learned modifier (f_learned, 1 = no effect) and its product with the
    classic fN (f_nutri_classic_learnable), columns the modifier inputs in
    `config.input_vars`. One color per plot, with a per-plot line through binned medians.
    `title_note`, if given, is added as a second line of the title.
    """
    names = config.input_vars
    inputs = _load_deposition(config.file_paths, names)
    fig, axes = plt.subplots(
        2,
        len(names),
        figsize=(4 + 4.5 * len(names), 9),
        layout="constrained",
        sharex="col",
        sharey=True,
        squeeze=False,
    )

    for row, (key, y_label) in enumerate(
        (
            ("learned_modifier", "f_learned (1 = no effect)"),
            ("pred_fitted_f_nutri_classic_learnable", _METRIC_LABELS["f_nutri_classic_learnable"]),
        )
    ):
        y_values = [series[key] for series in predicted_series]
        for col, (values, name) in enumerate(zip(inputs, names, strict=True)):
            ax = axes[row, col]
            _scatter_by_plot(ax, plot_ids, values, y_values)
            ax.axhline(1.0, color="black", linewidth=0.8, linestyle=":")
            ax.set_xlim(*_deposition_range(values, floor_at_zero=name in DEPOSITION_CHANNELS))
            input_label = _input_label(name)
            if row == 0:
                ax.set_title(input_label.split(" (")[0])
            else:
                ax.set_xlabel(input_label)
            if col == 0:
                ax.set_ylabel(y_label)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside right upper", fontsize=7, title="Plot")
    title = (
        "Nutrition modifiers against modifier inputs "
        "(solid: per-plot binned median, dashed: all plots)"
    )
    fig.suptitle(f"{title}\n{title_note}" if title_note else title)

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
    """Plot each plot's monthly `var_name`, with and without the modifier, against deposition.

    Rows are the runs with and without the modifier, columns the deposition inputs of the
    modifier (see `_modifier_deposition`), all on a shared y-axis. One color per plot, with
    a per-plot line through binned medians. Months with alpha_c = 0 without the modifier
    (no photosynthesis, e.g. winter) are left out.
    """
    deposition = _modifier_deposition(config)
    fig, axes = plt.subplots(
        2,
        len(deposition),
        figsize=(4 + 4.5 * len(deposition), 9),
        layout="constrained",
        sharex="col",
        sharey=True,
        squeeze=False,
    )

    run_labels = _run_labels(config)
    for row, (run_label, key) in enumerate(
        ((run_labels["fitted"], "fitted"), (run_labels["phys"], "phys"))
    ):
        for col, (values, dep_label) in enumerate(deposition):
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


def plot_modifier_residual_vs_deposition(
    config: NutritionModifierConfig,
    predicted_series: list[dict[str, np.ndarray]],
    plot_ids: list[str],
    var_name: str = "alpha_c",
    save_path: str | None = None,
    show: bool = True,
    title_note: str = "",
) -> None:
    """Plot each plot's monthly `var_name` with minus without the modifier against deposition.

    One column per deposition input of the modifier (see `_modifier_deposition`), on a
    shared y-axis. One color per plot, with a per-plot line through binned medians. Months
    with alpha_c = 0 without the modifier (no photosynthesis, e.g. winter) are left out.
    `title_note`, if given, is added as a second line of the title.
    """
    deposition = _modifier_deposition(config)
    fig, axes = plt.subplots(
        1,
        len(deposition),
        figsize=(4 + 4.5 * len(deposition), 5),
        layout="constrained",
        sharey=True,
        squeeze=False,
    )

    active = [series["pred_phys_alpha_c"] > 0 for series in predicted_series]
    residuals = [
        (series[f"pred_fitted_{var_name}"] - series[f"pred_phys_{var_name}"])[mask]
        for series, mask in zip(predicted_series, active, strict=True)
    ]
    for col, (values, dep_label) in enumerate(deposition):
        ax = axes[0, col]
        _scatter_by_plot(
            ax, plot_ids, [dep[mask] for dep, mask in zip(values, active, strict=True)], residuals
        )
        ax.axhline(0.0, color="black", linewidth=0.8, linestyle=":")
        ax.set_xlim(*_deposition_range(values))
        ax.set_xlabel(dep_label)
    axes[0, 0].set_ylabel(f"{_METRIC_LABELS[var_name]}\nwith − without modifier")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside right upper", fontsize=7, title="Plot")
    title = (
        f"Monthly {var_name} with − without modifier against deposition "
        "(solid: per-plot binned median, dashed: all plots)"
    )
    fig.suptitle(f"{title}\n{title_note}" if title_note else title)

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
            _scatter_observed(ax, config, observed_data, var_name)
            if f"dbh_derived_{var_name}" in series:
                ax.scatter(
                    series["dbh_derived_dates"],
                    series[f"dbh_derived_{var_name}"],
                    label="Derived from observed DBH (3-PG allometry)",
                    color="black",
                    marker="*",
                    s=40,
                    zorder=6,
                )
            _format_date_axis(ax)
            if row == 0:
                ax.set_title(_METRIC_LABELS.get(var_name, var_name))
            if col == 0:
                ax.set_ylabel(plot_id)

    _figure_legend(fig, axes)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()


def _scatter_observed(
    ax: Any, config: NutritionModifierConfig, observed_data: pl.DataFrame, var_name: str
) -> None:
    """Scatter a plot's observations of `var_name`, red for targets and gray otherwise."""
    if var_name not in observed_data.columns:
        return
    observed_dates = np.array(
        [
            np.datetime64(f"{year}-{month:02d}", "M")
            for year, month in zip(observed_data["year"], observed_data["month"], strict=True)
        ]
    )
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


def _figure_legend(fig: Any, axes: np.ndarray) -> None:
    """Add one legend above the figure with the entries of every column of the first row.

    Target and non-target observations sit in different columns, so every column is
    gathered, keeping the first handle per label.
    """
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


def plot_posterior_predictions(
    config: NutritionModifierConfig,
    plot_ids: list[str],
    posterior_series: list[dict[str, np.ndarray]],
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """Plot every plot's posterior prediction interval, default run and observations.

    Rows are plots (`config.file_paths`, labelled by `plot_ids`), columns are the plot
    variables (`config.plot_variables`, else `config.target_vars`). Each panel shows the
    posterior mean and 95% interval, the run with default parameters and no nutrition
    modifier, and the observations.

    Parameters
    ----------
    config : NutritionModifierConfig
        Plots and variables.
    plot_ids : list[str]
        Plot labels, in `config.file_paths` order.
    posterior_series : list[dict[str, np.ndarray]]
        One `pymc_nutrition_modifier.predict_posterior_bands` result per plot.
    save_path : str | None
        File to save the figure to.
    show : bool
        Whether to show the figure.
    """
    plot_variables = _plot_variables(config)
    n_rows, n_cols = len(plot_ids), len(plot_variables)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.5 * n_cols, 2.5 * n_rows),
        layout="constrained",
        squeeze=False,
    )

    for row, (plot_id, file_path, series) in enumerate(
        zip(plot_ids, config.file_paths, posterior_series, strict=True)
    ):
        observed_data = pl.read_excel(file_path, sheet_name=config.observed_sheet)
        dates = series["dates"]
        for col, var_name in enumerate(plot_variables):
            ax = axes[row, col]
            ax.fill_between(
                dates,
                series[f"lower_{var_name}"],
                series[f"upper_{var_name}"],
                color="tab:blue",
                alpha=0.3,
                label="95% interval (posterior)",
            )
            ax.plot(
                dates,
                series[f"mean_{var_name}"],
                color="tab:blue",
                linewidth=1.5,
                label="Posterior mean",
            )
            ax.plot(
                dates,
                series[f"default_{var_name}"],
                label="Default physiological parameters, without nutrition modifier",
                linewidth=1.5,
                **_RUN_STYLES["default"],
            )
            _scatter_observed(ax, config, observed_data, var_name)
            _format_date_axis(ax)
            if row == 0:
                ax.set_title(_METRIC_LABELS.get(var_name, var_name))
            if col == 0:
                ax.set_ylabel(plot_id)

    _figure_legend(fig, axes)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
