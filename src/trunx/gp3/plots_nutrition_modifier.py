"""Plots of the learnable nutrition modifier's training and fitted behaviour."""

from pathlib import Path
from typing import Any, Literal

import jax
import jax.numpy as jnp
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from trunx.gp3.extended_helper import prepare_modifier_inputs
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

_RUN_STYLES: dict[str, dict[str, str]] = {
    "fitted": {"color": "tab:blue", "linestyle": "-"},
    "phys": {"color": "tab:orange", "linestyle": "-."},
    "default": {"color": "tab:green", "linestyle": "--"},
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


def plot_modifier_response_surface(
    config: NutritionModifierConfig,
    plot_ids: list[str],
    fit_results: list[NutritionModifierFitResult],
    n_grid: int = 100,
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """Plot the fitted modifier over an (N, S) deposition grid, with each plot's monthly inputs.

    The grid covers the 1st-99th percentile of the pooled deposition values, padded
    by 20% to show how the modifier behaves just beyond the data; the few extreme
    months outside it are not drawn. With several `fit_results` (e.g. cross-validation
    folds), their mean modifier is drawn, next to its standard deviation across fits.
    """
    if set(config.input_vars) != {"N", "S"}:
        raise ValueError(f"Needs a modifier over exactly N and S, got {config.input_vars}")

    n_values, s_values = _load_deposition(config.file_paths)
    n_grid_values, s_grid_values = np.meshgrid(
        np.linspace(*_deposition_range(n_values), n_grid),
        np.linspace(*_deposition_range(s_values), n_grid),
    )
    channels = {"N": jnp.asarray(n_grid_values), "S": jnp.asarray(s_grid_values)}
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
    modifier = modifiers.mean(axis=0)

    # Diverging colors centered on 1 (no effect), symmetric around it
    spread = max(float(np.abs(modifier - 1.0).max()), 1e-6)
    n_panels = 1 if len(fit_results) == 1 else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(8 * n_panels, 6), layout="constrained")
    ax = axes if n_panels == 1 else axes[0]
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

    if n_panels == 2:
        std_ax = axes[1]
        std_surface = std_ax.contourf(
            n_grid_values, s_grid_values, modifiers.std(axis=0), levels=20, cmap="Greys"
        )
        fig.colorbar(std_surface, ax=std_ax, label="Standard deviation across fits")
        std_ax.set_xlabel("N deposition (kg ha⁻¹ month⁻¹)")
        std_ax.set_ylabel("S deposition (kg ha⁻¹ month⁻¹)")
        std_ax.set_title(f"Spread of the modifier across {len(fit_results)} fits")
        ax.set_title(f"Mean nutrition modifier response surface ({len(fit_results)} fits)")

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
