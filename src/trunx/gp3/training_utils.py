"""Shared helpers for gradient-descent-based 3PG parameter fitting."""

from pathlib import Path
from typing import Any

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

from trunx.gp3.model_inputs import SiteData


def build_observation_indices(observed_data: Any, site_data: SiteData) -> jnp.ndarray:
    """Build observation month indices relative to the simulation start month.

    Parameters
    ----------
    observed_data : Any
        A pandas or polars DataFrame with `year` and `month` columns.
    site_data : SiteData
        Site data whose `year_i`/`month_i` mark the simulation start month.

    Returns
    -------
    jnp.ndarray
        Zero-based month index of each observation relative to simulation start.
    """
    if not {"year", "month"}.issubset(set(observed_data.columns)):
        raise ValueError("Observed sheet must contain year and month columns")

    start_year = int(np.asarray(site_data.year_i).reshape(-1)[0])
    start_month = int(np.asarray(site_data.month_i).reshape(-1)[0])

    year = np.asarray(observed_data["year"]).astype(np.int32)
    month = np.asarray(observed_data["month"]).astype(np.int32)
    idx_values = (year - start_year) * 12 + (month - start_month)
    return jnp.asarray(idx_values, dtype=jnp.int32)


def build_optimizer(
    optimizer_name: str, learning_rate: float, global_clip_norm: float
) -> optax.GradientTransformation:
    """Build a gradient-clipped Optax optimizer.

    Parameters
    ----------
    optimizer_name : str
        Either `"adam"` or `"sgd"`.
    learning_rate : float
        Learning rate for the base optimizer.
    global_clip_norm : float
        Maximum global gradient norm before clipping.

    Returns
    -------
    optax.GradientTransformation
        The chained clip-then-optimize transformation.
    """
    if optimizer_name == "adam":
        base_optimizer = optax.adam(learning_rate=learning_rate)
    elif optimizer_name == "sgd":
        base_optimizer = optax.sgd(learning_rate=learning_rate)
    else:
        raise ValueError("optimizer_name must be either 'adam' or 'sgd'")

    return optax.chain(
        optax.clip_by_global_norm(global_clip_norm),
        base_optimizer,
    )


def weighted_squared_error(
    pg3_outputs: dict[str, jnp.ndarray],
    target_vars: list[str],
    obs_indices: jnp.ndarray,
    obs_values: dict[str, jnp.ndarray],
    obs_scales: dict[str, jnp.ndarray],
    species_index: int,
    variable_weights: dict[str, float],
) -> jnp.ndarray:
    """Sum weighted, NaN-masked squared residuals across target variables.

    Parameters
    ----------
    pg3_outputs : dict[str, jnp.ndarray]
        Simulated 3PG output series, keyed by variable name.
    target_vars : list[str]
        Variables to score.
    obs_indices : jnp.ndarray
        Simulation month index of each observation.
    obs_values : dict[str, jnp.ndarray]
        Observed values, keyed by variable name.
    obs_scales : dict[str, jnp.ndarray]
        Per-variable scale used to normalize residuals.
    species_index : int
        Species column to select when a variable is per-species.
    variable_weights : dict[str, float]
        Per-variable weight applied to its squared error (default 1.0).

    Returns
    -------
    jnp.ndarray
        Total weighted squared error, unnormalized by observation count.
    """
    total_squared_error = jnp.asarray(0.0, dtype=jnp.float32)
    for var_name in target_vars:
        pg3_predictions = pg3_outputs[var_name][obs_indices]
        if pg3_predictions.ndim == 2:
            pg3_predictions = pg3_predictions[:, species_index].reshape(-1)

        observed_values = obs_values[var_name].reshape(-1)
        scale = obs_scales[var_name]
        mask = ~(jnp.isnan(observed_values) | jnp.isnan(pg3_predictions))
        residuals = (pg3_predictions - observed_values) / scale
        squared = jnp.where(mask, residuals**2, 0.0)
        weight = variable_weights.get(var_name, 1.0)
        total_squared_error += weight * jnp.sum(squared)

    return total_squared_error


def save_and_show_figure(fig: plt.Figure, save_path: str | None, show: bool) -> None:
    """Save a figure to disk and/or display it.

    Parameters
    ----------
    fig : plt.Figure
        Figure to save/show.
    save_path : str | None
        Output image path; parent directories are created as needed.
    show : bool
        Whether to display the figure.
    """
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()


def plot_loss_over_iterations(
    loss_history: list[float],
    title: str,
    xlabel: str = "Iteration",
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """Plot optimization loss over iterations.

    Parameters
    ----------
    loss_history : list[float]
        Loss value at each iteration.
    title : str
        Plot title.
    xlabel : str
        X-axis label.
    save_path : str | None
        Optional output image path.
    show : bool
        Whether to display the figure.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(np.arange(len(loss_history)), loss_history, color="tab:blue", linewidth=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    save_and_show_figure(fig, save_path, show)


def plot_traces_grid(
    traces: list[tuple[str, np.ndarray]],
    suptitle: str,
    xlabel: str = "Iteration",
    save_path: str | None = None,
    show: bool = True,
    bounds: dict[str, tuple[float, float]] | None = None,
) -> None:
    """Plot a grid of named scalar traces, one subplot per trace.

    Parameters
    ----------
    traces : list[tuple[str, np.ndarray]]
        `(label, values_over_iterations)` pairs, one per subplot.
    suptitle : str
        Figure-level title.
    xlabel : str
        X-axis label for each subplot.
    save_path : str | None
        Optional output image path.
    show : bool
        Whether to display the figure.
    bounds : dict[str, tuple[float, float]] | None
        Optional `(lower, upper)` bound lines drawn on the matching trace.
    """
    if not traces:
        return

    n_traces = len(traces)
    n_cols = min(3, n_traces)
    n_rows = int(np.ceil(n_traces / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows), layout="constrained"
    )
    axes_list = np.ravel(np.atleast_1d(axes)).tolist()

    iterations = np.arange(len(traces[0][1]))
    for ax, (label, values) in zip(axes_list[:n_traces], traces, strict=True):
        ax.plot(iterations, values, color="tab:blue", linewidth=2, label=label)

        lower, upper = (bounds or {}).get(label, (-np.inf, np.inf))
        if np.isfinite(lower):
            ax.axhline(lower, color="tab:red", linestyle="--", linewidth=1.5, label="min")
        if np.isfinite(upper):
            ax.axhline(upper, color="tab:green", linestyle="--", linewidth=1.5, label="max")
        if np.isfinite(lower) or np.isfinite(upper):
            ax.legend(loc="best")

        ax.set_title(label)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Value")
        ax.grid(alpha=0.3)

    for ax in axes_list[n_traces:]:
        ax.set_visible(False)

    fig.suptitle(suptitle)
    save_and_show_figure(fig, save_path, show)
