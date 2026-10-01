"""Visualize a saved `pymc_param_est_multiplots.py` run against its plots' observations.

`run_pymc_multi_plot_analysis` fits one shared set of physiology parameters across
many plots at once and only saves the posterior trace (see `save_results`) — no
per-plot predicted time series like the single-plot pipeline's `predictions.npz`.
`plot_multiplot_predictions` re-runs the pooled 3PG forward pass at the posterior
mean to get those predictions back, for checking how well one shared parameter set
explains every plot it was fit on.
"""

import glob
import os
from typing import Any, cast

import arviz as az
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import root_mean_squared_error as rmse

from trunx.config import results_data_folder, threepg_data_folder
from trunx.gp3.bayesiancalibrations.bayesian_comparison_plots import (
    plot_convergence_comparison,
    plot_parameter_value_comparison,
    plot_posterior_comparison,
    plot_trace_and_posterior,
)
from trunx.gp3.bayesiancalibrations.bayesian_config import INITIAL_STATE_PARAMS, species_plot_ids
from trunx.gp3.bayesiancalibrations.jax_bayesian_param_est_multiplots import (
    load_and_pack_plots,
    run_packed_plots_forward,
)
from trunx.gp3.bayesiancalibrations.load_files import (
    load_plot_ids_from_file,
    load_priors_from_file,
)
from trunx.gp3.model_inputs import Params

PLOT_VARIABLES = ["BA", "DBH", "Height", "WS", "WR", "WF"]


def plot_multiplot_predictions(
    output_dir: str,
    plot_file: str,
    params_file: str,
    plot_ids: list[str] | None = None,
    plot_variables: list[str] = PLOT_VARIABLES,
    max_plots: int | None = None,
) -> tuple[Figure, pd.DataFrame]:
    """Plot predicted vs. observed growth curves for every plot in a saved multiplot run.

    Loads the posterior mean of each fitted physiology parameter from a saved
    `run_pymc_multi_plot_analysis` run, re-runs the pooled 3PG forward pass with
    those shared values, and plots the result against each plot's own
    observations — one row of subplots per plot (one column per
    `plot_variables` entry), so a single shared parameter set can be checked
    against every plot it was fit on at once.

    Parameters
    ----------
    output_dir : str
        Directory holding the saved `inference_data.nc` from
        `run_pymc_multi_plot_analysis`.
    plot_file : str
        Parquet file of packed plot data (see `prepare_multiplots_data.py`),
        the same one the saved run was calibrated on.
    params_file : str
        Shared physiology parameter bounds parquet, the same one the saved
        run used.
    plot_ids : list[str] | None
        Plot IDs to include. Defaults to every plot in `plot_file`.
    plot_variables : list[str]
        Output variables to plot per plot.
    max_plots : int | None
        Limit to this many plots (e.g. for a quick check on a large batch).

    Returns
    -------
    tuple[Figure, pd.DataFrame]
        The comparison figure and a per-(plot, variable) RMSE table (a
        variable with no observations for a given plot has no row).
    """
    if plot_ids is None:
        plot_ids = load_plot_ids_from_file(plot_file)
    if max_plots is not None:
        plot_ids = plot_ids[:max_plots]

    packed_plots, fixed_params = load_and_pack_plots(
        params_file, [(plot_file, plot_id) for plot_id in plot_ids]
    )

    idata = az.from_netcdf(os.path.join(output_dir, "inference_data.nc"))
    posterior = cast(Any, idata).posterior
    fitted_means = {
        name: float(posterior[name].mean())
        for name in Params._fields
        if name in posterior.data_vars
    }
    fitted_params = fixed_params._replace(
        **{
            name: jnp.full_like(getattr(fixed_params, name), value)
            for name, value in fitted_means.items()
        }
    )

    outputs = run_packed_plots_forward(packed_plots, fitted_params)

    n_plots = packed_plots.n_plots
    n_cols = len(plot_variables)
    fig, axes = plt.subplots(n_plots, n_cols, figsize=(4 * n_cols, 3 * n_plots), squeeze=False)

    rows = []
    for row_idx, plot_id in enumerate(packed_plots.plot_ids):
        n_months = int(packed_plots.climate.lengths[row_idx])
        start_date = pd.Timestamp(
            year=int(packed_plots.site.year_i[row_idx, 0]),
            month=int(packed_plots.site.month_i[row_idx, 0]),
            day=1,
        )
        time_months = pd.date_range(start=start_date, periods=n_months, freq="ME")

        for col_idx, var_name in enumerate(plot_variables):
            ax = axes[row_idx][col_idx]

            if var_name in outputs:
                predicted = np.asarray(outputs[var_name][row_idx, :n_months, 0])
                ax.plot(
                    time_months, predicted, color="tab:blue", label="Predicted (posterior mean)"
                )

            row: dict[str, str | float] = {"plot_id": plot_id, "variable": var_name}
            if var_name in packed_plots.observations:
                obs = packed_plots.observations[var_name]
                obs_mask = np.asarray(obs.mask[row_idx])
                obs_times = np.asarray(obs.times[row_idx])[obs_mask]
                obs_values = np.asarray(obs.values[row_idx, :, 0])[obs_mask]
                if obs_mask.any():
                    ax.scatter(
                        time_months[obs_times], obs_values, color="red", zorder=5, label="Observed"
                    )
                    if var_name in outputs:
                        predicted_at_obs = np.asarray(outputs[var_name][row_idx, obs_times, 0])
                        row["rmse"] = float(rmse(obs_values, predicted_at_obs))
            rows.append(row)

            if row_idx == 0:
                ax.set_title(var_name)
            if col_idx == 0:
                ax.set_ylabel(plot_id)
            ax.grid(alpha=0.3)
            ax.tick_params(axis="x", rotation=45)

    axes[0][0].legend(fontsize=8)
    fig.suptitle("Multiplot predictions vs. observations (posterior mean)")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))

    return fig, pd.DataFrame(rows)


def plot_multiplot_posterior(
    output_dir: str,
    params_file: str,
    param_names: list[str] | None = None,
) -> tuple[Figure, Figure]:
    """Plot the MCMC trace and posterior distributions for a saved multiplot run.

    Thin wrapper around `bayesian_comparison_plots.plot_trace_and_posterior` for
    a `run_pymc_multi_plot_analysis` run's shared physiology/`err_*`/`perr_*`
    parameters.

    Parameters
    ----------
    output_dir : str
        Directory holding the saved `inference_data.nc`.
    params_file : str
        Shared physiology parameter bounds parquet, the same one the saved run
        used — read for prior (lower, upper) bounds to mark on the plot.
    param_names : list[str] | None
        Parameters to plot. Defaults to every posterior variable except
        `INITIAL_STATE_PARAMS` (`WS0`/`WR0`/`WF0` — per-plot latent draws with
        `n_plots` values each, not single scalars like the rest).

    Returns
    -------
    tuple[Figure, Figure]
        The trace figure and the posterior figure.
    """
    inference_data_path = os.path.join(output_dir, "inference_data.nc")

    if param_names is None:
        idata = az.from_netcdf(inference_data_path)
        posterior_vars = set(cast(Any, idata).posterior.data_vars)
        param_names = sorted(posterior_vars - set(INITIAL_STATE_PARAMS))

    priors = load_priors_from_file(
        params_file, [name for name in param_names if name not in INITIAL_STATE_PARAMS]
    )
    return plot_trace_and_posterior(inference_data_path, param_names, priors)


def plot_multiplot_convergence(
    output_dir: str,
    step_method: str,
    param_names: list[str] | None = None,
) -> tuple[Figure, pd.DataFrame]:
    """Plot convergence diagnostics (r_hat, ESS, mean) for one saved multiplot run.

    Thin wrapper around `bayesian_comparison_plots.plot_convergence_comparison`
    for a single `run_pymc_multi_plot_analysis` run — use
    `plot_multiplot_method_comparison` instead to compare two step methods'
    saved runs against each other.

    Parameters
    ----------
    output_dir : str
        Directory holding the saved `inference_data.nc`.
    step_method : str
        Which step method produced this run: `"demetropolisz"` (PyMC) or
        `"nuts"` (HMC) — selects which of `plot_convergence_comparison`'s two
        method slots this run's path is passed as.
    param_names : list[str] | None
        Parameters to summarize. Defaults to every posterior variable except
        `INITIAL_STATE_PARAMS` (`WS0`/`WR0`/`WF0`).

    Returns
    -------
    tuple[Figure, pd.DataFrame]
        The convergence-diagnostics figure and its per-parameter summary table.
    """
    if step_method not in {"demetropolisz", "nuts"}:
        raise ValueError(f"step_method must be 'demetropolisz' or 'nuts', got {step_method!r}")

    inference_data_path = os.path.join(output_dir, "inference_data.nc")

    if param_names is None:
        idata = az.from_netcdf(inference_data_path)
        posterior_vars = set(cast(Any, idata).posterior.data_vars)
        param_names = sorted(posterior_vars - set(INITIAL_STATE_PARAMS))

    is_pymc = step_method == "demetropolisz"
    return plot_convergence_comparison(
        fit_params=param_names,
        pymc_inference_path=inference_data_path if is_pymc else None,
        hmc_inference_path=inference_data_path if not is_pymc else None,
        param_names=param_names,
        include_bayesian=is_pymc,
        include_hmc=not is_pymc,
    )


def plot_multiplot_method_comparison(
    demetropolisz_dir: str,
    nuts_dir: str,
    param_names: list[str] | None = None,
) -> tuple[Figure, pd.DataFrame, Figure, Figure, pd.DataFrame]:
    """Compare DEMetropolisZ vs. NUTS results for one saved multiplot combination.

    Thin wrapper around `bayesian_comparison_plots`' method-comparison plots
    (`plot_convergence_comparison`, `plot_posterior_comparison`,
    `plot_parameter_value_comparison`), with gradient descent/MAP excluded since
    `pymc_param_est_multiplots.py` only supports these two step methods.

    Parameters
    ----------
    demetropolisz_dir, nuts_dir : str
        Directories holding each method's saved `inference_data.nc`, for the
        same (species, literature_source, error_mode) combination.
    param_names : list[str] | None
        Parameters to compare. Defaults to every parameter present in both
        saved traces, excluding `INITIAL_STATE_PARAMS`.

    Returns
    -------
    tuple[Figure, pd.DataFrame, Figure, Figure, pd.DataFrame]
        The convergence-diagnostics figure and table, the posterior-overlay
        figure, and the fitted-value comparison figure and table.
    """
    demetropolisz_path = os.path.join(demetropolisz_dir, "inference_data.nc")
    nuts_path = os.path.join(nuts_dir, "inference_data.nc")

    if param_names is None:
        demetropolisz_vars = set(cast(Any, az.from_netcdf(demetropolisz_path)).posterior.data_vars)
        nuts_vars = set(cast(Any, az.from_netcdf(nuts_path)).posterior.data_vars)
        param_names = sorted((demetropolisz_vars & nuts_vars) - set(INITIAL_STATE_PARAMS))

    convergence_fig, convergence_df = plot_convergence_comparison(
        fit_params=param_names,
        pymc_inference_path=demetropolisz_path,
        hmc_inference_path=nuts_path,
        param_names=param_names,
    )
    posterior_fig = plot_posterior_comparison(
        pymc_inference_path=demetropolisz_path,
        hmc_inference_path=nuts_path,
        param_names=param_names,
    )
    value_fig, value_df = plot_parameter_value_comparison(
        pymc_inference_path=demetropolisz_path,
        hmc_inference_path=nuts_path,
        param_names=param_names,
        include_gradient_descent=False,
        include_map=False,
    )
    return convergence_fig, convergence_df, posterior_fig, value_fig, value_df


def plot_all_saved_multiplot_results(base_dir: str) -> None:
    """Plot every saved (species, literature_source, error_mode, step_method) run under `base_dir`.

    Parameters
    ----------
    base_dir : str
        Root directory to scan, e.g. `<results_data_folder>/pymc_multiplot_results`.
    """
    pattern = os.path.join(base_dir, "*", "*", "*", "*", "inference_data.nc")
    groups: dict[tuple[str, str, str], dict[str, str]] = {}
    for inference_path in sorted(glob.glob(pattern)):
        combo_dir = os.path.dirname(inference_path)
        step_method = os.path.basename(combo_dir)
        error_mode_dir = os.path.dirname(combo_dir)
        error_mode = os.path.basename(error_mode_dir)
        literature_source_dir = os.path.dirname(error_mode_dir)
        literature_source = os.path.basename(literature_source_dir)
        species_slug = os.path.basename(os.path.dirname(literature_source_dir))
        groups.setdefault((species_slug, literature_source, error_mode), {})[step_method] = (
            combo_dir
        )

    for (species_slug, literature_source, error_mode), step_method_dirs in groups.items():
        species_name = species_slug.replace("_", " ")
        plot_file = os.path.join(threepg_data_folder, f"icp_plot_data_{species_slug}.parquet")
        params_file = os.path.join(
            threepg_data_folder, f"params_bounds_{literature_source}_{species_slug}.parquet"
        )
        plot_ids = species_plot_ids.get(species_name)

        for step_method, combo_dir in step_method_dirs.items():
            print(
                f"Plotting {species_name} / {literature_source} / {error_mode} / {step_method}..."
            )
            try:
                fig, rmse_df = plot_multiplot_predictions(
                    output_dir=combo_dir,
                    plot_file=plot_file,
                    params_file=params_file,
                    plot_ids=plot_ids,
                )
                fig.savefig(
                    os.path.join(combo_dir, "predictions.png"), dpi=150, bbox_inches="tight"
                )
                plt.close(fig)
                rmse_df.to_csv(os.path.join(combo_dir, "predictions_rmse.csv"), index=False)

                trace_fig, posterior_fig = plot_multiplot_posterior(combo_dir, params_file)
                trace_fig.savefig(
                    os.path.join(combo_dir, "trace.png"), dpi=150, bbox_inches="tight"
                )
                posterior_fig.savefig(
                    os.path.join(combo_dir, "posterior.png"), dpi=150, bbox_inches="tight"
                )
                plt.close(trace_fig)
                plt.close(posterior_fig)

                convergence_fig, convergence_df = plot_multiplot_convergence(
                    combo_dir, step_method
                )
                convergence_fig.savefig(
                    os.path.join(combo_dir, "convergence.png"), dpi=150, bbox_inches="tight"
                )
                plt.close(convergence_fig)
                convergence_df.to_csv(os.path.join(combo_dir, "convergence.csv"), index=False)
            except Exception as exc:
                print(f"  Error: {exc}")

        if "demetropolisz" in step_method_dirs and "nuts" in step_method_dirs:
            print(f"Comparing methods for {species_name} / {literature_source} / {error_mode}...")
            try:
                comparison_dir = os.path.join(
                    os.path.dirname(step_method_dirs["demetropolisz"]), "plots"
                )
                os.makedirs(comparison_dir, exist_ok=True)
                convergence_fig, convergence_df, posterior_fig, value_fig, value_df = (
                    plot_multiplot_method_comparison(
                        step_method_dirs["demetropolisz"], step_method_dirs["nuts"]
                    )
                )
                convergence_fig.savefig(
                    os.path.join(comparison_dir, "method_convergence_comparison.png"),
                    dpi=150,
                    bbox_inches="tight",
                )
                convergence_df.to_csv(
                    os.path.join(comparison_dir, "method_convergence_comparison.csv"), index=False
                )
                posterior_fig.savefig(
                    os.path.join(comparison_dir, "method_posterior_comparison.png"),
                    dpi=150,
                    bbox_inches="tight",
                )
                value_fig.savefig(
                    os.path.join(comparison_dir, "method_value_comparison.png"),
                    dpi=150,
                    bbox_inches="tight",
                )
                value_df.to_csv(
                    os.path.join(comparison_dir, "method_value_comparison.csv"), index=False
                )
                plt.close(convergence_fig)
                plt.close(posterior_fig)
                plt.close(value_fig)
            except Exception as exc:
                print(f"  Error: {exc}")


if __name__ == "__main__":
    plot_all_saved_multiplot_results(os.path.join(results_data_folder, "pymc_multiplot_results"))
