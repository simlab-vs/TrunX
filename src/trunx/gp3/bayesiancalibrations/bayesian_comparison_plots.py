"""Compare default, gradient-descent, PyMC-Bayesian, and HMC-Bayesian 3PG predictions.

Plots all four prediction sources against observations for one site, with
per-variable RMSE/MAE printed for comparison.
"""

import os
from typing import Any, cast

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from matplotlib.figure import Figure
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import root_mean_squared_error as rmse

from trunx.config import data_folder, results_data_folder, threepg_data_folder
from trunx.gp3.bayesiancalibrations.load_files import load_param_defaults_from_file
from trunx.gp3.bayesiancalibrations.save_load_results import load_predictions
from trunx.gp3.gradient_descent import (
    GradientDescentConfig,
    apply_fitted_params,
    fit_with_gradient_descent,
)
from trunx.gp3.model_inputs import Params
from trunx.gp3.PG3_model_impl import prepare_data
from trunx.gp3.run_3pg import run_3pg

PLOT_VARIABLES = ["BA", "DBH", "Height", "WF", "WS", "WR"]
LABEL_MAP = {
    "BA": "Basal Area",
    "DBH": "DBH (cm)",
    "Height": "Height",
    "WS": "Stem biomass",
    "WR": "Root biomass",
    "WF": "Stem foliage",
}
FIT_PARAMS = [
    "pFS20",
    "aWS",
    "nWS",
    "pRn",
    "Tmin",
    "Topt",
    "Tmax",
    "fN0",
    "fNn",
    "MaxAge",
    "rAge",
    "gammaN1",
    "thinPower",
    "mS",
    "alphaCx",
    "rhoMin",
    "rhoMax",
    "aH",
    "nHB",
    "nHC",
]


def build_time_index(climate) -> pd.DatetimeIndex:
    """Build a month-end datetime index matching the simulation length."""
    n_months = len(climate.month) if hasattr(climate, "month") else len(climate.T_avg)
    if hasattr(climate, "year") and hasattr(climate, "month"):
        return pd.to_datetime(
            climate.year.astype(str) + "-" + climate.month.astype(str) + "-01"
        ) + pd.offsets.MonthEnd(0)
    return pd.date_range(start=pd.Timestamp("1967-01-01"), periods=n_months, freq="ME")


def run_default_model(file_path: str) -> dict[str, Any]:
    """Run 3PG with the file's default parameter values."""
    initial_state, climate, fixed_params, site_data, species_data, _, _ = prepare_data(file_path)
    param_defaults = load_param_defaults_from_file(file_path)
    phy_defaults = {k: v for k, v in param_defaults.items() if not k.startswith("err_")}
    fixed_params = fixed_params._replace(**phy_defaults)
    _, outputs = run_3pg(initial_state, climate, fixed_params, site_data, species_data)
    return outputs


def run_gradient_descent_model(
    file_path: str,
    target_vars: list[str],
    fit_params: list[str],
) -> dict[str, Any]:
    """Fit parameters with gradient descent and run 3PG with the fitted values."""
    config = GradientDescentConfig(
        target_vars=target_vars, fit_params=fit_params, file_path=file_path
    )
    fit_result = fit_with_gradient_descent(config)

    initial_state, climate, base_params, site_data, species_data, _, _ = prepare_data(file_path)
    fitted_params: Params = apply_fitted_params(
        base_params=base_params,
        fit_params=fit_params,
        fitted_values=fit_result.fitted_params,
        species_index=config.species_index,
    )
    _, outputs = run_3pg(initial_state, climate, fitted_params, site_data, species_data)
    return outputs


def run_bayesian_model(output_dir: str) -> dict[str, Any]:
    """Load posterior mean/lower/upper prediction bands from a saved PyMC inference run."""
    return load_predictions(os.path.join(output_dir, "predictions.npz"))


def run_hmc_model(output_dir: str) -> dict[str, Any]:
    """Load posterior mean/lower/upper prediction bands from a saved HMC (NUTS) run.

    See `trunx.gp3.bayesiancalibrations.parameter_estimation.run_hmc_analysis`.
    """
    return load_predictions(os.path.join(output_dir, "predictions.npz"))


def _obs_indices_in_time_series(time_months: pd.DatetimeIndex, obs_time: pd.Series) -> list[int]:
    """Map each observation time to its index in the full simulated time series."""
    time_lookup = {t: i for i, t in enumerate(time_months)}
    return [time_lookup[t] for t in obs_time if t in time_lookup]


def plot_comparison(
    file_path: str,
    bayesian_output_dir: str,
    hmc_output_dir: str,
    plot_variables: list[str] = PLOT_VARIABLES,
    fit_params: list[str] = FIT_PARAMS,
) -> tuple[Figure, pd.DataFrame]:
    """Plot default, gradient-descent, PyMC-, and HMC-Bayesian predictions vs. observations.

    Parameters
    ----------
    file_path : str
        3PG input Excel file (site, species, climate, observed sheets).
    bayesian_output_dir : str
        Directory containing a saved `predictions.npz` from a prior
        `run_pymc_analysis` run for the same file.
    hmc_output_dir : str
        Directory containing a saved `predictions.npz` from a prior
        `run_hmc_analysis` run for the same file (see `parameter_estimation.py`).
    plot_variables : list[str]
        Output variables to plot and score.
    fit_params : list[str]
        Parameter names to optimize during gradient descent (and that the
        loaded Bayesian runs were calibrated on).

    Returns
    -------
    tuple[Figure, pd.DataFrame]
        The comparison figure and a per-variable RMSE/MAE table.
    """
    climate = prepare_data(file_path)[1]
    time_months = build_time_index(climate)

    observations = pl.read_excel(file_path, sheet_name="observed")
    obs_time = pd.to_datetime(
        pd.Series(observations["year"].to_numpy()).astype(str)
        + "-"
        + pd.Series(observations["month"].to_numpy()).astype(str)
        + "-01"
    ) + pd.offsets.MonthEnd(0)
    obs_indices = _obs_indices_in_time_series(time_months, obs_time)

    default_outputs = run_default_model(file_path)
    gd_outputs = run_gradient_descent_model(file_path, plot_variables, fit_params)
    bay_predictions = run_bayesian_model(bayesian_output_dir)
    hmc_predictions = run_hmc_model(hmc_output_dir)

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    axes = axes.flatten()

    metrics = []
    for ax, var in zip(axes, plot_variables, strict=True):
        obs_values = np.asarray(observations[var], dtype=np.float64)
        default_at_obs = np.asarray([default_outputs[var][idx] for idx in obs_indices])
        gd_at_obs = np.asarray([gd_outputs[var][idx] for idx in obs_indices])

        mean_pred = np.asarray(bay_predictions[var][0])
        lower_pred = np.asarray(bay_predictions[var][1])
        upper_pred = np.asarray(bay_predictions[var][2])
        bay_at_obs = np.asarray([mean_pred[idx] for idx in obs_indices])

        hmc_mean_pred = np.asarray(hmc_predictions[var][0])
        hmc_lower_pred = np.asarray(hmc_predictions[var][1])
        hmc_upper_pred = np.asarray(hmc_predictions[var][2])
        hmc_at_obs = np.asarray([hmc_mean_pred[idx] for idx in obs_indices])

        ax.fill_between(time_months, lower_pred, upper_pred, alpha=0.3, label="PyMC (DEz) 95% CI")
        ax.plot(time_months, mean_pred, label="PyMC (DEz) mean")
        ax.fill_between(
            time_months, hmc_lower_pred, hmc_upper_pred, alpha=0.3, label="HMC (NUTS) 95% CI"
        )
        ax.plot(time_months, hmc_mean_pred, label="HMC (NUTS) mean")
        ax.plot(time_months, default_outputs[var], label="Default")
        ax.plot(time_months, gd_outputs[var], label="Gradient descent")
        ax.scatter(obs_time, obs_values, color="red", label="Observations", zorder=5)

        ax.set_xlabel("Year")
        ax.set_ylabel(LABEL_MAP.get(var, var))
        ax.grid(alpha=0.3)

        row = {
            "variable": var,
            "default_rmse": rmse(obs_values, default_at_obs),
            "default_mae": mae(obs_values, default_at_obs),
            "gd_rmse": rmse(obs_values, gd_at_obs),
            "gd_mae": mae(obs_values, gd_at_obs),
            "bayesian_rmse": rmse(obs_values, bay_at_obs),
            "bayesian_mae": mae(obs_values, bay_at_obs),
            "hmc_rmse": rmse(obs_values, hmc_at_obs),
            "hmc_mae": mae(obs_values, hmc_at_obs),
        }
        ax.set_title(
            f"RMSE — default: {row['default_rmse']:.2f}, GD: {row['gd_rmse']:.2f}, "
            f"PyMC: {row['bayesian_rmse']:.2f}, HMC: {row['hmc_rmse']:.2f}"
        )
        metrics.append(row)

    axes[0].legend()
    fig.tight_layout()

    return fig, pd.DataFrame(metrics)


def load_convergence_summary(inference_data_path: str, param_names: list[str]) -> pd.DataFrame:
    """Load a saved inference run and compute `az.summary` for the given parameters.

    Parameters
    ----------
    inference_data_path : str
        Path to a saved `inference_data.nc` (PyMC) or `numpyro_inference_data.nc` (HMC).
    param_names : list[str]
        Parameter names to summarize.

    Returns
    -------
    pd.DataFrame
        One row per parameter, with `az.summary`'s columns (mean, sd, r_hat,
        ess_bulk, ess_tail, ...) plus `n_chains`/`n_draws` read from the file
        itself (the actual retained posterior, not what a script requested).
    """
    idata = az.from_netcdf(inference_data_path)
    posterior = cast(Any, idata).posterior
    summary = cast(pd.DataFrame, az.summary(idata, var_names=param_names))
    summary = summary.reset_index().rename(columns={"index": "parameter"})
    summary["n_chains"] = posterior.sizes["chain"]
    summary["n_draws"] = posterior.sizes["draw"]
    return summary


def plot_convergence_comparison(
    pymc_inference_path: str,
    hmc_inference_path: str,
    pymc_num_warmup: int,
    hmc_num_warmup: int,
    param_names: list[str] | None = None,
) -> tuple[Figure, pd.DataFrame]:
    """Compare PyMC (DEMetropolisZ) vs. HMC (NUTS) convergence diagnostics per parameter.

    Parameters
    ----------
    pymc_inference_path : str
        Path to the saved PyMC `inference_data.nc`.
    hmc_inference_path : str
        Path to the saved HMC (NumPyro) `numpyro_inference_data.nc`.
    pymc_num_warmup, hmc_num_warmup : int
        Tuning/warmup draws used for each run. Not retained in either saved
        file, so these must be supplied from how the run was actually
        configured (not just read back from the code's current defaults).
    param_names : list[str] | None
        Parameters to compare. Defaults to `FIT_PARAMS` plus one `err_{var}`
        per variable in `PLOT_VARIABLES`.

    Returns
    -------
    tuple[Figure, pd.DataFrame]
        The comparison figure and the combined per-parameter summary table
        (one row per parameter per method).
    """
    if param_names is None:
        param_names = FIT_PARAMS + [f"err_{var}" for var in PLOT_VARIABLES]

    pymc_summary = load_convergence_summary(pymc_inference_path, param_names)
    pymc_summary["method"] = "PyMC (DEz)"
    pymc_summary["num_warmup"] = pymc_num_warmup

    hmc_summary = load_convergence_summary(hmc_inference_path, param_names)
    hmc_summary["method"] = "HMC (NUTS)"
    hmc_summary["num_warmup"] = hmc_num_warmup

    combined = pd.concat([pymc_summary, hmc_summary], ignore_index=True)

    diagnostic_metrics = ["r_hat", "ess_bulk", "ess_tail", "mean"]
    fig, axes = plt.subplots(len(diagnostic_metrics), 1, figsize=(14, 4 * len(diagnostic_metrics)))
    x = np.arange(len(param_names))
    width = 0.35
    colors = {"PyMC (DEz)": "tab:blue", "HMC (NUTS)": "tab:orange"}

    for ax, metric in zip(axes, diagnostic_metrics, strict=True):
        for offset, method in zip([-width / 2, width / 2], colors, strict=True):
            values = (
                combined[combined["method"] == method]
                .set_index("parameter")
                .reindex(param_names)[metric]
            )
            ax.bar(x + offset, values, width=width, label=method, color=colors[method])
        ax.set_ylabel(metric)
        ax.grid(alpha=0.3, axis="y")
        ax.legend()

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(param_names, rotation=90)
    for ax in axes[:-1]:
        ax.set_xticks(x)
        ax.set_xticklabels([])

    pymc_chains, pymc_draws = pymc_summary["n_chains"].iloc[0], pymc_summary["n_draws"].iloc[0]
    hmc_chains, hmc_draws = hmc_summary["n_chains"].iloc[0], hmc_summary["n_draws"].iloc[0]
    fig.suptitle(
        f"PyMC (DEz): {pymc_chains} chains × warmup={pymc_num_warmup:,}, samples={pymc_draws:,}"
        "   |   "
        f"HMC (NUTS): {hmc_chains} chains × warmup={hmc_num_warmup:,}, samples={hmc_draws:,}"
    )
    fig.tight_layout()

    return fig, combined


if __name__ == "__main__":
    _file_path = os.path.join(threepg_data_folder, "full_solling_data.xlsx")
    _bayesian_output_dir = os.path.join(
        results_data_folder, "results/full_pymc_inference_results_1M_1M"
    )
    _hmc_output_dir = os.path.join(data_folder, "hmc_results")
    _plot_output_dir = os.path.join(results_data_folder, "bayesian_test_plot")
    os.makedirs(_plot_output_dir, exist_ok=True)

    _fig, _metrics_df = plot_comparison(_file_path, _bayesian_output_dir, _hmc_output_dir)
    print(_metrics_df)
    _fig.savefig(
        os.path.join(_plot_output_dir, "prediction_comparison.png"), dpi=200, bbox_inches="tight"
    )

    _conv_fig, _conv_df = plot_convergence_comparison(
        pymc_inference_path=os.path.join(_bayesian_output_dir, "inference_data.nc"),
        hmc_inference_path=os.path.join(_hmc_output_dir, "numpyro_inference_data.nc"),
        pymc_num_warmup=1_000_000,
        hmc_num_warmup=100,
    )
    print(_conv_df)
    _conv_fig.savefig(
        os.path.join(_plot_output_dir, "convergence_comparison.png"), dpi=200, bbox_inches="tight"
    )
    print(f"Saved plots to {_plot_output_dir}")

    plt.show()
