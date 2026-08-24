"""Compare default, gradient-descent, PyMC-Bayesian, and HMC-Bayesian 3PG predictions.

Plots the default prediction plus any of gradient-descent/PyMC-/HMC-Bayesian
sources enabled via their `include_*` flags, against observations for one
site, with per-variable RMSE/MAE printed for comparison.
"""

import gc
import os
from typing import Any, cast

import arviz as az
import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import root_mean_squared_error as rmse

from trunx.config import data_folder, results_data_folder, threepg_data_folder
from trunx.gp3.bayesiancalibrations.bayesian_config import DIAGNOSTIC_ONLY_ERROR_NAMES, FIT_PARAMS
from trunx.gp3.bayesiancalibrations.load_files import (
    load_param_defaults_from_file,
    load_priors_from_file,
)
from trunx.gp3.bayesiancalibrations.save_load_results import (
    load_gradient_descent_result,
    load_map_estimate,
    load_predictions,
)
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
    "WF": "Foliage biomass",
}


def build_time_index(climate, site_data) -> pd.DatetimeIndex:
    """Build a month-end datetime index matching the simulation length.

    `ClimateData` only carries a repeating 1-12 month-of-year array with no
    year, so the true calendar start is anchored from `site_data.year_i`/
    `month_i` instead (same pattern as `gradient_descent.py`).
    """
    n_months = len(climate.month)
    start_date = pd.Timestamp(
        year=int(site_data.year_i[0]), month=int(site_data.month_i[0]), day=1
    )
    return pd.date_range(start=start_date, periods=n_months, freq="ME")


def run_default_model(file_path: str) -> dict[str, Any]:
    """Run 3PG with the file's default parameter values."""
    initial_state, climate, fixed_params, site_data, species_data, _, _ = prepare_data(file_path)
    param_defaults = load_param_defaults_from_file(file_path)
    phy_defaults = {k: v for k, v in param_defaults.items() if not k.startswith("err_")}
    fixed_params = fixed_params._replace(**phy_defaults)
    _, outputs = run_3pg(initial_state, climate, fixed_params, site_data, species_data)
    return outputs


def get_gradient_descent_fit(cache_dir: str) -> dict[str, float]:
    """Load a gradient descent fit saved by `run_calibration_sweep.py` (or similar).

    Plotting never runs gradient descent itself — the point estimate takes several
    minutes, which a plotting call shouldn't pay for — so this only loads a
    `gradient_descent_result.json` from `cache_dir` (see `save_gradient_descent_result`).

    Parameters
    ----------
    cache_dir : str
        Directory containing a `gradient_descent_result.json` from an earlier
        calibration run.

    Returns
    -------
    dict[str, float]
        Fitted parameter values.

    Raises
    ------
    FileNotFoundError
        If `cache_dir` has no saved fit yet.
    """
    cache_path = os.path.join(cache_dir, "gradient_descent_result.json")
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"No saved gradient descent fit at {cache_path!r}. Run the calibration "
            "(e.g. scripts/run_calibration_sweep.py) before plotting."
        )
    return load_gradient_descent_result(cache_path)


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


def dbh_from_observed_biomass(
    ws_per_ha: np.ndarray, stems_per_ha: np.ndarray, aWS: float, nWS: float
) -> np.ndarray:
    """3PG's own mean-tree DBH, applied to field biomass and stem count.

    Same inversion as `helper_function.compute_dbh`, but fed the observed WS/N instead
    of simulated output. Comparing this against the field-measured quadratic mean
    diameter (the `DBH` observation itself) isolates the Jensen's-inequality gap
    between the two aggregations: `DBH` is built by summing per-tree allometric
    equations over the stand's actual size distribution, while this — like the model's
    own DBH state — inverts a single mean tree. See TODO.md.
    """
    ws_per_tree_kg = (ws_per_ha * 1000.0) / stems_per_ha
    return (ws_per_tree_kg / aWS) ** (1.0 / nWS)


def _obs_indices_in_time_series(
    time_months: pd.DatetimeIndex, obs_time: pd.Series
) -> tuple[list[int], np.ndarray]:
    """Map each observation time to its index in the full simulated time series.

    Returns the matched indices together with a boolean mask (same length as
    `obs_time`) marking which observations fall inside the simulated range.
    Observations outside it (e.g. past the available climate data) have no
    index and must be dropped from any array aligned with `obs_time`.
    """
    time_lookup = {t: i for i, t in enumerate(time_months)}
    mask = np.asarray([t in time_lookup for t in obs_time])
    indices = [time_lookup[t] for t in obs_time if t in time_lookup]
    return indices, mask


def plot_comparison(
    file_path: str,
    fit_params: list[str],
    bayesian_output_dir: str | None = None,
    hmc_output_dir: str | None = None,
    plot_variables: list[str] = PLOT_VARIABLES,
    include_gradient_descent: bool = False,
    include_bayesian: bool = False,
    include_hmc: bool = False,
    bayesian_label: str = "PyMC (DEz)",
    hmc_label: str = "HMC (NUTS)",
    site_name: str | None = None,
) -> tuple[Figure, pd.DataFrame]:
    """Plot default, and optionally gradient-descent/PyMC-/HMC-Bayesian predictions vs. obs.

    Parameters
    ----------
    file_path : str
        3PG input Excel file (site, species, climate, observed sheets).
    bayesian_output_dir : str | None
        Directory containing a saved `predictions.npz` from a prior
        `run_pymc_analysis` run for the same file. Required if `include_bayesian`.
    hmc_output_dir : str | None
        Directory containing a saved `predictions.npz` from a prior
        `run_hmc_analysis` run for the same file (see `parameter_estimation.py`).
        Required if `include_hmc`.
    plot_variables : list[str]
        Output variables to plot and score.
    fit_params : list[str]
        Parameter names to optimize during gradient descent (and that the
        loaded Bayesian runs were calibrated on).
    include_gradient_descent, include_bayesian, include_hmc : bool
        Whether to run, plot, and score each source. The default model
        always runs.
    bayesian_label : str
        Legend/title label for the `bayesian_output_dir` source, e.g. "MAP + Laplace"
        when the saved predictions come from `run_map_analysis` instead of MCMC.
    hmc_label : str
        Legend/title label for the `hmc_output_dir` source. Despite the parameter name
        (`run_hmc_model` originally only loaded NUTS runs), this slot works for any
        saved `predictions.npz`, e.g. "MAP + Laplace" or "MCMC (DEz)".
    site_name : str | None
        If given, adds a figure title with this name, the mean RMSE across the
        variables actually calibrated on (`plot_variables` minus
        `bayesian_config.DIAGNOSTIC_ONLY_ERROR_NAMES` — DBH/BA/Height are excluded,
        since they're diagnostic-only, see TODO.md), and the MAP log posterior read
        from `bayesian_output_dir/map_estimate.json`, if present.

    Returns
    -------
    tuple[Figure, pd.DataFrame]
        The comparison figure and a per-variable RMSE/MAE table.
    """
    if include_bayesian and bayesian_output_dir is None:
        raise ValueError("bayesian_output_dir is required when include_bayesian is True")
    if include_hmc and hmc_output_dir is None:
        raise ValueError("hmc_output_dir is required when include_hmc is True")

    _, climate, _, site_data, _, _, _ = prepare_data(file_path)
    time_months = build_time_index(climate, site_data)

    observations = pl.read_excel(file_path, sheet_name="observed")
    obs_time = pd.to_datetime(
        pd.Series(observations["year"].to_numpy()).astype(str)
        + "-"
        + pd.Series(observations["month"].to_numpy()).astype(str)
        + "-01"
    ) + pd.offsets.MonthEnd(0)
    obs_indices, obs_mask = _obs_indices_in_time_series(time_months, obs_time)
    if not obs_mask.all():
        print(
            f"Warning: {int((~obs_mask).sum())} observation(s) fall outside the "
            "simulated time range and will be dropped"
        )
        observations = observations.filter(pl.Series(obs_mask))
        obs_time = obs_time[obs_mask]

    derived_dbh = None
    if "N" in observations.columns:
        stems_mask = observations["N"].is_not_null().to_numpy()
        if stems_mask.any():
            defaults = load_param_defaults_from_file(file_path, ["aWS", "nWS"])
            aWS, nWS = defaults["aWS"], defaults["nWS"]
            derived_dbh = (
                obs_time[stems_mask],
                dbh_from_observed_biomass(
                    ws_per_ha=observations["WS"].to_numpy()[stems_mask],
                    stems_per_ha=observations["N"].to_numpy()[stems_mask],
                    aWS=aWS,
                    nWS=nWS,
                ),
            )

    default_outputs = run_default_model(file_path)

    gd_outputs = (
        run_gradient_descent_model(file_path, plot_variables, fit_params)
        if include_gradient_descent
        else None
    )

    bay_predictions = None
    if include_bayesian:
        assert bayesian_output_dir is not None
        bay_predictions = run_bayesian_model(bayesian_output_dir)

    hmc_predictions = None
    if include_hmc:
        assert hmc_output_dir is not None
        hmc_predictions = run_hmc_model(hmc_output_dir)

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    axes = axes.flatten()

    metrics = []
    for ax, var in zip(axes, plot_variables, strict=True):
        obs_values = np.asarray(observations[var], dtype=np.float64)
        default_at_obs = np.asarray([default_outputs[var][idx] for idx in obs_indices])

        ax.plot(time_months, default_outputs[var], label="Default")
        row = {
            "variable": var,
            "default_rmse": rmse(obs_values, default_at_obs),
            "default_mae": mae(obs_values, default_at_obs),
        }
        title_parts = [f"default: {row['default_rmse']:.2f}"]

        if gd_outputs is not None:
            gd_at_obs = np.asarray([gd_outputs[var][idx] for idx in obs_indices])
            ax.plot(time_months, gd_outputs[var], label="Gradient descent")
            row["gd_rmse"] = rmse(obs_values, gd_at_obs)
            row["gd_mae"] = mae(obs_values, gd_at_obs)
            title_parts.append(f"GD: {row['gd_rmse']:.2f}")

        if bay_predictions is not None:
            mean_pred = np.asarray(bay_predictions[var][0])
            lower_pred = np.asarray(bay_predictions[var][1])
            upper_pred = np.asarray(bay_predictions[var][2])
            bay_at_obs = np.asarray([mean_pred[idx] for idx in obs_indices])
            ax.fill_between(
                time_months, lower_pred, upper_pred, alpha=0.3, label=f"{bayesian_label} 95% CI"
            )
            ax.plot(time_months, mean_pred, label=f"{bayesian_label} mean")
            row["bayesian_rmse"] = rmse(obs_values, bay_at_obs)
            row["bayesian_mae"] = mae(obs_values, bay_at_obs)
            title_parts.append(f"{bayesian_label}: {row['bayesian_rmse']:.2f}")

        if hmc_predictions is not None:
            hmc_mean_pred = np.asarray(hmc_predictions[var][0])
            hmc_lower_pred = np.asarray(hmc_predictions[var][1])
            hmc_upper_pred = np.asarray(hmc_predictions[var][2])
            hmc_at_obs = np.asarray([hmc_mean_pred[idx] for idx in obs_indices])
            ax.fill_between(
                time_months,
                hmc_lower_pred,
                hmc_upper_pred,
                alpha=0.3,
                label=f"{hmc_label} 95% CI",
            )
            ax.plot(time_months, hmc_mean_pred, label=f"{hmc_label} mean")
            row["hmc_rmse"] = rmse(obs_values, hmc_at_obs)
            row["hmc_mae"] = mae(obs_values, hmc_at_obs)
            title_parts.append(f"{hmc_label}: {row['hmc_rmse']:.2f}")

        ax.scatter(obs_time, obs_values, color="red", label="Observations", zorder=5)
        if var == "DBH" and derived_dbh is not None:
            derived_time, derived_values = derived_dbh
            ax.scatter(
                derived_time,
                derived_values,
                color="black",
                marker="x",
                label="3PG-derived (from obs WS, N)",
                zorder=5,
            )

        ax.set_xlabel("Year")
        ax.set_ylabel(LABEL_MAP.get(var, var))
        ax.grid(alpha=0.3)
        ax.set_title("RMSE — " + ", ".join(title_parts))
        metrics.append(row)

    axes[0].legend()

    if site_name is not None:
        diagnostic_only_vars = {name.removeprefix("err_") for name in DIAGNOSTIC_ONLY_ERROR_NAMES}
        calibrated_metrics = [
            row
            for row in metrics
            if row["variable"] not in diagnostic_only_vars and "bayesian_rmse" in row
        ]
        title = site_name
        if calibrated_metrics:
            mean_rmse = np.mean([row["bayesian_rmse"] for row in calibrated_metrics])
            calibrated_names = "/".join(row["variable"] for row in calibrated_metrics)
            title += f" — mean RMSE ({calibrated_names}): {mean_rmse:.2f}"
        if bayesian_output_dir is not None:
            map_estimate_path = os.path.join(bayesian_output_dir, "map_estimate.json")
            if os.path.exists(map_estimate_path):
                _, logp = load_map_estimate(map_estimate_path)
                title += f", log posterior: {logp:.2f}"
        fig.suptitle(title, fontsize=14)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    else:
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
        ess_bulk, ess_tail, ...) plus `n_chains`/`n_draws`/`num_warmup` read
        from the file itself (the actual retained posterior, not what a
        script requested). `num_warmup` is `None` for older saved files that
        predate `tuning_steps` being stashed in `posterior.attrs`.
    """
    idata = az.from_netcdf(inference_data_path)
    posterior = cast(Any, idata).posterior
    summary = cast(pd.DataFrame, az.summary(idata, var_names=param_names))
    summary = summary.reset_index().rename(columns={"index": "parameter"})
    summary["n_chains"] = posterior.sizes["chain"]
    summary["n_draws"] = posterior.sizes["draw"]
    summary["num_warmup"] = posterior.attrs.get("tuning_steps")
    return summary


def plot_trace_and_posterior(
    inference_data_path: str,
    param_names: list[str],
    priors: dict[str, tuple[float, float]],
) -> tuple[Figure, Figure]:
    """Plot MCMC trace and posterior distributions, with prior ranges marked.

    Parameters
    ----------
    inference_data_path : str
        Path to a saved `inference_data.nc` (PyMC) or `numpyro_inference_data.nc` (HMC).
    param_names : list[str]
        Parameter names to plot.
    priors : dict[str, tuple[float, float]]
        Prior (lower, upper) bounds per parameter, drawn as red lines. Parameters
        without a matching entry are plotted without prior lines.

    Returns
    -------
    tuple[Figure, Figure]
        The trace figure and the posterior figure.
    """
    idata = az.from_netcdf(inference_data_path)

    # `plot.max_subplots` (default 40) otherwise silently truncates the plot
    # instead of raising once `param_names` exceeds it, leaving fewer axes than
    # requested parameters.
    with az.rc_context(rc={"plot.max_subplots": None}):
        trace_axes = np.atleast_2d(az.plot_trace(idata, var_names=param_names))
        for row, param_name in zip(trace_axes, param_names, strict=True):
            bounds = priors.get(param_name)
            if bounds is None:
                continue
            density_ax, sample_ax = row
            for bound in bounds:
                density_ax.axvline(bound, color="red")
                sample_ax.axhline(bound, color="red")
        trace_fig = cast(Figure, trace_axes[0, 0].figure)
        trace_fig.tight_layout()

        # `plot_posterior`'s grid is sized to fit len(param_names) as squarely as
        # possible, leaving blank (`has_data() is False`) axes in leftover cells.
        posterior_axes = np.atleast_1d(az.plot_posterior(idata, var_names=param_names)).flatten()
        posterior_axes = [ax for ax in posterior_axes if ax.has_data()]
        for ax, param_name in zip(posterior_axes, param_names, strict=True):
            bounds = priors.get(param_name)
            if bounds is None:
                continue
            for bound in bounds:
                ax.axvline(bound, color="red")
        posterior_fig = cast(Figure, posterior_axes[0].figure)
        posterior_fig.tight_layout()

    return trace_fig, posterior_fig


def plot_convergence_comparison(
    fit_params: list[str],
    pymc_inference_path: str | None = None,
    hmc_inference_path: str | None = None,
    param_names: list[str] | None = None,
    include_bayesian: bool = True,
    include_hmc: bool = True,
) -> tuple[Figure, pd.DataFrame]:
    """Compare PyMC (DEMetropolisZ) and/or HMC (NUTS) convergence diagnostics per parameter.

    Parameters
    ----------
    pymc_inference_path : str | None
        Path to the saved PyMC `inference_data.nc`. Required if `include_bayesian`.
    hmc_inference_path : str | None
        Path to the saved HMC (NumPyro) `numpyro_inference_data.nc`. Required if `include_hmc`.
    param_names : list[str] | None
        Parameters to compare. Defaults to `FIT_PARAMS` plus one `err_{var}`
        per variable in `PLOT_VARIABLES`.
    include_bayesian, include_hmc : bool
        Whether to include each method. Tuning/warmup draws are read back
        from `posterior.attrs["tuning_steps"]` in the saved file itself
        (`None` for older saved files that predate this being stashed).

    Returns
    -------
    tuple[Figure, pd.DataFrame]
        The comparison figure and the combined per-parameter summary table
        (one row per parameter per included method).
    """
    if not include_bayesian and not include_hmc:
        raise ValueError("At least one of include_bayesian/include_hmc must be True")
    if include_bayesian and pymc_inference_path is None:
        raise ValueError("pymc_inference_path is required when include_bayesian is True")
    if include_hmc and hmc_inference_path is None:
        raise ValueError("hmc_inference_path is required when include_hmc is True")

    if param_names is None:
        param_names = fit_params + [f"err_{var}" for var in PLOT_VARIABLES]

    colors = {}
    summaries = []
    if include_bayesian:
        assert pymc_inference_path is not None
        pymc_summary = load_convergence_summary(pymc_inference_path, param_names)
        pymc_summary["method"] = "PyMC (DEz)"
        summaries.append(pymc_summary)
        colors["PyMC (DEz)"] = "tab:blue"

    if include_hmc:
        assert hmc_inference_path is not None
        hmc_summary = load_convergence_summary(hmc_inference_path, param_names)
        hmc_summary["method"] = "HMC (NUTS)"
        summaries.append(hmc_summary)
        colors["HMC (NUTS)"] = "tab:orange"

    combined = pd.concat(summaries, ignore_index=True)

    diagnostic_metrics = ["r_hat", "ess_bulk", "ess_tail", "mean"]
    fig, axes = plt.subplots(len(diagnostic_metrics), 1, figsize=(14, 4 * len(diagnostic_metrics)))
    x = np.arange(len(param_names))
    n_methods = len(colors)
    width = 0.7 / n_methods
    offsets = [width * (i - (n_methods - 1) / 2) for i in range(n_methods)]

    for ax, metric in zip(axes, diagnostic_metrics, strict=True):
        for offset, method in zip(offsets, colors, strict=True):
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

    title_parts = []
    for method in colors:
        method_df = combined[combined["method"] == method]
        chains, draws = method_df["n_chains"].iloc[0], method_df["n_draws"].iloc[0]
        warmup = method_df["num_warmup"].iloc[0]
        warmup_str = f"{warmup:,}" if warmup is not None else "unknown"
        title_parts.append(f"{method}: {chains} chains × warmup={warmup_str}, samples={draws:,}")
    fig.suptitle("   |   ".join(title_parts))
    fig.tight_layout()

    return fig, combined


def plot_parameter_value_comparison(
    pymc_inference_path: str | None = None,
    hmc_inference_path: str | None = None,
    param_names: list[str] = FIT_PARAMS,
    include_gradient_descent: bool = True,
    include_bayesian: bool = True,
    include_hmc: bool = True,
    gd_cache_dir: str | None = None,
) -> tuple[Figure, pd.DataFrame]:
    """Compare each parameter's fitted value across gradient descent, PyMC (DEz), and HMC (NUTS).

    One small subplot per parameter (values span very different scales, e.g.
    `mS` ~1e-4 vs. `MaxAge` ~400, so a single shared-axis bar chart would hide
    most of them), each with one bar per included method.

    Parameters
    ----------
    pymc_inference_path : str | None
        Path to the saved PyMC `inference_data.nc`. Required if `include_bayesian`.
    hmc_inference_path : str | None
        Path to the saved HMC (NumPyro) `numpyro_inference_data.nc`. Required if `include_hmc`.
    param_names : list[str]
        Physiology parameters to compare. Error/sigma terms are excluded — gradient
        descent doesn't fit them, so they'd have no value to compare against.
    include_gradient_descent, include_bayesian, include_hmc : bool
        Whether to include each method.
    gd_cache_dir : str | None
        Directory holding a saved gradient descent fit (see `get_gradient_descent_fit`).
        Required if `include_gradient_descent`; pass the same directory used for
        `plot_comparison`'s `gd_cache_dir` to compare against the same fit.

    Returns
    -------
    tuple[Figure, pd.DataFrame]
        The comparison figure and a long-form table (`parameter`, `method`, `value`).
    """
    if not (include_gradient_descent or include_bayesian or include_hmc):
        raise ValueError(
            "At least one of include_gradient_descent/include_bayesian/include_hmc must be True"
        )
    if include_gradient_descent and gd_cache_dir is None:
        raise ValueError("gd_cache_dir is required when include_gradient_descent is True")
    if include_bayesian and pymc_inference_path is None:
        raise ValueError("pymc_inference_path is required when include_bayesian is True")
    if include_hmc and hmc_inference_path is None:
        raise ValueError("hmc_inference_path is required when include_hmc is True")

    colors = {}
    rows = []

    if include_gradient_descent:
        assert gd_cache_dir is not None
        fitted_params = get_gradient_descent_fit(gd_cache_dir)
        for name in param_names:
            rows.append(
                {"parameter": name, "method": "Gradient descent", "value": fitted_params[name]}
            )
        colors["Gradient descent"] = "tab:green"

    if include_bayesian:
        assert pymc_inference_path is not None
        pymc_summary = load_convergence_summary(pymc_inference_path, param_names)
        for _, row in pymc_summary.iterrows():
            rows.append(
                {"parameter": row["parameter"], "method": "PyMC (DEz)", "value": row["mean"]}
            )
        colors["PyMC (DEz)"] = "tab:blue"

    if include_hmc:
        assert hmc_inference_path is not None
        hmc_summary = load_convergence_summary(hmc_inference_path, param_names)
        for _, row in hmc_summary.iterrows():
            rows.append(
                {"parameter": row["parameter"], "method": "HMC (NUTS)", "value": row["mean"]}
            )
        colors["HMC (NUTS)"] = "tab:orange"

    combined = pd.DataFrame(rows)

    ncols = 4
    nrows = int(np.ceil(len(param_names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = axes.flatten()

    for ax, name in zip(axes, param_names, strict=False):
        param_rows = combined[combined["parameter"] == name]
        for method in colors:
            value = param_rows[param_rows["method"] == method]["value"]
            if not value.empty:
                ax.bar(method, value.iloc[0], color=colors[method])
        ax.set_title(name)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(alpha=0.3, axis="y")

    for ax in axes[len(param_names) :]:
        ax.axis("off")

    legend_handles = [Rectangle((0, 0), 1, 1, color=color) for color in colors.values()]
    fig.legend(legend_handles, list(colors), loc="upper right")
    fig.suptitle("Fitted parameter values by calibration method")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

    return fig, combined


def plot_and_save(plot_id: str, output_dir: str, prefix: str = ""):
    """Plot and save comparison/convergence/trace/posterior figures for a plot.

    Parameters
    ----------
    plot_id : str
        ICP plot identifier.
    output_dir : str
        Directory to save the plots in.
    prefix : str
        Optional prefix for the saved filenames (e.g. "trot_").
    """
    if plot_id == "solling":
        _bayesian_output_dir = os.path.join(results_data_folder, "results/pymc_inference_results")
    else:
        _bayesian_output_dir = os.path.join(
            results_data_folder, f"results/{prefix}pymc_inference_results_{plot_id}"
        )

    _file_path = os.path.join(_bayesian_output_dir, f"{plot_id}_data.xlsx")

    if plot_id == "solling":
        fit_params = FIT_PARAMS
    else:
        _df = pl.read_excel(_file_path, sheet_name="param_bound")
        _df = _df.filter(pl.col("min").is_not_null() & pl.col("max").is_not_null())
        fit_params = _df["param_name"].to_list()

    _fig, _metrics_df = plot_comparison(
        _file_path,
        fit_params,
        _bayesian_output_dir,
        _hmc_output_dir,
        include_gradient_descent=_include_gradient_descent,
        include_bayesian=_include_bayesian,
        include_hmc=_include_hmc,
    )
    # print(_metrics_df)
    _fig.savefig(
        os.path.join(_plot_output_dir, f"{prefix}prediction_comparison_{plot_id}.png"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(_fig)

    _conv_fig, _conv_df = plot_convergence_comparison(
        pymc_inference_path=os.path.join(_bayesian_output_dir, "inference_data.nc"),
        hmc_inference_path=os.path.join(_hmc_output_dir, "numpyro_inference_data.nc"),
        include_bayesian=_include_bayesian,
        include_hmc=_include_hmc,
        fit_params=fit_params,
    )
    # print(_conv_df)
    _conv_fig.savefig(
        os.path.join(_plot_output_dir, f"{prefix}convergence_comparison_{plot_id}.png"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(_conv_fig)

    if _include_bayesian:
        _priors = load_priors_from_file(_file_path, fit_params)
        _trace_fig, _posterior_fig = plot_trace_and_posterior(
            os.path.join(_bayesian_output_dir, "inference_data.nc"), fit_params, _priors
        )
        _trace_fig.savefig(
            os.path.join(_plot_output_dir, f"{prefix}trace_{plot_id}.png"),
            dpi=200,
            bbox_inches="tight",
        )
        _posterior_fig.savefig(
            os.path.join(_plot_output_dir, f"{prefix}posterior_{plot_id}.png"),
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(_trace_fig)
        plt.close(_posterior_fig)

    print(f"Saved plots to {_plot_output_dir}")
    gc.collect()
    jax.clear_caches()


if __name__ == "__main__":
    _plot_output_dir = os.path.join(results_data_folder, "bayesian_test_plot")
    _hmc_output_dir = os.path.join(data_folder, "hmc_results")
    os.makedirs(_plot_output_dir, exist_ok=True)

    _include_bayesian = True
    _include_hmc = False
    _include_gradient_descent = True

    plot_ids = ["solling"]
    for plot_id in plot_ids:
        print(f"Processing plot_id={plot_id}...")
        plot_and_save(plot_id, _plot_output_dir, prefix="")
