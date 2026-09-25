"""Compare default, gradient-descent, PyMC-Bayesian, HMC-Bayesian, and MAP 3PG predictions.

Plots the default prediction plus any of gradient-descent/PyMC-/HMC-Bayesian/MAP
sources enabled via their `include_*` flags, against observations for one
site, with per-variable RMSE/MAE printed for comparison.
"""

import gc
import os
from collections.abc import Iterable
from typing import Any, cast

import arviz as az
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import root_mean_squared_error as rmse

from trunx.config import data_folder, results_data_folder, threepg_data_folder
from trunx.gp3.bayesiancalibrations.bayesian_config import (
    DIAGNOSTIC_ONLY_ERROR_NAMES,
    ERROR_MODES,
    FIT_PARAMS,
    INITIAL_STATE_PARAMS,
    PROCESS_ERROR_PARAM_NAMES,
    species_plot_ids,
)
from trunx.gp3.bayesiancalibrations.load_files import (
    fit_params_for_mode,
    load_param_defaults_from_file,
    load_priors_from_file,
)
from trunx.gp3.bayesiancalibrations.save_load_results import (
    load_gradient_descent_result,
    load_map_estimate,
    load_predictions,
)
from trunx.gp3.gradient_descent import apply_fitted_params
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
    input_data = prepare_data(file_path)
    param_defaults = load_param_defaults_from_file(file_path)
    # Excludes err_*/perr_* sigma defaults (and any other non-physiology entry
    # in the file's error_param sheet) rather than blacklisting prefixes, so a
    # third such prefix added later can't reopen this same AttributeError.
    phy_defaults = {k: v for k, v in param_defaults.items() if k in Params._fields}
    fixed_params = input_data.params._replace(
        **{
            name: jnp.full_like(getattr(input_data.params, name), value)
            for name, value in phy_defaults.items()
        }
    )
    _, outputs = run_3pg(
        input_data.initial_state,
        input_data.climate,
        fixed_params,
        input_data.site,
        input_data.species,
    )
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
    fit_params: list[str],
    cache_dir: str,
) -> dict[str, Any]:
    """Load a saved gradient descent fit and run 3PG with the fitted values.

    Plotting never runs gradient descent itself — the point estimate takes
    several minutes, which a plotting call shouldn't pay for — so this only
    loads a `gradient_descent_result.json` from `cache_dir` (see
    `scripts/run_calibration_sweep.py`, which computes and saves it).

    Parameters
    ----------
    file_path : str
        3PG input Excel file (site, species, climate, observed sheets).
    fit_params : list[str]
        Parameter names to apply from the saved fit. A name missing from it
        (e.g. a mode's calibrable-parameter set grew since the fit was last
        run) is silently left at the file's default value.
    cache_dir : str
        Directory containing a saved `gradient_descent_result.json`.

    Raises
    ------
    FileNotFoundError
        If `cache_dir` has no saved fit yet.
    """
    fitted_values = get_gradient_descent_fit(cache_dir)
    applied_fit_params = [name for name in fit_params if name in fitted_values]

    input_data = prepare_data(file_path)
    fitted_params: Params = apply_fitted_params(
        base_params=input_data.params,
        fit_params=applied_fit_params,
        fitted_values=fitted_values,
        species_index=0,
    )
    _, outputs = run_3pg(
        input_data.initial_state,
        input_data.climate,
        fitted_params,
        input_data.site,
        input_data.species,
    )
    return outputs


def run_bayesian_model(output_dir: str) -> dict[str, Any]:
    """Load posterior mean/lower/upper prediction bands from a saved PyMC inference run."""
    return load_predictions(os.path.join(output_dir, "predictions.npz"))


def run_hmc_model(output_dir: str) -> dict[str, Any]:
    """Load posterior mean/lower/upper prediction bands from a saved HMC (NUTS) run.

    See `trunx.gp3.bayesiancalibrations.parameter_estimation.run_hmc_analysis`.
    """
    return load_predictions(os.path.join(output_dir, "predictions.npz"))


def run_map_model(output_dir: str) -> dict[str, Any]:
    """Load mean/lower/upper prediction bands from a saved MAP (+ Laplace) run.

    See `trunx.gp3.bayesiancalibrations.map_param_est.run_map_analysis`.
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
    map_output_dir: str | None = None,
    plot_variables: list[str] = PLOT_VARIABLES,
    include_gradient_descent: bool = False,
    include_bayesian: bool = False,
    include_hmc: bool = False,
    include_map: bool = False,
    bayesian_label: str = "PyMC (DEz)",
    hmc_label: str = "HMC (NUTS)",
    map_label: str = "MAP + Laplace",
    site_name: str | None = None,
    gd_cache_dir: str | None = None,
    series_colors: dict[str, str] | None = None,
) -> tuple[Figure, pd.DataFrame]:
    """Plot default, and optionally gradient-descent/PyMC-/HMC-Bayesian/MAP predictions vs. obs.

    Parameters
    ----------
    file_path : str
        3PG input Excel file (site, species, climate, observed sheets).
    series_colors : dict[str, str] | None
        Override plot color per series name (`"default"`, `"gradient_descent"`,
        `"bayesian"`, `"hmc"`, `"map"`, `"observed"`). A series without a matching
        entry falls back to matplotlib's default color cycle (`"observed"` falls
        back to `"red"`). Set this so multiple series sharing one subplot stay
        distinguishable under a caller-set narrow-hue `axes.prop_cycle` (e.g. an
        all-green theme, where cycle order alone can leave two series looking
        too similar, or land on a shade too pale to see against a white figure).
    bayesian_output_dir : str | None
        Directory containing a saved `predictions.npz` from a prior
        `run_pymc_analysis` run for the same file. Required if `include_bayesian`.
    hmc_output_dir : str | None
        Directory containing a saved `predictions.npz` from a prior
        `run_hmc_analysis` run for the same file (see `parameter_estimation.py`).
        Required if `include_hmc`.
    map_output_dir : str | None
        Directory containing a saved `predictions.npz` from a prior
        `run_map_analysis` run for the same file. Required if `include_map`.
    plot_variables : list[str]
        Output variables to plot and score.
    fit_params : list[str]
        Parameter names to optimize during gradient descent (and that the
        loaded Bayesian runs were calibrated on).
    include_gradient_descent, include_bayesian, include_hmc, include_map : bool
        Whether to run, plot, and score each source. The default model
        always runs.
    gd_cache_dir : str | None
        Directory holding a saved gradient descent fit (see
        `run_gradient_descent_model`/`scripts/run_calibration_sweep.py`).
        Required if `include_gradient_descent` is True.
    bayesian_label : str
        Legend/title label for the `bayesian_output_dir` source.
    hmc_label : str
        Legend/title label for the `hmc_output_dir` source. Despite the parameter name
        (`run_hmc_model` originally only loaded NUTS runs), this slot works for any
        saved `predictions.npz`, e.g. "MCMC (DEz)".
    map_label : str
        Legend/title label for the `map_output_dir` source.
    site_name : str | None
        If given, adds a figure title with this name, the mean RMSE across the
        variables actually calibrated on (`plot_variables` minus
        `bayesian_config.DIAGNOSTIC_ONLY_ERROR_NAMES` — DBH/BA/Height are excluded,
        since they're diagnostic-only, see TODO.md), and the MAP log posterior read
        from `map_output_dir/map_estimate.json` (falling back to
        `bayesian_output_dir/map_estimate.json` if `map_output_dir` isn't given), if
        present.

    Returns
    -------
    tuple[Figure, pd.DataFrame]
        The comparison figure and a per-variable RMSE/MAE table.
    """
    if include_gradient_descent and gd_cache_dir is None:
        raise ValueError("gd_cache_dir is required when include_gradient_descent is True")
    if include_bayesian and bayesian_output_dir is None:
        raise ValueError("bayesian_output_dir is required when include_bayesian is True")
    if include_hmc and hmc_output_dir is None:
        raise ValueError("hmc_output_dir is required when include_hmc is True")
    if include_map and map_output_dir is None:
        raise ValueError("map_output_dir is required when include_map is True")

    input_data = prepare_data(file_path)
    time_months = build_time_index(input_data.climate, input_data.site)

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

    gd_outputs = None
    if include_gradient_descent:
        assert gd_cache_dir is not None
        gd_outputs = run_gradient_descent_model(file_path, fit_params, cache_dir=gd_cache_dir)

    bay_predictions = None
    if include_bayesian:
        assert bayesian_output_dir is not None
        bay_predictions = run_bayesian_model(bayesian_output_dir)

    hmc_predictions = None
    if include_hmc:
        assert hmc_output_dir is not None
        hmc_predictions = run_hmc_model(hmc_output_dir)

    map_predictions = None
    if include_map:
        assert map_output_dir is not None
        map_predictions = run_map_model(map_output_dir)

    n_cols = min(3, len(plot_variables))
    n_rows = int(np.ceil(len(plot_variables) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20 / 3 * n_cols, 5 * n_rows))
    axes = np.ravel(np.atleast_1d(axes))

    series_colors = series_colors or {}

    def _color_kwargs(name: str) -> dict[str, str]:
        return {"color": series_colors[name]} if name in series_colors else {}

    metrics = []
    for ax, var in zip(axes, plot_variables, strict=False):
        obs_values = np.asarray(observations[var], dtype=np.float64)
        default_at_obs = np.asarray([default_outputs[var][idx] for idx in obs_indices])

        ax.plot(time_months, default_outputs[var], label="Default", **_color_kwargs("default"))
        row = {
            "variable": var,
            "default_rmse": rmse(obs_values, default_at_obs),
            "default_mae": mae(obs_values, default_at_obs),
        }
        title_parts = [f"default: {row['default_rmse']:.2f}"]

        if gd_outputs is not None:
            gd_at_obs = np.asarray([gd_outputs[var][idx] for idx in obs_indices])
            ax.plot(
                time_months,
                gd_outputs[var],
                label="Gradient descent",
                **_color_kwargs("gradient_descent"),
            )
            row["gd_rmse"] = rmse(obs_values, gd_at_obs)
            row["gd_mae"] = mae(obs_values, gd_at_obs)
            title_parts.append(f"GD: {row['gd_rmse']:.2f}")

        if bay_predictions is not None:
            mean_pred = np.asarray(bay_predictions[var][0])
            lower_pred = np.asarray(bay_predictions[var][1])
            upper_pred = np.asarray(bay_predictions[var][2])
            bay_at_obs = np.asarray([mean_pred[idx] for idx in obs_indices])
            ax.fill_between(
                time_months,
                lower_pred,
                upper_pred,
                alpha=0.3,
                label=f"{bayesian_label} 95% CI",
                **_color_kwargs("bayesian"),
            )
            ax.plot(
                time_months, mean_pred, label=f"{bayesian_label} mean", **_color_kwargs("bayesian")
            )
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
                **_color_kwargs("hmc"),
            )
            ax.plot(time_months, hmc_mean_pred, label=f"{hmc_label} mean", **_color_kwargs("hmc"))
            row["hmc_rmse"] = rmse(obs_values, hmc_at_obs)
            row["hmc_mae"] = mae(obs_values, hmc_at_obs)
            title_parts.append(f"{hmc_label}: {row['hmc_rmse']:.2f}")

        if map_predictions is not None:
            map_mean_pred = np.asarray(map_predictions[var][0])
            map_lower_pred = np.asarray(map_predictions[var][1])
            map_upper_pred = np.asarray(map_predictions[var][2])
            map_at_obs = np.asarray([map_mean_pred[idx] for idx in obs_indices])
            ax.fill_between(
                time_months,
                map_lower_pred,
                map_upper_pred,
                alpha=0.3,
                label=f"{map_label} 95% CI",
                **_color_kwargs("map"),
            )
            ax.plot(time_months, map_mean_pred, label=f"{map_label} mean", **_color_kwargs("map"))
            row["map_rmse"] = rmse(obs_values, map_at_obs)
            row["map_mae"] = mae(obs_values, map_at_obs)
            title_parts.append(f"{map_label}: {row['map_rmse']:.2f}")

        ax.scatter(
            obs_time,
            obs_values,
            color=series_colors.get("observed", "red"),
            label="Observations",
            zorder=5,
        )
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
        # ax.set_title("RMSE — " + ", ".join(title_parts))
        metrics.append(row)

    for ax in axes[len(plot_variables) :]:
        ax.set_visible(False)

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
        map_estimate_dir = map_output_dir if map_output_dir is not None else bayesian_output_dir
        if map_estimate_dir is not None:
            map_estimate_path = os.path.join(map_estimate_dir, "map_estimate.json")
            if os.path.exists(map_estimate_path):
                _, logp = load_map_estimate(map_estimate_path)
                title += f", log posterior: {logp:.2f}"
        fig.suptitle(title, fontsize=14)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    else:
        fig.tight_layout()

    return fig, pd.DataFrame(metrics)


def _params_in_trace(inference_data_path: str, candidates: Iterable[str]) -> list[str]:
    """`candidates` names present as posterior variables in a saved trace.

    Returns `[]` if `inference_data_path` doesn't exist yet (that method
    hasn't been run for this scenario).
    """
    if not os.path.exists(inference_data_path):
        return []
    idata = az.from_netcdf(inference_data_path)
    posterior_vars = set(idata.posterior.data_vars)
    return [name for name in candidates if name in posterior_vars]


def _calibrated_params_from_results(
    candidates: frozenset[str],
    bayesian_output_dir: str,
    hmc_output_dir: str,
    map_output_dir: str,
    gradient_descent_dir: str,
    include_bayesian: bool,
    include_hmc: bool,
    include_map: bool,
    include_gradient_descent: bool,
) -> set[str]:
    """Union of `candidates` names actually present in whichever saved results are included.

    Reads straight from what each included method actually saved (MCMC
    posterior variables, `map_estimate.json`, `gradient_descent_result.json`)
    instead of trusting a computed/static parameter list, so plotting always
    matches what a given run was actually calibrated on — even across a
    `fit_params_for_mode`/`FIT_PARAMS` change made after that run was saved
    (see `run_gradient_descent_model`'s own stale-cache handling for the same
    concern on the gradient descent side).
    """
    found: set[str] = set()
    for included, method_dir in (
        (include_bayesian, bayesian_output_dir),
        (include_hmc, hmc_output_dir),
    ):
        if included:
            found.update(
                _params_in_trace(os.path.join(method_dir, "inference_data.nc"), candidates)
            )
    if include_map:
        map_estimate_path = os.path.join(map_output_dir, "map_estimate.json")
        if os.path.exists(map_estimate_path):
            map_estimate, _ = load_map_estimate(map_estimate_path)
            found.update(name for name in map_estimate if name in candidates)
    if include_gradient_descent:
        gd_path = os.path.join(gradient_descent_dir, "gradient_descent_result.json")
        if os.path.exists(gd_path):
            gd_fit = load_gradient_descent_result(gd_path)
            found.update(name for name in gd_fit if name in candidates)
    return found


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
        One row per parameter actually present in the saved posterior (params
        with no sigma prior in this run, e.g. `err_DBH` under a "biomass_only"
        calibration, are silently skipped — see the printed note), with
        `az.summary`'s columns (mean, sd, r_hat, ess_bulk, ess_tail, ...) plus
        `n_chains`/`n_draws`/`num_warmup` read from the file itself (the actual
        retained posterior, not what a script requested). `num_warmup` is
        `None` for older saved files that predate `tuning_steps` being stashed
        in `posterior.attrs`.
    """
    idata = az.from_netcdf(inference_data_path)
    posterior = cast(Any, idata).posterior
    available_names = [name for name in param_names if name in posterior.data_vars]
    missing_names = [name for name in param_names if name not in posterior.data_vars]
    if missing_names:
        print(
            f"Note: {inference_data_path} has no posterior for {missing_names} "
            "(not fit in this run) — skipping them"
        )
    summary = cast(pd.DataFrame, az.summary(idata, var_names=available_names))
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
        Parameter names to plot. A name with no posterior in this saved run
        (e.g. gathered from a different included method's results — see
        `_calibrated_params_from_results`) is silently skipped, matching
        `load_convergence_summary`'s own handling of the same case.
    priors : dict[str, tuple[float, float]]
        Prior (lower, upper) bounds per parameter, drawn as red lines. Parameters
        without a matching entry are plotted without prior lines.

    Returns
    -------
    tuple[Figure, Figure]
        The trace figure and the posterior figure.
    """
    idata = az.from_netcdf(inference_data_path)
    posterior_vars = set(cast(Any, idata).posterior.data_vars)
    missing_names = [name for name in param_names if name not in posterior_vars]
    if missing_names:
        print(
            f"Note: {inference_data_path} has no posterior for {missing_names} "
            "(not fit in this run) — skipping them"
        )
    param_names = [name for name in param_names if name in posterior_vars]

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


def plot_posterior_comparison(
    pymc_inference_path: str,
    hmc_inference_path: str,
    param_names: list[str],
    priors: dict[str, tuple[float, float]] | None = None,
    map_output_dir: str | None = None,
    gd_cache_dir: str | None = None,
    pymc_label: str = "PyMC (DEz)",
    hmc_label: str = "HMC (NUTS)",
    method_colors: dict[str, str] | None = None,
) -> Figure:
    """Overlay PyMC (DEMetropolisZ) and HMC (NUTS) posterior plots, per parameter.

    One small subplot per parameter, overlaying both methods' `arviz.plot_posterior`
    KDE curve (the same rendering as `plot_trace_and_posterior`'s posterior figure —
    one call per method on shared axes, each in its own color) with its own HDI bar
    and point-estimate text suppressed, since two independent sets of those labels
    stacked on one axis overlap and become unreadable. Values are shown instead as one
    color-coded vertical line per method: a dotted line at the posterior mean for the
    two full Bayesian calibrations (PyMC/HMC), and a bold dashed line for the two point
    estimates (MAP, gradient descent fit), each optional except the means, plus the
    prior (lower, upper) bounds marked in gray.

    Parameters
    ----------
    pymc_inference_path : str
        Path to the saved PyMC (DEMetropolisZ) `inference_data.nc`.
    hmc_inference_path : str
        Path to the saved HMC (NUTS) `inference_data.nc`.
    param_names : list[str]
        Parameters to plot. A name missing a posterior in either saved trace
        (e.g. gathered from a different included method's results — see
        `_calibrated_params_from_results`) is silently skipped, since this
        plot's whole point is overlaying both methods for the same parameter.
    priors : dict[str, tuple[float, float]] | None
        Prior (lower, upper) bounds per parameter, drawn as gray vertical lines.
        Parameters without a matching entry are plotted without prior lines.
    map_output_dir : str | None
        Directory containing a saved `map_estimate.json` (see `run_map_analysis`).
        A parameter missing from it is plotted without a MAP line.
    gd_cache_dir : str | None
        Directory containing a saved `gradient_descent_result.json` (see
        `get_gradient_descent_fit`). A parameter missing from it is plotted
        without a gradient descent line.
    pymc_label, hmc_label : str
        Legend labels for the two posterior plots.
    method_colors : dict[str, str] | None
        Override color per method (`pymc_label`/`hmc_label`/`"MAP"`/
        `"Gradient descent"`). Missing entries fall back to the default palette
        (blue/orange/reddish-purple/bluish-green — chosen, over a plain
        red/green pair, to stay distinguishable for red-green color blindness).

    Returns
    -------
    Figure
        The comparison figure.
    """
    pymc_idata = az.from_netcdf(pymc_inference_path)
    hmc_idata = az.from_netcdf(hmc_inference_path)

    pymc_vars = set(cast(Any, pymc_idata).posterior.data_vars)
    hmc_vars = set(cast(Any, hmc_idata).posterior.data_vars)
    missing_names = [name for name in param_names if name not in pymc_vars or name not in hmc_vars]
    if missing_names:
        print(
            f"Note: {missing_names} have no posterior in both {pymc_inference_path} and "
            f"{hmc_inference_path} — skipping them"
        )
    param_names = [name for name in param_names if name in pymc_vars and name in hmc_vars]

    priors = priors or {}
    map_estimate = (
        load_map_estimate(os.path.join(map_output_dir, "map_estimate.json"))[0]
        if map_output_dir is not None
        else {}
    )
    gd_fit = get_gradient_descent_fit(gd_cache_dir) if gd_cache_dir is not None else {}

    default_colors = {
        pymc_label: "#0072B2",  # blue
        hmc_label: "#E69F00",  # orange
        "MAP": "#CC79A7",  # reddish purple
        "Gradient descent": "#009E73",  # bluish green
    }
    method_colors = {**default_colors, **(method_colors or {})}
    bayesian_linestyle = ":"
    point_estimate_linestyle = "--"
    point_estimate_linewidth = 2.5

    ncols = 4
    nrows = int(np.ceil(len(param_names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.atleast_1d(axes).flatten()
    plot_axes = axes[: len(param_names)]

    # `plot.max_subplots` (default 40) otherwise silently truncates the plot
    # instead of raising once `param_names` exceeds it (same guard as
    # `plot_trace_and_posterior`). `point_estimate=None, hdi_prob="hide"` drop each
    # call's own text/HDI-bar annotations, which would otherwise overlap unreadably
    # once both methods' calls land on the same axes.
    with az.rc_context(rc={"plot.max_subplots": None}):
        az.plot_posterior(
            pymc_idata,
            var_names=param_names,
            color=method_colors[pymc_label],
            point_estimate=None,
            hdi_prob="hide",
            ax=plot_axes,
        )
        az.plot_posterior(
            hmc_idata,
            var_names=param_names,
            color=method_colors[hmc_label],
            point_estimate=None,
            hdi_prob="hide",
            ax=plot_axes,
        )

    line_handles = [
        Line2D(
            [],
            [],
            color=method_colors[pymc_label],
            linestyle=bayesian_linestyle,
            label=f"{pymc_label} mean",
        ),
        Line2D(
            [],
            [],
            color=method_colors[hmc_label],
            linestyle=bayesian_linestyle,
            label=f"{hmc_label} mean",
        ),
    ]
    if map_estimate:
        line_handles.append(
            Line2D(
                [],
                [],
                color=method_colors["MAP"],
                linestyle=point_estimate_linestyle,
                linewidth=point_estimate_linewidth,
                label="MAP",
            )
        )
    if gd_fit:
        line_handles.append(
            Line2D(
                [],
                [],
                color=method_colors["Gradient descent"],
                linestyle=point_estimate_linestyle,
                linewidth=point_estimate_linewidth,
                label="Gradient descent",
            )
        )
    if priors:
        line_handles.append(Line2D([], [], color="gray", linestyle="-", label="Prior bounds"))

    for ax, name in zip(plot_axes, param_names, strict=True):
        if name in pymc_idata.posterior.data_vars:
            ax.axvline(
                float(pymc_idata.posterior[name].mean()),
                color=method_colors[pymc_label],
                linestyle=bayesian_linestyle,
            )
        if name in hmc_idata.posterior.data_vars:
            ax.axvline(
                float(hmc_idata.posterior[name].mean()),
                color=method_colors[hmc_label],
                linestyle=bayesian_linestyle,
            )
        if name in map_estimate:
            ax.axvline(
                map_estimate[name],
                color=method_colors["MAP"],
                linestyle=point_estimate_linestyle,
                linewidth=point_estimate_linewidth,
            )
        if name in gd_fit:
            ax.axvline(
                gd_fit[name],
                color=method_colors["Gradient descent"],
                linestyle=point_estimate_linestyle,
                linewidth=point_estimate_linewidth,
            )
        bounds = priors.get(name)
        if bounds is not None:
            for bound in bounds:
                ax.axvline(bound, color="gray", linestyle="-")

    plot_axes[0].legend(handles=line_handles, fontsize=8)

    for ax in axes[len(param_names) :]:
        ax.axis("off")

    fig.suptitle(f"Posterior comparison: {pymc_label} vs. {hmc_label}")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

    return fig


def plot_posterior_across_plots(
    plot_ids: list[str],
    literature_source: str,
    error_terms: str,
    method: str = "demetropolisz",
    include_process_error: bool = True,
    param_names: list[str] | None = None,
    priors: dict[str, tuple[float, float]] | None = None,
    plot_colors: dict[str, str] | None = None,
) -> Figure:
    """Overlay parameters' posterior distributions across multiple plots.

    Each plot's saved run path is built the same way `plot_and_save` does; a
    plot with no saved run for the given combination is skipped with a
    printed note.

    Parameters
    ----------
    plot_ids : list[str]
        ICP plot identifiers (or "solling") to compare.
    literature_source : str
        Literature source the plots were calibrated with (e.g. "Forrester",
        "Trotsiuk"). Ignored for the "solling" plot id, which isn't
        literature-source-specific (see `plot_and_save`).
    error_terms : str
        Calibration scenario name (a key of `bayesian_config.ERROR_MODES`), used
        as the saved-run directory segment.
    method : str
        Sampler subdirectory to load from: "demetropolisz" (PyMC) or "nuts" (HMC).
    include_process_error : bool
        Whether to read from `results/latent_calibration_sweep` (True) or
        `results/calibration_sweep` (False) — see `plot_and_save`.
    param_names : list[str] | None
        Parameters to compare. Defaults to the union of physiological (i.e.
        excluding `err_*` observation-noise sigma terms) posterior variables
        across all given traces — every physiological parameter calibrated by
        at least one plot. A plot missing a posterior for a given parameter is
        left out of that parameter's subplot.
    priors : dict[str, tuple[float, float]] | None
        Prior (lower, upper) bounds per parameter, drawn as gray vertical
        lines. Parameters without a matching entry are plotted without prior
        lines.
    plot_colors : dict[str, str] | None
        Override color per plot label. Missing entries fall back to
        matplotlib's default color cycle.

    Returns
    -------
    Figure
        The comparison figure.
    """
    output_dir = os.path.join(
        results_data_folder,
        "latent_calibration_sweep" if include_process_error else "calibration_sweep",
    )
    resolved_paths: dict[str, str] = {}
    for plot_id in plot_ids:
        if plot_id == "solling":
            path = os.path.join(output_dir, f"{plot_id}/{error_terms}/{method}/inference_data.nc")
        else:
            path = os.path.join(
                output_dir,
                f"{plot_id}/{literature_source}/{error_terms}/{method}/inference_data.nc",
            )
        if not os.path.exists(path):
            print(f"Note: no saved run at {path} — skipping plot_id {plot_id!r}")
            continue
        resolved_paths[plot_id] = path

    if not resolved_paths:
        raise ValueError(
            f"No saved {method!r} runs found for any of {plot_ids} "
            f"(literature_source={literature_source!r}, error_terms={error_terms!r})"
        )

    priors = priors or {}
    plot_colors = plot_colors or {}
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    idatas = {label: az.from_netcdf(path) for label, path in resolved_paths.items()}
    colors = {
        label: plot_colors.get(label, color_cycle[i % len(color_cycle)])
        for i, label in enumerate(idatas)
    }

    if param_names is None:
        param_names = sorted(
            {
                name
                for idata in idatas.values()
                for name in cast(Any, idata).posterior.data_vars
                if not name.startswith("err_")
            }
        )

    ncols = min(4, len(param_names))
    nrows = int(np.ceil(len(param_names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.atleast_1d(axes).flatten()

    line_handles: dict[str, Line2D] = {}
    with az.rc_context(rc={"plot.max_subplots": None}):
        for ax, param_name in zip(axes, param_names, strict=False):
            plotted = False
            for label, idata in idatas.items():
                if param_name not in cast(Any, idata).posterior.data_vars:
                    continue

                color = colors[label]
                az.plot_posterior(
                    idata,
                    var_names=[param_name],
                    color=color,
                    point_estimate=None,
                    hdi_prob="hide",
                    ax=np.array([ax]),
                )
                ax.axvline(float(idata.posterior[param_name].mean()), color=color, linestyle=":")
                line_handles.setdefault(
                    label, Line2D([], [], color=color, linestyle=":", label=label)
                )
                plotted = True

            bounds = priors.get(param_name)
            if bounds is not None:
                for bound in bounds:
                    ax.axvline(bound, color="gray", linestyle="-")

            ax.set_title(param_name)
            if not plotted:
                ax.axis("off")

    if not line_handles:
        raise ValueError("No posterior for any of the given parameters found in any plot")

    if priors:
        line_handles.setdefault(
            "Prior bounds", Line2D([], [], color="gray", linestyle="-", label="Prior bounds")
        )

    for ax in axes[len(param_names) :]:
        ax.axis("off")

    fig.legend(handles=list(line_handles.values()), loc="upper right", fontsize=8)
    fig.suptitle("Posterior comparison across plots")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

    plt.show()
    return fig


def plot_convergence_comparison(
    fit_params: list[str],
    pymc_inference_path: str | None = None,
    hmc_inference_path: str | None = None,
    param_names: list[str] | None = None,
    excluded_error_names: frozenset[str] = DIAGNOSTIC_ONLY_ERROR_NAMES,
    include_bayesian: bool = True,
    include_hmc: bool = True,
    method_colors: dict[str, str] | None = None,
) -> tuple[Figure, pd.DataFrame]:
    """Compare PyMC (DEMetropolisZ) and/or HMC (NUTS) convergence diagnostics per parameter.

    Parameters
    ----------
    pymc_inference_path : str | None
        Path to the saved PyMC `inference_data.nc`. Required if `include_bayesian`.
    hmc_inference_path : str | None
        Path to the saved HMC (NumPyro) `numpyro_inference_data.nc`. Required if `include_hmc`.
    param_names : list[str] | None
        Parameters to compare. Defaults to `fit_params` plus one `err_{var}` per
        variable in `PLOT_VARIABLES` that isn't in `excluded_error_names`.
    excluded_error_names : frozenset[str]
        `err_*` names to leave out of the default `param_names` — the ones with no
        sigma prior (and so no posterior) in the run being plotted. Defaults to
        `DIAGNOSTIC_ONLY_ERROR_NAMES`; pass the scenario's own entry from
        `bayesian_config.ERROR_MODES` instead when plotting a specific named
        calibration scenario (e.g. `"all_error_terms"` fits every `err_*`, so its
        exclusion set is empty). Ignored if `param_names` is given explicitly.
    include_bayesian, include_hmc : bool
        Whether to include each method. Tuning/warmup draws are read back
        from `posterior.attrs["tuning_steps"]` in the saved file itself
        (`None` for older saved files that predate this being stashed).
    method_colors : dict[str, str] | None
        Override bar color per method name (`"PyMC (DEz)"`/`"HMC (NUTS)"`). Missing
        entries fall back to the default blue/orange.

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
        param_names = fit_params + [
            f"err_{var}" for var in PLOT_VARIABLES if f"err_{var}" not in excluded_error_names
        ]

    default_colors = {"PyMC (DEz)": "tab:blue", "HMC (NUTS)": "tab:orange"}
    method_colors = {**default_colors, **(method_colors or {})}

    colors = {}
    summaries = []
    if include_bayesian:
        assert pymc_inference_path is not None
        pymc_summary = load_convergence_summary(pymc_inference_path, param_names)
        pymc_summary["method"] = "PyMC (DEz)"
        summaries.append(pymc_summary)
        colors["PyMC (DEz)"] = method_colors["PyMC (DEz)"]

    if include_hmc:
        assert hmc_inference_path is not None
        hmc_summary = load_convergence_summary(hmc_inference_path, param_names)
        hmc_summary["method"] = "HMC (NUTS)"
        summaries.append(hmc_summary)
        colors["HMC (NUTS)"] = method_colors["HMC (NUTS)"]

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
    map_output_dir: str | None = None,
    param_names: list[str] = FIT_PARAMS,
    include_gradient_descent: bool = True,
    include_bayesian: bool = True,
    include_hmc: bool = True,
    include_map: bool = True,
    gd_cache_dir: str | None = None,
    method_colors: dict[str, str] | None = None,
) -> tuple[Figure, pd.DataFrame]:
    """Compare each parameter's fitted value across gradient descent, PyMC (DEz), HMC (NUTS), MAP.

    One small subplot per parameter (values span very different scales, e.g.
    `mS` ~1e-4 vs. `MaxAge` ~400, so a single shared-axis bar chart would hide
    most of them), each with one bar per included method.

    Parameters
    ----------
    pymc_inference_path : str | None
        Path to the saved PyMC `inference_data.nc`. Required if `include_bayesian`.
    hmc_inference_path : str | None
        Path to the saved HMC (NumPyro) `numpyro_inference_data.nc`. Required if `include_hmc`.
    map_output_dir : str | None
        Directory containing a saved `map_estimate.json` from a prior `run_map_analysis`
        run. Required if `include_map`. Parameters missing from the saved estimate (e.g.
        error/sigma terms dropped for that run) are silently skipped.
    param_names : list[str]
        Parameters to compare. A name a given method didn't fit (e.g.
        `INITIAL_STATE_PARAMS`/`PROCESS_ERROR_PARAM_NAMES` for gradient descent,
        which has no process-error mechanism, or an err_* term excluded from
        that run's error mode) is silently skipped for that method's bars.
    include_gradient_descent, include_bayesian, include_hmc, include_map : bool
        Whether to include each method.
    gd_cache_dir : str | None
        Directory holding a saved gradient descent fit (see `get_gradient_descent_fit`).
        Required if `include_gradient_descent`; pass the same directory used for
        `plot_comparison`'s `gd_cache_dir` to compare against the same fit.
    method_colors : dict[str, str] | None
        Override bar color per method name (`"Gradient descent"`/`"PyMC (DEz)"`/
        `"HMC (NUTS)"`/`"MAP"`). Missing entries fall back to the default
        green/blue/orange/red.

    Returns
    -------
    tuple[Figure, pd.DataFrame]
        The comparison figure and a long-form table (`parameter`, `method`, `value`).
    """
    if not (include_gradient_descent or include_bayesian or include_hmc or include_map):
        raise ValueError(
            "At least one of include_gradient_descent/include_bayesian/include_hmc/"
            "include_map must be True"
        )
    if include_gradient_descent and gd_cache_dir is None:
        raise ValueError("gd_cache_dir is required when include_gradient_descent is True")
    if include_bayesian and pymc_inference_path is None:
        raise ValueError("pymc_inference_path is required when include_bayesian is True")
    if include_hmc and hmc_inference_path is None:
        raise ValueError("hmc_inference_path is required when include_hmc is True")
    if include_map and map_output_dir is None:
        raise ValueError("map_output_dir is required when include_map is True")

    default_colors = {
        "Gradient descent": "tab:green",
        "PyMC (DEz)": "tab:blue",
        "HMC (NUTS)": "tab:orange",
        "MAP": "tab:red",
    }
    method_colors = {**default_colors, **(method_colors or {})}

    colors = {}
    rows = []

    if include_gradient_descent:
        assert gd_cache_dir is not None
        fitted_params = get_gradient_descent_fit(gd_cache_dir)
        for name in param_names:
            # Gradient descent has no process-error mechanism (see
            # GradientDescentConfig), so INITIAL_STATE_PARAMS/PROCESS_ERROR_PARAM_NAMES
            # names in param_names (see plot_and_save) have no value here — skipped,
            # matching the include_map branch's own missing-value handling below.
            if name in fitted_params:
                rows.append(
                    {"parameter": name, "method": "Gradient descent", "value": fitted_params[name]}
                )
        colors["Gradient descent"] = method_colors["Gradient descent"]

    if include_bayesian:
        assert pymc_inference_path is not None
        pymc_summary = load_convergence_summary(pymc_inference_path, param_names)
        for _, row in pymc_summary.iterrows():
            rows.append(
                {"parameter": row["parameter"], "method": "PyMC (DEz)", "value": row["mean"]}
            )
        colors["PyMC (DEz)"] = method_colors["PyMC (DEz)"]

    if include_hmc:
        assert hmc_inference_path is not None
        hmc_summary = load_convergence_summary(hmc_inference_path, param_names)
        for _, row in hmc_summary.iterrows():
            rows.append(
                {"parameter": row["parameter"], "method": "HMC (NUTS)", "value": row["mean"]}
            )
        colors["HMC (NUTS)"] = method_colors["HMC (NUTS)"]

    if include_map:
        assert map_output_dir is not None
        map_estimate, _ = load_map_estimate(os.path.join(map_output_dir, "map_estimate.json"))
        for name in param_names:
            if name in map_estimate:
                rows.append({"parameter": name, "method": "MAP", "value": map_estimate[name]})
        colors["MAP"] = method_colors["MAP"]

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


def plot_and_save(
    plot_id: str,
    plot_output_dir: str,
    error_terms: str,
    literature_source: str,
    method: str = "",
    include_process_error: bool = True,
    include_bayesian: bool = True,
    include_hmc: bool = True,
    include_gradient_descent: bool = True,
    include_map: bool = True,
):
    """Plot and save comparison/convergence/trace/posterior figures for a plot.

    Parameters
    ----------
    plot_id : str
        ICP plot identifier.
    plot_output_dir : str
        Directory to save the plots in.
    error_terms : str
        Calibration scenario name, used both for the saved filenames and to look up
        which `err_*` names were excluded from fitting via `bayesian_config.ERROR_MODES`
        (falls back to `DIAGNOSTIC_ONLY_ERROR_NAMES` if not a recognized scenario), so
        the convergence plot only asks for `err_*` posteriors that actually exist.
    literature_source : str
        Source of the literature for the saved filenames.
    include_bayesian, include_hmc, include_gradient_descent, include_map : bool
        Whether to include each method's results — forwarded to `plot_comparison`,
        `plot_convergence_comparison`, `plot_posterior_comparison`, and
        `plot_parameter_value_comparison`.
    method : str
        Calibration method to use.
    include_process_error : bool
        Selects both which saved sweep to read and how its calibrated parameters
        are determined. `False` (old method) reads `results/calibration_sweep`
        and selects calibrated parameters the original way, via
        `fit_params_for_mode`. `True` (new method) reads
        `results/latent_calibration_sweep` (whose runs fit WS0/WR0/WF0 —
        `INITIAL_STATE_PARAMS` — and their perr_WS/perr_WR/perr_WF sigmas —
        `PROCESS_ERROR_PARAM_NAMES` — as latent process-error parameters) and
        determines calibrated parameters straight from the saved results
        themselves instead (see `_calibrated_params_from_results`), since that
        sweep's per-mode saved runs may not fully agree with the current
        `fit_params_for_mode`.
    """
    plot_output_dir = os.path.join(plot_output_dir, plot_id)
    os.makedirs(plot_output_dir, exist_ok=True)

    output_dir = os.path.join(
        results_data_folder,
        "latent_calibration_sweep" if include_process_error else "calibration_sweep",
    )

    # Mirrors the calibration_sweep directory layout (site/[literature_source/]mode),
    # so filenames stay unique across every (literature_source, error_terms) combination
    # saved into the same plot_id folder instead of overwriting each other. Solling's
    # data is hand-curated and never varies by literature_source (see
    # resolve_source_file_path), so it's left out of solling's filenames.
    combo_parts = [error_terms] if plot_id == "solling" else [literature_source, error_terms]
    combo_parts = [part for part in combo_parts if part]
    combo_prefix = "_".join(combo_parts) + "_" if combo_parts else ""

    if plot_id == "solling":
        _bayesian_output_dir = os.path.join(output_dir, f"{plot_id}/{error_terms}/demetropolisz")
        _hmc_output_dir = os.path.join(output_dir, f"{plot_id}/{error_terms}/nuts")
        _gradient_descent_dir = os.path.join(
            output_dir, f"{plot_id}/{error_terms}/gradient_descent"
        )
        _map_output_dir = os.path.join(output_dir, f"{plot_id}/{error_terms}/map")
        _file_path = os.path.join(output_dir, f"{plot_id}/{plot_id}_data.xlsx")
    else:
        _bayesian_output_dir = os.path.join(
            output_dir, f"{plot_id}/{literature_source}/{error_terms}/demetropolisz"
        )
        _hmc_output_dir = os.path.join(
            output_dir, f"{plot_id}/{literature_source}/{error_terms}/nuts"
        )
        _gradient_descent_dir = os.path.join(
            output_dir, f"{plot_id}/{literature_source}/{error_terms}/gradient_descent"
        )
        _map_output_dir = os.path.join(
            output_dir, f"{plot_id}/{literature_source}/{error_terms}/map"
        )
        _file_path = os.path.join(output_dir, f"{plot_id}/{literature_source}/{plot_id}_data.xlsx")

    mode_name = error_terms if error_terms in ERROR_MODES else "biomass_only"

    if not include_process_error:
        # Old method: results/calibration_sweep's runs were all saved against
        # this same fixed selection — the hardcoded FIT_PARAMS list for solling
        # (hand-curated data, not mode-dependent), or otherwise every parameter
        # with both a min and max in the file's own param_bound sheet.
        fit_params = (
            FIT_PARAMS if plot_id == "solling" else list(load_priors_from_file(_file_path))
        )
        trace_param_names = fit_params
    else:
        # New method: results/latent_calibration_sweep's saved runs may
        # predate the current fit_params_for_mode (see
        # _calibrated_params_from_results), so calibrated parameters are
        # discovered straight from whichever saved results are included
        # instead. INITIAL_STATE_PARAMS/PROCESS_ERROR_PARAM_NAMES (WS0/WR0/WF0
        # and their perr_* sigmas) are in the candidates so they're picked up
        # automatically wherever a run actually fit them.
        _process_error_names = frozenset(INITIAL_STATE_PARAMS) | PROCESS_ERROR_PARAM_NAMES
        _candidate_names = frozenset(Params._fields) | _process_error_names
        _calibrated_params = _calibrated_params_from_results(
            _candidate_names,
            _bayesian_output_dir,
            _hmc_output_dir,
            _map_output_dir,
            _gradient_descent_dir,
            include_bayesian,
            include_hmc,
            include_map,
            include_gradient_descent,
        )
        if not _calibrated_params:
            # Nothing saved yet for any included method — fall back to the
            # config-computed set so a first-ever plotting call still works.
            _calibrated_params = set(fit_params_for_mode(_file_path, mode_name))

        # Physiology-only names (a real Params field) — gradient descent has no
        # process-error mechanism, so INITIAL_STATE_PARAMS/PROCESS_ERROR_PARAM_NAMES
        # are kept out of fit_params itself (it's also passed to run_gradient_descent_model).
        fit_params = sorted(_calibrated_params & set(Params._fields))
        trace_param_names = sorted(_calibrated_params)

    _fig, _metrics_df = plot_comparison(
        _file_path,
        fit_params,
        _bayesian_output_dir,
        _hmc_output_dir,
        _map_output_dir,
        include_gradient_descent=include_gradient_descent,
        include_bayesian=include_bayesian,
        include_hmc=include_hmc,
        include_map=include_map,
        gd_cache_dir=_gradient_descent_dir,
    )
    _fig.savefig(
        os.path.join(plot_output_dir, f"{combo_prefix}prediction_comparison_{plot_id}.png"),
        dpi=200,
        bbox_inches="tight",
    )

    _conv_fig, _conv_df = plot_convergence_comparison(
        pymc_inference_path=os.path.join(_bayesian_output_dir, "inference_data.nc"),
        hmc_inference_path=os.path.join(_hmc_output_dir, "inference_data.nc"),
        include_bayesian=include_bayesian,
        include_hmc=include_hmc,
        fit_params=trace_param_names,
        excluded_error_names=ERROR_MODES.get(error_terms, DIAGNOSTIC_ONLY_ERROR_NAMES),
    )
    # print(_conv_df)
    # plt.show()
    _conv_fig.savefig(
        os.path.join(plot_output_dir, f"{combo_prefix}convergence_comparison_{plot_id}.png"),
        dpi=200,
        bbox_inches="tight",
    )
    # plt.show()
    plt.close(_conv_fig)

    if include_bayesian or include_hmc:
        # WS0/WR0/WF0 aren't real param_bound/error_param rows in the file (they're
        # runtime-injected Normal priors, not Uniform — see run_map_analysis) and
        # load_priors_from_file raises on a name it can't find, so they're excluded
        # here; perr_WS/WR/WF are real error_param rows and stay included.
        _priors = load_priors_from_file(
            _file_path, [name for name in trace_param_names if name not in INITIAL_STATE_PARAMS]
        )
        for _method_name, _method_included, _method_output_dir in (
            ("demetropolisz", include_bayesian, _bayesian_output_dir),
            ("nuts", include_hmc, _hmc_output_dir),
        ):
            if not _method_included:
                continue
            _trace_fig, _posterior_fig = plot_trace_and_posterior(
                os.path.join(_method_output_dir, "inference_data.nc"), trace_param_names, _priors
            )
            _trace_fig.savefig(
                os.path.join(plot_output_dir, f"{combo_prefix}trace_{_method_name}_{plot_id}.png"),
                dpi=200,
                bbox_inches="tight",
            )
            _posterior_fig.savefig(
                os.path.join(
                    plot_output_dir, f"{combo_prefix}posterior_{_method_name}_{plot_id}.png"
                ),
                dpi=200,
                bbox_inches="tight",
            )
            plt.close(_trace_fig)
            plt.close(_posterior_fig)

    if include_bayesian and include_hmc:
        _posterior_comparison_fig = plot_posterior_comparison(
            pymc_inference_path=os.path.join(_bayesian_output_dir, "inference_data.nc"),
            hmc_inference_path=os.path.join(_hmc_output_dir, "inference_data.nc"),
            param_names=trace_param_names,
            priors=_priors,
            map_output_dir=_map_output_dir if include_map else None,
            gd_cache_dir=_gradient_descent_dir if include_gradient_descent else None,
        )
        _posterior_comparison_fig.savefig(
            os.path.join(plot_output_dir, f"{combo_prefix}posterior_comparison_{plot_id}.png"),
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(_posterior_comparison_fig)

    _param_fig, _param_df = plot_parameter_value_comparison(
        pymc_inference_path=os.path.join(_bayesian_output_dir, "inference_data.nc"),
        hmc_inference_path=os.path.join(_hmc_output_dir, "inference_data.nc"),
        map_output_dir=_map_output_dir,
        param_names=trace_param_names,
        include_gradient_descent=include_gradient_descent,
        include_bayesian=include_bayesian,
        include_hmc=include_hmc,
        include_map=include_map,
        gd_cache_dir=_gradient_descent_dir,
    )
    # print(_param_df)
    # plt.show()
    _param_fig.savefig(
        os.path.join(plot_output_dir, f"{combo_prefix}parameter_value_comparison_{plot_id}.png"),
        dpi=200,
        bbox_inches="tight",
    )
    # plt.close(_param_fig)

    print(f"Saved plots to {plot_output_dir}")
    gc.collect()
    jax.clear_caches()


if __name__ == "__main__":
    plot_ids = ["solling"]
    plot_ids = species_plot_ids["Picea abies"]

    plot_output_dir = os.path.join(data_folder, "results/comparison_plots")

    os.makedirs(plot_output_dir, exist_ok=True)

    _include_bayesian = True
    _include_hmc = True
    _include_gradient_descent = True
    _include_map = False
    _include_process_error = False

    for error_terms in ERROR_MODES:
        print(f"Processing error_terms={error_terms}...")
        for plot_id in plot_ids:
            print(f"Processing plot_id={plot_id}...")
            try:
                plot_and_save(
                    plot_id=plot_id,
                    plot_output_dir=plot_output_dir,
                    error_terms=error_terms,
                    literature_source="Forrester",
                    include_process_error=_include_process_error,
                    include_bayesian=_include_bayesian,
                    include_hmc=_include_hmc,
                    include_gradient_descent=_include_gradient_descent,
                    include_map=_include_map,
                )
            except Exception as e:
                print(f"Error processing plot_id={plot_id}: {e}")

        plot_posterior_across_plots(
            plot_ids=plot_ids,
            literature_source="Forrester",
            error_terms=error_terms,
            method="demetropolisz",
            include_process_error=_include_process_error,
        )
