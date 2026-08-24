"""Functions to save and load inference results and prediction uncertainty bands."""

import json
import os
from typing import Any, cast

import arviz as az
import jax.numpy as jnp
import numpy as np


def save_predictions(
    predictions: dict[str, tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]],
    output_dir: str,
    filename: str = "predictions.npz",
) -> str:
    """
    Save prediction uncertainty bands to a compressed .npz file.

    Parameters
    ----------
    predictions : dict[str, tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]
        Dictionary mapping variable names to (mean_pred, lower_pred, upper_pred),
        as returned by `predict_with_uncertainty`.
    output_dir : str
        Directory to save the file in (created if missing).
    filename : str
        Name of the .npz file.

    Returns
    -------
    str
        Full path to the saved file.
    """
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, filename)
    arrays: dict[str, np.ndarray] = {}
    for var_name, (mean_pred, lower_pred, upper_pred) in predictions.items():
        arrays[f"{var_name}_mean"] = np.asarray(mean_pred)
        arrays[f"{var_name}_lower"] = np.asarray(lower_pred)
        arrays[f"{var_name}_upper"] = np.asarray(upper_pred)
    np.savez(file_path, **arrays)  # ty: ignore[invalid-argument-type]  # pyright: ignore[reportArgumentType]
    return file_path


def save_results(
    mcmc: az.InferenceData,
    output_dir: str,
    predictions: dict[str, tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]] | None = None,
    filename: str = "inference_data.nc",
) -> az.InferenceData:
    """
    Save inference data and, if available, prediction uncertainty bands to disk.

    Parameters
    ----------
    mcmc : az.InferenceData
        Fitted inference data (from PyMC's `pm.sample` or `az.from_numpyro`).
    output_dir : str
        Directory to save results in.
    predictions : dict[str, tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]] | None
        Prediction uncertainty bands from `predict_with_uncertainty`, if computed.

    Returns
    -------
    az.InferenceData
        The inference data, so callers can reuse it for plotting without recomputing.
    """
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, filename)
    mcmc.to_netcdf(file_path)

    if predictions is not None:
        save_predictions(predictions, output_dir)

    return mcmc


def save_map_estimate(
    map_estimate: dict[str, float],
    logp: float,
    output_dir: str,
    filename: str = "map_estimate.json",
) -> str:
    """
    Save a MAP point estimate and its log posterior density to a JSON file.

    Parameters
    ----------
    map_estimate : dict[str, float]
        Parameter names mapped to their maximum a posteriori values.
    logp : float
        Log posterior density at the estimate, for comparing optimisation runs.
    output_dir : str
        Directory to save the file in (created if missing).
    filename : str
        Name of the JSON file.

    Returns
    -------
    str
        Full path to the saved file.
    """
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, filename)
    with open(file_path, "w") as f:
        json.dump({"logp": logp, "map_estimate": map_estimate}, f, indent=2)
    return file_path


def save_laplace_covariance(
    covariance: np.ndarray,
    names: list[str],
    output_dir: str,
    filename: str = "laplace_covariance.npz",
) -> str:
    """
    Save the unconstrained-space covariance of a Laplace approximation.

    Parameters
    ----------
    covariance : np.ndarray
        Covariance matrix from `fit_laplace`.
    names : list[str]
        Parameter names indexing the matrix axes.
    output_dir : str
        Directory to save the file in (created if missing).
    filename : str
        Name of the .npz file.

    Returns
    -------
    str
        Full path to the saved file.
    """
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, filename)
    np.savez(file_path, covariance=covariance, names=np.asarray(names))
    return file_path


def load_laplace_covariance(file_path: str) -> tuple[np.ndarray, list[str]]:
    """
    Load a covariance matrix previously saved with `save_laplace_covariance`.

    Parameters
    ----------
    file_path : str
        Path to the .npz file.

    Returns
    -------
    tuple[np.ndarray, list[str]]
        The covariance matrix and the parameter names indexing its axes.
    """
    data = np.load(file_path)
    return data["covariance"], [str(name) for name in data["names"]]


def load_map_estimate(file_path: str) -> tuple[dict[str, float], float]:
    """
    Load a MAP estimate previously saved with `save_map_estimate`.

    Parameters
    ----------
    file_path : str
        Path to the JSON file.

    Returns
    -------
    tuple[dict[str, float], float]
        The parameter estimates and their log posterior density.
    """
    with open(file_path) as f:
        state = json.load(f)
    return state["map_estimate"], state["logp"]


def save_gradient_descent_result(
    fitted_params: dict[str, float],
    output_dir: str,
    filename: str = "gradient_descent_result.json",
) -> str:
    """
    Save a gradient descent fit's parameter values to a JSON file.

    Parameters
    ----------
    fitted_params : dict[str, float]
        Parameter names mapped to their fitted (converged) values.
    output_dir : str
        Directory to save the file in (created if missing).
    filename : str
        Name of the JSON file.

    Returns
    -------
    str
        Full path to the saved file.
    """
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, filename)
    with open(file_path, "w") as f:
        json.dump({"fitted_params": fitted_params}, f, indent=2)
    return file_path


def load_gradient_descent_result(file_path: str) -> dict[str, float]:
    """
    Load fitted parameters previously saved with `save_gradient_descent_result`.

    Parameters
    ----------
    file_path : str
        Path to the JSON file.

    Returns
    -------
    dict[str, float]
        The fitted parameter values.
    """
    with open(file_path) as f:
        state = json.load(f)
    return state["fitted_params"]


def load_inference_data(file_path: str) -> az.InferenceData:
    """
    Load arviz InferenceData previously saved with `save_inference_data`.

    Parameters
    ----------
    file_path : str
        Path to the NetCDF file.

    Returns
    -------
    az.InferenceData
        Loaded inference data.
    """
    return az.from_netcdf(file_path)


def save_checkpoint(
    idata: az.InferenceData,
    draws_done: int,
    initvals: list[dict[str, float]],
    checkpoint_dir: str,
) -> None:
    """
    Save the posterior draws sampled so far and per-chain resume state to disk.

    Parameters
    ----------
    idata : az.InferenceData
        Posterior draws accumulated across all completed sampling chunks.
    draws_done : int
        Number of post-tuning draws per chain completed so far.
    initvals : list[dict[str, float]]
        Last sampled parameter values for each chain, used to seed the next chunk.
    checkpoint_dir : str
        Directory to save the checkpoint files in (created if missing).
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    data_path = os.path.join(checkpoint_dir, "checkpoint_inference_data.nc")
    # Write to a temp file and rename into place, so a checkpoint being loaded from
    # (its file handle still open) or a crash mid-write never corrupts the last good one.
    tmp_data_path = data_path + ".tmp"
    idata.to_netcdf(tmp_data_path)
    os.replace(tmp_data_path, data_path)

    state = {"draws_done": draws_done, "initvals": initvals}
    state_path = os.path.join(checkpoint_dir, "checkpoint_state.json")
    tmp_state_path = state_path + ".tmp"
    with open(tmp_state_path, "w") as f:
        json.dump(state, f)
    os.replace(tmp_state_path, state_path)


def load_checkpoint(
    checkpoint_dir: str,
) -> tuple[az.InferenceData, int, list[dict[str, float]]] | None:
    """
    Load a checkpoint previously saved with `save_checkpoint`, if one exists.

    Parameters
    ----------
    checkpoint_dir : str
        Directory passed to `save_checkpoint`.

    Returns
    -------
    tuple[az.InferenceData, int, list[dict[str, float]]] | None
        (posterior draws so far, draws completed per chain, per-chain resume values),
        or None if no checkpoint is found in `checkpoint_dir`.
    """
    state_path = os.path.join(checkpoint_dir, "checkpoint_state.json")
    data_path = os.path.join(checkpoint_dir, "checkpoint_inference_data.nc")
    if not (os.path.exists(state_path) and os.path.exists(data_path)):
        return None
    with open(state_path) as f:
        state = json.load(f)
    # Load eagerly and close the file handle, so the checkpoint file can be
    # overwritten later in the same process (e.g. by the next save_checkpoint call).
    idata = az.from_netcdf(data_path).load()
    _delist_attrs(idata)
    return idata, state["draws_done"], state["initvals"]


def _delist_attrs(idata: az.InferenceData) -> None:
    """Convert ndarray-valued group attrs (produced by a netcdf round-trip) back to lists.

    `az.concat` merges same-named attrs that differ across calls into a list under
    `combined_<attr>`, appending to it on each subsequent merge. A netcdf round-trip
    turns that list into a numpy array, which then breaks the next `.append` call —
    so this must be undone right after loading, before the checkpoint is concatenated
    with the next sampling chunk.
    """
    for group in idata.groups():
        attrs = getattr(idata, group).attrs
        for key, value in attrs.items():
            if isinstance(value, np.ndarray):
                attrs[key] = value.tolist()


def load_predictions(file_path: str) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Load prediction uncertainty bands previously saved with `save_predictions`.

    Parameters
    ----------
    file_path : str
        Path to the .npz file.

    Returns
    -------
    dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]
        Dictionary mapping variable names to (mean_pred, lower_pred, upper_pred).
    """
    data = np.load(file_path)
    var_names = sorted({key.rsplit("_", 1)[0] for key in data.files})
    return {
        var_name: (data[f"{var_name}_mean"], data[f"{var_name}_lower"], data[f"{var_name}_upper"])
        for var_name in var_names
    }
