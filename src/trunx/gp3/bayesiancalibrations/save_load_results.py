"""Functions to save and load inference results and prediction uncertainty bands."""

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
