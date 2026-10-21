"""Prepare ICP plot input files and run Bayesian calibration for each, in parallel."""

import argparse
import os
import shutil
import tempfile
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, cast

import pandas as pd
import polars as pl

from trunx.config import clean_data_folder, results_data_folder, threepg_data_folder
from trunx.gp3.bayesiancalibrations.bayesian_comparison_plots import run_dir
from trunx.gp3.bayesiancalibrations.bayesian_config import ERROR_MODES
from trunx.gp3.bayesiancalibrations.load_files import (
    fit_params_for_mode,
    load_observations_from_file,
    load_priors_from_file,
)
from trunx.gp3.bayesiancalibrations.pymc_param_est import plot_saved_results, run_pymc_analysis
from trunx.gp3.bayesiancalibrations.save_load_results import load_inference_data
from trunx.gp3.create_combined_inputs import create_plot_input_file, load_plot_tables
from trunx.gp3.prepare_data import prepare_data


def get_available_cpus() -> int:
    """Return the number of CPUs available to this process."""
    sched_getaffinity = getattr(os, "sched_getaffinity", None)
    if sched_getaffinity is not None:
        return len(sched_getaffinity(0))
    return os.cpu_count() or 1


def write_calibrated_input(
    file_path: str, plot_id: str, literature_source: str, error_mode: str
) -> str:
    """Copy a plot's input file with the calibration sweep's NUTS posterior means.

    Parameters
    ----------
    file_path : str
        The plot's uncalibrated input file.
    plot_id : str
        ICP plot identifier.
    literature_source : str
        Literature source of the calibration run.
    error_mode : str
        Error mode of the calibration run (a key of `bayesian_config.ERROR_MODES`).

    Returns
    -------
    str
        Path to the calibrated input Excel file, or `file_path` if the plot has no
        calibration run.
    """
    inference_path = os.path.join(
        run_dir(plot_id, literature_source, error_mode, include_process_error=False),
        "nuts",
        "inference_data.nc",
    )
    if not os.path.exists(inference_path):
        warnings.warn(
            f"No calibration of plot {plot_id} at {inference_path}, using its uncalibrated "
            "parameters",
            stacklevel=3,
        )
        return file_path
    posterior = cast(Any, load_inference_data(inference_path)).posterior

    parameters = pd.read_excel(file_path, sheet_name="parameters")
    species_columns = [column for column in parameters.columns if column != "parameter"]
    for name in parameters["parameter"]:
        if name in posterior.data_vars:
            parameters.loc[parameters["parameter"] == name, species_columns] = float(
                posterior[name].mean()
            )

    plot_dir = os.path.dirname(file_path)
    calibrated_path = os.path.join(plot_dir, f"{plot_id}_calibrated_{error_mode}_data.xlsx")
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".xlsx", prefix=f"{plot_id}_", dir=plot_dir)
    os.close(tmp_fd)
    try:
        shutil.copy(file_path, tmp_path)
        with pd.ExcelWriter(
            tmp_path, engine="openpyxl", mode="a", if_sheet_exists="replace"
        ) as writer:
            parameters.to_excel(writer, sheet_name="parameters", index=False)
        os.replace(tmp_path, calibrated_path)
    except Exception:
        os.remove(tmp_path)
        raise

    return calibrated_path


def prepare_plot_input(
    plot_id: str,
    literature_source: str,
    forest_source: str = "ICP",
    calibrated: bool = False,
    error_mode: str = "DBH_only",
) -> str:
    """Get one plot's 3PG input file for `literature_source`, building it once.

    Parameters
    ----------
    plot_id : str
        Plot identifier within `forest_source`.
    literature_source : str
        Forwarded to `create_combined_inputs.create_plot_input_file`; also the
        cache's subdirectory.
    forest_source : str
        Dataset of the plot: "ICP", "NFI", "EFM" or "LWF"; the cache is kept in
        `<forest_source>_plots`.
    calibrated : bool
        Whether to return a copy whose `parameters` sheet holds the plot's
        calibrated values from the calibration sweep, see `write_calibrated_input`.
    error_mode : str
        Error mode of the calibration run, used only if `calibrated`.

    Returns
    -------
    str
        Path to the plot's input Excel file.
    """
    plot_dir = os.path.join(
        threepg_data_folder, f"{forest_source.lower()}_plots", literature_source
    )
    os.makedirs(plot_dir, exist_ok=True)
    file_path = os.path.join(plot_dir, f"{plot_id}_data.xlsx")

    if not os.path.exists(file_path):
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".xlsx", prefix=f"{plot_id}_", dir=plot_dir)
        os.close(tmp_fd)
        try:
            create_plot_input_file(
                plot_id,
                forest_source,
                tmp_path,
                *load_plot_tables(plot_id, forest_source),
                literature_source=literature_source,
            )
            os.replace(tmp_path, file_path)
        except Exception:
            os.remove(tmp_path)
            raise

    if calibrated:
        return write_calibrated_input(file_path, plot_id, literature_source, error_mode)
    return file_path


def run_bayesian_for_plot(
    plot_id: str,
    chains: int = 3,
    cores: int | None = None,
    error_mode: str = "DBH_only",
    num_warmup: int = 10000,
    num_samples: int = 10000,
    literature_source: str = "Forrester",
    include_process_error: bool = False,
) -> str:
    """Prepare inputs and run Bayesian calibration for one ICP plot.

    Parameters
    ----------
    plot_id : str
        ICP plot identifier.
    chains, cores, num_warmup, num_samples
        Passed through to `run_pymc_analysis`.
    error_mode : str
        The error mode to use for the calibration.
    literature_source : str
        Forwarded to `prepare_plot_input`.
    include_process_error : bool
        Forwarded to `run_pymc_analysis` — see its docstring.

    Returns
    -------
    str
        The plot_id, on successful completion.
    """
    file_path = prepare_plot_input(plot_id, literature_source=literature_source)
    fit_params = fit_params_for_mode(file_path, error_mode)

    error_names = [
        name for name in load_priors_from_file(file_path) if name in ERROR_MODES[error_mode]
    ]

    output_dir = os.path.join(results_data_folder, f"pymc_inference_results_{plot_id}")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    os.mkdir(output_dir)
    shutil.copy(file_path, output_dir)

    run_pymc_analysis(
        output_dir=output_dir,
        file_path=file_path,
        param_to_optimize=fit_params + error_names,
        include_process_error=include_process_error,
        error_mode=error_mode,
        chains=chains,
        cores=cores,
        num_warmup=num_warmup,
        num_samples=num_samples,
    )
    print(f"Finished Bayesian calibration for plot_id={plot_id}")
    return plot_id


def run_bayesian_calibration(
    plot_ids: list[str],
    error_mode: str = "DBH_only",
    chains: int = 3,
    num_warmup: int = 100,
    num_samples: int = 100,
    literature_source: str = "Forrester",
    include_process_error: bool = False,
) -> None:
    """Run Bayesian calibration for multiple ICP plots in parallel.

    Parameters
    ----------
    plot_ids : list[str]
        List of ICP plot identifiers.
    chains, cores, num_warmup, num_samples
        Passed through to `run_pymc_analysis`.
    literature_source : str
        Forwarded to `prepare_plot_input`.
    include_process_error : bool
        Forwarded to `run_pymc_analysis` — see its docstring.
    """
    available_cpus = get_available_cpus()
    max_workers = max(1, min(len(plot_ids), available_cpus // chains))
    print(
        f"Available CPUs: {available_cpus}, chains per plot: {chains}, max_workers: {max_workers}"
    )

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                run_bayesian_for_plot,
                plot_id,
                chains=chains,
                cores=chains,
                num_warmup=num_warmup,
                num_samples=num_samples,
                literature_source=literature_source,
                error_mode=error_mode,
                include_process_error=include_process_error,
            ): plot_id
            for plot_id in plot_ids
        }
        for future in as_completed(futures):
            plot_id = futures[future]
            try:
                future.result()
            except Exception:
                print(f"Bayesian calibration failed for plot_id={plot_id}")
                raise


if __name__ == "__main__":
    species_plot_ids = {
        "Pinus sylvestris": [
            "01.0082",
            "04.1303",
            "51.0015",
            "53.0109",
            "53.0112",
            "53.0114",
            "53.0302",
            "53.0306",
            "53.0311",
            "53.0312",
            "53.0313",
            "53.0316",
            "53.0407",
            "53.0501",
            "53.0513",
            "53.0603",
            "53.0617",
            "53.0618",
            "53.0623",
            "59.0001",
            "59.0003",
        ],
        "Fagus sylvatica": ["04.0101", "04.0704", "08.0034", "53.0107"],
        "Picea abies": [
            "04.0302",
            "04.1402",
            "04.1403",
            "14.0017",
            "52.0010",
            "53.0701",
            "59.0008",
        ],
    }

    plot_ids = [plot_id for species in species_plot_ids.values() for plot_id in species]

    plot_ids = ["04.0302"]
    # Add argument parser
    parser = argparse.ArgumentParser(description="Run Bayesian calibration for ICP plots")
    parser.add_argument(
        "--plot-ids", nargs="+", required=False, help="Space-separated list of plot IDs"
    )
    parser.add_argument("--chains", type=int, default=3, help="Number of MCMC chains")
    parser.add_argument("--warmup", type=int, default=5000, help="Number of warmup samples")
    parser.add_argument("--samples", type=int, default=5000, help="Number of posterior samples")
    parser.add_argument(
        "--error-mode",
        type=str,
        default="DBH_only",
        choices=list(ERROR_MODES.keys()),
        help="Error mode to use for calibration",
    )
    parser.add_argument(
        "--include-process-error",
        action="store_true",
        help="Additionally treat the initial-state biomass pools WS0/WR0/WF0 as "
        "uncertain, fitted quantities (see bayesian_config.INITIAL_STATE_PARAMS).",
    )

    args = parser.parse_args()

    if args.plot_ids is None:
        args.plot_ids = plot_ids

    print(f"Processing plots: {args.plot_ids}")
    run_bayesian_calibration(
        plot_ids=args.plot_ids,
        chains=args.chains,
        num_warmup=args.warmup,
        num_samples=args.samples,
        error_mode=args.error_mode,
        include_process_error=args.include_process_error,
    )

    # Example plot
    plot_id = args.plot_ids[0]
    file_path = os.path.join(
        results_data_folder, f"pymc_inference_results_{plot_id}", f"{plot_id}_data.xlsx"
    )
    input_data = prepare_data(file_path)
    plot_saved_results(
        output_dir=os.path.join(results_data_folder, f"pymc_inference_results_{plot_id}"),
        observations=load_observations_from_file(file_path, site_data=input_data.site),
        climate=input_data.climate,
    )
