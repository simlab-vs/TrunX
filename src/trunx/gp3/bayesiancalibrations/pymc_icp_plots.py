"""Prepare ICP plot input files and run Bayesian calibration for each, in parallel."""

import argparse
import os
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

from trunx.config import results_data_folder, threepg_data_folder
from trunx.gp3.bayesiancalibrations.bayesian_config import DIAGNOSTIC_ONLY_ERROR_NAMES
from trunx.gp3.bayesiancalibrations.load_files import (
    load_observations_from_file,
    load_priors_from_file,
)
from trunx.gp3.bayesiancalibrations.pymc_param_est import plot_saved_results, run_pymc_analysis
from trunx.gp3.create_data_inputs import create_input_data
from trunx.gp3.prepare_data import prepare_data


def get_available_cpus() -> int:
    """Return the number of CPUs available to this process."""
    sched_getaffinity = getattr(os, "sched_getaffinity", None)
    if sched_getaffinity is not None:
        return len(sched_getaffinity(0))
    return os.cpu_count() or 1


_LITERATURE_SOURCES = {
    "Forrester": "literature_params_forrester_forrester.parquet",
    "Forrester_default": "literature_params_forrester_default.parquet",
    "Trotsiuk": "literature_params_trotsiuk.parquet",
}


def _load_species_param_bound(
    species_name: str, literature_source: str = "Forrester"
) -> pd.DataFrame:
    """Build a param_bound table for one species from the literature parquet.

    Parameters
    ----------
    species_name : str
        Species name as it appears in the literature parquet (e.g. "Picea abies").
    literature_source : str
        Which literature table to load bounds from — one of `_LITERATURE_SOURCES`
        (`"Forrester"`, `"Forrester_default"`, `"Trotsiuk"`).

    Returns
    -------
    pd.DataFrame
        Columns: param_name, default, min, max.
    """
    if literature_source not in _LITERATURE_SOURCES:
        raise ValueError(f"Unknown literature source: {literature_source}")
    literature_path = os.path.join(threepg_data_folder, _LITERATURE_SOURCES[literature_source])
    literature_bound = pd.read_parquet(literature_path)
    literature_bound = literature_bound[literature_bound["species"] == species_name]
    if literature_bound.empty:
        raise ValueError(f"No literature parameter bounds found for species: {species_name}")
    param_bound = literature_bound.rename(columns={"parameter": "param_name"})

    return param_bound[["param_name", "default", "min", "max"]]


def prepare_plot_input(plot_id: str, literature_source: str) -> str:
    """Get one plot's 3PG input file for `literature_source`, building it once.

    Parameters
    ----------
    plot_id : str
        ICP plot identifier.
    literature_source : str
        Forwarded to `_load_species_param_bound`; also the cache's subdirectory.

    Returns
    -------
    str
        Path to the plot's input Excel file.
    """
    plot_dir = os.path.join(threepg_data_folder, "icp_plots", literature_source)
    os.makedirs(plot_dir, exist_ok=True)
    file_path = os.path.join(plot_dir, f"{plot_id}_data.xlsx")

    if os.path.exists(file_path):
        return file_path

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".xlsx", prefix=f"{plot_id}_", dir=plot_dir)
    os.close(tmp_fd)
    try:
        create_input_data(tmp_path, plot_id)

        species_names = pd.read_excel(tmp_path, sheet_name="species")["species"].unique().tolist()
        if len(species_names) != 1:
            raise ValueError(
                f"Expected a single-species plot for {plot_id}, found: {species_names}"
            )
        species_name = species_names[0]

        solling_path = os.path.join(threepg_data_folder, "solling_data.xlsx")
        error_bound = pd.read_excel(solling_path, sheet_name="error_param")

        param_bound = _load_species_param_bound(species_name, literature_source=literature_source)

        observed_data = pd.read_excel(tmp_path, sheet_name="observed")
        observed_data = observed_data.rename(columns={"Date": "date", "stems_n": "N"})
        required_cols = ["month", "year", "date", "DBH", "WS", "WF", "WR", "BA", "Height"]
        # `N` (stems/ha) isn't required for calibration itself — it's kept only so plots can
        # show the "mean tree" DBH implied by inverting aWS/nWS on the observed WS/N, next to
        # the field-measured quadratic mean diameter, to visualize the Jensen's-gap between
        # the two aggregations. Dropped from the required set so a missing N doesn't discard
        # an otherwise-complete observation row.
        observed_data = observed_data[[*required_cols, "N"]].dropna(subset=required_cols)

        with pd.ExcelWriter(
            tmp_path, engine="openpyxl", mode="a", if_sheet_exists="replace"
        ) as writer:
            param_bound.to_excel(writer, sheet_name="param_bound", index=False)
            error_bound.to_excel(writer, sheet_name="error_param", index=False)
            observed_data.to_excel(writer, sheet_name="observed", index=False)
            pd.read_excel(tmp_path, sheet_name="observed").to_excel(
                writer, sheet_name="all_observed", index=False
            )

        os.replace(tmp_path, file_path)
    except Exception:
        os.remove(tmp_path)
        raise

    return file_path


def run_bayesian_for_plot(
    plot_id: str,
    chains: int = 3,
    cores: int | None = None,
    num_warmup: int = 10000,
    num_samples: int = 10000,
    literature_source: str = "Forrester",
) -> str:
    """Prepare inputs and run Bayesian calibration for one ICP plot.

    Parameters
    ----------
    plot_id : str
        ICP plot identifier.
    chains, cores, num_warmup, num_samples
        Passed through to `run_pymc_analysis`.
    literature_source : str
        Forwarded to `prepare_plot_input`.

    Returns
    -------
    str
        The plot_id, on successful completion.
    """
    file_path = prepare_plot_input(plot_id, literature_source=literature_source)
    param_bound = pd.read_excel(file_path, sheet_name="param_bound")
    fit_params = param_bound.dropna(subset=["min", "max"])["param_name"].tolist()
    error_names = [
        name
        for name in load_priors_from_file(file_path)
        if name.startswith("err_") and name not in DIAGNOSTIC_ONLY_ERROR_NAMES
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
        chains=chains,
        cores=cores,
        num_warmup=num_warmup,
        num_samples=num_samples,
    )
    print(f"Finished Bayesian calibration for plot_id={plot_id}")
    return plot_id


def run_bayesian_calibration(
    plot_ids: list[str],
    chains: int = 3,
    num_warmup: int = 100,
    num_samples: int = 100,
    literature_source: str = "Forrester",
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
    plot_ids = ["04.1402"]
    # Add argument parser
    parser = argparse.ArgumentParser(description="Run Bayesian calibration for ICP plots")
    parser.add_argument(
        "--plot-ids", nargs="+", required=False, help="Space-separated list of plot IDs"
    )
    parser.add_argument("--chains", type=int, default=3, help="Number of MCMC chains")
    parser.add_argument("--warmup", type=int, default=5000, help="Number of warmup samples")
    parser.add_argument("--samples", type=int, default=5000, help="Number of posterior samples")

    args = parser.parse_args()

    if args.plot_ids is None:
        args.plot_ids = plot_ids

    print(f"Processing plots: {args.plot_ids}")
    run_bayesian_calibration(
        plot_ids=args.plot_ids,
        chains=args.chains,
        num_warmup=args.warmup,
        num_samples=args.samples,
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
