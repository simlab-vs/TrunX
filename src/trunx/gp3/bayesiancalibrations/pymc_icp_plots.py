"""Prepare ICP plot input files and run Bayesian calibration for each, in parallel."""

import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

from trunx.config import results_data_folder, threepg_data_folder
from trunx.gp3.bayesiancalibrations.bayesian_config import FIT_PARAMS
from trunx.gp3.bayesiancalibrations.load_files import load_priors_from_file
from trunx.gp3.bayesiancalibrations.pymc_param_est import run_pymc_analysis
from trunx.gp3.create_data_inputs import create_input_data

plot_ids = ["01.0038", "01.0037", "04.0506"]
# plot_ids = [
#     "04.1605",
#     "14.0003",
#     "14.0019",
#     "14.0012",
# ]


def get_available_cpus() -> int:
    """Return the number of CPUs available to this process."""
    sched_getaffinity = getattr(os, "sched_getaffinity", None)
    if sched_getaffinity is not None:
        return len(sched_getaffinity(0))
    return os.cpu_count() or 1


def prepare_plot_input(plot_id: str) -> str:
    """Regenerate one plot's 3PG input file and align its calibration sheets with solling's.

    Parameters
    ----------
    plot_id : str
        ICP plot identifier.

    Returns
    -------
    str
        Path to the plot's generated input Excel file.
    """
    file_path = os.path.join(threepg_data_folder, f"icp_plots/{plot_id}_data.xlsx")
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Deleted: {file_path}")
    create_input_data(file_path, plot_id)

    solling_path = os.path.join(threepg_data_folder, "solling_data.xlsx")
    param_bound = pd.read_excel(solling_path, sheet_name="param_bound")
    error_bound = pd.read_excel(solling_path, sheet_name="error_param")

    observed_data = pd.read_excel(file_path, sheet_name="observed")
    observed_data = observed_data.rename(columns={"Date": "date"})
    observed_data = observed_data.dropna()

    with pd.ExcelWriter(
        file_path, engine="openpyxl", mode="a", if_sheet_exists="replace"
    ) as writer:
        param_bound.to_excel(writer, sheet_name="param_bound", index=False)
        error_bound.to_excel(writer, sheet_name="error_param", index=False)
        observed_data.to_excel(writer, sheet_name="observed", index=False)

    return file_path


def run_bayesian_for_plot(
    plot_id: str,
    chains: int = 3,
    cores: int | None = None,
    num_warmup: int = 10000,
    num_samples: int = 10000,
) -> str:
    """Prepare inputs and run Bayesian calibration for one ICP plot.

    Parameters
    ----------
    plot_id : str
        ICP plot identifier.
    chains, cores, num_warmup, num_samples
        Passed through to `run_pymc_analysis`.

    Returns
    -------
    str
        The plot_id, on successful completion.
    """
    file_path = prepare_plot_input(plot_id)
    error_names = [name for name in load_priors_from_file(file_path) if name.startswith("err_")]

    output_dir = os.path.join(results_data_folder, f"pymc_inference_results_{plot_id}")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    os.mkdir(output_dir)
    shutil.copy(file_path, output_dir)

    run_pymc_analysis(
        output_dir=output_dir,
        file_path=file_path,
        param_to_optimize=FIT_PARAMS + error_names,
        chains=chains,
        cores=cores,
        num_warmup=num_warmup,
        num_samples=num_samples,
    )
    print(f"Finished Bayesian calibration for plot_id={plot_id}")
    return plot_id


if __name__ == "__main__":
    chains = 3

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
                num_warmup=100,
                num_samples=100,
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
