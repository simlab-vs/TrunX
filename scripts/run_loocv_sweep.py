"""Leave-one-plot-out cross-validation of the nutrition modifier per literature source and species.

The plots of each species are the ICP plots passing `select_icp_plots` and the deposition
coverage rule (`deposition_coverage.selected_plot_ids`), except `EXCLUDED_PLOT_IDS`. There
is one job per literature source, species of `SPECIES` with parameters in that source and at least
`MIN_PLOTS` plots, and L2 penalty of `L2_VALUES`. `--input-vars` sets the modifier inputs
(N and S by default); other inputs get their own output folder, e.g.
`loocv_nutrition_modifier_N` for N only, and must be passed to every step. Likewise
`--modifier` sets the modifier function (`poly_nm` by default), e.g.
`loocv_nutrition_modifier_mlp_nm` for `mlp_nm`.

1. `--prepare` selects the plots, builds their 3PG input files and writes the jobs to
   `jobs.json` of the output folder; run it once before submitting, as the jobs only
   read the input files.
2. `--list-jobs` prints the jobs, to size the SLURM array.
3. `--job-index i` runs job i and saves to
   `<output folder>/<literature source>/<species>/l2=<value>/`: `loocv.csv`,
   `loocv_summary.csv`, `job.json` (plots and settings) and `images/`.
4. `--select`, once all jobs are done, picks the L2 penalty of each literature source and
   species with the lowest held-out RMSE (`select_l2`) and saves it to
   `<output folder>/<literature source>/<species>/l2_selection.csv`.
"""

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import polars as pl

from trunx.config import results_data_folder, threepg_data_folder
from trunx.gp3.bayesiancalibrations.pymc_icp_plots import prepare_plot_input
from trunx.gp3.create_combined_inputs import _LITERATURE_SOURCES
from trunx.gp3.deposition_coverage import selected_plot_ids
from trunx.gp3.extended_helper import mlp_nm, poly_nm
from trunx.gp3.loocv_nutrition_modifier import default_config, run_loocv, select_l2

logger = logging.getLogger(__name__)

LITERATURE_SOURCES = ["Forrester", "Trotsiuk"]
SPECIES = ["Fagus sylvatica", "Picea abies", "Pinus sylvestris"]
MIN_PLOTS = 3  # Fewer plots leave too few to train on in each fold
# Rounded, as linspace gives e.g. 6.8500000000000005, which ends up in the folder names
L2_VALUES = [round(float(l2), 2) for l2 in np.linspace(1, 10, 21)]
DEFAULT_INPUT_VARS = ("N", "S")
MODIFIERS = {"poly_nm": poly_nm, "mlp_nm": mlp_nm}
DEFAULT_MODIFIER = "poly_nm"
# Its 0.05 ha growth plot gives an implausibly dense stand (LAI 14-16, ~2000 stems/ha)
EXCLUDED_PLOT_IDS = ["55.0014"]


def output_dir(input_vars: tuple[str, ...], modifier: str = DEFAULT_MODIFIER) -> Path:
    """Results folder of the sweep, suffixed with the inputs and modifier unless the defaults."""
    parts = [] if input_vars == DEFAULT_INPUT_VARS else list(input_vars)
    if modifier != DEFAULT_MODIFIER:
        parts.append(modifier)
    suffix = "".join(f"_{part}" for part in parts)
    return results_data_folder / f"loocv_nutrition_modifier{suffix}"


def _literature_species(literature_source: str) -> set[str]:
    """Species with parameters in `literature_source`."""
    path = os.path.join(threepg_data_folder, _LITERATURE_SOURCES[literature_source])
    return set(pl.read_parquet(path)["species"].unique())


def prepare_jobs(input_vars: tuple[str, ...], modifier: str = DEFAULT_MODIFIER) -> list[dict]:
    """Select the plots, build their input files and write the jobs to `jobs.json`.

    Plots whose input file can't be built (e.g. a second species without parameters in
    the literature source) are left out of the job.

    Parameters
    ----------
    input_vars : tuple[str, ...]
        Modifier inputs, `DEPOSITION_CHANNELS` keys.
    modifier : str
        Modifier function, a `MODIFIERS` key.

    Returns
    -------
    list[dict]
        One job per literature source, species and L2 penalty: `literature_source`,
        `species`, `plot_ids`, `modifier_l2`, `input_vars` and `modifier`.
    """
    plot_ids_by_species = selected_plot_ids()
    jobs = []
    for source in LITERATURE_SOURCES:
        species_with_params = _literature_species(source)
        for species, plot_ids in plot_ids_by_species.items():
            if species not in SPECIES:
                continue
            plot_ids = [plot_id for plot_id in plot_ids if plot_id not in EXCLUDED_PLOT_IDS]
            if species not in species_with_params:
                logger.info("Skipping %s / %s: no parameters", source, species)
                continue
            built = []
            for plot_id in plot_ids:
                try:
                    prepare_plot_input(plot_id, literature_source=source)
                    built.append(plot_id)
                except Exception:
                    logger.exception("Skipping plot %s for %s / %s", plot_id, source, species)
            if len(built) < MIN_PLOTS:
                logger.info("Skipping %s / %s: %d plot(s)", source, species, len(built))
                continue
            jobs += [
                {
                    "literature_source": source,
                    "species": species,
                    "plot_ids": built,
                    "modifier_l2": l2,
                    "input_vars": list(input_vars),
                    "modifier": modifier,
                }
                for l2 in L2_VALUES
            ]

    output_dir(input_vars, modifier).mkdir(parents=True, exist_ok=True)
    (output_dir(input_vars, modifier) / "jobs.json").write_text(json.dumps(jobs, indent=2))
    return jobs


def load_jobs(input_vars: tuple[str, ...], modifier: str = DEFAULT_MODIFIER) -> list[dict]:
    """Read the jobs of `prepare_jobs` over `input_vars` and `modifier` from its `jobs.json`."""
    jobs_file = output_dir(input_vars, modifier) / "jobs.json"
    if not jobs_file.exists():
        raise FileNotFoundError(f"{jobs_file} not found; run with --prepare first")
    return json.loads(jobs_file.read_text())


def run_job(job: dict) -> None:
    """Cross-validate one job's plots and save the results in its folder.

    Parameters
    ----------
    job : dict
        Item of `prepare_jobs`.
    """
    file_paths = [
        prepare_plot_input(plot_id, literature_source=job["literature_source"])
        for plot_id in job["plot_ids"]
    ]
    input_vars = tuple(job.get("input_vars", DEFAULT_INPUT_VARS))
    modifier = job.get("modifier", DEFAULT_MODIFIER)
    config = default_config(file_paths, input_vars, MODIFIERS[modifier])
    job_dir = (
        output_dir(input_vars, modifier)
        / job["literature_source"]
        / job["species"].replace(" ", "_")
        / f"l2={job['modifier_l2']}"
    )
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "job.json").write_text(
        json.dumps(
            {
                **job,
                "input_vars": list(config.input_vars),
                "target_vars": config.target_vars,
            },
            indent=2,
        )
    )
    run_loocv(
        config,
        job["plot_ids"],
        [job["modifier_l2"]],
        job_dir,
        job_dir / "images",
        species=job["species"],
        literature_source=job["literature_source"],
    )


def select_jobs(input_vars: tuple[str, ...], modifier: str = DEFAULT_MODIFIER) -> None:
    """Select the L2 penalty of each literature source and species from its jobs' results.

    Parameters
    ----------
    input_vars : tuple[str, ...]
        Modifier inputs of the sweep, see `output_dir`.
    modifier : str
        Modifier function of the sweep, see `output_dir`.
    """
    for species_dir in sorted(output_dir(input_vars, modifier).glob("*/*")):
        summaries = sorted(species_dir.glob("l2=*/loocv_summary.csv"))
        if not summaries:
            continue
        selection = select_l2(
            pl.concat([pl.read_csv(path) for path in summaries], how="diagonal_relaxed")
        )
        selection.write_csv(species_dir / "l2_selection.csv")
        print(f"\n{species_dir.parent.name} / {species_dir.name}")
        print(selection)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true", help="select plots and write jobs")
    action.add_argument("--list-jobs", action="store_true", help="print the jobs")
    action.add_argument("--job-index", type=int, help="run this job")
    action.add_argument("--select", action="store_true", help="select the L2 penalties")
    parser.add_argument(
        "--input-vars", nargs="+", default=list(DEFAULT_INPUT_VARS), help="modifier inputs"
    )
    parser.add_argument(
        "--modifier", choices=list(MODIFIERS), default=DEFAULT_MODIFIER, help="modifier function"
    )
    args = parser.parse_args()
    input_vars = tuple(args.input_vars)

    if args.prepare:
        prepare_jobs(input_vars, args.modifier)
    if args.select:
        select_jobs(input_vars, args.modifier)
    elif args.prepare or args.list_jobs:
        jobs = load_jobs(input_vars, args.modifier)
        for index, job in enumerate(jobs):
            n_plots = len(job["plot_ids"])
            print(
                f"{index}: {job['literature_source']} / {job['species']}, {n_plots} plots, "
                f"l2={job['modifier_l2']}"
            )
        print(f"Submit with: sbatch --array=0-{len(jobs) - 1} scripts/slurm/loocv_sweep.sbatch")
    else:
        run_job(load_jobs(input_vars, args.modifier)[args.job_index])
