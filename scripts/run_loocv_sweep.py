"""Leave-one-plot-out cross-validation of the nutrition modifier per literature source and species.

The plots of each species are the ICP plots passing `select_icp_plots` and the deposition
coverage rule (`deposition_coverage.selected_plot_ids`), except `EXCLUDED_PLOT_IDS` and,
per literature source, those without DBH observations and, with `--plot-selection
poorly_fitted` (the default), those fitted well without the modifier
(`poorly_fitted_without_modifier`); `all_plots` keeps them. There
is one job per literature source, species of `SPECIES` with parameters in that source and at least
`MIN_PLOTS` plots, and L2 penalty of `L2_VALUES` (or `--l2-values`). `--input-vars` sets the
modifier inputs (N and S by default) and `--modifier` the modifier function (`poly_nm` by
default); all three must be passed to every step. The results go to the folders of
`nn_nutrition_modifier` and `loocv_nutrition_modifier`,
`nutrition_modifier/<literature source>/<inputs>/loocv/<plot selection>/<modifier>/`
(`run_dir`), with one subfolder per species.

1. `--prepare` selects the plots, builds their 3PG input files and writes the jobs to
   `jobs_file`; run it once before submitting, as the jobs only read the input files.
2. `--list-jobs` prints the jobs, to size the SLURM array.
3. `--job-index i` runs job i and saves to `<run_dir>/<species>/l2=<value>/`: `loocv.csv`,
   `loocv_summary.csv`, `job.json` (plots and settings) and `images/`.
4. `--select`, once all jobs are done, picks the L2 penalty of each literature source and
   species with the lowest held-out RMSE (`select_l2`) and saves it to
   `<run_dir>/<species>/l2_selection.csv`.
"""

import argparse
import json
import logging
import os
from pathlib import Path

import polars as pl

from trunx.config import results_data_folder, threepg_data_folder
from trunx.gp3.bayesiancalibrations.pymc_icp_plots import prepare_plot_input
from trunx.gp3.create_combined_inputs import _LITERATURE_SOURCES
from trunx.gp3.deposition_coverage import selected_plot_ids
from trunx.gp3.extended_helper import (
    bivariate_quadratic_nm,
    liebig_nm,
    mlp_nm,
    poly_nm,
    product_nm,
)
from trunx.gp3.loocv_nutrition_modifier import default_config, run_loocv, select_l2
from trunx.gp3.nn_nutrition_modifier import (
    poorly_fitted_without_modifier,
    with_target_observations,
)

logger = logging.getLogger(__name__)

LITERATURE_SOURCES = ["Forrester", "Trotsiuk"]
SPECIES = ["Fagus sylvatica", "Picea abies", "Pinus sylvestris"]
MIN_PLOTS = 3  # Fewer plots leave too few to train on in each fold

L2_VALUES = [
    0.0,
    0.003,
    0.01,
    0.03,
    0.05,
    0.1,
    0.2,
    0.3,
    1.0,
    3.0,
    5.0,
    7.0,
    10.0,
    20.0,
    30.0,
    50.0,
    70.0,
    100.0,
    300.0,
]
DEFAULT_INPUT_VARS = ("N", "S")
MODIFIERS = {
    "poly_nm": poly_nm,
    "mlp_nm": mlp_nm,
    "product_nm": product_nm,
    "liebig_nm": liebig_nm,
    "bivariate_quadratic_nm": bivariate_quadratic_nm,
}
DEFAULT_MODIFIER = "poly_nm"
# "poorly_fitted": only the plots `poorly_fitted_without_modifier` keeps; "all_plots": all
PLOT_SELECTIONS = ("poorly_fitted", "all_plots")
DEFAULT_PLOT_SELECTION = "poorly_fitted"
# Its 0.05 ha growth plot gives an implausibly dense stand (LAI 14-16, ~2000 stems/ha)
EXCLUDED_PLOT_IDS = ["55.0014"]


def run_dir(
    literature_source: str,
    input_vars: tuple[str, ...],
    modifier: str,
    plot_selection: str = DEFAULT_PLOT_SELECTION,
) -> Path:
    """Results folder of a literature source, inputs, plot selection and modifier.

    As in `nn_nutrition_modifier`, with the plot selection (`PLOT_SELECTIONS`) after `loocv`.
    """
    return (
        results_data_folder
        / "nutrition_modifier"
        / literature_source
        / "_".join(sorted(input_vars))  # Same folder for ("N", "S") and ("S", "N")
        / "loocv"
        / plot_selection
        / modifier
    )


def jobs_file(
    input_vars: tuple[str, ...], modifier: str, plot_selection: str = DEFAULT_PLOT_SELECTION
) -> Path:
    """Jobs of the sweep over `input_vars`, `modifier` and `plot_selection`.

    The sweep spans literature sources, so its jobs are outside their folders.
    """
    return (
        results_data_folder
        / "nutrition_modifier"
        / "loocv_sweep"
        / f"jobs_{'_'.join(sorted(input_vars))}_{plot_selection}_{modifier}.json"
    )


def _literature_species(literature_source: str) -> set[str]:
    """Species with parameters in `literature_source`."""
    path = os.path.join(threepg_data_folder, _LITERATURE_SOURCES[literature_source])
    return set(pl.read_parquet(path)["species"].unique())


def prepare_jobs(
    input_vars: tuple[str, ...],
    modifier: str = DEFAULT_MODIFIER,
    l2_values: list[float] = L2_VALUES,
    plot_selection: str = DEFAULT_PLOT_SELECTION,
) -> list[dict]:
    """Select the plots, build their input files and write the jobs to `jobs_file`.

    Plots whose input file can't be built (e.g. a second species without parameters in
    the literature source), without DBH observations or, with `plot_selection`
    "poorly_fitted", already fitted well without the modifier
    (`poorly_fitted_without_modifier`) are left out of the job.

    Parameters
    ----------
    input_vars : tuple[str, ...]
        Modifier inputs, `DEPOSITION_CHANNELS` keys.
    modifier : str
        Modifier function, a `MODIFIERS` key.
    l2_values : list[float]
        L2 penalties, one job each.
    plot_selection : str
        Plots to cross-validate, a `PLOT_SELECTIONS` item.

    Returns
    -------
    list[dict]
        One job per literature source, species and L2 penalty: `literature_source`,
        `species`, `plot_ids`, `modifier_l2`, `input_vars`, `modifier` and
        `plot_selection`.
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
            built, file_paths = [], []
            for plot_id in plot_ids:
                try:
                    file_paths.append(prepare_plot_input(plot_id, literature_source=source))
                    built.append(plot_id)
                except Exception:
                    logger.exception("Skipping plot %s for %s / %s", plot_id, source, species)
            config = default_config(file_paths, input_vars, MODIFIERS[modifier])
            built, file_paths = with_target_observations(config, built, file_paths)
            if plot_selection == "poorly_fitted":
                built, file_paths = poorly_fitted_without_modifier(config, built, file_paths)
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
                    "plot_selection": plot_selection,
                }
                for l2 in l2_values
            ]

    path = jobs_file(input_vars, modifier, plot_selection)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jobs, indent=2))
    return jobs


def load_jobs(
    input_vars: tuple[str, ...],
    modifier: str = DEFAULT_MODIFIER,
    plot_selection: str = DEFAULT_PLOT_SELECTION,
) -> list[dict]:
    """Read the jobs of `prepare_jobs` from `jobs_file`."""
    path = jobs_file(input_vars, modifier, plot_selection)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run with --prepare first")
    return json.loads(path.read_text())


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
    plot_selection = job.get("plot_selection", DEFAULT_PLOT_SELECTION)
    config = default_config(file_paths, input_vars, MODIFIERS[modifier])
    job_dir = (
        run_dir(job["literature_source"], input_vars, modifier, plot_selection)
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


def select_jobs(
    input_vars: tuple[str, ...],
    modifier: str = DEFAULT_MODIFIER,
    plot_selection: str = DEFAULT_PLOT_SELECTION,
) -> None:
    """Select the L2 penalty of each literature source and species from its jobs' results.

    Parameters
    ----------
    input_vars : tuple[str, ...]
        Modifier inputs of the sweep, see `run_dir`.
    modifier : str
        Modifier function of the sweep, see `run_dir`.
    plot_selection : str
        Plot selection of the sweep, see `run_dir`.
    """
    species_dirs = [
        species_dir
        for source in LITERATURE_SOURCES
        for species_dir in sorted(run_dir(source, input_vars, modifier, plot_selection).glob("*"))
    ]
    for species_dir in species_dirs:
        summaries = sorted(species_dir.glob("l2=*/loocv_summary.csv"))
        if not summaries:
            continue
        selection = select_l2(
            pl.concat([pl.read_csv(path) for path in summaries], how="diagonal_relaxed")
        )
        selection.write_csv(species_dir / "l2_selection.csv")
        print(f"\n{species_dir.parents[4].name} / {species_dir.name}")
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
    parser.add_argument(
        "--plot-selection",
        choices=PLOT_SELECTIONS,
        default=DEFAULT_PLOT_SELECTION,
        help="plots to cross-validate",
    )
    parser.add_argument(
        "--l2-values", nargs="+", type=float, default=L2_VALUES, help="L2 penalties (--prepare)"
    )
    args = parser.parse_args()
    input_vars = tuple(args.input_vars)

    if args.prepare:
        prepare_jobs(input_vars, args.modifier, args.l2_values, args.plot_selection)
    if args.select:
        select_jobs(input_vars, args.modifier, args.plot_selection)
    elif args.prepare or args.list_jobs:
        jobs = load_jobs(input_vars, args.modifier, args.plot_selection)
        for index, job in enumerate(jobs):
            n_plots = len(job["plot_ids"])
            print(
                f"{index}: {job['literature_source']} / {job['species']}, {n_plots} plots, "
                f"l2={job['modifier_l2']}"
            )
        print(f"Submit with: sbatch --array=0-{len(jobs) - 1} scripts/slurm/loocv_sweep.sbatch")
    else:
        run_job(load_jobs(input_vars, args.modifier, args.plot_selection)[args.job_index])
