"""Run DEMetropolisZ, NUTS, gradient descent, and MAP for every site, twice each.

Once fitting all error terms (DBH/BA/Height included), and once fitting only
biomass (WS/WF/WR) — the project's default, see `bayesian_config`'s
`DIAGNOSTIC_ONLY_ERROR_NAMES`. The two modes are produced by temporarily
overriding that constant rather than by any new calibration logic: DEMetropolisZ,
NUTS, and MAP all read it (via `pymc_param_est`/`map_param_est`) to decide which
error priors to drop before sampling/optimising; gradient descent has no sigma
priors at all, so its equivalent is fitting against all six `PLOT_VARIABLES`
instead of biomass alone.

Writes `<output-dir>/<site_id>/<mode>/{demetropolisz,nuts,map,gradient_descent}/`.

Every (site, mode, method) combination is independent, so each can run as its own
job instead of one long sequential script — see `--list-jobs`/`--job-index`, and
`scripts/slurm/calibration_sweep.sbatch` for submitting the whole sweep as a SLURM
array job, one task per combination.
"""

import argparse
import os
import shutil
import time
from collections.abc import Iterator
from contextlib import contextmanager

import trunx.gp3.bayesiancalibrations.map_param_est as map_param_est
import trunx.gp3.bayesiancalibrations.pymc_param_est as pymc_param_est
from trunx.config import results_data_folder, threepg_data_folder
from trunx.gp3.bayesiancalibrations.bayesian_comparison_plots import PLOT_VARIABLES
from trunx.gp3.bayesiancalibrations.bayesian_config import FIT_PARAMS
from trunx.gp3.bayesiancalibrations.load_files import load_priors_from_file
from trunx.gp3.bayesiancalibrations.save_load_results import save_gradient_descent_result
from trunx.gp3.gradient_descent import GradientDescentConfig, fit_with_gradient_descent

# The project's standard comparison sites (see scripts/run_comparison_site.py and
# scripts/slurm/map_mcmc_comparison.sbatch).
SITES = {
    "solling": "solling_data.xlsx",
}

ERROR_MODES: dict[str, frozenset[str]] = {
    "all_error_terms": frozenset(),
    "biomass_only": frozenset({"err_DBH", "err_BA", "err_Height"}),
}

METHODS = ["demetropolisz", "nuts", "map", "gradient_descent"]

# Modules that did `from bayesian_config import DIAGNOSTIC_ONLY_ERROR_NAMES` and so
# each hold their own binding of it — patched directly by `diagnostic_only_error_names`.
_PATCHED_MODULES = (pymc_param_est, map_param_est)


@contextmanager
def diagnostic_only_error_names(names: frozenset[str]) -> Iterator[None]:
    """Temporarily override `DIAGNOSTIC_ONLY_ERROR_NAMES` for `pymc_param_est`/`map_param_est`.

    Both modules imported the frozenset by name (`from bayesian_config import
    DIAGNOSTIC_ONLY_ERROR_NAMES`), which binds it into their own module namespace —
    reassigning `bayesian_config.DIAGNOSTIC_ONLY_ERROR_NAMES` itself wouldn't reach
    either already-bound name, so this patches each module's own attribute directly
    instead, restoring the original afterward.
    """
    originals = [module.DIAGNOSTIC_ONLY_ERROR_NAMES for module in _PATCHED_MODULES]
    for module in _PATCHED_MODULES:
        module.DIAGNOSTIC_ONLY_ERROR_NAMES = names
    try:
        yield
    finally:
        for module, original in zip(_PATCHED_MODULES, originals, strict=True):
            module.DIAGNOSTIC_ONLY_ERROR_NAMES = original


def list_jobs() -> list[tuple[str, str, str]]:
    """Every (site_id, mode_name, method) combination the sweep covers.

    Index into this list (e.g. via `SLURM_ARRAY_TASK_ID`) to run one combination
    as its own job with `run_job` — see `--job-index`.
    """
    return [
        (site_id, mode_name, method)
        for site_id in SITES
        for mode_name in ERROR_MODES
        for method in METHODS
    ]


def run_job(
    site_id: str,
    mode_name: str,
    method: str,
    output_dir: str,
    chains: int = 3,
    demetropolisz_num_warmup: int = 1_000_000,
    demetropolisz_num_samples: int = 5_000_000,
    nuts_num_warmup: int = 1000,
    nuts_num_samples: int = 1000,
    n_vmap_restarts: int = 2000,
    n_vmap_steps: int = 200,
    laplace_draws: int = 1000,
) -> None:
    """Run one (site, mode, method) combination — the unit a single job covers.

    Parameters
    ----------
    site_id, mode_name : str
        Keys into `SITES`/`ERROR_MODES`.
    method : str
        One of `METHODS`.
    output_dir : str
        Base directory; see the module docstring for the layout written under it.
    chains : int
        Forwarded to the PyMC runs (`run_pymc_analysis`).
    demetropolisz_num_warmup, demetropolisz_num_samples : int
        Forwarded to the DEMetropolisZ run. It needs far more draws than NUTS for a
        comparable effective sample size (see `pymc_param_est.run_pymc_inference`).
    nuts_num_warmup, nuts_num_samples : int
        Forwarded to the NUTS run.
    n_vmap_restarts, n_vmap_steps, laplace_draws : int
        Forwarded to the MAP run (`run_map_analysis`).
    """
    file_path = os.path.join(threepg_data_folder, SITES[site_id])
    error_names = [name for name in load_priors_from_file(file_path) if name.startswith("err_")]
    param_names = FIT_PARAMS + error_names
    diagnostic_only_names = ERROR_MODES[mode_name]
    site_dir = os.path.join(output_dir, site_id, mode_name)

    site_output_dir = os.path.join(output_dir, site_id)
    os.makedirs(site_output_dir, exist_ok=True)
    shutil.copy(file_path, site_output_dir)

    print(f"\n{'=' * 60}\n{site_id} / {mode_name} / {method}\n{'=' * 60}")
    start = time.perf_counter()

    if method == "demetropolisz":
        with diagnostic_only_error_names(diagnostic_only_names):
            pymc_param_est.run_pymc_analysis(
                output_dir=os.path.join(site_dir, "demetropolisz"),
                file_path=file_path,
                param_to_optimize=param_names,
                chains=chains,
                num_warmup=demetropolisz_num_warmup,
                num_samples=demetropolisz_num_samples,
                step_method="demetropolisz",
            )
    elif method == "nuts":
        with diagnostic_only_error_names(diagnostic_only_names):
            pymc_param_est.run_pymc_analysis(
                output_dir=os.path.join(site_dir, "nuts"),
                file_path=file_path,
                param_to_optimize=param_names,
                chains=chains,
                num_warmup=nuts_num_warmup,
                num_samples=nuts_num_samples,
                step_method="nuts",
            )
    elif method == "map":
        with diagnostic_only_error_names(diagnostic_only_names):
            map_param_est.run_map_analysis(
                output_dir=os.path.join(site_dir, "map"),
                file_path=file_path,
                param_to_optimize=param_names,
                n_vmap_restarts=n_vmap_restarts,
                n_vmap_steps=n_vmap_steps,
                laplace_draws=laplace_draws,
            )
    elif method == "gradient_descent":
        target_vars = [var for var in PLOT_VARIABLES if f"err_{var}" not in diagnostic_only_names]
        config = GradientDescentConfig(
            target_vars=target_vars, fit_params=FIT_PARAMS, file_path=file_path
        )
        fitted_params = fit_with_gradient_descent(config).fitted_params
        save_gradient_descent_result(fitted_params, os.path.join(site_dir, "gradient_descent"))
    else:
        raise ValueError(f"Unknown method {method!r}, expected one of {METHODS}")

    print(f"{method} done in {time.perf_counter() - start:.1f}s")


def run_sweep(output_dir: str, **run_job_kwargs) -> None:
    """Run every (site, mode, method) combination sequentially, for local/quick use.

    For a real sweep, prefer submitting each combination as its own job (see the
    module docstring) — `**run_job_kwargs` is forwarded to `run_job` unchanged.
    """
    for site_id, mode_name, method in list_jobs():
        run_job(site_id, mode_name, method, output_dir, **run_job_kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=os.path.join(results_data_folder, "calibration_sweep"),
        help="Base directory to write results into (default: %(default)s)",
    )
    parser.add_argument(
        "--list-jobs",
        action="store_true",
        help="Print every (site, mode, method) combination with its index, then exit "
        "— use this to size a SLURM array (--array=0-N).",
    )
    parser.add_argument(
        "--job-index",
        type=int,
        default=None,
        help="Run only this combination from --list-jobs, instead of the full sweep. "
        "Falls back to $SLURM_ARRAY_TASK_ID if set, so an array job needs no extra flag.",
    )
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--demetropolisz-num-warmup", type=int, default=1_000_000)
    parser.add_argument("--demetropolisz-num-samples", type=int, default=5_000_000)
    parser.add_argument("--nuts-num-warmup", type=int, default=1000)
    parser.add_argument("--nuts-num-samples", type=int, default=1000)
    parser.add_argument("--n-vmap-restarts", type=int, default=2000)
    parser.add_argument("--n-vmap-steps", type=int, default=200)
    parser.add_argument("--laplace-draws", type=int, default=1000)
    args = parser.parse_args()

    jobs = list_jobs()

    if args.list_jobs:
        for index, (site_id, mode_name, method) in enumerate(jobs):
            print(f"{index}\t{site_id}\t{mode_name}\t{method}")
        raise SystemExit(0)

    job_index = args.job_index
    if job_index is None and "SLURM_ARRAY_TASK_ID" in os.environ:
        job_index = int(os.environ["SLURM_ARRAY_TASK_ID"])

    run_job_kwargs = {
        "chains": args.chains,
        "demetropolisz_num_warmup": args.demetropolisz_num_warmup,
        "demetropolisz_num_samples": args.demetropolisz_num_samples,
        "nuts_num_warmup": args.nuts_num_warmup,
        "nuts_num_samples": args.nuts_num_samples,
        "n_vmap_restarts": args.n_vmap_restarts,
        "n_vmap_steps": args.n_vmap_steps,
        "laplace_draws": args.laplace_draws,
    }

    if job_index is None:
        run_sweep(output_dir=args.output_dir, **run_job_kwargs)
    else:
        site_id, mode_name, method = jobs[job_index]
        run_job(site_id, mode_name, method, args.output_dir, **run_job_kwargs)
