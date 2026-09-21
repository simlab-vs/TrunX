"""MAP+Laplace calibration sweep on GPU: every (site, mode) combination, MAP only.

`run_calibration_sweep.py`'s `METHODS` currently excludes `"map"` (see its module
docstring), and jobs already running index into its `list_jobs()` — changing that
list there would silently reindex those in-flight array tasks. This script covers
the exact same `(site_id, mode_name)` grid, via the same `SITES`/`ERROR_MODES` and
`run_job_with_retry`, with the method fixed to `"map"` instead of read from that
sweep's method list — so it can run as its own GPU-backed array without touching
the CPU sweep already in progress.

Size the array the same way as `run_calibration_sweep.py`'s:

    uv run --no-sync python scripts/run_map_sweep.py --list-jobs

Then submit one task per line printed above (N = that count - 1):

    sbatch --array=0-N scripts/slurm/map_sweep_gpu.sbatch

Override the checkout with TRUNX_PROJECT_DIR:

    sbatch --array=0-N --export=ALL,TRUNX_PROJECT_DIR=/path/to/checkout ...
"""

import argparse
import os
import sys

from trunx.config import results_data_folder

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_calibration_sweep import ERROR_MODES, SITES, run_job_with_retry  # noqa: E402


def list_jobs() -> list[tuple[str, str]]:
    """Every (site_id, mode_name) combination this MAP-only sweep covers."""
    return [(site_id, mode_name) for site_id in SITES for mode_name in ERROR_MODES]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=os.path.join(results_data_folder, "calibration_sweep"),
        help="Base directory to write results into (default: %(default)s). Matches "
        "run_calibration_sweep.py's so MAP output lands beside the other methods' "
        "for the same (site, mode).",
    )
    parser.add_argument(
        "--list-jobs",
        action="store_true",
        help="Print every (site, mode) combination with its index, then exit "
        "— use this to size a SLURM array (--array=0-N).",
    )
    parser.add_argument(
        "--job-index",
        type=int,
        default=None,
        help="Run only this combination from --list-jobs, instead of the full sweep. "
        "Falls back to $SLURM_ARRAY_TASK_ID if set, so an array job needs no extra flag.",
    )
    parser.add_argument(
        "--literature-source",
        default="Trotsiuk",
        help="Which literature source to use for parameter bounds (default: %(default)s), \
            options are 'Forrester', 'Forrester_default', 'Trotsiuk'.",
    )
    parser.add_argument("--n-vmap-restarts", type=int, default=2000)
    parser.add_argument("--n-vmap-steps", type=int, default=200)
    parser.add_argument("--laplace-draws", type=int, default=1000)
    parser.add_argument(
        "--include-process-error",
        action="store_true",
        help="Additionally treat the initial-state biomass pools WS0/WR0/WF0 as "
        "uncertain, fitted quantities (see bayesian_config.INITIAL_STATE_PARAMS).",
    )
    args = parser.parse_args()

    jobs = list_jobs()

    if args.list_jobs:
        for index, (site_id, mode_name) in enumerate(jobs):
            print(f"{index}\t{site_id}\t{mode_name}")
        raise SystemExit(0)

    job_index = args.job_index
    if job_index is None and "SLURM_ARRAY_TASK_ID" in os.environ:
        job_index = int(os.environ["SLURM_ARRAY_TASK_ID"])
    if job_index is None:
        raise SystemExit("Pass --job-index (see --list-jobs), or submit as a SLURM array")

    site_id, mode_name = jobs[job_index]
    run_job_with_retry(
        site_id,
        mode_name,
        "map",
        args.output_dir,
        literature_source=args.literature_source,
        include_process_error=args.include_process_error,
        n_vmap_restarts=args.n_vmap_restarts,
        n_vmap_steps=args.n_vmap_steps,
        laplace_draws=args.laplace_draws,
    )
