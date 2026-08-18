"""Run MAP+Laplace then MCMC calibration for one 3PG site, for a MAP-vs-MCMC comparison.

Both calibrations fit only on biomass observations (WS/WF/WR = "Wx"): DBH/BA/Height
have no sigma prior to score them against (see `bayesian_config.DIAGNOSTIC_ONLY_ERROR_NAMES`).
Writes `<output-dir>/map/` (MAP point + Laplace uncertainty draws) and
`<output-dir>/mcmc/` (DEMetropolisZ posterior draws, checkpointed so a requeued job
resumes instead of restarting).
"""

import argparse
import os
import time

from trunx.gp3.bayesiancalibrations.map_param_est import run_map_analysis
from trunx.gp3.bayesiancalibrations.pymc_param_est import run_pymc_analysis


def main() -> None:
    """Parse CLI args and run the MAP-then-MCMC comparison for one site."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file-path", required=True, help="3PG input Excel file for the site")
    parser.add_argument(
        "--output-dir", required=True, help="Directory to write map/ and mcmc/ into"
    )
    parser.add_argument("--n-vmap-restarts", type=int, default=2000)
    parser.add_argument("--n-vmap-steps", type=int, default=200)
    parser.add_argument("--laplace-draws", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=3)
    parser.add_argument("--num-warmup", type=int, default=10000)
    parser.add_argument("--num-samples", type=int, default=5000)
    args = parser.parse_args()

    map_dir = os.path.join(args.output_dir, "map")
    mcmc_dir = os.path.join(args.output_dir, "mcmc")

    print(f"=== MAP + Laplace: {args.file_path} -> {map_dir} ===")
    start = time.perf_counter()
    run_map_analysis(
        output_dir=map_dir,
        file_path=args.file_path,
        param_to_optimize=None,
        n_vmap_restarts=args.n_vmap_restarts,
        n_vmap_steps=args.n_vmap_steps,
        laplace_draws=args.laplace_draws,
    )
    print(f"MAP + Laplace done in {time.perf_counter() - start:.1f}s")

    print(f"=== MCMC (DEMetropolisZ): {args.file_path} -> {mcmc_dir} ===")
    start = time.perf_counter()
    run_pymc_analysis(
        output_dir=mcmc_dir,
        file_path=args.file_path,
        param_to_optimize=None,
        chains=args.chains,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
    )
    print(f"MCMC done in {time.perf_counter() - start:.1f}s")


if __name__ == "__main__":
    main()
