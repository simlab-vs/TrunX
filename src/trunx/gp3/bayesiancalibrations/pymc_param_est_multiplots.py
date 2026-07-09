"""
PyMC-parallel HMC parameter estimation across multiple forest plots.

Mirrors the single-plot PyTensor/JAX gradient bridge in `pymc_param_est.py`, but
evaluates the log-likelihood across a shared-parameter batch of plots packed by
`jax_bayesian_param_est_multiplots.py` (same plot loading/padding/vmap machinery
used by the pure-NumPyro multi-plot pipeline).
"""

import os
import time
from collections.abc import Sequence
from typing import Any, cast

import arviz as az
import jax
import numpy as np
import pymc as pm
import pytensor.tensor as pt
from jax import numpy as jnp
from jax.scipy.stats import t as jax_student_t
from pytensor.graph.basic import Apply, Variable
from pytensor.graph.op import Op, OutputStorageType

from trunx.config import results_data_folder, threepg_data_folder
from trunx.gp3.bayesiancalibrations.jax_bayesian_param_est_multiplots import (
    PackedPlotBatch,
    load_and_pack_plots,
    run_packed_plots_forward,
)
from trunx.gp3.bayesiancalibrations.load_files import (
    load_plot_ids_from_file,
    load_priors_from_file,
)
from trunx.gp3.bayesiancalibrations.pymc_param_est import (
    Run3PGLogLikeGrad,
    _configure_gpu_memory_sharing,
)
from trunx.gp3.bayesiancalibrations.save_load_results import save_results
from trunx.gp3.model_inputs import Params


class MultiPlotLogLikeOp(Op):
    """PyTensor Op that returns scalar log-likelihood across a batch of packed 3PG plots."""

    itypes = [pt.dvector]
    otypes = [pt.dscalar]

    def __init__(
        self,
        params_to_optimize: list[str],
        fixed_params: Params,
        packed_plots: PackedPlotBatch,
    ) -> None:
        self.params_to_optimize = params_to_optimize
        self.fixed_params = fixed_params
        self.packed_plots = packed_plots

        self._loglikelihood_jax = jax.jit(self._loglikelihood)
        self._grad_op = Run3PGLogLikeGrad(jax.jit(jax.grad(self._loglikelihood)))

    def _loglikelihood(self, param_values: jnp.ndarray) -> jnp.ndarray:
        """Compute the shared-parameter log-likelihood across all packed plots.

        JAX-differentiable. Uses one shared `err_{var}` observation-noise scale
        across all plots,
        matching the single-plot convention in `pymc_param_est.py` and
        `load_priors_from_file`, rather than the per-plot `sigma_{var}` sampled
        by the NumPyro multi-plot model in `jax_bayesian_param_est_multiplots.py`.
        """
        param_dict = dict(zip(self.params_to_optimize, param_values, strict=True))
        model_params = {
            name: value for name, value in param_dict.items() if not name.startswith("err_")
        }
        updated_params = self.fixed_params._replace(**model_params)

        outputs = run_packed_plots_forward(self.packed_plots, updated_params)

        log_likelihood = jnp.array(0.0)
        for var_name, obs in self.packed_plots.observations.items():
            sigma_name = f"err_{var_name}"
            if sigma_name not in param_dict or var_name not in outputs:
                continue

            pred_values = jnp.take_along_axis(outputs[var_name], obs.times[..., None], axis=1)
            log_probs = jax_student_t.logpdf(
                pred_values, df=3, loc=obs.values, scale=param_dict[sigma_name]
            )
            log_likelihood = log_likelihood + jnp.sum(
                jnp.where(obs.mask[..., None], log_probs, 0.0)
            )
        return log_likelihood

    def perform(
        self, node: Apply, inputs: Sequence[Any], output_storage: OutputStorageType
    ) -> None:
        """Compute the log-likelihood given the input parameters."""
        param_values = jnp.asarray(inputs[0], dtype=jnp.float64)
        log_likelihood = float(self._loglikelihood_jax(param_values))
        output_storage[0][0] = np.array(log_likelihood, dtype=np.float64)

    def grad(self, inputs: Sequence[Variable], output_grads: Sequence[Variable]) -> list[Variable]:
        """Return the gradient of the log-likelihood w.r.t. the input parameters."""
        (param_vector,) = inputs
        (output_grad,) = output_grads
        grad_value = cast(Any, self._grad_op(param_vector))
        return [cast(Any, output_grad) * grad_value]


def multi_plot_pymc_model(
    packed_plots: PackedPlotBatch,
    fixed_params: Params,
    priors: dict[str, tuple[float, float]],
) -> pm.Model:
    """Define a PyMC model for shared-parameter Bayesian calibration across plots."""
    param_to_optimize = list(priors.keys())
    loglike_op = MultiPlotLogLikeOp(
        params_to_optimize=param_to_optimize,
        fixed_params=fixed_params,
        packed_plots=packed_plots,
    )
    with pm.Model() as model:
        param_vars: dict[str, pt.TensorVariable] = {}
        for param_name, (lower, upper) in priors.items():
            param_vars[param_name] = pm.Uniform(param_name, lower=lower, upper=upper)

        param_vector = pt.stack([param_vars[name] for name in priors])
        loglike_value = cast(Any, loglike_op(param_vector))
        pm.Potential("likelihood", loglike_value)

    return model


def run_pymc_multi_plot_inference(
    packed_plots: PackedPlotBatch,
    fixed_params: Params,
    priors: dict[str, tuple[float, float]],
    num_warmup: int = 1000,
    num_samples: int = 1000,
    chains: int = 4,
    cores: int | None = None,
) -> tuple[az.InferenceData, pm.Model]:
    """Run PyMC inference for shared-parameter calibration across multiple plots.

    Parameters
    ----------
    chains : int
        Number of independent MCMC chains to run.
    cores : int | None
        Number of worker processes to run chains in. Defaults to `chains`
        (one process per chain). On a single GPU, pass a lower value (e.g. 1)
        so worker processes don't compete for device memory.
    """
    model = multi_plot_pymc_model(packed_plots, fixed_params, priors)

    if cores is None:
        cores = chains
    _configure_gpu_memory_sharing(cores)

    with model:
        step = pm.DEMetropolisZ()
        # step = pm.NUTS()
        trace = pm.sample(
            draws=num_samples,
            tune=num_warmup,
            step=step,
            chains=chains,
            cores=cores,
            # JAX's runtime is multithreaded and unsafe to fork; PyMC defaults to
            # fork/forkserver on macOS, so force spawn to run chains in parallel safely.
            mp_ctx="spawn",
            random_seed=42,
            return_inferencedata=True,
            progressbar=True,
            compute_convergence_checks=True,
        )

    return trace, model


def run_pymc_multi_plot_analysis(
    params_file: str,
    plot_files: list[tuple[str, str]],
    output_dir: str,
    param_names: list[str] | None = None,
    num_warmup: int = 500,
    num_samples: int = 500,
    chains: int = 4,
    cores: int | None = None,
) -> None:
    """Run shared-parameter PyMC calibration across many plots and save results."""
    priors = load_priors_from_file(params_file, param_names)
    packed_plots, fixed_params = load_and_pack_plots(params_file, plot_files)

    print(f"Loaded priors for {len(priors)} parameters")

    trace, model = run_pymc_multi_plot_inference(
        packed_plots=packed_plots,
        fixed_params=fixed_params,
        priors=priors,
        num_warmup=num_warmup,
        num_samples=num_samples,
        chains=chains,
        cores=cores,
    )

    print("\nConvergence diagnostics:")
    summary = az.summary(trace)
    print(summary)

    print("Saving results...")
    save_results(mcmc=trace, output_dir=output_dir)


def run_pymc_multi_plot_analysis_for_file(
    plot_file: str,
    params_file: str,
    output_dir: str,
    param_names: list[str] | None = None,
    num_warmup: int = 500,
    num_samples: int = 500,
    chains: int = 4,
    cores: int | None = None,
    max_plots: int | None = None,
) -> None:
    """Run shared-parameter PyMC calibration across all plots in one parquet file."""
    plot_ids = load_plot_ids_from_file(plot_file)
    if max_plots is not None:
        plot_ids = plot_ids[:max_plots]

    plot_files = [(plot_file, plot_id) for plot_id in plot_ids]
    print(f"Running PyMC shared-parameter calibration across {len(plot_ids)} plots")

    run_pymc_multi_plot_analysis(
        params_file=params_file,
        plot_files=plot_files,
        output_dir=output_dir,
        param_names=param_names,
        num_warmup=num_warmup,
        num_samples=num_samples,
        chains=chains,
        cores=cores,
    )


if __name__ == "__main__":
    start_time = time.perf_counter()

    species = "Picea_abies"
    run_pymc_multi_plot_analysis_for_file(
        plot_file=os.path.join(threepg_data_folder, f"icp_plot_data_{species}.parquet"),
        params_file=os.path.join(threepg_data_folder, "params_bounds.parquet"),
        output_dir=os.path.join(results_data_folder, "pymc_multiplot_results"),
        param_names=None,
        max_plots=4,  # Limit to 4 plots for quick testing; remove or increase for full analysis.
        num_warmup=100,
        num_samples=100,
        chains=3,
        cores=1,
    )

    elapsed_time = time.perf_counter() - start_time
    print(f"Total runtime: {elapsed_time:.2f} seconds")

    # plot_file=os.path.join(threepg_data_folder, f"icp_plot_data_{species}.parquet")
    # plot_ids = load_plot_ids_from_file(plot_file)

    # plot_ids = plot_ids[:4]

    # plot_files = [(plot_file, plot_id) for plot_id in plot_ids]
    # params_file=os.path.join(threepg_data_folder, "params_bounds.parquet")
    # packed_plots, fixed_params = load_and_pack_plots(params_file, plot_files)

    # outputs = run_packed_plots_forward(packed_plots, fixed_params)

    # print(outputs.shape)
