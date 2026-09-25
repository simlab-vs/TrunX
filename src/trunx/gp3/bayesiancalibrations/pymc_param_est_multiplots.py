"""
PyMC-parallel HMC parameter estimation across multiple forest plots.

Mirrors the single-plot PyTensor/JAX gradient bridge in `pymc_param_est.py`, but
evaluates the log-likelihood across a shared-parameter batch of plots packed by
`jax_bayesian_param_est_multiplots.py` (same plot loading/padding/vmap machinery
used by the pure-NumPyro multi-plot pipeline).
"""

import argparse
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

from trunx.config import SPECIES_INDICES, results_data_folder, threepg_data_folder
from trunx.gp3.bayesiancalibrations.bayesian_config import (
    ERROR_MODES,
    INITIAL_STATE_PARAMS,
    PROCESS_ERROR_PARAM_NAMES,
    species_plot_ids,
)
from trunx.gp3.bayesiancalibrations.calibration_utils import clip_defaults_to_priors
from trunx.gp3.bayesiancalibrations.jax_bayesian_param_est_multiplots import (
    PackedPlotBatch,
    load_and_pack_plots,
    run_packed_plots_forward,
)
from trunx.gp3.bayesiancalibrations.load_files import (
    fit_params_for_mode,
    literature_bounds_for_species,
    load_param_defaults_from_file,
    load_plot_ids_from_file,
    load_priors_from_file,
)
from trunx.gp3.bayesiancalibrations.pymc_param_est import (
    Run3PGLogLikeGrad,
    _configure_gpu_memory_sharing,
)
from trunx.gp3.bayesiancalibrations.save_load_results import save_results
from trunx.gp3.model_inputs import Params

# (name, size) for each entry a flat parameter vector is sliced into. `size` is 1 for a
# shared scalar (every ordinary physiology/`err_`/`perr_` parameter) or `n_plots` for an
# `INITIAL_STATE_PARAMS` entry, whose value is one latent draw per plot instead.
ParamBlocks = list[tuple[str, int]]


class MultiPlotLogLikeOp(Op):
    """PyTensor Op that returns scalar log-likelihood across a batch of packed 3PG plots."""

    itypes = [pt.dvector]
    otypes = [pt.dscalar]

    def __init__(
        self,
        param_blocks: ParamBlocks,
        fixed_params: Params,
        packed_plots: PackedPlotBatch,
    ) -> None:
        self.param_blocks = param_blocks
        self.fixed_params = fixed_params
        self.packed_plots = packed_plots

        self._loglikelihood_jax = jax.jit(self._loglikelihood)
        self._grad_op = Run3PGLogLikeGrad(jax.jit(jax.grad(self._loglikelihood)))

    def _unpack(self, param_values: jnp.ndarray) -> dict[str, jnp.ndarray]:
        """Slice the flat parameter vector back into named scalar/per-plot blocks."""
        param_dict = {}
        offset = 0
        for name, size in self.param_blocks:
            param_dict[name] = (
                param_values[offset] if size == 1 else param_values[offset : offset + size]
            )
            offset += size
        return param_dict

    def _loglikelihood(self, param_values: jnp.ndarray) -> jnp.ndarray:
        """Compute the shared-parameter log-likelihood across all packed plots.

        JAX-differentiable. Uses one shared `err_{var}` observation-noise scale
        across all plots, matching the single-plot convention in
        `pymc_param_est.py` and `load_priors_from_file`, rather than the
        per-plot `sigma_{var}` sampled by the NumPyro multi-plot model in
        `jax_bayesian_param_est_multiplots.py`. `INITIAL_STATE_PARAMS` entries
        (`"WS0"`/`"WR0"`/`"WF0"`) are one latent value per plot instead of a
        shared scalar — see `param_blocks`.
        """
        param_dict = self._unpack(param_values)
        model_params = {
            name: value
            for name, value in param_dict.items()
            if not name.startswith("err_")
            and name not in INITIAL_STATE_PARAMS
            and name not in PROCESS_ERROR_PARAM_NAMES
        }
        updated_params = self.fixed_params._replace(**model_params)

        # Uncertain initial-condition biomass pools (see INITIAL_STATE_PARAMS): override
        # the corresponding batched initial_state field with this draw's per-plot latent
        # value for any of WS0/WR0/WF0 present, broadcasting each plot's scalar across its
        # own species dimension (same shape run_packed_plots_forward's vmap expects).
        state_overrides = {
            field: jnp.broadcast_to(
                param_dict[name][:, None],
                getattr(self.packed_plots.initial_state, field).shape,
            )
            for name, field in INITIAL_STATE_PARAMS.items()
            if name in param_dict
        }
        packed_plots = (
            self.packed_plots._replace(
                initial_state=self.packed_plots.initial_state._replace(**state_overrides)
            )
            if state_overrides
            else self.packed_plots
        )

        outputs = run_packed_plots_forward(packed_plots, updated_params)

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
    """Define a PyMC model for shared-parameter Bayesian calibration across plots.

    `priors` may include `"WS0"`, `"WR0"` and/or `"WF0"` to treat the
    corresponding initial-state biomass pool as an uncertain, fitted quantity
    per plot instead of the fixed value already in `packed_plots`. Each gets
    its own `pm.Normal` prior per plot, centered on that plot's own measured
    value, with its spread controlled by a single shared `"perr_WS"`/
    `"perr_WR"`/`"perr_WF"` process-error sigma across every plot — itself a
    `pm.Uniform`-fitted parameter that must also be present in `priors`. See
    `INITIAL_STATE_PARAMS`/`PROCESS_ERROR_PARAM_NAMES`/`MultiPlotLogLikeOp`.
    """
    param_blocks: ParamBlocks = [
        (name, packed_plots.n_plots if name in INITIAL_STATE_PARAMS else 1) for name in priors
    ]
    loglike_op = MultiPlotLogLikeOp(
        param_blocks=param_blocks,
        fixed_params=fixed_params,
        packed_plots=packed_plots,
    )
    with pm.Model() as model:
        # Define priors for the parameters to be estimated. Two passes: WS0/WR0/WF0
        # need their own perr_WS/perr_WR/perr_WF sigma to already exist as a PyTensor
        # variable, and `priors` is a plain dict with no guaranteed ordering.
        param_vars: dict[str, pt.TensorVariable] = {}
        for param_name, (lower, upper) in priors.items():
            if param_name in INITIAL_STATE_PARAMS:
                continue
            param_vars[param_name] = pm.Uniform(param_name, lower=lower, upper=upper)

        for param_name, state_field in INITIAL_STATE_PARAMS.items():
            if param_name not in priors:
                continue
            sigma_name = f"perr_{state_field}"
            if sigma_name not in param_vars:
                raise KeyError(
                    f"'{param_name}' is in priors but its process-error sigma "
                    f"'{sigma_name}' is not — add a '{sigma_name}' entry to priors "
                    "to fit its uncertainty."
                )
            # Single-species plots: index 0 is each plot's only species column.
            nominal_values = np.asarray(getattr(packed_plots.initial_state, state_field))[:, 0]
            param_vars[param_name] = pm.Normal(
                param_name,
                mu=nominal_values,
                sigma=param_vars[sigma_name],
                shape=packed_plots.n_plots,
            )

        param_vector = pt.concatenate(
            [
                param_vars[name] if size > 1 else pt.stack([param_vars[name]])
                for name, size in param_blocks
            ]
        )
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
    step_method: str = "demetropolisz",
    target_accept: float = 0.9,
    param_defaults: dict[str, float] | None = None,
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
    step_method : str
        `"demetropolisz"` (derivative-free differential evolution) or `"nuts"`
        (gradient-based, via the same `MultiPlotLogLikeOp.grad` MAP already
        uses). NUTS needs far fewer draws for a comparable effective sample
        size but each draw costs more (multiple gradient evaluations via
        leapfrog steps), and can be more sensitive to a mode sitting on a
        prior bound.
    target_accept : float
        Target acceptance probability for `pm.NUTS`'s step-size adaptation.
        Ignored for `"demetropolisz"`.
    param_defaults : dict[str, float] | None
        Starting value for each shared scalar parameter (physiology/`err_`/
        `perr_`), used to seed every chain at the same point instead of
        PyMC's own default (the midpoint of each `pm.Uniform` prior, which
        can sit in a numerically unstable region far from any realistic
        parameter combination). Never covers `INITIAL_STATE_PARAMS`
        (`WS0`/`WR0`/`WF0`): those are per-plot `pm.Normal` draws already
        centered on each plot's own measured value, so PyMC's default start
        there is already sensible. If None, PyMC falls back to its own
        default (random/midpoint) initialization for every parameter.
    """
    if step_method not in {"demetropolisz", "nuts"}:
        raise ValueError(f"step_method must be 'demetropolisz' or 'nuts', got {step_method!r}")

    model = multi_plot_pymc_model(packed_plots, fixed_params, priors)

    if cores is None:
        cores = chains
    _configure_gpu_memory_sharing(cores)

    initvals = dict(param_defaults) if param_defaults is not None else None

    with model:
        step = (
            pm.NUTS(target_accept=target_accept) if step_method == "nuts" else pm.DEMetropolisZ()
        )
        trace = pm.sample(
            draws=num_samples,
            tune=num_warmup,
            step=step,
            chains=chains,
            cores=cores,
            initvals=cast(Any, initvals),
            # JAX's runtime is multithreaded and unsafe to fork; PyMC defaults to
            # fork/forkserver on macOS, so force spawn to run chains in parallel safely.
            mp_ctx="spawn",
            random_seed=42,
            return_inferencedata=True,
            progressbar=True,
            compute_convergence_checks=True,
        )

    return trace, model


def _species_names_in_batch(packed_plots: PackedPlotBatch) -> list[str]:
    """Species names in a packed batch (identical across plots, see `load_and_pack_plots`)."""
    index_to_species = {value: key for key, value in SPECIES_INDICES.items()}
    species_indices = np.asarray(packed_plots.species.specie[0]).astype(int).tolist()
    return [index_to_species[idx] for idx in species_indices]


def run_pymc_multi_plot_analysis(
    params_file: str,
    plot_files: list[tuple[str, str]],
    output_dir: str,
    param_names: list[str] | None = None,
    include_process_error: bool = False,
    error_mode: str = "biomass_only",
    num_warmup: int = 500,
    num_samples: int = 500,
    chains: int = 4,
    cores: int | None = None,
    step_method: str = "demetropolisz",
    target_accept: float = 0.9,
) -> None:
    """Run shared-parameter PyMC calibration across many plots and save results.

    Parameters
    ----------
    include_process_error : bool
        Whether to additionally treat each plot's initial-state biomass pools
        WS0/WR0/WF0 as uncertain, fitted quantities (see
        `multi_plot_pymc_model`/`INITIAL_STATE_PARAMS`), each with a per-plot
        `pm.Normal` prior whose spread is a single fitted `perr_WS`/`perr_WR`/
        `perr_WF` process-error sigma shared across every plot. Those three
        sigmas need `(min, max)` bounds in `params_file`; `WS0`/`WR0`/`WF0`
        themselves need no such row — they're added automatically with a
        placeholder bound, since their real prior comes from each plot's own
        nominal value plus the fitted sigma, not a file bound.
    error_mode : str
        Key into `ERROR_MODES` (`"all_error_terms"`, `"biomass_only"`,
        `"biomass_DBH_only"`, or `"DBH_only"`) selecting which `err_*`
        observation-noise terms are excluded from the fit. Also narrows the
        default physiology parameter set (see `param_names`) to those with
        a bearing on `error_mode`'s still-active outputs — matching
        `run_calibration_sweep.py`'s `fit_params_for_mode` usage.
    param_names : list[str] | None
        Parameter names to calibrate. If None, defaults to
        `fit_params_for_mode(params_file, error_mode)` (physiology
        parameters with a real prior and a bearing on an output still
        scored under `error_mode`) plus every `err_*` sigma in
        `params_file` — not every bounded parameter in the file.
    step_method, target_accept
        Forwarded to `run_pymc_multi_plot_inference`; see its docstring.
    """
    if error_mode not in ERROR_MODES:
        raise ValueError(f"error_mode must be one of {sorted(ERROR_MODES)}, got {error_mode!r}")

    packed_plots, fixed_params = load_and_pack_plots(params_file, plot_files)

    if param_names is None:
        fit_params = fit_params_for_mode(params_file, error_mode)
        error_names = [
            name for name in load_priors_from_file(params_file) if name.startswith("err_")
        ]
        param_names = fit_params + error_names

    priors_param_names = list(param_names)
    if include_process_error:
        priors_param_names = priors_param_names + list(PROCESS_ERROR_PARAM_NAMES)

    bound_overrides = literature_bounds_for_species(_species_names_in_batch(packed_plots))
    priors = load_priors_from_file(
        params_file, priors_param_names, bound_overrides=bound_overrides
    )
    for error_name in ERROR_MODES[error_mode]:
        priors.pop(error_name, None)
    if include_process_error:
        # WS0/WR0/WF0 get a per-plot pm.Normal prior in multi_plot_pymc_model, centered on
        # each plot's own nominal value with spread from perr_WS/WR/WF (loaded above) — not
        # a pm.Uniform one, so this bound is a required-but-otherwise-unused priors key.
        for name in INITIAL_STATE_PARAMS:
            priors[name] = (0.0, 0.0)

    param_defaults = load_param_defaults_from_file(
        params_file, [name for name in priors if name not in INITIAL_STATE_PARAMS]
    )
    param_defaults = clip_defaults_to_priors(param_defaults, priors)

    print(f"Loaded priors for {len(priors)} parameters")

    trace, model = run_pymc_multi_plot_inference(
        packed_plots=packed_plots,
        fixed_params=fixed_params,
        priors=priors,
        num_warmup=num_warmup,
        num_samples=num_samples,
        chains=chains,
        cores=cores,
        step_method=step_method,
        target_accept=target_accept,
        param_defaults=param_defaults,
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
    include_process_error: bool = False,
    error_mode: str = "biomass_only",
    num_warmup: int = 500,
    num_samples: int = 500,
    chains: int = 4,
    cores: int | None = None,
    plot_ids: list[str] | None = None,
    step_method: str = "demetropolisz",
    target_accept: float = 0.9,
    max_plots: int | None = None,
) -> None:
    """Run shared-parameter PyMC calibration across all plots in one parquet file."""
    if plot_ids is None:
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
        include_process_error=include_process_error,
        error_mode=error_mode,
        num_warmup=num_warmup,
        num_samples=num_samples,
        chains=chains,
        cores=cores,
        step_method=step_method,
        target_accept=target_accept,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--species",
        default="Picea abies",
        help="Species name (spaces or underscores both fine), matching "
        "icp_plot_data_<species>.parquet (default: %(default)s)",
    )
    parser.add_argument(
        "--plot-file",
        default=None,
        help="Parquet file of packed plot data. Defaults to "
        "<threepg_data_folder>/icp_plot_data_<species>.parquet",
    )
    parser.add_argument(
        "--params-file",
        default=None,
        help="Shared physiology parameter bounds parquet. Defaults to "
        "<threepg_data_folder>/params_bounds_<literature-source>_<species>.parquet "
        "(see prepare_multiplots_data.prepare_multiplot_param_bounds).",
    )
    parser.add_argument(
        "--literature-source",
        default="Trotsiuk",
        choices=["Forrester", "Trotsiuk"],
        help="Which literature-derived params_bounds file to use for --species when "
        "--params-file is omitted. Trotsiuk only covers Picea abies and Fagus "
        "sylvatica (default: %(default)s)",
    )
    parser.add_argument(
        "--plot-ids",
        nargs="+",
        default=None,
        metavar="PLOT_ID",
        help="Specific plot IDs to calibrate on. If omitted, uses the curated "
        "list for --species in bayesian_config.species_plot_ids, or every "
        "plot in --plot-file if --species has no curated list there.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save results to. Defaults to "
        "<results_data_folder>/pymc_multiplot_results",
    )
    parser.add_argument(
        "--param-names",
        nargs="+",
        default=None,
        metavar="NAME",
        help="Parameter names to estimate. If omitted, defaults to the physiology "
        "parameters with a bearing on --error-mode's active outputs (see "
        "fit_params_for_mode) plus every err_* sigma in --params-file.",
    )
    parser.add_argument(
        "--include-process-error",
        action="store_true",
        help="Treat each plot's initial-state biomass pools (WS0/WR0/WF0) as "
        "uncertain, fitted quantities. See run_pymc_multi_plot_analysis's docstring.",
    )
    parser.add_argument(
        "--error-mode",
        default="DBH_only",
        choices=sorted(ERROR_MODES),
        help="Which err_* observation-noise terms to exclude from the fit (default: %(default)s)",
    )
    parser.add_argument("--num-warmup", type=int, default=10000)
    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument(
        "--cores",
        type=int,
        default=4,
        help="Worker processes to run chains in. Single-GPU: keep at 1 so worker "
        "processes don't compete for device memory (default: %(default)s)",
    )
    parser.add_argument(
        "--step-method",
        default="demetropolisz",
        choices=["demetropolisz", "nuts"],
        help="'nuts' is faster but more sensitive to prior bounds (default: %(default)s)",
    )
    parser.add_argument("--target-accept", type=float, default=0.9)
    parser.add_argument(
        "--max-plots",
        type=int,
        default=None,
        help="Limit to this many plots, for a quick smoke test. Omit to use every plot.",
    )
    args = parser.parse_args()

    start_time = time.perf_counter()

    species_slug = args.species.replace(" ", "_")
    plot_file = args.plot_file or os.path.join(
        threepg_data_folder, f"icp_plot_data_{species_slug}.parquet"
    )
    params_file = args.params_file or os.path.join(
        threepg_data_folder, f"params_bounds_{args.literature_source}_{species_slug}.parquet"
    )
    output_dir = args.output_dir or os.path.join(results_data_folder, "pymc_multiplot_results")
    plot_ids = args.plot_ids or species_plot_ids.get(args.species)

    run_pymc_multi_plot_analysis_for_file(
        plot_file=plot_file,
        plot_ids=plot_ids,
        params_file=params_file,
        output_dir=output_dir,
        param_names=args.param_names,
        include_process_error=args.include_process_error,
        error_mode=args.error_mode,
        max_plots=args.max_plots,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        chains=args.chains,
        cores=args.cores,
        step_method=args.step_method,
        target_accept=args.target_accept,
    )

    elapsed_time = time.perf_counter() - start_time
    print(f"Total runtime: {elapsed_time:.2f} seconds")
