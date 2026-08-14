"""Bayesian calibration of 3PG parameters using PyMC and JAX."""

import os
import shutil
import time
from collections.abc import Callable, Sequence
from typing import Any, NamedTuple, cast

import arviz as az
import jax
import numpy as np
import polars as pl
import pymc as pm
import pytensor.tensor as pt
from jax import jit
from jax import numpy as jnp
from jax.scipy.stats import norm
from jax.scipy.stats import t as jax_student_t
from pytensor.graph.basic import Apply, Variable
from pytensor.graph.op import Op, OutputStorageType

from trunx.config import results_data_folder, threepg_data_folder
from trunx.gp3.bayesiancalibrations.bayesian_config import DIAGNOSTIC_ONLY_ERROR_NAMES, FIT_PARAMS
from trunx.gp3.bayesiancalibrations.calibration_utils import (
    plot_inference_results,
    predict_from_parameter_draws,
)
from trunx.gp3.bayesiancalibrations.load_files import (
    load_observations_from_file,
    load_param_defaults_from_file,
    load_priors_from_file,
    load_top_sensitive_params,
)
from trunx.gp3.bayesiancalibrations.save_load_results import (
    load_checkpoint,
    load_inference_data,
    load_predictions,
    save_checkpoint,
    save_results,
)
from trunx.gp3.model_inputs import ClimateData, Params, SiteData, SpeciesData, State
from trunx.gp3.run_3pg import run_3pg

jax.config.update("jax_enable_x64", True)


class PackedObservation(NamedTuple):
    """Prepacked observation arrays for one variable."""

    var_name: str
    sigma_name: str
    obs_times: jnp.ndarray
    obs_values: jnp.ndarray


class Run3PGLogLikeGrad(Op):
    """PyTensor Op that returns the JAX-computed gradient of the 3PG log-likelihood."""

    itypes = [pt.dvector]
    otypes = [pt.dvector]

    def __init__(self, grad_fn: Any) -> None:
        self.grad_fn = grad_fn

    def perform(
        self, node: Apply, inputs: Sequence[Any], output_storage: OutputStorageType
    ) -> None:
        """Compute the gradient of the log-likelihood w.r.t. the input parameters."""
        param_values = jnp.asarray(inputs[0], dtype=jnp.float64)
        output_storage[0][0] = np.asarray(self.grad_fn(param_values), dtype=np.float64)


def build_loglikelihood_fn(
    params_to_optimize: list[str],
    fixed_params: Params,
    state: State,
    climate: ClimateData,
    site: SiteData,
    species: SpeciesData,
    observations: dict[str, tuple[jnp.ndarray, jnp.ndarray]],
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Build a JAX-differentiable 3PG log-likelihood as a pure function of a parameter vector.

    Standalone so it can be reused outside `Run3PGLogLikeOp`'s PyTensor wrapping, e.g. by a
    plain-JAX optimiser that wants to `jax.vmap`/`jax.jit` it directly (see
    `map_param_est.batched_map_search`).

    Parameters
    ----------
    params_to_optimize : list[str]
        Names of the entries in the parameter vector passed to the returned function, in
        order. Names starting with `err_` are treated as observation-noise sigmas rather
        than 3PG physiology parameters.
    observations : dict[str, tuple[jnp.ndarray, jnp.ndarray]]
        Measured variables to score against, as (obs_times, obs_values).

    Returns
    -------
    Callable[[jnp.ndarray], jnp.ndarray]
        Maps a parameter vector (ordered as `params_to_optimize`) to the scalar
        log-likelihood of `observations` under the 3PG simulation it implies.
    """
    param_names = tuple(params_to_optimize)
    model_param_names = tuple(name for name in param_names if not name.startswith("err_"))
    packed_observations = tuple(
        PackedObservation(
            var_name=var_name,
            sigma_name=f"err_{var_name}",
            obs_times=jnp.asarray(obs_times, dtype=jnp.int32).reshape(-1),
            obs_values=jnp.asarray(obs_values, dtype=jnp.float64).reshape(-1),
        )
        for var_name, (obs_times, obs_values) in observations.items()
    )

    def loglikelihood(param_values: jnp.ndarray) -> jnp.ndarray:
        """Compute the log-likelihood for a parameter vector (JAX-differentiable)."""
        param_dict = dict(zip(param_names, param_values, strict=True))
        # Update the fixed_params with the new parameter values (excluding error/sigma terms)
        model_params = {name: param_dict[name] for name in model_param_names}
        updated_params = fixed_params._replace(**model_params)

        # Run the 3PG model
        _, sim_outputs = run_3pg(state, climate, updated_params, site, species)

        # Compute log-likelihood based on model outputs and observations. A Python loop
        # rather than a vmapped reduction: observations are ragged across variables (NaNs
        # are dropped independently per column, see load_observations_from_file), so they
        # can't generally be stacked into one array.
        log_likelihood = jnp.array(0.0)
        for observation in packed_observations:
            if observation.sigma_name not in param_dict or observation.var_name not in sim_outputs:
                continue
            pred_values = jnp.asarray(
                sim_outputs[observation.var_name][observation.obs_times]
            ).reshape(-1)
            # Predictions and observations must line up element-for-element;
            # a mismatch would broadcast into an (n_obs, n_obs) outer product
            # that silently scores every prediction against every observation.
            assert pred_values.shape == observation.obs_values.shape, (
                f"Likelihood shape mismatch for {observation.var_name}: "
                f"predictions {pred_values.shape} vs observations {observation.obs_values.shape}"
            )
            log_likelihood = log_likelihood + jnp.sum(
                norm.logpdf(
                    pred_values,
                    loc=observation.obs_values,
                    scale=param_dict[observation.sigma_name],
                )
            )
        return log_likelihood

    return loglikelihood


class Run3PGLogLikeOp(Op):
    """PyTensor Op that returns scalar log-likelihood from the 3PG simulator."""

    itypes = [pt.dvector]
    otypes = [pt.dscalar]

    def __init__(
        self,
        params_to_optimize: list[str],
        fixed_params: Params,
        state: State,
        climate: ClimateData,
        site: SiteData,
        species: SpeciesData,
        observations: dict[str, tuple[jnp.ndarray, jnp.ndarray]],
        n_species: int,
    ) -> None:
        self.params_to_optimize = tuple(params_to_optimize)
        self.n_species = n_species
        loglikelihood_fn = build_loglikelihood_fn(
            params_to_optimize=params_to_optimize,
            fixed_params=fixed_params,
            state=state,
            climate=climate,
            site=site,
            species=species,
            observations=observations,
        )

        self._loglikelihood_jax = jax.jit(loglikelihood_fn)
        self._grad_op = Run3PGLogLikeGrad(jax.jit(jax.grad(loglikelihood_fn)))

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


def _configure_gpu_memory_sharing(num_workers: int) -> None:
    """Cap JAX's GPU memory reservation so multiple sampler processes can share one GPU.

    Each `pm.sample` worker initializes its own JAX/XLA runtime; on a GPU backend
    they would otherwise each try to preallocate most of the device memory. Must
    be called before `pm.sample` spawns its worker processes, since it mutates
    `os.environ`, which spawned processes inherit and read when they lazily
    initialize their own JAX backend.
    """
    if num_workers <= 1 or jax.default_backend() != "gpu":
        return
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = f"{0.9 / num_workers:.3f}"


def pymc_model(
    climate: ClimateData,
    site: SiteData,
    species: SpeciesData,
    fixed_params: Params,
    state: State,
    observations: dict[str, tuple[jnp.ndarray, jnp.ndarray]],
    priors: dict[str, tuple[float, float]],
) -> pm.Model:
    """Define a PyMC model for Bayesian calibration of 3PG parameters."""
    param_to_optimize = list(priors.keys())
    loglike_op = Run3PGLogLikeOp(
        fixed_params=fixed_params,
        params_to_optimize=param_to_optimize,
        state=state,
        climate=climate,
        site=site,
        species=species,
        observations=observations,
        n_species=len(species.specie),
    )
    with pm.Model() as model:
        # Define priors for the parameters to be estimated
        param_vars: dict[str, pt.TensorVariable] = {}
        for param_name, (lower, upper) in priors.items():
            param_vars[param_name] = pm.Uniform(param_name, lower=lower, upper=upper)

        # Collect parameter values into a vector
        param_vector = pt.stack([param_vars[param_name] for param_name in priors])
        # Use the custom Op to define the likelihood
        loglike_value = cast(Any, loglike_op(param_vector))
        pm.Potential("likelihood", loglike_value)

    return model


def predict_with_uncertainity(
    trace: az.InferenceData,
    initial_state: State,
    climate: ClimateData,
    site: SiteData,
    species: SpeciesData,
    fixed_params: Params,
    observations: dict[str, tuple[jnp.ndarray, jnp.ndarray]],
    priors: dict[str, tuple[float, float]],
    num_predictions: int = 500,
):
    """Run the 3PG model with parameter samples from the posterior to generate predictions."""
    posterior = cast(Any, trace).posterior
    param_to_optimize = list(priors.keys())

    n_total = int(posterior.sizes["chain"] * posterior.sizes["draw"])
    n_pick = min(num_predictions, n_total)

    chain_indices, draw_indices = (
        np.random.randint(0, len(posterior.chain), size=n_pick),
        np.random.randint(0, len(posterior.draw), size=n_pick),
    )

    param_sets = {}
    for param_name in param_to_optimize:
        if param_name in posterior:
            param_sets[param_name] = posterior[param_name].values[chain_indices, draw_indices]

    return predict_from_parameter_draws(
        parameter_draws=param_sets,
        param_names=param_to_optimize,
        initial_state=initial_state,
        climate=climate,
        site=site,
        species=species,
        fixed_params=fixed_params,
        observations=observations,
        n_species=len(species.specie),
    )


def _extract_last_values(
    idata: az.InferenceData, param_names: Sequence[str]
) -> list[dict[str, float]]:
    """Extract each chain's last posterior draw, for use as the next chunk's initvals."""
    posterior = cast(Any, idata).posterior
    return [
        {name: float(posterior[name].isel(chain=chain, draw=-1).values) for name in param_names}
        for chain in range(posterior.sizes["chain"])
    ]


def run_pymc_inference(
    initial_state: State,
    climate: ClimateData,
    site: SiteData,
    species: SpeciesData,
    fixed_params: Params,
    observations: dict[str, tuple[jnp.ndarray, jnp.ndarray]],
    priors: dict[str, tuple[float, float]],
    num_warmup: int = 1000,
    num_samples: int = 1000,
    chains: int = 4,
    cores: int | None = None,
    param_defaults: dict[str, float] | None = None,
    checkpoint_dir: str | None = None,
    checkpoint_every: int = 500,
    resume_tune: int = 200,
    step_method: str = "demetropolisz",
    target_accept: float = 0.9,
) -> tuple[az.InferenceData, pm.Model]:
    """Run PyMC inference for Bayesian calibration of 3PG parameters.

    Parameters
    ----------
    chains : int
        Number of independent MCMC chains to run.
    cores : int | None
        Number of worker processes to run chains in. Defaults to `chains`
        (one process per chain). On a single GPU, pass a lower value (e.g. 1)
        so worker processes don't compete for device memory.
    param_defaults : dict[str, float] | None
        Starting value for each calibrated parameter, used to seed every
        chain at the same point instead of a random prior draw — matching
        the R reference's `createUniformPrior(min, max, best)`. If None,
        PyMC falls back to its default (random) initialization.
    checkpoint_dir : str | None
        Directory to save sampling checkpoints to and resume from. If a
        checkpoint from a previous (possibly interrupted) run is found there,
        sampling resumes from it instead of starting over. If None, sampling
        runs in a single pass with no checkpointing.
    checkpoint_every : int
        Number of post-tuning draws per chain to sample between checkpoints.
    resume_tune : int
        Number of tuning steps used to re-warm the sampler at the start of
        every chunk after the first (each chunk starts a fresh step object, so
        its proposal scale/step size needs to briefly readapt). For `"nuts"`,
        this re-tunes the step size and mass matrix from scratch each chunk,
        which is more wasteful than for `"demetropolisz"` — a larger
        `resume_tune` is worth considering there if checkpointing resumes often.
    step_method : str
        `"demetropolisz"` (derivative-free differential evolution) or `"nuts"`
        (gradient-based, via the same `Run3PGLogLikeOp.grad` MAP already uses).
        NUTS needs far fewer draws for a comparable effective sample size but
        each draw costs more (multiple gradient evaluations via leapfrog steps),
        and can be more sensitive to a mode sitting on a prior bound.
    target_accept : float
        Target acceptance probability for `pm.NUTS`'s step-size adaptation.
        Ignored for `"demetropolisz"`.
    """
    if step_method not in {"demetropolisz", "nuts"}:
        raise ValueError(f"step_method must be 'demetropolisz' or 'nuts', got {step_method!r}")

    checkpoint_every = max(500, num_samples // 10)
    model = pymc_model(
        climate=climate,
        site=site,
        species=species,
        fixed_params=fixed_params,
        state=initial_state,
        observations=observations,
        priors=priors,
    )

    if cores is None:
        cores = chains
    _configure_gpu_memory_sharing(cores)

    param_names = list(priors.keys())
    idata: az.InferenceData | None = None
    draws_done = 0
    initvals: Any = cast(Any, dict(param_defaults)) if param_defaults is not None else None

    if checkpoint_dir is not None:
        checkpoint = load_checkpoint(checkpoint_dir)
        if checkpoint is not None:
            idata, draws_done, initvals = checkpoint
            assert len(initvals) == chains, (
                f"Checkpoint has {len(initvals)} chains, but {chains} were requested"
            )
            print(f"Resuming from checkpoint: {draws_done}/{num_samples} draws already completed")

    with model:
        while draws_done < num_samples:
            chunk_draws = min(checkpoint_every, num_samples - draws_done)
            chunk_tune = num_warmup if idata is None else resume_tune
            step = (
                pm.NUTS(target_accept=target_accept)
                if step_method == "nuts"
                else pm.DEMetropolisZ()
            )
            chunk_trace = pm.sample(
                draws=chunk_draws,
                tune=chunk_tune,
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
                # Convergence is checked once on the full trace in run_pymc_analysis.
                compute_convergence_checks=False,
            )
            idata = (
                chunk_trace
                if idata is None
                else az.concat(cast(Any, idata), chunk_trace, dim="draw", inplace=False)
            )
            draws_done += chunk_draws
            initvals = _extract_last_values(idata, param_names)

            if checkpoint_dir is not None:
                save_checkpoint(idata, draws_done, initvals, checkpoint_dir)
                print(f"Checkpoint saved: {draws_done}/{num_samples} draws")

    return cast(az.InferenceData, idata), model


def run_pymc_analysis(
    output_dir: str,
    file_path: str = os.path.join(threepg_data_folder, "solling_data.xlsx"),
    param_to_optimize: list[str] | None = None,
    chains: int = 3,
    cores: int | None = None,
    num_warmup: int = 10000,
    num_samples: int = 5000,
    checkpoint_every: int = 500,
    resume_tune: int = 200,
    step_method: str = "demetropolisz",
    target_accept: float = 0.9,
):
    """Run PyMC inference for Bayesian calibration of 3PG parameters.

    Sampling is checkpointed to `output_dir` every `checkpoint_every` draws, so
    calling this again with the same `output_dir` resumes an interrupted run
    instead of starting over.

    Parameters
    ----------
    step_method, target_accept
        Forwarded to `run_pymc_inference`; see its docstring.
    """
    # Imported here so building the model doesn't require the input files that
    # `PG3_model_impl` reads at import time.
    from trunx.gp3.PG3_model_impl import prepare_data

    initial_state, climate, fixed_params, site_data, species_data, n_species, _ = prepare_data(
        file_path
    )

    priors = load_priors_from_file(file_path, param_to_optimize)
    for error_name in DIAGNOSTIC_ONLY_ERROR_NAMES:
        priors.pop(error_name, None)
    param_defaults = load_param_defaults_from_file(file_path, list(priors.keys()))
    observations = load_observations_from_file(file_path, site_data=site_data)

    skipped = [name for name in observations if f"err_{name}" not in priors]
    if skipped:
        print(f"Skipping observations with no matching sigma prior: {skipped}")

    print(f"Loaded priors for {len(priors)} parameters")
    print(f"Loaded observations for variables: {list(observations.keys())}")

    trace, model = run_pymc_inference(
        initial_state=initial_state,
        climate=climate,
        site=site_data,
        species=species_data,
        fixed_params=fixed_params,
        observations=observations,
        priors=priors,
        num_warmup=num_warmup,
        num_samples=num_samples,
        chains=chains,
        cores=cores,
        param_defaults=param_defaults,
        checkpoint_dir=output_dir,
        checkpoint_every=checkpoint_every,
        resume_tune=resume_tune,
        step_method=step_method,
        target_accept=target_accept,
    )

    print("\nConvergence diagnostics:")
    summary = az.summary(trace)
    print(summary)

    predictions = predict_with_uncertainity(
        trace=trace,
        initial_state=initial_state,
        climate=climate,
        site=site_data,
        species=species_data,
        fixed_params=fixed_params,
        observations=observations,
        priors=priors,
    )

    print("Saving results... ")
    save_results(
        mcmc=trace,
        output_dir=output_dir,
        predictions=predictions,
    )


def plot_saved_results(
    output_dir: str,
    params: list[str] | None = None,
    observations: dict[str, tuple[jnp.ndarray, jnp.ndarray]] | None = None,
    climate: ClimateData | None = None,
) -> None:
    """Load inference results saved by `run_pymc_analysis` and plot them.

    Parameters
    ----------
    output_dir : str
        Directory passed to `run_pymc_analysis`, containing `inference_data.nc`
        and, if predictions were computed, `predictions.npz`.
    params : list[str] | None
        Parameter names to plot. If None, plots all posterior variables.
    observations : dict[str, tuple[jnp.ndarray, jnp.ndarray]] | None
        Measured variables, overlaid on prediction plots when given.
    climate : ClimateData | None
        Climate data, needed to determine the prediction time axis when plotting
        predictions.
    """
    inf_data = load_inference_data(os.path.join(output_dir, "inference_data.nc"))

    predictions_path = os.path.join(output_dir, "predictions.npz")
    predictions = load_predictions(predictions_path) if os.path.exists(predictions_path) else None

    plot_inference_results(
        inf_data=inf_data,
        params=params,
        observations=observations,
        predictions=predictions,
        climate=climate,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    start_time = time.perf_counter()

    file_path = os.path.join(threepg_data_folder, "full_solling_data.xlsx")
    # morris_results_path = os.path.join(
    #     results_data_folder,
    #     "morris_analysis_results_jax",
    #     "morris_all_components.csv",
    # )

    error_names = [name for name in load_priors_from_file(file_path) if name.startswith("err_")]

    # top_params = load_top_sensitive_params(morris_results_path, n_top=5)

    # param_names = top_params + error_names
    param_names = FIT_PARAMS + error_names

    output_dir = os.path.join(results_data_folder, "pymc_inference_results")

    shutil.rmtree(output_dir)  # To rerun everthing from scratch uncomment this

    if load_checkpoint(output_dir) is None:
        # No checkpoint to resume from: start clean instead of appending to stale results.
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.mkdir(output_dir)
        shutil.copy(file_path, output_dir)

    run_pymc_analysis(
        output_dir=output_dir,
        file_path=file_path,
        param_to_optimize=param_names,
        chains=3,
        cores=3,
        num_warmup=3000,  # If you need to increase the warmup, rerun from scratch.
        num_samples=3000,  # If you just need to increase the number of samples, adjust here
    )

    elapsed_time = time.perf_counter() - start_time
    print(f"Total runtime: {elapsed_time:.2f} seconds")

    # plot_saved_results(
    #     output_dir=os.path.join(results_data_folder, "results/pymc_inference_results"),
    #     params=r_20_params,
    #     observations=load_observations_from_file(file_path),
    #     climate=prepare_data(file_path)[1],
    # )
