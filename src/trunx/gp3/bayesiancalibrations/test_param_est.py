"""Bayesian calibration of 3PG parameters using PyMC and JAX.

This is temp script file while could be used in TrunX once we verify JAX_DEMetropolitsZ.py
implementation is correct and matches the original PyMC implementation.

"""

import os
import shutil
import time
from collections.abc import Sequence
from typing import Any, NamedTuple, cast

import arviz as az
import jax
import numpy as np
import polars as pl
import pymc as pm
import pytensor
import pytensor.tensor as pt
from jax import jit
from jax import numpy as jnp
from jax.scipy.stats import norm
from jax.scipy.stats import t as jax_student_t
from line_profiler import profile
from pytensor.graph.basic import Apply, Variable
from pytensor.graph.op import Op, OutputStorageType

from trunx.config import results_data_folder, threepg_data_folder
from trunx.gp3.bayesiancalibrations.bayesian_config import FIT_PARAMS
from trunx.gp3.bayesiancalibrations.calibration_utils import (
    plot_inference_results,
    predict_from_parameter_draws,
)
from trunx.gp3.bayesiancalibrations.JAX_DEMetropolitsZ import run_demetropolisz_scan
from trunx.gp3.bayesiancalibrations.load_files import (
    load_observations_from_file,
    load_param_defaults_from_file,
    load_priors_from_file,
    load_top_sensitive_params,
)
from trunx.gp3.bayesiancalibrations.save_load_results import (
    load_inference_data,
    load_predictions,
    save_results,
)
from trunx.gp3.model_inputs import ClimateData, Params, SiteData, SpeciesData, State
from trunx.gp3.PG3_model_impl import prepare_data
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

    def __init__(self, grad_fn: Any) -> None:
        self.grad_fn = grad_fn

    def make_node(self, *inputs: Any) -> Apply:
        """Create the Apply node holding this Op's symbolic input/output.

        Follows PyMC's own recommended pattern for wrapping a JAX function
        in PyTensor (https://www.pymc.io/projects/examples/en/latest/howto/wrapping_jax_function.html):
        explicit `make_node` rather than the `itypes`/`otypes` shorthand.
        """
        param_vector = pt.as_tensor_variable(inputs[0])
        outputs = [param_vector.type()]
        return Apply(self, [param_vector], outputs)

    def perform(
        self, node: Apply, inputs: Sequence[Any], output_storage: OutputStorageType
    ) -> None:
        """Compute the gradient of the log-likelihood w.r.t. the input parameters."""
        param_values = jnp.asarray(inputs[0], dtype=jnp.float64)
        out_dtype = cast(Any, node.outputs[0]).dtype
        output_storage[0][0] = np.asarray(self.grad_fn(param_values), dtype=out_dtype)


def _build_loglikelihood(
    params_to_optimize: list[str],
    fixed_params: Params,
    state: State,
    climate: ClimateData,
    site: SiteData,
    species: SpeciesData,
    observations: dict[str, tuple[jnp.ndarray, jnp.ndarray]],
) -> Any:
    """Build a JAX-differentiable log-likelihood function for the 3PG simulator."""
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

        # Compute log-likelihood based on model outputs and observations, vmapped
        # over observations instead of looped in Python.
        active_observations = [
            observation
            for observation in packed_observations
            if observation.sigma_name in param_dict and observation.var_name in sim_outputs
        ]
        if not active_observations:
            return jnp.array(0.0)

        pred_stack = jnp.stack(
            [
                sim_outputs[observation.var_name][observation.obs_times].reshape(-1)
                for observation in active_observations
            ]
        )
        obs_values_stack = jnp.stack(
            [observation.obs_values.reshape(-1) for observation in active_observations]
        )
        sigma_stack = jnp.stack(
            [param_dict[observation.sigma_name] for observation in active_observations]
        )

        def _observation_loglik(
            pred_row: jnp.ndarray, obs_row: jnp.ndarray, sigma: jnp.ndarray
        ) -> jnp.ndarray:
            """Sum the log-density of one observation's predictions."""
            return jnp.sum(norm.logpdf(pred_row, loc=obs_row, scale=sigma))

        return jnp.sum(jax.vmap(_observation_loglik)(pred_stack, obs_values_stack, sigma_stack))

    return loglikelihood


class Run3PGLogLikeOp(Op):
    """PyTensor Op that returns scalar log-likelihood from the 3PG simulator."""

    def __init__(self, loglike_fn: Any, grad_fn: Any) -> None:
        self.loglike_fn = loglike_fn
        self._grad_op = Run3PGLogLikeGrad(grad_fn)

    def make_node(self, *inputs: Any) -> Apply:
        """Create the Apply node holding this Op's symbolic input/output."""
        # Convert our inputs to symbolic variables
        node_inputs = [pt.as_tensor_variable(inputs[0])]
        # Define the type of the output returned by the wrapped JAX function
        outputs = [pt.dscalar()]
        return Apply(self, node_inputs, outputs)

    def perform(
        self, node: Apply, inputs: Sequence[Any], output_storage: OutputStorageType
    ) -> None:
        """Compute the log-likelihood given the input parameters."""
        param_values = jnp.asarray(inputs[0], dtype=jnp.float64)
        result = self.loglike_fn(param_values)
        out_dtype = cast(Any, node.outputs[0]).dtype
        output_storage[0][0] = np.asarray(result, dtype=out_dtype)

    def grad(self, inputs: Sequence[Variable], output_grads: Sequence[Variable]) -> list[Variable]:
        """Return the gradient of the log-likelihood w.r.t. the input parameters."""
        (param_vector,) = inputs
        grad_wrt_param_vector = cast(Any, self._grad_op(param_vector))
        output_gradient = output_grads[0]
        return [cast(Any, output_gradient) * grad_wrt_param_vector]


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
    loglikelihood_fn = _build_loglikelihood(
        params_to_optimize=param_to_optimize,
        fixed_params=fixed_params,
        state=state,
        climate=climate,
        site=site,
        species=species,
        observations=observations,
    )
    loglike_op = Run3PGLogLikeOp(
        loglike_fn=jax.jit(loglikelihood_fn),
        grad_fn=jax.jit(jax.grad(loglikelihood_fn)),
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
    """
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

    initvals = cast(Any, dict(param_defaults)) if param_defaults is not None else None

    with model:
        step = pm.DEMetropolisZ()
        # step = pm.HamiltonianMC()
        # step = pm.NUTS()
        trace = pm.sample(
            draws=num_samples,
            tune=num_warmup,
            step=step,
            chains=chains,
            cores=cores,
            initvals=initvals,
            # JAX's runtime is multithreaded and unsafe to fork; PyMC defaults to
            # fork/forkserver on macOS, so force spawn to run chains in parallel safely.
            mp_ctx="spawn",
            random_seed=42,
            # discard_tuned_samples=False, # Saves warmup samples
            return_inferencedata=True,
            progressbar=True,
            compute_convergence_checks=True,
            # idata_kwargs={"save_warmup": True, "log_likelihood": True},
        )

    return trace, model


def _to_unconstrained(x: jnp.ndarray, lower: jnp.ndarray, upper: jnp.ndarray) -> jnp.ndarray:
    """Map bounded parameter values to PyMC's unconstrained Interval-transform space."""
    return jax.scipy.special.logit((x - lower) / (upper - lower))


def _to_constrained(y: jnp.ndarray, lower: jnp.ndarray, upper: jnp.ndarray) -> jnp.ndarray:
    """Map PyMC's unconstrained Interval-transform space back to bounded parameter values."""
    return lower + (upper - lower) * jax.nn.sigmoid(y)


def _log_jacobian(y: jnp.ndarray, lower: jnp.ndarray, upper: jnp.ndarray) -> jnp.ndarray:
    """Log-absolute-determinant of the Interval transform's Jacobian, summed over parameters."""
    return jnp.sum(jnp.log(upper - lower) + jax.nn.log_sigmoid(y) + jax.nn.log_sigmoid(-y))


def _build_log_posterior(loglikelihood_fn: Any, lower: jnp.ndarray, upper: jnp.ndarray) -> Any:
    """Combine the log-likelihood with the transform's Jacobian for unconstrained sampling."""

    def log_posterior(y: jnp.ndarray) -> jnp.ndarray:
        return loglikelihood_fn(_to_constrained(y, lower, upper)) + _log_jacobian(y, lower, upper)

    return log_posterior


def run_demetropolisz_jax(
    climate: ClimateData,
    site: SiteData,
    species: SpeciesData,
    fixed_params: Params,
    state: State,
    observations: dict[str, tuple[jnp.ndarray, jnp.ndarray]],
    priors: dict[str, tuple[float, float]],
    num_warmup: int = 1000,
    num_samples: int = 1000,
    chains: int = 4,
    param_defaults: dict[str, float] | None = None,
    seed: int = 42,
) -> az.InferenceData:
    """Run PyMC's DEMetropolisZ sampler via `JAX_DEMetropolitsZ.run_demetropolisz_scan`.

    Parameters
    ----------
    chains : int
        Number of independent chains, run in parallel via `jax.vmap`.
    param_defaults : dict[str, float] | None
        Starting value for each calibrated parameter, seeding every chain at
        the same point (matching `run_pymc_inference`'s own convention). If
        None, the midpoint of each parameter's prior bounds is used.
    seed : int
        PRNG seed; each chain gets an independent key split from it.

    Returns
    -------
    az.InferenceData
        Posterior draws in the same natural (bounded) parameter units as
        `run_pymc_inference`'s output, with one `chain`/`draw` dimension pair
        per parameter.
    """
    param_names = list(priors.keys())
    lower = jnp.array([priors[name][0] for name in param_names])
    upper = jnp.array([priors[name][1] for name in param_names])

    loglikelihood_fn = _build_loglikelihood(
        params_to_optimize=param_names,
        fixed_params=fixed_params,
        state=state,
        climate=climate,
        site=site,
        species=species,
        observations=observations,
    )
    log_posterior_fn = _build_log_posterior(loglikelihood_fn, lower, upper)

    if param_defaults is not None:
        initial_x = jnp.array([param_defaults[name] for name in param_names])
    else:
        initial_x = (lower + upper) / 2.0
    initial_position = _to_unconstrained(initial_x, lower, upper)

    draw_positions, draw_accepted = run_demetropolisz_scan(
        logp_fn=log_posterior_fn,
        initial_values=initial_position,
        num_warmup=num_warmup,
        num_samples=num_samples,
        chains=chains,
        seed=seed,
    )
    draw_values = _to_constrained(draw_positions, lower, upper)

    posterior = {name: np.asarray(draw_values[:, :, i]) for i, name in enumerate(param_names)}
    print(
        "Pure-JAX DEMetropolisZ mean acceptance rate (post-warmup): "
        f"{float(jnp.mean(draw_accepted)):.3f}"
    )

    return az.from_dict(posterior=posterior)


def run_pymc_analysis(
    output_dir: str,
    file_path: str = os.path.join(threepg_data_folder, "solling_data.xlsx"),
    param_to_optimize: list[str] | None = None,
    chains: int = 3,
    cores: int | None = None,
    num_warmup: int = 10000,
    num_samples: int = 5000,
    sampler: str = "pymc",
):
    """Run Bayesian calibration of 3PG parameters.

    Parameters
    ----------
    sampler : str
        Which sampler to run: "pymc" (PyMC's own `DEMetropolisZ`, via
        `run_pymc_inference`) or "jax" (the pure-JAX re-implementation, via
        `run_demetropolisz_jax`).
    """
    if sampler not in {"pymc", "jax"}:
        raise ValueError(f"sampler must be 'pymc' or 'jax', got {sampler!r}")

    input_data = prepare_data(file_path)

    priors = load_priors_from_file(file_path, param_to_optimize)
    param_defaults = load_param_defaults_from_file(file_path, list(priors.keys()))
    observations = load_observations_from_file(file_path, site_data=input_data.site)

    skipped = [name for name in observations if f"err_{name}" not in priors]
    if skipped:
        print(f"Skipping observations with no matching sigma prior: {skipped}")

    print(f"Loaded priors for {len(priors)} parameters")
    print(f"Loaded observations for variables: {list(observations.keys())}")

    if sampler == "pymc":
        trace, _ = run_pymc_inference(
            initial_state=input_data.initial_state,
            climate=input_data.climate,
            site=input_data.site,
            species=input_data.species,
            fixed_params=input_data.params,
            observations=observations,
            priors=priors,
            num_warmup=num_warmup,
            num_samples=num_samples,
            chains=chains,
            cores=cores,
            param_defaults=param_defaults,
        )
    else:
        trace = run_demetropolisz_jax(
            climate=input_data.climate,
            site=input_data.site,
            species=input_data.species,
            fixed_params=input_data.params,
            state=input_data.initial_state,
            observations=observations,
            priors=priors,
            num_warmup=num_warmup,
            num_samples=num_samples,
            chains=chains,
            param_defaults=param_defaults,
        )

    print("\nConvergence diagnostics:")
    summary = az.summary(trace)
    print(summary)

    predictions = predict_with_uncertainity(
        trace=trace,
        initial_state=input_data.initial_state,
        climate=input_data.climate,
        site=input_data.site,
        species=input_data.species,
        fixed_params=input_data.params,
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
        num_warmup=20000,
        num_samples=20000,
        sampler="jax",
    )

    elapsed_time = time.perf_counter() - start_time
    print(f"Total runtime: {elapsed_time:.2f} seconds")

    input_data = prepare_data(file_path)
    plot_saved_results(
        output_dir=os.path.join(results_data_folder, "pymc_inference_results"),
        params=FIT_PARAMS,
        observations=load_observations_from_file(file_path, input_data.site),
        climate=input_data.climate,
    )
