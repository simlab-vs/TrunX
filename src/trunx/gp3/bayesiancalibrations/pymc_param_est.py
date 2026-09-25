"""Bayesian calibration of 3PG parameters using PyMC and JAX."""

import itertools
import os
import shutil
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
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
from trunx.gp3.bayesiancalibrations.bayesian_config import (
    DIAGNOSTIC_ONLY_ERROR_NAMES,
    FIT_PARAMS,
    INITIAL_STATE_PARAMS,
    PROCESS_ERROR_PARAM_NAMES,
)
from trunx.gp3.bayesiancalibrations.calibration_utils import (
    clip_defaults_to_priors,
    plot_inference_results,
    predict_from_parameter_draws,
)
from trunx.gp3.bayesiancalibrations.load_files import (
    literature_bound_overrides,
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
    save_runtime,
)
from trunx.gp3.model_inputs import ClimateData, Params, SiteData, SpeciesData, State
from trunx.gp3.prepare_data import prepare_data
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
        order. Names starting with `err_` are treated as observation-noise sigmas, names
        in `INITIAL_STATE_PARAMS` (`"WS0"`, `"WR0"`, `"WF0"`) override the corresponding
        initial `State` field, and names in `PROCESS_ERROR_PARAM_NAMES` (`"perr_WS"`,
        `"perr_WR"`, `"perr_WF"`) are process-error sigmas consumed only by `pymc_model`'s
        prior construction — none of these three groups are a 3PG physiology parameter.
    observations : dict[str, tuple[jnp.ndarray, jnp.ndarray]]
        Measured variables to score against, as (obs_times, obs_values).

    Returns
    -------
    Callable[[jnp.ndarray], jnp.ndarray]
        Maps a parameter vector (ordered as `params_to_optimize`) to the scalar
        log-likelihood of `observations` under the 3PG simulation it implies.
    """
    param_names = tuple(params_to_optimize)
    model_param_names = tuple(
        name
        for name in param_names
        if not name.startswith("err_")
        and name not in INITIAL_STATE_PARAMS
        and name not in PROCESS_ERROR_PARAM_NAMES
    )
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

        # Uncertain initial-condition biomass pools (see INITIAL_STATE_PARAMS): override
        # the corresponding State field for any of WS0/WR0/WF0 present in this draw,
        # leaving the rest of the initial state (age, N, ASW, ...) untouched. Broadcast
        # to the field's own shape/dtype (jnp.full_like) rather than assigning the bare
        # scalar directly — run_3pg threads State through jax.lax.scan's carry, which
        # requires every iteration's shape to exactly match the initial one.
        state_overrides = {
            field: jnp.full_like(getattr(state, field), param_dict[name])
            for name, field in INITIAL_STATE_PARAMS.items()
            if name in param_dict
        }
        updated_state = state._replace(**state_overrides) if state_overrides else state

        # Run the 3PG model
        _, sim_outputs = run_3pg(updated_state, climate, updated_params, site, species)

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


def _child_pids(parent_pid: int) -> list[int]:
    """Return the PIDs of `parent_pid`'s direct child processes, read from /proc.

    Linux-only (this only runs inside the Apptainer container the sbatch job
    launches). Avoids adding psutil as a direct dependency for what
    `/proc/<pid>/stat`'s PPID field already gives us.
    """
    children = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat") as f:
                stat = f.read()
            # `comm` (2nd field) is parenthesized and may itself contain spaces/
            # parens, so split on the *last* ")" before reading the fields after it;
            # PPID is the first of those.
            ppid = int(stat.rsplit(")", 1)[1].split()[1])
        except (OSError, IndexError, ValueError):
            continue
        if ppid == parent_pid:
            children.append(int(entry))
    return children


@contextmanager
def _pin_sample_workers_to_distinct_cores() -> Iterator[None]:
    """Pin each of `pm.sample`'s chain-worker processes to one core apiece.

    `pm.sample(cores=N, mp_ctx="spawn")` spawns one process per chain; every one
    of them otherwise inherits this process's full CPU affinity (the SLURM job's
    whole cpuset), and any per-process thread pool that sizes itself off
    `sched_getaffinity` then creates its own *cpuset-wide* pool instead of a fair
    share of it — `chains`-fold oversubscription of the job's own CPU allocation.
    numpy/BLAS pools are already pinned to 1 thread via `*_NUM_THREADS` env vars
    (see calibration_sweep_test.sbatch), but JAX/XLA's CPU "Eigen" thread pool
    (used on every `run_3pg` log-likelihood evaluation here, regardless of step
    method) has no such override in the jaxlib build this runs on — confirmed
    empirically that neither `XLA_FLAGS=--xla_cpu_multi_thread_eigen=false` nor
    `OMP_NUM_THREADS=1` change its size, while restricting a process's own
    affinity does (it correctly follows `sched_getaffinity`, just not any
    thread-count env var). Restricting each worker's own affinity to a single
    core is therefore the only lever that actually caps it.

    Polls for newly spawned children rather than hooking `pm.sample` directly:
    PyMC's process-pool internals (`pymc.sampling.parallel.ProcessAdapter`)
    aren't a public API to inject a per-worker initializer into.
    """
    # sched_(get|set)affinity are Linux-only; absent on macOS/Windows dev machines
    # (see the same pattern in pymc_icp_plots.py's cpu-count fallback).
    sched_getaffinity = getattr(os, "sched_getaffinity", None)
    sched_setaffinity = getattr(os, "sched_setaffinity", None)
    if sched_getaffinity is None or sched_setaffinity is None:
        yield
        return

    cores = sorted(sched_getaffinity(0))
    if len(cores) <= 1:
        yield
        return

    stop_event = threading.Event()
    pinned: set[int] = set()
    core_cycle = itertools.cycle(cores)
    parent_pid = os.getpid()

    def _watch() -> None:
        # Poll fast: a freshly spawned worker takes at least a few hundred ms to
        # import jax/pymc before it can create any thread pool, but the pool is
        # sized once at creation and won't shrink if we pin affinity too late.
        while not stop_event.wait(0.05):
            for pid in _child_pids(parent_pid):
                if pid in pinned:
                    continue
                try:
                    sched_setaffinity(pid, {next(core_cycle)})
                except OSError:
                    # Process already exited between listing and pinning it.
                    continue
                pinned.add(pid)

    watcher = threading.Thread(target=_watch, daemon=True)
    watcher.start()
    try:
        yield
    finally:
        stop_event.set()
        watcher.join(timeout=1)


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
    """Define a PyMC model for Bayesian calibration of 3PG parameters.

    `priors` may include `"WS0"`, `"WR0"` and/or `"WF0"` to treat the corresponding
    initial-state biomass pool (`state.WS`/`state.WR`/`state.WF`) as an uncertain,
    fitted quantity instead of the fixed value in `state`. Each is given a
    `pm.Normal` prior centered on the fixed value already in `state`, with its
    spread controlled by a matching `"perr_WS"`/`"perr_WR"`/`"perr_WF"` process-error
    sigma — itself a `pm.Uniform`-fitted parameter that must also be present in
    `priors` — kept deliberately distinct from the `"err_WS"`/`"err_WR"`/`"err_WF"`
    observation-noise sigmas, which score the simulated *trajectory* against
    observations rather than the initial condition itself. See
    `INITIAL_STATE_PARAMS`/`PROCESS_ERROR_PARAM_NAMES`/`build_loglikelihood_fn`.
    """
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
            nominal_value = float(np.asarray(getattr(state, state_field)).reshape(-1)[0])
            param_vars[param_name] = pm.Normal(
                param_name, mu=nominal_value, sigma=param_vars[sigma_name]
            )

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
        chain at the same point instead of a random prior draw. If None,
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
            with _pin_sample_workers_to_distinct_cores():
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
    include_process_error: bool = False,
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
    include_process_error : bool
        Whether to additionally treat the initial-state biomass pools WS0/WR0/WF0
        as uncertain, fitted quantities (see `pymc_model`/`INITIAL_STATE_PARAMS`),
        each with a `pm.Normal` prior whose spread is a fitted `perr_WS`/`perr_WR`/
        `perr_WF` process-error sigma. Those three sigmas need `(min, max)` bounds in
        `file_path`'s `error_param` sheet; `WS0`/`WR0`/`WF0` themselves need no such
        row — they're added automatically with a placeholder bound, since their real
        prior comes from `state`'s own nominal value plus the fitted sigma, not a
        file bound. Independent of `param_to_optimize`'s observation-noise terms —
        e.g. combined with a `"biomass_only"`-style `param_to_optimize`, you'd fit 3
        observation-noise sigmas (err_WS/err_WR/err_WF) plus all 3 process-error
        sigmas (perr_WS/perr_WR/perr_WF).
    step_method, target_accept
        Forwarded to `run_pymc_inference`; see its docstring.
    """
    # Imported here so building the model doesn't require the input files that
    # `PG3_model_impl` reads at import time.
    from trunx.gp3.PG3_model_impl import prepare_data

    start_time = time.perf_counter()

    input_data = prepare_data(file_path)

    priors_param_names = param_to_optimize
    if include_process_error and param_to_optimize is not None:
        priors_param_names = list(param_to_optimize) + list(PROCESS_ERROR_PARAM_NAMES)

    priors = load_priors_from_file(
        file_path, priors_param_names, bound_overrides=literature_bound_overrides(file_path)
    )
    for error_name in DIAGNOSTIC_ONLY_ERROR_NAMES:
        priors.pop(error_name, None)
    if include_process_error:
        # WS0/WR0/WF0 get a pm.Normal prior in pymc_model, centered on state's own
        # nominal value with spread from perr_WS/WR/WF (loaded above) — not a
        # pm.Uniform one, so this bound is a required-but-otherwise-unused priors key.
        for name in INITIAL_STATE_PARAMS:
            priors[name] = (0.0, 0.0)

    param_defaults = load_param_defaults_from_file(
        file_path, [name for name in priors if name not in INITIAL_STATE_PARAMS]
    )
    param_defaults = clip_defaults_to_priors(param_defaults, priors)
    observations = load_observations_from_file(file_path, site_data=input_data.site)

    skipped = [name for name in observations if f"err_{name}" not in priors]
    if skipped:
        print(f"Skipping observations with no matching sigma prior: {skipped}")

    print(f"Loaded priors for {len(priors)} parameters")
    print(f"Loaded observations for variables: {list(observations.keys())}")
    print(f"Prior bounds: {priors}")

    trace, model = run_pymc_inference(
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
    elapsed_time = time.perf_counter() - start_time
    save_runtime(elapsed_time, output_dir)
    print(f"Total runtime: {elapsed_time:.2f} seconds")


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

    file_path = os.path.join(threepg_data_folder, "solling_data.xlsx")
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
        include_process_error=False,
        chains=3,
        cores=3,
        checkpoint_every=5000,
        step_method="demetropolisz",  # "nuts" is faster but more sensitive to prior bounds
        num_warmup=10000,  # If you need to increase the warmup, rerun from scratch.
        num_samples=10000,  # If you just need to increase the number of samples, adjust here
    )

    elapsed_time = time.perf_counter() - start_time
    print(f"Total runtime: {elapsed_time:.2f} seconds")

    # Imported here, not at module scope, so importing this module doesn't require the
    # input files that PG3_model_impl reads at import time (same reasoning as
    # run_pymc_analysis's own local import).
    from trunx.gp3.PG3_model_impl import prepare_data

    input_data = prepare_data(file_path)
    # include_process_error=True above also fits WS0/WR0/WF0 (INITIAL_STATE_PARAMS) and
    # their perr_WS/perr_WR/perr_WF sigmas (PROCESS_ERROR_PARAM_NAMES) — param_names
    # alone doesn't cover them (run_pymc_analysis adds them internally regardless of
    # what's passed as param_to_optimize), so they're added here too, or the trace/
    # posterior plots below would silently omit them despite being genuinely fit.
    plot_params = param_names + list(INITIAL_STATE_PARAMS) + list(PROCESS_ERROR_PARAM_NAMES)
    plot_saved_results(
        output_dir=output_dir,
        params=plot_params,
        observations=load_observations_from_file(file_path, site_data=input_data.site),
        climate=input_data.climate,
    )
