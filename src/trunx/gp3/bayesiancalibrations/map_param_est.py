"""Maximum a posteriori (MAP) calibration of 3PG parameters.

A point-estimate alternative to the `DEMetropolisZ` MCMC run in `pymc_param_est`:
it reuses the exact same PyMC model (same uniform priors, same JAX log-likelihood)
but maximises the posterior density with a gradient-based optimiser instead of
sampling from it. Fast, but it yields a single mode with no uncertainty estimate,
and only a local one — see `n_restarts`/`n_vmap_restarts`.
"""

import os
import shutil
import time
from typing import Any, cast

import arviz as az
import jax
import numpy as np
import optax
import pymc as pm
from jax import numpy as jnp
from jax.scipy.stats import norm

from trunx.config import results_data_folder, threepg_data_folder
from trunx.gp3.bayesiancalibrations.bayesian_config import (
    DIAGNOSTIC_ONLY_ERROR_NAMES,
    FIT_PARAMS,
    INITIAL_STATE_PARAMS,
    PROCESS_ERROR_PARAM_NAMES,
)
from trunx.gp3.bayesiancalibrations.load_files import (
    literature_bound_overrides,
    load_observations_from_file,
    load_param_defaults_from_file,
    load_priors_from_file,
)
from trunx.gp3.bayesiancalibrations.map_uncertainty import (
    LaplaceApproximation,
    fit_laplace,
    sample_laplace_posterior,
)
from trunx.gp3.bayesiancalibrations.pymc_param_est import (
    build_loglikelihood_fn,
    clip_defaults_to_priors,
    predict_with_uncertainity,
    pymc_model,
)
from trunx.gp3.bayesiancalibrations.save_load_results import (
    save_laplace_covariance,
    save_map_estimate,
    save_results,
    save_runtime,
)
from trunx.gp3.model_inputs import ClimateData, Params, SiteData, SpeciesData, State


def _prior_draw(
    priors: dict[str, tuple[float, float]], rng: np.random.Generator
) -> dict[str, float]:
    """Draw one starting point uniformly from the priors."""
    return {name: float(rng.uniform(lower, upper)) for name, (lower, upper) in priors.items()}


def _to_unconstrained(
    x: jnp.ndarray, lower: jnp.ndarray, upper: jnp.ndarray, identity_mask: jnp.ndarray
) -> jnp.ndarray:
    """Map bounded parameter values to PyMC's unconstrained Interval-transform space.

    `identity_mask` marks coordinates that pass through unchanged instead —
    `INITIAL_STATE_PARAMS` (`WS0`/`WR0`/`WF0`) are `Normal`-, not `Uniform`-,
    distributed and carry a placeholder `(0.0, 0.0)` "bound" (see
    `run_map_analysis`), so `lower == upper` would make the `Uniform` Interval
    transform below divide by zero.
    """
    bounded = jax.scipy.special.logit((x - lower) / (upper - lower))
    return jnp.where(identity_mask, x, bounded)


def _to_constrained(
    y: jnp.ndarray, lower: jnp.ndarray, upper: jnp.ndarray, identity_mask: jnp.ndarray
) -> jnp.ndarray:
    """Map PyMC's unconstrained Interval-transform space back to bounded parameter values.

    See `_to_unconstrained` for `identity_mask`.
    """
    bounded = lower + (upper - lower) * jax.nn.sigmoid(y)
    return jnp.where(identity_mask, y, bounded)


def _initial_state_prior_terms(
    param_names: list[str], state: State
) -> list[tuple[int, int, float]]:
    """`(value_index, sigma_index, nominal_value)` for each present `INITIAL_STATE_PARAMS` name.

    Indices are positions into `param_names`, for pulling the current draw's
    `WS0`/`WR0`/`WF0` value and its `perr_WS`/`perr_WR`/`perr_WF` sigma (itself
    being optimized in the same restart) out of a parameter vector — see
    `batched_map_search`'s `neg_log_posterior`. `nominal_value` uses the same
    lookup `pymc_model` does, so the two agree on what `WS0` etc. is centered at.
    """
    terms = []
    for name, field in INITIAL_STATE_PARAMS.items():
        if name not in param_names:
            continue
        sigma_name = f"perr_{field}"
        if sigma_name not in param_names:
            raise KeyError(
                f"'{name}' is in param_names but its process-error sigma "
                f"'{sigma_name}' is not — add it to fit its uncertainty."
            )
        nominal_value = float(np.asarray(getattr(state, field)).reshape(-1)[0])
        terms.append((param_names.index(name), param_names.index(sigma_name), nominal_value))
    return terms


def batched_map_search(
    priors: dict[str, tuple[float, float]],
    fixed_params: Params,
    state: State,
    climate: ClimateData,
    site: SiteData,
    species: SpeciesData,
    observations: dict[str, tuple[jnp.ndarray, jnp.ndarray]],
    n_restarts: int = 2000,
    n_steps: int = 200,
    seed: int = 42,
) -> tuple[dict[str, float], float]:
    """Search for the MAP by running many L-BFGS optimisations in parallel on the GPU.

    A `jax.vmap`-batched alternative to `run_map_estimation`'s sequential, CPU-only
    `n_restarts`: every restart's own `optax.lbfgs` trajectory (with a zoom linesearch)
    runs as one lane of a single vmapped/jitted computation, so thousands of restarts
    cost about the same wall-clock time as one. Optimises in the same unconstrained
    (Interval-transformed) space `pm.find_MAP` uses.

    Scored on the log-likelihood alone for `Uniform`-distributed parameters — their
    log-density is a flat constant within bounds, so it doesn't shift the optimum,
    matching `pm.find_MAP`'s `jacobian=False` scoring. `INITIAL_STATE_PARAMS`
    (`WS0`/`WR0`/`WF0`) are the exception: they're `Normal(nominal, perr_*)`-
    distributed (see `pymc_model`), whose log-density is *not* flat, so their
    explicit log-prior is added in — see `_initial_state_prior_terms`. Without it,
    `WS0` etc. would optimize completely unregularized against the data, ignoring
    `perr_*` and converging somewhere `pm.find_MAP`/NUTS/DEMetropolisZ wouldn't.

    This is an exploration step, not a replacement for `pm.find_MAP`: its winner is
    meant to seed one final scipy polish (see `run_map_estimation`'s `n_vmap_restarts`),
    whose exact stationary point is what `fit_laplace` needs.

    Parameters
    ----------
    n_restarts : int
        Number of independent random starting points, optimised in parallel — drawn
        uniformly from the priors for `Uniform`-distributed parameters, and as a
        small Gaussian jitter around the nominal value for `INITIAL_STATE_PARAMS`
        (whose priors's placeholder `(0.0, 0.0)` bound isn't a real range to draw
        from — see `_to_unconstrained`).
    n_steps : int
        Number of L-BFGS iterations per restart.
    seed : int
        Seed for the random starting points.

    Returns
    -------
    tuple[dict[str, float], float]
        The best restart's parameter estimates and its log posterior (equal to the
        log-likelihood when no `INITIAL_STATE_PARAMS` are present, since their
        log-prior term is then zero).
    """
    param_names = list(priors.keys())
    lower_np = np.array([priors[name][0] for name in param_names])
    upper_np = np.array([priors[name][1] for name in param_names])
    lower, upper = jnp.asarray(lower_np), jnp.asarray(upper_np)

    initial_state_terms = _initial_state_prior_terms(param_names, state)
    identity_mask_np = np.zeros(len(param_names), dtype=bool)
    for value_idx, _, _ in initial_state_terms:
        identity_mask_np[value_idx] = True
    identity_mask = jnp.asarray(identity_mask_np)

    loglikelihood_fn = build_loglikelihood_fn(
        params_to_optimize=param_names,
        fixed_params=fixed_params,
        state=state,
        climate=climate,
        site=site,
        species=species,
        observations=observations,
    )

    def neg_log_posterior(y: jnp.ndarray) -> jnp.ndarray:
        x = _to_constrained(y, lower, upper, identity_mask)
        log_prior = jnp.array(0.0)
        for value_idx, sigma_idx, nominal_value in initial_state_terms:
            log_prior = log_prior + norm.logpdf(
                x[value_idx], loc=nominal_value, scale=x[sigma_idx]
            )
        return -(loglikelihood_fn(x) + log_prior)

    solver = optax.lbfgs()
    value_and_grad = optax.value_and_grad_from_state(neg_log_posterior)

    def run_one(y0: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Run one L-BFGS trajectory from `y0`, returning its endpoint and log posterior."""

        def step(carry: tuple[Any, Any], _: None) -> tuple[tuple[Any, Any], None]:
            y, opt_state = carry
            value, grad = value_and_grad(y, state=opt_state)
            updates, opt_state = solver.update(
                grad, opt_state, y, value=value, grad=grad, value_fn=neg_log_posterior
            )
            return (optax.apply_updates(y, updates), opt_state), None

        (y_final, _), _ = jax.lax.scan(step, (y0, solver.init(y0)), xs=None, length=n_steps)
        return y_final, -neg_log_posterior(y_final)

    rng = np.random.default_rng(seed)
    x0_np = rng.uniform(lower_np, upper_np, size=(n_restarts, len(param_names)))
    for value_idx, _, nominal_value in initial_state_terms:
        # The placeholder (0.0, 0.0) "bound" isn't a real range to draw from — jitter
        # around the nominal value instead, at a scale unrelated to that bound.
        jitter_scale = max(abs(nominal_value) * 0.1, 1e-3)
        x0_np[:, value_idx] = nominal_value + rng.normal(scale=jitter_scale, size=n_restarts)
    y0_batch = _to_unconstrained(jnp.asarray(x0_np), lower, upper, identity_mask)

    y_final_batch, logp_batch = jax.jit(jax.vmap(run_one))(y0_batch)
    logp_np = np.asarray(logp_batch)
    finite = np.isfinite(logp_np)
    if not finite.any():
        raise ValueError("No vmapped restart produced a finite log posterior")

    best_index = int(np.argmax(np.where(finite, logp_np, -np.inf)))
    best_params = _to_constrained(y_final_batch[best_index], lower, upper, identity_mask)
    # A restart can converge with a coordinate pinned against a prior bound (e.g. an
    # error sigma driven to its floor by noise-free observations), where the sigmoid
    # saturates to exactly 0 or 1 in float64. Pulled back into the open interval so the
    # winner is a valid `pm.find_MAP` start — PyMC's own Interval transform maps an
    # exact bound to +/-inf and rejects it as a starting point. Skipped for
    # INITIAL_STATE_PARAMS, whose placeholder (0.0, 0.0) bound would otherwise clip
    # any real optimized value straight to 0.
    bound_margin = 1e-9 * (upper - lower)
    clipped = jnp.clip(best_params, lower + bound_margin, upper - bound_margin)
    best_params = jnp.where(identity_mask, best_params, clipped)
    print(
        f"Vmapped MAP search: {n_restarts} restarts x {n_steps} steps, best log "
        f"posterior = {logp_np[best_index]:.4f} (range over restarts: "
        f"[{logp_np[finite].min():.4f}, {logp_np[finite].max():.4f}])"
    )
    return (
        {name: float(v) for name, v in zip(param_names, best_params, strict=True)},
        float(logp_np[best_index]),
    )


def map_to_inference_data(map_estimate: dict[str, float]) -> az.InferenceData:
    """Wrap a MAP point as a single-draw `InferenceData`, for reuse of the MCMC tooling."""
    return az.from_dict(
        {name: np.asarray(value).reshape(1, 1) for name, value in map_estimate.items()}
    )


def run_map_estimation(
    initial_state: State,
    climate: ClimateData,
    site: SiteData,
    species: SpeciesData,
    fixed_params: Params,
    observations: dict[str, tuple[jnp.ndarray, jnp.ndarray]],
    priors: dict[str, tuple[float, float]],
    param_defaults: dict[str, float] | None = None,
    method: str = "L-BFGS-B",
    maxeval: int = 5000,
    n_restarts: int = 0,
    n_vmap_restarts: int = 0,
    n_vmap_steps: int = 200,
    seed: int = 42,
) -> tuple[dict[str, float], float, pm.Model]:
    """Maximise the posterior density of the 3PG calibration model.

    Parameters
    ----------
    param_defaults : dict[str, float] | None
        Starting value for each calibrated parameter. If None, the optimiser starts
        from PyMC's default initial point (the prior midpoint for a `Uniform`).
    method : str
        Any `scipy.optimize.minimize` method. The default `L-BFGS-B` uses the JAX
        gradient exposed by `Run3PGLogLikeOp`.
    maxeval : int
        Maximum number of posterior evaluations per optimisation run.
    n_restarts : int
        Number of extra optimisations started from random prior draws, run
        sequentially on CPU. The run with the highest posterior density wins. The
        likelihood surface of a stand simulator is multimodal, so a single run only
        finds a local mode.
    n_vmap_restarts : int
        Number of extra random-restart starting points explored in parallel on the
        GPU via `batched_map_search`, before the sequential `pm.find_MAP` runs. Cheap
        relative to `n_restarts` — thousands of GPU-parallel restarts cost about the
        same wall-clock time as one — so prefer this over `n_restarts` for broad
        exploration; only the winner is then polished by `pm.find_MAP`. 0 disables it.
    n_vmap_steps : int
        L-BFGS iterations per restart when `n_vmap_restarts > 0`.
    seed : int
        Seed for the restart draws.

    Returns
    -------
    tuple[dict[str, float], float, pm.Model]
        Best parameter estimates, their log posterior density, and the model.
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

    starts: list[dict[str, float] | None] = [
        {name: param_defaults[name] for name in priors if name in param_defaults}
        if param_defaults is not None
        else None
    ]
    if n_vmap_restarts > 0:
        vmap_point, vmap_logp = batched_map_search(
            priors=priors,
            fixed_params=fixed_params,
            state=initial_state,
            climate=climate,
            site=site,
            species=species,
            observations=observations,
            n_restarts=n_vmap_restarts,
            n_steps=n_vmap_steps,
            seed=seed,
        )
        print(f"Vmapped search winner: log posterior = {vmap_logp:.4f}")
        starts.append(vmap_point)
    rng = np.random.default_rng(seed)
    starts += [_prior_draw(priors, rng) for _ in range(n_restarts)]

    best_point: dict[str, Any] | None = None
    best_logp = -np.inf

    with model:
        # `find_MAP` optimises in the unconstrained space but scores with
        # `jacobian=False`, so the optimum is the mode of the constrained posterior.
        logp_fn = model.compile_logp(jacobian=False)
        value_names = [value_var.name for value_var in model.value_vars]

        for run_index, start in enumerate(starts):
            point = cast(
                dict[str, Any],
                pm.find_MAP(start=cast(Any, start), method=method, maxeval=maxeval),
            )
            logp = float(logp_fn({name: point[name] for name in value_names}))
            print(f"MAP run {run_index + 1}/{len(starts)}: log posterior = {logp:.4f}")
            if logp > best_logp:
                best_point, best_logp = point, logp

    assert best_point is not None, "No optimisation run produced a finite log posterior"
    map_estimate = {name: float(best_point[name]) for name in priors}
    return map_estimate, best_logp, model


def run_map_analysis(
    output_dir: str,
    file_path: str = os.path.join(threepg_data_folder, "solling_data.xlsx"),
    param_to_optimize: list[str] | None = None,
    include_process_error: bool = False,
    method: str = "L-BFGS-B",
    maxeval: int = 5000,
    n_restarts: int = 0,
    n_vmap_restarts: int = 0,
    n_vmap_steps: int = 200,
    laplace_draws: int = 0,
) -> dict[str, float]:
    """Calibrate 3PG by MAP estimation and save the estimates and predictions.

    Writes `map_estimate.json`, plus an `inference_data.nc` holding the posterior draws
    and a `predictions.npz` holding the prediction bands, so results load with the same
    helpers as an MCMC run.

    Parameters
    ----------
    include_process_error : bool
        Whether to additionally treat the initial-state biomass pools WS0/WR0/WF0 as
        uncertain, fitted quantities — see `pymc_param_est.run_pymc_analysis`'s
        parameter of the same name for the full explanation; identical behavior here.
    n_vmap_restarts, n_vmap_steps : int
        Forwarded to `run_map_estimation`'s GPU-parallel restart search.
    laplace_draws : int
        Number of draws to take from the Laplace approximation at the MAP, which gives
        the estimates and the predictions a local uncertainty band and writes
        `laplace_covariance.npz`. If 0, the MAP point is written as a single draw and
        the bands collapse onto the MAP trajectory itself.
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

    map_estimate, logp, model = run_map_estimation(
        initial_state=input_data.initial_state,
        climate=input_data.climate,
        site=input_data.site,
        species=input_data.species,
        fixed_params=input_data.params,
        observations=observations,
        priors=priors,
        param_defaults=param_defaults,
        method=method,
        maxeval=maxeval,
        n_restarts=n_restarts,
        n_vmap_restarts=n_vmap_restarts,
        n_vmap_steps=n_vmap_steps,
    )

    print(f"\nMAP estimate (log posterior = {logp:.4f}):")
    for name, (lower, upper) in priors.items():
        value = map_estimate[name]
        at_bound = "  <- at prior bound" if min(value - lower, upper - value) < 1e-6 else ""
        print(f"  {name:<12} {value:>12.6g}   [{lower:g}, {upper:g}]{at_bound}")

    laplace = _fit_laplace_or_warn(model, map_estimate) if laplace_draws > 0 else None
    if laplace is None:
        idata = map_to_inference_data(map_estimate)
        num_predictions = 1
    else:
        idata = sample_laplace_posterior(model, laplace, draws=laplace_draws)
        num_predictions = min(laplace_draws, 500)
        print("\nLaplace approximation at the MAP:")
        if laplace.fixed:
            print(f"  conditional on {sorted(laplace.fixed)} held at their MAP values")
        print(az.summary(idata, kind="stats"))

    predictions = predict_with_uncertainity(
        trace=idata,
        initial_state=input_data.initial_state,
        climate=input_data.climate,
        site=input_data.site,
        species=input_data.species,
        fixed_params=input_data.params,
        observations=observations,
        priors=priors,
        num_predictions=num_predictions,
    )

    print("Saving results... ")
    save_map_estimate(map_estimate, logp, output_dir)
    if laplace is not None:
        save_laplace_covariance(laplace.covariance, laplace.names, output_dir)
    save_results(mcmc=idata, output_dir=output_dir, predictions=predictions)
    elapsed_time = time.perf_counter() - start_time
    save_runtime(elapsed_time, output_dir)
    print(f"Total runtime: {elapsed_time:.2f} seconds")

    return map_estimate


def _fit_laplace_or_warn(
    model: pm.Model, map_estimate: dict[str, float]
) -> LaplaceApproximation | None:
    """Fit the Laplace approximation, downgrading a failure to a warning.

    The optimisation preceding this is expensive and its results are worth saving on
    their own, so a mode that admits no Gaussian approximation (typically one sitting
    on a prior bound) must not sink the whole run.
    """
    try:
        return fit_laplace(model, map_estimate)
    except ValueError as error:
        print(f"\nSkipping the Laplace approximation: {error}")
        return None


if __name__ == "__main__":
    start_time = time.perf_counter()

    file_path = os.path.join(threepg_data_folder, "full_solling_data.xlsx")

    error_names = [name for name in load_priors_from_file(file_path) if name.startswith("err_")]
    param_names = FIT_PARAMS + error_names

    output_dir = os.path.join(results_data_folder, "map_inference_results")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    shutil.copy(file_path, output_dir)

    run_map_analysis(
        output_dir=output_dir,
        file_path=file_path,
        param_to_optimize=param_names,
        include_process_error=True,
        n_restarts=4,
        laplace_draws=1000,
    )

    elapsed_time = time.perf_counter() - start_time
    print(f"Total runtime: {elapsed_time:.2f} seconds")
