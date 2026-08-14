"""Maximum a posteriori (MAP) calibration of 3PG parameters.

A point-estimate alternative to the `DEMetropolisZ` MCMC run in `pymc_param_est`:
it reuses the exact same PyMC model (same uniform priors, same JAX log-likelihood)
but maximises the posterior density with a gradient-based optimiser instead of
sampling from it. Fast, but it yields a single mode with no uncertainty estimate,
and only a local one — see `n_restarts`.
"""

import os
import shutil
import time
from typing import Any, cast

import arviz as az
import numpy as np
import pymc as pm
from jax import numpy as jnp

from trunx.config import results_data_folder, threepg_data_folder
from trunx.gp3.bayesiancalibrations.bayesian_config import (
    DIAGNOSTIC_ONLY_ERROR_NAMES,
    FIT_PARAMS,
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
    predict_with_uncertainity,
    pymc_model,
)
from trunx.gp3.bayesiancalibrations.save_load_results import (
    save_laplace_covariance,
    save_map_estimate,
    save_results,
)
from trunx.gp3.model_inputs import ClimateData, Params, SiteData, SpeciesData, State


def _prior_draw(
    priors: dict[str, tuple[float, float]], rng: np.random.Generator
) -> dict[str, float]:
    """Draw one starting point uniformly from the priors."""
    return {name: float(rng.uniform(lower, upper)) for name, (lower, upper) in priors.items()}


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
        Number of extra optimisations started from random prior draws. The run with
        the highest posterior density wins. The likelihood surface of a stand
        simulator is multimodal, so a single run only finds a local mode.
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
    method: str = "L-BFGS-B",
    maxeval: int = 5000,
    n_restarts: int = 0,
    laplace_draws: int = 0,
) -> dict[str, float]:
    """Calibrate 3PG by MAP estimation and save the estimates and predictions.

    Writes `map_estimate.json`, plus an `inference_data.nc` holding the posterior draws
    and a `predictions.npz` holding the prediction bands, so results load with the same
    helpers as an MCMC run.

    Parameters
    ----------
    laplace_draws : int
        Number of draws to take from the Laplace approximation at the MAP, which gives
        the estimates and the predictions a local uncertainty band and writes
        `laplace_covariance.npz`. If 0, the MAP point is written as a single draw and
        the bands collapse onto the MAP trajectory itself.
    """
    # Imported here so building the model doesn't require the input files that
    # `PG3_model_impl` reads at import time.
    from trunx.gp3.PG3_model_impl import prepare_data

    initial_state, climate, fixed_params, site_data, species_data, n_species, _ = prepare_data(
        file_path
    )

    priors = load_priors_from_file(
        file_path, param_to_optimize, bound_overrides=literature_bound_overrides(file_path)
    )
    for error_name in DIAGNOSTIC_ONLY_ERROR_NAMES:
        priors.pop(error_name, None)
    param_defaults = load_param_defaults_from_file(file_path, list(priors.keys()))
    observations = load_observations_from_file(file_path, site_data=site_data)

    skipped = [name for name in observations if f"err_{name}" not in priors]
    if skipped:
        print(f"Skipping observations with no matching sigma prior: {skipped}")

    print(f"Loaded priors for {len(priors)} parameters")
    print(f"Loaded observations for variables: {list(observations.keys())}")

    map_estimate, logp, model = run_map_estimation(
        initial_state=initial_state,
        climate=climate,
        site=site_data,
        species=species_data,
        fixed_params=fixed_params,
        observations=observations,
        priors=priors,
        param_defaults=param_defaults,
        method=method,
        maxeval=maxeval,
        n_restarts=n_restarts,
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
        initial_state=initial_state,
        climate=climate,
        site=site_data,
        species=species_data,
        fixed_params=fixed_params,
        observations=observations,
        priors=priors,
        num_predictions=num_predictions,
    )

    print("Saving results... ")
    save_map_estimate(map_estimate, logp, output_dir)
    if laplace is not None:
        save_laplace_covariance(laplace.covariance, laplace.names, output_dir)
    save_results(mcmc=idata, output_dir=output_dir, predictions=predictions)

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
        n_restarts=4,
        laplace_draws=1000,
    )

    elapsed_time = time.perf_counter() - start_time
    print(f"Total runtime: {elapsed_time:.2f} seconds")
