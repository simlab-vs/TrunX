"""Laplace uncertainty estimates around a MAP calibration of 3PG parameters.

`map_param_est` returns a single mode with no uncertainty. This approximates the
posterior near that mode by a Gaussian whose covariance is the inverse of the local
Hessian of the log posterior, turning the point estimate into draws that the MCMC
tooling (`predict_with_uncertainity`, `az.summary`, the saving helpers) consumes
unchanged.

The approximation is built in PyMC's unconstrained space, where the `Uniform` priors
are interval-transformed onto the whole real line, so the Gaussian fits far better
near a bound and every back-transformed draw lands inside the prior support. It
expands the same objective `pm.find_MAP` optimises (the log posterior scored with
`jacobian=False`), so the expansion is taken at a genuine stationary point.

Being local and Gaussian, it captures neither the multimodality of a stand simulator's
likelihood surface nor any skew in the posterior; it is a cheap alternative to MCMC,
not a replacement for it.
"""

import warnings
from typing import Any, NamedTuple, cast

import arviz as az
import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
from scipy.linalg import cho_solve

# Unconstrained coordinates are O(1) under the interval transform, so one absolute
# step suits every parameter; central differences make the truncation error O(step^2).
DEFAULT_STEP = 1e-4

# An interval-transformed coordinate this large puts the mode within 3e-7 of the prior
# range's edge, where the log posterior is flat and the Gaussian says nothing useful.
PINNED_COORDINATE = 15.0

# Smallest eigenvalue, in correlation units, that still counts as curvature rather than
# a flat direction. Inverting anything below this returns mostly round-off.
MIN_RELATIVE_CURVATURE = 1e-8


class LaplaceApproximation(NamedTuple):
    """Gaussian approximation to the posterior, in the unconstrained space.

    `mean`, `covariance` and `names` cover only the parameters the posterior actually
    curves around; `fixed` holds the rest at their MAP values, so the approximation is
    conditional on those.
    """

    mean: np.ndarray
    covariance: np.ndarray
    names: list[str]
    fixed: dict[str, float]


def _backward_function(model: pm.Model) -> Any:
    """Compile the map from a full unconstrained vector to constrained values."""
    backwards = [
        model.rvs_to_values[rv]
        if model.rvs_to_transforms[rv] is None
        else model.rvs_to_transforms[rv].backward(model.rvs_to_values[rv], *rv.owner.inputs)
        for rv in model.free_RVs
    ]
    return pytensor.function(
        cast(Any, model.value_vars), cast(Any, backwards), on_unused_input="ignore"
    )


def _forward_value(model: pm.Model, rv: Any, value: float) -> float:
    """Map one constrained value to its unconstrained coordinate."""
    transform = model.rvs_to_transforms[rv]
    if transform is None:
        return float(value)
    scalar = pt.dscalar(f"{rv.name}_constrained")
    forward = pytensor.function(
        [scalar], cast(Any, transform.forward(scalar, *rv.owner.inputs)), on_unused_input="ignore"
    )
    return float(forward(value))


def _unconstrained_vector(model: pm.Model, map_estimate: dict[str, float]) -> np.ndarray:
    """Map a MAP point to its unconstrained coordinates, in `model.free_RVs` order."""
    values = []
    for rv in model.free_RVs:
        if rv.name not in map_estimate:
            raise KeyError(f"MAP estimate is missing a value for model variable {rv.name!r}")
        values.append(_forward_value(model, rv, map_estimate[rv.name]))

    vector = np.asarray(values, dtype=np.float64)
    names = [rv.name for rv in model.free_RVs]

    at_bound = [name for name, value in zip(names, vector, strict=True) if not np.isfinite(value)]
    if at_bound:
        raise ValueError(
            "Cannot build a Laplace approximation: the MAP sits exactly on a prior bound "
            f"for {at_bound}, where the unconstrained coordinate is infinite. Widen those "
            "priors, or leave those parameters out of the approximation."
        )

    pinned = [
        name for name, value in zip(names, vector, strict=True) if abs(value) > PINNED_COORDINATE
    ]
    if pinned:
        warnings.warn(
            f"The MAP is pinned against a prior bound for {pinned}. The log posterior is "
            "flat there, so the approximation carries no information about these "
            "parameters and their draws will span the whole prior range.",
            stacklevel=3,
        )
    return vector


def log_posterior_hessian(
    model: pm.Model, map_estimate: dict[str, float], step: float = DEFAULT_STEP
) -> tuple[np.ndarray, list[str]]:
    """Hessian of the log posterior at the MAP, in the unconstrained space.

    Central-differences the exact gradient PyTensor already exposes (which reaches the
    JAX gradient of the 3PG likelihood through `Run3PGLogLikeOp.grad`), so only the
    outer difference is approximate. Costs two gradient evaluations per parameter.

    Parameters
    ----------
    model : pm.Model
        The calibration model the MAP was found in.
    map_estimate : dict[str, float]
        MAP values on the parameters' own scale, as returned by `run_map_estimation`.
    step : float
        Finite-difference step in the unconstrained space.

    Returns
    -------
    tuple[np.ndarray, list[str]]
        The symmetrised Hessian and the parameter names indexing its axes.
    """
    center = _unconstrained_vector(model, map_estimate)
    value_names = [value_var.name for value_var in model.value_vars]
    dlogp = model.compile_dlogp(jacobian=False)

    def gradient_at(vector: np.ndarray) -> np.ndarray:
        """Gradient of the log posterior at an unconstrained point."""
        point = dict(zip(value_names, [np.asarray(value) for value in vector], strict=True))
        return np.asarray(dlogp(point), dtype=np.float64).reshape(-1)

    hessian = np.empty((center.size, center.size), dtype=np.float64)
    for index in range(center.size):
        ahead, behind = center.copy(), center.copy()
        ahead[index] += step
        behind[index] -= step
        hessian[:, index] = (gradient_at(ahead) - gradient_at(behind)) / (2 * step)

    # Each mixed second derivative is estimated twice, once per column; averaging both
    # enforces symmetry and cancels part of the truncation error.
    return (hessian + hessian.T) / 2, [rv.name for rv in model.free_RVs]


def fit_laplace(
    model: pm.Model, map_estimate: dict[str, float], step: float = DEFAULT_STEP
) -> LaplaceApproximation:
    """Build the Gaussian approximation to the posterior at the MAP.

    Parameters
    ----------
    model : pm.Model
        The calibration model the MAP was found in.
    map_estimate : dict[str, float]
        MAP values on the parameters' own scale.
    step : float
        Finite-difference step used for the Hessian.

    Any parameter the posterior does not curve downward along — an unidentified one, or
    one whose mode sits on a prior bound, where the MAP is a boundary optimum rather
    than a stationary point — is held at its MAP value and reported in `fixed`, leaving
    the others with an uncertainty conditional on it.

    Returns
    -------
    LaplaceApproximation
        Mean and covariance in the unconstrained space, with their parameter names.

    Raises
    ------
    ValueError
        If no parameter has downward curvature, so there is no mode to expand around.
    """
    hessian, names = log_posterior_hessian(model, map_estimate, step=step)
    center = _unconstrained_vector(model, map_estimate)
    precision = -hessian

    kept = _identified_subset(precision)
    dropped = [name for index, name in enumerate(names) if index not in set(kept)]
    if not kept:
        raise ValueError(
            "The log posterior is not locally concave in any direction, so it has no "
            "Laplace approximation. The optimiser most likely stopped short of a mode."
        )
    if dropped:
        warnings.warn(
            f"Holding {dropped} fixed at their MAP values: the log posterior does not "
            "curve downward along them, which happens when a parameter is unidentified "
            "or its mode sits on a prior bound. The remaining parameters get a "
            "conditional uncertainty, which is narrower than a joint one would be.",
            stacklevel=2,
        )

    cholesky = np.linalg.cholesky(precision[np.ix_(kept, kept)])
    covariance = cho_solve((cholesky, True), np.eye(len(kept)))
    return LaplaceApproximation(
        mean=center[kept],
        covariance=(covariance + covariance.T) / 2,
        names=[names[index] for index in kept],
        fixed={name: map_estimate[name] for name in dropped},
    )


def _identified_subset(
    precision: np.ndarray, tolerance: float = MIN_RELATIVE_CURVATURE
) -> list[int]:
    """Find the parameters the posterior is genuinely peaked in, by index.

    Drops parameters the log posterior curves upward along, then repeatedly drops the
    one dominating the flattest remaining direction until the block is comfortably
    invertible. Conditioning on the dropped parameters — holding them at their MAP — is
    exactly what inverting the surviving block computes.

    The test runs on the correlation-scaled precision rather than the raw one. Curvatures
    here span many orders of magnitude simply because the parameters do, so a raw
    comparison would discard well-identified parameters that merely have small units.
    """
    kept = [index for index in range(len(precision)) if precision[index, index] > 0]
    while kept:
        block = precision[np.ix_(kept, kept)]
        scale = np.sqrt(np.diag(block))
        eigenvalues, eigenvectors = np.linalg.eigh(block / np.outer(scale, scale))
        # Scaling by a positive diagonal preserves definiteness, so a floor on the
        # smallest eigenvalue here both guarantees a Cholesky factor and caps how much
        # round-off the inversion can amplify.
        if eigenvalues[0] > tolerance:
            return kept
        kept.pop(int(np.argmax(np.abs(eigenvectors[:, 0]))))
    return kept


def sample_laplace_posterior(
    model: pm.Model, laplace: LaplaceApproximation, draws: int = 1000, seed: int = 42
) -> az.InferenceData:
    """Draw from the approximation and map the draws back onto the parameters' scale.

    The back-transform keeps every draw inside its prior bounds, at the cost of making
    the constrained-space marginals skewed rather than Gaussian.

    Parameters
    ----------
    model : pm.Model
        The calibration model the approximation was built from.
    laplace : LaplaceApproximation
        The fitted approximation.
    draws : int
        Number of posterior draws to generate.
    seed : int
        Seed for the Gaussian draws.

    Returns
    -------
    az.InferenceData
        The draws as a single-chain posterior, shaped like an MCMC trace. Every model
        parameter appears, with `laplace.fixed` ones repeated at their MAP value, so the
        result drops straight into the tooling that expects a full posterior.
    """
    rng = np.random.default_rng(seed)
    factor = np.linalg.cholesky(laplace.covariance)
    varying = laplace.mean[:, None] + factor @ rng.standard_normal((len(laplace.names), draws))

    # The back-transform needs every value variable at once, so slot the varying draws
    # and the parameters held at their MAP back into model order.
    positions = {name: index for index, name in enumerate(laplace.names)}
    unconstrained = np.empty((len(model.free_RVs), draws), dtype=np.float64)
    for index, rv in enumerate(model.free_RVs):
        if rv.name in positions:
            unconstrained[index] = varying[positions[rv.name]]
        else:
            unconstrained[index] = _forward_value(model, rv, laplace.fixed[rv.name])

    backward_fn = _backward_function(model)
    constrained = np.asarray(
        [backward_fn(*column) for column in unconstrained.T], dtype=np.float64
    )
    return az.from_dict(
        {
            rv.name: constrained[:, index].reshape(1, draws)
            for index, rv in enumerate(model.free_RVs)
        }
    )
