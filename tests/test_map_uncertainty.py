"""Checks for the Laplace uncertainty estimates built on top of a MAP calibration.

These run the real optimiser and the real PyMC model on the synthetic stand, but on a
two-parameter problem with noisy observations generated from a known ``alphaCx``, so
the mode is interior and the curvature around it can be checked against an independent
calculation.
"""

from typing import Any, cast
from unittest import mock

import jax.numpy as jnp
import numpy as np
import pytest
from synthetic_stand import OBS_MONTHS, TRUE_ALPHA_CX, build_inputs, one

from trunx.gp3.bayesiancalibrations.map_param_est import run_map_estimation
from trunx.gp3.bayesiancalibrations.map_uncertainty import (
    LaplaceApproximation,
    _identified_subset,
    _unconstrained_vector,
    fit_laplace,
    log_posterior_hessian,
    sample_laplace_posterior,
)
from trunx.gp3.run_3pg import run_3pg

PRIORS = {"alphaCx": (0.01, 0.12), "err_NPP": (0.001, 1.0)}
NOISE = 0.2


def _observations(
    replicates: int = 1, noise: float = NOISE, seed: int = 0
) -> dict[str, tuple[jnp.ndarray, jnp.ndarray]]:
    """NPP observations at ``TRUE_ALPHA_CX``, optionally repeated ``replicates`` times.

    Noise keeps the error term away from its lower prior bound, so the mode is interior.
    Repeating the same values leaves the mode untouched while scaling the log-likelihood
    (and hence its curvature) by exactly ``replicates``.
    """
    state, climate, params, site, species = build_inputs()
    _, outputs = run_3pg(state, climate, params, site, species)
    values = jnp.asarray(outputs["NPP"][OBS_MONTHS]).reshape(-1)
    if noise > 0.0:
        rng = np.random.default_rng(seed)
        values = values + jnp.asarray(rng.normal(0.0, noise, values.shape))
    return {
        "NPP": (
            jnp.concatenate([OBS_MONTHS] * replicates),
            jnp.concatenate([values] * replicates),
        )
    }


def _map_and_model(**observation_kwargs: Any) -> tuple[dict[str, float], Any]:
    """Run MAP estimation on the synthetic stand and return the estimate and model."""
    state, climate, params, site, species = build_inputs()
    map_estimate, _, model = run_map_estimation(
        initial_state=state,
        climate=climate,
        site=site,
        species=species,
        fixed_params=params._replace(alphaCx=one(0.02)),
        observations=_observations(**observation_kwargs),
        priors=PRIORS,
        param_defaults={"alphaCx": 0.02, "err_NPP": 0.5},
    )
    return map_estimate, model


@pytest.fixture(scope="module")
def fitted() -> tuple[dict[str, float], Any, LaplaceApproximation]:
    """A MAP fit on the noisy stand, with its Laplace approximation."""
    map_estimate, model = _map_and_model()
    return map_estimate, model, fit_laplace(model, map_estimate)


def test_hessian_matches_a_finite_difference_of_the_log_posterior(
    fitted: tuple[dict[str, float], Any, LaplaceApproximation],
) -> None:
    """Cross-check the gradient-based Hessian against one built from logp values alone."""
    map_estimate, model, _ = fitted
    hessian, names = log_posterior_hessian(model, map_estimate)

    logp_fn = model.compile_logp(jacobian=False)
    value_names = [value_var.name for value_var in model.value_vars]
    center = _unconstrained_vector(model, map_estimate)

    def logp_at(vector: np.ndarray) -> float:
        return float(logp_fn(dict(zip(value_names, [np.asarray(v) for v in vector], strict=True))))

    step = 1e-3
    expected = np.zeros_like(hessian)
    for i in range(len(names)):
        for j in range(len(names)):
            step_i, step_j = np.zeros(len(names)), np.zeros(len(names))
            step_i[i], step_j[j] = step, step
            expected[i, j] = (
                logp_at(center + step_i + step_j)
                - logp_at(center + step_i - step_j)
                - logp_at(center - step_i + step_j)
                + logp_at(center - step_i - step_j)
            ) / (4 * step * step)

    np.testing.assert_allclose(hessian, expected, rtol=1e-4, atol=1e-3)


def test_hessian_is_symmetric_and_negative_definite(
    fitted: tuple[dict[str, float], Any, LaplaceApproximation],
) -> None:
    """A MAP is a maximum, so the log posterior curves downward in every direction."""
    map_estimate, model, _ = fitted
    hessian, _ = log_posterior_hessian(model, map_estimate)

    np.testing.assert_allclose(hessian, hessian.T, rtol=0, atol=0)
    assert np.all(np.linalg.eigvalsh(hessian) < 0.0)


def test_covariance_is_the_inverse_of_the_negated_hessian(
    fitted: tuple[dict[str, float], Any, LaplaceApproximation],
) -> None:
    map_estimate, model, laplace = fitted
    hessian, names = log_posterior_hessian(model, map_estimate)

    assert laplace.names == names
    np.testing.assert_allclose(laplace.covariance @ -hessian, np.eye(len(names)), atol=1e-8)


def test_uncertainty_shrinks_with_the_square_root_of_the_sample_size() -> None:
    """Four copies of the data quadruple the curvature, so the standard errors halve.

    The mode itself is unmoved by the replication, so this isolates the curvature.
    """
    single_estimate, single_model = _map_and_model(replicates=1)
    quadruple_estimate, quadruple_model = _map_and_model(replicates=4)

    for name in PRIORS:
        assert quadruple_estimate[name] == pytest.approx(single_estimate[name], rel=1e-6)

    single = fit_laplace(single_model, single_estimate)
    quadruple = fit_laplace(quadruple_model, quadruple_estimate)

    ratio = np.sqrt(np.diag(single.covariance)) / np.sqrt(np.diag(quadruple.covariance))
    np.testing.assert_allclose(ratio, 2.0, rtol=1e-3)


def test_draws_are_centred_on_the_map_and_stay_inside_the_priors(
    fitted: tuple[dict[str, float], Any, LaplaceApproximation],
) -> None:
    map_estimate, model, laplace = fitted
    idata = sample_laplace_posterior(model, laplace, draws=4000)
    posterior = cast(Any, idata).posterior

    assert posterior.sizes["chain"] == 1
    assert posterior.sizes["draw"] == 4000
    assert set(posterior.data_vars) == set(PRIORS)

    for name, (lower, upper) in PRIORS.items():
        draws = posterior[name].values.reshape(-1)
        assert np.all((draws > lower) & (draws < upper))
        # The back-transform is nonlinear, so the draws are only loosely centred and
        # the MAP is a mode rather than a mean.
        assert draws.mean() == pytest.approx(map_estimate[name], rel=0.15)
        assert draws.std() > 0.0

    # The uncertainty on alphaCx has to be small enough to still identify the truth.
    alpha_draws = posterior["alphaCx"].values.reshape(-1)
    assert abs(alpha_draws.mean() - TRUE_ALPHA_CX) < 3 * alpha_draws.std()


def test_draws_reproduce_the_approximated_standard_deviation(
    fitted: tuple[dict[str, float], Any, LaplaceApproximation],
) -> None:
    """The sampler must return the covariance it was handed, in unconstrained terms."""
    _, model, laplace = fitted
    idata = sample_laplace_posterior(model, laplace, draws=20000, seed=7)
    posterior = cast(Any, idata).posterior

    # alphaCx is far from its bounds, so the interval transform is near-linear there and
    # the constrained spread is the unconstrained one scaled by the local derivative.
    lower, upper = PRIORS["alphaCx"]
    index = laplace.names.index("alphaCx")
    center = laplace.mean[index]
    derivative = (upper - lower) * np.exp(-center) / (1 + np.exp(-center)) ** 2
    expected = derivative * np.sqrt(laplace.covariance[index, index])

    assert posterior["alphaCx"].values.std() == pytest.approx(expected, rel=0.05)


def test_sampling_is_reproducible(
    fitted: tuple[dict[str, float], Any, LaplaceApproximation],
) -> None:
    _, model, laplace = fitted
    first = sample_laplace_posterior(model, laplace, draws=100, seed=3)
    second = sample_laplace_posterior(model, laplace, draws=100, seed=3)

    np.testing.assert_array_equal(
        cast(Any, first).posterior["alphaCx"].values,
        cast(Any, second).posterior["alphaCx"].values,
    )


def test_a_mode_pinned_to_a_prior_bound_warns() -> None:
    """Noise-free observations drive the error term onto its lower bound.

    The log posterior is flat there, so the approximation must say so rather than pass
    the result off as a real standard error.
    """
    map_estimate, model = _map_and_model(noise=0.0)
    lower = PRIORS["err_NPP"][0]
    assert map_estimate["err_NPP"] == pytest.approx(lower, abs=1e-6)

    with pytest.warns(UserWarning, match="err_NPP"):
        laplace = fit_laplace(model, map_estimate)

    # alphaCx stays identified whatever happens to the error term.
    assert "alphaCx" in laplace.names
    posterior = cast(Any, sample_laplace_posterior(model, laplace, draws=2000)).posterior
    assert set(posterior.data_vars) == set(PRIORS)
    assert 0.0 < posterior["alphaCx"].values.std() < 0.01


def test_unidentified_parameters_are_held_at_their_map(
    fitted: tuple[dict[str, float], Any, LaplaceApproximation],
) -> None:
    """A boundary or flat direction is conditioned on, not allowed to sink the fit.

    This is the situation the real Solling calibration lands in, where several
    parameters optimise onto their prior bounds.
    """
    map_estimate, model, _ = fitted
    hessian, names = log_posterior_hessian(model, map_estimate)

    # Flatten err_NPP by hand, standing in for a parameter the data cannot pin down.
    flattened = hessian.copy()
    index = names.index("err_NPP")
    flattened[index, :] = 0.0
    flattened[:, index] = 0.0

    with (
        mock.patch(
            "trunx.gp3.bayesiancalibrations.map_uncertainty.log_posterior_hessian",
            return_value=(flattened, names),
        ),
        pytest.warns(UserWarning, match="Holding.*err_NPP.*fixed at their MAP"),
    ):
        laplace = fit_laplace(model, map_estimate)

    assert laplace.names == ["alphaCx"]
    assert laplace.fixed == {"err_NPP": map_estimate["err_NPP"]}
    assert laplace.covariance.shape == (1, 1)

    # The fixed parameter still shows up in the draws, so downstream tooling is unaffected.
    posterior = cast(Any, sample_laplace_posterior(model, laplace, draws=200)).posterior
    assert set(posterior.data_vars) == set(PRIORS)
    np.testing.assert_allclose(
        posterior["err_NPP"].values.reshape(-1), map_estimate["err_NPP"], rtol=1e-9
    )
    assert posterior["alphaCx"].values.std() > 0.0


def test_identified_subset_ignores_the_scale_of_each_parameter() -> None:
    """A well-identified parameter must survive however small its natural units are.

    Curvatures across the real calibration span ten orders of magnitude purely because
    the parameters do, so picking what to drop by raw curvature throws away good
    parameters (all six error terms, in the Solling fit) and keeps degenerate ones.
    """
    # Parameters 0 and 2 are all but perfectly collinear; 1 is independent but is
    # measured in units that make its curvature the smallest of the three by far.
    correlation = np.array([[1.0, 0.0, 1.0 - 1e-9], [0.0, 1.0, 0.0], [1.0 - 1e-9, 0.0, 1.0]])
    scale = np.diag([1e2, 1e-3, 1e2])
    precision = scale @ correlation @ scale
    assert np.argmin(np.diag(precision)) == 1

    kept = _identified_subset(precision)

    assert 1 in kept, "the independent parameter was dropped because its units are small"
    assert len(kept) == 2, "one of the collinear pair should be conditioned on"


def test_missing_parameter_is_rejected(
    fitted: tuple[dict[str, float], Any, LaplaceApproximation],
) -> None:
    _, model, _ = fitted

    with pytest.raises(KeyError, match="err_NPP"):
        fit_laplace(model, {"alphaCx": 0.06})
