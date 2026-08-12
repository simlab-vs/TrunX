"""Cheap likelihood checks for 3PG Bayesian calibration.

These tests do not run MCMC and do not read any data files: they drive
``run_3pg`` on the synthetic stand from ``synthetic_stand`` and check the
observation log-likelihood used by the samplers. The likelihood math mirrors
``Run3PGLogLikeOp._loglikelihood`` in ``pymc_param_est`` and the ``model`` in
``parameter_estimation`` (Gaussian, summed over observations).
"""

import jax.numpy as jnp
import pytest
from jax.scipy.stats import norm
from synthetic_stand import OBS_MONTHS, TRUE_ALPHA_CX, build_inputs, one

from trunx.gp3.run_3pg import run_3pg


def _output_at(var_name: str, alpha_cx: float) -> jnp.ndarray:
    """One model output at the observation months — shape ``(n_obs, n_species)``."""
    state, climate, params, site, species = build_inputs()
    _, outputs = run_3pg(state, climate, params._replace(alphaCx=one(alpha_cx)), site, species)
    return outputs[var_name][OBS_MONTHS]


def _obs_loglik(pred: jnp.ndarray, obs: jnp.ndarray, sigma: float) -> jnp.ndarray:
    """Gaussian observation log-likelihood, as used by both single-site samplers."""
    pred = jnp.asarray(pred).reshape(-1)
    obs = jnp.asarray(obs).reshape(-1)
    assert pred.shape == obs.shape, f"shape mismatch: {pred.shape} vs {obs.shape}"
    return jnp.sum(norm.logpdf(pred, loc=obs, scale=sigma))


def test_prediction_and_observation_shapes_match() -> None:
    """The model's per-observation predictions carry the species axis observations need."""
    pred = _output_at("DBH", TRUE_ALPHA_CX)
    assert pred.shape == (OBS_MONTHS.shape[0], 1)  # (n_obs, n_species)

    # Observations aligned to the species axis (as the loaders now return them).
    obs = pred  # noise-free stand-in with the correct (n_obs, n_species) shape
    log_probs = norm.logpdf(pred, loc=obs, scale=1.0)
    assert log_probs.shape == (OBS_MONTHS.shape[0], 1)
    assert log_probs.size == OBS_MONTHS.shape[0]


def test_bare_species_axis_broadcasts_into_outer_product() -> None:
    """Regression witness: 1-D observations broadcast every prediction against every obs."""
    pred = _output_at("DBH", TRUE_ALPHA_CX)  # (n_obs, 1)
    obs_1d = pred.reshape(-1)  # the old (n_obs,) Excel-loader shape

    buggy = norm.logpdf(pred, loc=obs_1d, scale=1.0)
    n_obs = OBS_MONTHS.shape[0]
    assert buggy.shape == (n_obs, n_obs)  # the silent bug this guard prevents

    with pytest.raises(AssertionError):
        _obs_loglik(pred, obs_1d[:-1], sigma=1.0)  # genuinely mismatched lengths


def test_likelihood_recovers_known_parameter() -> None:
    """Log-likelihood peaks at the alphaCx that generated the (noise-free) observations.

    NPP is used as the observed variable because it responds monotonically to
    alphaCx in this minimal synthetic stand, giving a clean, deterministic
    recovery signal for the likelihood.
    """
    obs = _output_at("NPP", TRUE_ALPHA_CX)
    sigma = 1.0

    ll_true = _obs_loglik(_output_at("NPP", TRUE_ALPHA_CX), obs, sigma)
    ll_low = _obs_loglik(_output_at("NPP", TRUE_ALPHA_CX * 0.6), obs, sigma)
    ll_high = _obs_loglik(_output_at("NPP", TRUE_ALPHA_CX * 1.4), obs, sigma)

    assert ll_true > ll_low
    assert ll_true > ll_high
