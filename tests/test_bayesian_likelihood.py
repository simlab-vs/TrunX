"""Cheap likelihood checks for 3PG Bayesian calibration.

These tests do not run MCMC and do not read any data files: they build a
small synthetic single-species setup, drive ``run_3pg`` directly, and check
the observation log-likelihood used by the samplers. The likelihood math
mirrors ``Run3PGLogLikeOp._loglikelihood`` in ``pymc_param_est`` and the
``model`` in ``parameter_estimation`` (Gaussian, summed over observations).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.stats import norm

from trunx.gp3.model_inputs import ClimateData, Params, SiteData, SpeciesData, State
from trunx.gp3.run_3pg import run_3pg

# Enable double precision before any array is created, matching the samplers.
jax.config.update("jax_enable_x64", True)

# The synthetic stand below is intentionally minimal and only produces during
# its first months, so observations are placed there. n_obs (3) is kept
# distinct from n_species (1) so a stray species axis shows up as a shape error.
N_MONTHS = 72
OBS_MONTHS = jnp.asarray([0, 1, 2], dtype=jnp.int32)
TRUE_ALPHA_CX = 0.06


def _one(value: float) -> jnp.ndarray:
    """Single-species array holding ``value``."""
    return jnp.full((1,), float(value))


def _build_inputs() -> tuple[State, ClimateData, Params, SiteData, SpeciesData]:
    """Construct a synthetic single-species 3PG setup that runs without data files."""
    month = jnp.asarray(np.tile(np.arange(1, 13), N_MONTHS // 12), dtype=jnp.int32)
    climate = ClimateData(
        T_avg=jnp.full(N_MONTHS, 12.0),
        T_max=jnp.full(N_MONTHS, 18.0),
        VPD=jnp.full(N_MONTHS, 0.5),
        precip=jnp.full(N_MONTHS, 60.0),
        solar_rad=jnp.full(N_MONTHS, 12.0),
        frost_days=jnp.zeros(N_MONTHS),
        n_days=jnp.full(N_MONTHS, 30.0),
        co2=jnp.full(N_MONTHS, 400.0),
        d13catm=jnp.full(N_MONTHS, -8.0),
        month=month,
    )
    site = SiteData(
        latitude=jnp.asarray([51.0]),
        altitude=jnp.asarray([500.0]),
        soil_class=jnp.asarray([2.0]),
        ASW=jnp.asarray([200.0]),
        ASW_max=jnp.asarray([300.0]),
        ASW_min=jnp.asarray([0.0]),
        year_i=jnp.asarray([1970]),
        month_i=jnp.asarray([1]),
    )
    species = SpeciesData(
        specie=jnp.asarray([0]),
        FR=_one(0.5),
        WF=_one(5.0),
        WR=_one(5.0),
        WS=_one(50.0),
        N=_one(1000.0),
        year_p=jnp.asarray([1950]),
        month_p=jnp.asarray([1]),
    )
    state = State(
        WF=_one(5.0),
        WR=_one(5.0),
        WS=_one(50.0),
        N=_one(1000.0),
        ASW=_one(200.0),
        age=_one(240.0),
        WF_debt=_one(0.0),
        prev_month=jnp.full((1,), 12, dtype=jnp.int32),
    )

    defaults = {field: _one(1.0) for field in Params._fields}
    defaults.update(
        pFS2=_one(1.0),
        pFS20=_one(0.15),
        aWS=_one(0.1),
        nWS=_one(2.4),
        pRx=_one(0.4),
        pRn=_one(0.2),
        Tmin=_one(2.0),
        Topt=_one(16.0),
        Tmax=_one(32.0),
        leafgrow=_one(4),
        leaffall=_one(10),
        alphaCx=_one(TRUE_ALPHA_CX),
        gDM_mol=_one(24.0),
        molPAR_MJ=_one(2.3),
        Y=_one(0.47),
        MaxAge=_one(300.0),
        rAge=_one(0.95),
        nAge=_one(4.0),
        SLA0=_one(10.0),
        SLA1=_one(6.0),
        tSLA=_one(20.0),
        k=_one(0.5),
        fullCanAge=_one(20.0),
        MaxIntcptn=_one(0.15),
        LAImaxIntcptn=_one(5.0),
        aH=_one(2.0),
        nHB=_one(0.5),
        nHC=_one(0.0),
        aV=_one(0.01),
        nVB=_one(2.0),
        nVH=_one(1.0),
        rhoMin=_one(0.4),
        rhoMax=_one(0.5),
        tRho=_one(4.0),
        wSx1000=_one(300.0),
        thinPower=_one(1.5),
        mF=_one(0.0),
        mR=_one(0.2),
        mS=_one(0.2),
        gammaN0=_one(0.0),
        gammaN1=_one(0.0),
        tgammaN=_one(0.0),
        ngammaN=_one(1.0),
        gammaF0=_one(0.001),
        gammaF1=_one(0.08),
        tgammaF=_one(24.0),
        gammaR=_one(0.015),
    )
    return state, climate, Params(**defaults), site, species


def _output_at(var_name: str, alpha_cx: float) -> jnp.ndarray:
    """One model output at the observation months — shape ``(n_obs, n_species)``."""
    state, climate, params, site, species = _build_inputs()
    _, outputs = run_3pg(state, climate, params._replace(alphaCx=_one(alpha_cx)), site, species)
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
