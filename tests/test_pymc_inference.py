"""End-to-end checks for PyMC MCMC calibration on the synthetic stand.

Runs the real `pymc_model` and `pm.sample`, but on a two-parameter problem with
noise-free observations generated from a known `alphaCx`, so both the mode and a
converged sampler's diagnostics are known in advance.
"""

from typing import cast

import arviz as az
import jax.numpy as jnp
import pandas as pd
import pytest
from synthetic_stand import OBS_MONTHS, TRUE_ALPHA_CX, build_inputs, one

from trunx.gp3.bayesiancalibrations.pymc_param_est import run_pymc_inference
from trunx.gp3.run_3pg import run_3pg

PRIORS = {"alphaCx": (0.01, 0.12), "err_NPP": (0.001, 1.0)}


def _observations() -> dict[str, tuple[jnp.ndarray, jnp.ndarray]]:
    """Noise-free NPP observations generated at ``TRUE_ALPHA_CX``."""
    state, climate, params, site, species = build_inputs()
    _, outputs = run_3pg(state, climate, params, site, species)
    return {"NPP": (OBS_MONTHS, outputs["NPP"][OBS_MONTHS])}


def test_nuts_recovers_known_parameter_with_healthy_diagnostics() -> None:
    state, climate, params, site, species = build_inputs()

    idata, _ = run_pymc_inference(
        initial_state=state,
        climate=climate,
        site=site,
        species=species,
        fixed_params=params._replace(alphaCx=one(0.02)),
        observations=_observations(),
        priors=PRIORS,
        param_defaults={"alphaCx": 0.02, "err_NPP": 0.5},
        num_warmup=500,
        num_samples=500,
        chains=2,
        cores=1,
        step_method="nuts",
    )

    summary = cast(pd.DataFrame, az.summary(idata))
    assert summary.loc["alphaCx", "mean"] == pytest.approx(TRUE_ALPHA_CX, rel=0.05)
    # A converged NUTS run should mix far better than DEMetropolisZ's near-zero ESS
    # observed on real 3PG calibrations (see project notes on the DEz comparison).
    assert cast(float, summary.loc["alphaCx", "r_hat"]) < 1.05
    assert cast(float, summary.loc["alphaCx", "ess_bulk"]) > 50


def test_invalid_step_method_raises() -> None:
    state, climate, params, site, species = build_inputs()

    with pytest.raises(ValueError, match="step_method"):
        run_pymc_inference(
            initial_state=state,
            climate=climate,
            site=site,
            species=species,
            fixed_params=params,
            observations=_observations(),
            priors=PRIORS,
            step_method="bogus",
        )
