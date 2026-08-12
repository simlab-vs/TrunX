"""End-to-end checks for MAP calibration on the synthetic stand.

These run the real optimiser against the real PyMC model built by
``pymc_model``, but on a two-parameter problem with noise-free observations
generated from a known ``alphaCx``, so the mode is known in advance.
"""

from typing import Any, cast

import jax.numpy as jnp
import pytest
from synthetic_stand import OBS_MONTHS, TRUE_ALPHA_CX, build_inputs, one

from trunx.gp3.bayesiancalibrations.map_param_est import (
    map_to_inference_data,
    run_map_estimation,
)
from trunx.gp3.run_3pg import run_3pg

PRIORS = {"alphaCx": (0.01, 0.12), "err_NPP": (0.001, 1.0)}


def _observations() -> dict[str, tuple[jnp.ndarray, jnp.ndarray]]:
    """Noise-free NPP observations generated at ``TRUE_ALPHA_CX``."""
    state, climate, params, site, species = build_inputs()
    _, outputs = run_3pg(state, climate, params, site, species)
    return {"NPP": (OBS_MONTHS, outputs["NPP"][OBS_MONTHS])}


def _run_map(**kwargs) -> tuple[dict[str, float], float]:
    """Run MAP estimation on the synthetic stand."""
    state, climate, params, site, species = build_inputs()
    # Start away from the truth so recovery is the optimiser's doing, not the seed's.
    map_estimate, logp, _ = run_map_estimation(
        initial_state=state,
        climate=climate,
        site=site,
        species=species,
        fixed_params=params._replace(alphaCx=one(0.02)),
        observations=_observations(),
        priors=PRIORS,
        param_defaults={"alphaCx": 0.02, "err_NPP": 0.5},
        **kwargs,
    )
    return map_estimate, logp


def test_map_recovers_known_parameter() -> None:
    map_estimate, logp = _run_map()

    assert map_estimate["alphaCx"] == pytest.approx(TRUE_ALPHA_CX, rel=0.05)
    # Noise-free observations push the error term to its lower prior bound, so the
    # log posterior of the mode is large and positive.
    assert logp > 0.0
    for name, (lower, upper) in PRIORS.items():
        assert lower <= map_estimate[name] <= upper


def test_restarts_do_not_worsen_the_mode() -> None:
    _, logp_single = _run_map()
    _, logp_restarts = _run_map(n_restarts=2)

    assert logp_restarts >= logp_single


def test_map_to_inference_data_has_one_draw() -> None:
    map_estimate, _ = _run_map()
    idata = map_to_inference_data(map_estimate)

    posterior = cast(Any, idata).posterior
    assert posterior.sizes["chain"] == 1
    assert posterior.sizes["draw"] == 1
    assert set(posterior.data_vars) == set(PRIORS)
    assert float(posterior["alphaCx"].values[0, 0]) == map_estimate["alphaCx"]
