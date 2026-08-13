"""A minimal synthetic single-species 3PG stand, shared by the calibration tests.

The stand is intentionally tiny and only produces during its first months, so
observations are placed there. ``N_OBS`` (3) is kept distinct from ``n_species``
(1) so a stray species axis shows up as a shape error.
"""

import jax
import jax.numpy as jnp
import numpy as np

from trunx.gp3.model_inputs import ClimateData, Params, SiteData, SpeciesData, State

# Enable double precision before any array is created, matching the samplers.
jax.config.update("jax_enable_x64", True)

N_MONTHS = 72
OBS_MONTHS = jnp.asarray([0, 1, 2], dtype=jnp.int32)
TRUE_ALPHA_CX = 0.06


def one(value: float) -> jnp.ndarray:
    """Single-species array holding ``value``."""
    return jnp.full((1,), float(value))


def build_inputs() -> tuple[State, ClimateData, Params, SiteData, SpeciesData]:
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
        FR=one(0.5),
        WF=one(5.0),
        WR=one(5.0),
        WS=one(50.0),
        N=one(1000.0),
        year_p=jnp.asarray([1950]),
        month_p=jnp.asarray([1]),
    )
    state = State(
        WF=one(5.0),
        WR=one(5.0),
        WS=one(50.0),
        N=one(1000.0),
        ASW=one(200.0),
        age=one(240.0),
        WF_debt=one(0.0),
        prev_month=jnp.full((1,), 12, dtype=jnp.int32),
    )

    defaults = {field: one(1.0) for field in Params._fields}
    defaults.update(
        pFS2=one(1.0),
        pFS20=one(0.15),
        aWS=one(0.1),
        nWS=one(2.4),
        pRx=one(0.4),
        pRn=one(0.2),
        Tmin=one(2.0),
        Topt=one(16.0),
        Tmax=one(32.0),
        leafgrow=one(4),
        leaffall=one(10),
        alphaCx=one(TRUE_ALPHA_CX),
        gDM_mol=one(24.0),
        molPAR_MJ=one(2.3),
        Y=one(0.47),
        MaxAge=one(300.0),
        rAge=one(0.95),
        nAge=one(4.0),
        SLA0=one(10.0),
        SLA1=one(6.0),
        tSLA=one(20.0),
        k=one(0.5),
        fullCanAge=one(20.0),
        MaxIntcptn=one(0.15),
        LAImaxIntcptn=one(5.0),
        aH=one(2.0),
        nHB=one(0.5),
        nHC=one(0.0),
        aV=one(0.01),
        nVB=one(2.0),
        nVH=one(1.0),
        rhoMin=one(0.4),
        rhoMax=one(0.5),
        tRho=one(4.0),
        wSx1000=one(300.0),
        thinPower=one(1.5),
        mF=one(0.0),
        mR=one(0.2),
        mS=one(0.2),
        gammaN0=one(0.0),
        gammaN1=one(0.0),
        tgammaN=one(0.0),
        ngammaN=one(1.0),
        gammaF0=one(0.001),
        gammaF1=one(0.08),
        tgammaF=one(24.0),
        gammaR=one(0.015),
    )
    return state, climate, Params(**defaults), site, species
