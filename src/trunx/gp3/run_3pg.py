"""Run the 3PG model."""

import os

import jax
import jax.numpy as jnp
from jax import debug

from trunx.gp3.helper_function import (
    apply_self_thinning,
    calculate_base_conductance,
    calculate_day_length,
    compute_allocation_fraction,
    compute_asw,
    compute_canopy_cover,
    compute_dbh,
    compute_lai,
    compute_light_interception,
    compute_litterfall_rate,
    f_age,
    f_calpha,
    f_cg,
    f_exp_foliage,
    f_frost,
    f_nutrition,
    f_soil_water,
    f_temperature,
    f_temperature_gc,
    f_vpd,
    is_dormant,
)
from trunx.gp3.model_inputs import State


def model_step(state, climate_month, params, site, species, n_species):
    """Compute one model step."""
    T_avg, T_max, VPD, precip, solar_rad, frost_days, co2, n_days, month = climate_month
    WF, WR, WS, N, ASW, age_months, WF_debt, prev_month = state

    # Check if dormant
    dormant = is_dormant(month, params.leafgrow, params.leaffall)
    prev_dormant = is_dormant(prev_month, params.leafgrow, params.leaffall)

    first_dormant = jnp.array(dormant & ~prev_dormant)
    first_growing = jnp.array(~dormant & prev_dormant)

    WF_active = jnp.where(first_dormant, 0.0, WF)
    WF_debt_new = jnp.where(first_dormant, WF, WF_debt)

    WF_active = jnp.where(first_growing, WF_debt, WF_active)
    WF_debt_new = jnp.where(first_growing, 0.0, WF_debt_new)

    # Leaf area index
    LAI, SLA = compute_lai(WF, age_months, params.SLA0, params.SLA1, params.tSLA)
    LAI = jnp.clip(LAI, 0.0, 15.0)
    lai_total = jnp.sum(LAI)
    LAI_per = jnp.where(lai_total > 0.0, LAI / lai_total, 0.0)

    # Light interception (Beer's Law)
    canopy_cover = compute_canopy_cover(age_months, params.fullCanAge)
    lightIntcptn = compute_light_interception(params.k, LAI, canopy_cover)
    APAR = solar_rad * n_days * lightIntcptn * canopy_cover

    # Growth modifiers
    fT = f_temperature(T_avg, params.Tmin, params.Topt, params.Tmax)
    fF = f_frost(frost_days, params.kF)
    fN = f_nutrition(species.FR, params.fN0, params.fNn)
    fD = f_vpd(VPD, params.CoeffCond)
    fSW = f_soil_water(ASW, site.ASW_max, params.SWconst, params.SWpower, site.soil_class)
    fA = f_age(age_months, params.MaxAge, params.nAge, params.rAge)
    fcalpha = f_calpha(co2, params.fCalpha700)

    phi = fA * jnp.minimum(fD, fSW)
    # phi = fA * fD * fSW

    alpha_c = params.alphaCx * fT * fF * fN * phi * fcalpha
    alpha_c = jnp.where(LAI == 0.0, 0.0, alpha_c)
    # Primary production
    epsilon = params.gDM_mol * params.molPAR_MJ * alpha_c
    # GPP = alpha_c * APAR
    GPP = epsilon * APAR / 100
    NPP = params.Y * GPP

    DBH = compute_dbh(WS, N, params.aWS, params.nWS)

    pFS, eta_F, eta_S, eta_R = compute_allocation_fraction(
        species.FR, params.pRx, params.pRn, params.pFS2, params.pFS20, phi, DBH, params.m0
    )
    # Turnover
    # gammaF = compute_litterfall_rate(age_months, params.gammaF0, params.gammaF1, params.tgammaF)
    gammaF = f_exp_foliage(age_months, params.gammaF0, params.gammaF1, params.tgammaF)
    # Biomass updates
    # WF_new = jnp.clip(WF + eta_F * NPP - gammaF * WF, 0.0, None)
    WF_new = jnp.where(
        dormant,
        WF_active - gammaF * WF_active,  # Only litterfall in dormant months
        WF_active + eta_F * NPP - gammaF * WF_active,  # Normal growth
    )
    WF_new = jnp.clip(WF_new, 0.0, None)
    # WR_new = jnp.clip(WR + eta_R * NPP - params.gammaR * WR, 0.0, None)
    WR_new = jnp.where(dormant, WR, WR + eta_R * NPP - params.gammaR * WR)
    WR_new = jnp.clip(WR_new, 0.0, None)

    WS_new = jnp.clip(WS + eta_S * NPP, 0.0, None)

    # Self-thinning
    WS_new, N_new = apply_self_thinning(WS_new, N, params.wSx1000, params.thinPower)

    gC = calculate_base_conductance(lai_total, params.MaxCond, params.MinCond, params.LAIgcx)
    ftmp_gc = f_temperature_gc(T_avg, T_max, params.Tmin, params.Topt, params.Tmax)
    fcg = f_cg(co2, params.fCg700)
    conduct_canopy = gC * LAI_per * phi * ftmp_gc * fcg
    day_length = calculate_day_length(site.latitude, month)

    ASW_new, f_transp_scale = compute_asw(
        params,
        site,
        ASW=ASW,
        prcp=precip,
        solar_rad=solar_rad,
        VPD=VPD,
        day_length=day_length,
        days_in_month=n_days,
        conduct_canopy=conduct_canopy,
        lai=LAI,
    )

    GPP = GPP * f_transp_scale
    NPP = NPP * f_transp_scale

    new_state = State(
        WF=WF_new,
        WR=WR_new,
        WS=WS_new,
        N=N_new,
        ASW=ASW_new,
        age=jnp.asarray(age_months + 1),
        WF_debt=jnp.asarray(WF_debt_new),
        prev_month=jnp.full(n_species, month),
    )

    outputs = dict(
        GPP=GPP,
        NPP=NPP,
        LAI=LAI,
        APAR=APAR,
        DBH=DBH,
        fT=fT,
        fD=fD,
        fSW=fSW,
        fAge=fA,
        fN=fN,
        fF=fF,
        phi=phi,
        eta_R=eta_R,
        eta_F=eta_F,
        eta_S=eta_S,
        WF=WF_new,
        WR=WR_new,
        WS=WS_new,
        N=N_new,
        ASW=ASW_new,
        pFS=pFS,
        SLA=SLA,
        alpha_c=alpha_c,
        fcalpha=fcalpha,
        gammaF=gammaF,
        Volume=WS_new * 0.85 / (params.tRho + 1e-8),
    )

    return new_state, outputs


def run_3pg(initial_state, climate, params, site, species, n_species):
    """Run 3PG model."""
    climate_stack = jnp.stack(
        [
            climate.T_avg,
            climate.T_max,
            climate.VPD,
            climate.precip,
            climate.solar_rad,
            climate.frost_days,
            climate.co2,
            climate.n_days,
            climate.month,
        ],
        axis=-1,
    )

    def step(state, climate_row):
        return model_step(state, climate_row, params, site, species, n_species)

    return jax.lax.scan(step, initial_state, climate_stack)


def ws_final(alphaCx, CoeffCond, Y_val, params, initial_state, climate, site, species, n_species):
    """Compute final stem biomass as a scalar function."""
    p = params._replace(alphaCx=alphaCx, CoeffCond=CoeffCond, Y=Y_val)
    final_state, _ = run_3pg(initial_state, climate, p, site, species, n_species)
    return final_state.WS


def ws_final_vector(params_vec, params, initial_state, climate, site, species, n_species):
    """Compute final stem biomass for all species with params as a vector."""
    alphaCx, CoeffCond, Y_val = params_vec
    p = params._replace(alphaCx=alphaCx, CoeffCond=CoeffCond, Y=Y_val)
    final_state, _ = run_3pg(initial_state, climate, p, site, species, n_species)
    return final_state.WS


def loss_fn(log_params_arr, fixed_params, s0, climate, site, obs_WS, obs_times, species):
    """
    MSE loss for gradient-based calibration.

    log_params_arr:
        [log(alphaCx), log(CoeffCond), logit(Y)]
    """
    alphaCx = jnp.exp(log_params_arr[0])
    CoeffCond = jnp.exp(log_params_arr[1])
    Y = jax.nn.sigmoid(log_params_arr[2])

    params = fixed_params._replace(alphaCx=alphaCx, CoeffCond=CoeffCond, Y=Y)
    _, outputs = run_3pg(s0, climate, params, site, species, len(species))

    pred_WS = outputs["WS"][obs_times]
    return jnp.mean((pred_WS - obs_WS) ** 2)
