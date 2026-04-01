"""Run the 3PG model."""

import os

import jax
import jax.numpy as jnp
from jax import debug, lax

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

    first_dormant = jnp.logical_and(dormant, jnp.logical_not(prev_dormant))
    first_growing = jnp.logical_and(jnp.logical_not(dormant), prev_dormant)

    WF_active = jnp.where(first_dormant, 0.0, WF)
    WF_debt_new = jnp.where(first_dormant, WF, WF_debt)

    WF_active = jnp.where(first_growing, WF_debt, WF_active)
    WF_debt_new = jnp.where(first_growing, 0.0, WF_debt_new)

    # Leaf area index
    # Note with WF, we get better fitting.
    LAI, SLA = compute_lai(params, WF_active, age_months)

    lai_total = jnp.sum(LAI)
    LAI_per = jnp.where(lai_total > 0.0, LAI / lai_total, 0.0)

    # Light interception (Beer's Law)
    canopy_cover = compute_canopy_cover(params, age_months)
    lightIntcptn = compute_light_interception(params, LAI, canopy_cover)
    APAR = solar_rad * n_days * lightIntcptn * canopy_cover

    # Growth modifiers
    fT = f_temperature(params, T_avg)
    fF = f_frost(params, frost_days)
    fN = f_nutrition(species, params)
    fD = f_vpd(VPD, params.CoeffCond)
    fSW = f_soil_water(ASW, site, params)
    fA = f_age(params, age_months)
    fcalpha = f_calpha(params, co2)

    phi = fA * jnp.minimum(fD, fSW)
    # phi = fA * fD * fSW

    gC = calculate_base_conductance(params, lai_total)
    ftmp_gc = f_temperature_gc(params, T_avg, T_max)
    fcg = f_cg(params, co2)
    conduct_canopy = gC * LAI_per * phi * ftmp_gc * fcg

    alpha_c = params.alphaCx * fT * fF * fN * phi * fcalpha
    alpha_c = jnp.where(LAI == 0.0, 0.0, alpha_c)
    # Primary production
    epsilon = params.gDM_mol * params.molPAR_MJ * alpha_c
    # GPP = alpha_c * APAR
    GPP = epsilon * APAR / 100
    NPP = params.Y * GPP

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
    NPP_scaled = NPP * f_transp_scale

    DBH = compute_dbh(params, WS, N)

    pFS, eta_F, eta_S, eta_R = compute_allocation_fraction(species, params, phi, DBH)

    # Turnover
    # gammaF = compute_litterfall_rate(age_months, params.gammaF0, params.gammaF1, params.tgammaF)
    gammaF = f_exp_foliage(params, age_months)
    gammaF = jnp.clip(gammaF, 0.0, 1.0)

    WF_debt_after = WF_debt_new
    NPP_after_debt = NPP_scaled

    growing = ~dormant
    has_debt = WF_debt_new > jnp.zeros(n_species, dtype=float)

    # Calculate new debt and NPP after repayment
    WF_debt_after = jnp.where(
        growing & has_debt,
        jnp.where(NPP_scaled >= WF_debt_new, 0.0, WF_debt_new - NPP_scaled),
        WF_debt_new,
    )

    NPP_after_debt = jnp.where(
        growing & has_debt,
        jnp.where(NPP_scaled >= WF_debt_new, NPP_scaled - WF_debt_new, 0.0),
        NPP_scaled,
    )

    # Calculate biomass losses (litterfall) using current foliage
    # biom_loss_foliage = jnp.where(growing, gammaF * WF_active, 0.0)

    biom_loss_foliage = jnp.where(
        dormant & first_dormant,
        WF_debt,  # should be WF_debt_new
        jnp.where(growing, gammaF * WF_active, 0.0),
    )

    biom_loss_root = jnp.where(dormant, 0.0, params.gammaR * WR)

    # Calculate biomass increments (only in growing season)
    biom_incr_foliage = jnp.where(dormant, 0.0, NPP_after_debt * eta_F)
    biom_incr_root = jnp.where(dormant, 0.0, NPP_after_debt * eta_R)
    biom_incr_stem = jnp.where(dormant, 0.0, NPP_after_debt * eta_S)

    # Update biomass starting from current values
    WF_new = WF_active + biom_incr_foliage - biom_loss_foliage
    WF_new = jnp.clip(WF_new, 0.0, None)

    WR_new = WR + biom_incr_root - biom_loss_root
    WR_new = jnp.clip(WR_new, 0.0, None)

    WS_new = WS + biom_incr_stem
    WS_new = jnp.clip(WS_new, 0.0, None)

    # Self-thinning (only in growing season)
    WS_thinned, N_thinned = apply_self_thinning(params, WS_new, N)
    WS_new = jnp.where(~dormant, WS_thinned, WS_new)
    N_new = jnp.where(~dormant, N_thinned, N)

    # Recalculate LAI after all updates
    # LAI, SLA = compute_lai(params, WF_new, age_months + 1)
    # LAI = jnp.clip(LAI, 0.0, 15.0)

    new_state = State(
        WF=WF_new,
        WR=WR_new,
        WS=WS_new,
        N=N_new,
        ASW=ASW_new,
        age=jnp.asarray(age_months + 1),
        WF_debt=jnp.asarray(WF_debt_after),
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
        f_transp_scale=f_transp_scale,
        conduct_canopy=conduct_canopy,
        f_cg=gC,
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
