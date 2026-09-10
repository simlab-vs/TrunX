"""Run the 3PG model."""

import os
import warnings

import jax
import jax.numpy as jnp
from jax import debug, lax

from trunx.gp3.extended_helper import INPUT_VARIABLES, poly_nm
from trunx.gp3.helper_function import (
    apply_self_thinning_with_mortality_factors,
    apply_stress_mortality,
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
    f_exp_wood,
    f_frost,
    f_nutrition,
    f_soil_water,
    f_temperature,
    f_temperature_gc,
    f_vpd,
    is_dormant,
)
from trunx.gp3.model_inputs import State


def model_step(state, climate_month, params, site, species):
    """Compute one model step."""
    (
        T_avg,
        T_max,
        VPD,
        precip,
        solar_rad,
        frost_days,
        co2,
        n_days,
        month,
        fpoly_nn,
    ) = climate_month

    WF, WR, WS, N, ASW, age_months, WF_debt, prev_month = state

    # Check if dormant
    dormant = is_dormant(month, params.leafgrow, params.leaffall)
    prev_dormant = is_dormant(prev_month, params.leafgrow, params.leaffall)

    first_dormant = jnp.logical_and(dormant, jnp.logical_not(prev_dormant))
    first_growing = jnp.logical_and(jnp.logical_not(dormant), prev_dormant)

    WF_active = jnp.where(first_dormant, 0.0, WF)
    WF_debt_new = jnp.where(first_dormant, WF, WF_debt)

    WF_active = jnp.where(first_growing, WF_debt, WF_active)
    # WF_debt_new = jnp.where(first_growing, 0.0, WF_debt_new)

    # Leaf area index
    # Note with WF, we get better fitting.
    LAI, SLA = compute_lai(params, WF_active, age_months)

    lai_total = jnp.sum(LAI)
    lai_total = jnp.where(lai_total > 0.0, lai_total, 1.0)
    LAI_per = jnp.where(lai_total > 0.0, LAI / lai_total, 0.0)

    # Light interception (Beer's Law)
    canopy_cover = compute_canopy_cover(params, age_months)
    lightIntcptn = compute_light_interception(params, LAI, canopy_cover)
    APAR = solar_rad * n_days * lightIntcptn * canopy_cover

    # Growth modifiers
    fT = f_temperature(params, T_avg)
    fF = f_frost(params, frost_days, n_days)
    fN = f_nutrition(species, params)
    fD = f_vpd(VPD, params.CoeffCond)
    fSW = f_soil_water(ASW, site, params)
    fA = f_age(params, age_months)
    fcalpha = f_calpha(params, co2)

    # phi = fA * jnp.minimum(fD, fSW)
    phi = fA * fD * fSW

    gC = calculate_base_conductance(params, lai_total)
    ftmp_gc = f_temperature_gc(params, T_avg, T_max)
    # ftmp_gc = 1.0
    fcg = f_cg(params, co2)
    conduct_canopy = gC * LAI_per * phi * ftmp_gc * fcg

    alpha_c = params.alphaCx * fT * fF * fN * phi * fcalpha * fpoly_nn

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
        lai_total=lai_total,
        lai_per=LAI_per,
    )

    GPP = GPP * f_transp_scale
    NPP_scaled = NPP * f_transp_scale

    DBH = compute_dbh(params, WS, N)

    pFS, eta_F, eta_S, eta_R = compute_allocation_fraction(species, params, phi, DBH)

    # Turnover
    # gammaF = compute_litterfall_rate(age_months, params.gammaF0, params.gammaF1, params.tgammaF)
    gammaF = f_exp_foliage(params, age_months)
    gammaF = jnp.clip(gammaF, 0.0, 1.0)
    wood_density = f_exp_wood(params, age_months)
    WF_debt_after = WF_debt_new
    NPP_after_debt = NPP_scaled

    growing = ~dormant
    has_debt = WF_debt_new > 0.0

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
    biom_loss_foliage = jnp.where(
        dormant & first_dormant,
        WF_debt_new,
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

    mort_stress = jnp.zeros_like(N)

    # Stress mortality
    WS_stress, WF_stress, WR_stress, N_stress, mort_stress = apply_stress_mortality(
        params, age_months, WS_new, WF_new, WR_new, N, dormant
    )
    WS_new = WS_stress
    WF_new = WF_stress
    WR_new = WR_stress
    N_new = N_stress

    WS_thinned, WF_thinned, WR_thinned, N_thinned, mort_count = (
        apply_self_thinning_with_mortality_factors(params, WS_new, WF_new, WR_new, N_new, dormant)
    )

    WS_new = WS_thinned
    WF_new = WF_thinned
    WR_new = WR_thinned
    N_new = N_thinned

    # Recalculate DBH after all biomass events
    DBH_updated = compute_dbh(params, WS_new, N_new)

    # Recalculate LAI after all updates
    LAI, SLA = compute_lai(params, WF_new, age_months + 1)

    BA = jnp.pi * (DBH_updated / 200.0) ** 2 * N_new  # Basal area in m^2/ha
    competition_total = jnp.sum(wood_density * BA)
    H = params.aH * DBH_updated**params.nHB * competition_total**params.nHC  # Height in m

    # Forrester et al. (2021) height equation with 1.3 m offset for breast height
    # H = 1.3 + params.aH * jnp.exp(-params.nHB / DBH_updated) + \
    # params.nHC * competition_total * DBH_updated

    V = params.aV * DBH_updated**params.nVB * H**params.nVH  # Volume in m^3/ha

    new_state = State(
        WF=WF_new,
        WR=WR_new,
        WS=WS_new,
        N=N_new,
        ASW=ASW_new,
        age=jnp.asarray(age_months + 1),
        WF_debt=jnp.asarray(WF_debt_after),
        prev_month=jnp.full_like(N, month, dtype=jnp.int32),
    )

    outputs = dict(
        GPP=GPP,
        NPP=NPP,
        LAI=LAI,
        APAR=APAR,
        DBH=DBH_updated,
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
        f_cg=fcg,
        mort_stress=mort_stress,
        mort_thinn=mort_count,
        stems_n=N_new,
        Volume=V,
        Height=H,
        BA=BA,
    )

    return new_state, outputs


def run_3pg(
    initial_state,
    climate,
    params,
    site,
    species,
    deposition=None,
    extended_params=None,
    modifier_fn=poly_nm,
    input_vars=INPUT_VARIABLES,
):
    """Run 3PG model.

    Parameters
    ----------
    modifier_fn : Callable
        Nutrition modifier applied to `extended_params.modifier_params`, e.g.
        `poly_nm`, `saturating_nm`, `saturating_poly_nm`, or `mlp_nm` from
        `extended_helper.py`. Ignored when `extended_params` is None.
    input_vars : tuple[str, ...]
        Which of `("N", "S", "T_avg")` (nitrogen deposition, sulphur deposition,
        temperature) `modifier_fn` was built over, and in what order — must
        match `extended_params.modifier_params`'s axes/fields. Ignored when
        `extended_params` is None.
    """
    if extended_params is None:
        fpoly_nn = jnp.ones_like(climate.T_avg, dtype=float)
    else:
        channels = {"T_avg": climate.T_avg}
        if "N" in input_vars or "S" in input_vars:
            dep_n_tot = getattr(deposition, "dep_n_tot", None)
            dep_s_so4 = getattr(deposition, "dep_s_so4", None)
            if dep_n_tot is None or dep_s_so4 is None:
                warnings.warn(
                    "Missing deposition fields (dep_n_tot and/or dep_s_so4); "
                    "running 3PG without deposition effects using zeros.",
                    UserWarning,
                    stacklevel=2,
                )
                dep_n_tot = jnp.zeros_like(climate.T_avg, dtype=float)
                dep_s_so4 = jnp.zeros_like(climate.T_avg, dtype=float)
            channels["N"] = dep_n_tot
            channels["S"] = dep_s_so4

        inputs = jnp.stack([channels[name] for name in input_vars], axis=-1)
        fpoly_nn = modifier_fn(extended_params.modifier_params, inputs, input_vars)

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
            fpoly_nn,
        ],
        axis=-1,
    )

    def step(state, climate_row):
        return model_step(state, climate_row, params, site, species)

    return jax.lax.scan(step, initial_state, climate_stack)


def ws_final(alphaCx, CoeffCond, Y_val, params, initial_state, climate, site, species):
    """Compute final stem biomass as a scalar function."""
    p = params._replace(alphaCx=alphaCx, CoeffCond=CoeffCond, Y=Y_val)
    final_state, _ = run_3pg(initial_state, climate, p, site, species)
    return final_state.WS


def ws_final_vector(params_vec, params, initial_state, climate, site, species):
    """Compute final stem biomass for all species with params as a vector."""
    alphaCx, CoeffCond, Y_val = params_vec
    p = params._replace(alphaCx=alphaCx, CoeffCond=CoeffCond, Y=Y_val)
    final_state, _ = run_3pg(initial_state, climate, p, site, species)
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
    _, outputs = run_3pg(s0, climate, params, site, species)

    pred_WS = outputs["WS"][obs_times]
    return jnp.mean((pred_WS - obs_WS) ** 2)
