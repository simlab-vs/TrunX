"""Helper functions to implement 3PG model."""

import os
from typing import Optional

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import Array

from trunx.gp3.model_inputs import State


def f_temperature(params, T_avg: Array) -> Array:
    """
    Calculate the temperature response function (fT) for forest growth.

    The function is defined as:
    f_T = ((T - Tmin)/(Topt - Tmin)) *
         ((Tmax - T)/(Tmax - Topt))^((Tmax - Topt)/(Topt - Tmin))

    Parameters
    ----------
    T : Array
        Current temperature (monthly mean temperature).
    Tmin : Array
        Minimum temperature for growth.
    Topt : Array
        Optimum temperature for growth.
    Tmax : Array
        Maximum temperature for growth.

    If T <= Tmin or T >= Tmax, fT is set to 0.

    Returns
    -------
    jax.Array
        Temperature response function value (fT).
    """
    eps = 1e-8
    out_of_range = (T_avg <= params.Tmin) | (T_avg >= params.Tmax)

    a = (T_avg - params.Tmin) / (params.Topt - params.Tmin + eps)
    b = (params.Tmax - T_avg) / (params.Tmax - params.Topt + eps)
    b = jnp.where(b > 0.0, b, 1.0)
    power = (params.Tmax - params.Topt) / (params.Topt - params.Tmin + eps)

    fT = a * (b**power)
    return jnp.clip(jnp.where(out_of_range, 0.0, fT), 0.0, 1.0)


def f_frost(params, frost_days: Array, days_in_month: Array) -> Array:
    """
    Calculate the frost response function (fF) for forest growth.

    The function is defined as:
        fF = 1 - kF * frost_days / days_in_month

    Parameters
    ----------
    frost_days : Array
        Number of frost days in a month.
    days_in_month : Array
        Number of days in the current month.

    Returns
    -------
    Array
        Frost response function value (fF).
    """
    return jnp.clip(1.0 - params.kF * frost_days / (days_in_month + 1e-8), 0.0, 1.0)


def f_vpd(VPD: Array, CoeffCond: Array) -> Array:
    """
    Calculate the vapor pressure deficit response function (fVPD).

    The function is defined as:
    f_VPD = exp(-CoeffCond * VPD) # CoeffCond = k_g (Landsberg and Warin 1997)

    Parameters
    ----------
    VPD : Array
        Vapor pressure deficit in kPa.
    CoeffCond : Array
        Threshold for the vapor pressure deficit that significantly affects growth.

    Returns
    -------
    Array
        Vapor pressure deficit response function value (fVPD).
    """
    f_vpd = jnp.exp(-CoeffCond * VPD)
    return f_vpd


def f_age(
    params,
    age_months: Array,
    # MaxAge: Array, nAge: Array, rAge: Array | None = None
) -> Array:
    """
    Age-related growth modifier.

    The function is defined as:

        f_age = 1 / (1 + (FAge / rAge) ** nAge)

    where:
        FAge = (stand age in years) / MaxAge

    Parameters
    ----------
    age_months : Array
        Stand age in months.
    MaxAge : Array
        Maximum stand age used to scale relative age (years).
    nAge : Array
        Shape parameter controlling the steepness of the age-related decline.
        Higher values produce a sharper decline.
    rAge : Array, optional
        Relative age at which f_age equals 0.5 (default = 0.95).

    Returns
    -------
    F_age: Array
        Age modifier ranging from 0 to 1.
    """
    rAge = jnp.where(params.rAge is None, jnp.asarray(0.95), params.rAge)

    age_years = jnp.clip(
        jnp.where(age_months == 1.0, age_months / 12.0, (age_months - 1.0) / 12.0), 0.0, None
    )

    FAge = age_years / (params.MaxAge + 1e-8)
    f_age = 1.0 / (1.0 + (FAge / (rAge + 1e-8)) ** params.nAge)
    return f_age


def f_soil_water(
    ASW: Array,
    site,
    params,
) -> Array:
    """
    Soil water stress function.

    The function is defined as:

        SWdef = 1 - ASW / ASW_max

        f_sw = 1 / [ 1 + (SWdef / SWconst)^SWpower ]

    Parameters
    ----------
    ASW : Array
        Available soil water.
    ASW_max : Array
        Maximum available soil water.
    SWconst : Array
        Scaling constant controlling stress onset.
    SWpower : Array
        Exponent controlling stress sensitivity.

    Returns
    -------
    f_sw : Array
        Soil water stress factor clipped to [0, 1].
    """
    soil_class = jnp.asarray(site.soil_class)
    swconst_param = params.SWconst if params.SWconst is not None else jnp.asarray(0.0)
    swpower_param = params.SWpower if params.SWpower is not None else jnp.asarray(0.0)

    pos_class = soil_class > 0
    neg_class = soil_class < 0

    SWconst = jnp.where(
        pos_class,
        0.8 - 0.10 * soil_class,
        jnp.where(neg_class, swconst_param, jnp.asarray(999.0)),
    )
    SWpower = jnp.where(
        pos_class,
        11.0 - 2.0 * soil_class,
        jnp.where(neg_class, swpower_param, swpower_param),
    )

    SWdef = 1.0 - ASW / (site.ASW_max + 1e-8)
    f_sw = 1 / (1 + (SWdef / (SWconst + 1e-8)) ** SWpower)
    f_sw = jnp.clip(f_sw, 0.0, 1.0)

    return f_sw


def f_nutrition(
    species,
    params,
) -> Array:
    """
    Soil nutrition modifier from the 3-PG model.

    f_N = 1 - (1 - fN0) * (1 - FR)**fNn
    with fNn = 0 -> f_N = 1

    Parameters
    ----------
    fertility : Array
        Soil fertility index (0-1).
    fN0 : Array
        Minimum modifier at zero fertility.
    fNn : Array
        Nutrition response exponent.

    Returns
    -------
    f_N : Array
        Nutrition modifier.
    """
    f_N = 1.0 - (1.0 - params.fN0) * (1.0 - species.FR) ** params.fNn
    f_N = jnp.where(params.fNn == 0.0, 1.0, f_N)

    return f_N


def compute_dbh(params, WS: Array, N: Array) -> Array:
    """
    Compute DBH from stand-level values.

    DBH = (WS / aWs) ** (1 / nWs)

    Parameters
    ----------
    WS : Array
        Stem biomass.
    aWs : Array
        Stem biomass allometric coefficient.
    nWs : Array
        Stem biomass exponent.

    Returns
    -------
    dbh : Array
        Diameter at breast height (cm).
    """
    wS_per_tree = (WS * 1000.0) / (N + 1e-8)  # kg/tree
    DBH = (wS_per_tree / (params.aWS + 1e-8)) ** (1.0 / (params.nWS + 1e-8))
    return DBH


def compute_light_interception(params, LAI: Array, canopy_cover: Array | None = None):
    """
    Compute the light interception.

    Compute the fraction of incoming radiation intercepted by the canopy
    using the Beer-Lambert law.

    Parameters
    ----------
    k : Array
        Canopy light extinction coefficient (dimensionless).
    LAI : Array
        Leaf area index (m² leaf m⁻² ground).
    canopy_cover : Array, optional
        Fractional canopy cover (0 < canopy_cover ≤ 1). Default is 1.

    Returns
    -------
    lightIntcptn : Array
        Fraction of incident radiation intercepted by the canopy (0-1).
    """
    if canopy_cover is None:
        canopy_cover = jnp.asarray(1.0)

    lightIntcptn = 1.0 - jnp.exp(-params.k * LAI / (canopy_cover + 1e-8))
    return lightIntcptn


def compute_lai(params, WF: Array, age_months: Array) -> tuple[Array, Array]:
    """
    Compute Leaf Area Index (LAI) from foliage biomass and stand age.

    LAI is calculated using an age-dependent specific leaf area (SLA)
    following the 3-PG formulation:

        SLA(t) = SLA0 + SLA1 * exp(-ln(2) * t / tSLA)
        LAI    = WF * SLA(t) * 0.1

    where stand age t is expressed in years.

    The factor 0.1 is a unit conversion:
        1 t ha⁻¹ = 1000 kg / 10,000 m² = 0.1 kg m⁻²

    Multiplying foliage biomass (t ha⁻¹) by 0.1 converts it to kg m⁻².

    Parameters
    ----------
    WF : Array
        Foliage biomass per unit ground area (t ha⁻¹).
    stand_age_months : Array
        Stand age (months).
    SLA0 : Array
        Minimum SLA at old age (m² kg⁻¹).
    SLA1 : Array
        Difference between maximum and minimum SLA (m² kg⁻¹).
    tSLA : Array
        Half-life for SLA decline (years).

    Returns
    -------
    LAI : Array
        Leaf Area Index (m² leaf m⁻² ground).
    """
    age_year = jnp.where(age_months == 1.0, age_months / 12.0, (age_months - 1.0) / 12.0)

    # SLA = SLA1 * jnp.exp(-jnp.log(2.0) * stand_age_years / tSLA) + SLA0

    SLA = jnp.where(
        params.tSLA != 0,
        params.SLA1
        + (params.SLA0 - params.SLA1) * jnp.exp(-jnp.log(2.0) * (age_year / params.tSLA) ** 2),
        jnp.ones_like(age_year) * params.SLA1,
    )

    LAI = WF * SLA * 0.1

    return LAI, SLA


def compute_litterfall_rate(
    age_months: Array, gammaF0: Array, gammaF1: Array, tgammaF: Array
) -> Array:
    """
    Compute foliage litterfall rate as a function of stand age.

    Parameters
    ----------
    age_months : Array
        Stand age (months).
    gammaF0 : Array
        Litterfall rate at young age.
    gammaF1 : Array
        Minimum litterfall rate at old age.
    tgammaF : Array
        Characteristic age controlling litterfall decline (months).

    Returns
    -------
    gammaF : Array
        Foliage litterfall rate.
    """
    gammaF = gammaF1 + (gammaF0 - gammaF1) * jnp.exp(
        -jnp.log(2.0) * (age_months / (tgammaF + 1e-8)) ** 2
    )
    return gammaF


def _solve_mortality_newton(
    stems_n_ha: Array,
    stem_biomass_stand: Array,
    mS: Array,
    wSx1000: Array,
    thinPower: Array,
    max_iterations: int = 5,
    accuracy: float = 1e-3,
) -> Array:
    """Solve self-thinning mortality using Newton-Raphson iteration."""
    n = stems_n_ha / 1000.0
    x1 = 1000.0 * mS * stem_biomass_stand / jnp.maximum(stems_n_ha, 1e-8)
    converged = jnp.zeros_like(n, dtype=bool)

    for _ in range(max_iterations):
        active = (~converged) & (n > 0.0)
        x2 = wSx1000 * jnp.power(jnp.maximum(n, 1e-8), 1.0 - thinPower)
        fN = x2 - x1 * n - (1.0 - mS) * stem_biomass_stand
        dfN = (1.0 - thinPower) * x2 / jnp.maximum(n, 1e-8) - x1
        safe_dfN = jnp.where(jnp.abs(dfN) < 1e-8, jnp.where(dfN >= 0.0, 1e-8, -1e-8), dfN)
        dN = jnp.where(active, -fN / safe_dfN, 0.0)
        n = n + dN
        n = jnp.where(n <= 0.0, 1e-8, n)
        converged = converged | (~active) | (jnp.abs(dN) <= accuracy)

    mort_n = stems_n_ha - 1000.0 * n
    return jnp.maximum(mort_n, 0.0)


def apply_self_thinning_with_mortality_factors(
    params,
    WS: Array,
    WF: Array,
    WR: Array,
    N: Array,
    dormant: Array,
) -> tuple[Array, Array, Array, Array, Array]:
    """Apply self-thinning with mortality factors for stem, foliage, and roots."""
    thinPower = jnp.where(params.thinPower is None, jnp.asarray(1.5), params.thinPower)

    biom_tree = (WS * 1000.0) / jnp.maximum(N, 1e-8)
    wSmax_per_tree = params.wSx1000 * jnp.power(1000.0 / jnp.maximum(N, 1e-8), thinPower)
    should_thin = (biom_tree > wSmax_per_tree) & ~dormant

    mort_count = _solve_mortality_newton(
        stems_n_ha=N,
        stem_biomass_stand=WS,
        mS=params.mS,
        wSx1000=params.wSx1000,
        thinPower=thinPower,
    )
    mort_count = jnp.where(should_thin, mort_count, 0.0)
    mort_count = jnp.clip(mort_count, 0.0, N)

    N_new = jnp.maximum(N - mort_count, 1.0)

    WF_loss = params.mF * mort_count * (WF / jnp.maximum(N, 1e-8))
    WR_loss = params.mR * mort_count * (WR / jnp.maximum(N, 1e-8))
    WS_loss = params.mS * mort_count * (WS / jnp.maximum(N, 1e-8))

    WF_new = jnp.maximum(WF - WF_loss, 0.0)
    WR_new = jnp.maximum(WR - WR_loss, 0.0)
    WS_new = jnp.maximum(WS - WS_loss, 0.0)

    return WS_new, WF_new, WR_new, N_new, mort_count


def apply_stress_mortality(
    params,
    age_months: Array,
    WS: Array,
    WF: Array,
    WR: Array,
    N: Array,
    dormant: Array,
) -> tuple[Array, Array, Array, Array, Array]:
    """Apply age-dependent stress mortality and update biomass pools."""
    eps = 1e-8
    age_years = jnp.maximum(age_months, 0.0) / 12.0

    gammaN = params.gammaN1 + (params.gammaN0 - params.gammaN1) * jnp.exp(
        -jnp.log(2.0) * jnp.power(age_years / (params.tgammaN + eps), params.ngammaN)
    )

    active = (~dormant) & (gammaN > 0.0)
    mort_stress_raw = gammaN * N / 12.0 / 100.0
    mort_stress = jnp.where(active, jnp.minimum(mort_stress_raw, N), 0.0)

    WF_loss = params.mF * mort_stress * (WF / jnp.maximum(N, eps))
    WR_loss = params.mR * mort_stress * (WR / jnp.maximum(N, eps))
    WS_loss = params.mS * mort_stress * (WS / jnp.maximum(N, eps))

    WF_new = jnp.maximum(WF - WF_loss, 0.0)
    WR_new = jnp.maximum(WR - WR_loss, 0.0)
    WS_new = jnp.maximum(WS - WS_loss, 0.0)
    N_new = jnp.maximum(N - mort_stress, 0.0)

    return WS_new, WF_new, WR_new, N_new, mort_stress


def apply_self_thinning(
    params,
    WS: Array,
    N: Array,
    max_mortality: Array | None = None,
) -> tuple[Array, Array]:
    """
    Apply self-thinning mortality based on size-density constraints.

    Parameters
    ----------
    WS : Array
        Stand stem biomass (t ha⁻¹).
    N : Array
        Stocking density (trees ha⁻¹).
    wSx : Array
        Maximum stem biomass parameter.
    max_mortality : Array, optional
        Maximum fractional mortality per timestep.

    Returns
    -------
    WS_new : Array
        Updated stem biomass after self-thinning (t ha⁻¹).
    N_new : Array
        Updated stocking density after self-thinning (trees ha⁻¹).
    """
    if max_mortality is None:
        max_mortality = jnp.asarray(0.05)

    # if params.thinPower is None:
    #     thinPower = jnp.asarray(1.5)
    thinPower = jnp.where(params.thinPower is None, jnp.asarray(1.5), params.thinPower)

    wS = 1000.0 * WS / (N + 1e-8)

    wSmax = params.wSx1000 * (1000.0 / (N + 1e-8)) ** thinPower

    rel_excess = (wS - wSmax) / (wSmax + 1e-8)

    mort_frac = jnp.clip(rel_excess, 0.0, max_mortality)

    N_new = jnp.clip(N * (1.0 - mort_frac), 1.0, None)
    WS_new = WS * (1.0 - 0.8 * mort_frac)

    return WS_new, N_new


def compute_canopy_cover(params, age: Array):
    """
    Calculate fractional canopy cover.

    Parameters
    ----------
    age_years : Array
        Stand age in years
    fullCanAge : Array
        Age at canopy closure (years)

    Returns
    -------
    canopy_cover : Array
        Fractional canopy cover (0-1)
    """
    age_years = age / 12.0
    condition = (params.fullCanAge > 0) & (age_years < params.fullCanAge)

    # Calculate cover for young stands
    young_cover = (age_years + 0.01) / params.fullCanAge

    # Use jnp.where to select between young and mature cover
    canopy_cover = jnp.where(condition, young_cover, 1.0)

    return canopy_cover


def is_dormant(month: Array, leafgrow: Array, leaffall: Array) -> Array:
    """
    Determine if current month is in dormant period.

    Parameters
    ----------
    month : Array
        Current month (1-12)
    leafgrow : Array
        Month when leaves start growing
    leaffall : Array
        Month when leaves start falling

    Returns
    -------
    dormant : Array
        True if dormant period, False otherwise
    """
    # Default to False (evergreen)
    dormant = jnp.zeros_like(leafgrow, dtype=bool)

    cond_north = jnp.logical_and(
        leafgrow > leaffall, jnp.logical_and(month >= leaffall, month <= leafgrow)
    )

    cond_south = jnp.logical_and(
        leafgrow < leaffall, jnp.logical_or(month < leafgrow, month >= leaffall)
    )

    # Combine and ensure boolean type
    dormant = jnp.logical_or(cond_north, cond_south)

    return dormant


def f_calpha(params, co2: Array):
    """
    CO2 modifier for photosynthesis (alpha).

    Parameters
    ----------
    co2 : Array
        Atmospheric CO2 concentration (ppm)
    fCalphax : Array
        CO2 modifier parameter for photosynthesis

    Returns
    -------
    f_calpha : Array
        CO2 modifier for photosynthesis
    """
    fCalphax = params.fCalpha700 / (2.0 - params.fCalpha700 + 1e-8)
    fcalpha = fCalphax * co2 / (350.0 * (fCalphax - 1.0) + co2)
    return fcalpha


def compute_allocation_fraction(species, params, phi_phys: Array, DBH: Array):
    """
    Compute all allocation fractions (roots, foliage, stem) for 3-PG model.

    eta_R = (r_x * r_n) / (r_n + (r_x - r_n) * m)

    Parameters
    ----------
    B : Array
        Tree size (DBH in cm)
    FR : Array
        Fertility rating (0-1)
    phi_phys : Array
        Physiological modifier (0-1)
    pFS2 : Array
        Foliage:stem ratio at reference size 2 cm
    pFS20 : Array
        Foliage:stem ratio at reference size 20 cm
    pRx : Array
        Maximum root allocation ratio
    pRn : Array
        Minimum root allocation ratio
    m0 : Array, optional
        Base fertility effect parameter (default 0.5)

    Returns
    -------
    eta_R : Array
        Fraction of NPP allocated to roots
    eta_F : Array
        Fraction of NPP allocated to foliage
    eta_S : Array
        Fraction of NPP allocated to stem
    pFS : Array
        Foliage:stem ratio (intermediate value)
    """
    m0 = jnp.where(params.m0 is None, jnp.asarray(0.5), params.m0)
    m = m0 + (1.0 - params.m0) * species.FR

    eta_R = (params.pRx * params.pRn) / (params.pRn + (params.pRx - params.pRn) * phi_phys * m)

    pfsPower = jnp.log(params.pFS20 / (params.pFS2 + 1e-8)) / jnp.log(10.0)
    pfsConst = params.pFS2 / 2.0**pfsPower
    pFS = pfsConst * (jnp.clip(DBH, 0.1, None) ** pfsPower)

    eta_S = (1.0 - eta_R) / (1.0 + pFS)
    eta_F = 1.0 - eta_R - eta_S

    return pFS, eta_F, eta_S, eta_R


def calculate_interception(
    params,
    prcp: Array,
    lai_total: Array,
    lai_per: Array,
) -> tuple[Array, Array]:
    """
    Calculate rainfall interception per species, sharing the stand's total canopy.

    Interception saturates with the *stand's* total LAI (species compete for the
    same rainfall), then splits proportionally by each species' canopy share.

    Parameters
    ----------
    prcp : Array
        Monthly precipitation (mm)
    lai_total : Array
        Stand-level total Leaf Area Index (summed across species)
    lai_per : Array
        Each species' fraction of the stand's total LAI
    MaxIntcptn : Array
        Maximum interception fraction
    LAImaxIntcptn : Array
        LAI at which interception reaches maximum

    Returns
    -------
    prcp_interc_fract : Array
        Interception fraction
    prcp_interc : Array
        Interception amount (mm)
    """
    condition = params.LAImaxIntcptn > 0
    adjusted_fract = (
        params.MaxIntcptn * jnp.minimum(1.0, lai_total / (params.LAImaxIntcptn + 1e-8)) * lai_per
    )
    prcp_interc_fract = jnp.where(condition, adjusted_fract, params.MaxIntcptn)

    prcp_interc = prcp * prcp_interc_fract

    return prcp_interc_fract, prcp_interc


def calculate_transpiration(
    params,
    solar_rad: Array,
    day_length: Array,
    VPD: Array,
    conduct_canopy: Array,
    days_in_month: Array,
    rhoAir: Array | None = None,
    lambda_v: Array | None = None,
    VPDconv: Array | None = None,
    e20: Array | None = None,
) -> Array:
    """
    Calculate transpiration using Penman-Monteith.

    Returns transpiration in mm/month.
    """
    if rhoAir is None:
        rhoAir = jnp.array(1.2)
    if lambda_v is None:
        lambda_v = jnp.array(2460000.0)
    if VPDconv is None:
        VPDconv = jnp.array(0.000622)
    if e20 is None:
        e20 = jnp.array(2.2)

    # Convert solar radiation from MJ/m²/day to W/m² for daytime
    solar_rad_w = solar_rad * 1e6 / day_length

    # Net radiation (W/m²)
    netRad = params.Qa + params.Qb * solar_rad_w

    # Deficit term (related to VPD)
    defTerm = rhoAir * lambda_v * VPDconv * VPD * params.BLcond

    # Divisor (combined conductance term)
    div = conduct_canopy * (1.0 + e20) + params.BLcond

    # Transpiration rate (mm/s)
    transp_rate = conduct_canopy * (e20 * netRad + defTerm) / div / lambda_v

    # Convert to mm/month
    transp_veg = transp_rate * day_length * days_in_month
    transp_veg = jnp.maximum(0.0, transp_veg)

    # Handle VPD=0 case (no transpiration)
    transp_veg = jnp.where(VPD == 0.0, 0.0, transp_veg)

    return transp_veg


def update_soil_water(
    site,
    ASW: Array,
    prcp: Array,
    transp_veg: Array,
    evapotra_soil: Array,
    prcp_interc: Array,
    Irrig: Array | None = None,
    water_runoff_polled: Array | None = None,
    poolFractn: Array | None = None,
) -> tuple[Array, Array, Array]:
    """Update soil water balance."""
    if Irrig is None:
        Irrig = jnp.array(0.0)
    if water_runoff_polled is None:
        water_runoff_polled = jnp.array(0.0)
    if poolFractn is None:
        poolFractn = jnp.array(0.0)

    monthly_irrig = (100.0 * Irrig) / 12.0
    ASW = ASW + prcp + monthly_irrig + water_runoff_polled
    total_demand = transp_veg + evapotra_soil + prcp_interc
    evapo_transp = jnp.minimum(ASW, total_demand)
    excessSW = jnp.maximum(ASW - evapo_transp - site.ASW_max, 0.0)
    ASW = ASW - evapo_transp - excessSW

    # water_runoff_polled_new = poolFractn * excessSW
    # prcp_runoff = (1.0 - poolFractn) * excessSW
    # irrig_supl = jnp.maximum(asw_min - ASW, 0.0)

    ASW = jnp.maximum(ASW, site.ASW_min)
    f_transp_scale = jnp.where(total_demand == 0, 1.0, evapo_transp / (total_demand + 1e-8))

    return ASW, f_transp_scale, evapo_transp


def scale_transpiration(
    transp_veg: Array,
    evapotra_soil: Array,
    prcp_interc: Array,
    evapo_transp: Array,
    f_transp_scale: Array,
) -> tuple[Array, Array]:
    """Scale transpiration and evaporation when water-limited."""
    transp_total = transp_veg + evapotra_soil
    scale_factor = (evapo_transp - prcp_interc) / (transp_total + 1e-8)
    condition = (transp_total > 0) & (f_transp_scale < 1)

    transp_veg_scaled = jnp.where(condition, scale_factor * transp_veg, transp_veg)

    evapotra_soil_scaled = jnp.where(condition, scale_factor * evapotra_soil, evapotra_soil)

    return transp_veg_scaled, evapotra_soil_scaled


def compute_asw(
    params,
    site,
    # Input state
    ASW: Array,
    # Climate inputs
    prcp: Array,
    solar_rad: Array,
    VPD: Array,
    day_length: Array,
    days_in_month: Array,
    # Parameters
    conduct_canopy: Array,
    lai: Array,
    lai_total: Array,
    lai_per: Array,
    # Optional soil evaporation
    evapotra_soil: Array | None = None,
) -> tuple[Array, Array]:
    """
    Complete soil water balance for a stand.

    Parameters
    ----------
    ASW : Array
        Current available soil water (mm), identical across species
    prcp : Array
        Monthly precipitation (mm)
    solar_rad : Array
        Solar radiation (MJ/m²/day)
    VPD : Array
        Vapor pressure deficit (kPa)
    day_length : Array
        Day length (seconds)
    days_in_month : Array
        Number of days in the month
    conduct_canopy : Array
        Canopy conductance (m/s)
    lai : Array
        Leaf Area Index
    lai_total : Array
        Stand-level total Leaf Area Index (summed across species)
    lai_per : Array
        Each species' fraction of the stand's total LAI
    evapotra_soil : Array, optional
        Soil evaporation (mm), default 0.0

    Returns
    -------
    tuple[Array, Array]
    ASW
        Updated available soil water
    GPP_scale_factor
        Factor to scale GPP
    """
    if evapotra_soil is None:
        evapotra_soil = jnp.array(0.0)

    prcp_interc_fract, prcp_interc = calculate_interception(
        params=params,
        prcp=prcp,
        lai_total=lai_total,
        lai_per=lai_per,
    )

    # Calculate transpiration
    transp_veg = calculate_transpiration(
        params=params,
        solar_rad=solar_rad,
        day_length=day_length,
        VPD=VPD,
        conduct_canopy=conduct_canopy,
        days_in_month=days_in_month,
    )

    # Update the shared, stand-level soil water balance. All species
    # start each step with the same ASW, so any one of them represents the pool.
    ASW_stand, f_transp_scale_stand, evapo_transp = update_soil_water(
        site=site,
        ASW=jnp.asarray(ASW).reshape(-1)[0],
        prcp=prcp,
        transp_veg=jnp.sum(transp_veg),
        evapotra_soil=jnp.sum(jnp.broadcast_to(evapotra_soil, transp_veg.shape)),
        prcp_interc=jnp.sum(prcp_interc),
    )
    ASW = jnp.full_like(transp_veg, ASW_stand)
    f_transp_scale = jnp.full_like(transp_veg, f_transp_scale_stand)

    return ASW, f_transp_scale


def calculate_day_length(latitude: Array, month: Array) -> Array:
    """
    Calculate day length in seconds for a given latitude and month.

    Parameters
    ----------
    latitude : Array
        Latitude in degrees
    month : Array
        Current month (1-12)

    Returns
    -------
    day_length : Array
        Day length in seconds
    """
    lat_rad = jnp.radians(latitude)
    SLAt = jnp.sin(lat_rad)
    cLat = jnp.cos(lat_rad)

    day_of_year_values = jnp.array([15, 45, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349])

    month_idx = jnp.clip(month - 1, 0, 11).astype(int)
    day_of_year = day_of_year_values[month_idx]

    sinDec = 0.4 * jnp.sin(0.0172 * (day_of_year - 80.0))
    cosH0 = -sinDec * SLAt / (cLat * jnp.sqrt(1.0 - sinDec**2))
    day_length = jnp.arccos(jnp.clip(cosH0, -1.0, 1.0)) / jnp.pi

    day_length = jnp.where(cosH0 > 1.0, 0.0, day_length)
    day_length = jnp.where(cosH0 < -1.0, 1.0, day_length)

    day_length = 86400.0 * day_length
    return day_length


def calculate_base_conductance(params, lai: Array) -> Array:
    """
    Calculate base canopy conductance (gC) as function of LAI.

    Parameters
    ----------
    lai : Array
        LAI
    MaxCond : Array
        Maximum canopy conductance (m/s)
    MinCond : Array
        Minimum canopy conductance (m/s)
    LAIgcx : Array
        LAI at which conductance reaches maximum

    Returns
    -------
    gC : Array
        Base canopy conductance (m/s)
    """
    gC = params.MaxCond
    condition = lai <= params.LAIgcx
    scaled_cond = params.MinCond + (params.MaxCond - params.MinCond) * lai / (params.LAIgcx + 1e-8)
    gC = jnp.where(condition, scaled_cond, gC)

    return gC


def f_temperature_gc(
    params,
    T_avg: Array,
    T_max: Array,
) -> Array:
    """
    Temperature response function for canopy conductance.

    Uses (T_avg + T_max)/2 instead of just T_avg.

    Parameters
    ----------
    T_avg : Array
        Average monthly temperature (°C)
    T_max : Array
        Maximum monthly temperature (°C)
    Tmin : Array
        Minimum temperature for growth (°C)
    Topt : Array
        Optimum temperature for growth (°C)
    Tmax : Array
        Maximum temperature for growth (°C)

    Returns
    -------
    f_tmp_gc : Array
        Temperature modifier for canopy conductance (0-1)
    """
    eps = 1e-8
    T_mid = (T_avg + T_max) / 2.0

    a = jnp.clip((T_mid - params.Tmin) / (params.Topt - params.Tmin + eps), 0.0, None)
    b = jnp.clip((params.Tmax - T_mid) / (params.Tmax - params.Topt + eps), 0.0, None)
    power = (params.Tmax - params.Topt) / (params.Topt - params.Tmin + eps)

    f_tmp_gc = a * (b**power)
    return jnp.clip(f_tmp_gc, 0.0, 1.0)


def f_cg(params, co2: Array) -> Array:
    """
    CO2 modifier for canopy conductance.

    Parameters
    ----------
    co2 : Array
        Atmospheric CO2 concentration (ppm)
    fCg0 : Array
        CO2 modifier parameter for conductance

    Returns
    -------
    f_cg : Array
        CO2 modifier for canopy conductance
    """
    fCg0 = params.fCg700 / (2.0 * params.fCg700 - 1.0 + 1e-8)
    f_cg = fCg0 / (1.0 + (fCg0 - 1.0) * co2 / 350.0)
    return jnp.clip(f_cg, 0.0, 1.0)


def f_exp_foliage(params, age_months: Array) -> Array:
    """
    Exponential foliage growth function.

    Parameters
    ----------
    x : Array
        Input array (typically time in months).
    gammaF1 : Array
        Final/asymptotic value (maximum foliage biomass).
    gammaF0 : Array
        Initial value (initial foliage biomass).
    tgammaF : Array
        Time to reach a certain growth stage (months).

    Returns
    -------
    out : Array
    """
    eps = 1e-8
    kg = 12.0 * jnp.log(1.0 + params.gammaF1 / (params.gammaF0 + eps)) / (params.tgammaF + eps)
    age_year = jnp.where(age_months == 1.0, age_months / 12.0, (age_months - 1.0) / 12.0)

    denom = params.gammaF0 + (params.gammaF1 - params.gammaF0) * jnp.exp(-kg * age_year)
    denom = jnp.where(jnp.abs(denom) < eps, jnp.ones_like(denom), denom)
    out = jnp.where(
        (params.tgammaF * params.gammaF1) < eps,
        params.gammaF1,
        params.gammaF1 * params.gammaF0 / denom,
    )

    return out


def f_exp_wood(params, age_months: Array) -> Array:
    """Exponential wood density function."""
    eps = 1e-8
    age_years = age_months / 12.0
    out = jnp.where(
        params.tRho > eps,
        params.rhoMax
        + (params.rhoMin - params.rhoMax)
        * jnp.exp(-jnp.log(2.0) * (age_years / (params.tRho + eps))),
        params.rhoMax,
    )

    return out
