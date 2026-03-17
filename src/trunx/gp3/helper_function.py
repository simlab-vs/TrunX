"""Helper functions to implement 3PG model."""

import os
from typing import Optional

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import Array

from trunx.gp3.model_inputs import State


def f_temperature(T_avg: Array, T_min: Array, T_opt: Array, T_max: Array) -> Array:
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

    If T < Tmin or T > Tmax, fT is set to 0.

    Returns
    -------
    jax.Array
        Temperature response function value (fT).
    """
    eps = 1e-8
    a = jnp.clip((T_avg - T_min) / (T_opt - T_min + eps), 0.0, None)
    b = jnp.clip((T_max - T_avg) / (T_max - T_opt + eps), 0.0, None)
    power = (T_max - T_opt) / (T_opt - T_min + eps)
    return jnp.clip(a * (b**power), 0.0, 1.0)


def f_frost(frost_days: Array, k_F: Array):
    """
    Calculate the frost response function (fF) for forest growth.

    The function is defined as:
    fF = 1 - k_F * frost_days/30

    Parameters
    ----------
    frost_days : int
        Number of frost days in a month.
    frost_threshold : Array
        Threshold for the number of frost days that significantly affects growth.

    Returns
    -------
    Array
        Frost response function value (fF).
    """
    return jnp.clip(1.0 - k_F * frost_days / 30.0, 0.0, 1.0)


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


def f_age(age_months: Array, MaxAge: Array, nAge: Array, rAge: Array | None = None) -> Array:
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
    if rAge is None:
        rAge = jnp.asarray(0.95)

    age_years = age_months / 12.0
    FAge = age_years / (MaxAge + 1e-8)
    f_age = 1.0 / (1.0 + (FAge / (rAge + 1e-8)) ** nAge)
    return f_age


def f_soil_water(
    ASW: Array, ASW_max: Array, SWconst: Array, SWpower: Array, soil_class: Array
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
    if soil_class > 0:
        SWconst = 0.8 - 0.10 * soil_class
        SWpower = 11.0 - 2.0 * soil_class
    elif soil_class < 0:
        if SWconst is None or SWpower is None:
            raise ValueError("SWconst and SWpower must be provided when soil_class < 0")
        SWconst = SWconst
        SWpower = SWpower
    else:
        SWconst = jnp.asarray(999.0)
        SWpower = SWpower if SWpower is not None else jnp.asarray(0.0)

    SWdef = 1.0 - ASW / (ASW_max + 1e-8)
    f_sw = 1 / (1 + (SWdef / (SWconst + 1e-8)) ** SWpower)
    f_sw = jnp.clip(f_sw, 0.0, 1.0)

    return f_sw


def f_nutrition(FR: Array, fN0: Array, fNn: Array) -> Array:
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
    f_N = 1.0 - (1.0 - fN0) * (1.0 - FR) ** fNn
    f_N = jnp.where(fNn == 0.0, 1.0, f_N)

    return f_N


def compute_dbh(WS: Array, N: Array, aWs: Array, nWs: Array) -> Array:
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

    DBH = (wS_per_tree / (aWs + 1e-8)) ** (1.0 / (nWs + 1e-8))

    return DBH


def compute_light_interception(k: Array, LAI: Array, canopy_cover: Array | None = None):
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

    lightIntcptn = 1.0 - jnp.exp(-k * LAI / (canopy_cover + 1e-8))
    return lightIntcptn


def compute_lai(
    WF: Array,
    stand_age_months: Array,
    SLA0: Array,
    SLA1: Array,
    tSLA: Array,
) -> tuple[Array, Array]:
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
    stand_age_years = stand_age_months / 12.0

    # SLA = SLA1 * jnp.exp(-jnp.log(2.0) * stand_age_years / tSLA) + SLA0

    SLA = jnp.where(
        tSLA != 0,
        SLA1 + (SLA0 - SLA1) * jnp.exp(-jnp.log(2.0) * (stand_age_years / tSLA) ** 2),
        jnp.ones_like(stand_age_years) * SLA1,
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


def apply_self_thinning(
    WS: Array,
    N: Array,
    wSx: Array,
    max_mortality: Array | None = None,
    thinPower: Array | None = None,
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

    if thinPower is None:
        thinPower = jnp.asarray(1.5)

    wS = 1000.0 * WS / (N + 1e-8)

    wSmax = wSx * (1000.0 / (N + 1e-8)) ** thinPower

    rel_excess = (wS - wSmax) / (wSmax + 1e-8)

    mort_frac = jnp.clip(rel_excess, 0.0, max_mortality)

    N_new = jnp.clip(N * (1.0 - mort_frac), 1.0, None)
    WS_new = WS * (1.0 - 0.8 * mort_frac)

    return WS_new, N_new


def compute_canopy_cover(age: Array, fullCanAge: Array):
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
    condition = (fullCanAge > 0) & (age_years < fullCanAge)

    # Calculate cover for young stands
    young_cover = (age_years + 0.01) / fullCanAge

    # Use jnp.where to select between young and mature cover
    canopy_cover = jnp.where(condition, young_cover, 1.0)

    return canopy_cover


def is_dormant(month, leafgrow, leaffall):
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
    dormant = jnp.array(False)
    cond_north = (leafgrow > leaffall) & (month >= leaffall) & (month <= leafgrow)

    cond_south = (leafgrow < leaffall) & ((month < leafgrow) | (month >= leaffall))

    dormant = cond_north | cond_south

    return dormant


def f_calpha(co2: Array, fCalpha700: Array):
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
    fCalphax = fCalpha700 / (2.0 - fCalpha700 + 1e-8)
    fcalpha = fCalphax * co2 / (350.0 * (fCalphax - 1.0) + co2)
    return fcalpha


def compute_allocation_fraction(
    FR: Array,
    pRx: Array,
    pRn: Array,
    pFS2: Array,
    pFS20: Array,
    phi_phys: Array,
    DBH: Array,
    m0: Array | None = None,
):
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
    if m0 is None:
        m0 = jnp.asarray(0.5)

    m = m0 + (1.0 - m0) * FR

    eta_R = (pRx * pRn) / (pRn + (pRx - pRn) * phi_phys * m)

    pfsPower = jnp.log(pFS20 / (pFS2 + 1e-8)) / jnp.log(10.0)
    pfsConst = pFS2 / 2.0**pfsPower
    pFS = pfsConst * (jnp.clip(DBH, 0.1, None) ** pfsPower)

    eta_S = (1.0 - eta_R) / (1.0 + pFS)
    eta_F = 1.0 - eta_R - eta_S

    return pFS, eta_F, eta_S, eta_R


def calculate_interception(
    prcp: Array, lai: Array, MaxIntcptn: Array, LAImaxIntcptn: Array
) -> tuple[Array, Array]:
    """
    Calculate rainfall interception for a single species (JAX-compatible).

    From Fortran:
        prcp_interc_fract = MaxIntcptn
        if (LAImaxIntcptn > 0) then
            prcp_interc_fract = MaxIntcptn * min(1.0, lai / LAImaxIntcptn)
        end if
        prcp_interc = prcp * prcp_interc_fract

    Parameters
    ----------
    prcp : Array
        Monthly precipitation (mm)
    lai : Array
        Leaf Area Index
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
    condition = LAImaxIntcptn > 0
    adjusted_fract = MaxIntcptn * jnp.minimum(1.0, lai / (LAImaxIntcptn + 1e-8))
    prcp_interc_fract = jnp.where(condition, adjusted_fract, MaxIntcptn)

    prcp_interc = prcp * prcp_interc_fract

    return prcp_interc_fract, prcp_interc


def calculate_transpiration(
    solar_rad: Array,
    day_length: Array,
    VPD: Array,
    BLcond: Array,
    conduct_canopy: Array,
    days_in_month: Array,
    Qa: Array,
    Qb: Array,
    rhoAir: Array | None,
    lambda_v: Array | None,
    VPDconv: Array | None,
    e20: Array | None,
) -> tuple[Array, dict]:
    """
    Calculate transpiration using Penman-Monteith (JAX-compatible).

    Returns transpiration in mm/month and intermediate values.
    """
    if rhoAir is None:
        rhoAir = jnp.array(1.2)
    if lambda_v is None:
        lambda_v = jnp.array(2460000.0)
    if VPDconv is None:
        VPDconv = jnp.array(0.000622)
    if e20 is None:
        e20 = jnp.array(0.66)

    # Convert solar radiation from MJ/m²/day to W/m² for daytime
    solar_rad_w = solar_rad * 1e6 / day_length

    # Net radiation (W/m²)
    netRad = Qa + Qb * solar_rad_w

    # Deficit term (related to VPD)
    defTerm = rhoAir * lambda_v * VPDconv * VPD * BLcond

    # Divisor (combined conductance term)
    div = conduct_canopy * (1.0 + e20) + BLcond

    # Transpiration rate (mm/s)
    transp_rate = conduct_canopy * (e20 * netRad + defTerm) / div / lambda_v

    # Convert to mm/month
    transp_veg = transp_rate * day_length * days_in_month
    transp_veg = jnp.maximum(0.0, transp_veg)

    # Handle VPD=0 case (no transpiration)
    transp_veg = jnp.where(VPD == 0.0, 0.0, transp_veg)

    intermediates = {
        "solar_rad_w": solar_rad_w,
        "netRad": netRad,
        "defTerm": defTerm,
        "div": div,
        "transp_rate": transp_rate,
    }

    return transp_veg, intermediates


def update_soil_water(
    ASW: Array,
    prcp: Array,
    transp_veg: Array,
    evapotra_soil: Array,
    prcp_interc: Array,
    asw_max: Array,
    asw_min: Array,
    Irrig: Array | None,
    water_runoff_polled: Array | None,
    poolFractn: Array | None,
) -> dict[str, Array]:
    """
    Update soil water balance (JAX-compatible).

    From Fortran:
        ASW = ASW + prcp + (100 * Irrig / 12) + water_runoff_polled
        total_demand = transp_veg + evapotra_soil + prcp_interc
        evapo_transp = min(ASW, total_demand)
        excessSW = max(ASW - evapo_transp - asw_max, 0)
        ASW = ASW - evapo_transp - excessSW
        water_runoff_polled_new = poolFractn * excessSW
        prcp_runoff = (1 - poolFractn) * excessSW
        irrig_supl = max(asw_min - ASW, 0)
        ASW = max(ASW, asw_min)
        f_transp_scale = 1 if total_demand == 0 else evapo_transp / total_demand
    """
    if Irrig is None:
        Irrig = jnp.array(0.0)
    if water_runoff_polled is None:
        water_runoff_polled = jnp.array(0.0)
    if poolFractn is None:
        poolFractn = jnp.array(0.0)

    # Convert annual irrigation to monthly
    monthly_irrig = (100.0 * Irrig) / 12.0

    # Add water inputs
    ASW = ASW + prcp + monthly_irrig + water_runoff_polled

    # Total water demand
    total_demand = transp_veg + evapotra_soil + prcp_interc

    # Actual ET (can't exceed available water)
    evapo_transp = jnp.minimum(ASW, total_demand)

    # Excess above field capacity
    excessSW = jnp.maximum(ASW - evapo_transp - asw_max, 0.0)

    # Update ASW after ET and excess
    ASW = ASW - evapo_transp - excessSW

    # Split excess into runoff pool and immediate runoff
    water_runoff_polled_new = poolFractn * excessSW
    prcp_runoff = (1.0 - poolFractn) * excessSW

    # Check wilting point
    irrig_supl = jnp.maximum(asw_min - ASW, 0.0)
    ASW = jnp.maximum(ASW, asw_min)

    # Transpiration scaling factor
    f_transp_scale = jnp.where(total_demand == 0, 1.0, evapo_transp / total_demand)

    return {
        "ASW_final": ASW,
        "water_runoff_polled_new": water_runoff_polled_new,
        "prcp_runoff": prcp_runoff,
        "irrig_supl": irrig_supl,
        "f_transp_scale": f_transp_scale,
        "evapo_transp": evapo_transp,
        "excessSW": excessSW,
        "total_demand": total_demand,
        "monthly_irrig": monthly_irrig,
    }


def scale_transpiration(
    transp_veg: Array,
    evapotra_soil: Array,
    prcp_interc: Array,
    evapo_transp: Array,
    f_transp_scale: Array,
) -> tuple[Array, Array]:
    """
    Scale transpiration and evaporation when water-limited (JAX-compatible).

    From Fortran:
        if (transp_total > 0 and f_transp_scale < 1) then
            transp_veg = (evapo_transp - prcp_interc) / transp_total * transp_veg
            evapotra_soil = (evapo_transp - prcp_interc) / transp_total * evapotra_soil
        end if
    """
    transp_total = transp_veg + evapotra_soil
    scale_factor = (evapo_transp - prcp_interc) / (transp_total + 1e-8)
    condition = (transp_total > 0) & (f_transp_scale < 1)

    transp_veg_scaled = jnp.where(condition, scale_factor * transp_veg, transp_veg)

    evapotra_soil_scaled = jnp.where(condition, scale_factor * evapotra_soil, evapotra_soil)

    return transp_veg_scaled, evapotra_soil_scaled


def compute_asw(
    # Input state
    ASW: Array,
    water_runoff_polled: Array,
    # Climate inputs
    prcp: Array,
    solar_rad: Array,
    VPD: Array,
    day_length: Array,
    days_in_month: Array,
    # Parameters
    Qa: Array,
    Qb: Array,
    BLcond: Array,
    conduct_canopy: Array,
    MaxIntcptn: Array,
    lai: Array,
    LAImaxIntcptn: Array,
    asw_min: Array,
    asw_max: Array,
    # Optional soil evaporation
    evapotra_soil: Array | None = None,
    # Physical constants (with defaults)
    Irrig: Array | None = None,
    poolFractn: Array | None = None,
    rhoAir: Array | None = None,
    lambda_v: Array | None = None,
    VPDconv: Array | None = None,
    e20: Array | None = None,
) -> dict[str, Array]:
    """
    Complete soil water balance for a single species following Fortran 3-PG code.

    This function combines:
    1. Rainfall interception calculation
    2. Transpiration calculation (Penman-Monteith)
    3. Soil water balance update
    4. Transpiration scaling

    All operations are JAX-compatible for use in jit-compiled functions.

    Parameters
    ----------
    ASW : Array
        Current available soil water (mm)
    water_runoff_polled : Array
        Water from previous month's runoff pool (mm)
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
    Irrig : Array
        Annual irrigation (mm/year)
    poolFractn : Array
        Fraction of excess water that goes to runoff pool (0-1)
    Qa, Qb : Array
        Net radiation parameters
    BLcond : Array
        Boundary layer conductance (m/s)
    conduct_canopy : Array
        Canopy conductance (m/s)
    MaxIntcptn : Array
        Maximum interception fraction
    lai : Array
        Leaf Area Index
    LAImaxIntcptn : Array
        LAI at which interception reaches maximum
    asw_min : Array
        Minimum available soil water (wilting point) (mm)
    asw_max : Array
        Maximum available soil water (field capacity) (mm)
    evapotra_soil : Array, optional
        Soil evaporation (mm), default 0.0
    rhoAir : Array, optional
        Air density (kg/m³)
    lambda_v : Array, optional
        Latent heat of vaporization (J/kg)
    VPDconv : Array, optional
        VPD conversion factor (kPa⁻¹)
    e20 : Array, optional
        Constant for Penman-Monteith

    Returns
    -------
    dict[str, Array]
        Dictionary containing all calculated variables:
        - ASW_final: Updated available soil water
        - water_runoff_polled_new: Updated runoff pool for next month
        - prcp_runoff: Immediate runoff
        - irrig_supl: Irrigation supplement needed
        - f_transp_scale: Transpiration scaling factor
        - transp_veg: Calculated transpiration
        - transp_veg_scaled: Scaled transpiration
        - evapotra_soil_scaled: Scaled soil evaporation
        - prcp_interc: Rainfall interception
        - prcp_interc_fract: Interception fraction
        - evapo_transp: Actual evapotranspiration
        - excessSW: Excess water above field capacity
        - total_demand: Total water demand
        - monthly_irrig: Monthly irrigation amount
        - GPP_scale_factor: Factor to scale GPP (same as f_transp_scale)
    """
    if Irrig is None:
        Irrig = jnp.array(0.0)
    if water_runoff_polled is None:
        water_runoff_polled = jnp.array(0.0)
    if poolFractn is None:
        poolFractn = jnp.array(0.0)
    if evapotra_soil is None:
        evapotra_soil = jnp.array(0.0)
    if rhoAir is None:
        rhoAir = jnp.array(1.2)
    if lambda_v is None:
        lambda_v = jnp.array(2460000.0)
    if VPDconv is None:
        VPDconv = jnp.array(0.000622)
    if e20 is None:
        e20 = jnp.array(0.66)

    # Step 1: Calculate rainfall interception
    prcp_interc_fract, prcp_interc = calculate_interception(
        prcp=prcp, lai=lai, MaxIntcptn=MaxIntcptn, LAImaxIntcptn=LAImaxIntcptn
    )

    # Step 2: Calculate transpiration
    transp_veg, trans_intermediates = calculate_transpiration(
        solar_rad=solar_rad,
        day_length=day_length,
        VPD=VPD,
        BLcond=BLcond,
        conduct_canopy=conduct_canopy,
        days_in_month=days_in_month,
        Qa=Qa,
        Qb=Qb,
        rhoAir=rhoAir,
        lambda_v=lambda_v,
        VPDconv=VPDconv,
        e20=e20,
    )

    # Step 3: Update soil water balance
    water_results = update_soil_water(
        ASW=ASW,
        prcp=prcp,
        Irrig=Irrig,
        water_runoff_polled=water_runoff_polled,
        transp_veg=transp_veg,
        evapotra_soil=evapotra_soil,
        prcp_interc=prcp_interc,
        asw_max=asw_max,
        asw_min=asw_min,
        poolFractn=poolFractn,
    )

    # Step 4: Scale transpiration if needed
    transp_veg_scaled, evapotra_soil_scaled = scale_transpiration(
        transp_veg=transp_veg,
        evapotra_soil=evapotra_soil,
        prcp_interc=prcp_interc,
        evapo_transp=water_results["evapo_transp"],
        f_transp_scale=water_results["f_transp_scale"],
    )

    # Combine all results
    results = {
        # Soil water states
        "ASW_final": water_results["ASW_final"],
        "water_runoff_polled_new": water_results["water_runoff_polled_new"],
        "prcp_runoff": water_results["prcp_runoff"],
        "irrig_supl": water_results["irrig_supl"],
        # Scaling factors
        "f_transp_scale": water_results["f_transp_scale"],
        "GPP_scale_factor": water_results["f_transp_scale"],
        # Water balance components
        "transp_veg": transp_veg,
        "transp_veg_scaled": transp_veg_scaled,
        "evapotra_soil_scaled": evapotra_soil_scaled,
        "prcp_interc": prcp_interc,
        "prcp_interc_fract": prcp_interc_fract,
        "evapo_transp": water_results["evapo_transp"],
        "excessSW": water_results["excessSW"],
        "total_demand": water_results["total_demand"],
        "monthly_irrig": water_results["monthly_irrig"],
    }

    # Add transpiration intermediates for debugging
    results.update(trans_intermediates)

    return results


def initialize_water_variables(
    Irrig: Array | None = None,
    water_runoff_polled: Array | None = None,
    poolFractn: Array | None = None,
) -> dict[str, Array]:
    """
    Initialize water-related variables as in Fortran.

    From Fortran:
        Irrig = 0.d0
        water_runoff_polled = 0.d0
        poolFractn = 0.d0
        poolFractn = max(0.d0, min(1.d0, poolFractn))
    """
    if Irrig is None:
        Irrig = jnp.array(0.0)
    if water_runoff_polled is None:
        water_runoff_polled = jnp.array(0.0)
    if poolFractn is None:
        poolFractn = jnp.array(0.0)

    # Constrain poolFractn to [0, 1]
    poolFractn = jnp.clip(poolFractn, 0.0, 1.0)

    return {
        "Irrig": jnp.array(Irrig),
        "water_runoff_polled": jnp.array(0.0),
        "poolFractn": jnp.array(poolFractn),
    }


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
    # Day of year for middle of month (approximate)
    day_of_year_values = jnp.array([15, 45, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349])
    month_idx = jnp.clip(month - 1, 0, 11).astype(int)
    day_of_year = day_of_year_values[month_idx]

    # Solar declination (radians)
    decl = 0.4093 * jnp.sin(2 * jnp.pi * (284 + day_of_year) / 365)

    # Latitude in radians
    lat_rad = jnp.radians(latitude)

    # Hour angle at sunset (radians)
    cos_omega = -jnp.tan(lat_rad) * jnp.tan(decl)
    omega = jnp.arccos(jnp.clip(cos_omega, -1.0, 1.0))

    # Day length in hours, convert to seconds
    day_length_hours = (24.0 / jnp.pi) * omega
    day_length_seconds = day_length_hours * 3600.0

    return day_length_seconds


def calculate_base_conductance(lai: Array, MaxCond: Array, MinCond: Array, LAIgcx: Array) -> Array:
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
    # Default to MaxCond
    gC = MaxCond

    # For LAI below LAIgcx, scale between MinCond and MaxCond
    condition = lai <= LAIgcx
    scaled_cond = MinCond + (MaxCond - MinCond) * lai / (LAIgcx + 1e-8)
    gC = jnp.where(condition, scaled_cond, gC)

    return gC


def f_temperature_gc(
    T_avg: Array, T_max: Array, T_min: Array, T_opt: Array, T_max_val: Array
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
    T_min : Array
        Minimum temperature for growth (°C)
    T_opt : Array
        Optimum temperature for growth (°C)
    T_max_val : Array
        Maximum temperature for growth (°C)

    Returns
    -------
    f_tmp_gc : Array
        Temperature modifier for canopy conductance (0-1)
    """
    eps = 1e-8
    T_mid = (T_avg + T_max) / 2.0

    invalid = (T_mid <= T_min) | (T_mid >= T_max_val)

    a = (T_mid - T_min) / (T_opt - T_min + eps)
    b = (T_max_val - T_mid) / (T_max_val - T_opt + eps)
    power = (T_max_val - T_opt) / (T_opt - T_min + eps)

    f_tmp_gc = jnp.where(invalid, 0.0, a * (b**power))
    return jnp.clip(f_tmp_gc, 0.0, 1.0)


def f_cg(co2: Array, fCg700: Array) -> Array:
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
    fCg0 = fCg700 / (2.0 * fCg700 - 1.0 + 1e-8)
    f_cg = fCg0 / (1.0 + (fCg0 - 1.0) * co2 / 350.0)
    return jnp.clip(f_cg, 0.0, 1.0)
