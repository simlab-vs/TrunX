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
            raise ValueError("SWconst0 and SWpower0 must be provided when soil_class < 0")
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
    age_years : float
        Stand age in years
    fullCanAge : float
        Age at canopy closure (years)

    Returns
    -------
    canopy_cover : float
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


def f_cg(co2, fCg0) -> Array:
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
    f_cg = fCg0 / (1.0 + (fCg0 - 1.0) * co2 / 350.0)
    return f_cg


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
