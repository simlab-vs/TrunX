"""Helper functions to implement 3PG model."""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import Array
from model_inputs import State


def f_temperature(T_avg, T_min, T_opt, T_max):
    """
    Calculate the temperature response function (fT) for forest growth.

    The function is defined as:
    f_T = ((T - Tmin)/(Topt - Tmin)) *
         ((Tmax - T)/(Tmax - Topt))^((Tmax - Topt)/(Topt - Tmin))

    Parameters
    ----------
    T : float
        Current temperature (monthly mean temperature).
    Tmin : float
        Minimum temperature for growth.
    Topt : float
        Optimum temperature for growth.
    Tmax : float
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


def f_frost(frost_days, k_F):
    """
    Calculate the frost response function (fF) for forest growth.

    The function is defined as:
    fF = 1 - k_F * frost_days/30

    Parameters
    ----------
    frost_days : int
        Number of frost days in a month.
    frost_threshold : float
        Threshold for the number of frost days that significantly affects growth.

    Returns
    -------
    float
        Frost response function value (fF).
    """
    return jnp.clip(1.0 - k_F * frost_days / 30.0, 0.0, 1.0)


def f_vpd(VPD, CoeffCond):
    """
    Calculate the vapor pressure deficit response function (fVPD).

    The function is defined as:
    f_VPD = exp(-CoeffCond * VPD) # CoeffCond = k_g (Landsberg and Warin 1997)

    Parameters
    ----------
    VPD : float
        Vapor pressure deficit in kPa.
    CoeffCond : float
        Threshold for the vapor pressure deficit that significantly affects growth.

    Returns
    -------
    float
        Vapor pressure deficit response function value (fVPD).
    """
    f_vpd = jnp.exp(-CoeffCond * VPD)
    return f_vpd


def f_age(age_months, MaxAge, nAge, rAge=0.95):
    """
    Age-related growth modifier.

    The function is defined as:

        f_age = 1 / (1 + (FAge / rAge) ** nAge)

    where:
        FAge = (stand age in years) / MaxAge

    Parameters
    ----------
    age_months : float
        Stand age in months.
    MaxAge : float
        Maximum stand age used to scale relative age (years).
    nAge : float
        Shape parameter controlling the steepness of the age-related decline.
        Higher values produce a sharper decline.
    rAge : float, optional
        Relative age at which f_age equals 0.5 (default = 0.95).

    Returns
    -------
    F_age: float
        Age modifier ranging from 0 to 1.
    """
    age_years = age_months / 12.0
    FAge = age_years / (MaxAge + 1e-8)
    f_age = 1.0 / (1.0 + (FAge / (rAge + 1e-8)) ** nAge)
    return f_age


def f_soil_water(ASW, ASW_max, SWconst, SWpower):
    """
    Soil water stress function.

    The function is defined as:

        SWdef = 1 - ASW / ASW_max

        f_sw = 1 / [ 1 + (SWdef / SWconst)^SWpower ]

    Parameters
    ----------
    ASW : float
        Available soil water.
    ASW_max : float
        Maximum available soil water.
    SWconst : float
        Scaling constant controlling stress onset.
    SWpower : float
        Exponent controlling stress sensitivity.

    Returns
    -------
    f_sw : float
        Soil water stress factor clipped to [0, 1].
    """
    SWdef = 1.0 - ASW / (ASW_max + 1e-8)
    f_sw = 1 / (1 + (SWdef / (SWconst + 1e-8)) ** SWpower)
    f_sw = jnp.clip(f_sw, 0.0, 1.0)
    return f_sw


def f_nutrition(FR, fN0, fNn):
    """
    Soil nutrition modifier from the 3-PG model.

    f_N = 1 - (1 - fN0) * (1 - FR)**fNn
    with fNn = 0 -> f_N = 1

    Parameters
    ----------
    fertility : float
        Soil fertility index (0-1).
    fN0 : float
        Minimum modifier at zero fertility.
    fNn : float
        Nutrition response exponent.

    Returns
    -------
    f_N : float or ndarray
        Nutrition modifier.
    """
    f_N = 1.0 - (1.0 - fN0) * (1.0 - FR) ** fNn
    f_N = jnp.where(fNn == 0.0, 1.0, f_N)

    return f_N


def compute_dbh(WS: float, aWs: float, nWs: float) -> float:
    """
    Compute DBH from stem biomass per tree (3-PG).

    DBH = (WS / aWs) ** (1 / nWs)

    Parameters
    ----------
    WS : float
        Stem biomass per tree.
    aWs : float
        Stem biomass allometric coefficient.
    nWs : float
        Stem biomass exponent.

    Returns
    -------
    dbh : float
        Diameter at breast height (cm).
    """
    dbh = (WS / (aWs + 1e-8)) ** (1.0 / (nWs + 1e-8))
    return dbh


def compute_light_interception(k, LAI, canopy_cover=1.0):
    """
    Compute the light interception.

    Compute the fraction of incoming radiation intercepted by the canopy
    using the Beer-Lambert law.

    Parameters
    ----------
    k : float
        Canopy light extinction coefficient (dimensionless).
    LAI : float
        Leaf area index (m² leaf m⁻² ground).
    canopy_cover : float, optional
        Fractional canopy cover (0 < canopy_cover ≤ 1). Default is 1.

    Returns
    -------
    lightIntcptn : float
        Fraction of incident radiation intercepted by the canopy (0-1).
    """
    lightIntcptn = 1.0 - jnp.exp(-k * LAI / canopy_cover)
    return lightIntcptn


def compute_lai(
    WF: float,
    stand_age_months: float,
    SLA0: float,
    SLA1: float,
    tSLA: float,
):
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
    WF : float
        Foliage biomass per unit ground area (t ha⁻¹).
    stand_age_months : float
        Stand age (months).
    SLA0 : float
        Minimum SLA at old age (m² kg⁻¹).
    SLA1 : float
        Difference between maximum and minimum SLA (m² kg⁻¹).
    tSLA : float
        Half-life for SLA decline (years).

    Returns
    -------
    LAI : float
        Leaf Area Index (m² leaf m⁻² ground).
    """
    stand_age_years = stand_age_months / 12.0

    SLA = SLA1 * jnp.exp(-jnp.log(2.0) * stand_age_years / tSLA) + SLA0

    LAI = WF * SLA * 0.1

    return LAI


def compute_root_allocation(fN: float, phi_phys: float, r_x: float, r_n: float) -> float:
    """
    Compute the fraction of net production allocated to roots.

    Root allocation is controlled by nutrient availability and
    physiological limitation following the 3-PG formulation:

        m     = fN * phi_phys
        eta_R = (r_x * r_n) / (r_n + (r_x - r_n) * m)

    Parameters
    ----------
    fN : float
        Soil nutrition modifier (0-1).
    phi_phys : float
        Physiological modifier (0-1).
    r_x : float
        Maximum root allocation ratio.
    r_n : float
        Minimum root allocation ratio.

    Returns
    -------
    eta_R : float
        Fraction of net production allocated to roots.
    """
    m = fN * phi_phys

    eta_R = (r_x * r_n) / (r_n + (r_x - r_n) * m + 1e-8)

    return eta_R


def compute_allocation_fractions(B: float, eta_R: float, pFS2: float, pFS20: float):
    """
    Compute foliage and stem allocation fractions.

    Allocation depends on tree size through the foliage:stem ratio,
    following the standard 3-PG power-law formulation.

    Parameters
    ----------
    B : float
        Tree size variable (typically DBH or biomass proxy).
    eta_R : float
        Fraction of production allocated to roots.
    pFS2 : float
        Foliage:stem ratio at reference size 2.
    pFS20 : float
        Foliage:stem ratio at reference size 20.

    Returns
    -------
    eta_F : float
        Fraction of production allocated to foliage.
    eta_S : float
        Fraction of production allocated to stem.
    """
    np_alloc = jnp.log(pFS20 / (pFS2 + 1e-8)) / jnp.log(10.0)
    ap_alloc = pFS2 * (2.0 ** (-np_alloc))
    pFS = ap_alloc * jnp.clip(B, 0.1, None) ** np_alloc
    eta_F = (pFS / (1.0 + pFS)) * (1.0 - eta_R)
    eta_S = (1.0 / (1.0 + pFS)) * (1.0 - eta_R)

    return eta_F, eta_S


def compute_litterfall_rate(age_months: float, gammaF0: float, gammaF1: float, tgammaF: float):
    """
    Compute foliage litterfall rate as a function of stand age.

    Parameters
    ----------
    age_months : float
        Stand age (months).
    gammaF0 : float
        Litterfall rate at young age.
    gammaF1 : float
        Minimum litterfall rate at old age.
    tgammaF : float
        Characteristic age controlling litterfall decline (months).

    Returns
    -------
    gammaF : float
        Foliage litterfall rate.
    """
    gammaF = gammaF1 + (gammaF0 - gammaF1) * jnp.exp(
        -jnp.log(2.0) * (age_months / (tgammaF + 1e-8)) ** 2
    )
    return gammaF


def apply_self_thinning(WS: Array, N: float, wSx: float, max_mortality: float = 0.05):
    """
    Apply self-thinning mortality based on size-density constraints.

    Parameters
    ----------
    WS : float
        Stand stem biomass (t ha⁻¹).
    N : float
        Stocking density (trees ha⁻¹).
    wSx : float
        Maximum stem biomass parameter.
    max_mortality : float, optional
        Maximum fractional mortality per timestep.

    Returns
    -------
    WS_new : float
        Updated stem biomass after self-thinning (t ha⁻¹).
    N_new : float
        Updated stocking density after self-thinning (trees ha⁻¹).
    """
    wS = 1000.0 * WS / (N + 1e-8)

    wSmax = wSx * (1000.0 / (N + 1e-8)) ** 1.5

    rel_excess = (wS - wSmax) / (wSmax + 1e-8)

    mort_frac = jnp.clip(rel_excess, 0.0, max_mortality)

    N_new = jnp.clip(N * (1.0 - mort_frac), 1.0, None)
    WS_new = WS * (1.0 - 0.8 * mort_frac)

    return WS_new, N_new


def model_step(state, climate_month, params, site, species):
    """Compute one model step."""
    T_avg, VPD, precip, solar_rad, frost_days, n_days = climate_month
    WF, WR, WS, N, ASW, age_months = state

    # Leaf area index
    LAI = jnp.clip(compute_lai(WF, age_months, params.SLA0, params.SLA1, params.tSLA), 0.0, 15.0)

    # Light interception (Beer's Law)
    lightIntcptn = compute_light_interception(params.k, LAI)
    APAR = solar_rad * n_days * lightIntcptn

    # Growth modifiers
    fT = f_temperature(T_avg, params.Tmin, params.Topt, params.Tmax)
    fF = f_frost(frost_days, params.kF)
    fN = f_nutrition(species.FR, params.fN0, params.fNn)
    fD = f_vpd(VPD, params.CoeffCond)
    fSW = f_soil_water(ASW, site.ASW_max, params.SWconst, params.SWpower)
    fA = f_age(age_months, params.MaxAge, params.nAge, params.rAge)

    phi = fA * jnp.minimum(fD, fSW)
    alpha_c = params.alphaCx * fT * fF * fN * phi

    # Primary production
    GPP = alpha_c * APAR
    NPP = params.Y * GPP

    # Allocation
    eta_R = compute_root_allocation(fN, phi, params.pRx, params.pRn)

    B = compute_dbh(WS, params.aWS, params.nWS)
    eta_F, eta_S = compute_allocation_fractions(WS, eta_R, params.pFS2, params.pFS20)

    # Turnover
    gammaF = compute_litterfall_rate(age_months, params.gammaF0, params.gammaF1, params.tgammaF)

    # Biomass updates
    WF_new = jnp.clip(WF + eta_F * NPP - gammaF * WF, 0.0, None)
    WR_new = jnp.clip(WR + eta_R * NPP - params.gammaR * WR, 0.0, None)
    WS_new = jnp.clip(WS + eta_S * NPP, 0.0, None)

    # Self-thinning
    WS_new, N_new = apply_self_thinning(WS_new, N, params.wSx1000)

    ASW_new = jnp.clip(ASW + precip - VPD * n_days * 20.0 * phi, 0.0, site.ASW_max)

    new_state = State(WF=WF_new, WR=WR_new, WS=WS_new, N=N_new, ASW=ASW_new, age=age_months + 1.0)

    outputs = dict(
        GPP=GPP,
        NPP=NPP,
        LAI=LAI,
        DBH=B,
        fT=fT,
        fD=fD,
        fSW=fSW,
        fAge=fA,
        WF=WF_new,
        WR=WR_new,
        WS=WS_new,
        N=N_new,
        Volume=WS_new * 0.85 / (params.tRho + 1e-8),
    )

    return new_state, outputs


def run_3pg(initial_state, climate, params, site, species):
    """Run 3PG model."""
    climate_stack = jnp.stack(
        [
            climate.T_avg,
            climate.VPD,
            climate.precip,
            climate.solar_rad,
            climate.frost_days,
            climate.n_days,
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


def plot_outputs(outputs):
    """Visualize key 3-PG state variables over time."""
    months = jnp.arange(outputs["WS"].shape[0])

    fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharex=True)

    axes[0, 0].plot(months, outputs["DBH"])
    axes[0, 0].set_ylabel("DBH")

    axes[0, 1].plot(months, outputs["LAI"])
    axes[0, 1].set_ylabel("LAI")

    axes[0, 2].plot(months, outputs["GPP"])
    axes[0, 2].set_ylabel("GPP")

    axes[1, 0].plot(months, outputs["WS"])
    axes[1, 0].set_ylabel("Stem biomass")

    axes[1, 1].plot(months, outputs["WF"])
    axes[1, 1].set_ylabel("Foliage biomass")

    axes[1, 2].plot(months, outputs["WR"])
    axes[1, 2].set_ylabel("Root biomass")

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Time (months)")

    plt.tight_layout()
    plt.show()


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
