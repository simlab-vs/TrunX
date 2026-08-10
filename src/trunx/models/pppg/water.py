"""Water balance models (available soil water)."""

import numpy as np

from trunx.models.pppg.allometry import LeafAreaIndex
from trunx.models.pppg.environment import Modifiers
from trunx.models.pppg.schemas import SiteFactors, WaterParameters, WeatherData

type WaterHeight = float  # [mm]
type Conductance = float


def compute_conductance(
    params: WaterParameters, mods: Modifiers, LAI: LeafAreaIndex
) -> Conductance:
    """Compute the canopy conductance.

    Reference: Sands (2004) Eq. 16.
    """
    return (
        params.conductance_max
        * mods.physiological
        * np.minimum(1, LAI / params.lai_at_max_conductance)
    )


def compute_unconstrained_conductance(LAI: LeafAreaIndex, params: WaterParameters) -> Conductance:
    """Compute the unconstrained canopy conductance used in the P.M. equations.

    Reference: Sands and Landsberg (2011) Eq. 9.17.
    """
    if params.lai_at_max_conductance <= LAI:
        return params.conductance_max

    return params.conductance_at_lai0 + (params.conductance_max - params.conductance_at_lai0) * (
        LAI / params.lai_at_max_conductance
    )


def bulked_canopy_conductance(
    LAI: LeafAreaIndex, mods: Modifiers, params: WaterParameters
) -> Conductance:
    """Compute bulked (leaves + ground) canopy conductance for the P.M. equations.

    The bulked canopy conductance is a proxy used in the Penman-Monteith equations
    to compute total evapotranspiration.

    Reference: Sands and Landsberg (2011) Eq. 9.16.
    """
    conductance_unconstrained = compute_unconstrained_conductance(LAI, params)
    return mods.co2 * mods.physiological * conductance_unconstrained


def compute_interception_rate(LAI: LeafAreaIndex, params: WaterParameters) -> float:
    """Compute rainfall interception rate.

    Reference: Sands (2004) Eq. 15
    """
    return params.interception_rate_max * np.minimum(1, LAI / params.lai_at_max_interception)


def compute_transpiration() -> WaterHeight:
    """Compute evapotranspiration using the Penman-Monteith equations.

    Not implemented yet: this needs bulked canopy conductance, net radiation and
    VPD wired in as inputs first.
    """
    raise NotImplementedError("Penman-Monteith transpiration is not implemented yet.")


def compute_available_soil_water(
    asw_current: WaterHeight,
    LAI: LeafAreaIndex,
    forcings: WeatherData,
    params: WaterParameters,
    site: SiteFactors,
) -> WaterHeight:
    """Compute available soil water (ASW) at t+1.

    Reference: Sands (2004) Eq. 14.
    """
    transpiration = compute_transpiration()

    # effective rainfall = incoming - intercepted
    interception_rate = compute_interception_rate(LAI, params)
    rainfall_eff = (1 - interception_rate) * forcings.rainfall
    return np.clip(
        asw_current + rainfall_eff + forcings.irrigation - transpiration,
        a_min=site.asw_min,
        a_max=site.asw_max,
    )
