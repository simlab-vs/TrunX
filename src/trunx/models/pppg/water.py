"""Water balance models (available soil water).

This implementation follows the original from Sands (2004).
In particular, this does not include the constrained conductance model from r3PG.
"""

import numpy as np

from trunx.models.pppg.environment import Modifiers
from trunx.models.pppg.parameters import SiteFactors, WaterParameters, WeatherData
from trunx.models.pppg.quantities import Conductance, LeafAreaIndex, WaterHeight


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


def compute_interception_rate(LAI: LeafAreaIndex, params: WaterParameters) -> float:
    """Compute rainfall interception rate.

    Reference: Sands (2004) Eq. 15
    """
    epsilon = 1e-8
    if params.lai_at_max_interception > 0:
        return params.interception_rate_max * np.minimum(
            1.0, LAI / (params.lai_at_max_interception + epsilon)
        )
    else:
        return params.interception_rate_max


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
