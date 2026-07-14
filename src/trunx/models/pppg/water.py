"""Water balance models (available soil water)."""

import numpy as np

from trunx.models.pppg.allometry import leaf_area_index

type WaterHeight = float  # [mm]


# TODO: old implementation.
"""
def optimal_canopy_conductance(
    leaf_area_index, unconstrained_conductance, max_conductance, lai_at_max_conductance
):
    if leaf_area_index > lai_at_max_conductance:
        cond = max_conductance
    else:
        cond = unconstrained_conductance + (max_conductance - unconstrained_conductance) * (
            leaf_area_index / lai_at_max_conductance
        )

    return cond


def canopy_conductance(physiological_modifier, conductanc_modifier, leaf_area_index):
    return (
        physiological_modifier * conductanc_modifier * optimal_canopy_conductance(leaf_area_index)
    )
"""

def canopy_conductance(conductance_modifier, physiological_modifier, leaf_area_index, lai_at_max_conductance):
    """Compute the canopy conductance.

    Reference: Sands (2004) Eq. 16.

    """
    return conductance_modifier * physiological_modifier * np.min( 1, leaf_area_index / lai_at_max_conductance)


def rain_interception_rate(rain_interception_rate_max, leaf_area_index, lai_at_max_interception):
    """Compute rainfall interception rate.

    Reference: Sands (2004) Eq. 15

    """
    return rain_interception_rate_max * np.min(1, leaf_area_index / lai_at_max_interception)
