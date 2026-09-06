"""Light absorption sub-model."""

import numpy as np

from trunx.models.pppg.quantities import EnergyFlow, MassPerEnergy

photosynth_active_ratio = 0.5  # converts total radiation to photosynthetically active
carbon_moles_to_plant_grams = 24  # [g / mol] converts moles of produced carbon to grams of plant
par_photons_per_MJ = 4.6  # [mol / MJ] photosynthetically active photons per MJ of radiation
# TODO: the above is copied from the Sands book, but seems inverted.


def absorbed_radiation(
    extinction_coeff: float,
    leaf_ai: float,
    canopy_cover_fraction: float,
    total_radiation: EnergyFlow,
) -> EnergyFlow:
    """Compute absorbed radiation."""
    beer_factor = 1 - np.exp(-extinction_coeff * leaf_ai / canopy_cover_fraction)
    return photosynth_active_ratio * beer_factor * total_radiation * canopy_cover_fraction


def photo_efficiency(effective_quantum_efficiency: float) -> MassPerEnergy:
    """Compute the efficiency of conversion of absorber photosynthetically active solar radiation.

    Parameters
    ----------
    effective_quantum_efficiency: float
        Effective canopy quantum efficiency, computed by applying
        the environmental modifiers.
    """
    return carbon_moles_to_plant_grams * par_photons_per_MJ * effective_quantum_efficiency
