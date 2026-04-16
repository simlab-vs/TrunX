"""Allometric relations."""

from trunx.models.pppg.schemas import SurfaceBiomass

conversion_t_per_ha_kg_per_m2 = 0.1  # [tonnes/ha -> kg/m2]


def lea_area_index(foliage_mass: SurfaceBiomass, specific_leaf_area: float) -> float:
    """Compute leaf area index.

    Parameters
    ----------
    foliage_mass: SurfaceBiomass
        Total current foliage mass
    specific_leaf_area: float
        Species specific parameter (possibly age-dependent) [m2/kg]
    """
    return conversion_t_per_ha_kg_per_m2 * specific_leaf_area * foliage_mass
