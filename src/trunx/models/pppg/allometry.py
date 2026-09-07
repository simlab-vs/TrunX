"""Allometric relations.

Allometric relations allow to compute certain physical charateristics of trees:

- leaf area index (LAI)
- Diameter at breast height (DBH)
- Foliage to stem partitioning ratio

as a function of other charateristics of the tree (mostly biomass).
"""

import numpy as np

from trunx.models.pppg.parameters import AllometryParameters
from trunx.models.pppg.quantities import (
    DiameterBreastHeight,
    LeafAreaIndex,
    Month,
    SpecificArea,
    SurfaceBiomass,
)

conversion_t_per_ha_kg_per_m2 = 0.1  # [tonnes/ha -> kg/m2]
conversion_t_to_kg = 1000  # [tonnes -> kg]


def compute_LAI_from_biomass(
    foliage_mass: SurfaceBiomass, specific_leaf_area: float
) -> LeafAreaIndex:
    """Compute leaf area index from the current biomass.

    Parameters
    ----------
    foliage_mass: SurfaceBiomass
        Total current foliage mass
    specific_leaf_area: float
        Species specific parameter (possibly age-dependent) [m2/kg]
    """
    return conversion_t_per_ha_kg_per_m2 * specific_leaf_area * foliage_mass


def compute_dbh_from_biomass(
    stem_biomass: SurfaceBiomass,
    population: float,
    params: AllometryParameters,
) -> DiameterBreastHeight:
    """Compute the diameter at breast height (DBH)."""
    return (conversion_t_to_kg * stem_biomass / (params.dbh_scale * population)) ** (
        1 / params.dbh_power
    )


def compute_foliage_stem_ratio(
    dbh: DiameterBreastHeight, fs_ratio_2: float, fs_ratio_20: float
) -> float:
    """Compute the foliage to stem partition ration p_FS.

    Reference: Sands and Landsberg (2002), Eq. (A8, A10).

    Parameters
    ----------
    dbh: float
    fs_ratio_2: float
        Foliage to stem ratio at dbh = 2
    fs_ratio_2: float
        Foliage to stem ratio at dbh = 20
    fs_ratio_20: float

    """
    n = np.log(fs_ratio_20 / fs_ratio_2) / np.log(10)
    a = fs_ratio_2 / 2**n

    return a * dbh**n


def compute_specific_leaf_area(
    age: Month,
    specific_area_init: SpecificArea,
    specific_area_mature: SpecificArea,
    specific_area_age: Month,
) -> SpecificArea:
    """Compute the age-dependent specific leaf area.

    References
    ----------
      - Landsberg and Sands (2002): Eq (A.15)

    Parameters
    ----------
    age: float
        Current age of the stand
    specific_area_init: SpecificArea
        Species specific specific area at age 0
    specific_area_mature: SpecificArea
        Species specific specific area at maturity
    speficif_area_age: float
        Species specific parameter defining age at which
        specific leaf area reaches its mean value
    """
    return specific_area_mature + (specific_area_init - specific_area_mature) * np.exp(
        -np.log(2) * (age / specific_area_age) ** 2
    )
