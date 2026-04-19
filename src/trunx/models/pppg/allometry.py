"""Allometric relations."""

import numpy as np

from trunx.models.pppg.schemas import SurfaceBiomass

conversion_t_per_ha_kg_per_m2 = 0.1  # [tonnes/ha -> kg/m2]
conversion_t_to_kg = 1000  # [tonnes -> kg]


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


def compute_dbh(stem_biomass, population, dbh_allometric_param: tuple[float, float]) -> float:
    """Compute the diameter at breast height (DBH)."""
    return (conversion_t_to_kg * stem_biomass / (dbh_allometric_param[1] * population)) ** (
        1 / dbh_allometric_param[0]
    )


def compute_foliage_stem_ratio(dbh: float, fs_ratio_2: float, fs_ratio_20: float) -> float:
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
