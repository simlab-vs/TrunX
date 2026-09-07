"""Stand-level dynamics.

This submodule implements forest dynamics at the stand level. This includes
two main components:

- biomass pools evolution
- population (stem) dynamics

Reference: Landsberg and Sands (2011): Eqs (9.1) and (9.2)
"""

from dataclasses import dataclass

from trunx.models.pppg.quantities import (
    Month,
    PopulationDensity,
    SurfaceBiomass,
    WaterHeight,
)


@dataclass(frozen=True)
class StandState:
    """Stand state."""

    population: PopulationDensity
    foliage: SurfaceBiomass
    stem: SurfaceBiomass
    roots: SurfaceBiomass
    age: Month
    available_soil_water: WaterHeight
