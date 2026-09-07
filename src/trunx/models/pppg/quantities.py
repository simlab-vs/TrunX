"""Types for physical quantities used in the 3PG model."""

type Ratio = float  # ratio (for allocations etc...)

type Month = float
type Rate = float  # monthly rates [1 / month]

type PopulationDensity = float  # trees / ha

type SurfaceBiomass = float  # [t / ha]
type SurfaceMassRate = float  # [t / month * ha]

type EnergyFlow = float  # [MJ / day * m2]
type MassPerEnergy = float  # [g / MJ]

type Modifier = float  # [0-1]

type SpecificArea = float  # [m2 / kg]
type LeafAreaIndex = float  # [m2/m2]
type DiameterBreastHeight = float  # [cm]

type WaterHeight = float  # [mm]
type Conductance = float
