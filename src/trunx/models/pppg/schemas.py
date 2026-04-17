"""Dataclasses for parameters of the 3GP model.

Documentation can be found at [https://3pg.forestry.ubc.ca/files/2014/04/3PGpjs_UserManual.pdf](https://3pg.forestry.ubc.ca/files/2014/04/3PGpjs_UserManual.pdf).
"""

from pydantic import BaseModel, ConfigDict, Field

type SurfaceBiomass = float  # [t / ha]
type SurfaceMassRate = float  # [t / month * ha]
type EnergyFlow = float  # [MJ / day * m2]
type MassPerEnergy = float  # [g / MJ]


class Params(BaseModel):
    """Class for immutable model parameters."""

    model_config = ConfigDict(frozen=True)


class StandInitializationData(Params):
    """Stand initialization data."""

    population: float = Field(gt=0.0, description="initial population (stocking) [trees/ha]")
    foliage_biomass: float = Field(gt=0.0, description="initial foliage biomass [t/ha]")
    stem_biomass: float = Field(gt=0.0, description="initial stem biomass in [t/ha]")
    root_biomass: float = Field(gt=0.0, description="initial root biomass [t/ha]")
    age: float = Field(gt=0.0, description="initial stand age [months]")


class SiteFactors(Params):
    """Site-specific factors."""

    latitude: float = Field(
        ge=-90.0, le=90.0, description="site latitude in degrees, negative for S hemisphere"
    )
    fertility_rating: float = Field(
        ge=0.0,
        le=1.0,
        description="site fertilitiy rating, from concrete (0), to non-limited by nutrients (1)",
    )
    max_asw: float = Field(gt=0.0, description="maximum plant-available soil water [mm]")


class WeatherData(Params):
    """Monthly weather data."""

    average_max_temp: float = Field(
        description="monthly average of daily maximum temperature [C°]"
    )
    average_min_temp: float = Field(
        description="monthly average of daily minimum temperature [C°]"
    )
    average_radiation: float = Field(
        ge=0.0, description="monthly average daily solar radiation [MJ/m2*day]"
    )
    total_rainfall: float = Field(ge=0.0, description="total monthly rainfall [mm/month]")
    n_rain_days: float = Field(ge=0.0, le=31.0, description="number of rain days in the month")
    n_frost_days: float = Field(ge=0.0, le=30.0, description="number of frost days in the month")


class AllocationRatios(Params):
    """Allocation ratios of the net primary production to the different biomass pools."""

    foliage_ratio: float = Field(ge=0.0, le=0.0, description="foliage allocation ratio")
    stem_ratio: float = Field(ge=0.0, le=0.0, description="stem allocation ratio")
    roots_ratio: float = Field(ge=0.0, le=1.0, description="roots allocation ratio")


class SpeciesParameters(Params):
    """Species-specific parameters for the 3PG model."""

    # Allocation
    min_root_ratio: float = Field(
        gt=0.0, le=1.0, description="minimum fraction of NPP allocated to roots"
    )
    max_root_ratio: float = Field(
        gt=0.0, le=1.0, description="maximum fraction of NPP allocated to roots"
    )
    foliage_stem_ratio: float = Field(gt=0.0, description="ratio of foliage to stem allocation")
    fertility_allocation_param: float = Field(
        ge=0.0, le=1.0, description="modifier of root allocation response to fertility"
    )

    # Turnover
    root_turnover_rate: float = Field(
        gt=0.0, le=1.0, description="monthly root turnover rate [1/month]"
    )
    litterfall_init: float = Field(
        gt=0.0, description="foliage litterfall rate at age 0 [1/month]"
    )
    litterfall_mature: float = Field(
        gt=0.0, description="foliage litterfall rate at maturity [1/month]"
    )
    litterfall_age: float = Field(
        gt=0.0,
        description="age at which litterfall rate reaches its mean value [months]",
    )

    # Temperature response
    min_temp: float = Field(description="minimum temperature for growth [°C]")
    opt_temp: float = Field(description="optimum temperature for growth [°C]")
    max_temp: float = Field(description="maximum temperature for growth [°C]")

    # Other environmental
    frost_loss_coeff: float = Field(ge=0.0, description="growth days lost per frost day")
    vapour_pressure_coeff: float = Field(
        ge=0.0, description="coefficient for VPD effect on growth [1/kPa]"
    )
    max_age: float = Field(gt=0.0, description="maximum stand age [months]")

    # Light and allometry
    canopy_quantum_efficiency: float = Field(
        gt=0.0, description="canopy quantum efficiency [mol/mol]"
    )
    light_extinction_coeff: float = Field(
        gt=0.0, description="light extinction coefficient for Beer's law"
    )
    specific_leaf_area: float = Field(gt=0.0, description="specific leaf area [m2/kg]")


class TurnoverRates(Params):
    """Monthly turnover rates (loss) for the different biomass pools."""

    foliage_rate: float = Field(
        ge=0.0, le=1.0, description="monthly foliage turnover rate [1/month]"
    )
    stem_rate: float = Field(ge=0.0, le=1.0, description="monthly stem turnover rate [1/month]")
    roots_rate: float = Field(ge=0.0, le=1.0, description="monthly roots turnover rate [1/month]")
    stem_number_rate: float = Field(
        ge=0.0, le=1.0, description="monthly stem number turnover rate (mortality) [1/month]"
    )
