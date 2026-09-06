"""Dataclasses and validators for parameters of the 3GP model.

This submodule provides containers for parameters of the 3GP model.
These containers should be used to validate external data.
Data is only validated at the model boundaries.

Documentation of the 3GP model can be found at
[https://3pg.forestry.ubc.ca/files/2014/04/3PGpjs_UserManual.pdf](https://3pg.forestry.ubc.ca/files/2014/04/3PGpjs_UserManual.pdf).
"""

from pydantic import BaseModel, ConfigDict, Field


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
    asw_min: float = Field(ge=0.0, description="Minimum available soil water [mm]")
    asw_max: float = Field(ge=0.0, description="Maximum available soil water [mm]")


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
    rainfall: float = Field(ge=0.0, description="total monthly rainfall [mm/month]")
    irrigation: float = Field(ge=0.0, description="total monthly irrigation [mm/month]")
    n_rain_days: float = Field(ge=0.0, le=31.0, description="number of rain days in the month")
    n_frost_days: float = Field(ge=0.0, le=30.0, description="number of frost days in the month")


class AllocationRatios(Params):
    """Allocation ratios of the net primary production to the different biomass pools."""

    foliage_ratio: float = Field(ge=0.0, le=1.0, description="foliage allocation ratio")
    stem_ratio: float = Field(ge=0.0, le=1.0, description="stem allocation ratio")
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
    vapour_pressure_coefficient: float = Field(
        ge=0.0, description="coefficient for VPD effect on growth [1/mbar]"
    )
    max_age: float = Field(gt=0.0, description="maximum stand age [months]")

    # Light and allometry
    canopy_quantum_efficiency: float = Field(
        gt=0.0, description="canopy quantum efficiency [mol/mol]"
    )
    light_extinction_coeff: float = Field(
        gt=0.0, description="light extinction coefficient for Beer's law"
    )

    dbh_allometric_param: tuple[float, float] = (
        Field(description="parameters for the dbh to stem mass allometric equation (power)"),
        Field(description="parameters for the dbh to stem mass allometric equation (scaling)"),
    )
    fs_ratio_2: float = Field(gt=0.0, description="foliage to stem allocation ration at dbh=2")
    fs_ratio_20: float = Field(gt=0.0, description="foliage to stem allocation ration at dbh=20")
    specific_area_init: float = Field(gt=0.0, description="specific leaf area at age 0 [m2 / kg]")
    specific_area_mature: float = Field(
        gt=0.0, description="specific leaf area for mature stands [m2 / kg]"
    )
    specific_area_age: float = Field(
        gt=0.0, description="age at which specific leaf area reaches its mean value [month]"
    )


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


class WaterParameters(Params):
    """Parameters of the water model."""

    interception_rate_max: float = Field(
        ge=0.0, le=1.0, description="Maximum rainfall interception rate"
    )
    lai_at_max_interception: float = Field(
        ge=0.0, description="LAI at maximum rainfall interception"
    )
    lai_at_max_conductance: float = Field(ge=0.0, description="LAI at maximal canopy conductance")
    conductance_at_lai0: float = Field(ge=0.0, description="Canopy conductance at 0 LAI")
    conductance_max: float = Field(ge=0.0, description="Maximal canopy conductance")
