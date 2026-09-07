"""Constants and default values for the 3PG model.

Default species parameters are the ones for E. Globulus,
taken from Sands (2004).
Note that the values for the litterfall_init and litterfall_mature found in Sands (2004)
seem to be inverted.

"""

from trunx.models.pppg.parameters import (
    AllocationParameters,
    AllometryParameters,
    GrowthModifierParameters,
    LightParameters,
    SpeciesParameters,
    TurnoverParameters,
    WaterParameters,
)

default_allocation_parameters = AllocationParameters(
    min_root_ratio=0.25,
    max_root_ratio=0.8,
    fertility_allocation_param=0.0,
)

default_turnover_parameters = TurnoverParameters(
    root_turnover_rate=0.015,  # [1 / month]
    litterfall_init=0.001,  # [1 / month]
    litterfall_mature=0.027,  # [1 / month]
    litterfall_age=12,  # [month]
)

default_growth_mod_parameters = GrowthModifierParameters(
    min_temp=8.5,  # [C°]
    opt_temp=16,  # [C°]
    max_temp=40,  # [C°]
    frost_loss_coeff=0,  # [day]
    vapour_pressure_coefficient=0.05,  # [1 / mbar]
    max_age=50 * 12,  # [month]
)

default_light_parameters = LightParameters(
    canopy_quantum_efficiency=0.06,
    light_extinction_coeff=0.5,
)

default_allometry_parameters = AllometryParameters(
    dbh_power=2.4,
    dbh_scale=0.095,
    fs_ratio_2=1,
    fs_ratio_20=0.15,
    specific_area_init=11,  # [m2 / kg]
    specific_area_mature=4,  # [m2 / kg]
    specific_area_age=2.5 * 12,  # [months]
)

default_water_parameters = WaterParameters(
    interception_rate_max=0.15,
    lai_at_max_interception=0,
    lai_at_max_conductance=3.33,
    conductance_max=0.02,
)

default_species_parameters = SpeciesParameters(
    allocation=default_allocation_parameters,
    turnover=default_turnover_parameters,
    growth=default_growth_mod_parameters,
    light=default_light_parameters,
    allometry=default_allometry_parameters,
    water=default_water_parameters,
)
