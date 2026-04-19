"""Constants and default values for the 3PG model.

Default species parameters are the ones for E. Globulus,
taken from Sands (2004).

"""

from trunx.models.pppg.schemas import SpeciesParameters

default_species_parameters = SpeciesParameters(
    min_root_ratio=0.25,
    max_root_ratio=0.8,
    fertility_allocation_param=0.0,
    root_turnover_rate=0.015,  # [1 / month]
    litterfall_init=0.027,  # [1 / month]
    litterfall_mature=0.001,  # [1 / month]
    litterfall_age=12,  # [month]
    min_temp=8.5,  # [C°]
    opt_temp=16,  # [C°]
    max_temp=40,  # [C°]
    frost_loss_coeff=0,  # [day]
    vapour_pressure_coefficient=0.05,  # [1 / mbar]
    max_age=50 * 12,  # [month]
    canopy_quantum_efficiency=0.06,
    light_extinction_coeff=0.5,
    dbh_allometric_param=(2.4, 0.095),
    fs_ratio_2=1,
    fs_ratio_20=0.15,
    specific_area_init=11,  # [m2 / kg]
    specific_area_mature=4,  # [m2 / kg]
    specific_area_age=2.5 * 12,  # [months]
)
