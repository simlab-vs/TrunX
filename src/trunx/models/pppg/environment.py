"""Environmental modifiers."""

import numpy as np


def effective_quantum_efficiency(quantum_efficiency: float, modifiers: list[float]) -> float:
    """Compute the effective quantum efficiency.

    Parameters
    ----------
    quantum_efficiency: float
        Species specific canopy quantum efficiency [mol / mol]
    modifiers: list[float]
        List of multiplicative modifiers to apply. The traditional full list of
        modifiers is:
        [age, fros, fertility, salinity, co2, physiological]
    """
    return np.prod(modifiers).item() * quantum_efficiency


def physiological_modifier(age_modifier: float, vapour_modifier: float, water_modifier: float):
    """Compute the physiological modifier.

    Parameters
    ----------
    age_modifier: float
    vapour_modifier: float
    water_modifier: float

    """
    return age_modifier * np.min([vapour_modifier, water_modifier])


def temperature_modifier(
    average_temp: float, min_temp: float, max_temp: float, opt_temp: float
) -> float:
    """Compute the temperature modifier.

    Parameters
    ----------
    average_temp: float
        Monthly average daily temperature [C°]
    min_temp: float
        Species specific minimal growth temperature
    max_temp: float
        Species specific maximal growth temperature
    opt_temp: float
        Species specific optimal growth temperature

    """
    a = (average_temp - min_temp) / (opt_temp - min_temp)
    b = (max_temp - average_temp) / (max_temp - opt_temp)
    c = (max_temp - opt_temp) / (opt_temp - min_temp)

    return a * b**c


def frost_modifier(n_frost_days: float, frost_loss_coefficient) -> float:
    """Compute the frost modifier.

    Parameters
    ----------
    n_forst_days: float
        Number frost days in the month
    frost_loss_coefficient: float
        Species specific coefficient for the number of growth days
        lost for each frost day. Usually equal to 1, can be bigger
        for some species.

    """
    return 1 - frost_loss_coefficient * n_frost_days / 30.0


def vapour_modifier(vapour_deficit: float, vapour_coefficient: float) -> float:
    """Compute the vapour pressure deficit (VPD) modifier.

    Parameters
    ----------
    vapour_deficit: float
        Average day-time vapour pressure deficit [kPa]
    vapour_coefficient: float
        Species specific coefficient for the effect of vapour pressure
        deficit.

    """
    return np.exp(-vapour_coefficient * vapour_deficit)


def soil_modifier(available_water: float, soil_modifier_shape) -> float:
    """Compute the soil water modifier.

    Parameters
    ----------
    available_water: float
        Relative plant-available soil water
    soil_modifier_shape
        Soil texture specific shape parameters for the soil modifier
    """
    a = 1 - (1 - available_water) ** soil_modifier_shape[0]
    b = 1 + ((1 - available_water) / soil_modifier_shape[1]) ** soil_modifier_shape[0]
    return a / b


def fertility_modifier(
    fertility_rating: float, fertility_modifier_shape: tuple[float, float] = (1, 0.5)
) -> float:
    """Compute the fertility modifier.

    Parameters
    ----------
    fertility_rating: float
        Site fertility rating
    fertility_modifier_shape: tuple[float, float]
        Shape parameters for the fertility modifier. Usually generic, not species
        specific. Traditional value is (1 , 0.5). Defaults to this generic
        value.

    """
    a = 1 - fertility_modifier_shape[1]
    b = (1 - fertility_rating) ** fertility_modifier_shape[0]
    return 1 - a * b


def age_modifier(
    age: float, max_age: float, age_modifier_shape: tuple[float, float] = (4, 0.95)
) -> float:
    """Compute the age modifier.

    Parameters
    ----------
    vapour_deficit: float
        Average day-time vapour pressure deficit [kPa]
    max_age: float
        Species specific maximum age for a stand.
    age_modifier_shape: tuple[float, float]
        Shape parameters for the age modifier. Usually generic, not species
        specific. Traditional value is (4 , 0.95). Defaults to this generic
        value.

    """
    a = 1 + ((age / max_age) / age_modifier_shape[1]) ** age_modifier_shape[0]
    return 1 / a
