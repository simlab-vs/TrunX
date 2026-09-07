"""Environmental modifiers."""

from dataclasses import dataclass

import numpy as np

from trunx.models.pppg.quantities import Modifier


@dataclass(frozen=True)
class Modifiers:
    """Bundle of the environmental and physiological modifiers used across the 3PG equations."""

    age: Modifier
    frost: Modifier
    fertility: Modifier
    salinity: Modifier
    co2: Modifier
    physiological: Modifier
    vapour: Modifier
    water: Modifier
    temperature: Modifier


def effective_quantum_efficiency(quantum_efficiency: float, mods: Modifiers) -> float:
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
    return (
        mods.frost * mods.fertility * mods.temperature * mods.physiological
    ) * quantum_efficiency


def physiological_modifier(
    age_mod: Modifier, vapour_mod: Modifier, water_mod: Modifier
) -> Modifier:
    """Compute the physiological modifier.

    This modifier determines the allocation ratios, the effective quantum efficiency
    and the canopy conductance.

    Reference: Landsberg and Sands (2011) Eq. 9.9 and Sands (2004) Eq. 4.

    Parameters
    ----------
    age_mod: float
    vapour_mod: float
    water_mod: float

    """
    return age_mod * np.minimum(vapour_mod, water_mod)


def temperature_modifier(
    average_temp: float, min_temp: float, max_temp: float, opt_temp: float
) -> Modifier:
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


def frost_modifier(n_frost_days: float, frost_loss_coefficient) -> Modifier:
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


def vapour_modifier(vapour_deficit: float, vapour_pressure_coefficient: float) -> Modifier:
    """Compute the vapour pressure deficit (VPD) modifier.

    Parameters
    ----------
    vapour_deficit: float
        Average day-time vapour pressure deficit [mbar]
    vapour_coefficient: float
        Species specific coefficient for the effect of vapour pressure
        deficit [1 / mbar].

    """
    return np.exp(-vapour_pressure_coefficient * vapour_deficit)


def water_modifier(relative_available_water: float, water_modifier_shape) -> Modifier:
    """Compute the soil water modifier.

    Parameters
    ----------
    relative_available_water: float
        Relative plant-available soil water
    water_modifier_shape
        Soil texture specific shape parameters for the soil modifier
    """
    b = 1 + ((1 - relative_available_water) / water_modifier_shape[1]) ** water_modifier_shape[0]
    return 1 / b


def fertility_modifier(
    fertility_rating: float, fertility_modifier_shape: tuple[float, float] = (1, 0.5)
) -> Modifier:
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
) -> Modifier:
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
