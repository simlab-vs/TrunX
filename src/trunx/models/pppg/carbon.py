"""Carbon production sub-model."""

from trunx.models.pppg.schemas import EnergyFlow, MassPerEnergy, SurfaceMassRate

conversion_g_per_m2_t_per_ha = 0.01  # [grams/m2 -> tonnes/ha]


def gross_production(
    n_days: float, photo_efficiency: MassPerEnergy, absorbed_radiation: EnergyFlow
) -> SurfaceMassRate:
    """Compute gross primary carbon production from radiation and efficiency.

    Parameters
    ----------
    n_days: float
        Number of days in the month
    photo_efficiency: MassPerEnergy
        Efficiency of conversion of absorber photosynthetically
        active solar radiation.
    absorbed_radiation: EnergyFlow
        Absorbed photosynthetically active solar radiation.
        Computed by the light-submodel.

    """
    production = n_days * photo_efficiency * absorbed_radiation

    # Convert units g/m2 -> t/ha
    return conversion_g_per_m2_t_per_ha * production


def net_production(
    gross_production: SurfaceMassRate, carbon_use_efficiency: float = 0.47
) -> SurfaceMassRate:
    """Compute net primary carbon production.

    Parameters
    ----------
    gross_production: SurfaceMassRate
    carbon_use_efficiency: float
        Efficiency of carbon conversion, litterature has it as 0.47 +- 0.04.
        Defaults to 0.47.

    """
    return carbon_use_efficiency * gross_production
