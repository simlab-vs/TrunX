from trunx.models.pppg.schemas import (
    AllocationRatios,
    SiteFactors,
    SpeciesParameters,
    StandInitializationData,
    TurnoverRates,
    WeatherData,
)


def test_inits():
    stand_data = StandInitializationData(
        population=100, foliage_biomass=100, stem_biomass=100, root_biomass=100, age=1
    )
    site_factors = SiteFactors(latitude=50, fertility_rating=3, max_asw=0.5)

    weather_data = WeatherData(
        average_max_temp=10,
        average_min_temp=2,
        average_radiation=100,
        total_rainfall=20,
        n_rain_days=2,
        n_frost_days=3,
    )
    allocation_ratios = AllocationRatios(foliage_ratio=0.2, stem_ratio=0.6, roots_ratio=0.2)

    species_parameters = SpeciesParameters(
        min_root_ratio=0.1,
        max_root_ratio=0.1,
        foliage_stem_ratio=0.1,
        fertility_allocation_param=1.0,
        root_turnover_rate=0.3,
        litterfall_init=0.3,
        litterfall_mature=0.4,
        litterfall_age=10,
        min_temp=10,
        opt_temp=15,
        max_temp=20,
        frost_loss_coeff=1,
        vapour_pressure_coeff=2,
        max_age=100,
        canopy_quantum_efficiency=5,
        light_extinction_coeff=10,
        specific_leaf_area=2,
    )

    turnover_rates = TurnoverRates(
        foliage_rate=0.1, stem_rate=0.1, roots_rate=0.1, stem_number_rate=0.01
    )

    assert stand_data.population == 100
    assert site_factors.latitude == 50
    assert weather_data.average_max_temp == 10
    assert allocation_ratios.foliage_ratio == 0.2
    assert species_parameters.min_root_ratio == 0.1
    assert turnover_rates.foliage_rate == 0.1


def test_validation():
    pass


def test_constraints():
    pass
