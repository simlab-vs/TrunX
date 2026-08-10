from trunx.models.pppg.schemas import (
    AllocationRatios,
    SiteFactors,
    StandInitializationData,
    TurnoverRates,
    WeatherData,
)


def test_inits():
    stand_data = StandInitializationData(
        population=100, foliage_biomass=100, stem_biomass=100, root_biomass=100, age=1
    )
    site_factors = SiteFactors(latitude=50, fertility_rating=0.3, asw_min=0.0, asw_max=200.0)

    weather_data = WeatherData(
        average_max_temp=10,
        average_min_temp=2,
        average_radiation=100,
        rainfall=20,
        irrigation=0,
        n_rain_days=2,
        n_frost_days=3,
    )
    allocation_ratios = AllocationRatios(foliage_ratio=0.2, stem_ratio=0.6, roots_ratio=0.2)

    turnover_rates = TurnoverRates(
        foliage_rate=0.1, stem_rate=0.1, roots_rate=0.1, stem_number_rate=0.01
    )

    assert stand_data.population == 100
    assert site_factors.latitude == 50
    assert site_factors.asw_min == 0.0
    assert weather_data.average_max_temp == 10
    assert weather_data.irrigation == 0
    assert allocation_ratios.foliage_ratio == 0.2
    assert turnover_rates.foliage_rate == 0.1


def test_validation():
    pass


def test_constraints():
    pass
