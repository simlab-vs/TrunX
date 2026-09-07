from trunx.models.pppg.allocation import (
    AllocationRatios,
    TurnoverRates,
    compute_allocation,
    compute_turnover,
)
from trunx.models.pppg.constants import default_species_parameters
from trunx.models.pppg.dynamics import StandState
from trunx.models.pppg.parameters import (
    AllocationParameters,
    SiteFactors,
    TurnoverParameters,
    WeatherData,
)


def test_species_parameters_nesting():
    assert default_species_parameters.allocation.min_root_ratio == 0.25
    assert default_species_parameters.turnover.litterfall_age == 12
    assert default_species_parameters.water.conductance_max == 0.02


def test_inits():
    stand_state = StandState(
        population=100, foliage=100, stem=100, roots=100, age=1, available_soil_water=100
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

    assert stand_state.population == 100
    assert site_factors.latitude == 50
    assert site_factors.asw_min == 0.0
    assert weather_data.average_max_temp == 10
    assert weather_data.irrigation == 0


def test_allocation():
    allocation_params = AllocationParameters(
        min_root_ratio=0.25, max_root_ratio=0.8, fertility_allocation=0.0
    )
    allocation_ratios = compute_allocation(
        physiological_modifier=1.0,
        foliage_stem_ratio=0.3,
        fertility_rating=0.3,
        params=allocation_params,
    )

    turnover_params = TurnoverParameters(
        root_turnover_rate=0.015,
        litterfall_init=0.001,
        litterfall_mature=0.027,
        litterfall_age=12,
    )
    turnover_rates = compute_turnover(
        age=1, root_rate=0.1, stem_number_rate=0.01, params=turnover_params
    )

    assert isinstance(allocation_ratios, AllocationRatios)
    assert isinstance(turnover_rates, TurnoverRates)
    assert turnover_rates.stem_number == 0.01
    assert turnover_rates.stem == 0.0


def test_validation():
    pass


def test_constraints():
    pass
