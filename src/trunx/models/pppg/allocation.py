"""Allocation dynamics and turnovers."""

from dataclasses import dataclass

import numpy as np

from trunx.models.pppg.parameters import AllocationParameters, TurnoverParameters
from trunx.models.pppg.quantities import Modifier, Month, Rate, Ratio


@dataclass(frozen=True)
class TurnoverRates:
    """Monthly turnover rates (loss) for the different biomass pools."""

    foliage: Rate
    stem: Rate
    roots: Rate
    stem_number: Rate


@dataclass(frozen=True)
class AllocationRatios:
    """Allocation ratios of the net primary production to the different biomass pools."""

    foliage: Ratio
    stem: Ratio
    roots: Ratio


def compute_litterfall(age, params: TurnoverParameters) -> Rate:
    """Compute the age-dependent foliage turnover (litterfall) rate.

    References
    ----------
      - Landsberg and Sands (2011): Eq (A2.10)
      - Sands (2004): Eq. (17)

    Parameters
    ----------
    age: float
        Current age of the stand
    params: TurnoverParameters
    """
    a = params.litterfall_init * params.litterfall_mature
    b = np.exp(
        -(age / params.litterfall_age)
        * np.log(1 + params.litterfall_mature / params.litterfall_init)
    )
    c = params.litterfall_init + (params.litterfall_mature - params.litterfall_init) * b

    return a / c


def compute_turnover(
    age: Month, root_rate: Rate, stem_number_rate: Rate, params: TurnoverParameters
) -> TurnoverRates:
    """Compute turnover rates, for each biomass pool.

    These ratios determine what mass fraction of each pool gets shed.
    Among these parameters, the stem rate is usually set to 0, the root rate
    is a species specific constant and the foliage rate is age dependent.

    Reference: Landsberg and Sands (2011): Eqs (9.1) and (9.2)

    Parameters
    ----------
    age: Month
    root_rate: Rate
    stem_number_rate: Rate
    params: TurnoverParameters

    """
    return TurnoverRates(
        stem=0.0,
        foliage=compute_litterfall(age, params),
        roots=root_rate,
        stem_number=stem_number_rate,
    )


def compute_allocation(
    physiological_modifier: Modifier,
    foliage_stem_ratio: Ratio,
    fertility_rating: float,
    params: AllocationParameters,
) -> AllocationRatios:
    """Compute biomass allocation ratios.

    These ratios determine what fraction of the net primary carbon production is allocated
    to each of the biomass pools.

    Reference: Landsberg and Sands (2011): Eqs (9.11) and (9.12)

    Parameters
    ----------
    physiological_modifier: float
    foliage_stem_ratio: float
    fertility_rating: float
    params: AllocationParameters

    """
    site_limitation = (
        params.fertility_allocation + (1 - params.fertility_allocation) * fertility_rating
    ) * physiological_modifier
    root_ratio = (params.max_root_ratio * params.min_root_ratio) / (
        params.min_root_ratio + (params.max_root_ratio - params.min_root_ratio) * site_limitation
    )
    stem_ratio = (1 - root_ratio) / (1 + foliage_stem_ratio)
    return AllocationRatios(
        roots=root_ratio,
        stem=stem_ratio,
        foliage=foliage_stem_ratio * stem_ratio,
    )
