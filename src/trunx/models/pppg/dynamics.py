"""Stand-level dynamics.

This submodule implements forest dynamics at the stand level. This includes
two main components:

- biomass pools evolution
- population (stem) dynamics

Reference: Landsberg and Sands (2011): Eqs (9.1) and (9.2)
"""

from typing import Self

import numpy as np

type DayRate = float  # daily rates [1 / d]


class PoolsQuantity:
    """Convenience class to hold quantities that refer to forest biomass pools.

    Pools are always (foliage, stem, root).
    """

    def __init__(self, foliage_quantity: float, stem_quantity: float, root_quantity: float):
        self.pools = np.array([foliage_quantity, stem_quantity, root_quantity])

    @property
    def foliage(self) -> float:
        return self.pools[0]

    @property
    def stem(self) -> float:
        return self.pools[1]

    @property
    def roots(self) -> float:
        return self.pools[2]

    def __rmul__(self, val: float) -> Self:
        return self.__class__(val * self.pools[0], val * self.pools[1], val * self.pools[2])


class AllocationRatios(PoolsQuantity):
    """Biomass allocation ratios.

    These ratios determine what fraction of the net primary carbon production is allocated
    to each of the biomass pools.

    Reference: Landsberg and Sands (2011): Eqs (9.11) and (9.12)

    """

    def __init__(
        self,
        min_root_ratio: float,
        max_root_ratio: float,
        foliage_stem_ratio: float,
        physiological_modifier: float,
        fertility_allocation_param: float,
        fertility_rating: float,
    ):
        site_limitation = (
            fertility_allocation_param + (1 - fertility_allocation_param) * fertility_rating
        ) * physiological_modifier

        root = (max_root_ratio * min_root_ratio) / (
            min_root_ratio + (max_root_ratio - min_root_ratio) * site_limitation
        )
        stem = (1 - root) / (1 + foliage_stem_ratio)
        foliage = foliage_stem_ratio * stem

        return super().__init__(foliage, stem, root)


class TurnoverRates(PoolsQuantity):
    """Turnover rates, for each biomass pool.

    These ratios determine what mass fraction of each pool gets shed.
    Among these parameters, the stem rate is usually set to 0, the root rate
    is a species specific constant and the foliage rate is age dependent.

    Reference: Landsberg and Sands (2011): Eqs (9.1) and (9.2)

    """

    def __init__(self, age: float, root_rate, litterfall_init, litterfall_mature, litterfall_age):
        stem_rate = 0.0
        foliage_rate = litterfall_rate(age, litterfall_init, litterfall_mature, litterfall_age)

        return super().__init__(foliage_rate, stem_rate, root_rate)


def litterfall_rate(
    age, litterfall_init: DayRate, litterfall_mature: DayRate, litterfall_age
) -> DayRate:
    """Compute the age-dependent foliage turnover (litterfall) rate.

    References
    ----------
      - Landsberg and Sands (2011): Eq (A2.10)
      - Sands (2004): Eq. (17)

    Parameters
    ----------
    age: float
        Current age of the stand
    litterfal_init: DayRate
        Species specific litterfall rate at age 0
    litterfall_mature: DayRate
        Species specific litterfall rate at maturity
    litterfall_age: float
        Species specific parameter defining age at which litterfall rate
        reaches its mean value
    """
    a = litterfall_init * litterfall_mature
    b = np.exp(-(age / litterfall_age) * np.log(1 + litterfall_mature / litterfall_init))
    c = litterfall_init + (litterfall_mature - litterfall_init) * b

    return a / c


class Biomass(PoolsQuantity):
    """Biomass pools (foliage, stem, root) and their dynamics.

    This class implements the dynamics of the biomass pools, computing
    the right-hand side (vector field) of the evolution ODE.
    This RHS depends on 3 main quantities:

    - net primary carbon production
    - allocation ratios: depend only on the site fertility rating and the physiological modifier,
      which itself depends on age a vapour pressure deficit at current time
    - turnover rates: depend on age

    Reference: Landsberg and Sands (2011): Eqs (9.1) and (9.2)
    """

    def vector_field(
        self, production: float, allocation_ratios: AllocationRatios, turnover_rates: TurnoverRates
    ):
        return production * allocation_ratios - turnover_rates * self.pools
