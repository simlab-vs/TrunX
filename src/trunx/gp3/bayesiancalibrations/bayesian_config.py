"""Bayesian calibration configuration."""

FIT_PARAMS = [
    "pFS20",
    "aWS",
    "nWS",
    "pRn",
    "Tmin",
    "Topt",
    "Tmax",
    "fN0",
    "fNn",
    "MaxAge",
    "rAge",
    "gammaN1",
    "thinPower",
    "mS",
    "alphaCx",
    "rhoMin",
    "rhoMax",
    "aH",
    "nHB",
    "nHC",
]

# Prior bounds for the Solling stand (Picea abies, Fagus sylvatica) that widen the
# data file's range where it pins the MAP against a wall the literature doesn't
# support. Pass to `load_priors_from_file`'s `bound_overrides`.
#
# `thinPower` and `rhoMin` also pin against their current bounds, but published
# calibrations disagree by up to 2x on those two (e.g. Trotsiuk et al. 2020 vs.
# Forrester et al. 2021 for thinPower), so widening isn't obviously the right fix
# there; left alone pending a separate look at parameter identifiability.
SOLLING_BOUND_OVERRIDES: dict[str, tuple[float, float]] = {
    # Forrester et al. 2021 (central European calibration) reports [30, 45] for
    # both species; the data file's [25, 40] pins the Solling MAP near the ceiling.
    "Tmax": (25.0, 45.0),
    # Trotsiuk et al. 2020's species table gives Picea abies a ceiling of 500, not
    # the 400 the data file shares with Fagus sylvatica.
    "MaxAge": (200.0, 500.0),
}
