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

# `thinPower` and `rhoMin` also pin against their current bounds in some data files,
# but published calibrations disagree by up to 2x on those two (e.g. Trotsiuk et al.
# 2020 vs. Forrester et al. 2021 for thinPower), so widening isn't obviously the
# right fix there; left alone pending a separate look at parameter identifiability.
# See `load_files.literature_bound_overrides` for the species-dependent widening
# applied to Tmax and MaxAge.
