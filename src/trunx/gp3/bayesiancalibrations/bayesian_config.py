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

# `compute_dbh` derives DBH from a single stand-level "mean tree": it inverts
# aWS/nWS on the mean stem biomass per tree. `BA` and `Height` are then computed
# from that same DBH. The ICP observations for all three are instead built by
# summing per-tree allometric equations over each stand's actual DBH distribution
# (see create_data_inputs.py) — a different, distribution-aware aggregation that
# the model's single-mean-tree inversion cannot match whenever a stand has real
# size spread. Fitting err_DBH/err_BA/err_Height therefore pushes the optimizer
# to trade away real WS/WF/WR accuracy for a target the model can't correctly
# represent, so their sigma priors are excluded from calibration; the variables
# are still simulated and can be plotted for reference. See TODO.md.
DIAGNOSTIC_ONLY_ERROR_NAMES = frozenset({"err_DBH", "err_BA", "err_Height"})

# Named calibration scenarios (see `run_calibration_sweep.py`), each mapping to the
# error names excluded from that scenario's fit — i.e. the `DIAGNOSTIC_ONLY_ERROR_NAMES`
# override used while running it. Also consulted by `bayesian_comparison_plots.py` to
# know which err_* posteriors actually exist for a given scenario's saved run.
ERROR_MODES: dict[str, frozenset[str]] = {
    "all_error_terms": frozenset(),
    "biomass_only": frozenset({"err_DBH", "err_BA", "err_Height"}),
    "biomass_DBH_only": frozenset({"err_BA", "err_Height"}),
}
