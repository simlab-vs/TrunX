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

# Named calibration scenarios (see `run_calibration_sweep.py`), each mapping to the
# error names excluded from that scenario's fit — i.e. the `DIAGNOSTIC_ONLY_ERROR_NAMES`
# override used while running it. Also consulted by `bayesian_comparison_plots.py` to
# know which err_* posteriors actually exist for a given scenario's saved run.
ERROR_MODES: dict[str, frozenset[str]] = {
    "all_error_terms": frozenset(),
    "biomass_only": frozenset({"err_DBH", "err_BA", "err_Height"}),
    "biomass_DBH_only": frozenset({"err_BA", "err_Height"}),
    "DBH_only": frozenset({"err_BA", "err_Height", "err_WF", "err_WS", "err_WR"}),
}

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
DIAGNOSTIC_ONLY_ERROR_NAMES = ERROR_MODES["biomass_only"]

# Calibration parameter names that override an initial `State` field instead of a
# `Params` field, letting the initial-condition biomass pools be treated as uncertain
# (fitted) quantities rather than fixed inputs read from the site data. Include e.g.
# `"WS0"` in a `priors` dict to give it a `pm.Normal` prior like any other parameter
# — its posterior spread is then the quantified uncertainty in the initial state. See
# `pymc_param_est.build_loglikelihood_fn` and `calibration_utils.predict_from_parameter_draws`.
INITIAL_STATE_PARAMS: dict[str, str] = {"WS0": "WS", "WR0": "WR", "WF0": "WF"}

# The process-error sigma for each of INITIAL_STATE_PARAMS's initial-condition values,
# named `perr_{field}` (e.g. `"perr_WS"`) rather than reusing `err_{field}` — `err_WS`
# already means the *observation*-noise sigma scoring the simulated WS *trajectory*
# against observations (see `build_loglikelihood_fn`'s `packed_observations`), a
# different quantity from uncertainty in the *initial condition* itself. Include both
# `"WS0"` and `"perr_WS"` in a `priors` dict — e.g. via `run_pymc_analysis`/
# `run_map_analysis`'s `include_process_error` flag — to fit
# `WS0 ~ Normal(state.WS, perr_WS)` instead of leaving `state.WS` fixed. Unlike
# `err_{field}`, `perr_{field}` has no counterpart derived from `INITIAL_STATE_PARAMS`'s
# keys — it's keyed by the state *field* name, not the sampled *value* name.
PROCESS_ERROR_PARAM_NAMES: frozenset[str] = frozenset(
    f"perr_{field}" for field in INITIAL_STATE_PARAMS.values()
)

OUTPUT_PARAM_DEPENDENCIES: dict[str, dict[str, list[str]]] = {
    "DBH": {
        "direct": ["aWS", "nWS"],
        "indirect": [
            "pFS2",
            "pFS20",
            "pRx",
            "pRn",
            "m0",
            "alphaCx",
            "Tmin",
            "Topt",
            "Tmax",
            "fN0",
            "fNn",
            "MaxAge",
            "rAge",
            "gammaN1",
            "thinPower",
            "wSx1000",
            "mS",
        ],
    },
    "BA": {
        # No dedicated calibrated parameters of its own — BA is computed
        # directly from DBH and N, so all dependencies are indirect.
        "direct": [],
        "indirect": [
            "aWS",
            "nWS",
            "pFS2",
            "pFS20",
            "pRx",
            "pRn",
            "m0",
            "alphaCx",
            "Tmin",
            "Topt",
            "Tmax",
            "fN0",
            "fNn",
            "MaxAge",
            "rAge",
            "gammaN1",
            "thinPower",
            "wSx1000",
            "mS",
        ],
    },
    "Height": {
        "direct": ["aH", "nHB", "nHC", "rhoMin", "rhoMax", "tRho"],
        "indirect": [
            "aWS",
            "nWS",
            "pFS2",
            "pFS20",
            "pRx",
            "pRn",
            "m0",
            "alphaCx",
            "Tmin",
            "Topt",
            "Tmax",
            "fN0",
            "fNn",
            "MaxAge",
            "rAge",
            "gammaN1",
            "thinPower",
            "wSx1000",
            "mS",
        ],
    },
    "WF": {
        "direct": [
            "pFS2",
            "pFS20",
            "pRx",
            "pRn",
            "m0",
            "gammaF0",
            "gammaF1",
            "tgammaF",
        ],
        "indirect": [
            "aWS",
            "nWS",
            "alphaCx",
            "Tmin",
            "Topt",
            "Tmax",
            "fN0",
            "fNn",
            "MaxAge",
            "rAge",
            "gammaN1",
            "thinPower",
            "wSx1000",
            "mS",
        ],
    },
    "WS": {
        "direct": ["pFS2", "pFS20", "pRx", "pRn", "m0"],
        "indirect": [
            "aWS",
            "nWS",
            "alphaCx",
            "Tmin",
            "Topt",
            "Tmax",
            "fN0",
            "fNn",
            "MaxAge",
            "rAge",
            "gammaN1",
            "thinPower",
            "wSx1000",
            "mS",
        ],
    },
    "WR": {
        "direct": ["pRx", "pRn", "m0", "gammaR"],
        "indirect": [
            "pFS2",
            "pFS20",
            "aWS",
            "nWS",
            "alphaCx",
            "Tmin",
            "Topt",
            "Tmax",
            "fN0",
            "fNn",
            "MaxAge",
            "rAge",
            "gammaN1",
            "thinPower",
            "wSx1000",
            "mS",
        ],
    },
}


def error_mode_param_dependencies(mode: str) -> dict[str, list[str]]:
    """List direct and indirect parameters fit under a calibration error mode.

    Parameters
    ----------
    mode : str
        Key into `ERROR_MODES`.

    Returns
    -------
    dict[str, list[str]]
        `"direct"` and `"indirect"` parameter names, unioned over every
        output whose error term is active (not excluded) under `mode`. A
        parameter direct for at least one active output is reported as
        direct, even if indirect for another.
    """
    active_outputs = [
        output for output in OUTPUT_PARAM_DEPENDENCIES if f"err_{output}" not in ERROR_MODES[mode]
    ]
    direct = list(
        dict.fromkeys(
            param
            for output in active_outputs
            for param in OUTPUT_PARAM_DEPENDENCIES[output]["direct"]
        )
    )
    indirect = list(
        dict.fromkeys(
            param
            for output in active_outputs
            for param in OUTPUT_PARAM_DEPENDENCIES[output]["indirect"]
            if param not in direct
        )
    )

    parameter_set = set(direct + indirect)
    return {"parameters": list(parameter_set), "direct": direct, "indirect": indirect}


ERROR_MODE_PARAM_DEPENDENCIES: dict[str, dict[str, list[str]]] = {
    mode: error_mode_param_dependencies(mode) for mode in ERROR_MODES
}

if __name__ == "__main__":
    print("ERROR_MODE_PARAM_DEPENDENCIES:")
    for mode, deps in ERROR_MODE_PARAM_DEPENDENCIES.items():
        print(f"  {mode}:")
        print(f"    parameters: {deps['parameters']}")
        print(f"    length of parameters: {len(deps['parameters'])}")
