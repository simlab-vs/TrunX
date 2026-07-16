# pppg object model — design

Date: 2026-07-15
Status: approved design, pre-implementation

## Goal

Give the `trunx.models.pppg` (3PG) package a consistent object model so that every
sub-module follows the conventions already established in `water.py`: functions take
structured objects, return typed quantities, carry minimal docstrings, and pass
`uvx ty check`. Today the package mixes two typing mechanisms, duplicates class names
across modules, has a non-functional biomass update, and fails the type checker.

This design covers **structure only** — which objects exist, what layer they belong
to, and where code moves. Missing physics (see "Out of scope") is a separate pass.

## Organizing principles

Two axes classify every object.

**Layer — boundary vs. internal.**
- *Boundary* objects arrive from the outside world and need validation. They are
  pydantic models (`schemas.py`). Validation runs on construction.
- *Internal* objects are computed and passed around during a run. They are frozen
  dataclasses — no per-step validation cost, and (decisively) their `float`-typed
  fields accept `np.ndarray` when the model is later vectorized. A frozen pydantic
  model with `float` fields rejects arrays with `ValidationError`; a frozen dataclass
  does not. This is what keeps the vector door open.

**Lifetime.**
- *Parameters* — constant per run (`SpeciesParameters`, `SiteFactors`).
- *Forcings* — exogenous, one per timestep (`WeatherData`).
- *State* — integrated across steps (`StandState`).
- *Derived* — recomputed every step, never stored in state (`Modifiers`,
  `AllocationRatios`, `TurnoverRates`, LAI, DBH, SLA, NPP).

The load-bearing rule: **derived quantities are never cached in state.** LAI is needed
by both `water.py` and `light.py`; DBH feeds allocation. `step()` computes each once as
a local and passes it down — exactly as `water.py` already takes `LAI` as an argument.

## Execution model

Scalar now, vector door kept open. Every quantity is typed `float` today. Physics code
uses **elementwise numpy ufuncs only** — no reductions (`np.min`, `np.sum`) that would
collapse a per-stand array to a scalar. When vectorization is needed, the type aliases
in `quantities.py` widen in one place and every signature follows.

## Types and objects

### `quantities.py` (NEW) — the vocabulary

All quantity aliases live here as transparent `type X = float` with unit comments.
This is the single point that widens for vectorization.

```python
type Modifier = float             # [0-1]
type SurfaceBiomass = float       # [t/ha]
type SurfaceMassRate = float      # [t/(month·ha)]
type EnergyFlow = float           # [MJ/(day·m2)]
type MassPerEnergy = float        # [g/MJ]
type MonthRate = float            # [1/month]     (replaces mislabelled DayRate)
type SpecificArea = float         # [m2/kg]
type WaterHeight = float          # [mm]
type Conductance = float          # [m/s]
type LeafAreaIndex = float        # [m2/m2]        (was NewType)
type DiameterBreastHeight = float # [cm]           (was NewType("Vector"))
```

`NewType` is dropped: `ty` rejects arithmetic on a `NewType`-over-`float`
(`error[unsupported-operator]`), which is the sole cause of all 6 current type errors,
4 of them in `water.py`.

### Boundary — pydantic (`schemas.py`)

`schemas.py` is the outside-world validation interface only. It holds:

| Type | Role | Change from today |
|---|---|---|
| `Params` (base) | frozen pydantic base | keep |
| `SpeciesParameters` | species config | **nest into sub-models** (below); fix `dbh_allometric_param` |
| `SiteFactors` | site config | keep |
| `WeatherData` | per-step forcing | keep; ensure `irrigation` present |
| `StandInitializationData` | validated initial condition | **add initial `asw`** |

`SpeciesParameters` is composed of one sub-model per consuming module (generalizing the
existing `WaterParameters` slice):

```python
class SpeciesParameters(Params):
    allocation: AllocationParameters   # -> allocation.py
    turnover:   TurnoverParameters     # -> allocation.py
    growth:     GrowthModifierParameters  # -> environment.py
    light:      LightParameters        # -> light.py
    allometry:  AllometryParameters    # -> allometry.py
    water:      WaterParameters        # -> water.py
```

Field-to-sub-model mapping:

- `AllocationParameters`: `min_root_ratio`, `max_root_ratio`, `fertility_allocation_param`
- `TurnoverParameters`: `root_turnover_rate`, `litterfall_init`, `litterfall_mature`, `litterfall_age`
- `GrowthModifierParameters`: `min_temp`, `opt_temp`, `max_temp`, `frost_loss_coeff`, `vapour_pressure_coefficient`, `max_age`
- `LightParameters`: `canopy_quantum_efficiency`, `light_extinction_coeff`
- `AllometryParameters`: `dbh_allometric_param`, `fs_ratio_2`, `fs_ratio_20`, `specific_area_init`, `specific_area_mature`, `specific_area_age`
- `WaterParameters`: (existing) interception + conductance fields

Payoff: each function's signature names exactly the slice it consumes
(`temperature_modifier(weather, params: GrowthModifierParameters)`), rather than taking
a 25-field bag. `constants.py` becomes nested construction accordingly.

`dbh_allometric_param` is fixed from a tuple of two bare `Field(...)` calls (whose
default is currently two `FieldInfo` objects, `is_required=False`) to a required
`tuple[float, float]`.

No top-level holder object (`SimulationInputs`) is added now. It belongs at a future
`run()` entry point, not threaded through `step()`; weather is a series and cannot join
the run-constant bundle. Deferred (YAGNI).

### Internal — frozen dataclasses

| Type | Module | Note |
|---|---|---|
| `StandState` | `dynamics.py` | integrated state `step()` cycles; `from_init(StandInitializationData)` converts at the boundary once; includes `asw` |
| `Modifiers` | `environment.py` | stays a (frozen) dataclass; **add `temperature` field** |
| `AllocationRatios` | `allocation.py` (NEW) | 3-field container `(foliage, stem, root)`; removed from `schemas.py` |
| `TurnoverRates` | `allocation.py` (NEW) | container incl. stem-number (mortality) rate; removed from `schemas.py` |

`StandState`:

```python
@dataclass(frozen=True)
class StandState:
    foliage_biomass: SurfaceBiomass
    stem_biomass:    SurfaceBiomass
    root_biomass:    SurfaceBiomass
    population:      float
    age:             float
    asw:             WaterHeight
```

The `PoolsQuantity` hierarchy, both duplicate `AllocationRatios`/`TurnoverRates`
classes in `dynamics.py`, and `Biomass.vector_field` are **deleted**. `vector_field`
cannot run today: `production * allocation_ratios` dispatches to
`PoolsQuantity.__rmul__`, which calls the subclass constructor with 3 positional args,
but `AllocationRatios.__init__` requires 6 — immediate `TypeError`.

## Module layout

```
quantities.py   NEW  quantity type aliases + unit comments
schemas.py           boundary validation only: params, forcings, StandInitializationData
constants.py         default species params (nested construction)
allometry.py         LAI, DBH, foliage/stem ratio + compute_specific_leaf_area (moved in)
environment.py       Modifiers dataclass + modifier functions
light.py             absorbed radiation, photo efficiency
carbon.py            gross + net production
allocation.py   NEW  allocation ratios, turnover rates + litterfall_rate (moved in)
water.py             conductance, interception, transpiration, ASW
dynamics.py          StandState + step(): biomass update + mortality, StandState -> StandState
```

## What moves

| Item | From → To | Why |
|---|---|---|
| all `type X = float` aliases | `schemas.py`, `dynamics.py`, `allometry.py` → `quantities.py` | one vocabulary, single widening point |
| `LeafAreaIndex`, `DiameterBreastHeight` | `NewType` in `allometry.py` → aliases in `quantities.py` | clears the 6 `ty` errors |
| `compute_specific_leaf_area` | `dynamics.py` → `allometry.py` | feeds LAI |
| `litterfall_rate` | `dynamics.py` → `allocation.py` | feeds turnover |
| `AllocationRatios`, `TurnoverRates` physics | `dynamics.py` classes → free functions in `allocation.py` | matches `water.py` free-function style; returns internal containers |
| `AllocationRatios`, `TurnoverRates` containers | removed from `schemas.py` → `allocation.py` | derived, not boundary input |
| `StandState` | new → `dynamics.py` | lives with `step()`, its only consumer |

## Signature convention

Derived from `water.py`, whose functions already order args
**state/derived → forcings → modifiers → params → site**. The one outlier,
`compute_conductance(params, mods, LAI)`, is reordered to match. Every function returns
a typed quantity or a container; docstrings are one summary line plus a `Reference:`
line (no `Parameters`/`Notes`/`Examples` sections, per project style).

`dynamics.py` becomes purely the integrator, orchestrating
`allometry → environment → light → carbon → allocation → water` and returning a new
`StandState`.

## Bugs surfaced (fixed as part of this restructure)

Structural / provable, fixed here:

1. Duplicate `AllocationRatios`/`TurnoverRates` (schemas vs. dynamics) — resolved by the layer split.
2. `Biomass.vector_field` `TypeError` via `__rmul__`/constructor mismatch — deleted.
3. `NewType("Vector", float)` name mismatch and `NewType` arithmetic errors — aliases.
4. `dbh_allometric_param` default is two `FieldInfo` objects — fixed to required tuple.
5. `DayRate` (`[1/d]`) mislabels litterfall, which is `[1/month]` everywhere — `MonthRate`.
6. `np.min([vapour_mod, water_mod])` (`environment.py:59`) is a reduction; under the
   vector door it collapses to one scalar — change to `np.minimum`.
7. `tests/test_pppg_schema.py` is stale: passes `max_asw` (schema has `asw_min`/`asw_max`),
   omits required `irrigation`, and uses `fertility_rating=3` against `le=1.0` — updated.

## Out of scope (separate physics pass)

These are missing/incorrect *physics*, not structure, and are handled after the
restructure lands:

- **Age counted twice** in `effective_quantum_efficiency`: it multiplies `mods.age` and
  `mods.physiological`, but `physiological_modifier` already contains `age_mod`.
- **Temperature modifier never applied**: `temperature_modifier()` exists and is correct
  but nothing calls it and `Modifiers` has no temperature field (this design adds the
  field; wiring the value is the physics pass). Verify the intended chain against
  Sands (2004) Eq. 3.
- **`compute_transpiration()` is an empty stub** (Penman–Monteith), so
  `compute_available_soil_water` currently produces `None`-poisoned arithmetic.

## Testing

- `tests/test_pppg_schema.py` updated to the current boundary schemas and made to pass.
- Add a `StandState.from_init` round-trip test and a single `step()` smoke test that
  advances one month and returns a new `StandState` (guards against the deleted
  `vector_field` regressing).
- `uvx ty check src/trunx/models/pppg/` must report zero diagnostics.
- `uv run ruff check` and `uv run ruff format` clean.
