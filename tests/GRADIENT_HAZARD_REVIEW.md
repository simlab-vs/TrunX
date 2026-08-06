# 3-PG Gradient Hazard Review (Step 1: Discovery)

Scope: the **differentiated path only** — `run_3pg` → `model_step`
(`src/trunx/gp3/run_3pg.py`) and every helper it calls
(`src/trunx/gp3/helper_function.py`). Entry points that differentiate this core:
`gradient_descent.py`, `PG3_model_impl.py`, `jax_morris_loglikelihood.py`,
`bayesiancalibrations/pymc_param_est*.py`.

Explicitly **out of the autodiff path** (pure polars/numpy data prep, no
`jax.grad` reaches them): `allometrics.py`, `age_regression.py`,
`weather_processing.py`, `prepare_*.py`, `save_load_results.py`. Their
`min`/`max`/`sorted`/`for`-loops are not gradient hazards.

## Mechanism legend
- **tie** — a `where`/`min`/`max` whose two arguments can be equal; the
  subgradient is ambiguous exactly at equality.
- **kink** — `clip`/`min(·,const)`/`max(·,const)` where the derivative is
  discontinuous at the bound.
- **disc** — a genuine discontinuity / branch switch (dormancy transition,
  leaf-debt repayment); forward value jumps, gradient does not see the jump.

## Coverage legend (against `tests/test_grad_smoke.py`)
- **✓ smooth-side** — the harness evaluation point sits strictly on one side of
  this operator with margin, and `check_grads` + finite-difference exercises the
  gradient there. This catches NaN/Inf and a wrong smooth-branch rule, but **not**
  the behaviour *at* the tie or on the *other* side.
- **✗ not exercised** — the operator's branch is forced off by the harness
  construction (evergreen, no debt, self-thinning masked), so no gradient flows
  through it at all.
- **n/a** — operand is a fixed structural/integer index (not differentiated).

## Hazard table

| # | Function | file:line | Mechanism | Covered? |
|---|----------|-----------|-----------|----------|
| 1 | `model_step` co-limitation `phi = fA * min(fD, fSW)` | run_3pg.py:77 | **tie** (fD==fSW) | ✓ smooth-side — harness asserts `fD < fSW` with >0.05 margin. Tie itself **not** tested. |
| 2 | `model_step` dormancy transition `where(first_dormant/first_growing,…)` | run_3pg.py:49–53 | **disc** | ✗ not exercised — evergreen (`leafgrow==leaffall==0`) makes `is_dormant` always False. |
| 3 | `model_step` leaf-debt repayment `where(NPP_scaled >= WF_debt,…)` | run_3pg.py:127–135 | **disc/tie** | ✗ not exercised — `has_debt` always False for evergreen. |
| 4 | `model_step` dormant-gated litter/increment `where(dormant,…)` | run_3pg.py:140–151 | **disc** | ✗ not exercised — dormant always False. |
| 5 | `model_step` `where(lai_total>0)` / `where(LAI==0)` | run_3pg.py:60–61,86 | tie | ✓ smooth-side (LAI>0). |
| 6 | `model_step` `clip(gammaF,0,1)` and biomass `clip(·,0,None)` | run_3pg.py:118,155,158,161 | kink | ✓ smooth-side (all strictly interior/positive). |
| 7 | `f_temperature` out-of-range + `where(b>0,b,1)` + `clip(fT,0,1)` | helper_function.py:42,46,50 | tie/kink | ✓ smooth-side — `T=12` in `(Tmin,Tmax)`, `0<fT<1` asserted. NaN-guard pattern (guard `b` before `b**power`) is correct. |
| 8 | `f_frost` `clip(1-kF·…,0,1)` | helper_function.py:72 | kink | ✓ smooth-side (`fF=0.933`). |
| 9 | `f_nutrition` `where(fNn==0.0,1,f_N)` | helper_function.py:221 | tie | ✓ smooth-side (`fNn=1`). |
| 10 | `f_soil_water` soil-class `where` + `clip(f_sw,0,1)` | helper_function.py:178–191 | tie/kink | ✓ smooth-side — `soil_class=0` sentinel, `fSW≈1` unclipped. |
| 11 | `f_age` / `compute_lai` `where(age_months==1.0,…)` | helper_function.py:133,314 | tie | ✓ smooth-side (`age=48`). |
| 12 | `compute_allocation_fraction` `clip(DBH,0.1,None)**pfsPower` | helper_function.py:633 | kink | ✓ smooth-side (DBH≈5.9≫0.1). Not explicitly asserted. |
| 13 | `apply_stress_mortality` `min(mort_stress_raw, N)` + `where(active,…)` | helper_function.py:444 | **tie** | ✓ smooth-side — harness asserts `0 < mort_stress < N`. |
| 14 | `_solve_mortality_newton` unrolled Newton: `where(abs(dfN)<1e-8,…)`, `where(n<=0,…)`, `max(mort_n,0)` | helper_function.py:372–384 | tie/kink | ✗ **not exercised** — gradient killed downstream by masked `should_thin`. **Highest-risk untested zone.** |
| 15 | `apply_self_thinning_with_mortality_factors` `should_thin` mask + `where`/`clip`/`max` | helper_function.py:400,409–420 | disc/kink | ✗ not exercised — `should_thin` always False (biomass/tree ≪ wSmax). |
| 16 | `apply_self_thinning` (`clip(rel_excess,…)`, `clip(N·…,1,None)`) | helper_function.py:498,500 | kink | n/a — **dead code**; `model_step` calls the `_with_mortality_factors` variant instead. |
| 17 | `compute_canopy_cover` `where(age<fullCanAge,young,1)` | helper_function.py:529 | tie | ✓ smooth-side (age 4–6 yr < 20 yr). Other side untested. |
| 18 | `calculate_interception` `min(1.0, lai/LAImaxIntcptn)` | helper_function.py:668 | **kink** | ✓ smooth-side (LAI<5 → ratio branch). |
| 19 | `update_soil_water` `min(ASW,total_demand)`, `max(ASW-…-ASW_max,0)`, `max(ASW,ASW_min)`, `where(total_demand==0,…)` | helper_function.py:749,750,757,758 | **tie/kink** | ✓ smooth-side — huge reservoir → demand-limited, `f_transp_scale==1` asserted. Water-stressed side untested. |
| 20 | `scale_transpiration` `where((transp_total>0)&(f_transp_scale<1),…)` | helper_function.py:775,777 | disc | ✗ not exercised — `f_transp_scale==1` → condition always False. |
| 21 | `calculate_transpiration` `max(0,transp)` + `where(VPD==0,…)` | helper_function.py:719,722 | kink/tie | ✓ smooth-side (VPD=2, transp>0). |
| 22 | `calculate_base_conductance` `where(lai<=LAIgcx,scaled,MaxCond)` | helper_function.py:929–931 | **tie** | ✓ smooth-side (LAI<3.33 → scaled branch). Boundary untested. |
| 23 | `f_temperature_gc` `clip(a,0,None)`, `clip(b,0,None)`, `clip(·,0,1)` | helper_function.py:967,968,972 | kink | ✓ smooth-side (interior). |
| 24 | `f_cg` `clip(f_cg,0,1)` | helper_function.py:993 | kink | ✓ smooth-side (`0<f_cg<1` asserted). |
| 25 | `f_exp_foliage` `where((tgammaF·gammaF1)<eps,…)` | helper_function.py:1018 | tie | ✓ smooth-side (`0.72≫eps`). |
| 26 | `f_exp_wood` `where(tRho>eps,…)` | helper_function.py:1033 | tie | ✓ smooth-side (`tRho=4`). |
| 27 | `calculate_day_length` `clip(month-1,0,11).astype(int)` gather; `clip(cosH0,-1,1)`; `where(cosH0>1/<-1)` | helper_function.py:894,899,901–902 | kink | n/a (differentiated only wrt fixed `month`/`latitude`; not in the diff'd variable set). |

## Assessment

The existing harness (`tests/test_grad_smoke.py`, 4 tests, green) is a solid
**single-point interior smoke test**: it verifies `jax.grad` wrt 13 parameters
and 7 forcing channels is finite, matches central finite differences
(`atol/rtol=1e-3`), and is jit-invariant, at one deliberately non-degenerate
point. Rows marked ✓ are genuinely covered *on their smooth side*.

**Coverage gap** — the harness by design keeps every operator one-sided, so it
does **not** probe the three places where silently-wrong gradients actually bite
during real fitting:

1. **`_solve_mortality_newton` (row 14) — highest risk.** Reverse-mode AD runs
   through 5 unrolled Newton iterations containing `where(abs(dfN)<1e-8,…)` and
   `where(n<=0,…)` guards. In the harness `should_thin` is False, so the mask at
   `helper_function.py:409` zeroes the contribution and **no gradient ever flows
   through the solver**. A wrong subgradient here would be invisible today.
2. **Dormancy / leaf-debt transitions (rows 2–4, 20).** A deciduous species
   crossing `leafgrow`/`leaffall` hits the `WF_active`/`WF_debt` discontinuity —
   entirely untested.
3. **Co-limitation `min(fD,fSW)` and the water-stressed soil balance
   (rows 1, 13, 18, 19)** exactly at / on the other side of the tie.

**No confirmed gradient bug was found by inspection.** The NaN-guard pattern in
`f_temperature` (guarding `b` *before* the fractional power, row 7) is the
correct idiom and does not leak NaN through `where`. One dead-code item (row 16,
`apply_self_thinning`) is not a hazard but is worth deleting separately.

## Recommended Step 2 (targeted tie tests — not yet written)

Add finite-difference/`check_grads` tests at points that place the evaluation
*on* the tie and *on both sides*, prioritised by risk:

- **T1 (row 14):** a high-density / high-biomass stand that turns `should_thin`
  True, so gradients flow through the Newton solver; compare rev-mode grad to FD.
- **T2 (rows 2–4):** a deciduous species (`leafgrow≠leaffall`) run across a
  dormancy transition month.
- **T3 (rows 1,13,18,19):** parameter choices that put `min(fD,fSW)`,
  `min(mort_raw,N)`, `min(1,lai/LAImaxIntcptn)` and the soil-water `min`/`max`
  at equality, checking left- and right-derivative agreement with FD.

Any genuine bug these surface will get a **minimal, isolated fix + linked test,
flagged separately for approval** — no model equations changed otherwise.

## Step 2 (implemented): exact-tie regression tests

`tests/test_grad_ties.py` (14 tests, green, `smoke`-marked) now evaluates each
tie/kink **exactly at** the tie and pins the subgradient `jax.grad` returns
there, using **one-sided** finite differences from both directions (never a
central difference) and asserting consistency with forward, backward, or their
balanced midpoint — flagging, rather than silently passing, any gradient that
matches none.

Convention discovered and documented (JAX 0.4+):
- `jnp.minimum` / `jnp.maximum` / `jnp.clip` return the **balanced (50/50)
  subgradient** at an exact tie — the *mean* of the two one-sided derivatives,
  matching neither individually. Valid, not a bug; pinned so a change is caught.
- `jnp.where(cond, a, b)` with an equality `cond` follows the **selected
  branch**, so its gradient equals exactly one one-sided derivative.

Coverage added, by hazard row:

| Row(s) | Test | Convention pinned |
|--------|------|-------------------|
| 7 | `f_temperature@Tmin/@Tmax` | `<=`/`>=` guard → out-of-range (flat) branch, grad 0 |
| 8 | `f_frost@boundary` | `+1e-8` keeps `fd==dim` marginally unclipped → unclipped slope |
| 12 | `compute_allocation_fraction@DBH0.1` | balanced clip subgradient at lower bound |
| 17 | `compute_canopy_cover@closure` | strict `<` → mature (flat) branch at closure age |
| 18 | `calculate_interception@LAImax` | `+1e-8` keeps `lai==LAImax` on ratio side |
| 22 | `calculate_base_conductance@LAIgcx` | `<=` → scaled branch at `lai==LAIgcx` |
| 1 | `co_limitation_min_at_tie` + `..._permutation_equivariant` | balanced min; elementwise, permutation-equivariant |
| 13 | `stress_mortality_min_at_tie` | balanced `min(mort_raw, N)` |
| 19 | `soil_water_min_at_tie` | balanced `min(ASW, total_demand)` |
| 14/15 | `self_thinning_should_thin_at_boundary` + `..._newton_gradient_is_smooth_when_active` | strict `>` excludes boundary; **Newton solver gradient smooth & FD-matched when active** |
| 9 | `nutrition_fNn_sentinel_is_a_design_discontinuity` | **design flag — see Step 3** |

**Row 14 (highest risk) is now cleared.** Forcing `should_thin` True routes the
gradient through all 5 unrolled Newton iterations; `mort_count` and `N_new`
gradients are finite and match central FD to `1e-3` at an active point — the
solver's `where`/`max` guards do not leak a wrong subgradient.

**Permutation-invariance / `max([1,2,3,3,2])` bug class: absent.** The
differentiated path contains no hand-built `max`/`argmax` reduction over a
vector — only `jnp.sum` (linear) and the elementwise `jnp.minimum`
co-limitation, which is verified permutation-equivariant (a tie splits 50/50 on
the tied species; permuting species permutes the gradient identically, never
pinning to index 0).

## Step 3: design decisions requiring explicit sign-off

**`f_nutrition` `fNn == 0` sentinel (row 9) — needs a decision.**
`f_N = where(fNn == 0.0, 1.0, 1 - (1 - fN0) * (1 - FR)**fNn)` overrides `f_N` to
**1.0** at exactly `fNn == 0`, whereas the analytic limit of the else-branch as
`fNn → 0` is `fN0` (0.6 at the test point). Therefore, at `fNn == 0`:
- the forward **value is discontinuous** (jumps ≈0.6 → 1.0), and
- the autodiff gradient wrt `fNn` is exactly **0**, matching **neither** one-sided
  derivative (both ≈ ±4e5, opposite sign).

This reads as an intentional "nutrition disabled" sentinel, but it is a genuine
value+gradient discontinuity. `test_nutrition_fNn_sentinel_is_a_design_discontinuity`
pins the current behaviour as a regression guard (no fix applied). **Action:**
confirm the sentinel is intended and ensure calibration never places `fNn` at
exactly 0 (it is a fixed parameter, `fNn = 1` in the harness, so this is a
latent hazard, not an active one).

No other gradient bug was surfaced; every remaining tie resolves to a valid
(balanced or selected-branch) subgradient, all now pinned by tests.
