# TODO

## Model DBH/BA/Height from the stand's size distribution, not a single mean tree

`compute_dbh` (`src/trunx/gp3/helper_function.py`) derives DBH from a single
stand-level "mean tree": it inverts `aWS`/`nWS` on the mean stem biomass per
tree (`DBH = (WS_per_tree / aWS) ** (1 / nWS)`). `BA` and `Height` are then
computed from that same DBH (`run_3pg.py`).

The ICP observations for all three are instead built by applying the field
allometric equations per tree and *summing* over the stand's actual DBH
distribution (`create_data_inputs.py`, `allometrics.py`) — observed `DBH`
itself is the quadratic mean diameter (QMD) computed straight from the raw
per-tree measurements, independent of biomass.

These two aggregations aren't inverses of each other whenever a stand has
real size spread: summing a convex power function over a tree-size
distribution and then inverting at the mean is not the same operation as
inverting per tree and averaging (Jensen's inequality). Concretely,
`f⁻¹(mean(f(dbh_i)))` (what the model computes) is the order-`nWS` power mean
of the stand, not the QMD and not the arithmetic mean — for `nWS > 2` (true
here) it's structurally an upper bound on QMD given any spread. As a stopgap,
`err_DBH`/`err_BA`/`err_Height` have been excluded from calibration (see
`bayesian_config.DIAGNOSTIC_ONLY_ERROR_NAMES`) — these three are simulated
and can still be plotted, but no longer pull the optimizer away from a
correct WS/WF/WR fit.

The proper fix is to give 3PG a real notion of stand size *inequality*
instead of a single mean tree — e.g. a Weibull-shaped diameter distribution
(3-PGmix style), whose shape parameter evolves with self-thinning, so
`compute_dbh` can report a distribution-aware, bias-corrected DBH/QMD instead
of the current single-tree inversion. This is a real model extension (new
state variable + dynamics), not a small patch — scope it separately before
picking it up.

## `wSx1000`/`thinPower` are structurally unidentified at low-density stands

Comparing MAP+Laplace against full NUTS on the three ICP sites
(`04.1605`/`14.0003`/`14.0012`) showed `wSx1000`/`thinPower` pinned at
whatever value each method's optimiser/sampler happened to land on, with
NUTS's posterior spanning essentially the whole prior range. Root cause,
confirmed directly from the observed WS/N record: these two parameters only
enter the model through the self-thinning rule (`should_thin = biom_tree >
wSmax_per_tree`, `helper_function.py`), and it never triggers at any of the
three sites — `biom_tree` stays at 2-11% of `wSmax` across the whole observed
stand-density range. With `err_DBH`/`err_BA`/`err_Height` excluded from the
likelihood (see above), nothing left in the calibration data can distinguish
one `wSx1000`/`thinPower` value from another there, so the likelihood is
exactly flat along both directions. This isn't a Laplace-approximation bug —
`fit_laplace` (`map_uncertainty.py`) already detects the zero curvature
correctly and holds the parameter at its MAP value with a warning — but that
warning doesn't reach the saved outputs, so a MAP+Laplace run consumed on its
own (no NUTS to cross-check against) has no way to tell a genuinely
constrained parameter from a silently pinned one.

Options, not yet picked up (each needs a comparison rerun to validate):

- Fix `wSx1000`/`thinPower` at their literature/default values instead of
  fitting them, for sites where a pre-flight check (the same `biom_tree` vs.
  `wSmax` margin, evaluated at the prior default) shows self-thinning never
  triggers over the observed record — there's nothing in the data to fit
  them against.
- If self-thinning matters for out-of-sample extrapolation into denser
  future stand states, use literature-informed narrow priors for them
  instead of the current wide uniforms, the way `Tmax`/`MaxAge` already get
  species-derived bounds (`load_files.literature_bound_overrides`) — though
  note the comment above on `thinPower`: published sources disagree by up to
  2x, so "literature-informed" is less clear-cut here than for `Tmax`/`MaxAge`.
- Surface `fit_laplace`'s `dropped` (held-at-MAP) parameter list into the
  saved output (e.g. a field alongside `map_estimate.json`), so a MAP+Laplace
  run is self-diagnosing about which parameters it actually estimated versus
  silently pinned, without needing a parallel NUTS run to find out.
