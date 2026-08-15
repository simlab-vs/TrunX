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
