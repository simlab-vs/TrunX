"""Exact-tie / exact-kink gradient regression tests for the 3-PG forward pass.

Step 2 companion to ``tests/test_grad_smoke.py`` (which only exercises a single
strictly-interior point) and to ``tests/GRADIENT_HAZARD_REVIEW.md`` (which
enumerates every ``min``/``max``/``clip``/``where`` in the differentiated path).
Where the smoke test keeps every operator one-sided *with margin*, this module
places the evaluation **exactly at** the tie/kink of each hazard and pins down
the subgradient ``jax.grad`` actually returns there.

Methodology (per hazard):

* Construct an input **at the exact tie** (``eta_w == eta_s``, ``fD == fSW``, an
  input exactly at a ``jnp.clip`` bound, ``lai == LAIgcx`` …) — not near it.
* Take the reverse-mode autodiff gradient at that point.
* Take **one-sided** finite differences from *both* directions
  (``(f(x+eps) - f(x)) / eps`` and ``(f(x) - f(x-eps)) / eps``) — never a central
  difference, which would average across the discontinuity and hide the bug.
* Assert the autodiff gradient is consistent with **forward**, **backward**, or
  their **balanced midpoint**, and record which — so any refactor that silently
  moves the convention fails the test. A gradient matching *none* of the three
  is flagged as a hard failure, not a silent pass.

Convention documented by these tests (empirically, JAX 0.4+):

* ``jnp.minimum`` / ``jnp.maximum`` / ``jnp.clip`` use a **balanced (50/50)
  subgradient** at an exact tie: the returned value is the *mean* of the two
  one-sided derivatives, matching neither individually. This is a valid
  subgradient, not a bug; the tests pin it so a change is caught.
* ``jnp.where(cond, a, b)`` with an equality ``cond`` follows the **selected
  branch** (the branch taken when the condition holds at equality), so its
  gradient equals exactly one one-sided derivative.

Float64 is enabled so the finite-difference comparison is numerically reliable.
"""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from tests.test_grad_smoke import _col, make_params, make_site, make_species  # noqa: E402
from trunx.gp3 import helper_function as H  # noqa: E402

pytestmark = pytest.mark.smoke

_EPS = 1e-6
_ATOL = 1e-4
_RTOL = 1e-3

PARAMS = make_params()
SITE = make_site()
SPECIES = make_species()


def _grad_sum(f, x: jnp.ndarray) -> float:
    """Reverse-mode gradient of ``sum(f(x))`` reduced to a scalar."""
    return float(jnp.sum(jax.grad(lambda z: jnp.sum(f(z)))(x)))


def _one_sided(f, x: jnp.ndarray, eps: float = _EPS) -> tuple[float, float]:
    """Forward and backward one-sided finite differences of ``sum(f(x))``."""
    fx = jnp.sum(f(x))
    fwd = float((jnp.sum(f(x + eps)) - fx) / eps)
    bwd = float((fx - jnp.sum(f(x - eps))) / eps)
    return fwd, bwd


def _close(a: float, b: float) -> bool:
    """Relative+absolute closeness robust to the finite-difference floor."""
    return abs(a - b) <= _ATOL + _RTOL * max(abs(a), abs(b))


def _assert_consistent(ad: float, fwd: float, bwd: float, name: str) -> set[str]:
    """Assert ``ad`` matches a one-sided derivative or their balanced midpoint."""
    sides = {"forward": fwd, "backward": bwd, "balanced-midpoint": (fwd + bwd) / 2.0}
    matched = {k for k, v in sides.items() if _close(ad, v)}
    assert matched, (
        f"{name}: autodiff gradient {ad:.6g} is consistent with NEITHER one-sided "
        f"derivative (forward={fwd:.6g}, backward={bwd:.6g}) NOR their balanced "
        f"midpoint {(fwd + bwd) / 2.0:.6g} -- silently-wrong subgradient at the tie."
    )
    return matched


# Each case pins one hazard at its exact tie/kink. ``expect`` is the convention
# the current code uses there; asserting membership makes the convention a
# regression guard, not just "some side matched".
#   id, callable(x)->Array, x_at_tie, expected convention
_TIE_CASES = [
    # row 7: f_temperature out-of-range guard uses ``<=``/``>=`` -> exactly at
    # Tmin/Tmax the point is treated as out-of-range (fT=0, flat side).
    ("f_temperature@Tmin", lambda T: H.f_temperature(PARAMS, T), _col(2.0), "backward"),
    ("f_temperature@Tmax", lambda T: H.f_temperature(PARAMS, T), _col(36.0), "forward"),
    # row 8: f_frost clip(1-kF*fd/dim, 0, 1). The +1e-8 in the denominator keeps
    # fd==dim marginally on the unclipped side, so grad == the unclipped slope.
    ("f_frost@boundary", lambda fd: H.f_frost(PARAMS, fd, _col(30.0)), _col(30.0), "backward"),
    # row 12: compute_allocation_fraction clip(DBH, 0.1, None)**pfsPower at the
    # lower clip bound -> balanced clip subgradient.
    (
        "compute_allocation_fraction@DBH0.1",
        lambda D: H.compute_allocation_fraction(SPECIES, PARAMS, _col(0.7), D)[0],
        _col(0.1),
        "balanced-midpoint",
    ),
    # row 17: compute_canopy_cover where(age_years < fullCanAge, young, 1). Strict
    # ``<`` -> exactly at closure age selects the mature (flat) branch.
    (
        "compute_canopy_cover@closure",
        lambda a: H.compute_canopy_cover(PARAMS, a),
        _col(240.0),
        "forward",
    ),
    # row 18: calculate_interception min(1, lai/LAImaxIntcptn). +1e-8 keeps
    # lai==LAImaxIntcptn marginally on the ratio (unclipped) side.
    (
        "calculate_interception@LAImax",
        lambda lai: H.calculate_interception(PARAMS, _col(100.0), lai, _col(1.0))[0],
        _col(5.0),
        "backward",
    ),
    # row 22: calculate_base_conductance where(lai <= LAIgcx, scaled, MaxCond).
    # ``<=`` -> exactly at LAIgcx selects the scaled branch.
    (
        "calculate_base_conductance@LAIgcx",
        lambda lai: H.calculate_base_conductance(PARAMS, lai),
        _col(3.33),
        "backward",
    ),
]


@pytest.mark.parametrize("name, fn, x_tie, expect", _TIE_CASES, ids=[c[0] for c in _TIE_CASES])
def test_scalar_tie_subgradient(name: str, fn, x_tie: jnp.ndarray, expect: str) -> None:
    """Autodiff subgradient at an exact tie matches the documented one-sided rule."""
    ad = _grad_sum(fn, x_tie)
    fwd, bwd = _one_sided(fn, x_tie)
    matched = _assert_consistent(ad, fwd, bwd, name)
    assert expect in matched, (
        f"{name}: convention changed -- autodiff {ad:.6g} now matches {sorted(matched)}, "
        f"expected the '{expect}' side (fwd={fwd:.6g}, bwd={bwd:.6g})."
    )


def test_co_limitation_min_at_tie() -> None:
    """phi = fA * min(fD, fSW): at fD == fSW JAX returns the balanced subgradient."""

    def phi(fD: jnp.ndarray) -> jnp.ndarray:
        return _col(0.7) * jnp.minimum(fD, _col(0.8))

    ad = _grad_sum(phi, _col(0.8))
    fwd, bwd = _one_sided(phi, _col(0.8))
    matched = _assert_consistent(ad, fwd, bwd, "co-limitation min(fD, fSW)")
    assert "balanced-midpoint" in matched, (
        f"min(fD, fSW) tie no longer balanced: ad={ad:.6g}, fwd={fwd:.6g}, bwd={bwd:.6g}"
    )


def test_co_limitation_min_is_permutation_equivariant() -> None:
    """Elementwise min over species is permutation-equivariant, not pinned to index 0.

    Guards against the ``max([1, 2, 3, 3, 2])`` bug class: permuting the species
    axis must permute the gradient the same way (and split a tie 50/50 on the
    tied species), never leave it stuck on the first occurrence.
    """
    fA = jnp.array([0.7, 0.6, 0.5])
    fSW = jnp.array([0.8, 0.9, 0.95])
    fD = jnp.array([0.8, 0.5, 0.99])  # species-0 exactly at the tie fD == fSW

    def phi_sum(fD_in: jnp.ndarray, fSW_in: jnp.ndarray, fA_in: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(fA_in * jnp.minimum(fD_in, fSW_in))

    g = jax.grad(phi_sum)(fD, fSW, fA)
    # tied species -> 0.5 split; fD<fSW -> full weight; fD>fSW -> zero.
    assert jnp.allclose(g, jnp.array([0.5 * 0.7, 0.6, 0.0])), f"unexpected tie gradient {g}"

    perm = jnp.array([2, 0, 1])
    g_perm = jax.grad(phi_sum)(fD[perm], fSW[perm], fA[perm])
    assert jnp.allclose(g_perm, g[perm]), (
        f"co-limitation min gradient is not permutation-equivariant: {g_perm} != {g[perm]}"
    )


def test_stress_mortality_min_at_tie() -> None:
    """apply_stress_mortality min(mort_stress_raw, N) at mort_raw == N is balanced.

    gammaN is pinned to a constant so ``mort_raw = gammaN*N/1200`` crosses ``N``
    transversally as gammaN0 varies; the tie is at gammaN0 == 1200.
    """
    p = PARAMS._replace(gammaN1=_col(0.0), tgammaN=_col(1e8), ngammaN=_col(1.0))

    def mort(g0: jnp.ndarray) -> jnp.ndarray:
        pp = p._replace(gammaN0=g0)
        return H.apply_stress_mortality(
            pp, _col(0.0), _col(10.0), _col(3.0), _col(4.0), _col(1500.0), jnp.asarray([False])
        )[4]

    ad = _grad_sum(mort, _col(1200.0))
    fwd, bwd = _one_sided(mort, _col(1200.0))
    matched = _assert_consistent(ad, fwd, bwd, "apply_stress_mortality min(mort_raw, N)")
    assert "balanced-midpoint" in matched


def test_soil_water_min_at_tie() -> None:
    """update_soil_water min(ASW, total_demand) at ASW == total_demand is balanced.

    After adding precip (0 here) the reservoir holds 100; total_demand == transp,
    so the evapo_transp ``min`` ties exactly when transp == 100.
    """

    def f_transp_scale(transp: jnp.ndarray) -> jnp.ndarray:
        return H.update_soil_water(SITE, _col(100.0), _col(0.0), transp, _col(0.0), _col(0.0))[1]

    ad = _grad_sum(f_transp_scale, _col(100.0))
    fwd, bwd = _one_sided(f_transp_scale, _col(100.0))
    matched = _assert_consistent(ad, fwd, bwd, "update_soil_water min(ASW, total_demand)")
    assert "balanced-midpoint" in matched


def test_self_thinning_should_thin_at_boundary() -> None:
    """apply_self_thinning: should_thin uses strict ``>`` -> no thinning exactly at
    biom_tree == wSmax_per_tree, so the mortality gradient is the non-thinning side.

    With N==1000, thinPower==1.5, wSx1000==200: biom_tree == WS and
    wSmax_per_tree == 200, so the boundary is exactly WS == 200.
    """

    def mort(WS: jnp.ndarray) -> jnp.ndarray:
        return H.apply_self_thinning_with_mortality_factors(
            PARAMS, WS, _col(3.0), _col(4.0), _col(1000.0), jnp.asarray([False])
        )[4]

    ad = _grad_sum(mort, _col(200.0))
    fwd, bwd = _one_sided(mort, _col(200.0))
    matched = _assert_consistent(ad, fwd, bwd, "self-thinning should_thin boundary")
    assert "backward" in matched, (
        f"should_thin no longer excludes the exact boundary: ad={ad:.6g}, "
        f"fwd={fwd:.6g}, bwd={bwd:.6g}"
    )


def test_self_thinning_newton_gradient_is_smooth_when_active() -> None:
    """Highest-risk zone (review row 14): with should_thin True, the gradient flows
    through the 5-iteration unrolled Newton solver. Its internal guards
    (``where(abs(dfN)<1e-8,..)``, ``where(n<=0, 1e-8)``, ``max(mort_n, 0)``) must
    not corrupt the gradient -- it should be smooth and match central FD.
    """

    def mort(WS: jnp.ndarray) -> jnp.ndarray:
        return H.apply_self_thinning_with_mortality_factors(
            PARAMS, WS, _col(3.0), _col(4.0), _col(1000.0), jnp.asarray([False])
        )[4]

    def stems(WS: jnp.ndarray) -> jnp.ndarray:
        return H.apply_self_thinning_with_mortality_factors(
            PARAMS, WS, _col(3.0), _col(4.0), _col(1000.0), jnp.asarray([False])
        )[3]

    ws_active = _col(260.0)  # strictly inside the thinning region (WS > 200)
    for label, fn in (("mort_count", mort), ("N_new", stems)):
        ad = _grad_sum(fn, ws_active)
        fwd, bwd = _one_sided(fn, ws_active)
        assert not (jnp.isnan(jnp.asarray(ad)) or jnp.isinf(jnp.asarray(ad))), (
            f"{label}: non-finite grad"
        )
        central = (fwd + bwd) / 2.0
        assert _close(ad, central), (
            f"self-thinning Newton {label} gradient {ad:.6g} disagrees with central FD "
            f"{central:.6g} (fwd={fwd:.6g}, bwd={bwd:.6g}) at an active point."
        )
        # A smooth interior point: both one-sided derivatives agree with autodiff.
        assert _close(ad, fwd) and _close(ad, bwd), (
            f"self-thinning Newton {label} is not smooth at WS=260: this point was "
            f"chosen to be a clean interior; a kink here means the solver leaks a subgradient."
        )


def test_nutrition_fNn_sentinel_is_a_design_discontinuity() -> None:
    """REGRESSION GUARD + DESIGN FLAG (review row 9).

    f_nutrition applies ``where(fNn == 0.0, 1.0, f_N)``. At exactly fNn == 0 the
    model *overrides* f_N to 1.0 (nutrition disabled), whereas the analytic limit
    of f_N as fNn -> 0 is fN0 (= 0.6 here). Consequences pinned here:

    * the forward VALUE is discontinuous at fNn == 0 (jumps ~0.6 -> 1.0), and
    * the autodiff gradient wrt fNn is exactly 0 there, matching NEITHER one-sided
      finite difference (both are ~±4e5 and opposite in sign).

    This is intentional sentinel behaviour, not a smooth-subgradient case, so it
    is asserted as-is rather than run through ``_assert_consistent``. Flagged in
    GRADIENT_HAZARD_REVIEW.md (Step 3) as requiring explicit sign-off; calibration
    must never place fNn at exactly 0.
    """

    def f_N(fNn: jnp.ndarray) -> jnp.ndarray:
        return H.f_nutrition(SPECIES, PARAMS._replace(fNn=fNn))

    value_at_zero = float(jnp.sum(f_N(_col(0.0))))
    assert value_at_zero == pytest.approx(1.0), (
        f"fNn==0 sentinel no longer forces f_N=1.0 (got {value_at_zero})."
    )

    ad = _grad_sum(f_N, _col(0.0))
    fwd, bwd = _one_sided(f_N, _col(0.0))
    assert ad == pytest.approx(0.0, abs=1e-9), f"fNn==0 gradient expected 0, got {ad:.6g}"
    # Document (not assert-pass) that autodiff disagrees with both one-sided
    # derivatives here -- the whole point of the flag.
    assert not _close(ad, fwd) and not _close(ad, bwd), (
        "fNn==0 discontinuity has vanished: autodiff now agrees with a one-sided "
        f"derivative (ad={ad:.6g}, fwd={fwd:.6g}, bwd={bwd:.6g}); re-review the sentinel."
    )
