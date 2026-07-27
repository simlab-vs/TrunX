"""Gradient smoke test for the differentiable 3-PG forward pass.

This is a permanent regression gate against *silently wrong gradients* — the
failure mode where the forward pass is numerically correct but ``jax.grad``
returns ``NaN``/``Inf`` or a value that disagrees with finite differences
because the autodiff path crosses a kink of a ``jnp.minimum`` / ``jnp.maximum``
/ ``jnp.clip`` / ``jnp.where`` operator, or because a function silently
changes behaviour under ``jax.jit`` (tracer concretisation). Forward-value unit
tests cannot catch any of these.

Chosen evaluation point (see :func:`make_params` / :func:`make_climate` etc.)
and why it is strictly interior and non-degenerate — i.e. no co-limitation term
or ``min``/``max`` sits at or near a tie:

* **Single evergreen species** with ``leafgrow == leaffall == 0`` so
  :func:`is_dormant` is identically ``False``. This removes every
  dormancy / leaf-debt branch switch from the traced path.
* **Temperature** ``T_avg = 12`` lies strictly inside ``(Tmin=2, Topt=16,
  Tmax=36)`` and away from ``Topt``, so ``f_temperature`` is on its smooth
  interior (``0 < fT < 1``, unclipped, non-zero ``dfT/dT``).
* **Co-limitation** ``phi = fAge * min(fD, fSW)``: ``fD = exp(-0.15 * 2.0) =
  0.741`` while the site is deliberately well-watered (``soil_class = 0``
  selects the ``SWconst = 999`` sentinel and the soil holds a huge reservoir),
  giving ``fSW == 1.0`` at every step. The ``min`` therefore resolves to ``fD``
  with a fixed ~0.259 margin at every timestep and is never at a tie.
* **Soil-water balance**: the large reservoir (``ASW ~ 5e5``, ``ASW_max =
  1e6``) keeps every ``min``/``max`` clip in the water balance strictly
  one-sided — transpiration is strictly demand-limited, ``ASW`` stays strictly
  interior, and ``f_transp_scale == 1`` — so there is no water-balance kink.
* **Young, low-density stand** (``WS = 10 t/ha``, ``N = 1500/ha``) keeps
  self-thinning off (biomass/tree far below ``wSmax``/tree) and age-stress
  mortality strictly interior (``0 < mort << N``): the mortality masks never
  sit at their switch points.
* **Small canopy**: ``LAI`` stays below ``LAIgcx`` and ``LAImaxIntcptn`` and the
  stand age (4-6 yr over the run) stays below ``fullCanAge = 20 yr``, so the
  conductance, rainfall-interception and canopy-cover branches stay on one side.
* ``frost_days = 2 / 30`` gives ``fF = 0.933`` (unclipped) and ``co2 = 385``
  keeps the CO2 modifiers unclipped.

Differentiated variables:

* **Parameters**: a curated set of 13 continuous physiological / allocation
  parameters that have clean, non-zero sensitivity at this point.
* **Forcing**: the continuous climate channels. ``month`` and ``n_days`` are
  structural indices and are held fixed. (``T_max`` and ``precip`` have
  ~zero sensitivity in this well-watered, conductance-inert regime; they are
  kept for completeness and are covered by ``check_grads``' absolute tolerance.)

Float64 is enabled so the finite-difference comparison is numerically reliable.
"""

import jax
import jax.numpy as jnp
import pytest
from jax.test_util import check_grads

from trunx.gp3.model_inputs import ClimateData, Params, SiteData, SpeciesData, State
from trunx.gp3.run_3pg import run_3pg

# Reliable central finite differences need float64; set before any array is created.
jax.config.update("jax_enable_x64", True)

pytestmark = pytest.mark.smoke

N_STEPS = 24


def _col(x: float) -> jnp.ndarray:
    """Wrap a scalar as a one-species float64 column."""
    return jnp.asarray([x], dtype=jnp.float64)


def make_params() -> Params:
    """Realistic single-species 3-PG parameter set at a non-degenerate point."""
    return Params(
        pFS2=_col(1.0),
        pFS20=_col(0.15),
        aWS=_col(0.1),
        nWS=_col(2.4),
        pRx=_col(0.4),
        pRn=_col(0.25),
        gammaF1=_col(0.03),
        gammaF0=_col(0.001),
        tgammaF=_col(24.0),
        gammaR=_col(0.015),
        leafgrow=_col(0.0),
        leaffall=_col(0.0),
        Tmin=_col(2.0),
        Topt=_col(16.0),
        Tmax=_col(36.0),
        kF=_col(1.0),
        SWconst=_col(0.7),
        SWpower=_col(9.0),
        fCalpha700=_col(1.4),
        fCg700=_col(0.7),
        m0=_col(0.5),
        fN0=_col(0.6),
        fNn=_col(1.0),
        MaxAge=_col(150.0),
        nAge=_col(4.0),
        rAge=_col(0.95),
        gammaN1=_col(0.02),
        gammaN0=_col(0.2),
        tgammaN=_col(60.0),
        ngammaN=_col(1.0),
        wSx1000=_col(200.0),
        thinPower=_col(1.5),
        mF=_col(0.5),
        mR=_col(0.2),
        mS=_col(0.2),
        SLA0=_col(4.0),
        SLA1=_col(10.0),
        tSLA=_col(2.0),
        k=_col(0.5),
        fullCanAge=_col(20.0),
        MaxIntcptn=_col(0.15),
        LAImaxIntcptn=_col(5.0),
        cVPD=_col(5.0),
        alphaCx=_col(0.06),
        Y=_col(0.47),
        MinCond=_col(0.0),
        MaxCond=_col(0.02),
        LAIgcx=_col(3.33),
        CoeffCond=_col(0.15),
        BLcond=_col(0.2),
        RGcGw=_col(0.66),
        D13CTissueDif=_col(2.0),
        aFracDiffu=_col(4.4),
        bFracRubi=_col(27.0),
        fracBB0=_col(0.15),
        fracBB1=_col(0.05),
        tBB=_col(2.0),
        rhoMin=_col(0.4),
        rhoMax=_col(0.5),
        tRho=_col(4.0),
        crownshape=_col(2.0),
        aH=_col(1.5),
        nHB=_col(0.5),
        nHC=_col(0.1),
        aV=_col(0.1),
        nVB=_col(2.0),
        nVH=_col(1.0),
        nVBH=_col(0.0),
        aK=_col(0.5),
        nKB=_col(0.5),
        nKH=_col(0.1),
        nKC=_col(0.1),
        nKrh=_col(0.1),
        aHL=_col(0.5),
        nHLB=_col(0.5),
        nHLL=_col(0.1),
        nHLC=_col(0.1),
        nHLrh=_col(0.1),
        Qa=_col(-90.0),
        Qb=_col(0.8),
        gDM_mol=_col(24.0),
        molPAR_MJ=_col(2.3),
    )


def make_site() -> SiteData:
    """Well-watered site: huge soil reservoir keeps water-balance clips one-sided."""
    return SiteData(
        latitude=_col(47.0),
        altitude=_col(400.0),
        soil_class=_col(0.0),
        ASW=_col(500000.0),
        ASW_max=_col(1000000.0),
        ASW_min=_col(200.0),
        year_i=jnp.asarray([2000]),
        month_i=jnp.asarray([1]),
    )


def make_species() -> SpeciesData:
    """Single species at intermediate fertility."""
    return SpeciesData(
        specie=jnp.asarray([1]),
        FR=_col(0.5),
        WF=_col(3.0),
        WR=_col(4.0),
        WS=_col(10.0),
        N=_col(1500.0),
        year_p=jnp.asarray([1996]),
        month_p=jnp.asarray([1]),
    )


def make_state() -> State:
    """Young, low-density stand keeping mortality/thinning masks strictly off."""
    return State(
        WF=_col(3.0),
        WR=_col(4.0),
        WS=_col(10.0),
        N=_col(1500.0),
        ASW=_col(500000.0),
        age=_col(48.0),
        WF_debt=_col(0.0),
        prev_month=jnp.asarray([12], dtype=jnp.int32),
    )


def make_climate(n: int = N_STEPS) -> ClimateData:
    """Short, mild, growing-season forcing series with no clip saturation."""
    o = jnp.ones(n, dtype=jnp.float64)
    months = (jnp.arange(n, dtype=jnp.float64) % 12) + 1
    return ClimateData(
        T_avg=12.0 * o,
        T_max=18.0 * o,
        VPD=2.0 * o,
        precip=100.0 * o,
        solar_rad=15.0 * o,
        frost_days=2.0 * o,
        n_days=30.0 * o,
        co2=385.0 * o,
        d13catm=-8.0 * o,
        month=months,
    )


# Parameters differentiated in the gradient checks: continuous, with clean
# non-zero sensitivity at the chosen interior point.
PARAM_NAMES = [
    "alphaCx",
    "CoeffCond",
    "Y",
    "k",
    "pFS2",
    "pFS20",
    "pRx",
    "pRn",
    "aWS",
    "nWS",
    "gammaR",
    "fN0",
    "Topt",
]
# Continuous forcing channels differentiated in the gradient checks.
FORCING_NAMES = ["T_avg", "T_max", "VPD", "solar_rad", "frost_days", "co2", "precip"]
# Output pools reduced into the scalar loss (touch biomass pools and production).
REDUCE_VARS = ["WS", "WF", "WR", "GPP", "LAI"]


def _baseline_outputs():
    """Run the forward pass once at the baseline point."""
    _, outputs = run_3pg(make_state(), make_climate(), make_params(), make_site(), make_species())
    return outputs


def _make_loss_fns():
    """Build scalar losses of (params) and (forcing) around fixed synthetic targets."""
    outputs = _baseline_outputs()
    targets = {k: 0.9 * outputs[k] for k in REDUCE_VARS}

    def scalar_loss(o) -> jnp.ndarray:
        return sum((jnp.sum((o[k] - targets[k]) ** 2) for k in REDUCE_VARS), jnp.asarray(0.0))

    state0, climate, params = make_state(), make_climate(), make_params()
    site, species = make_site(), make_species()

    def loss_params(theta: jnp.ndarray) -> jnp.ndarray:
        upd = {name: jnp.reshape(theta[i], (1,)) for i, name in enumerate(PARAM_NAMES)}
        _, o = run_3pg(state0, climate, params._replace(**upd), site, species)
        return scalar_loss(o)

    def loss_forcing(*arrs: jnp.ndarray) -> jnp.ndarray:
        upd = dict(zip(FORCING_NAMES, arrs, strict=True))
        _, o = run_3pg(state0, climate._replace(**upd), params, site, species)
        return scalar_loss(o)

    theta0 = jnp.array([getattr(params, name)[0] for name in PARAM_NAMES])
    forcing0 = tuple(getattr(climate, name) for name in FORCING_NAMES)
    return loss_params, theta0, loss_forcing, forcing0


def test_gradients_are_finite() -> None:
    """grad wrt params and wrt forcing contains no NaN/Inf."""
    loss_params, theta0, loss_forcing, forcing0 = _make_loss_fns()

    g_params = jax.grad(loss_params)(theta0)
    assert not jnp.isnan(g_params).any(), "NaN in parameter gradient"
    assert not jnp.isinf(g_params).any(), "Inf in parameter gradient"

    g_forcing = jax.grad(loss_forcing, argnums=tuple(range(len(FORCING_NAMES))))(*forcing0)
    for name, g in zip(FORCING_NAMES, g_forcing, strict=True):
        assert not jnp.isnan(g).any(), f"NaN in forcing gradient for {name}"
        assert not jnp.isinf(g).any(), f"Inf in forcing gradient for {name}"


def test_gradients_match_finite_differences() -> None:
    """Reverse-mode grad matches central finite differences at the interior point."""
    loss_params, theta0, loss_forcing, forcing0 = _make_loss_fns()
    check_grads(loss_params, (theta0,), order=1, modes=("rev",), atol=1e-3, rtol=1e-3)
    check_grads(loss_forcing, forcing0, order=1, modes=("rev",), atol=1e-3, rtol=1e-3)


def test_jit_matches_unjit() -> None:
    """value_and_grad is identical jitted and un-jitted (catches tracer bugs)."""
    loss_params, theta0, loss_forcing, forcing0 = _make_loss_fns()

    for fn, args in ((loss_params, (theta0,)), (loss_forcing, forcing0)):
        v, g = jax.value_and_grad(fn, argnums=tuple(range(len(args))))(*args)
        vj, gj = jax.jit(jax.value_and_grad(fn, argnums=tuple(range(len(args)))))(*args)
        assert jnp.allclose(v, vj, rtol=1e-10, atol=1e-10), "loss differs under jit"
        for gi, gji in zip(g, gj, strict=True):
            assert jnp.allclose(gi, gji, rtol=1e-8, atol=1e-8), "grad differs under jit"


def test_point_is_non_degenerate() -> None:
    """The evaluation point sits strictly interior: no min/max/clip at a tie."""
    outputs = _baseline_outputs()

    for name, value in outputs.items():
        assert not jnp.isnan(value).any(), f"NaN in output {name}"
        assert not jnp.isinf(value).any(), f"Inf in output {name}"

    fD, fSW = outputs["fD"].ravel(), outputs["fSW"].ravel()
    assert jnp.all(fD < fSW), "co-limitation min(fD, fSW) not resolved by fD at every step"
    assert jnp.min(jnp.abs(fD - fSW)) > 0.05, "min(fD, fSW) arguments are near a tie"

    for name in ("fT", "fF", "fN", "f_cg"):
        v = outputs[name].ravel()
        assert jnp.all((v > 0.0) & (v < 1.0)), f"{name} is clip-saturated (at a kink)"

    assert jnp.max(outputs["mort_thinn"]) == 0.0, "self-thinning is active (mask at a tie)"
    stress = outputs["mort_stress"].ravel()
    assert jnp.all((stress > 0.0) & (stress < outputs["N"].ravel())), (
        "stress mortality not interior"
    )
