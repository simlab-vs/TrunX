"""Extended helper functions for 3PG model to have learnable componenets."""

import string
from collections.abc import Callable
from functools import cache
from typing import Any, Literal, NamedTuple

import jax
import jax.numpy as jnp
import polars as pl

from trunx.gp3.prepare_deposition import load_monthly_deposition

INPUT_VARIABLES = ("N", "S", "T_avg")
# Deposition modifier inputs and their `DepositionData` field: monthly N and S deposition,
# and their means and sums over the last 12 months
DEPOSITION_CHANNELS = {
    "N": "dep_n_tot",
    "S": "dep_s_so4",
    "N_avg": "avg_dep_n_tot",
    "S_avg": "avg_dep_s_so4",
    "N_sum": "sum_dep_n_tot",
    "S_sum": "sum_dep_s_so4",
}
# Percentile of all ICP plot-months each deposition input is clipped to at the top, so rare
# extreme months (e.g. canopy or pollen events) don't drive the fit, see `deposition_limits`
DEPOSITION_UPPER_QUANTILE = 0.99
# Form of the curvature `a_k ≥ 0` given by the raw curvature weights, see `_curvature`
CURVATURE_FORM: Literal["softplus", "squared"] = "softplus"
# Shift of the raw curvature weights of the "softplus" form, so all-zero weights (the
# neutral start, and where the L2 penalty pulls) give almost no curvature: softplus(-5) ≈ 0.007
CURVATURE_SHIFT = -5.0
# Starting raw curvature weight of each form: `raw**2` has no gradient at 0, so the
# "squared" start is slightly off zero, giving almost no curvature: 0.1**2 = 0.01
CURVATURE_INIT = {"softplus": 0.0, "squared": 0.1}[CURVATURE_FORM]
# Temperature of the smooth minimum of `liebig_nm`
LIEBIG_TAU = 0.02


@cache
def deposition_limits() -> dict[str, tuple[float, float]]:
    """Range each deposition input is clipped to before entering the modifier.

    From 0 to the `DEPOSITION_UPPER_QUANTILE` of all ICP plot-months in
    `icp_monthly_deposition.parquet`, rounded to two decimals.

    Returns
    -------
    dict[str, tuple[float, float]]
        (min, max) of each `DEPOSITION_CHANNELS` key: kg ha⁻¹ month⁻¹ for the monthly
        values and means, kg ha⁻¹ per 12 months for the sums.
    """
    upper = load_monthly_deposition().select(
        pl.col(column).quantile(DEPOSITION_UPPER_QUANTILE).alias(name)
        for name, column in DEPOSITION_CHANNELS.items()
    )
    return {name: (0.0, round(upper[name].item(), 2)) for name in DEPOSITION_CHANNELS}


def prepare_modifier_inputs(
    channels: dict[str, jnp.ndarray],
    input_vars: tuple[str, ...],
    input_mean: jnp.ndarray | float = 0.0,
    input_std: jnp.ndarray | float = 1.0,
) -> jnp.ndarray:
    """Stack the modifier inputs, clip each deposition to its `deposition_limits` and standardise.

    Parameters
    ----------
    channels : dict[str, jnp.ndarray]
        Monthly values of each input, keyed by name (`"T_avg"` or a `DEPOSITION_CHANNELS`
        key).
    input_vars : tuple[str, ...]
        Inputs to stack, in this order.
    input_mean, input_std : jnp.ndarray | float
        Mean and standard deviation of each input, one per `input_vars` item; the
        defaults leave the inputs unstandardised.

    Returns
    -------
    jnp.ndarray
        Prepared inputs, with `input_vars` along the last axis.
    """
    inputs = jnp.stack(
        [
            jnp.clip(channels[name], *deposition_limits()[name])
            if name in DEPOSITION_CHANNELS
            else jnp.asarray(channels[name])
            for name in input_vars
        ],
        axis=-1,
    )
    return (inputs - input_mean) / input_std


def evaluate_modifier(
    modifier_fn: Callable[[Any, jnp.ndarray, tuple[str, ...]], jnp.ndarray],
    modifier_params: Any,
    inputs: jnp.ndarray,
    input_vars: tuple[str, ...],
) -> jnp.ndarray:
    """Evaluate the nutrition modifier, neutral (1) in months with a missing input.

    Missing (NaN) inputs are replaced by 0 before `modifier_fn` runs, so no NaN reaches
    the modifier or its gradients; their months' result is then set to 1.

    Parameters
    ----------
    modifier_fn : Callable[[Any, jnp.ndarray, tuple[str, ...]], jnp.ndarray]
        Nutrition modifier, e.g. `poly_nm` or `mlp_nm`.
    modifier_params : Any
        Parameters of `modifier_fn`.
    inputs : jnp.ndarray
        Prepared inputs, see `prepare_modifier_inputs`.
    input_vars : tuple[str, ...]
        Inputs along the last axis of `inputs`, in order.

    Returns
    -------
    jnp.ndarray
        Modifier value of each month.
    """
    missing = jnp.isnan(inputs)
    modifier = modifier_fn(modifier_params, jnp.where(missing, 0.0, inputs), input_vars)
    return jnp.where(missing.any(axis=-1), 1.0, modifier)


def _squash(poly: jnp.ndarray) -> jnp.ndarray:
    """Map a raw score to (0, 2), centered at 1 (no effect) when `poly == 0`."""
    return 1.0 + jnp.tanh(poly)


def _channels(inputs: jnp.ndarray, input_vars: tuple[str, ...]) -> dict[str, jnp.ndarray]:
    """Split `inputs`'s last axis into a `{variable_name: channel}` mapping."""
    return {name: inputs[..., i] for i, name in enumerate(input_vars)}


def _cross_term_poly(
    channels: dict[str, jnp.ndarray], input_vars: tuple[str, ...], poly_params: jnp.ndarray
) -> jnp.ndarray:
    """Evaluate a full cross-term polynomial: one degree axis per selected variable.

    E.g. for `input_vars = ("N", "S")` and `poly_params.shape = (2, 2)`:
        poly = w[0,0] + w[1,0]*N + w[0,1]*S + w[1,1]*N*S
    """
    if poly_params.ndim != len(input_vars):
        raise ValueError(
            f"poly_params has {poly_params.ndim} axes but input_vars has "
            f"{len(input_vars)} entries {input_vars!r} — one degree axis per variable is required"
        )
    letters = string.ascii_lowercase[: len(input_vars)]
    powers = [
        channels[name][..., None] ** jnp.arange(degree)
        for name, degree in zip(input_vars, poly_params.shape, strict=True)
    ]
    operands = ",".join(f"...{letter}" for letter in letters)
    return jnp.einsum(f"{operands},{letters}->...", *powers, poly_params)


def poly_nm(
    poly_params: jnp.ndarray, inputs: jnp.ndarray, input_vars: tuple[str, ...] = INPUT_VARIABLES
) -> jnp.ndarray:
    """
    Polynomial nutrition modifier function.

    Full cross-term polynomial over `input_vars` (see `_cross_term_poly`), then
    squashed into (0, 2), centered at 1 (see `_squash`).
    """
    channels = _channels(inputs, input_vars)
    poly = _cross_term_poly(channels, input_vars, poly_params)
    return _squash(poly)


def constant_nm(
    params: jnp.ndarray, inputs: jnp.ndarray, input_vars: tuple[str, ...] = INPUT_VARIABLES
) -> jnp.ndarray:
    """Constant nutrition modifier, `1 + tanh(w)` in every month, ignoring the inputs.

    Baseline for `poly_nm`: its single weight plays the role of `poly_nm`'s `w00`.
    """
    return _squash(jnp.broadcast_to(params[0], inputs.shape[:-1]))


def _curvature(raw: jnp.ndarray) -> jnp.ndarray:
    """Non-negative curvature of `raw`, in the `CURVATURE_FORM` form.

    "softplus": `softplus(raw + CURVATURE_SHIFT)`, almost 0 when `raw == 0`;
    "squared": `raw**2`, 0 when `raw == 0`.
    """
    if CURVATURE_FORM == "squared":
        return raw**2
    return jax.nn.softplus(raw + CURVATURE_SHIFT)


def per_nutrient_nm(params: jnp.ndarray, inputs: jnp.ndarray) -> jnp.ndarray:
    """Evaluate the modifier of each input, `f_k = 1 + tanh(b_k + w_k z_k − ½ a_k z_k²)`.

    Parameters
    ----------
    params : jnp.ndarray
        `(b_k, w_k, raw a_k)` of each input, shape `(n_inputs, 3)`; `a_k` is
        `_curvature(raw a_k)`.
    inputs : jnp.ndarray
        z-scored inputs `z_k` along the last axis.

    Returns
    -------
    jnp.ndarray
        `f_k` of each input along the last axis.
    """
    b, w, a = params[:, 0], params[:, 1], _curvature(params[:, 2])
    return _squash(b + w * inputs - 0.5 * a * inputs**2)


def product_nm(
    params: jnp.ndarray, inputs: jnp.ndarray, input_vars: tuple[str, ...] = INPUT_VARIABLES
) -> jnp.ndarray:
    """Product of the per-input modifiers, `f = Π_k f_k` (see `per_nutrient_nm`)."""
    return per_nutrient_nm(params, inputs).prod(axis=-1)


def liebig_nm(
    params: jnp.ndarray, inputs: jnp.ndarray, input_vars: tuple[str, ...] = INPUT_VARIABLES
) -> jnp.ndarray:
    """Liebig's minimum of the per-input modifiers (see `per_nutrient_nm`).

    Smooth minimum `−τ log Σ_k exp(−f_k / τ)`, with `τ = LIEBIG_TAU`, so the gradient
    reaches every input.
    """
    f = per_nutrient_nm(params, inputs)
    return -LIEBIG_TAU * jax.nn.logsumexp(-f / LIEBIG_TAU, axis=-1)


def bivariate_quadratic_nm(
    params: jnp.ndarray, inputs: jnp.ndarray, input_vars: tuple[str, ...] = INPUT_VARIABLES
) -> jnp.ndarray:
    """Bivariate quadratic modifier, `f = 1 + tanh(b + wᵀz − ½ zᵀAz)` over two inputs.

    `params` is `(b, w_1, w_2, raw a_1, raw a_2, raw ρ)`, with `a_k = _curvature(raw a_k)`
    and `ρ = tanh(raw ρ)`, so `A = [[a_1, −ρ√(a_1 a_2)], [−ρ√(a_1 a_2), a_2]]` is positive
    semidefinite; `ρ > 0` means co-limitation, `ρ < 0` substitutable inputs.
    """
    if len(input_vars) != 2:
        raise ValueError(f"bivariate_quadratic_nm needs two inputs, got {input_vars!r}")
    b, w = params[0], params[1:3]
    a = _curvature(params[3:5])
    rho = jnp.tanh(params[5])
    z1, z2 = inputs[..., 0], inputs[..., 1]
    # zᵀAz = a_1 z_1² + a_2 z_2² + 2 A_12 z_1 z_2, with A_12 = −ρ√(a_1 a_2)
    quad = a[0] * z1**2 + a[1] * z2**2 - 2.0 * rho * jnp.sqrt(a[0] * a[1]) * z1 * z2
    return _squash(b + inputs @ w - 0.5 * quad)


def init_modifier_params(input_vars: tuple[str, ...] = INPUT_VARIABLES) -> jnp.ndarray:
    """Build a neutral (all-zero) starting point for `poly_nm`.

    Parameters
    ----------
    input_vars : tuple[str, ...]
        Which of `"T_avg"` and the `DEPOSITION_CHANNELS` keys the modifier is built
        over, and in what order — must match what's passed to `run_3pg`.
    """
    return jnp.zeros(tuple(2 for _ in input_vars))


class MLPLayer(NamedTuple):
    """One dense layer's weights: `x @ w + b`."""

    w: jnp.ndarray  # (fan_in, fan_out)
    b: jnp.ndarray  # (fan_out,)


class MLPModifierParams(NamedTuple):
    """Weights for a multi-hidden-layer MLP nutrition modifier.

    `layers[:-1]` are tanh-activated hidden layers; `layers[-1]` is the
    linear output layer (`fan_out=1`) — see `mlp_nm`/`init_mlp_modifier_params`.
    """

    layers: tuple[MLPLayer, ...]


def init_mlp_modifier_params(
    key: jax.Array,
    input_vars: tuple[str, ...] = INPUT_VARIABLES,
    hidden_sizes: tuple[int, ...] = (6,),
    init_scale: float = 0.1,
) -> MLPModifierParams:
    """Build a neutral starting point for `mlp_nm`."""
    sizes = (len(input_vars), *hidden_sizes, 1)
    n_layers = len(sizes) - 1
    keys = jax.random.split(key, n_layers)
    layers = []
    for i in range(n_layers):
        fan_in, fan_out = sizes[i], sizes[i + 1]
        is_output_layer = i == n_layers - 1
        w = (
            jnp.zeros((fan_in, fan_out))
            if is_output_layer
            else init_scale * jax.random.normal(keys[i], (fan_in, fan_out))
        )
        layers.append(MLPLayer(w=w, b=jnp.zeros(fan_out)))
    return MLPModifierParams(layers=tuple(layers))


def mlp_nm(
    mlp_params: MLPModifierParams,
    inputs: jnp.ndarray,
    input_vars: tuple[str, ...] = INPUT_VARIABLES,
) -> jnp.ndarray:
    """Multi-hidden-layer MLP nutrition modifier function."""
    x = inputs
    *hidden_layers, output_layer = mlp_params.layers
    for layer in hidden_layers:
        x = jnp.tanh(x @ layer.w + layer.b)
    out = (x @ output_layer.w + output_layer.b).squeeze(-1)
    return _squash(out)
