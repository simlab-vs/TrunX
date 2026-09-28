"""Extended helper functions for 3PG model to have learnable componenets."""

import string
from typing import NamedTuple

import jax
import jax.numpy as jnp

INPUT_VARIABLES = ("N", "S", "T_avg")


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


def init_modifier_params(input_vars: tuple[str, ...] = INPUT_VARIABLES) -> jnp.ndarray:
    """Build a neutral (all-zero) starting point for `poly_nm`.

    Parameters
    ----------
    input_vars : tuple[str, ...]
        Which of `("N", "S", "T_avg")` the modifier is built over, and in what
        order — must match what's passed to `run_3pg`.
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
