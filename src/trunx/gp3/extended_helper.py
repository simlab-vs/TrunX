"""Extended helper functions for 3PG model to have learnable componenets."""

import string

import jax.nn
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
