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


class MLPModifierParams(NamedTuple):
    """Weights for a single-hidden-layer MLP nutrition modifier."""

    w1: jnp.ndarray  # (n_input_vars, hidden_size)
    b1: jnp.ndarray  # (hidden_size,)
    w2: jnp.ndarray  # (hidden_size,)
    b2: jnp.ndarray  # scalar


def init_mlp_modifier_params(
    key: jax.Array,
    input_vars: tuple[str, ...] = INPUT_VARIABLES,
    hidden_size: int = 8,
    init_scale: float = 0.1,
) -> MLPModifierParams:
    """Build a neutral starting point for `mlp_nm`.

    `w1`/`b1` are randomly initialized (small scale) so hidden units aren't
    symmetric; `w2`/`b2` are zeroed so the network's output — and so its
    effect via `_squash` — starts at exactly 1 (no effect), matching
    `poly_nm`'s own neutral start (`init_modifier_params`'s all-zero grid).
    Zeroing `w1` too, the way `poly_nm`'s parameters are zeroed, would be a
    bug here: with `w2` also zero, gradients could never reach `w1` (the
    chain rule multiplies through `w2`), so the hidden layer could never
    learn. `poly_nm` has no such hidden bottleneck layer, so it doesn't share
    this failure mode.
    """
    w1 = init_scale * jax.random.normal(key, (len(input_vars), hidden_size))
    b1 = jnp.zeros(hidden_size)
    w2 = jnp.zeros(hidden_size)
    b2 = jnp.zeros(())
    return MLPModifierParams(w1=w1, b1=b1, w2=w2, b2=b2)


def mlp_nm(
    mlp_params: MLPModifierParams,
    inputs: jnp.ndarray,
    input_vars: tuple[str, ...] = INPUT_VARIABLES,
) -> jnp.ndarray:
    """
    Single-hidden-layer MLP nutrition modifier function.

    `inputs`'s last axis already holds one channel per `input_vars`, in
    order (see `_channels`), so it's fed straight into the first linear
    layer. Squashed into (0, 2), centered at 1, same convention as `poly_nm`
    (see `_squash`) — a drop-in alternative `modifier_fn` for `run_3pg`.
    """
    del input_vars  # part of the shared modifier_fn signature; unused here
    h = jnp.tanh(inputs @ mlp_params.w1 + mlp_params.b1)
    out = h @ mlp_params.w2 + mlp_params.b2
    return _squash(out)
