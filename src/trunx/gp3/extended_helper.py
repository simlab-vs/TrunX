"""Extended helper functions for 3PG model to have learnable componenets."""

import jax.nn
import jax.numpy as jnp


def poly_nm(poly_params, inputs):
    """
    Polynomial nutrition modifier function.

    Bilinear-plus-cross-term polynomial:
        poly = w[0,0] + w[1,0]*N + w[0,1]*S + w[1,1]*N*S
    then squashed through sigmoid into (0, 1). N, S can be scalars or
    batched arrays of the same shape.
    """
    N = inputs[..., 0]
    S = inputs[..., 1]
    deg_N, deg_S = poly_params.shape
    N_powers = N[..., None] ** jnp.arange(deg_N)
    S_powers = S[..., None] ** jnp.arange(deg_S)
    poly = jnp.einsum("...i,...j,ij->...", N_powers, S_powers, poly_params)
    return jax.nn.sigmoid(poly)
