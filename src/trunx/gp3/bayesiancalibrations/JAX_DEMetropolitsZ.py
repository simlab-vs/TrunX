"""JAX port of PyMC's DEMetropolisZ sampler, keeping the reference implementation's variable names.

This is claude generated generated implementation, needs to be verified thoroughly against
the original PyMC implementation, and is not yet used in TrunX.

"""  # noqa: E501

import functools
import math
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


def tune(scale: jnp.ndarray, acc_rate: jnp.ndarray) -> jnp.ndarray:
    """
    Tune the scaling parameter for the proposal distribution.

    Uses the acceptance rate over the last tune_interval.

    Rate    Variance adaptation
    ----    -------------------
    <0.001        x 0.1
    <0.05         x 0.5
    <0.2          x 0.9
    >0.5          x 1.1
    >0.75         x 2
    >0.95         x 10
    """
    return scale * jnp.where(
        acc_rate < 0.001,
        0.1,
        jnp.where(
            acc_rate < 0.05,
            0.5,
            jnp.where(
                acc_rate < 0.2,
                0.9,
                jnp.where(
                    acc_rate > 0.95,
                    10.0,
                    jnp.where(acc_rate > 0.75, 2.0, jnp.where(acc_rate > 0.5, 1.1, 1.0)),
                ),
            ),
        ),
    )


class _ScanCarry(NamedTuple):
    """Scan carry for the `jax.lax.scan`-compiled DEMetropolisZ chain."""

    q0d: jnp.ndarray
    q0d_logp: jnp.ndarray
    history: jnp.ndarray
    history_len: jnp.ndarray
    history_start: jnp.ndarray
    scaling: jnp.ndarray
    interval_accepted: jnp.ndarray
    key: jax.Array


def _scan_step(
    carry: _ScanCarry,
    _: None,
    *,
    logp_fn: Any,
    lamb: float,
    do_tune: bool,
    tune_interval: int,
) -> tuple[_ScanCarry, tuple[jnp.ndarray, jnp.ndarray]]:
    """One DEMetropolisZ iteration, written to be traced once by `jax.lax.scan`.

    Same algorithm as `DEMetropolisZ.astep`, but with the Python-level
    `_history` list/`while` loop replaced by a fixed-size buffer and
    branch-free index arithmetic, so the whole per-step loop compiles to a
    single XLA program instead of being dispatched step by step in Python.
    """
    key, key_iz1, key_iz2, key_eps, key_accept = jax.random.split(carry.key, 5)
    dim = carry.q0d.shape[0]

    it = carry.history_len - carry.history_start
    iz1 = jax.random.randint(key_iz1, (), carry.history_start, carry.history_len)
    iz2_upper = jnp.maximum(carry.history_start + 1, carry.history_len - 1)
    iz2_raw = jax.random.randint(key_iz2, (), carry.history_start, iz2_upper)
    iz2 = jnp.where(iz2_raw >= iz1, iz2_raw + 1, iz2_raw)

    z1 = carry.history[iz1]
    z2 = carry.history[iz2]
    # use the DE-MCMC-Z proposal scheme as soon as the history has 2 entries,
    # otherwise propose just with noise (dx=0), matching PyMC's own `it > 1` guard
    dx = jnp.where(it > 1, z1 - z2, 0.0) * lamb
    epsilon = jax.random.normal(key_eps, (dim,)) * carry.scaling
    q = carry.q0d + dx + epsilon

    q_logp = logp_fn(q)
    accept = q_logp - carry.q0d_logp
    accepted = jnp.log(jax.random.uniform(key_accept)) < accept

    q_new = jnp.where(accepted, q, carry.q0d)
    q_new_logp = jnp.where(accepted, q_logp, carry.q0d_logp)

    new_history = carry.history.at[carry.history_len].set(q_new)
    new_history_len = carry.history_len + 1
    new_interval_accepted = carry.interval_accepted + accepted.astype(jnp.int32)

    new_scaling = carry.scaling
    if do_tune:
        is_boundary = jnp.equal(jnp.mod(new_history_len - carry.history_start, tune_interval), 0)
        acc_rate = new_interval_accepted / tune_interval
        new_scaling = jnp.where(is_boundary, tune(carry.scaling, acc_rate), carry.scaling)
        new_interval_accepted = jnp.where(is_boundary, 0, new_interval_accepted)

    new_carry = carry._replace(
        q0d=q_new,
        q0d_logp=q_new_logp,
        history=new_history,
        history_len=new_history_len,
        scaling=new_scaling,
        interval_accepted=new_interval_accepted,
        key=key,
    )
    return new_carry, (q_new, accepted)


def run_demetropolisz_scan(
    logp_fn: Any,
    initial_values: jnp.ndarray,
    num_warmup: int,
    num_samples: int,
    chains: int = 1,
    lamb: float | None = None,
    scaling: float = 0.001,
    tune_interval: int = 100,
    tune_drop_fraction: float = 0.9,
    seed: int = 0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Run one or more DEMetropolisZ chains, fully compiled via `jax.lax.scan`.

    Same algorithm as the `DEMetropolisZ` class (ter Braak & Vrugt, 2008), but
    the whole per-step loop is compiled once instead of being dispatched step
    by step in Python, and multiple chains run in parallel via `jax.vmap`.
    Use this entry point when speed matters; use the `DEMetropolisZ` class
    when you need to inspect or interact with the sampler step by step.

    Parameters
    ----------
    logp_fn : callable
        JAX-differentiable function mapping a parameter vector to its log-density.
    initial_values : array
        Starting position, shared by every chain.
    chains : int
        Number of independent chains, run in parallel via `jax.vmap`.
    seed : int
        PRNG seed; each chain gets an independent key split from it.

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray]
        `(draws, accepted)`, shaped `(chains, num_samples, dim)` and
        `(chains, num_samples)` respectively.
    """
    dim = initial_values.shape[0]
    if lamb is None:
        lamb = 2.38 / math.sqrt(2.0 * dim)

    def run_one_chain(key: jax.Array) -> tuple[jnp.ndarray, jnp.ndarray]:
        buffer_size = num_warmup + num_samples
        carry = _ScanCarry(
            q0d=initial_values,
            q0d_logp=logp_fn(initial_values),
            history=jnp.zeros((buffer_size, dim)),
            history_len=jnp.array(0, dtype=jnp.int32),
            history_start=jnp.array(0, dtype=jnp.int32),
            scaling=jnp.array(scaling, dtype=jnp.float64),
            interval_accepted=jnp.array(0, dtype=jnp.int32),
            key=key,
        )

        tune_step = functools.partial(
            _scan_step, logp_fn=logp_fn, lamb=lamb, do_tune=True, tune_interval=tune_interval
        )
        carry, _ = jax.lax.scan(tune_step, carry, xs=None, length=num_warmup)

        # Drop the first `tune_drop_fraction` of the tuning-phase history so
        # future proposals aren't informed by unconverged early positions
        # (matches `DEMetropolisZ.stop_tuning`).
        it = carry.history_len - carry.history_start
        n_drop = jnp.floor(it.astype(jnp.float64) * tune_drop_fraction).astype(jnp.int32)
        carry = carry._replace(history_start=carry.history_start + n_drop)

        draw_step = functools.partial(
            _scan_step, logp_fn=logp_fn, lamb=lamb, do_tune=False, tune_interval=tune_interval
        )
        carry, (draws, accepted) = jax.lax.scan(draw_step, carry, xs=None, length=num_samples)
        return draws, accepted

    keys = jax.random.split(jax.random.PRNGKey(seed), chains)
    draws, accepted = jax.jit(jax.vmap(run_one_chain))(keys)
    return draws, accepted
