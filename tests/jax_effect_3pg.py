"""Benchmark `run_3pg` with synchronized JAX timing and perturbed inputs."""

import os
import time

import jax
import jax.numpy as jnp

from trunx.config import threepg_data_folder
from trunx.gp3.PG3_model_impl import prepare_data
from trunx.gp3.run_3pg import run_3pg as run_3pg_orig


def _wait_for_outputs(outputs) -> None:
    """Block until JAX finishes computing all output arrays."""
    jax.tree_util.tree_map(
        lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
        outputs,
    )


def _perturb_float_arrays(tree, key: jax.Array, scale: float):
    """Perturb only floating-point arrays while keeping shape and dtype unchanged."""
    data = dict(tree._asdict())
    for name, value in data.items():
        if hasattr(value, "dtype") and jnp.issubdtype(value.dtype, jnp.floating):
            key, subkey = jax.random.split(key)
            noise = jax.random.uniform(
                subkey,
                shape=value.shape,
                minval=-scale,
                maxval=scale,
                dtype=value.dtype,
            )
            one = jnp.asarray(1.0, dtype=value.dtype)
            data[name] = value * (one + noise)
    return tree._replace(**data), key


def perturb_inputs(params, climate, site, species, initial_state, scale=0.02, seed=42):
    """Create one perturbed set of float inputs for benchmarking stability."""
    key = jax.random.PRNGKey(seed)
    params_p, key = _perturb_float_arrays(params, key, scale)
    climate_p, key = _perturb_float_arrays(climate, key, scale * 0.2)
    site_p, key = _perturb_float_arrays(site, key, scale * 0.2)
    species_p, key = _perturb_float_arrays(species, key, scale * 0.2)
    state_p, key = _perturb_float_arrays(initial_state, key, scale * 0.2)
    return params_p, climate_p, site_p, species_p, state_p


file_path = os.path.join(threepg_data_folder, "solling_data.xlsx")
initial_state, climate, params, site_data, species_data, n_species, _ = prepare_data(file_path)

print("JIT-compiling run_3pg...")
run_3pg = jax.jit(run_3pg_orig)
print("JIT-compiled run_3pg")

# First run: compile + execute (synchronized timing)
start_compile = time.perf_counter()
_, outputs = run_3pg(initial_state, climate, params, site_data, species_data)
_wait_for_outputs(outputs)
compile_time = time.perf_counter() - start_compile
print(f"First-run/Compile+exec time: {compile_time:.4f}s")

PERTURB_INPUTS = []
for num in range(100):
    perturbed_inputs = perturb_inputs(
        params,
        climate,
        site_data,
        species_data,
        initial_state,
        scale=0.02,
        seed=42 + num,
    )
    PERTURB_INPUTS.append(perturbed_inputs)

time.sleep(5)

for num in range(100):
    params_p, climate_p, site_p, species_p, state_p = PERTURB_INPUTS[num]
    start_exec = time.perf_counter()
    _, outputs = run_3pg(state_p, climate_p, params_p, site_p, species_p)
    _wait_for_outputs(outputs)
    exec_time = time.perf_counter() - start_exec

    if num % 10 == 0:
        print(f"Execution time at simulation {num}: {exec_time:.4f}s")
