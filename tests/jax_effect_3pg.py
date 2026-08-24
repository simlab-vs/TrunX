"""Benchmark `run_3pg`: R (CPU, parallel) vs JAX vmap (CPU/GPU).

Runs the same comparison at several problem scales (see `_SCALES`) to show how
each implementation's wall time and memory footprint change as the number of
passes grows. For each scale we collect:
  - wall time for a single, non-parallelizable pass
  - wall time for many passes (parallel across CPU cores for R, vmapped for
    JAX)
  - peak/mean process memory (RSS) while running the many-passes case

One-time costs (JAX JIT compilation, R package loading and data reading) are
kept out of the timed regions on both sides, since including them would not
reflect the steady-state cost these implementations are actually chosen for.

"""

import gc
import os
import statistics
import subprocess
import threading
import time
import warnings
from typing import NamedTuple

import jax
import jax.numpy as jnp
import polars as pl
import psutil

from trunx.config import project_root, threepg_data_folder
from trunx.gp3.PG3_model_impl import prepare_data
from trunx.gp3.run_3pg import run_3pg as run_3pg_orig

warnings.filterwarnings("ignore")

_SCALES = [1, 1_000, 1_0_000]
_R_SCRIPT = os.path.join(project_root, "models", "r3PG", "benchmark_run_3pg.R")


class MemorySampler:
    """Poll a process's RSS on a background thread to get peak/mean memory."""

    def __init__(self, pid: int | None = None, interval: float = 0.02) -> None:
        self._process = psutil.Process(pid)
        self._interval = interval
        self._samples: list[int] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._samples.append(self._process.memory_info().rss)
            except psutil.NoSuchProcess:
                break
            time.sleep(self._interval)

    def __enter__(self) -> "MemorySampler":
        self._thread.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)

    @property
    def peak_mb(self) -> float:
        """Peak RSS observed during the sampled window, in MB."""
        return max(self._samples) / 1e6 if self._samples else float("nan")

    @property
    def mean_mb(self) -> float:
        """Mean RSS observed during the sampled window, in MB."""
        return statistics.fmean(self._samples) / 1e6 if self._samples else float("nan")


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


def perturb_inputs(
    params, climate, site, species, initial_state, key: jax.Array, scale: float = 0.02
):
    """Create one perturbed set of float inputs for benchmarking stability."""
    params_p, key = _perturb_float_arrays(params, key, scale)
    climate_p, key = _perturb_float_arrays(climate, key, scale * 0.2)
    site_p, key = _perturb_float_arrays(site, key, scale * 0.2)
    species_p, key = _perturb_float_arrays(species, key, scale * 0.2)
    state_p, key = _perturb_float_arrays(initial_state, key, scale * 0.2)
    return params_p, climate_p, site_p, species_p, state_p


class ScaleResult(NamedTuple):
    """Benchmark results for one problem scale (number of passes)."""

    scale: int
    vmap_exec: float
    vmap_peak_mb: float
    vmap_mean_mb: float
    r_total: float | None
    r_peak_mb: float
    r_mean_mb: float


def build_perturbed_inputs(
    params, climate, site_data, species_data, initial_state, n_runs, seed: int = 42
):
    """Vmap `perturb_inputs` over `n_runs` independent keys to build a batch of perturbed inputs.

    Replaces a Python-level loop of `n_runs` unjitted calls (each doing many
    tiny per-field `jax.random` dispatches) with a single jitted, vectorized
    computation.

    Returns
    -------
    tuple of 5 pytrees
        (params_batch, climate_batch, site_batch, species_batch, state_batch),
        each leaf carrying a leading `n_runs` batch dimension.
    """
    keys = jax.random.split(jax.random.PRNGKey(seed), n_runs)
    batched_perturb = jax.jit(
        jax.vmap(perturb_inputs, in_axes=(None, None, None, None, None, 0, None))
    )

    start = time.perf_counter()
    batch = batched_perturb(params, climate, site_data, species_data, initial_state, keys, 0.02)
    _wait_for_outputs(batch)
    elapsed = time.perf_counter() - start
    print(f"  build_perturbed_inputs ({n_runs} runs) compile+exec: {elapsed:.4f}s")

    return batch


def run_jax_vmap(perturbed_batch):
    """Vmap the batch of perturbed inputs; returns (compile_seconds, exec_seconds, mem)."""
    params_batch, climate_batch, site_batch, species_batch, state_batch = perturbed_batch

    time.sleep(5)

    run_3pg = jax.jit(run_3pg_orig)
    run_3pg_vmapped = jax.vmap(run_3pg)

    start_compile = time.perf_counter()
    _, outputs_vmap = run_3pg_vmapped(
        state_batch, climate_batch, params_batch, site_batch, species_batch
    )
    _wait_for_outputs(outputs_vmap)
    compile_seconds = time.perf_counter() - start_compile

    with MemorySampler() as mem:
        start_exec = time.perf_counter()
        _, outputs_vmap = run_3pg_vmapped(
            state_batch, climate_batch, params_batch, site_batch, species_batch
        )
        _wait_for_outputs(outputs_vmap)
        exec_seconds = time.perf_counter() - start_exec

    return compile_seconds, exec_seconds, mem


def run_r_benchmark(file_path: str, n_runs: int):
    """Run the R benchmark script.

    Returns (single_seconds, total_seconds, mem), or None on failure.
    """
    try:
        r_proc = subprocess.Popen(
            ["Rscript", _R_SCRIPT, file_path, str(n_runs)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        with MemorySampler(pid=r_proc.pid) as mem:
            r_stdout, _ = r_proc.communicate()

        r_results = {}
        for line in r_stdout.splitlines():
            if line.startswith("RESULT_"):
                key, value = line.split(maxsplit=1)
                r_results[key] = value

        single_seconds = float(r_results["RESULT_SINGLE_SECONDS"])
        total_seconds = float(r_results["RESULT_TOTAL_SECONDS"])
        return single_seconds, total_seconds, mem
    except (FileNotFoundError, KeyError, subprocess.SubprocessError) as exc:
        print(f"  R benchmark unavailable ({exc}); skipping.")
        return None


file_path = os.path.join(threepg_data_folder, "solling_data.xlsx")
input_data = prepare_data(file_path)

print(f"JAX backend: {jax.default_backend()} (devices: {jax.devices()})")

print("JIT-compiling run_3pg...")
run_3pg = jax.jit(run_3pg_orig)
start_compile = time.perf_counter()
_, outputs = run_3pg(
    input_data.initial_state,
    input_data.climate,
    input_data.params,
    input_data.site,
    input_data.species,
)
_wait_for_outputs(outputs)
compile_time = time.perf_counter() - start_compile
print(f"JIT-compiled run_3pg (first-run/compile+exec time: {compile_time:.4f}s)")

time.sleep(5)

results = []
for scale in _SCALES:
    print(f"\n--- Scale: {scale} pass(es) ---")
    perturbed_inputs = build_perturbed_inputs(
        input_data.params,
        input_data.climate,
        input_data.site,
        input_data.species,
        input_data.initial_state,
        scale,
    )

    vmap_compile, vmap_exec, vmap_mem = run_jax_vmap(perturbed_inputs)
    print(f"  JAX vmap compile+exec: {vmap_compile:.4f}s, exec only: {vmap_exec:.4f}s")

    print(f"  Running R benchmark ({scale} runs)...")
    r_result = run_r_benchmark(file_path, scale)

    if r_result is not None:
        r_single, r_total, r_mem = r_result
        print(f"  R single-pass: {r_single:.4f}s, total ({scale} runs): {r_total:.4f}s")
    else:
        r_single = r_total = None
        r_mem = None

    results.append(
        ScaleResult(
            scale=scale,
            vmap_exec=vmap_exec,
            vmap_peak_mb=vmap_mem.peak_mb,
            vmap_mean_mb=vmap_mem.mean_mb,
            r_total=r_total,
            r_peak_mb=r_mem.peak_mb if r_mem is not None else float("nan"),
            r_mean_mb=r_mem.mean_mb if r_mem is not None else float("nan"),
        )
    )

    # Release this scale's batched arrays and compiled programs before the
    # next (larger) scale runs, so peak/mean RSS reflects each scale on its
    # own instead of accumulating across the whole run.
    del perturbed_inputs
    jax.clear_caches()
    gc.collect()

summary_rows = []
for r in results:
    for name, total, peak, mean in [
        ("R (parallel)", r.r_total, r.r_peak_mb, r.r_mean_mb),
        ("JAX vmap", r.vmap_exec, r.vmap_peak_mb, r.vmap_mean_mb),
    ]:
        summary_rows.append(
            {
                "scale": r.scale,
                "implementation": name,
                "total_s": total,
                "per_call_s": total / r.scale if total is not None else None,
                "peak_rss_mb": peak,
                "mean_rss_mb": mean,
                "speedup_vs_r": (
                    r.r_total / total if r.r_total is not None and total is not None else None
                ),
            }
        )

summary_df = pl.DataFrame(summary_rows)
print("\nBenchmark summary")
print(summary_df)
