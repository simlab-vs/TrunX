#!/bin/bash
# MAP+Laplace calibration sweep on GPU: every (site, mode) combination, MAP method
# only (see scripts/run_map_sweep.py — kept separate from calibration_sweep_test.sbatch
# because that sweep's METHODS list currently excludes "map", and jobs already running
# index into it, so it can't be changed without reindexing those in-flight tasks).
#
# MAP's heavy lifting is `n_vmap_restarts` (2000 by default) independent L-BFGS
# restarts batched into one jax.vmap call (see map_param_est.batched_map_search) —
# thousands of restarts cost about the same wall-clock time as one, which is what
# makes this GPU-shaped: it's one big batched computation on a single device, not
# multiple chains/processes the way pymc_param_est's DEMetropolisZ/NUTS runs are.
# So, unlike calibration_sweep_test.sbatch, there is no per-chain CPU-affinity
# pinning to worry about here — just one process, one GPU.
#
# First, size the array to the current number of (site, mode) combinations:
#
#   uv run --no-sync python scripts/run_map_sweep.py --list-jobs
#
# Then submit one task per line printed above (N = that count - 1):
#
#   sbatch --array=0-N scripts/slurm/map_sweep_gpu.sbatch
#
# Override the checkout with TRUNX_PROJECT_DIR:
#
#   sbatch --array=0-N --export=ALL,TRUNX_PROJECT_DIR=/path/to/checkout ...

#SBATCH --job-name=trunx-map-sweep
#SBATCH --output=data/results/calibration_sweep/logs/%x-%A_%a.out
#SBATCH --error=data/results/calibration_sweep/logs/%x-%A_%a.err
#SBATCH --partition=Dance
#SBATCH --qos=standby_qos
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
# Standby jobs are preempted (SIGTERM) whenever a normal/premium/standard QoS job
# needs the resources. Unlike the DEMetropolisZ/NUTS sweep, MAP has no mid-run
# checkpoint, so --requeue restarts a preempted (site, mode) combination from
# scratch rather than resuming it — acceptable here since one combination's MAP
# run is minutes, not the days a DEMetropolisZ chain needs.
#SBATCH --requeue

set -euo pipefail

# The Dance partition spans chacha (uv via snap, already on PATH) and disco (uv
# installed to ~/.local/bin, per-node local home — not on a batch shell's default
# PATH). Harmless no-op on chacha.
export PATH="$HOME/.local/bin:$PATH"

# stdout isn't a tty once redirected to the log file, so progress bars and print()
# block-buffer instead of flushing per line — the log otherwise looks stalled for
# long stretches even while the job is actively computing.
export PYTHONUNBUFFERED=1

# Let JAX see the GPU --gres granted this job (contrast calibration_sweep_test.sbatch,
# which forces CPU). Don't preallocate most of the device's memory up front: this
# job doesn't need it (see batched_map_search's docstring — a few thousand restarts'
# worth of L-BFGS state, not a huge model), and holding it needlessly blocks other
# jobs sharing the same physical GPU (this partition also exposes `shard:...` GRES
# for fractional GPU sharing — switch --gres above to that if running many of these
# concurrently and 2000 vmap restarts don't need a whole H100/A100 to itself).
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

# Host-side data loading (pandas/polars/numpy) still runs on CPU before the GPU
# takes over for the vmapped search — same reasoning as calibration_sweep_test.sbatch,
# just mattering less here since this is one process, not `chains` of them sharing
# a cpuset.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export POLARS_MAX_THREADS=1
export RAYON_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export NUMBA_THREADING_LAYER=workqueue

PROJECT_DIR="${TRUNX_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT_DIR"
mkdir -p logs

JOB_INDEX="${SLURM_ARRAY_TASK_ID:?Submit as an array (see the header comment for sizing it)}"

echo "Job:       ${SLURM_JOB_ID:-interactive}, array task ${JOB_INDEX}, on $(hostname)"
echo "Project:   $PROJECT_DIR"
echo "GPU:       ${CUDA_VISIBLE_DEVICES:-$(nvidia-smi -L 2>/dev/null || echo unknown)}"
echo

# Run in the Apptainer container, binding the checkout and data directories into
# it so it can read/write them. `apptainer exec ... /app/.venv/bin/python3 <script>`
# rather than `apptainer run` on purpose (see calibration_sweep_test.sbatch for why:
# the image's %runscript's `uv run` re-scans the shared, bind-mounted .venv on every
# invocation and races with sibling array tasks starting at the same time) — with
# --nv added here to pass the GPU through into the container.
# Build trunkx-app.sif with `apptainer build trunkx-app.sif scripts/slurm/apptainer.def`.

srun --cpus-per-task="$SLURM_CPUS_PER_TASK" \
    apptainer exec --nv \
        --env OMP_NUM_THREADS=1 \
        --env MKL_NUM_THREADS=1 \
        --env OPENBLAS_NUM_THREADS=1 \
        --env NUMEXPR_NUM_THREADS=1 \
        --env POLARS_MAX_THREADS=1 \
        --env RAYON_NUM_THREADS=1 \
        --env NUMBA_NUM_THREADS=1 \
        --env NUMBA_THREADING_LAYER=workqueue \
        --env PYTHONPATH=/app/src \
        --bind .:/app \
        --bind ./src:/app/src \
        --bind ./scripts:/app/scripts \
        --bind ./tests:/app/tests \
        --bind ./data:/app/data \
        --bind /tmp:/tmp \
        --pwd /app \
        trunkx-app.sif \
        /app/.venv/bin/python3 \
        /app/scripts/run_map_sweep.py \
        --job-index "$JOB_INDEX" \
        --output-dir data/results/latent_calibration_sweep \
        --include-process-error
