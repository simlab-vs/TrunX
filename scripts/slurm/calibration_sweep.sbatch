#!/bin/bash
# Calibration sweep: DEMetropolisZ, NUTS, gradient descent, and MAP, on every site
# and error mode, one job per (site, mode, method) combination.
#
# Each combination is independent (see scripts/run_calibration_sweep.py), so this
# submits the whole sweep as a SLURM array instead of one long sequential job.
#
# First, size the array to the current number of combinations:
#
#   uv run --no-sync python scripts/run_calibration_sweep.py --list-jobs
#
# Then submit one task per line printed above (N = that count - 1):
#
#   sbatch --array=0-N scripts/slurm/calibration_sweep.sbatch
#
# Override the checkout with TRUNX_PROJECT_DIR:
#
#   sbatch --array=0-N --export=ALL,TRUNX_PROJECT_DIR=/path/to/checkout ...
#
# The DEMetropolisZ jobs (1e6 warmup + 5e6 samples, see run_calibration_sweep.py)
# dominate the wall-clock time this --time limit needs to cover; NUTS/MAP/gradient
# descent tasks in the same array finish in a small fraction of it.

#SBATCH --job-name=trunx-calibration-sweep
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --partition=Dance
#SBATCH --qos=standby_qos
#SBATCH --cpus-per-task=5
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
# Standby jobs are preempted (SIGTERM) whenever a normal/premium/standard QoS job
# needs the resources; --requeue puts the task back in the queue instead of it
# just dying. The PyMC methods checkpoint (run_pymc_analysis writes to
# <output-dir>/.../{demetropolisz,nuts}/), so a requeue only redoes the sampling
# since the last checkpointed chunk, not the whole run.
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

# Force JAX onto CPU: no --gres is requested here, and NUTS on this workload was
# benchmarked CPU-faster than GPU (see scripts/slurm/map_mcmc_comparison.sbatch).
export CUDA_VISIBLE_DEVICES=
export JAX_PLATFORM_NAME=cpu

PROJECT_DIR="${TRUNX_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT_DIR"
mkdir -p logs

JOB_INDEX="${SLURM_ARRAY_TASK_ID:?Submit as an array (see the header comment for sizing it)}"

echo "Job:       ${SLURM_JOB_ID:-interactive}, array task ${JOB_INDEX}, on $(hostname)"
echo "Project:   $PROJECT_DIR"
echo

# Run the calibration sweep script in the Apptainer container, binding the
# checkout and data directories into the container so it can read/write them.
# Build trunkx-app.sif with `apptainer build trunkx-app.sif scripts/slurm/apptainer.def`.

apptainer run \
    --bind .:/app \
    --bind ./src:/app/src \
    --bind ./scripts:/app/scripts \
    --bind ./tests:/app/tests \
    --bind ./data:/app/data \
    --bind /tmp:/tmp \
    --pwd /app \
    trunkx-app.sif \
    python scripts/run_calibration_sweep.py \
    --job-index "$JOB_INDEX" \
    --output-dir "data/results/calibration_sweep" \
    --chains 4 \
    --demetropolisz-num-warmup 50 \
    --demetropolisz-num-samples 50 \
    --nuts-num-warmup 50 \
    --nuts-num-samples 50 \
    --n-vmap-restarts 10 \
    --n-vmap-steps 10 \
    --laplace-draws 10
