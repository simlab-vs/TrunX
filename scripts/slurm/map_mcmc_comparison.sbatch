#!/bin/bash
# MAP+Laplace vs MCMC (DEMetropolisZ) calibration comparison, on a few 3PG sites.
#
# One array task per site, each requesting its own GPU for the vmap-batched MAP
# search (see map_param_est.batched_map_search) and then running MCMC. Assumes
# data/threepg_inputs and literature/ are already present under the project data
# root (synced separately, e.g. by rsync — this job does not pull from S3).
#
# Submit from the project checkout:
#
#   sbatch scripts/slurm/map_mcmc_comparison.sbatch
#
# Override the checkout with TRUNX_PROJECT_DIR:
#
#   sbatch --export=ALL,TRUNX_PROJECT_DIR=/path/to/checkout scripts/slurm/map_mcmc_comparison.sbatch

#SBATCH --job-name=trunx-map-mcmc
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --partition=Dance
#SBATCH --qos=standby_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
# This account is capped at 1 GPU at a time (see standby QoS docs), so tasks run
# one at a time (%1) rather than across the partition's 2 nodes concurrently.
#SBATCH --array=0-3%1
# Standby jobs are preempted (SIGTERM) whenever a normal/premium/standard QoS job
# needs the resources; --requeue puts the task back in the queue instead of it
# just dying. MCMC checkpoints (run_pymc_analysis writes to <output-dir>/mcmc/),
# so a requeue only redoes the (fast) MAP+Laplace half of a restarted task.
#SBATCH --requeue

set -euo pipefail

# The Dance partition spans chacha (uv via snap, already on PATH) and disco (uv
# installed to ~/.local/bin, per-node local home — not on a batch shell's default
# PATH). Harmless no-op on chacha.
export PATH="$HOME/.local/bin:$PATH"

PROJECT_DIR="${TRUNX_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT_DIR"
mkdir -p logs

# site_id -> input file, relative to data/threepg_inputs.
SITE_IDS=(solling 04.1605 14.0003 14.0012)
SITE_FILES=(full_solling_data.xlsx icp_plots/04.1605_data.xlsx icp_plots/14.0003_data.xlsx icp_plots/14.0012_data.xlsx)

SITE_ID="${SITE_IDS[$SLURM_ARRAY_TASK_ID]}"
SITE_FILE="${SITE_FILES[$SLURM_ARRAY_TASK_ID]}"

echo "Job:       ${SLURM_JOB_ID:-interactive}[${SLURM_ARRAY_TASK_ID:-0}] on $(hostname)"
echo "Project:   $PROJECT_DIR"
echo "Site:      $SITE_ID ($SITE_FILE)"
echo

uv run --no-sync python scripts/run_comparison_site.py \
    --file-path "data/threepg_inputs/$SITE_FILE" \
    --output-dir "data/results/comparison_${SITE_ID}" \
    --n-vmap-restarts 2000 \
    --n-vmap-steps 200 \
    --laplace-draws 1000 \
    --chains 3 \
    --num-warmup 10000 \
    --num-samples 5000
