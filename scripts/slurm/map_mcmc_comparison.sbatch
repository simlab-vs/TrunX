#!/bin/bash
# MAP+Laplace vs MCMC (NUTS) calibration comparison, on a few 3PG sites.
#
# MAP+Laplace results already exist under data/results/comparison_<site>/map/
# (from an earlier GPU vmap-batched run), so this job always passes --skip-map
# and only runs the MCMC (NUTS) phase. NUTS on this problem was benchmarked as
# CPU-faster than GPU (a 21-param site finished in ~28min on CPU vs. still not
# done after 50+min on GPU — NUTS only parallelizes ~4x across chains, so a
# GPU's batch-throughput design doesn't pay off the way it does for MAP's
# 2000-way vmapped restart search), so no --gres/GPU is requested and JAX is
# forced onto CPU explicitly.
#
# The two Dance nodes (chacha, disco) have independent, non-shared
# filesystems, so each site's data/results/comparison_<site>/map/ only exists
# on the node it was originally computed on. Submit one job per site with an
# explicit --nodelist matching that node, e.g.:
#
#   sbatch --nodelist=chacha --export=ALL,TRUNX_SITE_INDEX=1 scripts/slurm/map_mcmc_comparison.sbatch
#   sbatch --nodelist=chacha --export=ALL,TRUNX_SITE_INDEX=2 scripts/slurm/map_mcmc_comparison.sbatch
#   sbatch --nodelist=chacha --export=ALL,TRUNX_SITE_INDEX=3 scripts/slurm/map_mcmc_comparison.sbatch
#   sbatch --nodelist=disco  --export=ALL,TRUNX_SITE_INDEX=0 scripts/slurm/map_mcmc_comparison.sbatch
#
# (index -> site: 0=solling, 1=04.1605, 2=14.0003, 3=14.0012)
#
# Override the checkout with TRUNX_PROJECT_DIR:
#
#   sbatch --export=ALL,TRUNX_SITE_INDEX=0,TRUNX_PROJECT_DIR=/path/to/checkout ...

#SBATCH --job-name=trunx-map-mcmc
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=Dance
#SBATCH --qos=standby_qos
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=12:00:00
# Standby jobs are preempted (SIGTERM) whenever a normal/premium/standard QoS
# job needs the resources; --requeue puts the task back in the queue instead
# of it just dying. MCMC checkpoints (run_pymc_analysis writes to
# <output-dir>/mcmc/), so a requeue only redoes the sampling since the last
# checkpointed chunk, not the whole run.
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

# Force JAX onto CPU: benchmarked faster than GPU for NUTS on this workload (see
# header), and this also sidesteps GPU contention/standby-QoS GPU limits since
# no --gres is requested here.
export CUDA_VISIBLE_DEVICES=
export JAX_PLATFORM_NAME=cpu

PROJECT_DIR="${TRUNX_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT_DIR"
mkdir -p logs

# site_id -> input file, relative to data/threepg_inputs.
SITE_IDS=(solling 04.1605 14.0003 14.0012)
SITE_FILES=(full_solling_data.xlsx icp_plots/04.1605_data.xlsx icp_plots/14.0003_data.xlsx icp_plots/14.0012_data.xlsx)

SITE_INDEX="${SLURM_ARRAY_TASK_ID:-${TRUNX_SITE_INDEX:?Set TRUNX_SITE_INDEX (0-3) via --export, or submit as an array}}"
SITE_ID="${SITE_IDS[$SITE_INDEX]}"
SITE_FILE="${SITE_FILES[$SITE_INDEX]}"

echo "Job:       ${SLURM_JOB_ID:-interactive} on $(hostname)"
echo "Project:   $PROJECT_DIR"
echo "Site:      $SITE_ID ($SITE_FILE)"
echo

uv run --no-sync python scripts/run_comparison_site.py \
    --file-path "data/threepg_inputs/$SITE_FILE" \
    --output-dir "data/results/comparison_${SITE_ID}" \
    --skip-map \
    --step-method nuts \
    --target-accept 0.9 \
    --chains 4 \
    --num-warmup 1000 \
    --num-samples 1000
