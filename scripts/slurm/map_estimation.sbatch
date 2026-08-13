#!/bin/bash
# MAP calibration of the 3PG parameters on the full Solling stand.
#
# Refreshes the 3PG input files from the S3 bucket into the shared data root, then runs
# the MAP estimation against them. Files already current on the cluster are left alone,
# so a repeated submission only pays for what actually changed upstream.
#
# Submit from the project checkout:
#
#   sbatch scripts/slurm/map_estimation.sbatch
#
# Override the data root or the checkout with TRUNX_BASE_DIR / TRUNX_PROJECT_DIR:
#
#   sbatch --export=ALL,TRUNX_BASE_DIR=/scratch/$USER/trunx scripts/slurm/map_estimation.sbatch

#SBATCH --job-name=trunx-map
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
# Uncomment to run on a GPU node; the linux extras install jax[cuda12].
##SBATCH --gres=gpu:1

set -euo pipefail

PROJECT_DIR="${TRUNX_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"

# Shared data root on the cluster. trunx.config derives every data path from it, so the
# pull below and the MAP run agree on where the inputs live.
export TRUNX_BASE_DIR="${TRUNX_BASE_DIR:-$HOME/shared_sim_lab/trunx}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

cd "$PROJECT_DIR"
mkdir -p logs "$TRUNX_BASE_DIR"

echo "Job:       ${SLURM_JOB_ID:-interactive} on $(hostname)"
echo "Project:   $PROJECT_DIR"
echo "Data root: $TRUNX_BASE_DIR"
echo

echo "== Syncing 3PG inputs =="
uv run scripts/pull_data.py threepg_inputs --dest "$TRUNX_BASE_DIR"
echo

# Forrester et al. (2017) Table A.5 lives beside the checkout, not under the data root,
# because trunx.gp3.allometrics resolves it from project_root. It is gitignored, so the
# bucket is the only way onto a fresh cluster node.
echo "== Syncing literature supplements =="
uv run scripts/pull_data.py literature --dest "$PROJECT_DIR"
echo

# See _DEFAULT_XLSX in src/trunx/gp3/allometrics.py; create_data_inputs reads it at
# import time, so a missing file otherwise surfaces as a bare traceback mid-run.
FORRESTER_XLSX="$PROJECT_DIR/literature/1-s2.0-S0378112717301238-mmc1.xlsx"
if [[ ! -f "$FORRESTER_XLSX" ]]; then
    echo "ERROR: missing $FORRESTER_XLSX" >&2
    echo "Upload it once from a machine that has it, then re-submit:" >&2
    echo "  aws --profile simlab-zhw-a s3 cp <local copy> s3://trunx/literature/" >&2
    exit 1
fi

echo "== Running MAP estimation =="
uv run python -m trunx.gp3.bayesiancalibrations.map_param_est
