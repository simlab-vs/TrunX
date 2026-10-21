#!/bin/bash
# Leave-one-plot-out cross-validation of the nutrition modifier: one task per
# (literature source, species, L2 penalty), see scripts/run_loocv_sweep.py.
#
# First, select the plots, build their 3PG input files and write the job list
# (data/results/nutrition_modifier/loocv_sweep/jobs_<inputs>_<selection>_<modifier>.json). Run this once, on the login node
# or locally before syncing data/, as the array tasks only read the input files:
#
#   uv run --no-sync python scripts/run_loocv_sweep.py --prepare
#
# It prints the jobs and the array size; submit one task per job (N = count - 1):
#
#   mkdir -p data/results/nutrition_modifier/loocv_sweep/logs
#   sbatch --array=0-N scripts/slurm/loocv_sweep.sbatch
#
# Fit the modifier over other inputs with INPUT_VARS (default "N S"), which must match
# the --input-vars given to --prepare:
#
#   uv run --no-sync python scripts/run_loocv_sweep.py --prepare --input-vars N
#   sbatch --array=0-N --export=ALL,INPUT_VARS=N scripts/slurm/loocv_sweep.sbatch
#
# Likewise MODIFIER (default poly_nm) must match the --modifier given to --prepare, e.g.
# mlp_nm:
#
#   uv run --no-sync python scripts/run_loocv_sweep.py --prepare --modifier mlp_nm
#   sbatch --array=0-N --export=ALL,MODIFIER=mlp_nm scripts/slurm/loocv_sweep.sbatch
#
# And PLOT_SELECTION (default poorly_fitted) the --plot-selection, e.g. all_plots to keep
# the plots fitted well without the modifier:
#
#   uv run --no-sync python scripts/run_loocv_sweep.py --prepare --plot-selection all_plots
#   sbatch --array=0-N --export=ALL,PLOT_SELECTION=all_plots scripts/slurm/loocv_sweep.sbatch
#
# Override the checkout with TRUNX_PROJECT_DIR:
#
#   sbatch --array=0-N --export=ALL,TRUNX_PROJECT_DIR=/path/to/checkout ...
#
# Results: data/results/nutrition_modifier/<source>/<inputs>/loocv/<selection>/<modifier>/
# <species>/l2=<value>/
# (loocv.csv, loocv_summary.csv, job.json, images/). A preempted task is requeued and
# restarts its cross-validation from the beginning (no checkpointing).
#
# Once all tasks are done, select the L2 penalty of each source and species
# (.../loocv/<modifier>/<species>/l2_selection.csv):
#
#   uv run --no-sync python scripts/run_loocv_sweep.py --select

#SBATCH --job-name=trunx-loocv-sweep
#SBATCH --output=data/results/nutrition_modifier/loocv_sweep/logs/%x-%A_%a.out
#SBATCH --error=data/results/nutrition_modifier/loocv_sweep/logs/%x-%A_%a.err
#SBATCH --partition=Dance
#SBATCH --qos=standby_qos
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --requeue

set -euo pipefail

# The Dance partition spans chacha (uv via snap, already on PATH) and disco (uv
# installed to ~/.local/bin, per-node local home — not on a batch shell's default
# PATH). Harmless no-op on chacha.
export PATH="$HOME/.local/bin:$PATH"

# Flush print() per line, so the log follows the training while it runs.
export PYTHONUNBUFFERED=1
# Figures are only saved, never shown.
export MPLBACKEND=Agg

# Force JAX onto CPU: no --gres is requested here (see calibration_sweep.sbatch).
export CUDA_VISIBLE_DEVICES=
export JAX_PLATFORM_NAME=cpu

PROJECT_DIR="${TRUNX_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT_DIR"

JOB_INDEX="${SLURM_ARRAY_TASK_ID:?Submit as an array (see the header comment for sizing it)}"
INPUT_VARS="${INPUT_VARS:-N S}"
MODIFIER="${MODIFIER:-poly_nm}"
PLOT_SELECTION="${PLOT_SELECTION:-poorly_fitted}"

echo "Job:     ${SLURM_JOB_ID:-interactive}, array task ${JOB_INDEX}, on $(hostname)"
echo "Project: $PROJECT_DIR"
echo "Inputs:  $INPUT_VARS"
echo "Modifier: $MODIFIER"
echo "Plots:   $PLOT_SELECTION"
echo

# Run in the Apptainer container, binding the checkout and data directories into it.
# `apptainer exec ... /app/.venv/bin/python3` rather than `apptainer run`, so sibling
# array tasks don't race on the shared .venv (see map_sweep_gpu.sbatch).
# Build trunkx-app.sif with `apptainer build trunkx-app.sif scripts/slurm/apptainer.def`.

srun --cpus-per-task="$SLURM_CPUS_PER_TASK" \
    apptainer exec \
        --env PYTHONPATH=/app/src \
        --env MPLBACKEND=Agg \
        --bind .:/app \
        --bind ./src:/app/src \
        --bind ./scripts:/app/scripts \
        --bind ./tests:/app/tests \
        --bind ./data:/app/data \
        --bind /tmp:/tmp \
        --pwd /app \
        trunkx-app.sif \
        /app/.venv/bin/python3 \
        /app/scripts/run_loocv_sweep.py \
        --job-index "$JOB_INDEX" \
        --input-vars $INPUT_VARS \
        --modifier "$MODIFIER" \
        --plot-selection "$PLOT_SELECTION"
