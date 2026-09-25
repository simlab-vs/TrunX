#!/bin/bash
# PyMC multiplot calibration sweep: every (literature_source, error_mode, step_method)
# combination for one species, using pymc_param_est_multiplots.py's shared-parameter
# model that fits every plot for that species jointly in one run (see
# run_pymc_multi_plot_analysis).
#
# Unlike calibration_sweep.sbatch (one job per single plot), each task here already
# covers every plot for the species, so the sweep is only over
# (literature_source, error_mode, step_method) — LITERATURE_SOURCES x ERROR_MODES x
# STEP_METHODS below, kept in sync by hand with pymc_param_est_multiplots.py's
# --literature-source/--step-method choices and bayesian_config.ERROR_MODES's keys.
#
# Warmup/sample counts are method-specific, not swept independently: DEMetropolisZ
# needs a much longer chain to mix well (1e6 warmup + 5e6 samples, matching
# calibration_sweep.sbatch's own DEMetropolisZ jobs), while NUTS's gradient-informed
# proposals converge with far fewer (1000 + 1000).
#
# One-time setup (the SBATCH --output/--error directives need this to exist before
# the array can even start):
#
#   mkdir -p data/results/pymc_multiplot_results/logs
#
# Submit one task per (source, mode, method) combination, N = len(LITERATURE_SOURCES)
# * len(ERROR_MODES) * len(STEP_METHODS) - 1 (16 combinations -> --array=0-15 as of
# writing):
#
#   sbatch --array=0-15 scripts/slurm/pymc_multiplot_sweep.sbatch
#
# Override the species (default "Picea abies") or the checkout:
#
#   sbatch --array=0-15 --export=ALL,TRUNX_MULTIPLOT_SPECIES="Fagus sylvatica" ...
#   sbatch --array=0-15 --export=ALL,TRUNX_PROJECT_DIR=/path/to/checkout ...
#
# run_pymc_multi_plot_analysis has no mid-run checkpoint (unlike the single-plot
# pymc_param_est.py path calibration_sweep.sbatch uses), so a preempted standby
# task restarts its combination from scratch rather than resuming it.

#SBATCH --job-name=trunx-pymc-multiplot-sweep
#SBATCH --output=data/results/pymc_multiplot_results/logs/%x-%A_%a.out
#SBATCH --error=data/results/pymc_multiplot_results/logs/%x-%A_%a.err
#SBATCH --partition=Dance
#SBATCH --qos=standby_qos
#SBATCH --cpus-per-task=5
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
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

# Force JAX onto CPU: no --gres is requested here (see calibration_sweep.sbatch).
export CUDA_VISIBLE_DEVICES=
export JAX_PLATFORM_NAME=cpu

PROJECT_DIR="${TRUNX_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT_DIR"

SPECIES="${TRUNX_MULTIPLOT_SPECIES:-Picea abies}"

# Keep these lists in sync with pymc_param_est_multiplots.py's --literature-source/
# --step-method choices and bayesian_config.ERROR_MODES's keys.
LITERATURE_SOURCES=(Forrester Trotsiuk)
ERROR_MODES=(DBH_only all_error_terms biomass_DBH_only biomass_only)
STEP_METHODS=(demetropolisz nuts)

JOB_INDEX="${SLURM_ARRAY_TASK_ID:?Submit as an array (see the header comment for sizing it)}"

N_MODES=${#ERROR_MODES[@]}
N_SOURCES=${#LITERATURE_SOURCES[@]}
N_METHODS=${#STEP_METHODS[@]}
TOTAL=$((N_SOURCES * N_MODES * N_METHODS))
if (( JOB_INDEX >= TOTAL )); then
    echo "Job index $JOB_INDEX out of range (0-$((TOTAL - 1)) for $N_SOURCES sources x $N_MODES modes x $N_METHODS methods)" >&2
    exit 1
fi
SOURCE_IDX=$((JOB_INDEX / (N_MODES * N_METHODS)))
MODE_IDX=$(((JOB_INDEX / N_METHODS) % N_MODES))
METHOD_IDX=$((JOB_INDEX % N_METHODS))
LITERATURE_SOURCE="${LITERATURE_SOURCES[$SOURCE_IDX]}"
ERROR_MODE="${ERROR_MODES[$MODE_IDX]}"
STEP_METHOD="${STEP_METHODS[$METHOD_IDX]}"

# DEMetropolisZ needs a much longer chain to mix well; NUTS's gradient-informed
# proposals converge with far fewer draws (see header comment).
if [[ "$STEP_METHOD" == "demetropolisz" ]]; then
    NUM_WARMUP=1000000
    NUM_SAMPLES=5000000
else
    NUM_WARMUP=1000
    NUM_SAMPLES=1000
fi

SPECIES_SLUG="${SPECIES// /_}"
OUTPUT_DIR="data/results/pymc_multiplot_results/${SPECIES_SLUG}/${LITERATURE_SOURCE}/${ERROR_MODE}/${STEP_METHOD}"

echo "Job:               ${SLURM_JOB_ID:-interactive}, array task ${JOB_INDEX}, on $(hostname)"
echo "Project:           $PROJECT_DIR"
echo "Species:           $SPECIES"
echo "Literature source: $LITERATURE_SOURCE"
echo "Error mode:        $ERROR_MODE"
echo "Step method:       $STEP_METHOD"
echo "Warmup/samples:    $NUM_WARMUP / $NUM_SAMPLES"
echo "Output dir:        $OUTPUT_DIR"
echo

# Run in the Apptainer container, binding the checkout and data directories into it
# so it can read/write them. `apptainer exec ... /app/.venv/bin/python3 <script>`
# rather than `apptainer run` on purpose (see map_sweep_gpu.sbatch: the image's
# %runscript's `uv run` re-scans the shared, bind-mounted .venv on every invocation
# and races with sibling array tasks starting at the same time).
# Build trunkx-app.sif with `apptainer build trunkx-app.sif scripts/slurm/apptainer.def`.

srun --cpus-per-task="$SLURM_CPUS_PER_TASK" \
    apptainer exec \
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
        /app/src/trunx/gp3/bayesiancalibrations/pymc_param_est_multiplots.py \
        --species "$SPECIES" \
        --literature-source "$LITERATURE_SOURCE" \
        --error-mode "$ERROR_MODE" \
        --step-method "$STEP_METHOD" \
        --output-dir "$OUTPUT_DIR" \
        --num-warmup "$NUM_WARMUP" \
        --num-samples "$NUM_SAMPLES" \
        --chains 4 \
        --cores 4
