#!/bin/bash
# Bayesian comparison plots sweep: one task per individual job from
# bayesian_comparison_plots.py's own flattened job list (see its --task-index),
# covering every (literature_source, error_mode, plot_id) comparison plot plus
# the (literature_source, error_mode, method) cross-plot posterior plots.
#
# Unlike pymc_multiplot_sweep.sbatch, this only reads already-saved MCMC/MAP/
# gradient-descent results and renders matplotlib figures — no sampling, no
# checkpointing, each task finishes in minutes. Memory is still sized for the
# worst case, not the typical one: a task comparing against a DEMetropolisZ run
# loads its full multi-million-draw inference_data.nc into memory (~29G observed
# for one such task), well above what a NUTS-only comparison needs.
#
# One-time setup (the SBATCH --output/--error directives need this to exist
# before the array can even start):
#
#   mkdir -p data/results/bayesian_comparison_plots/logs
#
# Get the current total job count (prints "Total jobs: N" then exits), and
# submit the full sweep as --array=0-$((N-1)):
#
#   apptainer exec --bind .:/app --pwd /app trunkx-app.sif \
#       /app/.venv/bin/python3 /app/src/trunx/gp3/bayesiancalibrations/bayesian_comparison_plots.py \
#       --task-index 999999999
#   sbatch --array=0-187 scripts/slurm/bayesian_comparison_plots_sweep.sbatch
#
# (188 jobs as of writing; re-run the count above after changing the sweep's
# literature sources, error modes, or species_plot_ids, since --task-index N
# means whatever job is at position N in that flattened list.)
#
# Override the checkout:
#
#   sbatch --array=0-187 --export=ALL,TRUNX_PROJECT_DIR=/path/to/checkout ...

#SBATCH --job-name=trunx-bayesian-comparison-plots
#SBATCH --output=data/results/bayesian_comparison_plots/logs/%x-%A_%a.out
#SBATCH --error=data/results/bayesian_comparison_plots/logs/%x-%A_%a.err
#SBATCH --partition=Dance
#SBATCH --qos=standby_qos
#SBATCH --cpus-per-task=2
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --requeue

set -euo pipefail

# The Dance partition spans chacha (uv via snap, already on PATH) and disco (uv
# installed to ~/.local/bin, per-node local home — not on a batch shell's default
# PATH). Harmless no-op on chacha.
export PATH="$HOME/.local/bin:$PATH"

# stdout isn't a tty once redirected to the log file, so print() block-buffers
# instead of flushing per line — the log otherwise looks stalled while running.
export PYTHONUNBUFFERED=1

# Force JAX onto CPU: no --gres is requested here (see calibration_sweep.sbatch).
export CUDA_VISIBLE_DEVICES=
export JAX_PLATFORM_NAME=cpu

PROJECT_DIR="${TRUNX_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT_DIR"

JOB_INDEX="${SLURM_ARRAY_TASK_ID:?Submit as an array (see the header comment for sizing it)}"

echo "Job:        ${SLURM_JOB_ID:-interactive}, array task ${JOB_INDEX}, on $(hostname)"
echo "Project:    $PROJECT_DIR"
echo

# Run in the Apptainer container, binding the checkout and data directories into
# it so it can read/write them. `apptainer exec ... python3 <script>` rather than
# `apptainer run` on purpose (see map_sweep_gpu.sbatch: the image's %runscript's
# `uv run` re-scans the shared, bind-mounted .venv on every invocation and races
# with sibling array tasks starting at the same time).
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
        /app/src/trunx/gp3/bayesiancalibrations/bayesian_comparison_plots.py \
        --task-index "$JOB_INDEX"
