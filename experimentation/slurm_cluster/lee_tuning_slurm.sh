#!/bin/bash

## --- Job Configuration ---
#SBATCH --job-name=lee_tune_fig8
#SBATCH --partition=batch

## --- Resources Requested ---
#SBATCH --time=100:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=31G

## --- Job Array: 10 repetitions ---
#SBATCH --array=0-9

## --- Slurm Job Logs ---
#SBATCH --output=./out/%x_%A_%a.out
#SBATCH --error=./out/%x_%A_%a.err

# --- Job Info ---
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on node: $(hostname)"
echo "Job started at: $(date)"

# --- Environment Setup ---
VENV_PATH=/home/jed/workspace/airevolve/.venv
PROJECT_DIR=/home/jed/workspace/airevolve

source "$VENV_PATH/bin/activate"

export PATH="$VENV_PATH/bin:$PATH"

echo "Using Python: $(which python3)"
python3 --version

# --- Create output directories ---
mkdir -p "$PROJECT_DIR/out"

# --- Set up temporary and final output directories ---
TMP_DIR="/tmp/lee_tuning_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
FINAL_DIR="/scratch/jed/airevolve_data_180226"

mkdir -p "$TMP_DIR"
echo "Temporary data directory: $TMP_DIR"

# --- Run Evolution with Lee Tuning ---
cd "$PROJECT_DIR"

srun python3 examples/run_evolution_with_lee_tuning.py \
    --gate-cfg figure8 \
    --population-size 16 \
    --generations 50 \
    --num-mutate 16 \
    --num-crossover 0 \
    --max-evals 500 \
    --cma-workers 1 \
    --sim-time 20.0 \
    --dt 0.005 \
    --timeout 30.0 \
    --num-workers 32 \
    --min-narms 6 \
    --max-narms 6 \
    --init-pop-max-evals 500 \
    --init-pop-gates-threshold 8 \
    --init-pop-tuning-workers 32 \
    --log-dir "$TMP_DIR/lee_tuning_runs"

# --- Move results to scratch ---
echo "Moving results from $TMP_DIR to $FINAL_DIR ..."
mkdir -p "$FINAL_DIR"
mv "$TMP_DIR" "$FINAL_DIR/lee_tuning_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Results saved to $FINAL_DIR/lee_tuning_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

echo "Task $SLURM_ARRAY_TASK_ID finished at: $(date)"
echo "done"
