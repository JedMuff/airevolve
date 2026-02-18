#!/bin/bash

## --- Job Configuration ---
#SBATCH --job-name=curriculum_evo
#SBATCH --partition=batch

## --- Resources Requested ---
#SBATCH --time=100:00:00
#SBATCH --cpus-per-task=14
#SBATCH --mem=64G

## --- Job Array: 8 repetitions x 2 arm configs = 16 tasks ---
## Even tasks (0,2,4,...18): 4 arms, Odd tasks (1,3,5,...19): 6 arms
#SBATCH --array=0-19

## --- Slurm Job Logs ---
#SBATCH --output=./out/%x_%A_%a.out
#SBATCH --error=./out/%x_%A_%a.err

# --- Job Info ---
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on node: $(hostname)"
echo "Job started at: $(date)"

# --- Environment Setup ---
VENV_PATH=/home/jed/workspaces/airevolve/.venv
PROJECT_DIR=/home/jed/workspaces/airevolve

source "$VENV_PATH/bin/activate"

export PATH="$VENV_PATH/bin:$PATH"

echo "Using Python: $(which python3)"
python3 --version

# --- Create output directories ---
mkdir -p "$PROJECT_DIR/out"

# --- Determine arm configuration from array task ID ---
# Even tasks (0,2,4,...): 4 arms, Odd tasks (1,3,5,...): 6 arms
if [ $(( SLURM_ARRAY_TASK_ID % 2 )) -eq 0 ]; then
    NARMS=4
else
    NARMS=6
fi

echo "Arm configuration: ${NARMS} arms"

# --- Run Evolution with Curriculum ---
cd "$PROJECT_DIR"

srun python3 examples/run_evolution_with_curriculum.py \
    --num-workers 32 \
    --min-narms $NARMS \
    --max-narms $NARMS \
    --log-dir ./logs/curriculum_runs

echo "Task $SLURM_ARRAY_TASK_ID finished at: $(date)"
echo "done"
