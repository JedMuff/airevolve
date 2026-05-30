#!/usr/bin/env bash
#SBATCH --job-name=drone_ppo_ea
#SBATCH --output=slurm_logs/exp_%A_%a.out
#SBATCH --error=slurm_logs/exp_%A_%a.err
#SBATCH --array=1-10
#SBATCH --time=48:00:00  # Set to 48 hours to safely cover the 25h duration
#SBATCH --cpus-per-task=16 # 12 workers + 4 envs as per the original script
#SBATCH --mem=32G        # Adjust if more memory is needed
#SBATCH --partition=compute # Update with your specific SLURM partition

# This script runs the 10 experiments in parallel across the SLURM cluster.

# Load any necessary modules here (e.g., Python, CUDA) if your cluster requires it
# module load python/3.10

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXP_SCRIPT="${REPO_ROOT}/scripts/run_exp_standard_ppo_power_ea.sh"

# SLURM_ARRAY_TASK_ID will range from 1 to 10
RUN_ID=${SLURM_ARRAY_TASK_ID}

RESULTS_DIR="${REPO_ROOT}/results_${RUN_ID}"
LOG_DIR="${REPO_ROOT}/logs_${RUN_ID}"

# Create slurm_logs directory if it doesn't exist
mkdir -p "${REPO_ROOT}/slurm_logs"

echo "Starting SLURM job array task ${RUN_ID}"

bash "$EXP_SCRIPT" \
    --results-dir "$RESULTS_DIR" \
    --log-dir "$LOG_DIR"
