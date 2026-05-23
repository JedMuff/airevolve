#!/bin/bash
## --- Job Configuration ---
#SBATCH --job-name=airevolve_ea         # Name of your job
#SBATCH --partition=batch-gpu           # The partition/queue to run in

## --- Resources Requested ---
#SBATCH --time=48:00:00                 # Time limit (2 days)
#SBATCH --cpus-per-task=5               # Number of CPU cores per task
#SBATCH --gres=gpu:1                    # Request 1 GPU
#SBATCH --mem=128G                      # Memory required per node

## --- Job Array Configuration ---
#SBATCH --array=0-1                     # Create 2 tasks, indexed 0 and 1

## --- Slurm Job Logs ---
#SBATCH --output=./logs/%x_%A_%a.out    # Stdout: JobName_ArrayID_Index.out
#SBATCH --error=./logs/%x_%A_%a.err     # Stderr: JobName_ArrayID_Index.err

# --- Script Logic ---
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on node: $(hostname)"

# Make sure logs and results directories exist
mkdir -p ./logs
mkdir -p ./results

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    echo "Running STANDARD PPO experiment (no energy penalties in RL training)"
    bash scripts/run_exp_standard_ppo_power_ea.sh \
        --num-workers 5 \
        --results-dir ./results \
        --run-id exp_standard_ppo_power_ea
elif [ "$SLURM_ARRAY_TASK_ID" -eq 1 ]; then
    echo "Running POWER-AWARE PPO experiment (sparse_weight=0.004, overdraw_weight=0.01)"
    bash scripts/run_exp_power_ppo_power_ea.sh \
        --num-workers 5 \
        --results-dir ./results \
        --run-id exp_power_ppo_power_ea
fi
