#!/bin/bash
#SBATCH -J airevolve_ea
#SBATCH -p gpu_a100
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH -t 48:00:00
#SBATCH --mem=128G

#SBATCH --array=0-1

#SBATCH --output=./logs/%x_%A_%a.out
#SBATCH --error=./logs/%x_%A_%a.err

module load 2022
module load Python/3.10.4-GCCcore-11.3.0

echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on node: $(hostname)"
echo "Submit directory: $SLURM_SUBMIT_DIR"

mkdir -p ./logs

SCRATCH_DIR="/scratch-shared/$USER/airevolve_tmp_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$SCRATCH_DIR/results"
mkdir -p "$SCRATCH_DIR/logs"
echo "Scratch directory: $SCRATCH_DIR"

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    echo "Running STANDARD PPO experiment"
    srun bash scripts/run_exp_standard_ppo_power_ea.sh \
        --num-workers 5 \
        --device cuda \
        --results-dir "$SCRATCH_DIR/results" \
        --log-dir "$SCRATCH_DIR/logs" \
        --run-id exp_standard_ppo_power_ea
elif [ "$SLURM_ARRAY_TASK_ID" -eq 1 ]; then
    echo "Running POWER-AWARE PPO experiment"
    srun bash scripts/run_exp_power_ppo_power_ea.sh \
        --num-workers 5 \
        --device cuda \
        --results-dir "$SCRATCH_DIR/results" \
        --log-dir "$SCRATCH_DIR/logs" \
        --run-id exp_power_ppo_power_ea
fi

echo ""
echo "Copying results from scratch to $SLURM_SUBMIT_DIR ..."
mkdir -p "$SLURM_SUBMIT_DIR/results"
mkdir -p "$SLURM_SUBMIT_DIR/logs"
cp -r "$SCRATCH_DIR/results/"* "$SLURM_SUBMIT_DIR/results/" 2>/dev/null || true
cp -r "$SCRATCH_DIR/logs/"* "$SLURM_SUBMIT_DIR/logs/" 2>/dev/null || true
echo "Done. Results copied to $SLURM_SUBMIT_DIR/results/"