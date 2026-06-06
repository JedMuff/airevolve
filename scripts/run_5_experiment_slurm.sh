#!/bin/bash
#SBATCH -J airevolve_ea_5rep
#SBATCH -p gpu_a100
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH -t 48:00:00
#SBATCH --mem=120G
#SBATCH --array=0-4
#SBATCH --output=./logs/%x_%A_%a.out
#SBATCH --error=./logs/%x_%A_%a.err

module load 2022
module load Python/3.10.4-GCCcore-11.3.0

# Activate project venv
source /projects/prjs2127/airevolve/venv/bin/activate

echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on node: $(hostname)"
echo "Submit directory: $SLURM_SUBMIT_DIR"

mkdir -p ./logs

SCRATCH_DIR="/scratch-shared/$USER/airevolve_tmp_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$SCRATCH_DIR/results"
mkdir -p "$SCRATCH_DIR/logs"
echo "Scratch directory: $SCRATCH_DIR"

REP_ID=$((SLURM_ARRAY_TASK_ID + 1))
echo "Running STANDARD PPO experiment - Repetition $REP_ID / 5"

srun bash scripts/run_exp_standard_ppo_power_ea.sh \
    --num-workers 16 \
    --device cpu \
    --results-dir "$SCRATCH_DIR/results" \
    --log-dir "$SCRATCH_DIR/logs" \
    --run-id "exp_standard_ppo_power_ea_rep${REP_ID}"

echo ""
echo "Copying results from scratch to $SLURM_SUBMIT_DIR ..."
mkdir -p "$SLURM_SUBMIT_DIR/results"
mkdir -p "$SLURM_SUBMIT_DIR/logs"
cp -r "$SCRATCH_DIR/results/"* "$SLURM_SUBMIT_DIR/results/" 2>/dev/null || true
cp -r "$SCRATCH_DIR/logs/"* "$SLURM_SUBMIT_DIR/logs/" 2>/dev/null || true
echo "Done. Results copied to $SLURM_SUBMIT_DIR/results/"
