#!/bin/bash
#SBATCH -J airevolve_ea_10rep
#SBATCH -p genoa
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --exclusive
#SBATCH -t 22:00:00
#SBATCH --mem=320G
#SBATCH --array=1-10
#SBATCH --output=./logs/%x_%A_%a.out
#SBATCH --error=./logs/%x_%A_%a.err

module load 2022
module load Python/3.10.4-GCCcore-11.3.0

source /projects/prjs2127/airevolve/venv/bin/activate

mkdir -p ./logs

SCRATCH_DIR="/scratch-shared/$USER/airevolve_tmp_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$SCRATCH_DIR/results"
mkdir -p "$SCRATCH_DIR/logs"

srun bash scripts/run_exp_standard_ppo_power_ea.sh \
    --num-workers 32 \
    --num-envs 5 \
    --device cpu \
    --training-timesteps 3000000 \
    --results-dir "$SCRATCH_DIR/results" \
    --log-dir "$SCRATCH_DIR/logs" \
    --run-id "exp_standard_ppo_power_ea_rep${SLURM_ARRAY_TASK_ID}"

mkdir -p "$SLURM_SUBMIT_DIR/results"
mkdir -p "$SLURM_SUBMIT_DIR/logs"
cp -r "$SCRATCH_DIR/results/"* "$SLURM_SUBMIT_DIR/results/" 2>/dev/null || true
cp -r "$SCRATCH_DIR/logs/"* "$SLURM_SUBMIT_DIR/logs/" 2>/dev/null || true