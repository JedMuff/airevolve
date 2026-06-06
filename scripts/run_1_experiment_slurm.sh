#!/bin/bash
#SBATCH -J airevolve_ea_single
#SBATCH -p genoa
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH -t 48:00:00
#SBATCH --mem=120G
#SBATCH --output=./logs/%x_%j.out
#SBATCH --error=./logs/%x_%j.err

module load 2022
module load Python/3.10.4-GCCcore-11.3.0

# Activate project venv
source /projects/prjs2127/airevolve/venv/bin/activate

echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Submit directory: $SLURM_SUBMIT_DIR"

mkdir -p ./logs

SCRATCH_DIR="/scratch-shared/$USER/airevolve_single_${SLURM_JOB_ID}"
mkdir -p "$SCRATCH_DIR/results"
mkdir -p "$SCRATCH_DIR/logs"
echo "Scratch directory: $SCRATCH_DIR"

echo "Running STANDARD PPO experiment - Single run on GENOA"

srun bash scripts/run_exp_standard_ppo_power_ea.sh \
    --num-workers 32 \
    --num-envs 1 \
    --device cpu \
    --results-dir "$SCRATCH_DIR/results" \
    --log-dir "$SCRATCH_DIR/logs" \
    --run-id "exp_standard_ppo_power_ea"

echo ""
echo "Copying results from scratch to $SLURM_SUBMIT_DIR ..."
mkdir -p "$SLURM_SUBMIT_DIR/results"
mkdir -p "$SLURM_SUBMIT_DIR/logs"
cp -r "$SCRATCH_DIR/results/"* "$SLURM_SUBMIT_DIR/results/" 2>/dev/null || true
cp -r "$SCRATCH_DIR/logs/"* "$SLURM_SUBMIT_DIR/logs/" 2>/dev/null || true
echo "Done. Results copied to $SLURM_SUBMIT_DIR/results/"