#!/bin/bash
#SBATCH -J airevolve_resume
#SBATCH -p genoa
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --exclusive
#SBATCH -t 08:15:00
#SBATCH --mem=320G
#SBATCH --array=2,4,5,8,9
#SBATCH --output=./logs/%x_%A_%a.out
#SBATCH --error=./logs/%x_%A_%a.err

module load 2022
module load Python/3.10.4-GCCcore-11.3.0

source /projects/prjs2127/airevolve/venv/bin/activate

mkdir -p ./logs

NEW_SCRATCH_DIR="/scratch-shared/$USER/airevolve_resume_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$NEW_SCRATCH_DIR/results"
mkdir -p "$NEW_SCRATCH_DIR/logs"

cleanup() {
    mkdir -p "$SLURM_SUBMIT_DIR/results_resumed" "$SLURM_SUBMIT_DIR/logs_resumed"
    cp -r "$NEW_SCRATCH_DIR/results/"* "$SLURM_SUBMIT_DIR/results_resumed/" 2>/dev/null || true
    cp -r "$NEW_SCRATCH_DIR/logs/"* "$SLURM_SUBMIT_DIR/logs_resumed/" 2>/dev/null || true
    exit 1
}

trap cleanup SIGTERM SIGINT

if [ "$SLURM_ARRAY_TASK_ID" -le 5 ]; then
    GATE_CFG="figure8"
    REP_ID="$SLURM_ARRAY_TASK_ID"
    RUN_ID_BASE="figure8"
else
    GATE_CFG="backandforth"
    REP_ID=$(( SLURM_ARRAY_TASK_ID - 5 ))
    RUN_ID_BASE="shuttlerun"
fi

OLD_RUN_ID="exp_standard_ppo_power_ea_${RUN_ID_BASE}_rep${REP_ID}"
OLD_SCRATCH_DIR="/scratch-shared/$USER/airevolve_tmp_23732190_${SLURM_ARRAY_TASK_ID}"
LATEST_GEN_DIR=$(ls -d ${OLD_SCRATCH_DIR}/results/${OLD_RUN_ID}/snapshots/gen_* 2>/dev/null | sort -V | tail -n 1)
RESUME_FILE="${LATEST_GEN_DIR}/genomes.npy"
NEW_RUN_ID="${OLD_RUN_ID}_resumed"

srun python experimentation/run_exp_resume.py \
    --training-timesteps 10000000 \
    --generations 1 \
    --population-size 24 \
    --num-mutate 32 \
    --num-workers 24 \
    --num-envs 5 \
    --torch-threads 8 \
    --device cpu \
    --genome spherical \
    --gate-cfg "$GATE_CFG" \
    --z-drag-multiplier 1.0 \
    --results-dir "$NEW_SCRATCH_DIR/results" \
    --run-id "$NEW_RUN_ID" \
    --resume-genomes "$RESUME_FILE"

mkdir -p "$SLURM_SUBMIT_DIR/results_resumed" "$SLURM_SUBMIT_DIR/logs_resumed"
cp -r "$NEW_SCRATCH_DIR/results/"* "$SLURM_SUBMIT_DIR/results_resumed/" 2>/dev/null || true
cp -r "$NEW_SCRATCH_DIR/logs/"* "$SLURM_SUBMIT_DIR/logs_resumed/" 2>/dev/null || true