#!/bin/bash

## --- Job Configuration ---
#SBATCH --job-name=lee_tune_all_geno
#SBATCH --partition=batch

## --- Resources Requested ---
#SBATCH --time=100:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=31G

## --- Job Array: 10 reps x 3 genotypes x 4 tasks = 120 jobs ---
## Index layout: task varies fastest, then genotype, then rep
## idx = rep * 12 + geno * 4 + task
## e.g. 0=backandforth/spherical/r0, 1=figure8/spherical/r0, ...
#SBATCH --array=0-119

## --- Slurm Job Logs ---
#SBATCH --output=./out/%x_%A_%a.out
#SBATCH --error=./out/%x_%A_%a.err

# --- Derive task, genotype, and repetition from array index ---
TASKS=("backandforth" "figure8" "circle" "slalom")
GENOTYPES=("spherical" "cppn" "hybrid-cppn")
NUM_TASKS=${#TASKS[@]}
NUM_GENOS=${#GENOTYPES[@]}

TASK_IDX=$(( SLURM_ARRAY_TASK_ID % NUM_TASKS ))
GENO_IDX=$(( (SLURM_ARRAY_TASK_ID / NUM_TASKS) % NUM_GENOS ))
REP_IDX=$(( SLURM_ARRAY_TASK_ID / (NUM_TASKS * NUM_GENOS) ))
TASK="${TASKS[$TASK_IDX]}"
GENOTYPE="${GENOTYPES[$GENO_IDX]}"

# --- Job Info ---
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Task: $TASK (index $TASK_IDX)"
echo "Genotype: $GENOTYPE (index $GENO_IDX)"
echo "Repetition: $REP_IDX"
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
TMP_DIR="/tmp/lee_tuning_${TASK}_${GENOTYPE}_rep${REP_IDX}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
FINAL_DIR="/scratch/jed/airevolve_data_180226"

mkdir -p "$TMP_DIR"
echo "Temporary data directory: $TMP_DIR"

# --- Build genotype-specific flags ---
EXTRA_ARGS=""
if [ "$GENOTYPE" = "cppn" ]; then
    EXTRA_ARGS="--num-segments 8 --initial-hidden-nodes 0"
elif [ "$GENOTYPE" = "hybrid-cppn" ]; then
    EXTRA_ARGS="--initial-hidden-nodes 0"
fi

# --- Run Evolution with Lee Tuning ---
cd "$PROJECT_DIR"

srun python3 examples/evolution/run_evolution_with_lee_tuning.py \
    --genome-handler "$GENOTYPE" \
    --gate-cfg "$TASK" \
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
    --log-dir "$TMP_DIR/lee_tuning_runs" \
    $EXTRA_ARGS

# --- Move results to scratch ---
echo "Moving results from $TMP_DIR to $FINAL_DIR ..."
mkdir -p "$FINAL_DIR"
mv "$TMP_DIR" "$FINAL_DIR/lee_tuning_${TASK}_${GENOTYPE}_rep${REP_IDX}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Results saved to $FINAL_DIR/lee_tuning_${TASK}_${GENOTYPE}_rep${REP_IDX}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

echo "Task $SLURM_ARRAY_TASK_ID ($TASK/$GENOTYPE rep $REP_IDX) finished at: $(date)"
echo "done"
