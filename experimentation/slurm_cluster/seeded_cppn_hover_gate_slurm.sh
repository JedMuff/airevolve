#!/bin/bash

## --- Job Configuration ---
#SBATCH --job-name=seeded_cppn_hg
#SBATCH --partition=batch

## --- Resources Requested ---
#SBATCH --time=100:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=31G

## --- Job Array: 10 reps x 2 genotypes x 3 tasks x 2 strategies = 120 jobs ---
## Index layout: strategy varies fastest, then task, then genotype, then rep
## idx = rep * 12 + geno * 6 + task * 2 + strategy
#SBATCH --array=0-119

## --- Slurm Job Logs ---
#SBATCH --output=./out/%x_%A_%a.out
#SBATCH --error=./out/%x_%A_%a.err

# --- Derive strategy, task, genotype, and repetition from array index ---
TASKS=("backandforth" "figure8" "circle")
GENOTYPES=("cppn" "hybrid-cppn")
STRATEGIES=("mulambda" "neat")
NUM_STRATEGIES=${#STRATEGIES[@]}
NUM_TASKS=${#TASKS[@]}
NUM_GENOS=${#GENOTYPES[@]}

STRAT_IDX=$(( SLURM_ARRAY_TASK_ID % NUM_STRATEGIES ))
TASK_IDX=$(( (SLURM_ARRAY_TASK_ID / NUM_STRATEGIES) % NUM_TASKS ))
GENO_IDX=$(( (SLURM_ARRAY_TASK_ID / (NUM_STRATEGIES * NUM_TASKS)) % NUM_GENOS ))
REP_IDX=$(( SLURM_ARRAY_TASK_ID / (NUM_STRATEGIES * NUM_TASKS * NUM_GENOS) ))
TASK="${TASKS[$TASK_IDX]}"
GENOTYPE="${GENOTYPES[$GENO_IDX]}"
STRATEGY="${STRATEGIES[$STRAT_IDX]}"

# --- Job Info ---
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Task: $TASK (index $TASK_IDX)"
echo "Genotype: $GENOTYPE (index $GENO_IDX)"
echo "Strategy: $STRATEGY (index $STRAT_IDX)"
echo "Repetition: $REP_IDX"
echo "Running on node: $(hostname)"
echo "Job started at: $(date)"

# --- Environment Setup ---
VENV_PATH=/home/user/workspace/airevolve/.venv
PROJECT_DIR=/home/user/workspace/airevolve

source "$VENV_PATH/bin/activate"

export PATH="$VENV_PATH/bin:$PATH"

echo "Using Python: $(which python3)"
python3 --version

# --- Create output directories ---
mkdir -p "$PROJECT_DIR/out"

# --- Set up temporary and final output directories ---
TMP_DIR="/tmp/seeded_cppn_hg_${STRATEGY}_${TASK}_${GENOTYPE}_rep${REP_IDX}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
FINAL_DIR="/scratch/user/airevolve_data_seeded_cppn"

mkdir -p "$TMP_DIR"
echo "Temporary data directory: $TMP_DIR"

# --- Build genotype-specific flags ---
EXTRA_ARGS=""
if [ "$GENOTYPE" = "cppn" ]; then
    EXTRA_ARGS="--num-segments 8 --initial-hidden-nodes 0 --init-topology seeded"
elif [ "$GENOTYPE" = "hybrid-cppn" ]; then
    EXTRA_ARGS="--initial-hidden-nodes 0 --init-topology seeded"
fi

# --- Run evolution with the appropriate strategy ---
cd "$PROJECT_DIR"

if [ "$STRATEGY" = "neat" ]; then
    srun python3 experimentation/run_neat_combined_hover_gate_evolution.py \
        --genome-handler "$GENOTYPE" \
        --gate-cfg "$TASK" \
        --population-size 32 \
        --generations 50 \
        --crossover-rate 0.75 \
        --compatibility-threshold 3.0 \
        --target-species-count 5 \
        --stagnation-limit 15 \
        --max-evals 300 \
        --cma-workers 1 \
        --sim-time 20.0 \
        --dt 0.005 \
        --timeout 30.0 \
        --num-workers 32 \
        --min-narms 6 \
        --max-narms 6 \
        --log-dir "$TMP_DIR/seeded_${STRATEGY}_runs" \
        $EXTRA_ARGS
elif [ "$STRATEGY" = "mulambda" ]; then
    srun python3 experimentation/run_combined_hover_gate_evolution.py \
        --genome-handler "$GENOTYPE" \
        --gate-cfg "$TASK" \
        --population-size 32 \
        --generations 50 \
        --num-mutate 32 \
        --num-crossover 0 \
        --strategy-type plus \
        --max-evals 300 \
        --cma-workers 1 \
        --sim-time 20.0 \
        --dt 0.005 \
        --timeout 30.0 \
        --num-workers 32 \
        --min-narms 6 \
        --max-narms 6 \
        --log-dir "$TMP_DIR/seeded_${STRATEGY}_runs" \
        $EXTRA_ARGS
fi

# --- Move results to scratch ---
echo "Moving results from $TMP_DIR to $FINAL_DIR ..."
mkdir -p "$FINAL_DIR"
mv "$TMP_DIR" "$FINAL_DIR/seeded_${STRATEGY}_${TASK}_${GENOTYPE}_rep${REP_IDX}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Results saved to $FINAL_DIR/seeded_${STRATEGY}_${TASK}_${GENOTYPE}_rep${REP_IDX}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

echo "Task $SLURM_ARRAY_TASK_ID ($STRATEGY/$TASK/$GENOTYPE rep $REP_IDX) finished at: $(date)"
echo "done"
