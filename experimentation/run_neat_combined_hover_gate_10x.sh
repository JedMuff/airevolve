#!/bin/bash
# Run NEAT combined hover+gate evolution 10 times sequentially
# for a single genotype + task combination.
#
# Usage:
#   ./experimentation/run_neat_combined_hover_gate_10x.sh <genotype> <task>
#
# Example:
#   ./experimentation/run_neat_combined_hover_gate_10x.sh spherical figure8

set -e

GENOTYPE="${1:?Usage: $0 <genotype> <task>  (genotype: spherical|cppn|hybrid-cppn)}"
TASK="${2:?Usage: $0 <genotype> <task>  (task: backandforth|figure8|circle|slalom)}"
NUM_REPS=9

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${PROJECT_DIR}/.data"

# Build genotype-specific flags
EXTRA_ARGS=""
if [ "$GENOTYPE" = "cppn" ]; then
    EXTRA_ARGS="--num-segments 8 --initial-hidden-nodes 0"
elif [ "$GENOTYPE" = "hybrid-cppn" ]; then
    EXTRA_ARGS="--initial-hidden-nodes 0"
fi

echo "============================================================"
echo "NEAT Combined Hover+Gate: ${GENOTYPE} / ${TASK} x ${NUM_REPS}"
echo "Project dir: ${PROJECT_DIR}"
echo "Log dir: ${LOG_DIR}"
echo "============================================================"

for REP in $(seq 0 $((NUM_REPS - 1))); do
    echo ""
    echo "--- Repetition ${REP}/${NUM_REPS} started at $(date) ---"

    python3 "${PROJECT_DIR}/experimentation/run_neat_combined_hover_gate_evolution.py" \
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
        --log-dir "$LOG_DIR" \
        $EXTRA_ARGS

    echo "--- Repetition ${REP}/${NUM_REPS} finished at $(date) ---"
done

echo ""
echo "All ${NUM_REPS} repetitions complete."
