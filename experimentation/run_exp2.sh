#!/usr/bin/env bash
# =============================================================================
# run_exp2.sh — Experiment 2: Sparse End-of-Episode Energy Penalty
# =============================================================================
#
# Reward shaping:  reward_terminal -= sparse_weight × E_episode_joules
#                  (no in-flight penalty; agent only penalised at episode end)
#
# Weight sweep:    [0.0 0.001 0.002 0.003 0.004 0.005 0.006 0.007 0.008 0.009 0.01 0.02 0.03 0.04 0.05]
# Total runs:      6
# Steps / run:     50_000_000
# Output root:     /local/data/mdu219/drone-experiment-2/
#
# Estimated wall time: ~80 min/run × 6 runs ≈ 8.0 hours (sequential)
# =============================================================================

set -euo pipefail

source /local/data/mdu219/venvs/drone-venv/bin/activate
DEVICE="cuda:0"
NUM_ENVS=64
TOTAL_STEPS=50000000
MAX_STEPS=1200

BASE_DIR="/local/data/mdu219/drone-experiment-2"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="${REPO_ROOT}/examples/learning/run_power_experiment.py"

WEIGHTS=(0.0 0.001 0.002 0.003 0.004 0.005 0.006 0.007 0.008 0.009 0.01 0.02 0.03 0.04 0.05)

echo "========================================================================"
echo "  Experiment 2 — Sparse End-of-Episode Energy Penalty"
echo "  Sweep: ${WEIGHTS[*]}"
echo "  Runs:  ${#WEIGHTS[@]}  ×  ${TOTAL_STEPS} steps  ×  ${NUM_ENVS} envs"
echo "  GPU:   ${DEVICE}   |   Repo: ${REPO_ROOT}"
echo "  Output: ${BASE_DIR}"
echo "========================================================================"
echo ""

mkdir -p "${BASE_DIR}"

TOTAL_RUNS=${#WEIGHTS[@]}
RUN_NUM=0

for W in "${WEIGHTS[@]}"; do
    RUN_NUM=$((RUN_NUM + 1))
    RUN_DIR="${BASE_DIR}/sparse_w${W}"
    echo "--------------------------------------------------------------------"
    echo "  Run ${RUN_NUM}/${TOTAL_RUNS}  |  sparse_weight=${W}  |  ${RUN_DIR}"
    echo "  Started: $(date)"
    echo "--------------------------------------------------------------------"

    python "${SCRIPT}" \
        --experiment    2       \
        --sparse-weight "${W}"  \
        --num-envs      "${NUM_ENVS}"    \
        --total-steps   "${TOTAL_STEPS}" \
        --max-steps     "${MAX_STEPS}"   \
        --device        "${DEVICE}"      \
        --save-dir      "${RUN_DIR}"

    echo "  Finished: $(date)"
    echo ""
done

echo "========================================================================"
echo "  Experiment 2 complete."
echo "  TensorBoard: tensorboard --logdir ${BASE_DIR}"
echo "========================================================================"
