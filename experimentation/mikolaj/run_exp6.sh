#!/usr/bin/env bash
# =============================================================================
# run_exp3.sh — Experiment 3: Hybrid (Dense Penalty + Sparse SoC Bonus)
# =============================================================================
#
# Reward shaping:  reward_t          -= dense_weight  × P_instantaneous_t
#                  reward_terminal    += sparse_bonus  × SoC_final_%
#                                        (ONLY if battery was NOT depleted)
#
# Pair sweep:
#   (dense_weight, sparse_bonus) = (0.0,    0 )  ← baseline (no penalty)
#                                  (0.0001, 50)
#                                  (0.0001,100)
#                                  (0.0005, 50)
#                                  (0.0005,100)
#
# Total runs:     5
# Steps / run:    50_000_000
# Output root:    /local/data/mdu219/drone-experiment-6/
#
# Estimated wall time: ~80 min/run × 5 runs ≈ 6.6 hours (sequential)
# =============================================================================

set -euo pipefail

source /local/data/mdu219/venvs/drone-venv/bin/activate
DEVICE="cuda:0"
NUM_ENVS=64
TOTAL_STEPS=50000000
MAX_STEPS=1200

BASE_DIR="/local/data/mdu219/drone-experiment-6"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="${REPO_ROOT}/examples/learning/run_power_experiment.py"

# Paired sweep: "dense_weight:sparse_bonus"
PAIRS=(
    "0.0:0"
    "0.0001:1"
    "0.0001:5"
    "0.0001:10"
    "0.0001:20"
    "0.0005:1"
    "0.0005:5"
    "0.0005:10"
    "0.0005:50"
)


echo "========================================================================"
echo "  Experiment 6 — Hybrid (Dense Penalty + Sparse SoC Survival Bonus)"
echo "  Pairs (dw:sb): ${PAIRS[*]}"
echo "  Runs:  ${#PAIRS[@]}  ×  ${TOTAL_STEPS} steps  ×  ${NUM_ENVS} envs"
echo "  GPU:   ${DEVICE}   |   Repo: ${REPO_ROOT}"
echo "  Output: ${BASE_DIR}"
echo "========================================================================"
echo ""

mkdir -p "${BASE_DIR}"

TOTAL_RUNS=${#PAIRS[@]}
RUN_NUM=0

for PAIR in "${PAIRS[@]}"; do
    RUN_NUM=$((RUN_NUM + 1))

    # Split "dense_weight:sparse_bonus" on the colon
    DW="${PAIR%%:*}"
    SB="${PAIR##*:}"

    RUN_DIR="${BASE_DIR}/d${DW}_b${SB}"
    echo "--------------------------------------------------------------------"
    echo "  Run ${RUN_NUM}/${TOTAL_RUNS}  |  dense_weight=${DW}  sparse_bonus=${SB}"
    echo "  Output: ${RUN_DIR}"
    echo "  Started: $(date)"
    echo "--------------------------------------------------------------------"

    python "${SCRIPT}" \
        --experiment    3       \
        --dense-weight  "${DW}" \
        --sparse-bonus  "${SB}" \
        --num-envs      "${NUM_ENVS}"    \
        --total-steps   "${TOTAL_STEPS}" \
        --max-steps     "${MAX_STEPS}"   \
        --device        "${DEVICE}"      \
        --save-dir      "${RUN_DIR}"

    echo "  Finished: $(date)"
    echo ""
done

echo "========================================================================"
echo "  Experiment 6 complete."
echo "  TensorBoard: tensorboard --logdir ${BASE_DIR}"
echo "========================================================================"
