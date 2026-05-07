#!/usr/bin/env bash
# =============================================================================
# run_exp1.sh — Experiment 1: Dense Timestep Power Penalty
# =============================================================================
#
# Reward shaping:  reward_t -= dense_weight × P_instantaneous_t
#
# Weight sweep:    [0.0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
# Total runs:      8
# Steps / run:     10_000_000
# Output root:     /local/data/mdu219/drone-experiment-1/
#
# Server requirements
# -------------------
# • AMD EPYC 9375F (64 threads) + NVIDIA L4 24GB  ← target machine
# • Conda environment with PyTorch CUDA 12.x, stable-baselines3, sympy
# • Estimated wall time: ~80 min/run × 8 runs ≈ 10.6 hours (sequential)
#
# Usage:
#   chmod +x run_exp1.sh
#   ./run_exp1.sh                     # runs all 8 sweeps sequentially
#   ./run_exp1.sh 2>&1 | tee exp1.log # capture all output to file
#
# TensorBoard (once at least one run has started):
#   tensorboard --logdir /local/data/mdu219/drone-experiment-1/
# =============================================================================

set -euo pipefail

# ── Environment ───────────────────────────────────────────────────────────────
CONDA_ENV="airevolve"          # change to your conda/venv name if different
DEVICE="cuda:0"
NUM_ENVS=64                    # optimal for L4 + 64-thread EPYC (see profiling notes)
TOTAL_STEPS=10000000
MAX_STEPS=1200                 # 12 s per episode at dt=0.01

# ── Paths ─────────────────────────────────────────────────────────────────────
# All output goes to /local/data/mdu219 to avoid home-directory quota limits.
BASE_DIR="/local/data/mdu219/drone-experiment-1"
# Resolve repo root as the directory containing this script
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="${REPO_ROOT}/examples/learning/run_power_experiment.py"

# ── Weights to sweep ──────────────────────────────────────────────────────────
WEIGHTS=(0.0 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1)

# ── Activate conda environment ────────────────────────────────────────────────
# Source conda so 'conda activate' works in non-login shells
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi
conda activate "${CONDA_ENV}"

echo "========================================================================"
echo "  Experiment 1 — Dense Timestep Penalty"
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
    RUN_DIR="${BASE_DIR}/dense_w${W}"
    echo "--------------------------------------------------------------------"
    echo "  Run ${RUN_NUM}/${TOTAL_RUNS}  |  dense_weight=${W}  |  ${RUN_DIR}"
    echo "  Started: $(date)"
    echo "--------------------------------------------------------------------"

    python "${SCRIPT}" \
        --experiment    1       \
        --dense-weight  "${W}"  \
        --num-envs      "${NUM_ENVS}"   \
        --total-steps   "${TOTAL_STEPS}" \
        --max-steps     "${MAX_STEPS}"  \
        --device        "${DEVICE}"     \
        --save-dir      "${RUN_DIR}"

    echo "  Finished: $(date)"
    echo ""
done

echo "========================================================================"
echo "  Experiment 1 complete."
echo "  TensorBoard: tensorboard --logdir ${BASE_DIR}"
echo "========================================================================"
