#!/usr/bin/env bash
# =============================================================================
# run_exp_power_ppo_power_ea.sh
#
# Launch script for: Power-Aware PPO Training + Power-Aware NSGA-II Evaluation
# Hardware target  : AMD Threadripper Pro 32c/64t, 128 GB RAM
#
# Architecture
# ------------
#   RL  : PowerAwareDroneEnv — with energy penalties in the reward.
#         sparse_weight=0.004, overdraw_weight=0.01, strict_voltage_kill=False.
#   EA  : Power-aware bi-objective NSGA-II.
#         Fitness = (gates_passed ↑, total_energy_j ↓).
#         LiPoBatteryModel(strict_voltage_kill=True) in the 12-second eval.
#
# Usage
# -----
#   # Full run (default hyperparameters):
#   bash scripts/run_exp_power_ppo_power_ea.sh
#
#   # Dry-run (validate config, no training):
#   bash scripts/run_exp_power_ppo_power_ea.sh --dry-run
#
#   # Override any hyperparameter:
#   bash scripts/run_exp_power_ppo_power_ea.sh \
#       --population-size 16 --generations 8 \
#       --training-timesteps 500000 --num-workers 8
#
# Options
# -------
#   --dry-run     Print config without running evolution.
#   --device      PyTorch device (default: cpu).
#   Any extra flags are forwarded verbatim to the Python runner.
# =============================================================================

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_ACTIVATE="drone-venv/bin/activate"
RUNNER="${REPO_ROOT}/experimentation/run_exp_power_ppo_power_ea.py"
RESULTS_DIR="${REPO_ROOT}/results"
LOG_DIR="${REPO_ROOT}/logs/exp_power_ppo_power_ea"

TRAINING_TIMESTEPS=10000000
GENERATIONS=32
POPULATION_SIZE=32
NUM_WORKERS=32
NUM_ENVS=4
DEVICE="cpu"
GENOME="spherical"
GATE_CFG="figure8"
MIN_NARMS=6
MAX_NARMS=6
INIT_POP_MODE="hover_repair"
DRY_RUN=false
SPARSE_WEIGHT=0.004
OVERDRAW_WEIGHT=0.01

EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)               DRY_RUN=true                        ;;
        --device)                DEVICE="$2";         shift          ;;
        --training-timesteps)    TRAINING_TIMESTEPS="$2"; shift      ;;
        --generations)           GENERATIONS="$2";    shift          ;;
        --population-size)       POPULATION_SIZE="$2"; shift         ;;
        --num-workers)           NUM_WORKERS="$2";    shift          ;;
        --num-envs)              NUM_ENVS="$2";       shift          ;;
        --genome)                GENOME="$2";         shift          ;;
        --gate-cfg)              GATE_CFG="$2";       shift          ;;
        --results-dir)           RESULTS_DIR="$2";    shift          ;;
        --sparse-weight)         SPARSE_WEIGHT="$2";  shift          ;;
        --overdraw-weight)       OVERDRAW_WEIGHT="$2"; shift         ;;
        --run-id)                EXTRA_ARGS+=("--run-id" "$2"); shift ;;
        *)                       EXTRA_ARGS+=("$1")                   ;;
    esac
    if [[ $# -gt 0 ]]; then
        shift
    fi
done

mkdir -p "${LOG_DIR}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/run_${TIMESTAMP}.log"

echo "════════════════════════════════════════════════════════════════════════"
echo " Experiment: Power-Aware PPO + Power-Aware NSGA-II"
echo "════════════════════════════════════════════════════════════════════════"
echo "  training_timesteps : ${TRAINING_TIMESTEPS}"
echo "  generations        : ${GENERATIONS}"
echo "  population_size    : ${POPULATION_SIZE}"
echo "  num_workers (EA)   : ${NUM_WORKERS}"
echo "  num_envs (PPO)     : ${NUM_ENVS}"
echo "  device             : ${DEVICE}"
echo "  genome             : ${GENOME}"
echo "  gate_cfg           : ${GATE_CFG}"
echo "  results_dir        : ${RESULTS_DIR}"
echo "  log_file           : ${LOG_FILE}"
echo ""
echo "  RL flags  : sparse_weight=${SPARSE_WEIGHT}  overdraw_weight=${OVERDRAW_WEIGHT}  strict_kill=False  use_power_env=True"
echo "  EA flags  : strict_kill=True   fitness=(gates↑, energy↓)"
echo "════════════════════════════════════════════════════════════════════════"

CMD=(
    python "${RUNNER}"
    --training-timesteps  "${TRAINING_TIMESTEPS}"
    --generations         "${GENERATIONS}"
    --population-size     "${POPULATION_SIZE}"
    --num-workers         "${NUM_WORKERS}"
    --num-envs            "${NUM_ENVS}"
    --device              "${DEVICE}"
    --genome              "${GENOME}"
    --gate-cfg            "${GATE_CFG}"
    --min-narms           "${MIN_NARMS}"
    --max-narms           "${MAX_NARMS}"
    --init-pop-mode       "${INIT_POP_MODE}"
    --results-dir         "${RESULTS_DIR}"
    --sparse-weight       "${SPARSE_WEIGHT}"
    --overdraw-weight     "${OVERDRAW_WEIGHT}"
    "${EXTRA_ARGS[@]}"
)

if $DRY_RUN; then
    CMD+=(--dry-run)
fi

if [[ -f "${VENV_ACTIVATE}" ]]; then
    # shellcheck source=/dev/null
    source "${VENV_ACTIVATE}"
    echo "  venv activated: ${VENV_ACTIVATE}"
else
    echo "  [warn] venv not found at ${VENV_ACTIVATE} — using system Python"
fi

echo ""
echo "  Command: ${CMD[*]}"
echo "  Log    : ${LOG_FILE}"
echo ""

if $DRY_RUN; then
    "${CMD[@]}"
else
    START_WALL="$(date '+%Y-%m-%d %H:%M:%S')"
    SECONDS=0
    echo "  Started at : ${START_WALL}"
    echo ""

    "${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
    EXIT_CODE=${PIPESTATUS[0]}

    ELAPSED=${SECONDS}
    END_WALL="$(date '+%Y-%m-%d %H:%M:%S')"
    ELAPSED_H=$(( ELAPSED / 3600 ))
    ELAPSED_M=$(( (ELAPSED % 3600) / 60 ))
    ELAPSED_S=$(( ELAPSED % 60 ))
    ELAPSED_FMT=$(printf '%02d:%02d:%02d' "${ELAPSED_H}" "${ELAPSED_M}" "${ELAPSED_S}")

    echo ""
    echo "════════════════════════════════════════════════════════════════════════"
    if [[ "${EXIT_CODE}" -eq 0 ]]; then
        echo " Experiment COMPLETE."
        echo " Results → ${RESULTS_DIR}/exp_power_ppo_power_ea/"
    else
        echo " Experiment FAILED with exit code ${EXIT_CODE}."
        echo " Log      → ${LOG_FILE}"
    fi
    echo ""
    echo "  Started  : ${START_WALL}"
    echo "  Finished : ${END_WALL}"
    echo "  Elapsed  : ${ELAPSED_FMT}  (${ELAPSED}s)"
    echo "════════════════════════════════════════════════════════════════════════"
    exit "${EXIT_CODE}"
fi
