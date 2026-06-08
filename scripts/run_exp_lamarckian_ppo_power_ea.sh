#!/usr/bin/env bash
# =============================================================================
# run_exp_lamarckian_ppo_power_ea.sh
#
# Launch script for: Lamarckian Standard-PPO Training + Power-Aware NSGA-II
# Hardware target  : AMD EPYC 9654 (Genoa), 192 cores, 320 GB RAM
#
# Worker tuning (Genoa 192-core node)
#   24 workers × 8 torch_threads = 192 cores (full node saturation)
#   24 workers × (1 + 25 SubprocVecEnv) = 624 peak processes
#
# Architecture
# ------------
#   RL  : Standard DroneGateEnv — no power penalties, no voltage kill.
#         sparse_weight=0.0, overdraw_weight=0.0, strict_voltage_kill=False.
#   EA  : Power-aware bi-objective NSGA-II.
#         Fitness = (gates_passed ↑, total_energy_j ↓).
#         LiPoBatteryModel(strict_voltage_kill=True) in the 12-second eval.
#   Lamarckian:
#         Gen 0  → Darwinian: 10M ts, no parent policy.
#         Gen 1+ → Inherit parent policy.zip, continue training (2M ts).
#         Fallback → Missing policy.zip: train from scratch (3M ts).
#
# Usage
# -----
#   # Full run (default hyperparameters for Genoa):
#   bash scripts/run_exp_lamarckian_ppo_power_ea.sh
#
#   # Dry-run (validate config, no training):
#   bash scripts/run_exp_lamarckian_ppo_power_ea.sh --dry-run
#
#   # Override any hyperparameter:
#   bash scripts/run_exp_lamarckian_ppo_power_ea.sh \
#       --population-size 16 --generations 8 \
#       --num-workers 8 --torch-threads 4
#
# Options
# -------
#   --dry-run     Print config without running evolution.
#   --device      PyTorch device (default: cpu).
#   Any extra flags are forwarded verbatim to the Python runner.
# =============================================================================

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_ACTIVATE="venv/bin/activate"
RUNNER="${REPO_ROOT}/experimentation/final_standard_ppo_power_ea.py"
RESULTS_DIR="${REPO_ROOT}/results"
LOG_DIR="${REPO_ROOT}/logs/exp_lamarckian_ppo_power_ea"

# ── Default hyperparameters (tuned for Genoa 192-core node) ──────────────────
TRAINING_TIMESTEPS=10000000       # Gen 0 Darwinian start (10M ts)
GENERATIONS=32
POPULATION_SIZE=24
NUM_WORKERS=24
NUM_ENVS=5                        # SubprocVecEnv parallelism (24 workers * 5 envs = 120 procs, avoids core thrashing)
TORCH_THREADS=8                   # 24 workers × 8 threads = 192 cores
DEVICE="cpu"
GENOME="spherical"
GATE_CFG="figure8"
MIN_NARMS=6
MAX_NARMS=6
INIT_POP_MODE="hover_repair"
Z_DRAG_MULTIPLIER=5.0
DRY_RUN=false

EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)               DRY_RUN=true                             ;;
        --device)                DEVICE="$2";              shift          ;;
        --training-timesteps)    TRAINING_TIMESTEPS="$2";  shift          ;;
        --generations)           GENERATIONS="$2";         shift          ;;
        --population-size)       POPULATION_SIZE="$2";     shift          ;;
        --num-workers)           NUM_WORKERS="$2";         shift          ;;
        --num-envs)              NUM_ENVS="$2";            shift          ;;
        --torch-threads)         TORCH_THREADS="$2";       shift          ;;
        --genome)                GENOME="$2";              shift          ;;
        --gate-cfg)              GATE_CFG="$2";            shift          ;;
        --z-drag-multiplier)     Z_DRAG_MULTIPLIER="$2";  shift          ;;
        --results-dir)           RESULTS_DIR="$2";         shift          ;;
        --log-dir)               LOG_DIR="$2";             shift          ;;
        --run-id)                EXTRA_ARGS+=("--run-id" "$2"); shift     ;;
        *)                       EXTRA_ARGS+=("$1")                        ;;
    esac
    if [[ $# -gt 0 ]]; then
        shift
    fi
done

mkdir -p "${LOG_DIR}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/run_${TIMESTAMP}.log"

TOTAL_CORES=$(( NUM_WORKERS * TORCH_THREADS ))
PEAK_PROCS=$(( NUM_WORKERS * (1 + NUM_ENVS) ))

echo "════════════════════════════════════════════════════════════════════════"
echo " Experiment: Lamarckian Standard-PPO + Power-Aware NSGA-II"
echo "════════════════════════════════════════════════════════════════════════"
echo "  training_timesteps : ${TRAINING_TIMESTEPS} (Gen 0 Darwinian)"
echo "  generations        : ${GENERATIONS}"
echo "  population_size    : ${POPULATION_SIZE}"
echo "  num_workers (EA)   : ${NUM_WORKERS}"
echo "  torch_threads      : ${TORCH_THREADS}  (${NUM_WORKERS}×${TORCH_THREADS}=${TOTAL_CORES} total cores)"
echo "  num_envs (PPO)     : ${NUM_ENVS}  (${PEAK_PROCS} peak processes)"
echo "  device             : ${DEVICE}"
echo "  genome             : ${GENOME}"
echo "  gate_cfg           : ${GATE_CFG}"
echo "  z_drag_multiplier  : ${Z_DRAG_MULTIPLIER}"
echo "  results_dir        : ${RESULTS_DIR}"
echo "  log_file           : ${LOG_FILE}"
echo ""
echo "  RL env  : DroneGateEnv (gate_train_power.py) — no power penalties"
echo "  EA eval : BiObjectiveFitness → gate_train_power.evaluate_individual"
echo "  Lamarckian: Gen 0=10M ts | Gen 1+=2M ts (parent) | Fallback=3M ts"
echo "════════════════════════════════════════════════════════════════════════"

CMD=(
    python "${RUNNER}"
    --training-timesteps  "${TRAINING_TIMESTEPS}"
    --generations         "${GENERATIONS}"
    --population-size     "${POPULATION_SIZE}"
    --num-workers         "${NUM_WORKERS}"
    --num-envs            "${NUM_ENVS}"
    --torch-threads       "${TORCH_THREADS}"
    --device              "${DEVICE}"
    --genome              "${GENOME}"
    --gate-cfg            "${GATE_CFG}"
    --min-narms           "${MIN_NARMS}"
    --max-narms           "${MAX_NARMS}"
    --init-pop-mode       "${INIT_POP_MODE}"
    --z-drag-multiplier   "${Z_DRAG_MULTIPLIER}"
    --results-dir         "${RESULTS_DIR}"
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
        echo " Results → ${RESULTS_DIR}/exp_lamarckian_standard_ppo_power_ea/"
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
