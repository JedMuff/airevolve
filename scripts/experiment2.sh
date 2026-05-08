#!/usr/bin/env bash
# =============================================================================
# experiment2.sh — Experiment 2: Dense penalty (3 variants)
#
# Runs three NSGA-II variants:
#   dense  w = 0.0
#   dense  w = 0.00001  (1e-5)
#   dense  w = 0.00005  (5e-5)
#
# Usage:
#   bash scripts/experiment2.sh [--device DEVICE] [--parallel] [--dry-run]
#
# Options:
#   --device   PyTorch device string (default: cuda:0).
#   --parallel Run all three variants concurrently.
#              WARNING: requires enough GPU memory for 3 × num_envs=100
#              training environments simultaneously.
#              Default: sequential.
#   --dry-run  Print commands without executing them.
# =============================================================================

set -uo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
PARALLEL=false
DEVICE="cuda:0"
DRY_RUN=false
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ── Parse flags ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --parallel)  PARALLEL=true  ;;
        --device)    DEVICE="$2"; shift ;;
        --dry-run)   DRY_RUN=true   ;;
        *) echo "Unknown flag: $1" >&2; exit 1 ;;
    esac
    shift
done

# ── Shared hyperparameters ────────────────────────────────────────────────────
POP_SIZE=24
GENS=30
TIMESTEPS=50000000
NUM_ENVS=100
GENOME="spherical"
GATE_CFG="figure8"
RESULTS_DIR="${REPO_ROOT}/results"
LOG_DIR="${REPO_ROOT}/logs"

mkdir -p "${LOG_DIR}"

# ── Helper: run one experiment variant ───────────────────────────────────────
run_exp() {
    local exp_type="$1"
    local weight="$2"
    local log_tag="${exp_type}_w_${weight}"
    local log_file="${LOG_DIR}/${log_tag}.log"

    local -a cmd=(
        python "${REPO_ROOT}/experimentation/run_energy_ablation.py"
        --experiment-type  "${exp_type}"
        --penalty-weight   "${weight}"
        --genome           "${GENOME}"
        --population-size  "${POP_SIZE}"
        --generations      "${GENS}"
        --training-timesteps "${TIMESTEPS}"
        --num-envs         "${NUM_ENVS}"
        --device           "${DEVICE}"
        --gate-cfg         "${GATE_CFG}"
        --results-dir      "${RESULTS_DIR}"
    )

    echo "────────────────────────────────────────────────────────────────────"
    echo "  exp_type : ${exp_type}    weight : ${weight}"
    echo "  log      : ${log_file}"
    echo "────────────────────────────────────────────────────────────────────"

    if $DRY_RUN; then
        echo "[dry-run] ${cmd[*]}"
        return 0
    fi

    if $PARALLEL; then
        "${cmd[@]}" > "${log_file}" 2>&1 &
        echo "  Launched in background — PID $!"
    else
        "${cmd[@]}" 2>&1 | tee "${log_file}"
        echo "  Completed: ${log_tag}"
    fi
}

# ── Experiment 2 ─────────────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════════════════════"
echo " Experiment 2 — Dense penalty  (3 variants)"
echo "   r -= dense_weight × P_instant  every step"
echo "════════════════════════════════════════════════════════════════════════"

run_exp "dense" "0.0"
run_exp "dense" "0.00001"
run_exp "dense" "0.00005"

# ── Wait for background jobs ──────────────────────────────────────────────────
if $PARALLEL && ! $DRY_RUN; then
    echo ""
    echo "Waiting for all dense-penalty background jobs to finish…"
    wait
    echo "Experiment 2 complete."
fi

echo ""
echo "Results →"
echo "  ${RESULTS_DIR}/exp_dense_w_0/"
echo "  ${RESULTS_DIR}/exp_dense_w_1e-05/"
echo "  ${RESULTS_DIR}/exp_dense_w_5e-05/"
