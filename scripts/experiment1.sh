#!/usr/bin/env bash
# =============================================================================
# experiment1.sh — Experiment 1: Baseline (no energy penalty)
#
# Runs one NSGA-II variant:
#   baseline  w = 0.0
#
# Usage:
#   bash scripts/experiment1.sh [--device DEVICE] [--parallel] [--dry-run]
#
# Options:
#   --device   PyTorch device string (default: cuda:0).
#   --parallel No-op here (only one variant); kept for interface consistency.
#   --dry-run  Print the command without executing it.
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

# ── Experiment 1 ─────────────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════════════════════"
echo " Experiment 1 — Baseline (standard RL, no energy penalty)"
echo "════════════════════════════════════════════════════════════════════════"

run_exp "baseline" "0.0"

# ── Wait for background jobs (no-op for a single variant) ─────────────────────
if $PARALLEL && ! $DRY_RUN; then
    echo ""
    echo "Waiting for background jobs to finish…"
    wait
    echo "Experiment 1 complete."
fi

echo ""
echo "Results → ${RESULTS_DIR}/exp_baseline_w_0/"
