#!/usr/bin/env bash
# =============================================================================
# run_all_experiments.sh
#
# Sequentially launches all 8 NSGA-II energy-ablation runs:
#   Exp 1  — Baseline          (1 variant)
#   Exp 2  — Dense penalty     (3 variants: w = 0.0, 1e-5, 5e-5)
#   Exp 3  — Sparse penalty    (4 variants: w = 0.0, 0.001, 0.004, 0.007)
#
# Logs are written to logs/exp_<type>_w_<weight>.log
#
# Usage:
#   cd <repo_root>
#   bash scripts/run_all_experiments.sh [--parallel] [--device DEVICE]
#
# Options:
#   --parallel   Run all experiments concurrently (requires enough GPU memory
#                for all num_envs=100 training environments simultaneously).
#                Default: sequential (safer on a single GPU).
#   --device     PyTorch device string (default: cuda:0).
#   --dry-run    Print commands without executing them.
# =============================================================================

set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────────
PARALLEL=false
DEVICE="cuda:0"
DRY_RUN=false
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ── Parse flags ──────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --parallel)  PARALLEL=true  ;;
        --device)    DEVICE="$2"; shift ;;
        --dry-run)   DRY_RUN=true   ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
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

# ── Helper: build the common argument string ──────────────────────────────────
common_args() {
    echo "--genome ${GENOME} \
          --population-size ${POP_SIZE} \
          --generations ${GENS} \
          --training-timesteps ${TIMESTEPS} \
          --num-envs ${NUM_ENVS} \
          --device ${DEVICE} \
          --gate-cfg ${GATE_CFG} \
          --results-dir ${RESULTS_DIR}"
}

# ── Helper: run one experiment ────────────────────────────────────────────────
run_exp() {
    local exp_type="$1"
    local weight="$2"
    local log_tag="${exp_type}_w_${weight}"
    local log_file="${LOG_DIR}/${log_tag}.log"

    local cmd="python ${REPO_ROOT}/experimentation/run_energy_ablation.py \
        --experiment-type ${exp_type} \
        --penalty-weight ${weight} \
        $(common_args)"

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Starting: exp_type=${exp_type}  weight=${weight}"
    echo "  Log file: ${log_file}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if $DRY_RUN; then
        echo "[dry-run] $cmd"
        return
    fi

    if $PARALLEL; then
        eval "$cmd" >"${log_file}" 2>&1 &
        echo "  PID: $!"
    else
        eval "$cmd" 2>&1 | tee "${log_file}"
        echo "  Finished exp_type=${exp_type} weight=${weight}"
    fi
}

# ── Experiment 1 — Baseline ──────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════════════════"
echo " EXPERIMENT 1 — Baseline (no energy penalty)"
echo "════════════════════════════════════════════════════════════════════════"
run_exp "baseline" "0.0"

# ── Experiment 2 — Dense penalty ─────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════════════════"
echo " EXPERIMENT 2 — Dense penalty  (3 variants)"
echo "════════════════════════════════════════════════════════════════════════"
run_exp "dense" "0.0"
run_exp "dense" "0.00001"
run_exp "dense" "0.00005"

# ── Experiment 3 — Sparse penalty ────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════════════════"
echo " EXPERIMENT 3 — Sparse penalty  (4 variants)"
echo "════════════════════════════════════════════════════════════════════════"
run_exp "sparse" "0.0"
run_exp "sparse" "0.001"
run_exp "sparse" "0.004"
run_exp "sparse" "0.007"

# ── Wait for background jobs (parallel mode) ──────────────────────────────────
if $PARALLEL && ! $DRY_RUN; then
    echo ""
    echo "Waiting for all background experiments to complete…"
    wait
    echo "All experiments finished."
fi

echo ""
echo "════════════════════════════════════════════════════════════════════════"
echo " All experiments complete.  Results in: ${RESULTS_DIR}"
echo "════════════════════════════════════════════════════════════════════════"

# ── Optional: generate comparison Pareto plot across all results ──────────────
if ! $DRY_RUN; then
    echo ""
    echo "Generating cross-experiment Pareto comparison plot…"
    python "${REPO_ROOT}/airevolve/evolution_tools/inspection_tools/plot_pareto_front.py" \
        "${RESULTS_DIR}/exp_baseline_w_0.0/evolution_data.csv" \
        "${RESULTS_DIR}/exp_dense_w_1e-05/evolution_data.csv" \
        "${RESULTS_DIR}/exp_dense_w_5e-05/evolution_data.csv" \
        "${RESULTS_DIR}/exp_sparse_w_0.0/evolution_data.csv" \
        "${RESULTS_DIR}/exp_sparse_w_1e-03/evolution_data.csv" \
        "${RESULTS_DIR}/exp_sparse_w_4e-03/evolution_data.csv" \
        "${RESULTS_DIR}/exp_sparse_w_7e-03/evolution_data.csv" \
        --labels "Baseline" "Dense 0" "Dense 1e-5" "Dense 5e-5" \
                 "Sparse 0" "Sparse 1e-3" "Sparse 4e-3" "Sparse 7e-3" \
        --output "${RESULTS_DIR}/pareto_comparison_all.png" \
        --title "Energy Ablation — All Experiments" \
        2>/dev/null || echo "[warn] Comparison plot skipped (some runs may be incomplete)"
fi
