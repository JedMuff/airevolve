#!/usr/bin/env bash
# Script to run the experiment 10 times sequentially.
# IMPORTANT: To ensure this script doesn't stop if your SSH session disconnects,
# you MUST run it inside a terminal multiplexer like 'tmux' or 'screen', or use 'nohup'.
# For example: 
#   tmux new -s exp_run
#   bash scripts/run_10_experiments_local.sh
#   (then you can detach with Ctrl+B, D)

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXP_SCRIPT="${REPO_ROOT}/scripts/run_exp_standard_ppo_power_ea.sh"

for i in {1..10}; do
    echo "======================================"
    echo "Starting Experiment Run $i of 10"
    echo "======================================"
    
    RESULTS_DIR="${REPO_ROOT}/results_$i"
    LOG_DIR="${REPO_ROOT}/logs_$i"
    
    bash "$EXP_SCRIPT" \
        --device cpu \
        --results-dir "$RESULTS_DIR" \
        --log-dir "$LOG_DIR"
        
    echo "Finished Run $i"
done
