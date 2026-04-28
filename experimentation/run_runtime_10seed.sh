#!/usr/bin/env bash
# Session 5: 10-seed parity sweep with airevolve's env wrapper running the
# *new runtime path* (no --reference-dynamics flag). The runtime DroneSimulator
# now uses reference-form dynamics; this sweep validates that the runtime
# path matches the refdyn-2inch baseline (comparison_results_session4_refdyn_2inch).
#
# 10M steps per run, sequential. ~50-80 min on RTX A4000.
set -euo pipefail
PY=/home/jed/miniconda3/envs/isaaclab/bin/python
ROOT=/home/jed/workspaces/airevolve/experimentation/comparison_results_session5_runtime
TOTAL=1e7

mkdir -p "$ROOT"
echo "=== session 5 runtime 10-seed parity (10M steps each) started $(date) ===" | tee "$ROOT/run.log"

# Only run airevolve seeds. The optimal baseline is reused from
# comparison_results_session4_refdyn_2inch via symlink (the optimal
# framework hasn't changed, so re-running it would be wasted compute).
mkdir -p "$ROOT/airevolve" "$ROOT/optimal"
SESSION4=/home/jed/workspaces/airevolve/experimentation/comparison_results_session4_refdyn_2inch

for s in 1 2 3 4 5 6 7 8 9 10; do
    AOUT=$ROOT/airevolve/seed_$s
    OOUT=$ROOT/optimal/seed_$s
    mkdir -p "$AOUT"
    if [ ! -e "$OOUT" ] && [ -d "$SESSION4/optimal/seed_$s" ]; then
        ln -s "$SESSION4/optimal/seed_$s" "$OOUT"
    fi

    echo ">>> [$(date +%H:%M:%S)] airevolve runtime seed=$s -> $AOUT" | tee -a "$ROOT/run.log"
    cd /home/jed/workspaces/airevolve/experimentation
    $PY optimal_parity_quad.py \
        --prop-size 2 --seed "$s" --total-steps "$TOTAL" --tag s5-runtime \
        --out-dir "$AOUT" > "$AOUT/run.log" 2>&1
done

echo "=== all seeds done $(date) ===" | tee -a "$ROOT/run.log"
