#!/usr/bin/env bash
# Session 4 follow-up: 10-seed parity sweep with airevolve's env wrapper +
# the *2-inch derived* sysid'd dynamics (--reference-params 2inch_derived).
# Compares against optimal_quad_control_RL's reference framework on the
# same track. 10M steps per run, sequential. ~80 min on RTX A4000.
set -euo pipefail
PY=/home/jed/miniconda3/envs/isaaclab/bin/python
ROOT=/home/jed/workspaces/airevolve/experimentation/comparison_results_session4_refdyn_2inch
TOTAL=1e7

mkdir -p "$ROOT"
echo "=== session 4 refdyn-2inch 10-seed parity (10M steps each) started $(date) ===" | tee "$ROOT/run.log"

for s in 1 2 3 4 5 6 7 8 9 10; do
    AOUT=$ROOT/airevolve/seed_$s
    OOUT=$ROOT/optimal/seed_$s
    mkdir -p "$AOUT" "$OOUT"

    echo ">>> [$(date +%H:%M:%S)] airevolve+refdyn-2inch seed=$s -> $AOUT" | tee -a "$ROOT/run.log"
    cd /home/jed/workspaces/airevolve/experimentation
    $PY optimal_parity_quad.py --reference-dynamics --reference-params 2inch_derived \
        --prop-size 2 --seed "$s" --total-steps "$TOTAL" --tag s4-2in \
        --out-dir "$AOUT" > "$AOUT/run.log" 2>&1

    echo ">>> [$(date +%H:%M:%S)] optimal                  seed=$s -> $OOUT" | tee -a "$ROOT/run.log"
    cd /home/jed/workspaces/optimal_quad_control_RL
    $PY parity_train.py --seed "$s" --total-steps "$TOTAL" --out-dir "$OOUT" \
        > "$OOUT/run.log" 2>&1
done

echo "=== all seeds done $(date) ===" | tee -a "$ROOT/run.log"
