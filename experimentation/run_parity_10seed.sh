#!/usr/bin/env bash
# 10-seed parity sweep: airevolve vs optimal_quad_control_RL
# 10M steps per run, sequential (single GPU, both use 100 envs).
# Total wall-clock: ~80 min on RTX A4000.
set -euo pipefail
PY=/home/jed/miniconda3/envs/isaaclab/bin/python
ROOT=/home/jed/workspaces/airevolve/experimentation/comparison_results
TOTAL=1e7

mkdir -p "$ROOT"
echo "=== 10-seed parity (10M steps each, sequential) started $(date) ===" | tee "$ROOT/run.log"

for s in 1 2 3 4 5 6 7 8 9 10; do
    AOUT=$ROOT/airevolve/seed_$s
    OOUT=$ROOT/optimal/seed_$s
    mkdir -p "$AOUT" "$OOUT"

    echo ">>> [$(date +%H:%M:%S)] airevolve seed=$s -> $AOUT" | tee -a "$ROOT/run.log"
    cd /home/jed/workspaces/airevolve/experimentation
    $PY optimal_parity_quad.py --prop-size 2 \
        --seed "$s" --total-steps "$TOTAL" --tag 10seed --out-dir "$AOUT" \
        > "$AOUT/run.log" 2>&1

    echo ">>> [$(date +%H:%M:%S)] optimal   seed=$s -> $OOUT" | tee -a "$ROOT/run.log"
    cd /home/jed/workspaces/optimal_quad_control_RL
    $PY parity_train.py --seed "$s" --total-steps "$TOTAL" --out-dir "$OOUT" \
        > "$OOUT/run.log" 2>&1
done

echo "=== all seeds done $(date) ===" | tee -a "$ROOT/run.log"
