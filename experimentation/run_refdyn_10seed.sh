#!/usr/bin/env bash
# Session 4: 10-seed parity sweep with airevolve's env wrapper but
# optimal_quad_control_RL's sysid'd 5-inch dynamics swapped in
# (--reference-dynamics). 10M steps per run, sequential.
#
# Compares against the optimal reference framework on the same track.
# Output dir mirrors run_parity_10seed.sh's layout (airevolve/, optimal/)
# so analyze_parity_10seed.py works against it via --root-dir.
#
# Total wall-clock: ~80 min on RTX A4000.
set -euo pipefail
PY=/home/jed/miniconda3/envs/isaaclab/bin/python
ROOT=/home/jed/workspaces/airevolve/experimentation/comparison_results_session4_refdyn
TOTAL=1e7

mkdir -p "$ROOT"
echo "=== session 4 refdyn 10-seed parity (10M steps each) started $(date) ===" | tee "$ROOT/run.log"

for s in 1 2 3 4 5 6 7 8 9 10; do
    AOUT=$ROOT/airevolve/seed_$s
    OOUT=$ROOT/optimal/seed_$s
    mkdir -p "$AOUT" "$OOUT"

    echo ">>> [$(date +%H:%M:%S)] airevolve+refdyn seed=$s -> $AOUT" | tee -a "$ROOT/run.log"
    cd /home/jed/workspaces/airevolve/experimentation
    $PY optimal_parity_quad.py --reference-dynamics --prop-size 2 \
        --seed "$s" --total-steps "$TOTAL" --tag s4 --out-dir "$AOUT" \
        > "$AOUT/run.log" 2>&1

    echo ">>> [$(date +%H:%M:%S)] optimal          seed=$s -> $OOUT" | tee -a "$ROOT/run.log"
    cd /home/jed/workspaces/optimal_quad_control_RL
    $PY parity_train.py --seed "$s" --total-steps "$TOTAL" --out-dir "$OOUT" \
        > "$OOUT/run.log" 2>&1
done

echo "=== all seeds done $(date) ===" | tee -a "$ROOT/run.log"
