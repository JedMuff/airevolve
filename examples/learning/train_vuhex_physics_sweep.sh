#!/usr/bin/env bash
# Train vuhex gate (figure-8) and hover policies in physics mode, the
# simulator evolution uses (--drone physics_vuhex: the hex3 layout with
# "vuhex" motors and airframe calibrated from params_vuhex), then export
# each policy as flight C code. Same setup as train_vuhex_drag_sweep.sh,
# whose real_vuhex runs are the sysid-mode counterparts.
#
#   bash examples/learning/train_vuhex_physics_sweep.sh
#   SEED=1 PY=.venv/bin/python THREADS=8 bash examples/learning/train_vuhex_physics_sweep.sh
#
# SEED=<n> seeds the envs and appends _s<n> to the run names. The two runs
# train in parallel with THREADS torch threads each.
#
# Policies: __data__/rl/models/vuhex_physics/<name>/, C code: c_code/<name>/,
# logs: __data__/rl/vuhex_physics_sweep_logs/<name>.log. Re-running a name
# deletes its previous checkpoints (train_rl.py).
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-python}
THREADS=${THREADS:-8}
SEED=${SEED:-}
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS
SESSION=vuhex_physics
LOGS=__data__/rl/vuhex_physics_sweep_logs
mkdir -p "$LOGS"

#    name                        drone               task   timesteps
RUNS=(
    "vuhex_physics_gate_r30      physics_vuhex       gates  1e8"
    "vuhex_physics_hover_r30     physics_vuhex       hover  2e7"
)

train_and_export() {
    local name=$1${SEED:+_s$SEED} drone=$2 task=$3 timesteps=$4
    echo ">>> [$(date +%H:%M:%S)] train $name ($drone, $task, $timesteps steps)"
    $PY examples/learning/train_rl.py "$SESSION" "$name" --drone "$drone" --task "$task" \
        --randomization 30 --timesteps "$timesteps" ${SEED:+--seed "$SEED"} > "$LOGS/$name.log" 2>&1
    echo ">>> [$(date +%H:%M:%S)] C code $name"
    $PY examples/deployment/generate_c_code.py "__data__/rl/models/$SESSION/$name" >> "$LOGS/$name.log" 2>&1
    echo ">>> [$(date +%H:%M:%S)] done $name"
}

pids=()
for run in "${RUNS[@]}"; do
    read -r name drone task timesteps <<< "$run"
    train_and_export "$name" "$drone" "$task" "$timesteps" &
    pids+=($!)
done

failed=0
for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
done
[ "$failed" = 0 ] && echo ">>> [$(date +%H:%M:%S)] all done" || { echo ">>> some runs failed, see $LOGS"; exit 1; }
