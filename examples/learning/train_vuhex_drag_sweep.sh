#!/usr/bin/env bash
# Train vuhex gate (figure-8) and hover policies with the identified
# params_vuhex and with its body drag scaled 100x (real_vuhex_drag100),
# then export each policy as flight C code.
#
#   bash examples/learning/train_vuhex_drag_sweep.sh
#   PY=.venv/bin/python THREADS=8 bash examples/learning/train_vuhex_drag_sweep.sh
#
# SEED=<n> seeds the envs and appends _s<n> to the run names, so several
# copies can run at once without overwriting each other:
#
#   SEED=1 bash examples/learning/train_vuhex_drag_sweep.sh &
#   SEED=2 bash examples/learning/train_vuhex_drag_sweep.sh &
#
# The four runs train in parallel. A single run stops getting faster
# beyond ~8 torch threads (the env and the 64x3 MLP are small), so each
# gets THREADS of them; on the 64-core ripper 4 runs x 8 threads take
# ~31 s per 1M steps each vs ~24 s for one run alone. Two copies (8 runs)
# fill the 64 cores: ~43 s per 1M steps each, ~1.4x the total throughput.
#
# Policies: __data__/rl/models/vuhex/<name>/, C code: c_code/<name>/,
# logs: __data__/rl/vuhex_drag_sweep_logs/<name>.log. Re-running a name
# deletes its previous checkpoints (train_rl.py).
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-python}
THREADS=${THREADS:-8}
SEED=${SEED:-}
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS
SESSION=vuhex
LOGS=__data__/rl/vuhex_drag_sweep_logs
mkdir -p "$LOGS"

#    name                        drone               task   timesteps
RUNS=(
    "vuhex_gate_r30              real_vuhex          gates  1e8"
    "vuhex_hover_r30             real_vuhex          hover  2e7"
    "vuhex_drag100_gate_r30      real_vuhex_drag100  gates  1e8"
    "vuhex_drag100_hover_r30     real_vuhex_drag100  hover  2e7"
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
