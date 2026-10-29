#!/usr/bin/env bash
# Launch train_vuhex_drag_sweep.sh (or SWEEP=physics:
# train_vuhex_physics_sweep.sh) for several seeds in a detached tmux
# session, one window per seed, so the runs survive logouts, closed
# terminals and VS Code restarts.
#
#   bash examples/learning/launch_vuhex_drag_sweep_tmux.sh            # seeds 1 2
#   SEEDS="1 2 3" THREADS=5 bash examples/learning/launch_vuhex_drag_sweep_tmux.sh
#   SWEEP=physics bash examples/learning/launch_vuhex_drag_sweep_tmux.sh   # tmux session vuhex_physics_sweep
#
#   tmux attach -t vuhex_sweep     # watch (Ctrl-b n / p switches seed, Ctrl-b d detaches)
#   tmux kill-session -t vuhex_sweep   # stop everything
#
# Refuses to start while a sweep is already training, since two runs with
# the same name delete and overwrite each other's checkpoints.
set -euo pipefail
cd "$(dirname "$0")/../.."
SWEEP=${SWEEP:-drag}
case $SWEEP in
    drag)    SCRIPT=train_vuhex_drag_sweep.sh;    TRAIN_SESSION=vuhex;         OUT=sweep ;;
    physics) SCRIPT=train_vuhex_physics_sweep.sh; TRAIN_SESSION=vuhex_physics; OUT=sweep_physics ;;
    *) echo "unknown SWEEP=$SWEEP (drag or physics)"; exit 1 ;;
esac
SESSION=${SESSION:-${TRAIN_SESSION}_sweep}
SEEDS=${SEEDS:-"1 2"}
PY=${PY:-.venv/bin/python}
THREADS=${THREADS:-8}

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session $SESSION already exists: tmux attach -t $SESSION (or kill it first)"
    exit 1
fi
if pgrep -f "train_rl.py $TRAIN_SESSION " > /dev/null; then
    echo "$TRAIN_SESSION training is already running:"
    pgrep -af "train_rl.py $TRAIN_SESSION " | cut -c1-120
    exit 1
fi

first=1
for seed in $SEEDS; do
    cmd="SEED=$seed PY=$PY THREADS=$THREADS bash examples/learning/$SCRIPT 2>&1 | tee ${OUT}_s$seed.out; exec bash"
    if [ "$first" = 1 ]; then
        tmux new-session -d -s "$SESSION" -n "s$seed" -c "$PWD" "$cmd"
        first=0
    else
        tmux new-window -t "$SESSION" -n "s$seed" -c "$PWD" "$cmd"
    fi
done
echo "Started seeds $SEEDS in tmux session $SESSION: tmux attach -t $SESSION"
