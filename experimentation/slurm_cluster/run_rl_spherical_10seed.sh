#!/bin/bash

#SBATCH --job-name=rl-sph-10
#SBATCH --output=out_files/rl-sph-%A_%a.out
#SBATCH --error=out_files/rl-sph-%A_%a.err
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --array=0-9
# Exclude sm_61 nodes (gtx1070 / gtx1070ti): venv torch is built for sm_70+.
#SBATCH --exclude=node01,node03,node04

set -euo pipefail

REPO=/home/jed/workspace/airevolve
VENV_PATH=$REPO/.venv

# Local scratch for in-flight writes; final location for kept results.
TMP_DIR=/tmp/${USER}/airevolve_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
FINAL_DIR=$REPO/__data__/evolution

RUN_TAG=rl_spherical_hover_repair_hg_figure8_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}

cd "$REPO"
mkdir -p out_files
mkdir -p "$TMP_DIR"

# Always move whatever the job produced — successful or not — so partial
# results are preserved on failure / timeout / preemption.
cleanup() {
    rc=$?
    if [ -d "$TMP_DIR" ] && [ -n "$(ls -A "$TMP_DIR" 2>/dev/null || true)" ]; then
        echo "Moving results from $TMP_DIR to $FINAL_DIR/$RUN_TAG ..."
        mkdir -p "$FINAL_DIR"
        mv "$TMP_DIR" "$FINAL_DIR/$RUN_TAG"
        echo "Results saved to $FINAL_DIR/$RUN_TAG"
    else
        echo "No results in $TMP_DIR to move (rc=$rc)"
        rm -rf "$TMP_DIR" || true
    fi
    exit $rc
}
trap cleanup EXIT

source "$VENV_PATH/bin/activate"

echo "Node:         $(hostname)"
echo "Job:          $SLURM_JOB_ID  Array task: $SLURM_ARRAY_TASK_ID"
echo "Python:       $(which python)"
python --version
nvidia-smi -L || true
echo "Tmp dir:      $TMP_DIR"
echo "Final dir:    $FINAL_DIR/$RUN_TAG"
echo "Started at:   $(date)"

srun python examples/evolution/run_evolution.py \
    --brain rl \
    --genome spherical \
    --fitness gate \
    --hover-gradient \
    --init-pop-mode hover_repair \
    --per-individual-repair \
    --strategy-type plus \
    --population-size 16 \
    --num-mutate 16 \
    --generations 50 \
    --training-timesteps 10000000 \
    --num-envs 100 \
    --device cuda:0 \
    --num-workers 2 \
    --sim-time 20.0 \
    --log-dir "$TMP_DIR"

echo "Task $SLURM_ARRAY_TASK_ID finished at: $(date)"
echo "done"
