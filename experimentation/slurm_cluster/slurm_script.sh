#!/bin/bash

#SBATCH --output=out_files/quad-%A_%a.out
#SBATCH --error=out_files/quad-%A_%a.err
#SBATCH --time=100:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --array=15-24

VENV_PATH=/home/jed/workspace/airevolve/.venv

echo "Node: $(hostname)"
echo "Using Python from: $VENV_PATH"

export PATH="$VENV_PATH/bin:$PATH"
export PYTHONPATH="$VENV_PATH/lib/python3.10/site-packages:$PYTHONPATH"
export PYTHONPATH="/home/jed/workspace/drone-hover:$PYTHONPATH"

which python3
python3 --version
# Commands to execute as part of your task

source /home/jed/workspace/airevolve/.venv/bin/activate

if (($SLURM_ARRAY_TASK_ID == 0)) ; then
    srun python3 examples/run_evolution.py --population-size 12 --generations 40 --gate-cfg circle --num-envs 100 --strategy-type plus --symmetry none --log-dir ./logs/
elif (($SLURM_ARRAY_TASK_ID == 1)) ; then
    srun python3 examples/run_evolution.py --population-size 12 --generations 40 --gate-cfg circle --num-envs 100 --strategy-type plus --symmetry none --log-dir ./logs/
elif (($SLURM_ARRAY_TASK_ID == 2)) ; then
    srun python3 examples/run_evolution.py --population-size 12 --generations 40 --gate-cfg circle --num-envs 100 --strategy-type plus --symmetry none --log-dir ./logs/
elif (($SLURM_ARRAY_TASK_ID == 3)) ; then
    srun python3 examples/run_evolution.py --population-size 12 --generations 40 --gate-cfg circle --num-envs 100 --strategy-type plus --symmetry none --log-dir ./logs/
elif (($SLURM_ARRAY_TASK_ID == 4)) ; then
    srun python3 examples/run_evolution.py --population-size 12 --generations 40 --gate-cfg circle --num-envs 100 --strategy-type plus --symmetry none --log-dir ./logs/
elif (($SLURM_ARRAY_TASK_ID == 5)) ; then
    srun python3 examples/run_evolution.py --population-size 12 --generations 40 --gate-cfg slalom --num-envs 100 --strategy-type plus --symmetry none --log-dir ./logs/
elif (($SLURM_ARRAY_TASK_ID == 6)) ; then
    srun python3 examples/run_evolution.py --population-size 12 --generations 40 --gate-cfg slalom --num-envs 100 --strategy-type plus --symmetry none --log-dir ./logs/
elif (($SLURM_ARRAY_TASK_ID == 7)) ; then
    srun python3 examples/run_evolution.py --population-size 12 --generations 40 --gate-cfg slalom --num-envs 100 --strategy-type plus --symmetry none --log-dir ./logs/
elif (($SLURM_ARRAY_TASK_ID == 8)) ; then
    srun python3 examples/run_evolution.py --population-size 12 --generations 40 --gate-cfg slalom --num-envs 100 --strategy-type plus --symmetry none --log-dir ./logs/
elif (($SLURM_ARRAY_TASK_ID == 9)) ; then
    srun python3 examples/run_evolution.py --population-size 12 --generations 40 --gate-cfg slalom --num-envs 100 --strategy-type plus --symmetry none --log-dir ./logs/
elif (($SLURM_ARRAY_TASK_ID == 10)) ; then
    srun python3 examples/run_evolution.py --population-size 12 --generations 40 --gate-cfg backandforth --num-envs 100 --strategy-type plus --symmetry none --log-dir ./logs/
elif (($SLURM_ARRAY_TASK_ID == 11)) ; then
    srun python3 examples/run_evolution.py --population-size 12 --generations 40 --gate-cfg backandforth --num-envs 100 --strategy-type plus --symmetry none --log-dir ./logs/
elif (($SLURM_ARRAY_TASK_ID == 12)) ; then
    srun python3 examples/run_evolution.py --population-size 12 --generations 40 --gate-cfg backandforth --num-envs 100 --strategy-type plus --symmetry none --log-dir ./logs/
elif (($SLURM_ARRAY_TASK_ID == 13)) ; then
    srun python3 examples/run_evolution.py --population-size 12 --generations 40 --gate-cfg backandforth --num-envs 100 --strategy-type plus --symmetry none --log-dir ./logs/
elif (($SLURM_ARRAY_TASK_ID == 14)) ; then
    srun python3 examples/run_evolution.py --population-size 12 --generations 40 --gate-cfg backandforth --num-envs 100 --strategy-type plus --symmetry none --log-dir ./logs/
elif (($SLURM_ARRAY_TASK_ID == 15)) ; then
    srun python3 examples/run_evolution.py --population-size 12 --generations 40 --gate-cfg figure8 --num-envs 100 --strategy-type plus --symmetry none --log-dir ./logs/
elif (($SLURM_ARRAY_TASK_ID == 16)) ; then
    srun python3 examples/run_evolution.py --population-size 12 --generations 40 --gate-cfg figure8 --num-envs 100 --strategy-type plus --symmetry none --log-dir ./logs/
elif (($SLURM_ARRAY_TASK_ID == 17)) ; then
    srun python3 examples/run_evolution.py --population-size 12 --generations 40 --gate-cfg figure8 --num-envs 100 --strategy-type plus --symmetry none --log-dir ./logs/
elif (($SLURM_ARRAY_TASK_ID == 18)) ; then
    srun python3 examples/run_evolution.py --population-size 12 --generations 40 --gate-cfg figure8 --num-envs 100 --strategy-type plus --symmetry none --log-dir ./logs/
elif (($SLURM_ARRAY_TASK_ID == 19)) ; then
    srun python3 examples/run_evolution.py --population-size 12 --generations 40 --gate-cfg figure8 --num-envs 100 --strategy-type plus --symmetry none --log-dir ./logs/
elif (($SLURM_ARRAY_TASK_ID == 20)) ; then
    srun python3 examples/run_evolution.py --population-size 12 --generations 40 --gate-cfg figure8 --num-envs 100 --strategy-type plus --symmetry yz --log-dir ./logs/
elif (($SLURM_ARRAY_TASK_ID == 21)) ; then
    srun python3 examples/run_evolution.py --population-size 12 --generations 40 --gate-cfg figure8 --num-envs 100 --strategy-type plus --symmetry yz --log-dir ./logs/
elif (($SLURM_ARRAY_TASK_ID == 22)) ; then
    srun python3 examples/run_evolution.py --population-size 12 --generations 40 --gate-cfg figure8 --num-envs 100 --strategy-type plus --symmetry yz --log-dir ./logs/
elif (($SLURM_ARRAY_TASK_ID == 23)) ; then
    srun python3 examples/run_evolution.py --population-size 12 --generations 40 --gate-cfg figure8 --num-envs 100 --strategy-type plus --symmetry yz --log-dir ./logs/
elif (($SLURM_ARRAY_TASK_ID == 24)) ; then
    srun python3 examples/run_evolution.py --population-size 12 --generations 40 --gate-cfg figure8 --num-envs 100 --strategy-type plus --symmetry yz --log-dir ./logs/
fi

echo "done"