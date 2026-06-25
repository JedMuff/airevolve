#!/bin/bash
#SBATCH -J airevolve_master
#SBATCH -p genoa
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 10:00:00
#SBATCH --output=./logs/master_run_%j.out
#SBATCH --error=./logs/master_run_%j.err

module load 2022
module load Python/3.10.4-GCCcore-11.3.0

cd /projects/prjs2127/airevolve
source venv/bin/activate

# Force PyTorch and NumPy to use exactly 1 thread per drone
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p ./logs

echo "Starting all 86 drones simultaneously..."

# Launch all 86 training tasks in the background
for i in {1..86}; do
    python retrain_pareto_launcher.py $i > logs/drone_run_${i}.out 2> logs/drone_run_${i}.err &
done

# This tells the script to wait until all 86 background tasks finish
wait

echo "All 86 drones have finished training!"