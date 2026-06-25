#!/bin/bash
#SBATCH -J airevolve_pareto
#SBATCH -p genoa
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -t 10:00:00
#SBATCH --mem=4G
#SBATCH --array=1-86
#SBATCH --output=./logs/pareto_%A_%a.out
#SBATCH --error=./logs/pareto_%A_%a.err

module load 2022
module load Python/3.10.4-GCCcore-11.3.0

cd /projects/prjs2127/airevolve
source venv/bin/activate

# Create logs directory if it doesn't exist
mkdir -p ./logs

# Pass the array ID to the python launcher
srun python retrain_pareto_launcher.py $SLURM_ARRAY_TASK_ID