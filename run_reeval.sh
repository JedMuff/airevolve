#!/bin/bash
#SBATCH -J airevolve_fix
#SBATCH -p genoa
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -t 10:00:00
#SBATCH --mem=8G
#SBATCH --array=0-9
#SBATCH --output=./logs/%x_%A_%a.out
#SBATCH --error=./logs/%x_%A_%a.err

module load 2022
module load Python/3.10.4-GCCcore-11.3.0

source /projects/prjs2127/airevolve/venv/bin/activate

mkdir -p ./logs

# Execute the parallel Python script
srun python reevaluate_corrected.py $SLURM_ARRAY_TASK_ID