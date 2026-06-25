import os
import sys
import glob
from pathlib import Path
import numpy as np

# Force exactly 1 CPU thread to save your SLURM budget
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import torch
torch.set_num_threads(1)

# Ensure the repository root is in the path
REPO_ROOT = Path("/projects/prjs2127/airevolve")
sys.path.insert(0, str(REPO_ROOT))

# Import the exact training function from your gate_train.py script
from airevolve.evolution_tools.evaluators.gate_train import evaluate_individual

def main():
    if len(sys.argv) < 2:
        print("Error: Missing SLURM_ARRAY_TASK_ID")
        sys.exit(1)
    
    # SLURM array is 1-indexed (1 to 86), Python lists are 0-indexed
    task_id = int(sys.argv[1]) - 1 
    
    # Gather all 86 individual directories
    base_dir = REPO_ROOT / "re-train"
    figure8_dirs = sorted(glob.glob(str(base_dir / "figure8" / "rep*")))
    shuttle_dirs = sorted(glob.glob(str(base_dir / "shuttlerun" / "rep*")))
    all_dirs = figure8_dirs + shuttle_dirs
    
    if task_id < 0 or task_id >= len(all_dirs):
        print(f"Error: task_id {task_id+1} out of bounds (found {len(all_dirs)} folders)")
        sys.exit(1)
        
    target_dir = Path(all_dirs[task_id])
    print(f"Task ID {task_id+1} assigned to: {target_dir}")
    
    # Map the environment configurations
    if "shuttlerun" in target_dir.parent.name:
        gate_cfg = "backandforth"
    else:
        gate_cfg = "figure8"
        
    # Load the specific genome
    genome_path = target_dir / "genome.npy"
    if not genome_path.exists():
        print(f"Error: {genome_path} not found!")
        sys.exit(1)
        
    genome = np.load(genome_path, allow_pickle=True).astype(np.float32)
    print(f"Running evaluate_individual for {target_dir.name} on {gate_cfg}...")
    
    # Launch the 1e8 training loop, forcing output into the target_dir
    fitness = evaluate_individual(
        individual=genome,
        ind_save_dir=str(target_dir),   # Saves logs/policies right next to genome.npy
        training_ts=100_000_000,        # The 1e8 Gold Standard convergence
        num_envs=100,                   # Vectorized batch collection (Insanely fast)
        gate_cfg=gate_cfg,
        device="cpu",                   # Safe single-core execution
        max_steps=1200,
        random_start=True,
        verbose=1,
        progress_bar=False
    )
    
    print(f"Finished {target_dir.name} with fitness: {fitness}")

if __name__ == "__main__":
    main()