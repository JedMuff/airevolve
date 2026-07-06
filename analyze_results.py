import os
import sys
import glob
import subprocess

def main():
    # SLURM script outputs to $SLURM_SUBMIT_DIR/results/ which defaults to "results" relative to repo root
    base_dir = "results"
    
    # We have two main tasks
    tasks = ["figure8", "shuttlerun"]
    
    # Scripts that process ALL runs for a given task
    multi_run_scripts = [
        "examples/visualization/pareto_fronts/plot_objectives_all.py",
        "examples/visualization/pareto_fronts/plot_diversity_all.py",
        "examples/visualization/plot_learning_metrics.py",
        "examples/visualization/plot_symmetry.py"
    ]
    
    # Scripts that process a SINGLE run
    single_run_scripts = [
        "examples/visualization/pareto_fronts/plot_pareto_evolution.py",
        "examples/visualization/pareto_fronts/plot_hypervolume.py",
        "examples/visualization/pareto_fronts/plot_pareto_3d.py",
        "examples/visualization/pareto_fronts/plot_pareto_diversity.py",
        "examples/visualization/pareto_fronts/plot_swarm_evolution.py",
        "examples/visualization/pareto_fronts/plot_morphological_diversity.py",
        "examples/visualization/pareto_fronts/plot_random_pareto.py"
    ]

    print("==================================================")
    print("         STARTING ANALYSIS PIPELINE               ")
    print("==================================================")
    
    # Run multi-run scripts for each task
    for task in tasks:
        print(f"\n>>> Processing MULTI-RUN scripts for task: {task.upper()}")
        
        # Find all run directories for this task
        run_dirs = sorted(glob.glob(os.path.join(base_dir, f"exp_standard_ppo_power_ea_{task}_rep*")))
        if not run_dirs:
            print(f"No run directories found for task {task} in {base_dir}")
            continue
            
        for script in multi_run_scripts:
            script_path = os.path.abspath(script)
            if not os.path.exists(script_path):
                print(f"Script not found: {script_path}")
                continue
                
            print(f"  Running {os.path.basename(script)}...")
            cmd = ["python", script_path, "--base_dir", base_dir, "--task_name", task, "--run_dirs"] + run_dirs
                
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"  [ERROR] {os.path.basename(script)} failed: {e}")

    # Run single-run scripts for ALL 10 experiments
    print("\n>>> Processing SINGLE-RUN scripts for all individual runs")
    all_runs = sorted(glob.glob(os.path.join(base_dir, "exp_standard_ppo_power_ea_*_rep*")))
    
    for run_dir in all_runs:
        run_name = os.path.basename(run_dir)
        print(f"\n--- Analyzing Run: {run_name} ---")
        
        csv_path = os.path.join(run_dir, "evolution_data.csv")
        if not os.path.exists(csv_path):
            print(f"  [WARNING] {csv_path} not found. Skipping...")
            continue
            
        for script in single_run_scripts:
            script_path = os.path.abspath(script)
            if not os.path.exists(script_path):
                print(f"  Script not found: {script_path}")
                continue
                
            print(f"  Running {os.path.basename(script)}...")
            cmd = ["python", script_path, csv_path]
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"  [ERROR] {os.path.basename(script)} failed for {run_name}: {e}")

    print("\n==================================================")
    print("         ANALYSIS PIPELINE COMPLETE               ")
    print("==================================================")

if __name__ == "__main__":
    main()
