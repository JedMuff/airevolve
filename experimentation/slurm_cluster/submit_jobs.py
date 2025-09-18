#!/usr/bin/env python3
"""
Script to generate and submit SLURM jobs for drone evaluation.
Creates optimized job batching to utilize 6 nodes × 3 GPUs = 18 parallel jobs.
"""

import os
import subprocess
import itertools
from math import ceil

def create_slurm_script(job_configs, base_dir="drone_evaluations"):
    """Create SLURM array job script with 3 evaluations per GPU"""
    
    # Each array task will handle 3 evaluations on 1 GPU
    num_array_tasks = ceil(len(job_configs) / 3)
    
    script_content = f"""#!/bin/bash

#SBATCH --output=out_files/drone_eval-%A_%a.out
#SBATCH --error=out_files/drone_eval-%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --array=0-{num_array_tasks-1}
#SBATCH --exclude=ripper2

VENV_PATH=/home/jed/workspace/airevolve/.venv

echo "Node: $(hostname)"
echo "Using Python from: $VENV_PATH"

export PATH="$VENV_PATH/bin:$PATH"
export PYTHONPATH="$VENV_PATH/lib/python3.10/site-packages:$PYTHONPATH"
export PYTHONPATH="/home/jed/workspace/drone-hover:$PYTHONPATH"

which python3
python3 --version

# Create output directory if it doesn't exist
mkdir -p out_files

source /home/jed/workspace/airevolve/.venv/bin/activate

echo "Starting drone evaluation array job"
echo "Array task ID: $SLURM_ARRAY_TASK_ID"
echo "Job started at: $(date)"

# Define all job configurations (flattened arrays)
"""
    
    # Flatten all job configurations
    designs = []
    tasks = []
    reps = []
    
    for design, task, rep in job_configs:
        designs.append(f'"{design}"')
        tasks.append(f'"{task}"')
        reps.append(str(rep))
    
    script_content += f"""
designs=({' '.join(designs)})
tasks=({' '.join(tasks)})
reps=({' '.join(reps)})

# Calculate which 3 evaluations this array task should handle
start_idx=$((SLURM_ARRAY_TASK_ID * 3))
end_idx=$((start_idx + 3))

echo "Array task $SLURM_ARRAY_TASK_ID handling evaluations $start_idx to $((end_idx-1))"

# Function to run evaluation in background
run_evaluation() {{
    local design=$1
    local task=$2
    local rep=$3
    local eval_num=$4
    
    echo "Starting evaluation $eval_num: $design, $task, rep $rep"
    
    python3 main_evaluation.py \\
        --design "$design" \\
        --task "$task" \\
        --rep $rep \\
        --base_dir "{base_dir}"
    
    local exit_code=$?
    echo "Completed evaluation $eval_num: $design, $task, rep $rep (exit code: $exit_code)"
    return $exit_code
}}

# Run up to 3 evaluations in parallel on this GPU
pids=()
eval_results=()

for i in $(seq $start_idx $((end_idx-1))); do
    # Check if this index exists in our arrays
    if [ $i -lt ${{#designs[@]}} ]; then
        design=${{designs[$i]}}
        task=${{tasks[$i]}}
        rep=${{reps[$i]}}
        
        echo "Launching evaluation $i: $design $task rep_$rep"
        run_evaluation "$design" "$task" "$rep" "$i" &
        pids+=($!)
        eval_results+=("$i")
    fi
done

# Wait for all evaluations to complete
echo "Waiting for ${{#pids[@]}} evaluations to complete..."
failed_count=0

for i in "${{!pids[@]}}"; do
    pid=${{pids[$i]}}
    eval_num=${{eval_results[$i]}}
    
    wait $pid
    exit_code=$?
    
    if [ $exit_code -ne 0 ]; then
        echo "ERROR: Evaluation $eval_num failed with exit code $exit_code"
        ((failed_count++))
    else
        echo "SUCCESS: Evaluation $eval_num completed successfully"
    fi
done

echo "Array task $SLURM_ARRAY_TASK_ID completed at: $(date)"
echo "Results: ${{#pids[@]}} total, $failed_count failed, $(((${{#pids[@]}} - failed_count))) succeeded"

# Exit with error code if any evaluations failed
if [ $failed_count -gt 0 ]; then
    echo "WARNING: $failed_count evaluations failed in this array task"
    exit 1
else
    echo "All evaluations in this array task completed successfully"
    exit 0
fi
"""
    
    return script_content

def main():
    # Define all combinations
    designs = [#'slalom_design_rank_02', 'shuttlerun_design_rank_01', 'circle_design_rank_01', 
               #'design_rank_01', 'spiderhex', 'crosshexcopter'
        'sym_circle_design_rank_01',
        'sym_figure8_design_rank_01',
        'sym_shuttlerun_design_rank_01',
        'sym_slalom_design_rank_01']
    tasks = ['backandforth', 'figure8', 'circle', 'slalom']
    repetitions = list(range(10))  # 0-9
    
    # Create all job combinations
    all_jobs = list(itertools.product(designs, tasks, repetitions))
    print(f"Total evaluations needed: {len(all_jobs)}")
    
    # Create output directories
    os.makedirs("slurm_scripts", exist_ok=True)
    os.makedirs("out_files", exist_ok=True)
    
    # With 3 evaluations per GPU, we need fewer array jobs
    # Each array task handles 3 evaluations in parallel
    evals_per_gpu = 3
    max_concurrent_gpus = 18  # 6 nodes × 3 GPUs per node
    max_concurrent_evals = max_concurrent_gpus * evals_per_gpu  # 54 parallel evaluations!
    
    # Split jobs into manageable array sizes
    # Each array job will have multiple array tasks, each running 3 evaluations
    max_evals_per_array = max_concurrent_evals  # 54 evaluations per array job
    num_arrays = ceil(len(all_jobs) / max_evals_per_array)
    
    print(f"Creating {num_arrays} SLURM array jobs")
    print(f"Each GPU will run {evals_per_gpu} evaluations in parallel")  
    print(f"Maximum concurrent evaluations: {max_concurrent_evals}")
    print(f"Expected completion time: ~4 hours (with sufficient nodes)")
    
    job_scripts = []
    
    for array_id in range(num_arrays):
        start_idx = array_id * max_evals_per_array
        end_idx = min(start_idx + max_evals_per_array, len(all_jobs))
        array_jobs = all_jobs[start_idx:end_idx]
        
        if not array_jobs:  # Skip empty arrays
            continue
            
        script_content = create_slurm_script(array_jobs)
        script_filename = f"slurm_scripts/drone_eval_array_{array_id:02d}.sh"
        
        with open(script_filename, 'w') as f:
            f.write(script_content)
        
        # Make script executable
        os.chmod(script_filename, 0o755)
        job_scripts.append(script_filename)
        
        num_array_tasks = ceil(len(array_jobs) / evals_per_gpu)
        print(f"Array {array_id:2d}: {len(array_jobs):3d} evaluations in {num_array_tasks:2d} array tasks -> {script_filename}")
    
    # Create submission script
    submit_script = """#!/bin/bash
# Script to submit all drone evaluation array jobs

echo "Submitting optimized drone evaluation jobs..."
echo "Each GPU runs 3 evaluations in parallel for maximum efficiency!"
echo "Total arrays: """ + str(len(job_scripts)) + """"
echo

"""
    
    for i, script in enumerate(job_scripts):
        submit_script += f"""
echo "Submitting array {i+1}/{len(job_scripts)}: {script}"
sbatch {script}
sleep 2  # Small delay between submissions
"""
    
    submit_script += """
echo "All optimized array jobs submitted!"
echo "Expected performance:"
echo "  - Up to 54 evaluations running simultaneously"  
echo "  - ~4 hours total completion time (with 6 nodes available)"
echo "  - Each GPU efficiently runs 3 evaluations in parallel"
echo
echo "Monitor with: squeue -u $USER"
echo "Check logs in: out_files/"
echo "Results will be in: drone_evaluations/"
"""
    
    with open("submit_all_jobs.sh", 'w') as f:
        f.write(submit_script)
    
    os.chmod("submit_all_jobs.sh", 0o755)
    
    print("\n" + "="*70)
    print("OPTIMIZED SLURM ARRAY JOB GENERATION COMPLETE!")
    print("="*70)
    print(f"Created {len(job_scripts)} array job scripts in slurm_scripts/")
    print(f"Total evaluations: {len(all_jobs)}")
    print(f"Evaluations per GPU: {evals_per_gpu} (parallel)")
    print(f"Maximum concurrent evaluations: {max_concurrent_evals}")
    print(f"Time limit: 4 hours per array task")
    print(f"Excludes: ripper2 node")
    
    print(f"\n🚀 PERFORMANCE BOOST:")
    print(f"Previous approach: ~18 evaluations in parallel")
    print(f"New approach:      ~54 evaluations in parallel")
    print(f"Speed improvement: ~3x faster!")
    
    print("\nTo submit all jobs:")
    print("  ./submit_all_jobs.sh")
    print("\nTo submit individual arrays:")
    for script in job_scripts[:3]:  # Show first 3 as examples
        print(f"  sbatch {script}")
    if len(job_scripts) > 3:
        print("  ...")
    
    print(f"\nLogs will be in:")
    print("  out_files/drone_eval-JOBID_ARRAYINDEX.out")
    print("  out_files/drone_eval-JOBID_ARRAYINDEX.err")
    
    print(f"\nMonitor jobs with:")
    print("  squeue -u $USER")
    print("  ./monitor_progress.sh")

if __name__ == "__main__":
    main()