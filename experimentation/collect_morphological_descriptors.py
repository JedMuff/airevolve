"""
Clean script to collect morphological descriptors data from experiments
and save as CSV files for each experiment directory.
Uses parallelization to compute metrics for individuals concurrently.
Now includes fitness data alongside morphological descriptors.
"""

import os
import pickle
import re
import functools
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

# Import morphological descriptor functions
from airevolve.inspection_tools.morphological_descriptors.central_symmetry import compute_symmetry as compute_central_symmetry
from airevolve.inspection_tools.morphological_descriptors.biradial_symmetry import compute_symmetry as compute_biradial_symmetry
from airevolve.inspection_tools.morphological_descriptors.hovering_info import max_thrust_to_weight, controlability


def convert_str_to_nparray(s):
    """Convert string representation of numpy array back to numpy array."""
    s = s.replace("nan", "np.nan")
    s = s.replace("0. ", "0.0")
    s = re.sub(r"(-?\d|np\.nan)\s+(?=-?\d|np\.nan)", r"\1, ", s)
    s = re.sub(r"(\d\.)\s+(?=\d)", r"\1, ", s)
    s = re.sub(r"\]\s+\[", "], [", s)
    s = s.replace(". ", " ")
    s = s.replace(".,", ",")
    
    arr = np.array(eval(s))
    
    if np.isnan(arr).any():
        print("Warning: np.nan found in array")
    
    return arr


def aggregate_population_with_fitness(df):
    """Aggregate population data by generation, including fitness data."""
    df['offspring'] = df['offspring'].apply(convert_str_to_nparray)
    
    # Also convert fitness data if it's stored as strings
    if 'fitness' in df.columns and df['fitness'].dtype == 'object':
        df['fitness'] = df['fitness'].apply(convert_str_to_nparray)
    
    grouped_offspring = df.groupby('generation')['offspring'].apply(np.array)
    grouped_fitness = df.groupby('generation')['fitness'].apply(np.array) if 'fitness' in df.columns else None
    
    max_len = 24
    pop_array = np.full((len(grouped_offspring), max_len), np.nan, dtype=object)
    fitness_array = np.full((len(grouped_offspring), max_len), np.nan) if grouped_fitness is not None else None
    
    for i, pop_list in enumerate(grouped_offspring):
        pop_array[i, :len(pop_list)] = pop_list[:24]
        
        if grouped_fitness is not None:
            fitness_list = grouped_fitness.iloc[i]
            if len(fitness_list) > 0:
                fitness_array[i, :len(fitness_list)] = fitness_list[:24]
    
    return pop_array, fitness_array


def compute_individual_descriptors(args):
    """Compute morphological descriptors for a single individual."""
    individual, generation_idx, individual_idx, fitness_value, md_funcs, md_titles = args
    
    if individual is None or np.isnan(individual).any():
        return None
    
    row_data = {
        'generation': generation_idx,
        'individual': individual_idx,
        'fitness': fitness_value if fitness_value is not None and not np.isnan(fitness_value) else np.nan
    }
    
    for func, title in zip(md_funcs, md_titles):
        try:
            value = func(individual)
            row_data[title] = value
        except Exception as e:
            print(f"Error computing {title} for gen {generation_idx}, ind {individual_idx}: {e}")
            row_data[title] = np.nan
    
    return row_data


def compute_morphological_descriptors(pop_data, fitness_data, md_funcs, md_titles, n_workers=None):
    """Compute morphological descriptors for population data using parallelization."""
    if n_workers is None:
        n_workers = min(cpu_count(), 8)  # Cap at 8 to avoid overwhelming the system
    
    print(f"Computing morphological descriptors using {n_workers} workers...")
    
    # Prepare arguments for parallel processing
    args_list = []
    for generation_idx, generation in enumerate(pop_data):
        for individual_idx, individual in enumerate(generation):
            # Get corresponding fitness value
            fitness_value = None
            if fitness_data is not None and generation_idx < len(fitness_data):
                if individual_idx < len(fitness_data[generation_idx]):
                    fitness_value = fitness_data[generation_idx][individual_idx]
            
            args_list.append((individual, generation_idx, individual_idx, fitness_value, md_funcs, md_titles))
    
    md_data = []
    
    # Process individuals in parallel
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all tasks
        future_to_args = {
            executor.submit(compute_individual_descriptors, args): args 
            for args in args_list
        }
        
        # Collect results as they complete
        completed = 0
        total = len(args_list)
        
        for future in as_completed(future_to_args):
            try:
                result = future.result()
                if result is not None:
                    md_data.append(result)
                
                completed += 1
                if completed % 100 == 0:  # Progress update every 100 individuals
                    print(f"Processed {completed}/{total} individuals ({100*completed/total:.1f}%)")
                    
            except Exception as e:
                args = future_to_args[future]
                print(f"Error processing individual gen {args[1]}, ind {args[2]}: {e}")
    
    print(f"Completed processing {len(md_data)} valid individuals")
    return pd.DataFrame(md_data)


def process_experiment_directory(experiment_dir, save_dir, n_workers=None):
    """Process all runs in an experiment directory and save combined morphological descriptors."""
    print(f"Processing experiment directory: {experiment_dir}")
    
    # Define reduced morphological descriptors
    compute_bilateral_symmetry = functools.partial(compute_biradial_symmetry, fixed_plane=[1,0,0,0])
    md_funcs = [compute_central_symmetry, compute_bilateral_symmetry, max_thrust_to_weight, controlability]
    md_titles = ["Central_Asymmetry", "Bilateral_Asymmetry", "Max_Thrust_to_Weight", "Controlability"]
    
    all_md_data = []
    
    # List all run directories
    run_dirs = [d for d in os.listdir(experiment_dir) if d != ".DS_Store"]
    
    for run_idx, run_dir in enumerate(run_dirs):
        run_path = os.path.join(experiment_dir, run_dir)
        evolution_data_path = os.path.join(run_path, "evolution_data.csv")
        
        if not os.path.exists(evolution_data_path):
            print(f"Warning: evolution_data.csv not found in {run_path}")
            continue
        
        print(f"Processing run {run_idx + 1}/{len(run_dirs)}: {run_dir}")
        
        # Load evolution data
        evo_data = pd.read_csv(evolution_data_path)
        
        # Print column names to debug
        print(f"Evolution data columns: {evo_data.columns.tolist()}")
        
        # Aggregate population data with fitness
        pop_data, fitness_data = aggregate_population_with_fitness(evo_data)
        
        # Compute morphological descriptors with parallelization
        md_df = compute_morphological_descriptors(pop_data, fitness_data, md_funcs, md_titles, n_workers=n_workers)
        md_df['run'] = run_idx
        
        all_md_data.append(md_df)
    
    # Combine all runs
    if all_md_data:
        combined_md_data = pd.concat(all_md_data, ignore_index=True)
        
        # Save to CSV
        output_path = os.path.join(save_dir, "morphological_descriptors_data.csv")
        combined_md_data.to_csv(output_path, index=False)
        print(f"Saved morphological descriptors data to: {output_path}")
        print(f"Total records: {len(combined_md_data)}")
        
        # Print summary statistics
        if 'fitness' in combined_md_data.columns:
            valid_fitness = combined_md_data['fitness'].dropna()
            if len(valid_fitness) > 0:
                print(f"Fitness statistics: mean={valid_fitness.mean():.3f}, std={valid_fitness.std():.3f}")
                print(f"Fitness range: [{valid_fitness.min():.3f}, {valid_fitness.max():.3f}]")
            else:
                print("Warning: No valid fitness values found")
        
        return combined_md_data
    else:
        print(f"No valid data found in {experiment_dir}")
        return None


def main():
    """Main function to process all experiment directories."""
    
    # Define experiment directories
    experiment_dirs = {
        "asym_figure8": "data_backup/asym_figure8/",
        "asym_circle": "data_backup/asym_circle/",
        "asym_slalom": "data_backup/asym_slalom/",
        "asym_backnforth": "data_backup/asym_backnforth/",
    }
    
    # Set number of workers (None for auto-detection, or specify a number)
    n_workers = min(cpu_count(), 24)  # Will use min(cpu_count(), 8)
    
    print(f"Available CPU cores: {cpu_count()}")
    print(f"Using {n_workers if n_workers else min(cpu_count(), 24)} worker processes")
    
    # Process each experiment directory
    for exp_name, exp_dir in experiment_dirs.items():
        if os.path.exists(exp_dir):
            print(f"\n{'='*50}")
            print(f"Starting {exp_name}")
            print(f"{'='*50}")
            process_experiment_directory(exp_dir, exp_dir, n_workers=n_workers)
        else:
            print(f"Warning: Directory {exp_dir} does not exist")


if __name__ == "__main__":
    main()