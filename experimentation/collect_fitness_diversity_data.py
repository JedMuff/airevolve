"""
Clean script to collect fitness and diversity data from multiple experimental runs
and save them as CSV files for each experiment.
"""

import os
import re
import numpy as np
import pandas as pd
from airevolve.evolution_tools.evaluators.edit_distance import compute_individual_population_edit_distance


def convert_str_to_nparray(s):
    """Convert string representation of array to numpy array."""
    # Replace "nan" with "np.nan" for NumPy compatibility
    s = s.replace("nan", "np.nan")
    
    # Add commas where necessary
    s = s.replace("0. ", "0.0")
    s = re.sub(r"(-?\d|np\.nan)\s+(?=-?\d|np\.nan)", r"\1, ", s)
    s = re.sub(r"(\d\.)\s+(?=\d)", r"\1, ", s)
    s = re.sub(r"\]\s+\[", "], [", s)
    s = s.replace(". ", " ")
    s = s.replace(".,", ",")
    
    # Convert to NumPy array
    arr = np.array(eval(s))
    
    if np.isnan(arr).any():
        print("Warning: np.nan found in array")
    
    return arr


def aggregate_fitness(df, column='fitness', max_individuals=24):
    """Aggregate fitness data by generation."""
    grouped = df.groupby('generation')[column].apply(np.array)
    
    # Create 2D array with NaN for varying lengths
    fitness_array = np.full((len(grouped), max_individuals), np.nan)
    
    for i, fitness_list in enumerate(grouped):
        fitness_array[i, :len(fitness_list)] = fitness_list[:max_individuals]
    
    return fitness_array


def aggregate_population(df, max_individuals=24):
    """Aggregate population data by generation."""
    df['offspring'] = df['offspring'].apply(convert_str_to_nparray)
    grouped = df.groupby('generation')['offspring'].apply(np.array)
    
    pop_array = np.full((len(grouped), max_individuals), np.nan, dtype=object)
    
    for i, pop_list in enumerate(grouped):
        pop_array[i, :len(pop_list)] = pop_list[:max_individuals]
    
    return pop_array


def calculate_diversity_data(pop_data, min_max_params=None):
    """Calculate diversity metrics from population data using edit distance functions."""
    if min_max_params is None:
        # Default parameters as used in the original template
        min_max_params = np.array([[0.09, 0.4], [0, 2*np.pi], [0, 2*np.pi], 
                                  [0, 2*np.pi], [0, 2*np.pi], [0, 1]])
    
    min_vals = min_max_params[:, 0]
    max_vals = min_max_params[:, 1]
    diversity_values = []
    
    for generation in pop_data:
        # Filter out NaN individuals
        valid_individuals = [ind for ind in generation if not pd.isna(ind).any() if isinstance(ind, np.ndarray)]
        
        if len(valid_individuals) == 0:
            diversity_values.append(np.nan)
            continue
        
        novelties = np.zeros(len(valid_individuals))
        for i, ind in enumerate(valid_individuals):
            try:
                novelties[i] = compute_individual_population_edit_distance(ind, valid_individuals, min_vals, max_vals)
            except Exception as e:
                print(f"Warning: Error calculating diversity for individual {i}: {e}")
                novelties[i] = np.nan
        
        # Calculate mean diversity for this generation
        mean_diversity = np.nanmean(novelties) if len(novelties) > 0 else np.nan
        diversity_values.append(mean_diversity)
    
    return np.array(diversity_values)


def process_experiment_runs(experiment_dir, experiment_name, save_dir):
    """Process all runs for a single experiment and save aggregated data."""
    print(f"Processing experiment: {experiment_name}")
    
    if not os.path.exists(experiment_dir):
        print(f"Warning: Directory {experiment_dir} does not exist")
        return
    
    all_fitness_data = []
    all_diversity_data = []
    
    # Process each run in the experiment directory
    run_folders = [d for d in os.listdir(experiment_dir) if d != ".DS_Store"]
    
    for run_folder in run_folders:
        run_path = os.path.join(experiment_dir, run_folder)
        evolution_file = os.path.join(run_path, "evolution_data.csv")
        
        if not os.path.exists(evolution_file):
            print(f"Warning: {evolution_file} not found, skipping run {run_folder}")
            continue
        
        print(f"  Processing run: {run_folder}")
        
        # Load evolution data
        evo_data = pd.read_csv(evolution_file)
        
        # Aggregate fitness and population data
        fit_data = aggregate_fitness(evo_data, column='fitness')
        pop_data = aggregate_population(evo_data)
        
        # Calculate fitness statistics per generation
        max_fitness = np.nanmax(fit_data, axis=1)
        mean_fitness = np.nanmean(fit_data, axis=1)
        std_fitness = np.nanstd(fit_data, axis=1)
        
        # Convert population data to proper numpy arrays
        pop_data = np.array([[np.array(ind) for ind in pop] for pop in pop_data])
        
        # Calculate diversity using EditDistanceEvaluator
        diversity_data = calculate_diversity_data(pop_data)
        
        # Store data for this run
        run_data = {
            'experiment': experiment_name,
            'run': run_folder,
            'generation': range(len(max_fitness)),
            'max_fitness': max_fitness,
            'mean_fitness': mean_fitness,
            'std_fitness': std_fitness,
            'diversity': diversity_data[:len(max_fitness)]  # Ensure same length
        }
        
        all_fitness_data.append(pd.DataFrame(run_data))
    
    if not all_fitness_data:
        print(f"No valid runs found for experiment {experiment_name}")
        return
    
    # Combine all runs
    combined_data = pd.concat(all_fitness_data, ignore_index=True)
    
    # Save to CSV
    output_file = os.path.join(save_dir, f"{experiment_name}/fitness_diversity_data.csv")
    combined_data.to_csv(output_file, index=False)
    print(f"  Saved data to: {output_file}")
    
    return combined_data


def main():
    """Main function to process all experiments."""
    
    # Define experiment directories and names
    experiment_dirs = {
        # "asym_figure8": "data_backup/asym_figure8",
        # "asym_circle": "data_backup/asym_circle/",
        # "asym_slalom": "data_backup/asym_slalom/",
        # "asym_backnforth": "data_backup/asym_backnforth/",
        "sym_data/sym_fig8": "data_backup/sym_data/sym_fig8/",
        "sym_data/sym_circle": "data_backup/sym_data/sym_circle/",
        "sym_data/sym_slalom": "data_backup/sym_data/sym_slalom/",
        "sym_data/sym_shuttlerun": "data_backup/sym_data/sym_shuttlerun/",
    }
    
    # Output directory for CSV files
    save_dir = "data_backup/"
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Process each experiment
    all_experiment_data = []
    
    for experiment_name, experiment_dir in experiment_dirs.items():
        experiment_data = process_experiment_runs(experiment_dir, experiment_name, save_dir)
        if experiment_data is not None:
            all_experiment_data.append(experiment_data)
    
    # Optionally, create a combined file with all experiments
    # if all_experiment_data:
    #     combined_all = pd.concat(all_experiment_data, ignore_index=True)
    #     combined_file = os.path.join(save_dir, "all_experiments_fitness_diversity_data.csv")
    #     combined_all.to_csv(combined_file, index=False)
    #     print(f"Combined data saved to: {combined_file}")
    
    print("Processing complete!")


if __name__ == "__main__":
    main()