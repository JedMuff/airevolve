"""
Optimized script to collect best individuals from evolutionary experiments and save as CSV files.
Uses multiprocessing and performance optimizations for faster execution.
"""

import pickle
import os
import re
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial
import time
from airevolve.evolution_tools.evaluators.edit_distance import compute_edit_distance


def convert_str_to_nparray_fast(s):
    """Optimized string to numpy array conversion."""
    if isinstance(s, np.ndarray):
        return s
    
    # Fast replacements
    s = s.replace("nan", "np.nan").replace("0. ", "0.0")
    
    # Use regex substitution more efficiently
    s = re.sub(r"(-?\d+\.?\d*|np\.nan)\s+(?=-?\d+\.?\d*|np\.nan)", r"\1, ", s)
    s = re.sub(r"\]\s+\[", "], [", s)
    s = s.replace(". ", " ").replace(".,", ",")
    
    try:
        return np.array(eval(s))
    except:
        print(f"Error converting: {s[:100]}...")
        return np.array([])


def process_single_run(args):
    """Process a single run directory - designed for multiprocessing."""
    run_path, k, similarity_threshold, use_similarity = args
    
    evolution_csv = os.path.join(run_path, "evolution_data.csv")
    
    if not os.path.exists(evolution_csv):
        return []
    
    try:
        # Load data more efficiently
        evo_data = pd.read_csv(evolution_csv)
        
        # Quick fitness aggregation - avoid creating full arrays if not needed
        fitness_by_gen = {}
        pop_by_gen = {}
        
        for _, row in evo_data.iterrows():
            gen = row['generation']
            if gen not in fitness_by_gen:
                fitness_by_gen[gen] = []
                pop_by_gen[gen] = []
            
            fitness_by_gen[gen].append(row['fitness'])
            pop_by_gen[gen].append(convert_str_to_nparray_fast(row['offspring']))
        
        # Convert to arrays only for generations we have
        max_gen = max(fitness_by_gen.keys()) + 1
        fit_data = np.full((max_gen, 24), np.nan)
        pop_data = np.full((max_gen, 24), np.nan, dtype=object)
        
        for gen in fitness_by_gen:
            fit_list = fitness_by_gen[gen][:24]
            pop_list = pop_by_gen[gen][:24]
            
            fit_data[gen, :len(fit_list)] = fit_list
            pop_data[gen, :len(pop_list)] = pop_list
        
        # Get top individuals with optimized version
        if use_similarity:
            top_individuals, top_fitnesses, top_indices = get_top_k_individuals_fast(
                pop_data, fit_data, k, similarity_threshold=similarity_threshold
            )
        else:
            top_individuals, top_fitnesses, top_indices = get_top_k_individuals_simple(
                pop_data, fit_data, k
            )
        
        # Return results
        results = []
        for i, (individual, fitness, (gen_idx, ind_idx)) in enumerate(zip(top_individuals, top_fitnesses, top_indices)):
            results.append({
                'run_directory': run_path,
                'rank': i + 1,
                'fitness': fitness,
                'generation': gen_idx,
                'individual_index': ind_idx,
                'individual_data': individual.tolist() if hasattr(individual, 'tolist') else individual
            })
        
        return results
        
    except Exception as e:
        print(f"Error processing {run_path}: {e}")
        return []


def get_top_k_individuals_simple(population, fitnesses, k):
    """Simplified version without similarity checking - much faster."""
    updated_fitnesses = []
    
    for gen_idx, generation in enumerate(population):
        for ind_idx, fitness in enumerate(fitnesses[gen_idx]):
            if not np.isnan(fitness):  # Skip NaN fitnesses
                updated_fitnesses.append((fitness, gen_idx, ind_idx))
    
    # Sort by fitness (descending)
    sorted_fitnesses = sorted(updated_fitnesses, key=lambda x: x[0], reverse=True)
    
    top_k = sorted_fitnesses[:k]
    top_k_population = [population[gen_idx][ind_idx] for _, gen_idx, ind_idx in top_k]
    top_k_fitnesses = [fitness for fitness, _, _ in top_k]
    top_k_indices = [(gen_idx, ind_idx) for _, gen_idx, ind_idx in top_k]
    
    return top_k_population, top_k_fitnesses, top_k_indices


def get_top_k_individuals_fast(population, fitnesses, k, similarity_threshold=3.0):
    """Optimized version with similarity checking."""
    parameter_limits = np.array([[0.09,0.4], [0, 2*np.pi], [0, 2*np.pi], [0, 2*np.pi], [0, 2*np.pi], [0,1]])
    min_vals = parameter_limits[:, 0]
    max_vals = parameter_limits[:, 1]
    
    # Pre-filter valid individuals
    valid_individuals = []
    for gen_idx, generation in enumerate(population):
        for ind_idx, individual in enumerate(generation):
            fitness = fitnesses[gen_idx][ind_idx]
            if not np.isnan(fitness) and individual is not np.nan:
                valid_individuals.append((fitness, gen_idx, ind_idx, individual))
    
    # Sort by fitness
    valid_individuals.sort(key=lambda x: x[0], reverse=True)
    
    selected_population = []
    selected_fitnesses = []
    selected_indices = []
    
    for fitness, gen_idx, ind_idx, individual in valid_individuals:
        if len(selected_population) >= k:
            break
            
        # Check similarity with already selected individuals
        is_similar = False
        for selected_ind in selected_population:
            try:
                if compute_edit_distance(individual, selected_ind, min_vals, max_vals) < similarity_threshold:
                    is_similar = True
                    break
            except:
                # If distance computation fails, assume they're different
                continue
        
        if not is_similar:
            selected_population.append(individual)
            selected_fitnesses.append(fitness)
            selected_indices.append((gen_idx, ind_idx))
    
    return selected_population, selected_fitnesses, selected_indices


def collect_best_individuals_from_experiment_parallel(experiment_dir, k=3, similarity_threshold=3.0, 
                                                    use_similarity=True, n_processes=None):
    """
    Parallel version of best individuals collection.
    """
    if n_processes is None:
        n_processes = min(cpu_count(), 8)  # Don't use too many processes
    
    # Get all run directories
    run_dirs = [os.path.join(experiment_dir, d) for d in os.listdir(experiment_dir) 
                if d != ".DS_Store" and os.path.isdir(os.path.join(experiment_dir, d))]
    
    if not run_dirs:
        print(f"No run directories found in {experiment_dir}")
        return []
    
    print(f"Processing {len(run_dirs)} runs using {n_processes} processes...")
    
    # Prepare arguments for multiprocessing
    args = [(run_path, k, similarity_threshold, use_similarity) for run_path in run_dirs]
    
    # Process in parallel
    with Pool(n_processes) as pool:
        results = pool.map(process_single_run, args)
    
    # Flatten results
    all_best_individuals = []
    for run_results in results:
        all_best_individuals.extend(run_results)
    
    return all_best_individuals


def main():
    """Main function with performance optimizations."""
    
    # Define experiment directories
    experiment_dirs = {
        # "asym_figure8": "/media/jed/MyPassport/data_backup/asym_figure8",
        # "asym_circle": "/media/jed/MyPassport/data_backup/asym_circle/",
        # "asym_slalom": "/media/jed/MyPassport/data_backup/asym_slalom/",
        # "asym_backnforth": "/media/jed/MyPassport/data_backup/asym_backnforth/",
        "sym_figure8": "/media/jed/MyPassport/data_backup/sym_figure8/",
        "sym_circle": "/media/jed/MyPassport/data_backup/sym_circle/",
        "sym_slalom": "/media/jed/MyPassport/data_backup/sym_slalom/",
        "sym_shuttlerun": "/media/jed/MyPassport/data_backup/sym_shuttlerun/",
    }
    
    # Parameters
    k_best = 10
    similarity_threshold = 3.0
    use_similarity = True  # Set to False for much faster processing
    n_processes = None  # None = auto-detect optimal number
    
    total_start_time = time.time()
    
    # Process each experiment
    for experiment_name, experiment_dir in experiment_dirs.items():
        if not os.path.exists(experiment_dir):
            print(f"Warning: Directory {experiment_dir} does not exist")
            continue
        
        # print(f"Processing experiment: {experiment_name}")
        # print(f"Directory: {experiment_dir}")
        
        start_time = time.time()
        
        # Collect best individuals data using parallel processing
        best_individuals_data = collect_best_individuals_from_experiment_parallel(
            experiment_dir, 
            k=k_best, 
            similarity_threshold=similarity_threshold,
            use_similarity=use_similarity,
            n_processes=n_processes
        )
        # print(best_individuals_data)
        
        processing_time = time.time() - start_time
        
        if not best_individuals_data:
            print(f"No data found for experiment {experiment_name}")
            continue
        
        # Create DataFrame and save
        df = pd.DataFrame(best_individuals_data)
        output_file = os.path.join(experiment_dir, "best_individual_data.csv")
        df.to_csv(output_file, index=False)
        
        print(f"Saved {len(best_individuals_data)} best individuals to {output_file}")
        print(f"Best fitness found: {df['fitness'].max():.4f}")
        print(f"Processing time: {processing_time:.2f} seconds")
        print("-" * 50)
    
    total_time = time.time() - total_start_time
    print(f"Total processing time: {total_time:.2f} seconds")


if __name__ == "__main__":
    main()