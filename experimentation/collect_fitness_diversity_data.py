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
    """Aggregate population data by generation (from CSV offspring column)."""
    df['offspring'] = df['offspring'].apply(convert_str_to_nparray)
    grouped = df.groupby('generation')['offspring'].apply(np.array)

    pop_array = np.full((len(grouped), max_individuals), np.nan, dtype=object)

    for i, pop_list in enumerate(grouped):
        pop_array[i, :len(pop_list)] = pop_list[:max_individuals]

    return pop_array


def load_phenotypes_from_dirs(run_path):
    """Load phenotype arrays from genome.npy files in generation/individual dirs.

    Returns a list-of-lists structure compatible with calculate_diversity_data:
    a list indexed by generation, each element is a list of (narms, 6) arrays.
    Missing individuals are simply omitted.
    """
    phenotypes_by_gen = {}

    # Scan for generation directories
    for entry in os.listdir(run_path):
        m = re.match(r"generation_(\d+)$", entry)
        if not m:
            continue
        gen_idx = int(m.group(1))
        gen_path = os.path.join(run_path, entry)
        if not os.path.isdir(gen_path):
            continue

        gen_phenotypes = []
        for ind_entry in os.listdir(gen_path):
            if not ind_entry.startswith("individual_"):
                continue
            genome_file = os.path.join(gen_path, ind_entry, "phenotype.npy")
            if not os.path.exists(genome_file):
                genome_file = os.path.join(gen_path, ind_entry, "genome.npy")
            if not os.path.exists(genome_file):
                continue
            try:
                phenotype = np.load(genome_file)
                if phenotype.ndim == 2 and not np.isnan(phenotype).all():
                    gen_phenotypes.append(phenotype)
            except Exception as e:
                print(f"  Warning: failed to load {genome_file}: {e}")
                continue

        phenotypes_by_gen[gen_idx] = gen_phenotypes

    if not phenotypes_by_gen:
        return []

    max_gen = max(phenotypes_by_gen.keys()) + 1
    result = []
    for g in range(max_gen):
        result.append(phenotypes_by_gen.get(g, []))

    return result


def calculate_diversity_data(pop_data, min_max_params=None):
    """Calculate diversity metrics from population data using edit distance functions.

    pop_data: either a 2D object array (from aggregate_population) where each
    element is an ndarray or NaN, OR a list-of-lists where each inner list
    contains (narms, 6) arrays.
    """
    if min_max_params is None:
        # Default parameters as used in the original template
        min_max_params = np.array([[0.09, 0.4], [0, 2*np.pi], [0, 2*np.pi],
                                  [0, 2*np.pi], [0, 2*np.pi], [0, 1]])

    min_vals = min_max_params[:, 0]
    max_vals = min_max_params[:, 1]
    diversity_values = []

    for generation in pop_data:
        # Filter out NaN / invalid individuals
        if isinstance(generation, np.ndarray):
            # Object array from aggregate_population
            valid_individuals = [ind for ind in generation if isinstance(ind, np.ndarray) and not np.isnan(ind).any()]
        else:
            # List of arrays from load_phenotypes_from_dirs
            valid_individuals = [ind for ind in generation if isinstance(ind, np.ndarray) and not np.isnan(ind).any()]

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


def process_experiment_runs(experiment_dir, experiment_name, save_dir,
                            compute_diversity=True, genome_type="spherical"):
    """Process all runs for a single experiment and save aggregated data."""
    print(f"Processing experiment: {experiment_name}")

    if not os.path.exists(experiment_dir):
        print(f"Warning: Directory {experiment_dir} does not exist")
        return

    all_fitness_data = []

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

        # Aggregate fitness data
        fit_data = aggregate_fitness(evo_data, column='fitness')

        # Calculate fitness statistics per generation
        max_fitness = np.nanmax(fit_data, axis=1)
        mean_fitness = np.nanmean(fit_data, axis=1)
        std_fitness = np.nanstd(fit_data, axis=1)

        # Store data for this run
        run_data = {
            'experiment': experiment_name,
            'run': run_folder,
            'generation': range(len(max_fitness)),
            'max_fitness': max_fitness,
            'mean_fitness': mean_fitness,
            'std_fitness': std_fitness,
        }

        if compute_diversity:
            # Prefer loading phenotypes from genome.npy files in directory structure
            pop_data = load_phenotypes_from_dirs(run_path)

            if len(pop_data) == 0 and genome_type == "spherical" and 'offspring' in evo_data.columns:
                # Fallback: parse offspring column from CSV (v1 spherical data)
                pop_data = aggregate_population(evo_data)

            if len(pop_data) > 0:
                diversity_data = calculate_diversity_data(pop_data)
                run_data['diversity'] = diversity_data[:len(max_fitness)]
            else:
                print(f"  Warning: no phenotype data found for diversity in {run_folder}")
                run_data['diversity'] = np.nan
        else:
            run_data['diversity'] = np.nan

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

    base_dir = "/media/jed/My Passport/airevolve030326/v2"

    tasks = ["backandforth", "figure8", "circle", "slalom"]
    genotypes = [
        {"name": "spherical",   "diversity": True, "genome_type": "spherical"},
        {"name": "cppn",        "diversity": True, "genome_type": "cppn"},
        {"name": "hybrid_cppn", "diversity": True, "genome_type": "hybrid_cppn"},
    ]

    all_experiment_data = []

    for task in tasks:
        print(f"\n{'='*60}")
        print(f"Task: {task}")
        print(f"{'='*60}")

        for geno in genotypes:
            experiment_dir = os.path.join(base_dir, task, geno["name"])
            save_dir = os.path.join(base_dir, task)
            os.makedirs(os.path.join(save_dir, geno["name"]), exist_ok=True)

            experiment_data = process_experiment_runs(
                experiment_dir, geno["name"], save_dir,
                compute_diversity=geno["diversity"],
                genome_type=geno["genome_type"],
            )
            if experiment_data is not None:
                all_experiment_data.append(experiment_data)

    print("Processing complete!")


if __name__ == "__main__":
    main()
