"""
Optimized script to collect best individuals from evolutionary experiments and save as CSV files.
Uses multiprocessing and performance optimizations for faster execution.
"""

import os
import re
import time
import pickle
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from airevolve.evolution_tools.evaluators.edit_distance import compute_edit_distance


def detect_genome_type(evo_data):
    """Detect genome type from the offspring column content."""
    sample = str(evo_data['offspring'].iloc[0])
    if sample.startswith("CPPNNetwork("):
        return "cppn"
    if sample.startswith("HybridGenome("):
        return "hybrid_cppn"
    return "spherical"


def load_phenotype_from_individual_dir(run_path, generation, individual_index):
    """Load phenotype from genome.npy in the individual's directory.

    Scans generation_{gen}/individual_*/ directories to find the individual
    by its position index in the generation.
    """
    gen_dir = os.path.join(run_path, f"generation_{generation:02d}")
    if not os.path.isdir(gen_dir):
        # Try without zero-padding
        gen_dir = os.path.join(run_path, f"generation_{generation}")
        if not os.path.isdir(gen_dir):
            return None, None

    # Collect individual dirs sorted by individual number
    ind_dirs = []
    for entry in os.listdir(gen_dir):
        m = re.match(r"individual_(\d+)$", entry)
        if m:
            ind_dirs.append((int(m.group(1)), entry))
    ind_dirs.sort()

    if individual_index >= len(ind_dirs):
        return None, None

    ind_num, ind_name = ind_dirs[individual_index]
    ind_path = os.path.join(gen_dir, ind_name)

    # Load phenotype
    genome_file = os.path.join(ind_path, "genome.npy")
    phenotype = None
    if os.path.exists(genome_file):
        try:
            phenotype = np.load(genome_file)
        except Exception:
            pass

    # Check for genotype.pkl
    genotype_pkl = os.path.join(ind_path, "genotype.pkl")
    genotype_path = genotype_pkl if os.path.exists(genotype_pkl) else None

    return phenotype, genotype_path


def process_single_run(args):
    """Process a single run directory - designed for multiprocessing."""
    run_path, k, similarity_threshold, use_similarity = args

    evolution_csv = os.path.join(run_path, "evolution_data.csv")

    if not os.path.exists(evolution_csv):
        return []

    try:
        evo_data = pd.read_csv(evolution_csv)
        genome_type = detect_genome_type(evo_data)

        # Build per-generation fitness and phenotype data from genome.npy files
        fitness_by_gen = {}
        index_by_gen = {}

        for idx, row in evo_data.iterrows():
            gen = row['generation']
            if gen not in fitness_by_gen:
                fitness_by_gen[gen] = []
                index_by_gen[gen] = []
            fitness_by_gen[gen].append(row['fitness'])
            index_by_gen[gen].append(len(fitness_by_gen[gen]) - 1)

        # Collect all (fitness, gen, ind_idx) tuples
        candidates = []
        for gen in fitness_by_gen:
            for ind_idx, fitness in enumerate(fitness_by_gen[gen]):
                if not np.isnan(fitness):
                    candidates.append((fitness, gen, ind_idx))

        # Sort by fitness descending
        candidates.sort(key=lambda x: x[0], reverse=True)

        # Select top-k (with optional similarity filtering)
        parameter_limits = np.array([[0.09, 0.4], [0, 2*np.pi], [0, 2*np.pi],
                                     [0, 2*np.pi], [0, 2*np.pi], [0, 1]])
        min_vals = parameter_limits[:, 0]
        max_vals = parameter_limits[:, 1]

        selected = []
        selected_phenotypes = []

        for fitness, gen, ind_idx in candidates:
            if len(selected) >= k:
                break

            phenotype, genotype_path = load_phenotype_from_individual_dir(
                run_path, gen, ind_idx)

            if phenotype is None:
                continue

            # Similarity check
            if use_similarity and selected_phenotypes:
                is_similar = False
                for sel_pheno in selected_phenotypes:
                    try:
                        if compute_edit_distance(phenotype, sel_pheno, min_vals, max_vals) < similarity_threshold:
                            is_similar = True
                            break
                    except Exception:
                        continue
                if is_similar:
                    continue

            selected.append({
                'run_directory': run_path,
                'rank': len(selected) + 1,
                'fitness': fitness,
                'generation': gen,
                'individual_index': ind_idx,
                'individual_data': phenotype.tolist(),
                'genome_type': genome_type,
                'genotype_pkl_path': genotype_path if genotype_path else '',
            })
            selected_phenotypes.append(phenotype)

        return selected

    except Exception as e:
        print(f"Error processing {run_path}: {e}")
        return []


def collect_best_individuals_from_experiment_parallel(experiment_dir, k=3, similarity_threshold=3.0,
                                                    use_similarity=True, n_processes=None):
    """Parallel version of best individuals collection."""
    if n_processes is None:
        n_processes = min(cpu_count(), 8)

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
    base_dir = "/media/jed/My Passport/airevolve030326"
    experiment_dirs = {
        "spherical": os.path.join(base_dir, "spherical"),
        "cppn": os.path.join(base_dir, "cppn"),
        "hybrid_cppn": os.path.join(base_dir, "hybrid_cppn"),
    }

    # Parameters
    k_best = 10
    similarity_threshold = 3.0
    use_similarity = True
    n_processes = None  # None = auto-detect optimal number

    total_start_time = time.time()

    # Process each experiment
    for experiment_name, experiment_dir in experiment_dirs.items():
        if not os.path.exists(experiment_dir):
            print(f"Warning: Directory {experiment_dir} does not exist")
            continue

        start_time = time.time()

        # Collect best individuals data using parallel processing
        best_individuals_data = collect_best_individuals_from_experiment_parallel(
            experiment_dir,
            k=k_best,
            similarity_threshold=similarity_threshold,
            use_similarity=use_similarity,
            n_processes=n_processes
        )

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
