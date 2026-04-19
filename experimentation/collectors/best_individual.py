"""Collect best individuals from evolutionary experiments and save as CSV.

Uses multiprocessing and edit-distance-based similarity filtering to select
diverse top-k individuals per run.
"""

import os
import re

import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count

from airevolve.evolution_tools.evaluators.edit_distance import compute_edit_distance

from experimentation.collection_utils import (
    iter_experiments,
    iter_runs,
    load_evolution_csv,
)
from experimentation.config import PARAMETER_LIMITS, get_min_max, BASE_DIR, TASKS, GENOTYPES


def detect_genome_type(evo_data):
    """Detect genome type from the genome column content."""
    sample = str(evo_data['genome'].iloc[0])
    if sample.startswith("CPPNNetwork("):
        return "cppn"
    if sample.startswith("HybridGenome("):
        return "hybrid_cppn"
    return "spherical"


def load_phenotype_from_individual_dir(run_path, generation, individual_index):
    """Load phenotype from genome.npy in the individual's directory by position index."""
    gen_dir = os.path.join(run_path, f"generation_{generation:02d}")
    if not os.path.isdir(gen_dir):
        gen_dir = os.path.join(run_path, f"generation_{generation}")
        if not os.path.isdir(gen_dir):
            return None, None

    ind_dirs = []
    for entry in os.listdir(gen_dir):
        m = re.match(r"individual_(\d+)$", entry)
        if m:
            ind_dirs.append((int(m.group(1)), entry))
    ind_dirs.sort()

    if individual_index >= len(ind_dirs):
        return None, None

    _ind_num, ind_name = ind_dirs[individual_index]
    ind_path = os.path.join(gen_dir, ind_name)

    phenotype = None
    genome_file = os.path.join(ind_path, "genome.npy")
    if os.path.exists(genome_file):
        try:
            phenotype = np.load(genome_file)
        except Exception:
            pass

    genotype_pkl = os.path.join(ind_path, "genotype.pkl")
    genotype_path = genotype_pkl if os.path.exists(genotype_pkl) else None

    return phenotype, genotype_path


def process_single_run(args):
    """Process a single run directory (designed for multiprocessing)."""
    run_path, k, similarity_threshold, use_similarity = args

    evo_csv = os.path.join(run_path, "evolution_data.csv")
    if not os.path.exists(evo_csv):
        return []

    try:
        evo_data = pd.read_csv(evo_csv)
        genome_type = detect_genome_type(evo_data)

        fitness_by_gen = {}
        for _idx, row in evo_data.iterrows():
            gen = row['generation']
            if gen not in fitness_by_gen:
                fitness_by_gen[gen] = []
            fitness_by_gen[gen].append(row['fitness'])

        candidates = []
        for gen in fitness_by_gen:
            for ind_idx, fitness in enumerate(fitness_by_gen[gen]):
                if not np.isnan(fitness):
                    candidates.append((fitness, gen, ind_idx))

        candidates.sort(key=lambda x: x[0], reverse=True)

        min_vals, max_vals = get_min_max()

        selected = []
        selected_phenotypes = []

        for fitness, gen, ind_idx in candidates:
            if len(selected) >= k:
                break

            phenotype, genotype_path = load_phenotype_from_individual_dir(run_path, gen, ind_idx)
            if phenotype is None:
                continue

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


def collect_best_individuals_parallel(experiment_dir, k=3, similarity_threshold=3.0,
                                      use_similarity=True, n_processes=None):
    """Parallel collection of best individuals across runs."""
    if n_processes is None:
        n_processes = min(cpu_count(), 8)

    run_dirs = [
        os.path.join(experiment_dir, d)
        for d in os.listdir(experiment_dir)
        if d != ".DS_Store" and os.path.isdir(os.path.join(experiment_dir, d))
    ]

    if not run_dirs:
        return []

    pool_args = [(rp, k, similarity_threshold, use_similarity) for rp in run_dirs]

    with Pool(n_processes) as pool:
        results = pool.map(process_single_run, pool_args)

    all_best = []
    for run_results in results:
        all_best.extend(run_results)
    return all_best


def run(base_dir, tasks, genotypes, *, k_best=10, similarity_threshold=3.0,
        use_similarity=True, n_processes=None):
    """Collect best individuals for all experiments.

    Parameters
    ----------
    base_dir : str
        Root directory containing task/genotype experiment folders.
    tasks : list[str]
        Task names to process.
    genotypes : list[str]
        Genotype names to process.
    k_best : int
        Number of top individuals to keep per run.
    similarity_threshold : float
        Edit distance threshold for similarity filtering.
    use_similarity : bool
        Whether to apply similarity filtering.
    n_processes : int, optional
        Number of parallel processes. Defaults to min(cpu_count(), 8).
    """
    for task, genotype, exp_dir in iter_experiments(base_dir, tasks, genotypes):
        print(f"Processing {task}/{genotype}")

        best_individuals_data = collect_best_individuals_parallel(
            exp_dir,
            k=k_best,
            similarity_threshold=similarity_threshold,
            use_similarity=use_similarity,
            n_processes=n_processes,
        )

        if not best_individuals_data:
            continue

        df = pd.DataFrame(best_individuals_data)
        output_file = os.path.join(exp_dir, "best_individual_data.csv")
        df.to_csv(output_file, index=False)
        print(f"  Saved: {output_file} ({len(best_individuals_data)} individuals)")


if __name__ == "__main__":
    run(BASE_DIR, TASKS, GENOTYPES)
