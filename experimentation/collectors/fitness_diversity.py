"""Collect fitness and diversity data from evolutionary experiments and save as CSV.

Computes per-generation fitness statistics and phenotypic diversity using edit distance.
"""

import os
import re

import numpy as np
import pandas as pd
from airevolve.evolution_tools.evaluators.edit_distance import compute_edit_distance

from experimentation.collection_utils import (
    convert_str_to_nparray,
    iter_experiments,
    iter_runs,
    iter_generations,
    iter_individuals,
    load_phenotype,
    load_evolution_csv,
)
from experimentation.config import PARAMETER_LIMITS, get_min_max, BASE_DIR, TASKS, GENOTYPES


def aggregate_fitness(df, column='fitness', max_individuals=24):
    """Aggregate fitness data by generation."""
    grouped = df.groupby('generation')[column].apply(np.array)
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
    """Load phenotype arrays from generation/individual directories.

    Returns a list-of-lists indexed by generation, each containing (narms, 6) arrays.
    """
    phenotypes_by_gen = {}

    for gen_idx, gen_path in iter_generations(run_path):
        gen_phenotypes = []
        for _ind_id, ind_path in iter_individuals(gen_path):
            phenotype = load_phenotype(ind_path)
            if phenotype is not None:
                gen_phenotypes.append(phenotype)
        phenotypes_by_gen[gen_idx] = gen_phenotypes

    if not phenotypes_by_gen:
        return []

    max_gen = max(phenotypes_by_gen.keys()) + 1
    return [phenotypes_by_gen.get(g, []) for g in range(max_gen)]


def calculate_diversity_data(pop_data, min_max_params=None):
    """Calculate diversity metrics from population data using edit distance.

    Parameters
    ----------
    pop_data : list or ndarray
        Population data indexed by generation.
    min_max_params : ndarray, optional
        Parameter limits array. Defaults to PARAMETER_LIMITS from config.
    """
    if min_max_params is None:
        min_max_params = PARAMETER_LIMITS

    min_vals = min_max_params[:, 0]
    max_vals = min_max_params[:, 1]
    diversity_values = []

    for generation in pop_data:
        valid_individuals = [
            ind for ind in generation
            if isinstance(ind, np.ndarray) and not np.isnan(ind).any()
        ]

        if len(valid_individuals) == 0:
            diversity_values.append(np.nan)
            continue

        n = len(valid_individuals)
        if n < 2:
            diversity_values.append(np.nan)
            continue

        dists = []
        for i in range(n):
            for j in range(i + 1, n):
                try:
                    dists.append(compute_edit_distance(
                        valid_individuals[i], valid_individuals[j],
                        min_vals, max_vals,
                    ))
                except Exception:
                    pass

        diversity_values.append(np.mean(dists) if dists else np.nan)

    return np.array(diversity_values)


def process_experiment_runs(experiment_dir, save_dir, experiment_name,
                            compute_diversity=True, genome_type="spherical"):
    """Process all runs for a single experiment and save aggregated data."""
    all_fitness_data = []

    for run_name, run_path in iter_runs(experiment_dir):
        evo_data = load_evolution_csv(run_path)
        if evo_data is None:
            continue

        fit_data = aggregate_fitness(evo_data, column='fitness')

        max_fitness = np.nanmax(fit_data, axis=1)
        mean_fitness = np.nanmean(fit_data, axis=1)
        std_fitness = np.nanstd(fit_data, axis=1)

        run_data = {
            'experiment': experiment_name,
            'run': run_name,
            'generation': range(len(max_fitness)),
            'max_fitness': max_fitness,
            'mean_fitness': mean_fitness,
            'std_fitness': std_fitness,
        }

        if compute_diversity:
            pop_data = load_phenotypes_from_dirs(run_path)

            if len(pop_data) == 0 and genome_type in ("spherical",) and 'offspring' in evo_data.columns:
                pop_data = aggregate_population(evo_data)

            if len(pop_data) > 0:
                run_data['diversity'] = calculate_diversity_data(pop_data)[:len(max_fitness)]
            else:
                run_data['diversity'] = np.nan
        else:
            run_data['diversity'] = np.nan

        all_fitness_data.append(pd.DataFrame(run_data))

    if not all_fitness_data:
        return None

    combined_data = pd.concat(all_fitness_data, ignore_index=True)

    output_file = os.path.join(save_dir, f"{experiment_name}/fitness_diversity_data.csv")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    combined_data.to_csv(output_file, index=False)
    print(f"  Saved: {output_file}")

    return combined_data


def run(base_dir, tasks, genotypes, **kwargs):
    """Collect fitness and diversity data for all experiments.

    Parameters
    ----------
    base_dir : str
        Root directory containing task/genotype experiment folders.
    tasks : list[str]
        Task names to process.
    genotypes : list[str]
        Genotype names to process.
    """
    for task, genotype, exp_dir in iter_experiments(base_dir, tasks, genotypes):
        print(f"Processing {task}/{genotype}")
        save_dir = os.path.join(base_dir, task)
        os.makedirs(os.path.join(save_dir, genotype), exist_ok=True)
        process_experiment_runs(
            exp_dir, save_dir, experiment_name=genotype,
            compute_diversity=True, genome_type=genotype,
        )


if __name__ == "__main__":
    run(BASE_DIR, TASKS, GENOTYPES)
