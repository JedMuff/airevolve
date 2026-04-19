"""Collect morphological descriptors data from experiments and save as CSV.

Uses parallelization to compute metrics for individuals concurrently.
Includes fitness data alongside morphological descriptors.
"""

import functools
import json
import os
import re

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from numpy.linalg import norm, eig

from airevolve.evolution_tools.inspection_tools.morphological_descriptors.central_symmetry import (
    compute_symmetry as compute_central_symmetry,
)
from airevolve.evolution_tools.inspection_tools.morphological_descriptors.biradial_symmetry import (
    compute_symmetry as compute_biradial_symmetry,
)
from airevolve.evolution_tools.inspection_tools.morphological_descriptors.hovering_info import get_sim
from airevolve.evolution_tools.inspection_tools.morphological_descriptors.avr_arm_length import compute_avr_arm_length
from airevolve.evolution_tools.inspection_tools.morphological_descriptors.var_arm_length import compute_var_arm_length
from airevolve.evolution_tools.inspection_tools.morphological_descriptors.num_arms import compute_num_arms
from airevolve.evolution_tools.inspection_tools.morphological_descriptors.mass import compute_total_mass
from airevolve.evolution_tools.inspection_tools.morphological_descriptors.area import compute_area
from airevolve.evolution_tools.inspection_tools.morphological_descriptors.volume import compute_volume
from airevolve.evolution_tools.inspection_tools.morphological_descriptors.proportion import compute_proportion
from airevolve.evolution_tools.inspection_tools.morphological_descriptors.thrust_alignment import thrust_alignment
from airevolve.evolution_tools.inspection_tools.morphological_descriptors.xyz import avr_x, avr_y, avr_z

from experimentation.collection_utils import (
    convert_str_to_nparray,
    iter_experiments,
    iter_runs,
    iter_generations,
    iter_individuals,
    load_phenotype,
    load_hover_breakdown,
    build_fitness_lookup,
)
from experimentation.config import PARAMETER_LIMITS, get_min_max, BASE_DIR, TASKS, GENOTYPES

# Constants for hover fitness computation (match run_combined_hover_gate_evolution.py)
_G = 9.81
_C_AUTHORITY = 300.0


def compute_hover_components_from_matrices(Bf, Bm):
    """Compute hover fitness components from Bf/Bm matrices (pure numpy, fork-safe).

    Returns dict with f_rank, f_force, f_torque, hover_fitness.
    """
    rank_f = np.linalg.matrix_rank(Bf)
    rank_m = np.linalg.matrix_rank(Bm)
    f_rank = (rank_f + rank_m) / 6.0

    n_props = Bf.shape[1]
    eta_max = np.ones(n_props)
    f_vec = Bf @ eta_max
    thrust_ratio = norm(f_vec) / _G
    f_force = min(thrust_ratio, 2.0) / 2.0

    gram_m = Bm @ Bm.T
    eigs = np.real(eig(gram_m)[0])
    eigs = np.maximum(eigs, 0.0)

    lambda_min = np.min(eigs)
    lambda_max = np.max(eigs)

    condition = lambda_min / (lambda_max + 1e-12)
    authority = np.sqrt(lambda_min)
    authority_sat = authority / (authority + _C_AUTHORITY)
    f_torque = condition * authority_sat

    ctrl = float(lambda_min)
    mtw = float(norm(f_vec) / _G)

    return {
        "Max_Thrust_to_Weight": mtw,
        "Controlability": ctrl,
        "f_rank": float(f_rank),
        "f_force": float(f_force),
        "f_torque": float(f_torque),
        "hover_fitness": float(f_rank + f_force + f_torque),
    }


def aggregate_population_with_fitness(df):
    """Aggregate population data by generation from CSV offspring column, including fitness data."""
    df['offspring'] = df['offspring'].apply(convert_str_to_nparray)

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


def load_population_from_dirs(run_path):
    """Load phenotypes, fitness, and paths from generation/individual directory structure.

    Returns (pop_data, fitness_data, paths_data) as lists-of-lists indexed by generation.
    """
    fitness_lookup = build_fitness_lookup(run_path)

    phenotypes_by_gen = {}
    fitness_by_gen = {}
    paths_by_gen = {}

    for gen_idx, gen_path in iter_generations(run_path):
        gen_phenotypes = []
        gen_fitness = []
        gen_paths = []
        for ind_id, ind_path in iter_individuals(gen_path):
            phenotype = load_phenotype(ind_path)
            if phenotype is None:
                continue
            gen_phenotypes.append(phenotype)
            gen_paths.append(ind_path)
            fit = fitness_lookup.get((gen_idx, ind_id), np.nan)
            gen_fitness.append(fit)

        phenotypes_by_gen[gen_idx] = gen_phenotypes
        fitness_by_gen[gen_idx] = gen_fitness
        paths_by_gen[gen_idx] = gen_paths

    if not phenotypes_by_gen:
        return [], [], []

    max_gen = max(phenotypes_by_gen.keys()) + 1
    pop_data = [phenotypes_by_gen.get(g, []) for g in range(max_gen)]
    fitness_data = [fitness_by_gen.get(g, []) for g in range(max_gen)]
    paths_data = [paths_by_gen.get(g, []) for g in range(max_gen)]

    return pop_data, fitness_data, paths_data


def compute_individual_descriptors(args):
    """Compute morphological descriptors for a single individual."""
    individual, generation_idx, individual_idx, fitness_value, md_funcs, md_titles, ind_path = args

    if individual is None or not isinstance(individual, np.ndarray):
        return None
    if individual.ndim != 2 or np.isnan(individual).any():
        return None

    row_data = {
        'generation': generation_idx,
        'individual': individual_idx,
        'fitness': fitness_value if fitness_value is not None and not np.isnan(fitness_value) else np.nan,
    }

    for func, title in zip(md_funcs, md_titles):
        try:
            row_data[title] = func(individual)
        except Exception:
            row_data[title] = np.nan

    row_data['num_arms'] = int(np.sum(~np.isnan(individual).all(axis=1)))

    # Hover metrics from Bf/Bm matrices
    nan_hover = {
        "Max_Thrust_to_Weight": np.nan, "Controlability": np.nan,
        "f_rank": np.nan, "f_force": np.nan, "f_torque": np.nan, "hover_fitness": np.nan,
    }
    try:
        sim = get_sim(individual)
        if sim is not None:
            Bf = np.array(sim.Bf)
            Bm = np.array(sim.Bm)
            del sim
            row_data.update(compute_hover_components_from_matrices(Bf, Bm))
        else:
            row_data.update(nan_hover)
    except Exception:
        row_data.update(nan_hover)

    # Hover breakdown from JSON
    if ind_path is not None:
        hb = load_hover_breakdown(ind_path)
        if hb is not None:
            row_data['gates_passed'] = hb.get('gates_passed', np.nan)
            row_data['hover_status'] = hb.get('status', '')
        else:
            row_data['gates_passed'] = np.nan
            row_data['hover_status'] = ''
    else:
        row_data['gates_passed'] = np.nan
        row_data['hover_status'] = ''

    return row_data


def compute_morphological_descriptors(pop_data, fitness_data, md_funcs, md_titles,
                                      paths_data=None, n_workers=None):
    """Compute morphological descriptors for population data using parallelization."""
    if n_workers is None:
        n_workers = min(cpu_count(), 8)

    args_list = []
    for generation_idx, generation in enumerate(pop_data):
        for individual_idx, individual in enumerate(generation):
            fitness_value = None
            if fitness_data is not None and generation_idx < len(fitness_data):
                gen_fitness = fitness_data[generation_idx]
                if isinstance(gen_fitness, (list, np.ndarray)) and individual_idx < len(gen_fitness):
                    fitness_value = gen_fitness[individual_idx]

            ind_path = None
            if paths_data is not None and generation_idx < len(paths_data):
                gen_paths = paths_data[generation_idx]
                if isinstance(gen_paths, list) and individual_idx < len(gen_paths):
                    ind_path = gen_paths[individual_idx]

            args_list.append((individual, generation_idx, individual_idx, fitness_value, md_funcs, md_titles, ind_path))

    md_data = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_args = {executor.submit(compute_individual_descriptors, a): a for a in args_list}
        completed = 0
        total = len(args_list)
        for future in as_completed(future_to_args):
            try:
                result = future.result()
                if result is not None:
                    md_data.append(result)
                completed += 1
                if completed % 500 == 0:
                    print(f"  {completed}/{total} individuals processed")
            except Exception:
                pass

    return pd.DataFrame(md_data)


def _build_md_funcs():
    """Return (md_funcs, md_titles) lists for all morphological descriptors."""
    compute_bilateral_symmetry = functools.partial(compute_biradial_symmetry, fixed_plane=[1, 0, 0, 0])
    md_funcs = [
        compute_central_symmetry,
        compute_bilateral_symmetry,
        compute_avr_arm_length,
        compute_var_arm_length,
        compute_num_arms,
        compute_total_mass,
        compute_area,
        compute_volume,
        compute_proportion,
        thrust_alignment,
        avr_x,
        avr_y,
        avr_z,
    ]
    md_titles = [
        "Central_Asymmetry",
        "Bilateral_Asymmetry",
        "Avg_Arm_Length",
        "Var_Arm_Length",
        "Num_Arms",
        "Total_Mass",
        "Area",
        "Volume",
        "Proportion",
        "Thrust_Alignment",
        "Avg_X",
        "Avg_Y",
        "Avg_Z",
    ]
    return md_funcs, md_titles


def process_experiment_directory(experiment_dir, save_dir, n_workers=None):
    """Process all runs in an experiment directory and save combined morphological descriptors."""
    md_funcs, md_titles = _build_md_funcs()
    all_md_data = []

    runs = list(iter_runs(experiment_dir))
    for run_idx, (run_name, run_path) in enumerate(runs):
        print(f"  Run {run_idx + 1}/{len(runs)}: {run_name}")

        evo_data = pd.read_csv(os.path.join(run_path, "evolution_data.csv"))

        pop_data, fitness_data, paths_data = load_population_from_dirs(run_path)

        if len(pop_data) == 0 and 'offspring' in evo_data.columns:
            pop_data, fitness_data = aggregate_population_with_fitness(evo_data)
            paths_data = None

        if len(pop_data) == 0:
            continue

        md_df = compute_morphological_descriptors(
            pop_data, fitness_data, md_funcs, md_titles,
            paths_data=paths_data, n_workers=n_workers,
        )
        md_df['run'] = run_idx
        all_md_data.append(md_df)

    if all_md_data:
        combined_md_data = pd.concat(all_md_data, ignore_index=True)
        output_path = os.path.join(save_dir, "morphological_descriptors_data.csv")
        combined_md_data.to_csv(output_path, index=False)
        print(f"  Saved: {output_path} ({len(combined_md_data)} records)")
        return combined_md_data

    return None


def run(base_dir, tasks, genotypes, *, n_workers=None):
    """Collect morphological descriptors for all experiments.

    Parameters
    ----------
    base_dir : str
        Root directory containing task/genotype experiment folders.
    tasks : list[str]
        Task names to process.
    genotypes : list[str]
        Genotype names to process.
    n_workers : int, optional
        Number of parallel workers. Defaults to min(cpu_count(), 24).
    """
    if n_workers is None:
        n_workers = min(cpu_count(), 24)

    for task, genotype, exp_dir in iter_experiments(base_dir, tasks, genotypes):
        print(f"Processing {task}/{genotype}")
        save_dir = exp_dir
        os.makedirs(save_dir, exist_ok=True)
        process_experiment_directory(exp_dir, save_dir, n_workers=n_workers)


if __name__ == "__main__":
    run(BASE_DIR, TASKS, GENOTYPES)
