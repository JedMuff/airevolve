"""Collect per-generation bloodline survival data and founder-to-descendant divergence.

Each gen-0 individual starts a unique bloodline. For mutation-only genotypes
every individual traces back to exactly one founder. For NEAT genotypes that
use crossover, each individual is assigned to the founder of its *primary*
parent (parent_ids[0], the fitter parent whose topology dominates). This script
counts how many distinct founder bloodlines survive in the population at each
generation, and measures how much descendants have diverged from their founders
phenotypically and genotypically.
"""

import ast
import os

import numpy as np
import pandas as pd

from experimentation.collection_utils import (
    iter_experiments,
    iter_runs,
    iter_generations,
    iter_individuals,
    load_evolution_csv,
    load_phenotype,
    load_genotype,
    lazy_import_cppn_compat,
)
from experimentation.config import BASE_DIR, TASKS, GENOTYPES, get_min_max
from experimentation.collectors.genome_diversity import hybrid_genome_distance
from airevolve.evolution_tools.evaluators.edit_distance import (
    compute_edit_distance,
    compute_euclidean_distance,
)


def _parse_parent_ids(val):
    """Parse parent_ids column (string repr of list) into a list.

    Normalizes string IDs (e.g. '0011') to int to match pandas-parsed CSV id column.
    """
    if pd.isna(val):
        return [None, None]
    try:
        result = ast.literal_eval(str(val))
        if isinstance(result, list):
            normalized = []
            for x in result:
                if x is None:
                    normalized.append(None)
                else:
                    normalized.append(str(x))
            return normalized
        return [None, None]
    except (ValueError, SyntaxError):
        return [None, None]


def _build_founder_map(df):
    """Build a mapping from individual id to its gen-0 founder.

    Uses iterative ancestor tracing with memoization.
    """
    parent_of = {}
    gen_of = {}
    for _, row in df.iterrows():
        ind_id = row["id"]
        if ind_id not in parent_of:
            parents = _parse_parent_ids(row["parent_ids"])
            parent_of[ind_id] = parents[0] if parents[0] is not None else None
            gen_of[ind_id] = int(row["generation"])

    founder_cache = {}

    def find_founder(ind_id):
        chain = []
        current = ind_id
        while current not in founder_cache:
            parent = parent_of.get(current)
            if parent is None or gen_of.get(current, 0) == 0:
                founder_cache[current] = current
                break
            chain.append(current)
            current = parent

        founder = founder_cache.get(current, current)
        for c in chain:
            founder_cache[c] = founder

        return founder_cache.get(ind_id, ind_id)

    for ind_id in parent_of:
        find_founder(ind_id)

    return founder_cache


def process_run(run_name, run_path):
    """Process a single run and return bloodline rows."""
    df = load_evolution_csv(run_path)
    if df is None or len(df) == 0:
        return []

    founder_map = _build_founder_map(df)

    rows = []
    in_pop_df = df[df["in_pop"] == True]
    for gen, gen_df in in_pop_df.groupby("generation"):
        ids_in_pop = gen_df["id"].unique()
        founders = set()
        for ind_id in ids_in_pop:
            founders.add(founder_map.get(ind_id, ind_id))

        n_bloodlines = len(founders)
        pop_size = len(ids_in_pop)
        rows.append({
            "run": run_name,
            "generation": int(gen),
            "n_bloodlines": n_bloodlines,
            "population_size": pop_size,
            "bloodline_ratio": n_bloodlines / pop_size if pop_size > 0 else 0.0,
        })

    return rows


def _load_phenotype_lookup(run_path):
    """Build ind_id -> phenotype array lookup from individual directories.

    Each individual's phenotype is stored once in its birth generation directory.
    Keys use string IDs to match the CSV id column (e.g. "0000").
    """
    lookup = {}
    for gen_idx, gen_path in iter_generations(run_path):
        for ind_id, ind_path in iter_individuals(gen_path):
            if ind_id not in lookup:
                pheno = load_phenotype(ind_path)
                if pheno is not None:
                    lookup[ind_id] = pheno
    return lookup


def process_run_divergence(run_name, run_path):
    """Compute per-generation mean phenotype divergence from founders."""
    df = load_evolution_csv(run_path)
    if df is None or len(df) == 0:
        return []

    founder_map = _build_founder_map(df)
    pheno_lookup = _load_phenotype_lookup(run_path)
    min_vals, max_vals = get_min_max()

    rows = []
    in_pop_df = df[df["in_pop"] == True]
    for gen, gen_df in in_pop_df.groupby("generation"):
        gen = int(gen)
        ids_in_pop = gen_df["id"].unique()
        distances = []

        for ind_id in ids_in_pop:
            founder_id = founder_map.get(ind_id, ind_id)
            desc_pheno = pheno_lookup.get(ind_id)
            founder_pheno = pheno_lookup.get(founder_id)

            if desc_pheno is not None and founder_pheno is not None:
                try:
                    dist = compute_edit_distance(
                        desc_pheno, founder_pheno, min_vals, max_vals
                    )
                    distances.append(dist)
                except Exception:
                    pass

        if distances:
            rows.append({
                "run": run_name,
                "generation": gen,
                "mean_divergence": np.mean(distances),
                "std_divergence": np.std(distances),
                "n_compared": len(distances),
            })

    return rows


def _load_genotype_lookup(run_path):
    """Build ind_id -> genotype object lookup from individual directories.

    Keys use string IDs to match the CSV id column (e.g. "0000").
    """
    lookup = {}
    for gen_idx, gen_path in iter_generations(run_path):
        for ind_id, ind_path in iter_individuals(gen_path):
            if ind_id not in lookup:
                geno = load_genotype(ind_path)
                if geno is not None:
                    lookup[ind_id] = geno
    return lookup


def _get_genotypic_distance_fn(genotype):
    """Return the appropriate genotypic distance function for a genotype."""
    if genotype in ("spherical", "neat_spherical"):
        min_vals, max_vals = get_min_max()
        def pheno_as_geno_dist(a, b):
            return compute_edit_distance(a, b, min_vals, max_vals)
        return pheno_as_geno_dist, True  # uses phenotype arrays
    elif genotype in ("cppn", "neat_cppn"):
        return lazy_import_cppn_compat(), False
    elif genotype in ("hybrid_cppn", "neat_hybrid_cppn"):
        return hybrid_genome_distance, False
    return None, False


def process_run_genotypic_divergence(run_name, run_path, genotype):
    """Compute per-generation mean genotypic divergence from founders."""
    df = load_evolution_csv(run_path)
    if df is None or len(df) == 0:
        return []

    dist_fn, uses_phenotype = _get_genotypic_distance_fn(genotype)
    if dist_fn is None:
        return []

    founder_map = _build_founder_map(df)

    if uses_phenotype:
        lookup = _load_phenotype_lookup(run_path)
    else:
        lookup = _load_genotype_lookup(run_path)

    rows = []
    in_pop_df = df[df["in_pop"] == True]
    for gen, gen_df in in_pop_df.groupby("generation"):
        gen = int(gen)
        ids_in_pop = gen_df["id"].unique()
        distances = []

        for ind_id in ids_in_pop:
            founder_id = founder_map.get(ind_id, ind_id)
            desc = lookup.get(ind_id)
            founder = lookup.get(founder_id)

            if desc is not None and founder is not None:
                try:
                    distances.append(dist_fn(desc, founder))
                except Exception:
                    pass

        if distances:
            rows.append({
                "run": run_name,
                "generation": gen,
                "mean_divergence": np.mean(distances),
                "std_divergence": np.std(distances),
                "n_compared": len(distances),
            })

    return rows


def run(base_dir, tasks, genotypes, **kwargs):
    """Collect bloodline survival and founder divergence data.

    Parameters
    ----------
    base_dir : str
        Root experiment directory.
    tasks : list[str]
        Task names to process.
    genotypes : list[str]
        Genotype names to process.
    """
    for task, genotype, experiment_dir in iter_experiments(base_dir, tasks, genotypes):
        all_rows = []
        all_div_rows = []
        all_geno_div_rows = []

        for run_name, run_path in iter_runs(experiment_dir):
            run_rows = process_run(run_name, run_path)
            all_rows.extend(run_rows)

            div_rows = process_run_divergence(run_name, run_path)
            all_div_rows.extend(div_rows)

            geno_div_rows = process_run_genotypic_divergence(
                run_name, run_path, genotype)
            all_geno_div_rows.extend(geno_div_rows)

        if all_rows:
            out_df = pd.DataFrame(all_rows)
            output_path = os.path.join(experiment_dir, "bloodline_data.csv")
            out_df.to_csv(output_path, index=False)
            print(f"Saved {len(out_df)} bloodline records to {output_path}")

        if all_div_rows:
            div_df = pd.DataFrame(all_div_rows)
            div_path = os.path.join(experiment_dir, "bloodline_divergence_data.csv")
            div_df.to_csv(div_path, index=False)
            print(f"Saved {len(div_df)} divergence records to {div_path}")

        if all_geno_div_rows:
            geno_div_df = pd.DataFrame(all_geno_div_rows)
            geno_div_path = os.path.join(
                experiment_dir, "bloodline_genotypic_divergence_data.csv")
            geno_div_df.to_csv(geno_div_path, index=False)
            print(f"Saved {len(geno_div_df)} genotypic divergence records to {geno_div_path}")


if __name__ == "__main__":
    run(BASE_DIR, TASKS, GENOTYPES)
