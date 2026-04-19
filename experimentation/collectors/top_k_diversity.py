"""Collect diversity among the top-K best individuals across independent runs.

Takes the single best individual from each of the top 10 runs (by best fitness)
and computes pairwise phenotypic and genome-level distances.
"""

import os
from itertools import combinations

import numpy as np
import pandas as pd

from experimentation.collection_utils import (
    iter_experiments,
    iter_runs,
    load_phenotype,
    load_genotype,
    load_evolution_csv,
    lazy_import_cppn_compat,
)
from experimentation.config import get_min_max

from airevolve.evolution_tools.evaluators.edit_distance import (
    compute_edit_distance,
    compute_euclidean_distance,
)


# Direct parameter limits for hybrid
_DIRECT_MIN = np.array([0.09, 0.0, 0.0])
_DIRECT_MAX = np.array([0.4, 2 * np.pi, 2 * np.pi])


def find_best_individual(run_path):
    """Find the best individual in a run by fitness.

    Returns (fitness, gen_idx, ind_id, ind_path) or None.
    """
    evo_df = load_evolution_csv(run_path)
    if evo_df is None or len(evo_df) == 0:
        return None

    best_row = evo_df.loc[evo_df["fitness"].idxmax()]
    gen_idx = int(best_row["generation"])
    ind_id = str(best_row["id"])
    fitness = float(best_row["fitness"])

    # Try unpadded first, then zero-padded (NEAT uses generation_00/individual_0000)
    ind_path = os.path.join(run_path, f"generation_{gen_idx}", f"individual_{ind_id}")
    if not os.path.isdir(ind_path):
        ind_path = os.path.join(run_path, f"generation_{gen_idx:02d}", f"individual_{ind_id}")
    if not os.path.isdir(ind_path):
        return None

    return fitness, gen_idx, ind_id, ind_path


def compute_genome_distance(g1, g2, genotype):
    """Compute genome-level distance between two genomes."""
    cppn_compatibility_distance = lazy_import_cppn_compat()
    if genotype in ("cppn", "neat_cppn"):
        return cppn_compatibility_distance(g1, g2)
    elif genotype in ("hybrid_cppn", "neat_hybrid_cppn"):
        cppn_dist = cppn_compatibility_distance(g1.cppn, g2.cppn)
        direct_dist = compute_euclidean_distance(g1.direct, g2.direct, _DIRECT_MIN, _DIRECT_MAX)
        return cppn_dist + direct_dist
    else:
        return np.nan


def process_experiment(task, genotype, experiment_dir, top_k=10):
    """Find top-K best individuals across runs and compute pairwise distances."""
    min_vals, max_vals = get_min_max()

    # Collect best individual from each run
    run_bests = []
    for run_name, run_path in iter_runs(experiment_dir):
        result = find_best_individual(run_path)
        if result is None:
            continue
        fitness, gen_idx, ind_id, ind_path = result
        run_bests.append({
            "run": run_name,
            "fitness": fitness,
            "ind_path": ind_path,
        })

    if len(run_bests) < 2:
        return pd.DataFrame(), pd.DataFrame()

    # Sort by fitness descending, take top K
    run_bests.sort(key=lambda x: x["fitness"], reverse=True)
    selected = run_bests[:top_k]

    # Load phenotypes and genotypes for selected individuals
    for entry in selected:
        entry["phenotype"] = load_phenotype(entry["ind_path"])
        entry["genome"] = load_genotype(entry["ind_path"])

    # Compute pairwise distances
    detail_rows = []
    for a, b in combinations(selected, 2):
        pheno_dist = np.nan
        if a["phenotype"] is not None and b["phenotype"] is not None:
            pheno_dist = compute_edit_distance(a["phenotype"], b["phenotype"], min_vals, max_vals)

        genome_dist = np.nan
        if a["genome"] is not None and b["genome"] is not None and genotype != "spherical":
            try:
                genome_dist = compute_genome_distance(a["genome"], b["genome"], genotype)
            except Exception:
                pass

        detail_rows.append({
            "task": task,
            "genotype": genotype,
            "ind_i_run": a["run"],
            "ind_i_fitness": a["fitness"],
            "ind_j_run": b["run"],
            "ind_j_fitness": b["fitness"],
            "phenotypic_distance": pheno_dist,
            "genome_distance": genome_dist,
        })

    detail_df = pd.DataFrame(detail_rows)

    # Summary
    pheno_dists = detail_df["phenotypic_distance"].dropna()
    genome_dists = detail_df["genome_distance"].dropna()

    summary = {
        "task": task,
        "genotype": genotype,
        "n_individuals": len(selected),
        "mean_phenotypic_distance": pheno_dists.mean() if len(pheno_dists) > 0 else np.nan,
        "std_phenotypic_distance": pheno_dists.std() if len(pheno_dists) > 0 else np.nan,
        "min_phenotypic_distance": pheno_dists.min() if len(pheno_dists) > 0 else np.nan,
        "max_phenotypic_distance": pheno_dists.max() if len(pheno_dists) > 0 else np.nan,
        "mean_genome_distance": genome_dists.mean() if len(genome_dists) > 0 else np.nan,
        "std_genome_distance": genome_dists.std() if len(genome_dists) > 0 else np.nan,
    }
    summary_df = pd.DataFrame([summary])

    return detail_df, summary_df


def run(base_dir, tasks, genotypes, **kwargs):
    """Collect top-K cross-run diversity data for all experiments."""
    top_k = kwargs.get("top_k", 10)
    all_summaries = []

    for task, genotype, experiment_dir in iter_experiments(base_dir, tasks, genotypes):
        print(f"Processing {task}/{genotype}")
        detail_df, summary_df = process_experiment(task, genotype, experiment_dir, top_k)

        if len(detail_df) == 0:
            continue

        detail_path = os.path.join(experiment_dir, "top_k_cross_run_diversity.csv")
        detail_df.to_csv(detail_path, index=False)
        print(f"  Saved {len(detail_df)} pairwise comparisons to {detail_path}")

        all_summaries.append(summary_df)

    if all_summaries:
        combined = pd.concat(all_summaries, ignore_index=True)
        summary_path = os.path.join(base_dir, "top_k_cross_run_diversity_summary.csv")
        combined.to_csv(summary_path, index=False)
        print(f"Saved combined summary to {summary_path}")


if __name__ == "__main__":
    from experimentation.config import BASE_DIR, TASKS, GENOTYPES
    run(BASE_DIR, TASKS, GENOTYPES)
