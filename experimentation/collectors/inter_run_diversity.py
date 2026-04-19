"""Collect inter-run phenotypic diversity to measure search determinism.

Compares within-run vs between-run phenotypic edit distances at every 10th
generation to assess how much independent runs diverge.
"""

import os
from itertools import combinations

import numpy as np
import pandas as pd

from experimentation.collection_utils import (
    iter_experiments,
    iter_generations,
    iter_individuals,
    iter_runs,
    load_phenotype,
)
from experimentation.config import get_min_max

from airevolve.evolution_tools.evaluators.edit_distance import compute_edit_distance


def load_run_phenotypes_by_gen(run_path, gen_interval=10):
    """Load phenotypes for every gen_interval-th generation.

    Returns dict: gen_idx -> list of phenotype arrays.
    """
    result = {}
    for gen_idx, gen_path in iter_generations(run_path):
        if gen_idx % gen_interval != 0:
            continue
        phenotypes = []
        for ind_id, ind_path in iter_individuals(gen_path):
            p = load_phenotype(ind_path)
            if p is not None:
                phenotypes.append(p)
        if phenotypes:
            result[gen_idx] = phenotypes
    return result


def mean_pairwise_edit_distance(phenotypes, min_vals, max_vals):
    """Compute mean pairwise edit distance for a list of phenotypes."""
    n = len(phenotypes)
    if n < 2:
        return 0.0
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            dists.append(compute_edit_distance(phenotypes[i], phenotypes[j], min_vals, max_vals))
    return np.mean(dists)


def mean_cross_edit_distance(phenos_a, phenos_b, min_vals, max_vals):
    """Compute mean edit distance between all pairs from two populations."""
    dists = []
    for a in phenos_a:
        for b in phenos_b:
            dists.append(compute_edit_distance(a, b, min_vals, max_vals))
    return np.mean(dists) if dists else 0.0


def process_experiment(task, genotype, experiment_dir, gen_interval=10):
    """Compute within-run and between-run diversity for an experiment."""
    min_vals, max_vals = get_min_max()

    # Load phenotypes per run per generation
    runs_data = {}  # run_name -> {gen_idx -> [phenotypes]}
    for run_name, run_path in iter_runs(experiment_dir):
        run_phenos = load_run_phenotypes_by_gen(run_path, gen_interval)
        if run_phenos:
            runs_data[run_name] = run_phenos

    if len(runs_data) < 2:
        return pd.DataFrame(), pd.DataFrame()

    run_names = sorted(runs_data.keys())

    # Find common generations across all runs
    all_gens = set()
    for rd in runs_data.values():
        all_gens.update(rd.keys())
    # Also include the last generation from each run
    for run_name, run_path in iter_runs(experiment_dir):
        all_gen_idxs = []
        for gen_idx, _ in iter_generations(run_path):
            all_gen_idxs.append(gen_idx)
        if all_gen_idxs:
            max_gen = max(all_gen_idxs)
            if max_gen not in all_gens:
                # Load it
                for gen_idx, gen_path in iter_generations(run_path):
                    if gen_idx == max_gen:
                        phenotypes = []
                        for ind_id, ind_path in iter_individuals(gen_path):
                            p = load_phenotype(ind_path)
                            if p is not None:
                                phenotypes.append(p)
                        if phenotypes:
                            if run_name not in runs_data:
                                runs_data[run_name] = {}
                            runs_data[run_name][max_gen] = phenotypes
                            all_gens.add(max_gen)

    detail_rows = []
    summary_rows = []

    for gen in sorted(all_gens):
        # Collect runs that have data for this generation
        available_runs = [(rn, runs_data[rn][gen]) for rn in run_names
                          if gen in runs_data.get(rn, {})]

        if len(available_runs) < 2:
            continue

        # Within-run diversity
        within_divs = {}
        for rn, phenos in available_runs:
            within_divs[rn] = mean_pairwise_edit_distance(phenos, min_vals, max_vals)

        # Between-run diversity for each pair
        between_dists = []
        for (rn_i, ph_i), (rn_j, ph_j) in combinations(available_runs, 2):
            bd = mean_cross_edit_distance(ph_i, ph_j, min_vals, max_vals)
            detail_rows.append({
                "task": task,
                "genotype": genotype,
                "generation": gen,
                "run_i": rn_i,
                "run_j": rn_j,
                "within_run_i": within_divs[rn_i],
                "within_run_j": within_divs[rn_j],
                "between_run_diversity": bd,
            })
            between_dists.append(bd)

        within_vals = list(within_divs.values())
        mean_within = np.mean(within_vals)
        std_within = np.std(within_vals)
        mean_between = np.mean(between_dists)
        std_between = np.std(between_dists)
        divergence_ratio = mean_between / mean_within if mean_within > 0 else np.nan

        summary_rows.append({
            "task": task,
            "genotype": genotype,
            "generation": gen,
            "mean_within_run": mean_within,
            "std_within_run": std_within,
            "mean_between_run": mean_between,
            "std_between_run": std_between,
            "divergence_ratio": divergence_ratio,
        })

    return pd.DataFrame(detail_rows), pd.DataFrame(summary_rows)


def run(base_dir, tasks, genotypes, **kwargs):
    """Collect inter-run diversity data for all experiments."""
    gen_interval = kwargs.get("gen_interval", 1)

    for task, genotype, experiment_dir in iter_experiments(base_dir, tasks, genotypes):
        print(f"Processing {task}/{genotype}")
        detail_df, summary_df = process_experiment(task, genotype, experiment_dir, gen_interval)

        if len(detail_df) == 0:
            continue

        detail_path = os.path.join(experiment_dir, "inter_run_diversity_data.csv")
        summary_path = os.path.join(experiment_dir, "inter_run_diversity_summary.csv")
        detail_df.to_csv(detail_path, index=False)
        summary_df.to_csv(summary_path, index=False)
        print(f"  Saved {len(detail_df)} detail rows to {detail_path}")
        print(f"  Saved {len(summary_df)} summary rows to {summary_path}")


if __name__ == "__main__":
    from experimentation.config import BASE_DIR, TASKS, GENOTYPES
    run(BASE_DIR, TASKS, GENOTYPES)
