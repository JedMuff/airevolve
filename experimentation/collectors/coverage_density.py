"""Collect phenotype space coverage and density metrics per generation.

For each generation's population, computes pairwise phenotypic edit distances
and derives:
- mean_pairwise_distance: average distance between all pairs (spread)
- mean_nn_distance: average nearest-neighbour distance (local sparsity)
- coverage_ratio: mean_nn / mean_pairwise (higher = sparser coverage)
"""

import os

import numpy as np
import pandas as pd

from airevolve.evolution_tools.evaluators.edit_distance import compute_edit_distance

from experimentation.collection_utils import (
    iter_experiments,
    iter_runs,
    iter_generations,
    iter_individuals,
    load_phenotype,
)
from experimentation.config import BASE_DIR, TASKS, GENOTYPES, get_min_max


def _compute_distance_matrix(phenotypes, min_vals, max_vals):
    """Compute full pairwise edit-distance matrix for a list of phenotypes."""
    n = len(phenotypes)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            try:
                d = compute_edit_distance(phenotypes[i], phenotypes[j],
                                          min_vals, max_vals)
            except Exception:
                d = np.nan
            D[i, j] = d
            D[j, i] = d
    return D


def _generation_metrics(phenotypes, min_vals, max_vals):
    """Compute coverage/density metrics for one generation's phenotypes.

    Returns (mean_pairwise_distance, mean_nn_distance, coverage_ratio) or
    (nan, nan, nan) if fewer than 2 valid individuals.
    """
    if len(phenotypes) < 2:
        return np.nan, np.nan, np.nan

    D = _compute_distance_matrix(phenotypes, min_vals, max_vals)

    # Upper triangle for mean pairwise distance
    triu = D[np.triu_indices_from(D, k=1)]
    valid = triu[np.isfinite(triu)]
    if len(valid) == 0:
        return np.nan, np.nan, np.nan

    mean_pairwise = np.mean(valid)

    # Nearest-neighbour: min of each row (excluding self = 0 on diagonal)
    np.fill_diagonal(D, np.inf)
    nn_dists = np.nanmin(D, axis=1)
    nn_dists = nn_dists[np.isfinite(nn_dists)]
    if len(nn_dists) == 0:
        return mean_pairwise, np.nan, np.nan

    mean_nn = np.mean(nn_dists)
    ratio = mean_nn / mean_pairwise if mean_pairwise > 1e-12 else np.nan

    return mean_pairwise, mean_nn, ratio


def _process_run(run_name, run_path, min_vals, max_vals):
    """Process one run, returning a list of per-generation metric dicts."""
    rows = []
    for gen_idx, gen_path in iter_generations(run_path):
        phenotypes = []
        for _ind_id, ind_path in iter_individuals(gen_path):
            pheno = load_phenotype(ind_path)
            if pheno is not None:
                phenotypes.append(pheno)

        mpd, mnn, ratio = _generation_metrics(phenotypes, min_vals, max_vals)
        rows.append({
            "run": run_name,
            "generation": gen_idx,
            "n_individuals": len(phenotypes),
            "mean_pairwise_distance": mpd,
            "mean_nn_distance": mnn,
            "coverage_ratio": ratio,
        })
    return rows


def run(base_dir, tasks, genotypes, **kwargs):
    """Collect coverage/density data for all experiments.

    Parameters
    ----------
    base_dir : str
        Root experiment directory.
    tasks : list[str]
        Task names to process.
    genotypes : list[str]
        Genotype names to process.
    """
    min_vals, max_vals = get_min_max()

    for task, genotype, experiment_dir in iter_experiments(base_dir, tasks, genotypes):
        print(f"Processing coverage_density: {task}/{genotype}")
        all_rows = []

        for run_name, run_path in iter_runs(experiment_dir):
            run_rows = _process_run(run_name, run_path, min_vals, max_vals)
            all_rows.extend(run_rows)

        if not all_rows:
            print(f"  No coverage/density data for {task}/{genotype}")
            continue

        out_df = pd.DataFrame(all_rows)
        output_path = os.path.join(experiment_dir, "coverage_density_data.csv")
        out_df.to_csv(output_path, index=False)
        print(f"  Saved {len(out_df)} rows to {output_path}")


if __name__ == "__main__":
    run(BASE_DIR, TASKS, GENOTYPES)
