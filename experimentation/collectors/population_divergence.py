"""Collect per-generation population divergence from the initial generation.

Measures how much the current population has diverged from generation 0 by
computing the mean phenotypic edit distance between all individuals in gen N
and all individuals in gen 0 (cross-population distance).  Unlike bloodline
divergence this does not require lineage tracking, so it works for both
mu+lambda and NEAT (crossover) experiments.
"""

import os

import numpy as np
import pandas as pd

from experimentation.collection_utils import (
    iter_experiments,
    iter_runs,
    iter_generations,
    iter_individuals,
    load_phenotype,
)
from experimentation.config import get_min_max

from airevolve.evolution_tools.evaluators.edit_distance import compute_edit_distance


def _load_generation_phenotypes(gen_path):
    """Load all phenotypes for a single generation directory."""
    phenotypes = []
    for ind_id, ind_path in iter_individuals(gen_path):
        p = load_phenotype(ind_path)
        if p is not None:
            phenotypes.append(p)
    return phenotypes


def _mean_cross_distance(phenos_a, phenos_b, min_vals, max_vals):
    """Mean edit distance between all pairs from two populations."""
    dists = []
    for a in phenos_a:
        for b in phenos_b:
            dists.append(compute_edit_distance(a, b, min_vals, max_vals))
    return np.mean(dists) if dists else np.nan


def process_run(run_name, run_path, min_vals, max_vals):
    """Compute per-generation divergence from gen-0 for a single run."""
    # Load gen-0 phenotypes
    gen0_phenotypes = None
    all_gen_phenotypes = {}

    for gen_idx, gen_path in iter_generations(run_path):
        phenotypes = _load_generation_phenotypes(gen_path)
        if not phenotypes:
            continue
        all_gen_phenotypes[gen_idx] = phenotypes
        if gen_idx == 0:
            gen0_phenotypes = phenotypes

    if gen0_phenotypes is None or len(gen0_phenotypes) < 2:
        return []

    # Baseline: mean pairwise distance within gen-0 itself
    baseline = _mean_cross_distance(
        gen0_phenotypes, gen0_phenotypes, min_vals, max_vals
    )

    rows = []
    for gen_idx in sorted(all_gen_phenotypes.keys()):
        gen_phenotypes = all_gen_phenotypes[gen_idx]
        cross_dist = _mean_cross_distance(
            gen_phenotypes, gen0_phenotypes, min_vals, max_vals
        )
        rows.append({
            "run": run_name,
            "generation": gen_idx,
            "mean_divergence": cross_dist - baseline,
            "n_current": len(gen_phenotypes),
            "n_gen0": len(gen0_phenotypes),
        })

    return rows


def run(base_dir, tasks, genotypes, **kwargs):
    """Collect population divergence from gen-0 for all experiments.

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
        print(f"Processing {task}/{genotype}")
        all_rows = []

        for run_name, run_path in iter_runs(experiment_dir):
            run_rows = process_run(run_name, run_path, min_vals, max_vals)
            all_rows.extend(run_rows)

        if all_rows:
            df = pd.DataFrame(all_rows)
            output_path = os.path.join(experiment_dir, "population_divergence_data.csv")
            df.to_csv(output_path, index=False)
            print(f"  Saved {len(df)} records to {output_path}")


if __name__ == "__main__":
    from experimentation.config import BASE_DIR, TASKS, GENOTYPES
    run(BASE_DIR, TASKS, GENOTYPES)
