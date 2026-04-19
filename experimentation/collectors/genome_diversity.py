"""Collect per-generation genotypic diversity for CPPN and hybrid genomes.

For spherical genomes, genotype = phenotype so genome diversity equals phenotypic
edit distance (already collected in fitness_diversity_data.csv). This script
focuses on CPPN/hybrid where genome-level distance differs from phenotypic distance.
"""

import os

import numpy as np
import pandas as pd

from experimentation.collection_utils import (
    iter_experiments,
    iter_runs,
    iter_generations,
    iter_individuals,
    load_genotype,
    load_phenotype,
    lazy_import_cppn_compat,
)
from experimentation.config import PARAMETER_LIMITS, get_min_max

from airevolve.evolution_tools.evaluators.edit_distance import compute_euclidean_distance


# Direct parameter limits for hybrid: [magnitude, arm_yaw, arm_pitch]
_DIRECT_MIN = np.array([0.09, 0.0, 0.0])
_DIRECT_MAX = np.array([0.4, 2 * np.pi, 2 * np.pi])


def hybrid_genome_distance(g1, g2):
    """Compute distance between two HybridGenome objects.

    Combines CPPN compatibility distance with normalized Euclidean distance
    of the direct arm parameters.
    """
    cppn_compatibility_distance = lazy_import_cppn_compat()
    cppn_dist = cppn_compatibility_distance(g1.cppn, g2.cppn)
    direct_dist = compute_euclidean_distance(g1.direct, g2.direct, _DIRECT_MIN, _DIRECT_MAX)
    return cppn_dist + direct_dist


def compute_mean_pairwise_distance(genomes, distance_fn):
    """Compute mean pairwise distance for a list of genomes."""
    n = len(genomes)
    if n < 2:
        return 0.0

    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += distance_fn(genomes[i], genomes[j])
            count += 1

    return total / count if count > 0 else 0.0


def process_experiment(experiment_dir, genotype):
    """Process all runs and compute per-generation genome diversity."""
    rows = []

    if genotype in ("spherical", "neat_spherical"):
        # For spherical, genotype = phenotype. Recompute using phenotypic edit distance.
        min_vals, max_vals = get_min_max()

        for run_name, run_path in iter_runs(experiment_dir):
            for gen_idx, gen_path in iter_generations(run_path):
                phenotypes = []
                for ind_id, ind_path in iter_individuals(gen_path):
                    p = load_phenotype(ind_path)
                    if p is not None:
                        phenotypes.append(p)

                if len(phenotypes) < 2:
                    continue

                def pheno_dist(a, b):
                    from airevolve.evolution_tools.evaluators.edit_distance import compute_edit_distance
                    return compute_edit_distance(a, b, min_vals, max_vals)

                dists = []
                for i in range(len(phenotypes)):
                    for j in range(i + 1, len(phenotypes)):
                        dists.append(pheno_dist(phenotypes[i], phenotypes[j]))

                rows.append({
                    "experiment": genotype,
                    "run": run_name,
                    "generation": gen_idx,
                    "genome_diversity_mean": np.mean(dists),
                    "genome_diversity_std": np.std(dists),
                })
        return pd.DataFrame(rows)

    # CPPN or hybrid
    cppn_compat_fn = lazy_import_cppn_compat()

    for run_name, run_path in iter_runs(experiment_dir):
        for gen_idx, gen_path in iter_generations(run_path):
            genomes = []
            for ind_id, ind_path in iter_individuals(gen_path):
                genome = load_genotype(ind_path)
                if genome is not None:
                    genomes.append(genome)

            if len(genomes) < 2:
                continue

            if genotype in ("cppn", "neat_cppn"):
                dist_fn = cppn_compat_fn
            elif genotype in ("hybrid_cppn", "neat_hybrid_cppn"):
                dist_fn = hybrid_genome_distance
            else:
                continue

            dists = []
            for i in range(len(genomes)):
                for j in range(i + 1, len(genomes)):
                    try:
                        dists.append(dist_fn(genomes[i], genomes[j]))
                    except Exception:
                        pass

            if dists:
                rows.append({
                    "experiment": genotype,
                    "run": run_name,
                    "generation": gen_idx,
                    "genome_diversity_mean": np.mean(dists),
                    "genome_diversity_std": np.std(dists),
                })

    return pd.DataFrame(rows)


def run(base_dir, tasks, genotypes, **kwargs):
    """Collect genome diversity data for all experiments."""
    for task, genotype, experiment_dir in iter_experiments(base_dir, tasks, genotypes):
        print(f"Processing {task}/{genotype}")
        df = process_experiment(experiment_dir, genotype)

        if len(df) == 0:
            continue

        output_path = os.path.join(experiment_dir, "genome_diversity_data.csv")
        df.to_csv(output_path, index=False)
        print(f"  Saved {len(df)} records to {output_path}")


if __name__ == "__main__":
    from experimentation.config import BASE_DIR, TASKS, GENOTYPES
    run(BASE_DIR, TASKS, GENOTYPES)
