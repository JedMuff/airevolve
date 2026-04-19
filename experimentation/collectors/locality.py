"""Collect parent-child locality data: distance vs fitness difference.

Locality measures whether small changes in genotype/phenotype produce small
changes in fitness. For each parent-child pair we compute:
- phenotypic distance (edit distance on phenotype arrays)
- genotypic distance (CPPN compatibility for cppn/hybrid, edit distance for spherical)
- fitness difference (child - parent)
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

from airevolve.evolution_tools.evaluators.edit_distance import (
    compute_edit_distance,
    compute_euclidean_distance,
)


# Direct parameter limits for hybrid: [magnitude, arm_yaw, arm_pitch]
_DIRECT_MIN = np.array([0.09, 0.0, 0.0])
_DIRECT_MAX = np.array([0.4, 2 * np.pi, 2 * np.pi])


def _parse_parent_ids(val):
    """Parse parent_ids column (string repr of list) into a list of ints/Nones."""
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
                    try:
                        normalized.append(int(x))
                    except (ValueError, TypeError):
                        normalized.append(x)
            return normalized
        return [None, None]
    except (ValueError, SyntaxError):
        return [None, None]


def _build_individual_lookup(run_path):
    """Build int_id -> ind_path mapping across all generations.

    Keeps first-seen path per ID (birth generation) so that elite copies
    in later generations don't overwrite the original.
    """
    lookup = {}
    for gen_idx, gen_path in iter_generations(run_path):
        for ind_id, ind_path in iter_individuals(gen_path):
            try:
                key_id = int(ind_id)
            except ValueError:
                key_id = ind_id
            if key_id not in lookup:
                lookup[key_id] = ind_path
    return lookup


def _hybrid_genome_distance(g1, g2):
    """Compute distance between two HybridGenome objects."""
    cppn_compatibility_distance = lazy_import_cppn_compat()
    cppn_dist = cppn_compatibility_distance(g1.cppn, g2.cppn)
    direct_dist = compute_euclidean_distance(g1.direct, g2.direct, _DIRECT_MIN, _DIRECT_MAX)
    return cppn_dist + direct_dist


def _compute_genotypic_distance(geno1, geno2, genotype):
    """Dispatch to the correct genotypic distance function."""
    if genotype in ("cppn", "neat_cppn"):
        cppn_compatibility_distance = lazy_import_cppn_compat()
        return cppn_compatibility_distance(geno1, geno2)
    elif genotype in ("hybrid_cppn", "neat_hybrid_cppn"):
        return _hybrid_genome_distance(geno1, geno2)
    return None


def _process_run(run_name, run_path, genotype):
    """Process one run and return locality rows."""
    df = load_evolution_csv(run_path)
    if df is None or len(df) == 0:
        return []

    ind_lookup = _build_individual_lookup(run_path)
    min_vals, max_vals = get_min_max()

    # Build id -> fitness lookup from in-population individuals.
    # Use the last occurrence per id (latest generation where fitness was evaluated).
    fitness_lookup = {}
    for _, row in df.iterrows():
        try:
            fitness_lookup[int(row["id"])] = float(row["fitness"])
        except (ValueError, TypeError):
            pass

    is_spherical = genotype in ("spherical", "neat_spherical")
    needs_genotype = not is_spherical

    rows = []
    skipped = 0

    for _, row in df.iterrows():
        gen = int(row["generation"])
        if gen == 0:
            continue

        try:
            child_id = int(row["id"])
        except (ValueError, TypeError):
            continue

        parent_ids = _parse_parent_ids(row.get("parent_ids"))
        child_in_pop = bool(row.get("in_pop", False))
        child_fitness = fitness_lookup.get(child_id)
        if child_fitness is None or np.isnan(child_fitness):
            continue

        child_path = ind_lookup.get(child_id)
        if child_path is None:
            skipped += 1
            continue

        child_pheno = load_phenotype(child_path)
        if child_pheno is None:
            skipped += 1
            continue

        child_geno = None
        if needs_genotype:
            child_geno = load_genotype(child_path)

        for parent_id in parent_ids:
            if parent_id is None:
                continue

            parent_fitness = fitness_lookup.get(parent_id)
            if parent_fitness is None or np.isnan(parent_fitness):
                continue

            parent_path = ind_lookup.get(parent_id)
            if parent_path is None:
                skipped += 1
                continue

            parent_pheno = load_phenotype(parent_path)
            if parent_pheno is None:
                skipped += 1
                continue

            # Phenotypic distance
            try:
                pheno_dist = compute_edit_distance(
                    child_pheno, parent_pheno, min_vals, max_vals
                )
            except Exception:
                pheno_dist = np.nan

            # Genotypic distance
            geno_dist = np.nan
            if is_spherical:
                geno_dist = pheno_dist
            elif needs_genotype:
                parent_geno = load_genotype(parent_path)
                if child_geno is not None and parent_geno is not None:
                    try:
                        geno_dist = _compute_genotypic_distance(
                            child_geno, parent_geno, genotype
                        )
                    except Exception:
                        geno_dist = np.nan

            fitness_diff = child_fitness - parent_fitness

            rows.append({
                "run": run_name,
                "generation": gen,
                "child_id": child_id,
                "parent_id": parent_id,
                "in_pop": child_in_pop,
                "phenotypic_distance": pheno_dist,
                "genotypic_distance": geno_dist,
                "child_fitness": child_fitness,
                "parent_fitness": parent_fitness,
                "fitness_diff": fitness_diff,
                "abs_fitness_diff": abs(fitness_diff),
            })

    if skipped > 0:
        print(f"    {run_name}: skipped {skipped} pairs (missing data)")

    return rows


def run(base_dir, tasks, genotypes, **kwargs):
    """Collect locality data for all experiments.

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
        print(f"Processing locality: {task}/{genotype}")
        all_rows = []

        for run_name, run_path in iter_runs(experiment_dir):
            run_rows = _process_run(run_name, run_path, genotype)
            all_rows.extend(run_rows)

        if not all_rows:
            print(f"  No locality data for {task}/{genotype}")
            continue

        out_df = pd.DataFrame(all_rows)
        output_path = os.path.join(experiment_dir, "locality_data.csv")
        out_df.to_csv(output_path, index=False)
        print(f"  Saved {len(out_df)} locality records to {output_path}")


if __name__ == "__main__":
    run(BASE_DIR, TASKS, GENOTYPES)
