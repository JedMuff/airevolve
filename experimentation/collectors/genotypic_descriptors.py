"""Collect structural properties of CPPN/hybrid genomes from genotype.pkl files."""

import os
from collections import defaultdict

import numpy as np
import pandas as pd

from experimentation.collection_utils import (
    iter_experiments,
    iter_runs,
    iter_generations,
    iter_individuals,
    load_genotype,
    build_fitness_lookup,
    lazy_import_cppn,
)
from experimentation.config import BASE_DIR, TASKS, GENOTYPES


def compute_longest_path(network):
    """Compute longest input->output path in the CPPN DAG."""
    _, topological_sort_fn = lazy_import_cppn()
    try:
        order = topological_sort_fn(network)
    except ValueError:
        return 0

    children = defaultdict(list)
    for conn in network.get_enabled_connections():
        children[conn.source_id].append(conn.target_id)

    dist = {nid: 0 for nid in order}
    for nid in order:
        for child in children[nid]:
            dist[child] = max(dist[child], dist[nid] + 1)

    output_ids = {n.node_id for n in network.get_output_nodes()}
    return max((dist[nid] for nid in output_ids), default=0)


def compute_cppn_descriptors(network):
    """Extract structural metrics from a CPPNNetwork."""
    return {
        "n_hidden_nodes": len(network.get_hidden_nodes()),
        "n_enabled_connections": len(network.get_enabled_connections()),
        "n_total_connections": len(network.connections),
        "longest_path": compute_longest_path(network),
    }


def process_experiment(experiment_dir, genotype):
    """Process all runs in an experiment directory."""
    if genotype in ("spherical", "neat_spherical"):
        return pd.DataFrame()

    CPPNNetwork, _ = lazy_import_cppn()
    rows = []

    for run_name, run_path in iter_runs(experiment_dir):
        fitness_lookup = build_fitness_lookup(run_path)

        for gen_idx, gen_path in iter_generations(run_path):
            for ind_id, ind_path in iter_individuals(gen_path):
                genome = load_genotype(ind_path)
                if genome is None:
                    continue

                fitness = fitness_lookup.get((gen_idx, ind_id), np.nan)

                row = {
                    "run": run_name,
                    "generation": gen_idx,
                    "individual": ind_id,
                    "fitness": fitness,
                    "genome_type": genotype,
                }

                if genotype in ("cppn", "neat_cppn"):
                    if isinstance(genome, CPPNNetwork):
                        row.update(compute_cppn_descriptors(genome))
                    else:
                        continue
                elif genotype in ("hybrid_cppn", "neat_hybrid_cppn"):
                    if hasattr(genome, "cppn") and hasattr(genome, "direct"):
                        row.update(compute_cppn_descriptors(genome.cppn))
                        row["n_direct_arms"] = genome.direct.shape[0]
                    else:
                        continue
                else:
                    continue

                rows.append(row)

    return pd.DataFrame(rows)


def run(base_dir, tasks, genotypes, **kwargs):
    """Collect CPPN/hybrid genome structural descriptors.

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
        df = process_experiment(experiment_dir, genotype)

        if len(df) == 0:
            continue

        output_path = os.path.join(experiment_dir, "genotypic_descriptors_data.csv")
        df.to_csv(output_path, index=False)
        print(f"Saved {len(df)} records to {output_path}")


if __name__ == "__main__":
    run(BASE_DIR, TASKS, GENOTYPES)
