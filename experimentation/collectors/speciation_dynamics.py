"""Collect per-generation speciation dynamics from NEAT evolution_data.csv.

Extracts species count, size distribution, fitness per species, and
species birth/extinction events from the species_id column.
"""

import os

import numpy as np
import pandas as pd

from experimentation.collection_utils import (
    iter_experiments,
    iter_runs,
    load_evolution_csv,
)


def process_run(run_name, run_path):
    """Extract speciation dynamics for a single run.

    Returns a list of per-generation row dicts, or empty list if no species_id.
    """
    evo_df = load_evolution_csv(run_path)
    if evo_df is None or "species_id" not in evo_df.columns:
        return []

    # Filter to in-population individuals
    if "in_pop" in evo_df.columns:
        pop_df = evo_df[evo_df["in_pop"] == True]  # noqa: E712
    else:
        pop_df = evo_df

    rows = []
    prev_species = set()

    for gen, gen_df in sorted(pop_df.groupby("generation")):
        species_ids = gen_df["species_id"].dropna().astype(int)
        current_species = set(species_ids.unique())

        # Species counts and sizes
        species_counts = species_ids.value_counts()
        n_species = len(species_counts)
        sizes = species_counts.values

        # Fitness per species
        species_fitness = gen_df.groupby("species_id")["fitness"].agg(["mean", "max"])

        # Birth/extinction
        born = current_species - prev_species if prev_species else current_species
        extinct = prev_species - current_species if prev_species else set()

        rows.append({
            "run": run_name,
            "generation": int(gen),
            "n_species": n_species,
            "mean_species_size": float(np.mean(sizes)),
            "std_species_size": float(np.std(sizes)),
            "min_species_size": int(np.min(sizes)),
            "max_species_size": int(np.max(sizes)),
            "mean_species_mean_fitness": float(species_fitness["mean"].mean()),
            "mean_species_max_fitness": float(species_fitness["max"].mean()),
            "max_species_max_fitness": float(species_fitness["max"].max()),
            "species_born": len(born),
            "species_extinct": len(extinct),
        })

        prev_species = current_species

    return rows


def run(base_dir, tasks, genotypes, **kwargs):
    """Collect speciation dynamics for all experiments.

    Skips non-NEAT genotypes (those without species_id in evolution_data.csv).
    """
    for task, genotype, experiment_dir in iter_experiments(base_dir, tasks, genotypes):
        print(f"Processing {task}/{genotype}")
        all_rows = []

        for run_name, run_path in iter_runs(experiment_dir):
            run_rows = process_run(run_name, run_path)
            all_rows.extend(run_rows)

        if not all_rows:
            print(f"  No speciation data found (species_id column missing)")
            continue

        df = pd.DataFrame(all_rows)
        output_path = os.path.join(experiment_dir, "speciation_dynamics_data.csv")
        df.to_csv(output_path, index=False)
        print(f"  Saved {len(df)} records to {output_path}")


if __name__ == "__main__":
    from experimentation.config import BASE_DIR, TASKS, GENOTYPES
    run(BASE_DIR, TASKS, GENOTYPES)
