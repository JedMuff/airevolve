"""Collect CMA-ES learning curve descriptors from learning_curve.json files.

Each individual that passed the hover check has a learning_curve.json with:
    {"stage1": [...], "stage2": [...], "combined": [...]}
where each list contains best-so-far fitness per CMA-ES iteration.
"""

import os

import numpy as np
import pandas as pd

from experimentation.collection_utils import (
    iter_experiments,
    iter_runs,
    iter_generations,
    iter_individuals,
    load_learning_curve,
    build_fitness_lookup,
)
from experimentation.config import BASE_DIR, TASKS, GENOTYPES


def compute_convergence_point(values, epsilon=1e-3):
    """Find first index where remaining values stay within epsilon of final value."""
    if len(values) == 0:
        return 0
    final = values[-1]
    for i in range(len(values)):
        if all(abs(v - final) <= epsilon for v in values[i:]):
            return i
    return len(values) - 1


def compute_learning_descriptors(curve):
    """Extract learning descriptors from a learning_curve.json dict."""
    combined = curve.get("combined", [])
    stage1 = curve.get("stage1", [])
    stage2 = curve.get("stage2", [])

    if not combined:
        return None

    learning_delta = combined[-1] - combined[0]

    # Stability: mean absolute difference between consecutive entries
    if len(combined) > 1:
        diffs = [abs(combined[i + 1] - combined[i]) for i in range(len(combined) - 1)]
        learning_stability = np.mean(diffs)
    else:
        learning_stability = 0.0

    convergence_point = compute_convergence_point(combined)
    asymptotic_performance = max(combined)

    n_stage1_iters = len(stage1)
    n_stage2_iters = len(stage2)

    stage1_improvement = (stage1[-1] - stage1[0]) if len(stage1) > 0 else np.nan
    if len(stage2) > 0:
        stage2_improvement = max(stage2) - stage2[0]
    else:
        stage2_improvement = np.nan

    return {
        "learning_delta": learning_delta,
        "learning_stability": learning_stability,
        "convergence_point": convergence_point,
        "asymptotic_performance": asymptotic_performance,
        "n_stage1_iters": n_stage1_iters,
        "n_stage2_iters": n_stage2_iters,
        "stage1_improvement": stage1_improvement,
        "stage2_improvement": stage2_improvement,
    }


def process_experiment(experiment_dir):
    """Process all runs in an experiment directory."""
    rows = []

    for run_name, run_path in iter_runs(experiment_dir):
        fitness_lookup = build_fitness_lookup(run_path)

        for gen_idx, gen_path in iter_generations(run_path):
            for ind_id, ind_path in iter_individuals(gen_path):
                curve = load_learning_curve(ind_path)
                if curve is None:
                    continue

                descriptors = compute_learning_descriptors(curve)
                if descriptors is None:
                    continue

                fitness = fitness_lookup.get((gen_idx, ind_id), np.nan)

                row = {
                    "run": run_name,
                    "generation": gen_idx,
                    "individual": ind_id,
                    "fitness": fitness,
                }
                row.update(descriptors)
                rows.append(row)

    return pd.DataFrame(rows)


def run(base_dir, tasks, genotypes, **kwargs):
    """Collect CMA-ES learning curve descriptors.

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
        df = process_experiment(experiment_dir)

        if len(df) == 0:
            continue

        output_path = os.path.join(experiment_dir, "learning_descriptors_data.csv")
        df.to_csv(output_path, index=False)
        print(f"Saved {len(df)} records to {output_path}")


if __name__ == "__main__":
    run(BASE_DIR, TASKS, GENOTYPES)
