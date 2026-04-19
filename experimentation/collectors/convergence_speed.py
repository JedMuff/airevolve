"""Collect convergence speed metrics from fitness_diversity_data.csv files.

Derived analysis — reads already-collected fitness/diversity CSVs and computes
per-run convergence metrics: generation-to-threshold, normalized AUC, and
early-generation fitness snapshots.
"""

import os

import numpy as np
import pandas as pd

from experimentation.collection_utils import iter_experiments


def _gen_to_threshold(cummax_series, generations, threshold):
    """First generation where cummax(max_fitness) >= threshold.

    Returns generation number or NaN if never reached.
    """
    mask = cummax_series >= threshold
    if mask.any():
        return int(generations[mask.idxmin() if hasattr(mask, 'idxmin') else np.argmax(mask)])
    return np.nan


def _process_run(run_df):
    """Compute convergence metrics for a single run."""
    run_df = run_df.sort_values("generation")
    generations = run_df["generation"].values
    max_fitness = run_df["max_fitness"].values

    # Cumulative max to handle re-evaluation noise
    cummax = np.maximum.accumulate(max_fitness)

    final_fitness = max_fitness[-1]
    n_gens = generations[-1] - generations[0]

    result = {"final_fitness": final_fitness}

    # Generation-to-threshold (using cummax for monotonicity)
    for pct in (50, 75, 90):
        key = f"gen_to_{pct}pct"
        if final_fitness <= 0:
            result[key] = np.nan
        else:
            threshold = pct / 100.0 * final_fitness
            reached = cummax >= threshold
            if reached.any():
                result[key] = float(generations[np.argmax(reached)])
            else:
                result[key] = np.nan

    # AUC
    auc_raw = float(np.trapezoid(max_fitness, generations))
    result["auc_raw"] = auc_raw
    if final_fitness > 0 and n_gens > 0:
        result["auc_normalized"] = auc_raw / (final_fitness * n_gens)
    else:
        result["auc_normalized"] = np.nan

    # Early-generation fitness snapshots
    gen_lookup = dict(zip(generations, max_fitness))
    for g in (5, 10, 15, 25):
        result[f"fitness_gen_{g}"] = gen_lookup.get(g, np.nan)

    return result


def run(base_dir, tasks, genotypes, **kwargs):
    """Collect convergence speed metrics for all experiments."""
    for task, genotype, exp_dir in iter_experiments(base_dir, tasks, genotypes):
        csv_path = os.path.join(exp_dir, "fitness_diversity_data.csv")
        if not os.path.exists(csv_path):
            print(f"  Skipping {task}/{genotype}: no fitness_diversity_data.csv")
            continue

        df = pd.read_csv(csv_path)
        rows = []
        for run_name, run_df in df.groupby("run"):
            metrics = _process_run(run_df)
            metrics["task"] = task
            metrics["genotype"] = genotype
            metrics["run"] = run_name
            rows.append(metrics)

        if not rows:
            continue

        out_df = pd.DataFrame(rows)
        # Reorder columns
        cols = ["task", "genotype", "run", "final_fitness",
                "gen_to_50pct", "gen_to_75pct", "gen_to_90pct",
                "auc_raw", "auc_normalized",
                "fitness_gen_5", "fitness_gen_10", "fitness_gen_15", "fitness_gen_25"]
        out_df = out_df[[c for c in cols if c in out_df.columns]]

        out_path = os.path.join(exp_dir, "convergence_speed_data.csv")
        out_df.to_csv(out_path, index=False)
        print(f"  Saved: {out_path} ({len(rows)} runs)")
