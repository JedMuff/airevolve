#!/usr/bin/env python3
"""Illustrate typical mutations for each genome type.

For each encoding (spherical, cppn, hybrid_cppn), finds real parent-child pairs
whose phenotypic distance is closest to Q1, median, and Q3 of the mutation
distance distribution. Saves individual drone images, phenotype/genotype files,
and metadata for manual figure composition.
"""

import json
import os
import shutil
import sys
import types

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Mock fcl before importing genome handler modules
sys.modules.setdefault("fcl", types.ModuleType("fcl"))

from airevolve.evolution_tools.inspection_tools.drone_visualizer import (
    DroneVisualizer,
    VisualizationConfig,
)
from experimentation.collection_utils import (
    load_phenotype,
    load_genotype,
    iter_runs,
)
from experimentation.collectors.locality import _build_individual_lookup
from experimentation.config import (
    BASE_DIR,
    GENOTYPE_LABELS,
    figures_dir,
)
from experimentation.plot_utils import setup_style, save_figure

TASKS = ["backandforth", "figure8", "circle"]
GENOTYPES = ["spherical", "cppn", "hybrid_cppn"]
QUANTILES = [("q1", 25), ("median", 50), ("q3", 75)]


# ── Data loading ─────────────────────────────────────────────────────────────

def _load_pooled_locality(base_dir, genotype):
    """Load and concatenate locality data across all tasks for one genotype."""
    frames = []
    for task in TASKS:
        path = os.path.join(base_dir, task, genotype, "locality_data.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        df["task"] = task
        frames.append(df)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def _compute_distribution_stats(distances):
    """Compute mean, Q1, median, Q3 from an array of distances."""
    clean = distances[np.isfinite(distances)]
    q1, med, q3 = np.percentile(clean, [25, 50, 75])
    return {
        "mean": float(np.mean(clean)),
        "q1": float(q1),
        "median": float(med),
        "q3": float(q3),
        "n": int(len(clean)),
    }


# ── Pair finding ─────────────────────────────────────────────────────────────

def _find_pair(df, target_distance, mean_fitness, base_dir, genotype, lookup_cache):
    """Find a parent-child pair closest to target_distance and mean fitness.

    Ranks candidates by a combined score of normalised distance delta and
    normalised fitness delta (average of parent and child fitness vs mean).

    Returns dict with pair info and loaded phenotypes, or None.
    """
    df = df.copy()

    # Normalise distance delta to [0, 1]
    dist_delta = (df["phenotypic_distance"] - target_distance).abs()
    dist_range = dist_delta.max()
    df["_dist_score"] = dist_delta / dist_range if dist_range > 1e-12 else 0.0

    # Normalise fitness delta to [0, 1] (average of parent+child vs mean)
    avg_fitness = (df["parent_fitness"] + df["child_fitness"]) / 2.0
    fit_delta = (avg_fitness - mean_fitness).abs()
    fit_range = fit_delta.max()
    df["_fit_score"] = fit_delta / fit_range if fit_range > 1e-12 else 0.0

    # Combined score (equal weight)
    df["_score"] = df["_dist_score"] + df["_fit_score"]
    df = df.sort_values("_score")

    for _, row in df.iterrows():
        task = row["task"]
        run = row["run"]
        child_id = int(row["child_id"])
        parent_id = int(row["parent_id"])

        # Build/cache individual lookup for this run
        cache_key = (genotype, task, run)
        if cache_key not in lookup_cache:
            run_path = os.path.join(base_dir, task, genotype, run)
            if not os.path.isdir(run_path):
                continue
            lookup_cache[cache_key] = _build_individual_lookup(run_path)

        lookup = lookup_cache[cache_key]
        child_path = lookup.get(child_id)
        parent_path = lookup.get(parent_id)
        if child_path is None or parent_path is None:
            continue

        child_pheno = load_phenotype(child_path)
        parent_pheno = load_phenotype(parent_path)
        if child_pheno is None or parent_pheno is None:
            continue

        return {
            "task": task,
            "run": run,
            "generation": int(row["generation"]),
            "child_id": child_id,
            "parent_id": parent_id,
            "phenotypic_distance": float(row["phenotypic_distance"]),
            "parent_fitness": float(row["parent_fitness"]),
            "child_fitness": float(row["child_fitness"]),
            "fitness_diff": float(row["fitness_diff"]),
            "child_path": child_path,
            "parent_path": parent_path,
            "child_pheno": child_pheno,
            "parent_pheno": parent_pheno,
        }

    return None


# ── Output helpers ───────────────────────────────────────────────────────────

def _save_drone_image(phenotype, out_dir, filename):
    """Save a standalone 3D drone visualization."""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    viz = DroneVisualizer(VisualizationConfig(
        elevation=30, azimuth=45,
        show_axis_ticks=False, show_axis=False,
    ))
    viz.plot_3d(phenotype, ax=ax)
    save_figure(fig, out_dir, filename)


def _save_artifacts(pair, out_dir, quantile_label, target_distance, genotype, dist_stats):
    """Save drone images, phenotype/genotype files, and metadata for one pair."""
    # Drone images
    _save_drone_image(pair["parent_pheno"], out_dir, f"{quantile_label}_parent")
    _save_drone_image(pair["child_pheno"], out_dir, f"{quantile_label}_child")

    # Phenotype arrays
    np.save(os.path.join(out_dir, f"{quantile_label}_parent_phenotype.npy"),
            pair["parent_pheno"])
    np.save(os.path.join(out_dir, f"{quantile_label}_child_phenotype.npy"),
            pair["child_pheno"])

    # Genotype pickles (copy from source if they exist)
    for role in ("parent", "child"):
        src_pkl = os.path.join(pair[f"{role}_path"], "genotype.pkl")
        if os.path.exists(src_pkl):
            shutil.copy2(src_pkl,
                         os.path.join(out_dir, f"{quantile_label}_{role}_genotype.pkl"))

    # Metadata JSON
    meta = {
        "genotype": genotype,
        "quantile": quantile_label,
        "target_distance": target_distance,
        "actual_distance": pair["phenotypic_distance"],
        "parent_fitness": pair["parent_fitness"],
        "child_fitness": pair["child_fitness"],
        "fitness_diff": pair["fitness_diff"],
        "source_task": pair["task"],
        "source_run": pair["run"],
        "source_generation": pair["generation"],
        "parent_id": pair["parent_id"],
        "child_id": pair["child_id"],
        "distribution_stats": dist_stats,
    }
    with open(os.path.join(out_dir, f"{quantile_label}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


# ── Main ─────────────────────────────────────────────────────────────────────

def main(base_dir=None):
    base_dir = base_dir or BASE_DIR
    setup_style()

    print("=" * 70)
    print("Typical Mutation Illustration")
    print("=" * 70)

    for genotype in GENOTYPES:
        label = GENOTYPE_LABELS.get(genotype, genotype)
        print(f"\n{'─' * 60}")
        print(f"Genome type: {label} ({genotype})")
        print(f"{'─' * 60}")

        df = _load_pooled_locality(base_dir, genotype)
        if df is None or len(df) == 0:
            print("  No locality data found.")
            continue

        # Filter to pairs where both parent and child can fly
        df = df[(df["parent_fitness"] > 5) & (df["child_fitness"] > 5)]
        if len(df) == 0:
            print("  No pairs with both parent and child fitness > 5.")
            continue

        distances = df["phenotypic_distance"].dropna().values
        dist_stats = _compute_distribution_stats(distances)

        # Mean fitness (for combined ranking)
        all_fitness = pd.concat([df["parent_fitness"], df["child_fitness"]]).dropna()
        mean_fitness = float(all_fitness.mean())
        dist_stats["mean_fitness"] = mean_fitness

        print(f"  Distribution (n={dist_stats['n']}):")
        print(f"    mean_dist={dist_stats['mean']:.4f}  "
              f"Q1={dist_stats['q1']:.4f}  "
              f"median={dist_stats['median']:.4f}  "
              f"Q3={dist_stats['q3']:.4f}")
        print(f"    mean_fitness={mean_fitness:.3f}")

        out_dir = figures_dir(base_dir, plotter_name=f"typical_mutation/{genotype}")
        lookup_cache = {}

        for q_label, q_pct in QUANTILES:
            target = np.percentile(distances[np.isfinite(distances)], q_pct)
            pair = _find_pair(df, target, mean_fitness, base_dir, genotype, lookup_cache)

            if pair is None:
                print(f"  {q_label}: no valid pair found")
                continue

            _save_artifacts(pair, out_dir, q_label, target, genotype, dist_stats)

            print(f"  {q_label} (target={target:.4f}):")
            print(f"    actual_dist={pair['phenotypic_distance']:.4f}  "
                  f"parent_fit={pair['parent_fitness']:.3f}  "
                  f"child_fit={pair['child_fitness']:.3f}  "
                  f"diff={pair['fitness_diff']:+.3f}")
            print(f"    source: {pair['task']}/{pair['run']}  "
                  f"gen={pair['generation']}  "
                  f"parent={pair['parent_id']}  child={pair['child_id']}")

    print(f"\n{'=' * 70}")
    print("Done.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
