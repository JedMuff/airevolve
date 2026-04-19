#!/usr/bin/env python3
"""Config-driven runner for experiment data collection scripts.

Usage:
    python -m experimentation.run_collection
    python -m experimentation.run_collection --config path/to/config.yaml
    python -m experimentation.run_collection --only fitness_diversity,bloodline
"""

import argparse
import importlib
import os
import sys
import time

# Allow running as `python experimentation/run_collection.py` from repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experimentation.config import load_config, DEFAULT_CONFIG_PATH


COLLECTOR_MODULES = {
    "morphological_descriptors": "experimentation.collectors.morphological_descriptors",
    "fitness_diversity": "experimentation.collectors.fitness_diversity",
    "best_individual": "experimentation.collectors.best_individual",
    "bloodline": "experimentation.collectors.bloodline",
    "learning_descriptors": "experimentation.collectors.learning_descriptors",
    "genome_diversity": "experimentation.collectors.genome_diversity",
    "genotypic_descriptors": "experimentation.collectors.genotypic_descriptors",
    "inter_run_diversity": "experimentation.collectors.inter_run_diversity",
    "top_k_diversity": "experimentation.collectors.top_k_diversity",
    "convergence_speed": "experimentation.collectors.convergence_speed",
    "speciation_dynamics": "experimentation.collectors.speciation_dynamics",
    "locality": "experimentation.collectors.locality",
    "population_divergence": "experimentation.collectors.population_divergence",
    "coverage_density": "experimentation.collectors.coverage_density",
}


def run_collector(name, module_path, base_dir, tasks, genotypes, **kwargs):
    """Import and run a single collector module."""
    try:
        mod = importlib.import_module(module_path)
    except ImportError as e:
        print(f"  Skipping {name}: {e}")
        return False

    print(f"\n{'─' * 60}")
    print(f"Running collector: {name}")
    print(f"{'─' * 60}")

    start = time.time()
    try:
        mod.run(base_dir, tasks, genotypes, **kwargs)
    except Exception as e:
        print(f"  ERROR in {name}: {e}")
        return False

    elapsed = time.time() - start
    print(f"  Completed {name} in {elapsed:.1f}s")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run experiment data collectors")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH,
                        help="Path to YAML config file")
    parser.add_argument("--only", default=None,
                        help="Comma-separated list of collectors to run (e.g. fitness_diversity,bloodline)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    base_dir = cfg["base_dir"]
    tasks = cfg["tasks"]
    genotypes = cfg["genotypes"]
    collectors = cfg["collectors"]

    if args.only:
        collectors = [c.strip() for c in args.only.split(",")]

    print(f"Base dir: {base_dir}")
    print(f"Tasks: {tasks}")
    print(f"Genotypes: {genotypes}")
    print(f"Collectors: {collectors}")

    succeeded = 0
    failed = 0
    for name in collectors:
        module_path = COLLECTOR_MODULES.get(name)
        if module_path is None:
            print(f"\n  Unknown collector: {name}")
            failed += 1
            continue

        kwargs = {}
        if name == "morphological_descriptors":
            kwargs["n_workers"] = cfg.get("n_workers", 24)

        if run_collector(name, module_path, base_dir, tasks, genotypes, **kwargs):
            succeeded += 1
        else:
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Collection complete: {succeeded} succeeded, {failed} failed")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
