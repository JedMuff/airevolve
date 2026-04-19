#!/usr/bin/env python3
"""Config-driven runner for experiment plotting scripts.

Usage:
    python -m experimentation.run_plotting
    python -m experimentation.run_plotting --config path/to/config.yaml
    python -m experimentation.run_plotting --only fitness_diversity,bloodline
    python -m experimentation.run_plotting --latex
"""

import argparse
import importlib
import os
import sys
import time

# Allow running as `python experimentation/run_plotting.py` from repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experimentation.config import load_config, DEFAULT_CONFIG_PATH
from experimentation.plot_utils import setup_style


PLOTTER_MODULES = {
    "fitness_diversity": "experimentation.plotters.fitness_diversity",
    "morphological_descriptors": "experimentation.plotters.morphological_descriptors",
    "learning_descriptors": "experimentation.plotters.learning_descriptors",
    "bloodline": "experimentation.plotters.bloodline",
    "genome_diversity": "experimentation.plotters.genome_diversity",
    "genotypic_descriptors": "experimentation.plotters.genotypic_descriptors",
    "inter_run_diversity": "experimentation.plotters.inter_run_diversity",
    "top_k_diversity": "experimentation.plotters.top_k_diversity",
    "best_individual": "experimentation.plotters.best_individual",
    "best_evaluations": "experimentation.plotters.best_evaluations",
    "genome_visualization": "experimentation.plotters.genome_visualization",
    "convergence_speed": "experimentation.plotters.convergence_speed",
    "speciation_dynamics": "experimentation.plotters.speciation_dynamics",
    "locality": "experimentation.plotters.locality",
    "population_divergence": "experimentation.plotters.population_divergence",
    "morphological_descriptors_against_fitness": "experimentation.plotters.morphological_descriptors_against_fitness",
    "coverage_density": "experimentation.plotters.coverage_density",
}


def run_plotter(name, module_path, base_dir, tasks, genotypes):
    """Import and run a single plotter module."""
    try:
        mod = importlib.import_module(module_path)
    except ImportError as e:
        print(f"  Skipping {name}: {e}")
        return False

    print(f"\n{'─' * 60}")
    print(f"Running plotter: {name}")
    print(f"{'─' * 60}")

    start = time.time()
    try:
        mod.run(base_dir, tasks, genotypes)
    except Exception as e:
        print(f"  ERROR in {name}: {e}")
        import traceback
        traceback.print_exc()
        return False

    elapsed = time.time() - start
    print(f"  Completed {name} in {elapsed:.1f}s")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run experiment plotters")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH,
                        help="Path to YAML config file")
    parser.add_argument("--only", default=None,
                        help="Comma-separated list of plotters to run")
    parser.add_argument("--latex", action="store_true",
                        help="Generate LaTeX figure code after plotting")
    args = parser.parse_args()

    cfg = load_config(args.config)
    base_dir = cfg["base_dir"]
    tasks = cfg["tasks"]
    genotypes = cfg["genotypes"]
    plotters = cfg["plotters"]

    # Apply style from config
    setup_style(cfg)

    if args.only:
        plotters = [p.strip() for p in args.only.split(",")]

    print(f"Base dir: {base_dir}")
    print(f"Tasks: {tasks}")
    print(f"Genotypes: {genotypes}")
    print(f"Plotters: {plotters}")

    succeeded = 0
    failed = 0
    for name in plotters:
        module_path = PLOTTER_MODULES.get(name)
        if module_path is None:
            print(f"\n  Unknown plotter: {name}")
            failed += 1
            continue

        if run_plotter(name, module_path, base_dir, tasks, genotypes):
            succeeded += 1
        else:
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Plotting complete: {succeeded} succeeded, {failed} failed")
    print(f"{'=' * 60}")

    if args.latex:
        print("\nGenerating LaTeX figures...")
        from experimentation.generate_latex_figures import generate_latex
        generate_latex(base_dir)


if __name__ == "__main__":
    main()
