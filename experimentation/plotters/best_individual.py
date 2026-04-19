"""Plotter for best individual blueprint visualizations.

Produces per-task, per-genotype figures:
  - One standalone 3D isometric blueprint per individual (no subplot grids)
  - numpy array text export of designs
"""

import os
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from airevolve.evolution_tools.inspection_tools.drone_visualizer import DroneVisualizer

from experimentation.config import (
    BASE_DIR, TASKS, GENOTYPES, COLORS, GENOTYPE_LABELS, TASK_LABELS,
    figures_dir, csv_dir,
)
from experimentation.plot_utils import (
    setup_style, save_figure, apply_axis_style,
    print_analysis_header, print_analysis_footer,
)

warnings.filterwarnings("ignore")

PLOTTER_NAME = "best_individual"


# ── Data loading ─────────────────────────────────────────────────────────────

def _load_individual_data(csv_path):
    """Load individual data from CSV, converting individual_data strings to numpy arrays."""
    df = pd.read_csv(csv_path)
    for idx, row in df.iterrows():
        try:
            raw = row["individual_data"]
            if isinstance(raw, float):
                df.at[idx, "individual_data"] = None
                continue
            individual_data = eval(raw)
            df.at[idx, "individual_data"] = np.array(individual_data)
        except Exception:
            df.at[idx, "individual_data"] = None
    return df


# ── Single-individual blueprint ──────────────────────────────────────────────

def _plot_single_blueprint(individual, out_dir, filename):
    """Plot a single isometric 3D blueprint and save it."""
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")

    visualizer = DroneVisualizer()
    visualizer.plot_3d(individual, ax=ax, elevation=30, azimuth=45,
                       include_motor_orientation=True)

    save_figure(fig, out_dir, filename)


# ── Numpy export ─────────────────────────────────────────────────────────────

def _save_designs_as_numpy(df, out_dir, experiment_label):
    """Save individual designs as copy-pasteable numpy array text."""
    output_file = os.path.join(out_dir, "designs_as_numpy_arrays.txt")
    os.makedirs(out_dir, exist_ok=True)

    with open(output_file, "w") as f:
        f.write(f"# Best Individual Designs - {experiment_label}\n")
        f.write(f"# Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Total individuals: {len(df)}\n\n")
        f.write("import numpy as np\n\n")

        for _, row in df.iterrows():
            individual = row["individual_data"]
            if individual is None:
                f.write(f"# Rank {row['rank']}: Invalid data\n\n")
                continue
            f.write(f"# Rank {row['rank']} - Fitness: {row['fitness']:.6f} "
                    f"- Generation: {row['generation']}\n")
            var_name = f"design_rank_{row['rank']:02d}"
            array_str = np.array2string(individual, precision=6,
                                        separator=", ", suppress_small=True,
                                        max_line_width=100)
            f.write(f"{var_name} = np.array({array_str})\n\n")

        # Convenience dict
        f.write("# Convenience dictionary with all designs\n")
        f.write("all_designs = {\n")
        for _, row in df.iterrows():
            if row["individual_data"] is not None:
                var_name = f"design_rank_{row['rank']:02d}"
                f.write(f"    {row['rank']}: {{'design': {var_name}, "
                        f"'fitness': {row['fitness']:.6f}}},\n")
        f.write("}\n")

    print(f"  Saved designs to {output_file}")


# ── Main entry point ────────────────────────────────────────────────────────

def run(base_dir, tasks, genotypes, output_base=None, max_individuals=15):
    """Generate blueprint figures for best individuals.

    Args:
        base_dir: root data directory
        tasks: list of task names
        genotypes: list of genotype names
        output_base: override for output root (defaults to config.output_dir)
        max_individuals: max number of individuals to plot per experiment
    """
    from experimentation.config import output_dir as _output_dir
    if output_base is None:
        output_base = _output_dir(base_dir)

    setup_style()

    for task in tasks:
        for genotype in genotypes:
            experiment_dir = os.path.join(base_dir, task, genotype)
            csv_path = os.path.join(experiment_dir, "best_individual_data.csv")

            if not os.path.exists(csv_path):
                continue

            print(f"\n{'=' * 60}")
            print(f"Blueprints: {TASK_LABELS.get(task, task)} / "
                  f"{GENOTYPE_LABELS.get(genotype, genotype)}")
            print(f"{'=' * 60}")

            df = _load_individual_data(csv_path)
            if df.empty:
                continue

            if max_individuals is not None:
                df = df.head(max_individuals)

            out_dir = figures_dir(base_dir, task=task, plotter_name=PLOTTER_NAME)

            # Print summary header
            print_analysis_header(
                f"Best Individuals - {TASK_LABELS.get(task, task)} / "
                f"{GENOTYPE_LABELS.get(genotype, genotype)}"
            )

            successful = 0
            for _, row in df.iterrows():
                individual = row["individual_data"]
                if individual is None:
                    continue

                filename = (f"{genotype}_rank_{row['rank']:02d}_"
                           f"fitness_{row['fitness']:.4f}_"
                           f"gen_{row['generation']}")
                try:
                    _plot_single_blueprint(individual, out_dir, filename)
                    successful += 1
                except Exception as e:
                    print(f"  Error plotting rank {row['rank']}: {e}")

            print(f"  Generated {successful}/{len(df)} blueprints")

            # Export numpy arrays
            experiment_label = f"{task}/{genotype}"
            _save_designs_as_numpy(df, out_dir, experiment_label)

            print_analysis_footer(out_dir)


if __name__ == "__main__":
    run(BASE_DIR, TASKS, GENOTYPES)
