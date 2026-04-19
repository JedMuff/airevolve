"""Plotter for genome visualizations alongside drone blueprints.

Produces per-task, per-genotype standalone figures for each best individual:
  - drone blueprint (3D isometric view)
  - CPPN graph (for CPPN / hybrid genotypes)
  - phenotype heatmap (for spherical / hybrid genotypes)
"""

import ast
import os
import pickle
import sys
import types
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Mock fcl before importing genome handler modules
sys.modules.setdefault("fcl", types.ModuleType("fcl"))

from airevolve.evolution_tools.inspection_tools.drone_visualizer import (
    DroneVisualizer,
    VisualizationConfig,
)
from airevolve.evolution_tools.genome_handlers.cppn.network import CPPNNetwork
from airevolve.evolution_tools.genome_handlers.hybrid_cppn_genome_handler import HybridGenome

from experimentation.config import (
    BASE_DIR,
    TASKS,
    GENOTYPES,
    COLORS,
    GENOTYPE_LABELS,
    figures_dir,
)
from experimentation.plot_utils import (
    setup_style,
    save_figure,
    apply_axis_style,
    draw_cppn_graph,
    print_analysis_header,
    print_analysis_footer,
)

warnings.filterwarnings("ignore")

PLOTTER_NAME = "genome_visualization"

# Column labels for phenotype arrays
SPHERICAL_COLUMNS = ["magnitude", "arm_rot", "arm_pitch", "motor_rot", "motor_pitch", "direction"]
HYBRID_DIRECT_COLUMNS = ["magnitude", "arm_yaw", "arm_pitch"]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _load_genotype(pkl_path):
    """Load a genotype from a pickle file."""
    if not pkl_path or not isinstance(pkl_path, str) or not os.path.exists(pkl_path):
        return None
    try:
        with open(pkl_path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def draw_phenotype_heatmap(phenotype, ax, columns=None):
    """Draw a heatmap of the (narms, ncols) phenotype array."""
    if columns is None:
        columns = [f"col_{i}" for i in range(phenotype.shape[1])]

    im = ax.imshow(phenotype, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(phenotype.shape[0]))
    ax.set_yticklabels([f"arm {i}" for i in range(phenotype.shape[0])], fontsize=7)

    for i in range(phenotype.shape[0]):
        for j in range(phenotype.shape[1]):
            val = phenotype[i, j]
            color = "white" if val < (phenotype.max() + phenotype.min()) / 2 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=5, color=color)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _make_drone_figure(phenotype):
    """Create a standalone 3D drone blueprint figure."""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    viz = DroneVisualizer(VisualizationConfig(elevation=30, azimuth=45, show_axis_ticks=False))
    viz.plot_3d(phenotype, ax=ax)
    return fig


def _make_cppn_graph_figure(cppn):
    """Create a standalone CPPN graph figure."""
    fig, ax = plt.subplots(figsize=(8, 6))
    draw_cppn_graph(cppn, ax)
    return fig


def _make_phenotype_heatmap_figure(phenotype_data, columns):
    """Create a standalone phenotype heatmap figure."""
    fig, ax = plt.subplots(figsize=(8, 6))
    draw_phenotype_heatmap(phenotype_data, ax, columns=columns)
    return fig


# ── Per-genotype figure generation ───────────────────────────────────────────

def _generate_spherical_figures(phenotype, genotype_name, rank, out_dir):
    """Generate figures for a spherical genome individual."""
    fig = _make_drone_figure(phenotype)
    save_figure(fig, out_dir, f"{genotype_name}_rank{rank:02d}_drone")

    fig = _make_phenotype_heatmap_figure(phenotype, SPHERICAL_COLUMNS)
    save_figure(fig, out_dir, f"{genotype_name}_rank{rank:02d}_phenotype_heatmap")


def _generate_cppn_figures(phenotype, genotype, genotype_name, rank, out_dir):
    """Generate figures for a CPPN genome individual."""
    fig = _make_drone_figure(phenotype)
    save_figure(fig, out_dir, f"{genotype_name}_rank{rank:02d}_drone")

    if isinstance(genotype, CPPNNetwork):
        fig = _make_cppn_graph_figure(genotype)
        save_figure(fig, out_dir, f"{genotype_name}_rank{rank:02d}_cppn_graph")


def _generate_hybrid_figures(phenotype, genotype, genotype_name, rank, out_dir):
    """Generate figures for a hybrid genome individual."""
    fig = _make_drone_figure(phenotype)
    save_figure(fig, out_dir, f"{genotype_name}_rank{rank:02d}_drone")

    if isinstance(genotype, HybridGenome):
        fig = _make_phenotype_heatmap_figure(genotype.direct, HYBRID_DIRECT_COLUMNS)
        save_figure(fig, out_dir, f"{genotype_name}_rank{rank:02d}_phenotype_heatmap")

        fig = _make_cppn_graph_figure(genotype.cppn)
        save_figure(fig, out_dir, f"{genotype_name}_rank{rank:02d}_cppn_graph")
    else:
        fig = _make_phenotype_heatmap_figure(phenotype, SPHERICAL_COLUMNS)
        save_figure(fig, out_dir, f"{genotype_name}_rank{rank:02d}_phenotype_heatmap")


# ── Main entry point ────────────────────────────────────────────────────────

def run(base_dir, tasks, genotypes, output_base=None):
    """Generate genome visualization figures for best individuals.

    Args:
        base_dir: root data directory
        tasks: list of task names
        genotypes: list of genotype names
        output_base: unused, kept for interface consistency
    """
    setup_style()

    for task in tasks:
        for genotype_name in genotypes:
            experiment_dir = os.path.join(base_dir, task, genotype_name)
            csv_path = os.path.join(experiment_dir, "best_individual_data.csv")

            if not os.path.exists(csv_path):
                continue

            print_analysis_header(
                f"Genome Visualization: {task} / "
                f"{GENOTYPE_LABELS.get(genotype_name, genotype_name)}"
            )

            df = pd.read_csv(csv_path)
            out_dir = figures_dir(base_dir, task=task, plotter_name=PLOTTER_NAME)

            successful = 0
            for _, row in df.iterrows():
                rank = row["rank"]
                fitness = row["fitness"]
                genome_type = row.get("genome_type", "spherical")

                # Parse phenotype
                try:
                    individual_data = ast.literal_eval(row["individual_data"])
                    phenotype = np.array(individual_data)
                except Exception:
                    continue

                # Load genotype if available
                genotype_path = row.get("genotype_pkl_path", "")
                genotype = _load_genotype(genotype_path) if genotype_path else None

                try:
                    if genome_type == "hybrid_cppn":
                        _generate_hybrid_figures(phenotype, genotype, genotype_name, rank, out_dir)
                    elif genome_type == "cppn":
                        _generate_cppn_figures(phenotype, genotype, genotype_name, rank, out_dir)
                    else:
                        _generate_spherical_figures(phenotype, genotype_name, rank, out_dir)
                    successful += 1
                except Exception as e:
                    print(f"  Error plotting rank {rank}: {e}")

            print(f"  Generated {successful}/{len(df)} individuals")
            print_analysis_footer(out_dir)


if __name__ == "__main__":
    run(BASE_DIR, TASKS, GENOTYPES)
