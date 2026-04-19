"""Plotter for morphological descriptors against fitness.

Produces per-task figures:
  - One scatter plot per numeric metric showing descriptor value vs fitness
    for all genotypes, with optional trend lines.
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from experimentation.config import (
    BASE_DIR, TASKS, GENOTYPES, COLORS, GENOTYPE_LABELS, TASK_LABELS,
    figures_dir,
)
from experimentation.plot_utils import (
    setup_style, save_figure, apply_axis_style,
    print_analysis_header, print_summary_stats,
    print_stat_tests, print_analysis_footer,
)

warnings.filterwarnings("ignore")

PLOTTER_NAME = "morphological_descriptors_against_fitness"

# Columns that are bookkeeping, not plottable metrics
_EXCLUDE_COLS = {"generation", "individual", "run", "fitness", "hover_status", "offspring"}


# ── Data loading ─────────────────────────────────────────────────────────────

def _load_task_data(base_dir, task, genotypes):
    """Load morphological descriptors CSVs for each genotype under a task."""
    data = {}
    for genotype in genotypes:
        csv_path = os.path.join(base_dir, task, genotype,
                                "morphological_descriptors_data.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # Drop rows without valid fitness
            if "fitness" in df.columns:
                df = df.dropna(subset=["fitness"])
            if len(df) > 0:
                data[genotype] = df
    return data


def _discover_metrics(data_by_genotype):
    """Auto-discover numeric metric columns across all genotype DataFrames."""
    metric_set = set()
    for df in data_by_genotype.values():
        for col in df.select_dtypes(include="number").columns:
            if col not in _EXCLUDE_COLS:
                metric_set.add(col)
    return sorted(metric_set)


# ── Per-metric scatter plot ──────────────────────────────────────────────────

def _plot_metric_vs_fitness(metric, data_by_genotype, genotypes, out_dir):
    """One figure: scatter plot of metric vs fitness for all genotypes."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for genotype in genotypes:
        if genotype not in data_by_genotype:
            continue
        df = data_by_genotype[genotype]
        if metric not in df.columns or "fitness" not in df.columns:
            continue

        valid = df[[metric, "fitness"]].dropna()
        if len(valid) == 0:
            continue

        color = COLORS.get(genotype, None)
        label = GENOTYPE_LABELS.get(genotype, genotype)
        ax.scatter(valid[metric], valid["fitness"],
                   color=color, label=label, alpha=0.3, s=10, edgecolors="none")

        # Linear trend line
        if len(valid) >= 2:
            coeffs = np.polyfit(valid[metric], valid["fitness"], 1)
            x_range = np.linspace(valid[metric].min(), valid[metric].max(), 100)
            ax.plot(x_range, np.polyval(coeffs, x_range),
                    color=color, linewidth=2, linestyle="--", alpha=0.8)

    pretty_name = metric.replace("_", " ")
    apply_axis_style(ax, pretty_name, "Fitness")
    ax.legend()
    ax.grid(True, alpha=0.3)

    filename = metric.lower()
    save_figure(fig, out_dir, filename)

    # Analytical output: correlation per genotype
    print_analysis_header(f"{pretty_name} vs Fitness")
    for genotype in genotypes:
        if genotype not in data_by_genotype:
            continue
        df = data_by_genotype[genotype]
        if metric not in df.columns or "fitness" not in df.columns:
            continue
        valid = df[[metric, "fitness"]].dropna()
        if len(valid) < 3:
            continue
        r = np.corrcoef(valid[metric], valid["fitness"])[0, 1]
        label = GENOTYPE_LABELS.get(genotype, genotype)
        print(f"  {label}: Pearson r = {r:+.3f}  (n={len(valid)})")
    print_analysis_footer(os.path.join(out_dir, filename))


# ── Main entry point ────────────────────────────────────────────────────────

def run(base_dir, tasks, genotypes, output_base=None):
    """Run all morphological descriptor vs fitness plots.

    Args:
        base_dir: root data directory
        tasks: list of task names
        genotypes: list of genotype names
        output_base: override for output root (defaults to config.output_dir)
    """
    from experimentation.config import output_dir as _output_dir
    if output_base is None:
        output_base = _output_dir(base_dir)

    setup_style()

    for task in tasks:
        print(f"\n{'=' * 60}")
        print(f"Morphological Descriptors vs Fitness: {TASK_LABELS.get(task, task)}")
        print(f"{'=' * 60}")

        data_by_genotype = _load_task_data(base_dir, task, genotypes)
        if not data_by_genotype:
            continue

        out_dir = figures_dir(base_dir, task=task, plotter_name=PLOTTER_NAME)
        metrics = _discover_metrics(data_by_genotype)

        if not metrics:
            print("  No numeric metric columns found.")
            continue

        for metric in metrics:
            _plot_metric_vs_fitness(metric, data_by_genotype, genotypes, out_dir)


if __name__ == "__main__":
    run(BASE_DIR, TASKS, GENOTYPES)
