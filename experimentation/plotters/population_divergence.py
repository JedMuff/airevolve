"""Population divergence from generation 0 plotter.

Produces per-task figures:
  - population_divergence_over_generations (all genotypes on one plot)
Cross-task figure:
  - population_divergence_heatmap
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from experimentation.config import (
    BASE_DIR, TASKS, GENOTYPES, COLORS, GENOTYPE_LABELS, TASK_LABELS,
    figures_dir,
)
from experimentation.plot_utils import (
    setup_style, save_figure, apply_axis_style, aggregate_by_generation,
    print_analysis_header, print_summary_stats, print_trajectory_samples,
    print_stat_tests, print_analysis_footer,
)

warnings.filterwarnings("ignore")

PLOTTER_NAME = "population_divergence"


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_data(base_dir, task, genotype):
    path = os.path.join(base_dir, task, genotype, "population_divergence_data.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


# ── Per-task plot ─────────────────────────────────────────────────────────────

def _plot_divergence_over_generations(task, data_dict, genotypes, out_dir):
    """Line plot of population divergence from gen-0 over generations."""
    fig, ax = plt.subplots(figsize=(10, 6))

    final_vals = {}
    traj_data = {}

    for genotype in genotypes:
        df = data_dict.get(genotype)
        if df is None:
            continue
        mean, se = aggregate_by_generation(df, "mean_divergence")
        generations = mean.index
        ax.plot(generations, mean.values, color=COLORS[genotype],
                label=GENOTYPE_LABELS[genotype])
        ax.fill_between(generations,
                        mean.values - 1.96 * se.values,
                        mean.values + 1.96 * se.values,
                        color=COLORS[genotype], alpha=0.2)

        max_gen = df["generation"].max()
        final_vals[genotype] = df[df["generation"] == max_gen]["mean_divergence"].values
        traj_data[genotype] = df

    apply_axis_style(ax, xlabel="Generation",
                     ylabel="Phenotypic Distance from Gen 0")
    ax.legend()
    fig.tight_layout()

    filename = "population_divergence_over_generations"
    save_figure(fig, out_dir, filename)

    print_analysis_header(f"Population Divergence from Gen 0 — {task}")
    print_summary_stats(final_vals, metric_label="divergence from gen 0")
    print_trajectory_samples(traj_data, value_col="mean_divergence")
    print_stat_tests(final_vals)
    print_analysis_footer(os.path.join(out_dir, filename))


# ── Cross-task heatmap ────────────────────────────────────────────────────────

def _plot_divergence_heatmap(heatmap_data, tasks, genotypes, out_dir):
    """Heatmap of final-generation population divergence across tasks and genotypes."""
    tasks_present = [t for t in tasks if t in heatmap_data]
    if not tasks_present:
        return

    matrix = np.full((len(tasks_present), len(genotypes)), np.nan)
    for i, task in enumerate(tasks_present):
        for j, genotype in enumerate(genotypes):
            matrix[i, j] = heatmap_data[task].get(genotype, np.nan)

    fig, ax = plt.subplots(
        figsize=(max(6, 3 * len(genotypes)), max(4, 2 * len(tasks_present))))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")

    for i in range(len(tasks_present)):
        for j in range(len(genotypes)):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=20, fontweight="bold", color="black")

    ax.set_xticks(np.arange(len(genotypes)))
    ax.set_xticklabels([GENOTYPE_LABELS[g] for g in genotypes])
    ax.set_yticks(np.arange(len(tasks_present)))
    ax.set_yticklabels([TASK_LABELS.get(t, t) for t in tasks_present])
    apply_axis_style(ax, xlabel="Genotype", ylabel="Task")

    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label("Divergence from Gen 0", fontsize=25)
    fig.tight_layout()

    filename = "population_divergence_heatmap"
    save_figure(fig, out_dir, filename)

    # Analytical output
    flat_vals = {}
    for j, genotype in enumerate(genotypes):
        col = matrix[:, j]
        col = col[~np.isnan(col)]
        if len(col) > 0:
            flat_vals[genotype] = col

    print_analysis_header("Cross-Task Population Divergence Heatmap")
    print_summary_stats(flat_vals, metric_label="divergence from gen 0")
    print_stat_tests(flat_vals)
    print_analysis_footer(os.path.join(out_dir, filename))


# ── Public entry point ────────────────────────────────────────────────────────

def run(base_dir, tasks, genotypes, output_base=None):
    """Generate all population divergence figures.

    Args:
        base_dir: root data directory
        tasks: list of task names
        genotypes: list of genotype names
        output_base: override for output root (defaults to config.output_dir)
    """
    setup_style()

    out_base = output_base if output_base else base_dir
    heatmap_data = {}

    for task in tasks:
        task_data = {}
        for genotype in genotypes:
            df = _load_data(base_dir, task, genotype)
            if df is not None:
                task_data[genotype] = df

        if not task_data:
            continue

        out_dir = figures_dir(out_base, task=task, plotter_name=PLOTTER_NAME)
        _plot_divergence_over_generations(task, task_data, genotypes, out_dir)

        # Collect final divergence for heatmap
        heatmap_data[task] = {}
        for genotype, df in task_data.items():
            max_gen = df["generation"].max()
            final_rows = df[df["generation"] == max_gen]
            heatmap_data[task][genotype] = final_rows["mean_divergence"].mean()

    # Cross-task heatmap
    if heatmap_data:
        cross_dir = figures_dir(out_base, task="cross_task",
                                plotter_name=PLOTTER_NAME)
        _plot_divergence_heatmap(heatmap_data, tasks, genotypes, cross_dir)


if __name__ == "__main__":
    from experimentation.config import BASE_DIR, TASKS, GENOTYPES
    run(BASE_DIR, TASKS, GENOTYPES)
