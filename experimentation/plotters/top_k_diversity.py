"""Plotter for top-K cross-run diversity data.

Produces:
  - Cross-task: top_k_diversity_comparison (grouped bar chart, all genotypes)
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from experimentation.config import (
    BASE_DIR, TASKS, GENOTYPES, COLORS, GENOTYPE_LABELS, TASK_LABELS,
    figures_dir, csv_dir,
)
from experimentation.plot_utils import (
    setup_style, save_figure, apply_axis_style,
    aggregate_by_generation, pad_generations,
    print_analysis_header, print_summary_stats, print_trajectory_samples,
    print_stat_tests, print_analysis_footer,
)

warnings.filterwarnings("ignore")

PLOTTER_NAME = "top_k_diversity"


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_summary_csv(base_dir):
    """Load the cross-task summary CSV."""
    path = os.path.join(base_dir, "top_k_cross_run_diversity_summary.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


# ── Cross-task grouped bar chart ──────────────────────────────────────────────

def _plot_diversity_comparison(summary_df, tasks, genotypes, out_dir):
    """Grouped bar chart of mean phenotypic distance by genotype, grouped by task."""
    task_order = [t for t in tasks if t in summary_df["task"].values]
    n_tasks = len(task_order)
    n_geno = len(genotypes)
    bar_width = 0.25
    x = np.arange(n_tasks)

    fig, ax = plt.subplots(figsize=(10, 6))

    final_vals = {}

    for g_idx, genotype in enumerate(genotypes):
        means = []
        stds = []
        geno_vals = []
        for task in task_order:
            subset = summary_df[
                (summary_df["task"] == task) & (summary_df["genotype"] == genotype)
            ]
            if len(subset) == 0:
                means.append(np.nan)
                stds.append(0.0)
            else:
                val = subset["mean_phenotypic_distance"].values[0]
                means.append(val)
                stds.append(subset["std_phenotypic_distance"].values[0])
                geno_vals.append(val)

        offsets = x + g_idx * bar_width - bar_width * (n_geno - 1) / 2
        ax.bar(
            offsets, means, bar_width, yerr=stds,
            label=GENOTYPE_LABELS[genotype], color=COLORS[genotype],
            capsize=5, alpha=0.85, error_kw={"elinewidth": 1.5},
        )
        if geno_vals:
            final_vals[genotype] = np.array(geno_vals)

    ax.set_xticks(x)
    ax.set_xticklabels([TASK_LABELS.get(t, t.capitalize()) for t in task_order])
    apply_axis_style(ax, xlabel="Task", ylabel="Mean Phenotypic Distance")
    ax.legend()
    fig.tight_layout()

    filename = "top_k_diversity_comparison"
    save_figure(fig, out_dir, filename)

    print_analysis_header("Top-K Cross-Run Diversity Comparison")
    print_summary_stats(final_vals, metric_label="Mean Phenotypic Distance")
    print_stat_tests(final_vals)
    print_analysis_footer(os.path.join(out_dir, filename))


# ── Public entry point ────────────────────────────────────────────────────────

def run(base_dir, tasks, genotypes, output_base=None):
    """Generate all top-K diversity figures.

    Args:
        base_dir: root data directory
        tasks: list of task names
        genotypes: list of genotype names
        output_base: override for output root (defaults to config.output_dir)
    """
    setup_style()

    out_base = output_base if output_base else base_dir

    summary_df = _load_summary_csv(base_dir)
    if summary_df is not None:
        cross_dir = figures_dir(out_base, task="cross_task",
                                plotter_name=PLOTTER_NAME)
        _plot_diversity_comparison(summary_df, tasks, genotypes, cross_dir)


if __name__ == "__main__":
    from experimentation.config import BASE_DIR as _BASE_DIR, TASKS as _TASKS, GENOTYPES as _GENOTYPES
    run(_BASE_DIR, _TASKS, _GENOTYPES)
