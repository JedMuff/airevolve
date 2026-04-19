"""Plotter for CMA-ES learning curve descriptors.

Produces per-task figures:
  - learning_delta, learning_stability, convergence_point, asymptotic_performance
    (one standalone figure each, split from the original 2x2 grid)
  - stage_iteration_counts, stage_fitness_improvements (split from stage comparison)
  - convergence_distribution (box plot)
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

PLOTTER_NAME = "learning_descriptors"


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_data(base_dir, task, genotypes):
    """Load learning descriptors CSV for each genotype under a task."""
    data = {}
    for genotype in genotypes:
        csv_path = os.path.join(base_dir, task, genotype, "learning_descriptors_data.csv")
        if os.path.exists(csv_path):
            data[genotype] = pd.read_csv(csv_path)
    return data


# ── Individual metric figures ─────────────────────────────────────────────────

_METRICS = [
    ("learning_delta", "Learning Delta"),
    ("learning_stability", "Learning Stability"),
    ("convergence_point", "Convergence Point"),
    ("asymptotic_performance", "Asymptotic Performance"),
]


def _plot_metric_over_generations(data, metric, metric_label, genotypes, out_dir):
    """Single standalone figure for one learning metric over generations."""
    fig, ax = plt.subplots(figsize=(10, 6))

    traj_data = {}
    final_vals = {}

    for genotype in genotypes:
        if genotype not in data:
            continue
        df = pad_generations(data[genotype].copy())
        if metric not in df.columns:
            continue

        mean, se = aggregate_by_generation(df, metric)
        generations = mean.index

        color = COLORS[genotype]
        label = GENOTYPE_LABELS[genotype]
        ax.plot(generations, mean, color=color, label=label, linewidth=2)
        ax.fill_between(generations, mean - 1.96 * se, mean + 1.96 * se,
                        color=color, alpha=0.2)

        traj_data[genotype] = df[["generation", metric]].rename(
            columns={metric: "value"})
        max_gen = df["generation"].max()
        final_vals[genotype] = df.loc[df["generation"] == max_gen, metric].dropna().values

    apply_axis_style(ax, xlabel="Generation", ylabel=metric_label)
    ax.legend()
    fig.tight_layout()

    filename = metric
    save_figure(fig, out_dir, filename)

    fig_path = os.path.join(out_dir, filename)
    print_analysis_header(f"{metric_label} Over Generations")
    print_summary_stats(final_vals, metric_label=metric_label)
    print_trajectory_samples(traj_data)
    print_stat_tests(final_vals)
    print_analysis_footer(fig_path)


# ── Stage comparison (split into two individual figures) ──────────────────────

def _plot_stage_iterations(data, genotypes, out_dir):
    """Grouped bar chart: stage 1 vs stage 2 iteration counts."""
    x = np.arange(len(genotypes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    stage1_means, stage1_ses = [], []
    stage2_means, stage2_ses = [], []
    final_vals = {}

    for genotype in genotypes:
        if genotype in data and "n_stage1_iters" in data[genotype].columns:
            s1 = data[genotype]["n_stage1_iters"].dropna()
            s2 = data[genotype]["n_stage2_iters"].dropna()
            stage1_means.append(s1.mean())
            stage1_ses.append(s1.sem())
            stage2_means.append(s2.mean())
            stage2_ses.append(s2.sem())
            final_vals[genotype] = s1.values
        else:
            stage1_means.append(np.nan)
            stage1_ses.append(0)
            stage2_means.append(np.nan)
            stage2_ses.append(0)

    ax.bar(x - width / 2, stage1_means, width, yerr=stage1_ses,
           label="Stage 1", color="#4477AA", capsize=4)
    ax.bar(x + width / 2, stage2_means, width, yerr=stage2_ses,
           label="Stage 2", color="#EE6677", capsize=4)

    ax.set_xticks(x)
    ax.set_xticklabels([GENOTYPE_LABELS[g] for g in genotypes])
    apply_axis_style(ax, xlabel="Genotype", ylabel="Mean Iterations")
    ax.legend()
    fig.tight_layout()

    filename = "stage_iteration_counts"
    save_figure(fig, out_dir, filename)

    print_analysis_header("Stage Iteration Counts")
    print_summary_stats(final_vals, metric_label="Stage 1 Iterations")
    print_stat_tests(final_vals)
    print_analysis_footer(os.path.join(out_dir, filename))


def _plot_stage_improvements(data, genotypes, out_dir):
    """Grouped bar chart: stage 1 vs stage 2 fitness improvements."""
    x = np.arange(len(genotypes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    imp1_means, imp1_ses = [], []
    imp2_means, imp2_ses = [], []
    final_vals = {}

    for genotype in genotypes:
        if genotype in data and "stage1_improvement" in data[genotype].columns:
            i1 = data[genotype]["stage1_improvement"].dropna()
            i2 = data[genotype]["stage2_improvement"].dropna()
            imp1_means.append(i1.mean())
            imp1_ses.append(i1.sem())
            imp2_means.append(i2.mean())
            imp2_ses.append(i2.sem())
            final_vals[genotype] = i1.values
        else:
            imp1_means.append(np.nan)
            imp1_ses.append(0)
            imp2_means.append(np.nan)
            imp2_ses.append(0)

    ax.bar(x - width / 2, imp1_means, width, yerr=imp1_ses,
           label="Stage 1", color="#4477AA", capsize=4)
    ax.bar(x + width / 2, imp2_means, width, yerr=imp2_ses,
           label="Stage 2", color="#EE6677", capsize=4)

    ax.set_xticks(x)
    ax.set_xticklabels([GENOTYPE_LABELS[g] for g in genotypes])
    apply_axis_style(ax, xlabel="Genotype", ylabel="Mean Improvement")
    ax.legend()
    fig.tight_layout()

    filename = "stage_fitness_improvements"
    save_figure(fig, out_dir, filename)

    print_analysis_header("Stage Fitness Improvements")
    print_summary_stats(final_vals, metric_label="Stage 1 Improvement")
    print_stat_tests(final_vals)
    print_analysis_footer(os.path.join(out_dir, filename))


# ── Convergence distribution ─────────────────────────────────────────────────

def _plot_convergence_distribution(data, genotypes, out_dir):
    """Box plot of convergence_point by genotype."""
    fig, ax = plt.subplots(figsize=(10, 6))

    plot_data = []
    positions = []
    tick_labels = []
    final_vals = {}

    for geno_idx, genotype in enumerate(genotypes):
        if genotype not in data or "convergence_point" not in data[genotype].columns:
            continue
        values = data[genotype]["convergence_point"].dropna().values
        if len(values) == 0:
            continue
        plot_data.append(values)
        positions.append(geno_idx)
        tick_labels.append(GENOTYPE_LABELS[genotype])
        final_vals[genotype] = values

    if not plot_data:
        plt.close(fig)
        return

    bp = ax.boxplot(plot_data, positions=positions, patch_artist=True, widths=0.5)
    for patch, pos in zip(bp["boxes"], positions):
        patch.set_facecolor(COLORS[genotypes[pos]])
        patch.set_alpha(0.7)

    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels)
    apply_axis_style(ax, xlabel="Genotype", ylabel="Convergence Point")
    fig.tight_layout()

    filename = "convergence_distribution"
    save_figure(fig, out_dir, filename)

    print_analysis_header("Convergence Point Distribution")
    print_summary_stats(final_vals, metric_label="Convergence Point")
    print_stat_tests(final_vals)
    print_analysis_footer(os.path.join(out_dir, filename))


# ── Public entry point ────────────────────────────────────────────────────────

def run(base_dir, tasks, genotypes, output_base=None):
    """Generate all learning-descriptor figures.

    Args:
        base_dir: root data directory
        tasks: list of task names
        genotypes: list of genotype names
        output_base: override for output root (defaults to config.output_dir)
    """
    setup_style()

    for task in tasks:
        data = _load_data(base_dir, task, genotypes)
        if not data:
            continue

        out_dir = figures_dir(
            output_base if output_base else base_dir,
            task=task, plotter_name=PLOTTER_NAME,
        )

        # Four individual metric-over-generation figures
        for metric, metric_label in _METRICS:
            _plot_metric_over_generations(data, metric, metric_label, genotypes, out_dir)

        # Stage comparison (two separate figures)
        _plot_stage_iterations(data, genotypes, out_dir)
        _plot_stage_improvements(data, genotypes, out_dir)

        # Convergence distribution
        _plot_convergence_distribution(data, genotypes, out_dir)


if __name__ == "__main__":
    from experimentation.config import BASE_DIR as _BASE_DIR, TASKS as _TASKS, GENOTYPES as _GENOTYPES
    run(_BASE_DIR, _TASKS, _GENOTYPES)
