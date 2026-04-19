"""Plotter for morphological descriptor data.

Produces per-task figures:
  - One standalone figure per numeric metric showing all genotypes compared
    over generations (mean with 95% CI shading).
"""

import os
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from experimentation.config import (
    BASE_DIR, TASKS, GENOTYPES, COLORS, GENOTYPE_LABELS, TASK_LABELS,
    Y_LIMITS, figures_dir, csv_dir,
)
from experimentation.plot_utils import (
    setup_style, save_figure, apply_axis_style,
    aggregate_by_generation, pad_generations,
    print_analysis_header, print_summary_stats, print_trajectory_samples,
    print_stat_tests, print_analysis_footer,
)

warnings.filterwarnings("ignore")

PLOTTER_NAME = "morphological_descriptors"

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
            data[genotype] = pd.read_csv(csv_path)
    return data


def _discover_metrics(data_by_genotype):
    """Auto-discover numeric metric columns across all genotype DataFrames."""
    metric_set = set()
    for df in data_by_genotype.values():
        for col in df.select_dtypes(include="number").columns:
            if col not in _EXCLUDE_COLS:
                metric_set.add(col)
    return sorted(metric_set)


# ── Per-metric comparison plot ───────────────────────────────────────────────

def _plot_metric_comparison(metric, data_by_genotype, genotypes, out_dir):
    """One figure: metric over generations for all genotypes (mean +/- 95% CI)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    final_vals = {}

    for genotype in genotypes:
        if genotype not in data_by_genotype:
            continue
        df = data_by_genotype[genotype]
        if metric not in df.columns:
            continue

        padded = pad_generations(df)
        gen_stats = (padded
                     .groupby("generation")[metric]
                     .agg(["mean", "std", "count"])
                     .reset_index())
        gen_stats["se"] = gen_stats["std"] / np.sqrt(gen_stats["count"])
        ci = 1.96 * gen_stats["se"]

        color = COLORS.get(genotype, None)
        label = GENOTYPE_LABELS.get(genotype, genotype)
        ax.plot(gen_stats["generation"], gen_stats["mean"],
                label=label, color=color, linewidth=1.5)
        ax.fill_between(gen_stats["generation"],
                        gen_stats["mean"] - ci,
                        gen_stats["mean"] + ci,
                        color=color, alpha=0.2)

        # Collect final-generation values for stats
        final_gen = df["generation"].max()
        final = df[df["generation"] == final_gen][metric].dropna().values
        if len(final) > 0:
            final_vals[genotype] = final

    pretty_name = metric.replace("_", " ")
    apply_axis_style(ax, "Generation", pretty_name)
    ax.legend()
    ax.grid(True, alpha=0.3)

    filename = metric.lower()
    save_figure(fig, out_dir, filename)

    # Analytical output
    if final_vals:
        print_analysis_header(f"{pretty_name} (Final Generation)")
        print_summary_stats(final_vals, pretty_name)

        # Trajectory samples
        traj_data = {}
        for genotype in genotypes:
            if genotype in data_by_genotype and metric in data_by_genotype[genotype].columns:
                traj_data[genotype] = data_by_genotype[genotype][["generation", metric]].rename(
                    columns={metric: "value"})
        if traj_data:
            print_trajectory_samples(traj_data, generation_col="generation",
                                     value_col="value")

        print_stat_tests(final_vals)
        print_analysis_footer(os.path.join(out_dir, filename))


# ── Combined cross-task plot ─────────────────────────────────────────────────

def _plot_combined_metric(metric, all_task_data, tasks, genotypes, out_dir):
    """Combined plot: metric averaged across tasks, one line per genotype."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for genotype in genotypes:
        task_series = []
        for task in tasks:
            df = all_task_data.get((task, genotype))
            if df is None or metric not in df.columns:
                continue
            padded = pad_generations(df)
            gen_mean = padded.groupby("generation")[metric].mean()
            task_series.append(gen_mean)

        if not task_series:
            continue

        combined = pd.concat(task_series, axis=1)
        mean_vals = combined.mean(axis=1)
        se_vals = combined.std(axis=1) / np.sqrt(combined.count(axis=1))
        generations = mean_vals.index

        color = COLORS.get(genotype, None)
        label = GENOTYPE_LABELS.get(genotype, genotype)
        ax.plot(generations, mean_vals.values, color=color, linewidth=1.5,
                label=label)
        ax.fill_between(generations,
                        (mean_vals - 1.96 * se_vals).values,
                        (mean_vals + 1.96 * se_vals).values,
                        color=color, alpha=0.2)

    pretty_name = metric.replace("_", " ")
    apply_axis_style(ax, "Generation", pretty_name)
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_figure(fig, out_dir, f"combined_{metric.lower()}")


# ── Main entry point ────────────────────────────────────────────────────────

def run(base_dir, tasks, genotypes, output_base=None):
    """Run all morphological descriptor plots.

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

    all_task_data = {}  # (task, genotype) -> df for combined plots

    for task in tasks:
        print(f"\n{'=' * 60}")
        print(f"Morphological Descriptors: {TASK_LABELS.get(task, task)}")
        print(f"{'=' * 60}")

        data_by_genotype = _load_task_data(base_dir, task, genotypes)
        if not data_by_genotype:
            continue

        for genotype, df in data_by_genotype.items():
            all_task_data[(task, genotype)] = df

        out_dir = figures_dir(base_dir, task=task, plotter_name=PLOTTER_NAME)
        metrics = _discover_metrics(data_by_genotype)

        if not metrics:
            print("  No numeric metric columns found.")
            continue

        for metric in metrics:
            _plot_metric_comparison(metric, data_by_genotype, genotypes, out_dir)

    # Combined cross-task thrust alignment plot
    if all_task_data:
        cross_dir = figures_dir(base_dir, task="cross_task", plotter_name=PLOTTER_NAME)
        _plot_combined_metric("Thrust_Alignment", all_task_data, tasks, genotypes, cross_dir)


if __name__ == "__main__":
    run(BASE_DIR, TASKS, GENOTYPES)
