"""Plotter for inter-run diversity data.

Produces per-task figures:
  - divergence_ratio_over_generations (all genotypes on one plot)
  - final_divergence_comparison (bar chart)
Cross-task figure:
  - divergence_ratio_heatmap
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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

PLOTTER_NAME = "inter_run_diversity"


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_summary(base_dir, task, genotype):
    """Load inter-run diversity summary CSV."""
    path = os.path.join(base_dir, task, genotype, "inter_run_diversity_summary.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


# ── Divergence ratio over generations (all genotypes) ─────────────────────────

def _plot_divergence_ratio(data, genotypes, out_dir):
    """Line plot of divergence ratio over generations for all genotypes."""
    fig, ax = plt.subplots(figsize=(10, 6))

    traj_data = {}
    final_vals = {}

    for genotype in genotypes:
        df = data.get(genotype)
        if df is None:
            continue
        generations = df["generation"].values
        ratio = df["divergence_ratio"].values

        ax.plot(generations, ratio, color=COLORS[genotype], linewidth=2,
                label=GENOTYPE_LABELS[genotype])

        traj_data[genotype] = pd.DataFrame({
            "generation": generations, "value": ratio,
        })
        max_gen = df["generation"].max()
        final_vals[genotype] = df.loc[
            df["generation"] == max_gen, "divergence_ratio"
        ].dropna().values

    ax.axhline(y=1.0, color="black", linestyle="--", linewidth=1.5)

    apply_axis_style(ax, xlabel="Generation",
                     ylabel="Divergence Ratio")
    ax.legend()
    fig.tight_layout()

    filename = "divergence_ratio_over_generations"
    save_figure(fig, out_dir, filename)

    print_analysis_header("Divergence Ratio Over Generations")
    print_summary_stats(final_vals, metric_label="Final Divergence Ratio")
    print_trajectory_samples(traj_data)
    print_stat_tests(final_vals)
    print_analysis_footer(os.path.join(out_dir, filename))


# ── Final divergence bar chart ────────────────────────────────────────────────

def _plot_final_divergence_comparison(data, genotypes, out_dir):
    """Bar chart of final-generation divergence ratio per genotype."""
    fig, ax = plt.subplots(figsize=(10, 6))

    bar_labels = []
    bar_values = []
    bar_errors = []
    bar_colors = []
    final_vals = {}

    for genotype in genotypes:
        df = data.get(genotype)
        if df is None:
            continue
        max_gen = df["generation"].max()
        final_rows = df[df["generation"] == max_gen]
        mean_ratio = final_rows["divergence_ratio"].mean()
        err = final_rows["divergence_ratio"].std() if len(final_rows) > 1 else 0.0

        bar_labels.append(GENOTYPE_LABELS[genotype])
        bar_values.append(mean_ratio)
        bar_errors.append(err)
        bar_colors.append(COLORS[genotype])
        final_vals[genotype] = final_rows["divergence_ratio"].dropna().values

    x = np.arange(len(bar_labels))
    ax.bar(x, bar_values, yerr=bar_errors, color=bar_colors, capsize=6,
           width=0.5, error_kw={"elinewidth": 2})
    ax.axhline(y=1.0, color="black", linestyle="--", linewidth=1.5)

    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels)
    apply_axis_style(ax, xlabel="Genotype", ylabel="Final Divergence Ratio")
    fig.tight_layout()

    filename = "final_divergence_comparison"
    save_figure(fig, out_dir, filename)

    print_analysis_header("Final Divergence Comparison")
    print_summary_stats(final_vals, metric_label="Final Divergence Ratio")
    print_stat_tests(final_vals)
    print_analysis_footer(os.path.join(out_dir, filename))


# ── Cross-task divergence heatmap ─────────────────────────────────────────────

def _plot_divergence_heatmap(heatmap_data, tasks, genotypes, out_dir):
    """Heatmap of final divergence ratio across tasks and genotypes."""
    tasks_present = [t for t in tasks if t in heatmap_data]
    if not tasks_present:
        return

    matrix = np.full((len(tasks_present), len(genotypes)), np.nan)
    for i, task in enumerate(tasks_present):
        for j, genotype in enumerate(genotypes):
            matrix[i, j] = heatmap_data[task].get(genotype, np.nan)

    # Diverging colormap centred at 1
    vmin = np.nanmin(matrix)
    vmax = np.nanmax(matrix)
    half_range = max(abs(vmax - 1.0), abs(1.0 - vmin), 0.1)
    norm = mcolors.TwoSlopeNorm(
        vmin=1.0 - half_range, vcenter=1.0, vmax=1.0 + half_range)

    fig, ax = plt.subplots(
        figsize=(max(6, 3 * len(genotypes)), max(4, 2 * len(tasks_present))))
    im = ax.imshow(matrix, cmap="RdYlBu_r", norm=norm, aspect="auto")

    for i in range(len(tasks_present)):
        for j in range(len(genotypes)):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=20, fontweight="bold", color="black")

    ax.set_xticks(np.arange(len(genotypes)))
    ax.set_xticklabels([GENOTYPE_LABELS[g] for g in genotypes])
    ax.set_yticks(np.arange(len(tasks_present)))
    ax.set_yticklabels([TASK_LABELS.get(t, t) for t in tasks_present])
    apply_axis_style(ax, xlabel="Genotype", ylabel="Task")

    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label("Divergence Ratio", fontsize=25)
    fig.tight_layout()

    filename = "divergence_ratio_heatmap"
    save_figure(fig, out_dir, filename)

    # Analytical output: flatten heatmap values per genotype
    flat_vals = {}
    for j, genotype in enumerate(genotypes):
        col = matrix[:, j]
        col = col[~np.isnan(col)]
        if len(col) > 0:
            flat_vals[genotype] = col

    print_analysis_header("Cross-Task Divergence Ratio Heatmap")
    print_summary_stats(flat_vals, metric_label="Divergence Ratio")
    print_stat_tests(flat_vals)
    print_analysis_footer(os.path.join(out_dir, filename))


# ── Combined cross-task divergence ratio ──────────────────────────────────────

def _plot_combined_divergence_ratio(all_task_data, tasks, genotypes, out_dir):
    """Combined plot: divergence ratio averaged across tasks, one line per genotype."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for genotype in genotypes:
        # Collect per-task ratio series for this genotype
        task_series = []
        for task in tasks:
            df = all_task_data.get((task, genotype))
            if df is None:
                continue
            series = df.set_index("generation")["divergence_ratio"]
            task_series.append(series)

        if not task_series:
            continue

        # Align on common generations and compute mean +/- SE across tasks
        combined = pd.concat(task_series, axis=1)
        mean_vals = combined.mean(axis=1)
        se_vals = combined.std(axis=1) / np.sqrt(combined.count(axis=1))
        generations = mean_vals.index

        ax.plot(generations, mean_vals.values, color=COLORS[genotype], linewidth=2,
                label=GENOTYPE_LABELS[genotype])
        ax.fill_between(generations,
                        (mean_vals - 1.96 * se_vals).values,
                        (mean_vals + 1.96 * se_vals).values,
                        color=COLORS[genotype], alpha=0.2)

    ax.axhline(y=1.0, color="black", linestyle="--", linewidth=1.5)

    apply_axis_style(ax, xlabel="Generation", ylabel="Divergence Ratio")
    if "combined_divergence_ratio" in Y_LIMITS:
        ax.set_ylim(0, Y_LIMITS["combined_divergence_ratio"])
    ax.legend()
    fig.tight_layout()

    save_figure(fig, out_dir, "combined_divergence_ratio")


# ── Public entry point ────────────────────────────────────────────────────────

def run(base_dir, tasks, genotypes, output_base=None):
    """Generate all inter-run diversity figures.

    Args:
        base_dir: root data directory
        tasks: list of task names
        genotypes: list of genotype names
        output_base: override for output root (defaults to config.output_dir)
    """
    setup_style()

    out_base = output_base if output_base else base_dir
    heatmap_data = {}
    all_task_data = {}  # (task, genotype) -> df for combined plot

    for task in tasks:
        task_data = {}
        for genotype in genotypes:
            df = _load_summary(base_dir, task, genotype)
            if df is not None:
                task_data[genotype] = df
                all_task_data[(task, genotype)] = df

        if not task_data:
            continue

        out_dir = figures_dir(out_base, task=task, plotter_name=PLOTTER_NAME)

        _plot_divergence_ratio(task_data, genotypes, out_dir)
        _plot_final_divergence_comparison(task_data, genotypes, out_dir)

        # Collect final divergence ratios for heatmap
        heatmap_data[task] = {}
        for genotype, df in task_data.items():
            max_gen = df["generation"].max()
            final_rows = df[df["generation"] == max_gen]
            heatmap_data[task][genotype] = final_rows["divergence_ratio"].mean()

    # Cross-task plots
    if heatmap_data:
        cross_dir = figures_dir(out_base, task="cross_task",
                                plotter_name=PLOTTER_NAME)
        _plot_divergence_heatmap(heatmap_data, tasks, genotypes, cross_dir)
        _plot_combined_divergence_ratio(all_task_data, tasks, genotypes, cross_dir)


if __name__ == "__main__":
    from experimentation.config import BASE_DIR as _BASE_DIR, TASKS as _TASKS, GENOTYPES as _GENOTYPES
    run(_BASE_DIR, _TASKS, _GENOTYPES)
