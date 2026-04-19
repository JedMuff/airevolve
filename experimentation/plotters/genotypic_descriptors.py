"""Genotypic descriptors plotter.

Plots CPPN network structural metrics (hidden nodes, enabled connections,
longest path) over generations, connection density, and cross-task box plots.
Only cppn and hybrid_cppn genotypes have data.
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
    setup_style, save_figure, apply_axis_style, apply_legend,
    aggregate_by_generation, pad_generations, print_analysis_header,
    print_summary_stats, print_trajectory_samples, print_stat_tests,
    print_analysis_footer,
)

warnings.filterwarnings("ignore")

PLOTTER_NAME = "genotypic_descriptors"

# Only CPPN-based genotypes have genotypic descriptors
_CPPN_GENOTYPES = ["cppn", "hybrid_cppn"]

COMPLEXITY_METRICS = ["n_hidden_nodes", "n_enabled_connections", "longest_path"]
METRIC_LABELS = {
    "n_hidden_nodes": "Hidden Nodes",
    "n_enabled_connections": "Enabled Connections",
    "longest_path": "Longest Path",
}


# ── Data loading ─────────────────────────────────────────────────────────────

def _load_data(base_dir, task, genotypes):
    """Load genotypic descriptor CSVs for CPPN-based genotypes."""
    data = {}
    for genotype in genotypes:
        if genotype not in _CPPN_GENOTYPES:
            continue
        csv_path = os.path.join(base_dir, task, genotype, "genotypic_descriptors_data.csv")
        if os.path.exists(csv_path):
            data[genotype] = pd.read_csv(csv_path)
    return data


def _gen_stats(df, metric):
    """Return (generations, mean, lower_ci, upper_ci) for a metric with generation padding."""
    padded = pad_generations(df)
    stats = (
        padded.groupby("generation")[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    stats["se"] = stats["std"] / np.sqrt(stats["count"])
    ci = 1.96 * stats["se"]
    return stats["generation"], stats["mean"], stats["mean"] - ci, stats["mean"] + ci


# ── Per-task plots (individual figures per metric) ───────────────────────────

def _plot_complexity_metric(task, task_data, metric, out_dir):
    """One standalone figure for a single complexity metric over generations."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for genotype, df in task_data.items():
        if metric not in df.columns:
            continue
        gens, mean, lo, hi = _gen_stats(df, metric)
        color = COLORS[genotype]
        label = GENOTYPE_LABELS[genotype]
        ax.plot(gens, mean, color=color, linewidth=2, label=label)
        ax.fill_between(gens, lo, hi, color=color, alpha=0.2)

    apply_axis_style(ax, xlabel="Generation", ylabel=METRIC_LABELS[metric])
    apply_legend(ax)

    fname = f"{metric}_over_generations"
    fig_path = os.path.join(out_dir, fname)
    save_figure(fig, out_dir, fname)

    # Analytical output
    print_analysis_header(f"{METRIC_LABELS[metric]} Over Generations — {task}")
    final_vals = {}
    traj_data = {}
    for genotype, df in task_data.items():
        if metric not in df.columns:
            continue
        padded = pad_generations(df)
        max_gen = padded["generation"].max()
        final_vals[genotype] = padded[padded["generation"] == max_gen][metric].values
        traj_data[genotype] = padded.rename(columns={metric: "value"})
    print_summary_stats(final_vals, metric_label=METRIC_LABELS[metric])
    print_trajectory_samples(
        {g: d.rename(columns={"value": metric}) for g, d in traj_data.items()},
        value_col=metric,
    )
    print_stat_tests(final_vals)
    print_analysis_footer(f"{fig_path}.{{pdf,png}}")


def _plot_connection_density(task, task_data, out_dir):
    """Standalone figure: connection density (enabled/total) over generations."""
    required = ["n_enabled_connections", "n_total_connections"]
    fig, ax = plt.subplots(figsize=(10, 6))

    has_data = False
    traj_data = {}
    final_vals = {}

    for genotype, df in task_data.items():
        if not all(c in df.columns for c in required):
            continue
        df = df.copy()
        df["connection_density"] = np.where(
            df["n_total_connections"] > 0,
            df["n_enabled_connections"] / df["n_total_connections"],
            np.nan,
        )
        gens, mean, lo, hi = _gen_stats(df, "connection_density")
        ax.plot(gens, mean, color=COLORS[genotype], linewidth=2,
                label=GENOTYPE_LABELS[genotype])
        ax.fill_between(gens, lo, hi, color=COLORS[genotype], alpha=0.2)
        has_data = True

        padded = pad_generations(df)
        max_gen = padded["generation"].max()
        final_vals[genotype] = padded[padded["generation"] == max_gen]["connection_density"].values
        traj_data[genotype] = padded

    if not has_data:
        plt.close(fig)
        return

    apply_axis_style(ax, xlabel="Generation", ylabel="Connection Density (enabled/total)")
    apply_legend(ax)

    fig_path = os.path.join(out_dir, "connection_density_over_generations")
    save_figure(fig, out_dir, "connection_density_over_generations")

    print_analysis_header(f"Connection Density — {task}")
    print_summary_stats(final_vals, metric_label="connection density")
    print_trajectory_samples(traj_data, value_col="connection_density")
    print_stat_tests(final_vals)
    print_analysis_footer(f"{fig_path}.{{pdf,png}}")


# ── Cross-task box plots (individual figures per metric) ─────────────────────

def _plot_cross_task_boxplot(cross_task_data, metric, genotypes, out_dir):
    """One standalone cross-task box plot for a single metric."""
    if not cross_task_data:
        return

    df = pd.DataFrame(cross_task_data)
    if metric not in df.columns:
        return

    cppn_genotypes = [g for g in genotypes if g in _CPPN_GENOTYPES]
    fig, ax = plt.subplots(figsize=(10, 6))

    task_positions = {task: i for i, task in enumerate(TASKS)}
    n_genotypes = len(cppn_genotypes)
    width = 0.35
    offsets = np.linspace(-(n_genotypes - 1) * width / 2,
                          (n_genotypes - 1) * width / 2,
                          n_genotypes)

    final_vals = {}

    for g_idx, genotype in enumerate(cppn_genotypes):
        gdf = df[df["genotype"] == genotype]
        color = COLORS[genotype]

        positions = []
        box_data = []
        for task in TASKS:
            tdf = gdf[gdf["task"] == task]
            if tdf.empty or metric not in tdf.columns:
                continue
            vals = tdf[metric].dropna().values
            if len(vals) == 0:
                continue
            positions.append(task_positions[task] + offsets[g_idx])
            box_data.append(vals)

        if not box_data:
            continue

        bp = ax.boxplot(box_data, positions=positions, widths=width * 0.8,
                        patch_artist=True,
                        medianprops=dict(color="black", linewidth=1.5),
                        flierprops=dict(marker="o", markersize=3, alpha=0.5))
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Legend entry
        ax.plot([], [], color=color, linewidth=6, alpha=0.7,
                label=GENOTYPE_LABELS[genotype])

        all_vals = gdf[metric].dropna().values
        if len(all_vals) > 0:
            final_vals[genotype] = all_vals

    ax.set_xticks(range(len(TASKS)))
    ax.set_xticklabels([TASK_LABELS.get(t, t) for t in TASKS])
    apply_axis_style(ax, xlabel="Task", ylabel=METRIC_LABELS.get(metric, metric))
    apply_legend(ax)

    fname = f"{metric}_cross_task"
    fig_path = os.path.join(out_dir, fname)
    save_figure(fig, out_dir, fname)

    print_analysis_header(f"{METRIC_LABELS.get(metric, metric)} — Cross-Task")
    print_summary_stats(final_vals, metric_label=METRIC_LABELS.get(metric, metric))
    print_stat_tests(final_vals)
    print_analysis_footer(f"{fig_path}.{{pdf,png}}")


# ── Public entry point ───────────────────────────────────────────────────────

def run(base_dir, tasks, genotypes, output_base=None):
    """Generate all genotypic descriptor plots and analytical output.

    Args:
        base_dir: root data directory
        tasks: list of task names
        genotypes: list of genotype names (non-CPPN genotypes are silently skipped)
        output_base: override for output root (defaults to config.output_dir)
    """
    setup_style()

    if output_base is None:
        from experimentation.config import output_dir
        output_base = output_dir(base_dir)

    cross_task_rows = []

    for task in tasks:
        task_data = _load_data(base_dir, task, genotypes)
        if not task_data:
            continue

        out_dir = figures_dir(base_dir, task=task, plotter_name=PLOTTER_NAME)

        # Individual complexity metric plots
        for metric in COMPLEXITY_METRICS:
            _plot_complexity_metric(task, task_data, metric, out_dir)

        # Connection density
        _plot_connection_density(task, task_data, out_dir)

        # Collect final-generation data for cross-task plots
        for genotype, df in task_data.items():
            for run_id in df["run"].unique():
                run_df = df[df["run"] == run_id]
                max_gen = run_df["generation"].max()
                final_df = run_df[run_df["generation"] == max_gen]
                for _, row in final_df.iterrows():
                    entry = {"task": task, "genotype": genotype}
                    for m in COMPLEXITY_METRICS:
                        if m in row.index:
                            entry[m] = row[m]
                    cross_task_rows.append(entry)

    # Cross-task box plots (one per metric)
    cross_out = figures_dir(base_dir, task="cross_task", plotter_name=PLOTTER_NAME)
    for metric in COMPLEXITY_METRICS:
        _plot_cross_task_boxplot(cross_task_rows, metric, genotypes, cross_out)


if __name__ == "__main__":
    from experimentation.config import BASE_DIR, TASKS, GENOTYPES
    run(BASE_DIR, TASKS, GENOTYPES)
