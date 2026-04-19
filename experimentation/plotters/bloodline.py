"""Bloodline survival and divergence plotter.

Plots surviving bloodlines over generations and phenotype divergence from
founders, per task.  Cross-task summary tables are printed as analytical output.
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from experimentation.config import (
    BASE_DIR, TASKS, GENOTYPES, COLORS, GENOTYPE_LABELS, TASK_LABELS,
    Y_LIMITS, figures_dir, csv_dir,
)
from experimentation.plot_utils import (
    setup_style, save_figure, apply_axis_style, aggregate_by_generation,
    pad_generations, print_analysis_header, print_summary_stats,
    print_trajectory_samples, print_stat_tests, print_analysis_footer,
)

warnings.filterwarnings("ignore")

PLOTTER_NAME = "bloodline"


# ── Data loading ─────────────────────────────────────────────────────────────

def _load_bloodline_data(base_dir, task, genotype):
    path = os.path.join(base_dir, task, genotype, "bloodline_data.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def _load_divergence_data(base_dir, task, genotype):
    path = os.path.join(base_dir, task, genotype, "bloodline_divergence_data.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def _load_genotypic_divergence_data(base_dir, task, genotype):
    path = os.path.join(base_dir, task, genotype, "bloodline_genotypic_divergence_data.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


# ── Per-task plots ───────────────────────────────────────────────────────────

def _plot_bloodlines_over_generations(task, data_dict, genotypes, out_dir):
    """One standalone figure: surviving bloodlines over generations."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for genotype in genotypes:
        df = data_dict.get(genotype)
        if df is None:
            continue
        mean, se = aggregate_by_generation(df, "n_bloodlines")
        generations = mean.index
        ax.plot(generations, mean.values, color=COLORS[genotype],
                label=GENOTYPE_LABELS[genotype])
        ax.fill_between(generations,
                        mean.values - 1.96 * se.values,
                        mean.values + 1.96 * se.values,
                        color=COLORS[genotype], alpha=0.2)

    apply_axis_style(ax, xlabel="Generation", ylabel="Surviving Bloodlines")
    ax.legend()

    fig_path = os.path.join(out_dir, "bloodlines_over_generations")
    save_figure(fig, out_dir, "bloodlines_over_generations")

    # Analytical output
    print_analysis_header(f"Bloodline Survival — {task}")
    final_vals = {}
    traj_data = {}
    for genotype in genotypes:
        df = data_dict.get(genotype)
        if df is None:
            continue
        max_gen = df["generation"].max()
        final_vals[genotype] = df[df["generation"] == max_gen]["n_bloodlines"].values
        traj_data[genotype] = df
    print_summary_stats(final_vals, metric_label="surviving bloodlines")
    print_trajectory_samples(traj_data, value_col="n_bloodlines")
    print_stat_tests(final_vals)
    print_analysis_footer(f"{fig_path}.{{pdf,png}}")


def _plot_divergence_over_generations(task, div_data_dict, genotypes, out_dir):
    """One standalone figure: phenotype divergence from founders over generations."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for genotype in genotypes:
        df = div_data_dict.get(genotype)
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

    apply_axis_style(ax, xlabel="Generation", ylabel="Phenotypic Divergence")
    if "bloodline_phenotypic_divergence" in Y_LIMITS:
        ax.set_ylim(0, Y_LIMITS["bloodline_phenotypic_divergence"])
    ax.legend()

    fig_path = os.path.join(out_dir, "bloodline_divergence_over_generations")
    save_figure(fig, out_dir, "bloodline_divergence_over_generations")

    # Analytical output
    print_analysis_header(f"Bloodline Divergence — {task}")
    final_vals = {}
    traj_data = {}
    for genotype in genotypes:
        df = div_data_dict.get(genotype)
        if df is None:
            continue
        max_gen = df["generation"].max()
        final_vals[genotype] = df[df["generation"] == max_gen]["mean_divergence"].values
        traj_data[genotype] = df
    print_summary_stats(final_vals, metric_label="divergence from founder")
    print_trajectory_samples(traj_data, value_col="mean_divergence")
    print_stat_tests(final_vals)
    print_analysis_footer(f"{fig_path}.{{pdf,png}}")


def _plot_genotypic_divergence_over_generations(task, geno_div_data_dict, genotypes, out_dir):
    """One standalone figure: genotypic divergence from founders over generations."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for genotype in genotypes:
        df = geno_div_data_dict.get(genotype)
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

    apply_axis_style(ax, xlabel="Generation", ylabel="Genotypic Divergence")
    if "bloodline_genotypic_divergence" in Y_LIMITS:
        ax.set_ylim(0, Y_LIMITS["bloodline_genotypic_divergence"])
    ax.legend()

    fig_path = os.path.join(out_dir, "bloodline_genotypic_divergence_over_generations")
    save_figure(fig, out_dir, "bloodline_genotypic_divergence_over_generations")

    # Analytical output
    print_analysis_header(f"Bloodline Genotypic Divergence — {task}")
    final_vals = {}
    traj_data = {}
    for genotype in genotypes:
        df = geno_div_data_dict.get(genotype)
        if df is None:
            continue
        max_gen = df["generation"].max()
        final_vals[genotype] = df[df["generation"] == max_gen]["mean_divergence"].values
        traj_data[genotype] = df
    print_summary_stats(final_vals, metric_label="genotypic divergence from founder")
    print_trajectory_samples(traj_data, value_col="mean_divergence")
    print_stat_tests(final_vals)
    print_analysis_footer(f"{fig_path}.{{pdf,png}}")


# ── Combined cross-task bloodline divergence ─────────────────────────────────

def _plot_combined_bloodline_divergence(div_summary, tasks, genotypes, out_dir):
    """Combined plot: bloodline divergence averaged across tasks, one line per genotype."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for genotype in genotypes:
        task_series = []
        for task in tasks:
            df = div_summary.get((task, genotype))
            if df is None:
                continue
            mean, _ = aggregate_by_generation(df, "mean_divergence")
            task_series.append(mean)

        if not task_series:
            continue

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

    apply_axis_style(ax, xlabel="Generation", ylabel="Phenotypic Divergence")
    if "bloodline_phenotypic_divergence" in Y_LIMITS:
        ax.set_ylim(0, Y_LIMITS["bloodline_phenotypic_divergence"])
    ax.legend()
    fig.tight_layout()

    save_figure(fig, out_dir, "combined_bloodline_divergence")


# ── Cross-task summary tables ────────────────────────────────────────────────

def _print_cross_task_bloodline_table(summary, tasks, genotypes):
    """Print formatted cross-task summary table of final-generation bloodlines."""
    print_analysis_header("Cross-Task Bloodline Counts (final generation)")
    table = {}
    for task in tasks:
        table[task] = {}
        for genotype in genotypes:
            df = summary.get((task, genotype))
            if df is None:
                table[task][genotype] = None
                continue
            max_gen = df["generation"].max()
            table[task][genotype] = df[df["generation"] == max_gen]["n_bloodlines"].values

    header = f"{'Task':<15}" + "".join(f"{GENOTYPE_LABELS[g]:>18}" for g in genotypes)
    print(header)
    print("-" * len(header))
    for task in tasks:
        row = f"{TASK_LABELS.get(task, task):<15}"
        for genotype in genotypes:
            vals = table[task][genotype]
            if vals is None or len(vals) == 0:
                row += f"{'---':>18}"
            else:
                row += f"{np.mean(vals):>8.1f} ({np.std(vals):>4.1f})  "
        print(row)
    print_analysis_footer("(table only)")


def _print_cross_task_divergence_table(div_summary, tasks, genotypes, title=None):
    """Print formatted cross-task table of final-generation divergence."""
    print_analysis_header(title or "Cross-Task Divergence from Founders (final generation)")
    table = {}
    for task in tasks:
        table[task] = {}
        for genotype in genotypes:
            df = div_summary.get((task, genotype))
            if df is None:
                table[task][genotype] = None
                continue
            max_gen = df["generation"].max()
            table[task][genotype] = df[df["generation"] == max_gen]["mean_divergence"].values

    header = f"{'Task':<15}" + "".join(f"{GENOTYPE_LABELS[g]:>18}" for g in genotypes)
    print(header)
    print("-" * len(header))
    for task in tasks:
        row = f"{TASK_LABELS.get(task, task):<15}"
        for genotype in genotypes:
            vals = table[task][genotype]
            if vals is None or len(vals) == 0:
                row += f"{'---':>18}"
            else:
                row += f"{np.mean(vals):>8.3f} ({np.std(vals):>5.3f})  "
        print(row)
    print_analysis_footer("(table only)")


# ── Public entry point ───────────────────────────────────────────────────────

def run(base_dir, tasks, genotypes, output_base=None):
    """Generate all bloodline plots and analytical output.

    Args:
        base_dir: root data directory
        tasks: list of task names
        genotypes: list of genotype names
        output_base: override for output root (defaults to config.output_dir)
    """
    setup_style()

    if output_base is None:
        from experimentation.config import output_dir
        output_base = output_dir(base_dir)

    summary = {}
    div_summary = {}
    geno_div_summary = {}

    for task in tasks:
        data_dict = {}
        div_data_dict = {}
        geno_div_data_dict = {}

        for genotype in genotypes:
            df = _load_bloodline_data(base_dir, task, genotype)
            data_dict[genotype] = df
            summary[(task, genotype)] = df

            div_df = _load_divergence_data(base_dir, task, genotype)
            div_data_dict[genotype] = div_df
            div_summary[(task, genotype)] = div_df

            geno_div_df = _load_genotypic_divergence_data(base_dir, task, genotype)
            geno_div_data_dict[genotype] = geno_div_df
            geno_div_summary[(task, genotype)] = geno_div_df

        out_dir = figures_dir(base_dir, task=task, plotter_name=PLOTTER_NAME)

        if any(v is not None for v in data_dict.values()):
            _plot_bloodlines_over_generations(task, data_dict, genotypes, out_dir)

        if any(v is not None for v in div_data_dict.values()):
            _plot_divergence_over_generations(task, div_data_dict, genotypes, out_dir)

        if any(v is not None for v in geno_div_data_dict.values()):
            _plot_genotypic_divergence_over_generations(
                task, geno_div_data_dict, genotypes, out_dir)

    # Cross-task summary tables
    _print_cross_task_bloodline_table(summary, tasks, genotypes)
    _print_cross_task_divergence_table(div_summary, tasks, genotypes)
    _print_cross_task_divergence_table(geno_div_summary, tasks, genotypes,
                                       title="Cross-Task Genotypic Divergence from Founders (final generation)")

    # Combined cross-task bloodline divergence plot
    if any(v is not None for v in div_summary.values()):
        cross_dir = figures_dir(base_dir, task="cross_task",
                                plotter_name=PLOTTER_NAME)
        _plot_combined_bloodline_divergence(div_summary, tasks, genotypes, cross_dir)


if __name__ == "__main__":
    from experimentation.config import BASE_DIR, TASKS, GENOTYPES
    run(BASE_DIR, TASKS, GENOTYPES)
