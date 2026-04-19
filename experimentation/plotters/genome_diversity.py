"""Genome diversity plotter.

Plots genomic (and phenotypic) diversity over generations per task,
genotypic-vs-phenotypic scatter per genotype, and cross-task box plots.
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from experimentation.config import (
    BASE_DIR, TASKS, GENOTYPES, COLORS, GENOTYPE_LABELS, TASK_LABELS,
    Y_LIMITS, figures_dir, csv_dir,
)
from experimentation.plot_utils import (
    setup_style, save_figure, apply_axis_style, apply_legend,
    aggregate_by_generation, pad_generations, print_analysis_header,
    print_summary_stats, print_trajectory_samples, print_stat_tests,
    print_analysis_footer,
)

warnings.filterwarnings("ignore")

PLOTTER_NAME = "genome_diversity"


# ── Data loading ─────────────────────────────────────────────────────────────

def _load_genome_diversity(base_dir, task, genotype):
    path = os.path.join(base_dir, task, genotype, "genome_diversity_data.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def _load_phenotypic_diversity(base_dir, task, genotype):
    path = os.path.join(base_dir, task, genotype, "fitness_diversity_data.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if "phenotypic_diversity" in df.columns:
        df = df.rename(columns={"phenotypic_diversity": "diversity_col"})
    elif "diversity" in df.columns:
        df = df.rename(columns={"diversity": "diversity_col"})
    else:
        return None
    return df


# ── Per-task plots ───────────────────────────────────────────────────────────

def _plot_genome_diversity_over_generations(task, genome_data_dict, pheno_data_dict,
                                            genotypes, out_dir):
    """Standalone figure: genomic diversity (solid) with optional phenotypic overlay (dashed)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for genotype in genotypes:
        gdf = genome_data_dict.get(genotype)
        if gdf is None:
            continue
        color = COLORS[genotype]
        label = GENOTYPE_LABELS[genotype]

        gen_mean, gen_se = aggregate_by_generation(gdf, "genome_diversity_mean")
        generations = gen_mean.index
        ax.plot(generations, gen_mean.values, color=color, linestyle="-",
                label=label)
        ax.fill_between(generations,
                        gen_mean.values - 1.96 * gen_se.values,
                        gen_mean.values + 1.96 * gen_se.values,
                        color=color, alpha=0.2)

        pdf = pheno_data_dict.get(genotype)
        if pdf is not None:
            pheno_mean, pheno_se = aggregate_by_generation(pdf, "diversity_col")
            pheno_gens = pheno_mean.index
            ax.plot(pheno_gens, pheno_mean.values, color=color, linestyle="--",
                    label="_nolegend_")
            ax.fill_between(pheno_gens,
                            pheno_mean.values - 1.96 * pheno_se.values,
                            pheno_mean.values + 1.96 * pheno_se.values,
                            color=color, alpha=0.1)

    apply_axis_style(ax, xlabel="Generation", ylabel="Diversity")
    if "genome_diversity" in Y_LIMITS:
        ax.set_ylim(0, Y_LIMITS["genome_diversity"])
    apply_legend(ax, loc="upper left")

    fig_path = os.path.join(out_dir, "genome_diversity_over_generations")
    save_figure(fig, out_dir, "genome_diversity_over_generations")

    # Analytical output
    print_analysis_header(f"Genome Diversity Over Generations — {task}")
    final_vals = {}
    traj_data = {}
    for genotype in genotypes:
        gdf = genome_data_dict.get(genotype)
        if gdf is None:
            continue
        max_gen = gdf["generation"].max()
        final_vals[genotype] = gdf[gdf["generation"] == max_gen]["genome_diversity_mean"].values
        traj_data[genotype] = gdf
    print_summary_stats(final_vals, metric_label="genome diversity")
    print_trajectory_samples(traj_data, value_col="genome_diversity_mean")
    print_stat_tests(final_vals)
    print_analysis_footer(f"{fig_path}.{{pdf,png}}")


def _plot_genotypic_vs_phenotypic_diversity(task, genome_data_dict, pheno_data_dict,
                                            genotypes, out_dir):
    """One standalone scatter figure per genotype: genomic vs phenotypic diversity."""
    for genotype in genotypes:
        gdf = genome_data_dict.get(genotype)
        pdf = pheno_data_dict.get(genotype)
        if gdf is None or pdf is None:
            continue

        gen_mean, _ = aggregate_by_generation(gdf, "genome_diversity_mean")
        pheno_mean, _ = aggregate_by_generation(pdf, "diversity_col")
        common_gens = gen_mean.index.intersection(pheno_mean.index)
        if len(common_gens) < 2:
            continue

        x = gen_mean.loc[common_gens].values
        y = pheno_mean.loc[common_gens].values

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r, pval = pearsonr(x, y)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(x, y, color=COLORS[genotype], alpha=0.7)
        ax.annotate(f"r={r:.2f}  p={pval:.1e}", xy=(0.03, 0.95),
                    xycoords="axes fraction", fontsize=30, va="top",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

        label = GENOTYPE_LABELS[genotype]
        apply_axis_style(ax, xlabel="Genomic Diversity", ylabel="Phenotypic Diversity")
        apply_legend(ax, [label])

        fname = f"genotypic_vs_phenotypic_{genotype}"
        fig_path = os.path.join(out_dir, fname)
        save_figure(fig, out_dir, fname)

        print_analysis_header(f"Genotypic vs Phenotypic Diversity — {task} — {label}")
        print(f"  Pearson r = {r:.4f}, p = {pval:.4e}")
        print_analysis_footer(f"{fig_path}.{{pdf,png}}")


# ── Cross-task plot ──────────────────────────────────────────────────────────

def _plot_genome_diversity_cross_task(cross_task_data, genotypes, out_dir):
    """Cross-task box plot of final-generation genome diversity."""
    if not cross_task_data:
        return

    df = pd.DataFrame(cross_task_data)
    fig, ax = plt.subplots(figsize=(14, 7))

    n_tasks = len(TASKS)
    n_geno = len(genotypes)
    group_width = 0.8
    bar_width = group_width / n_geno
    x_positions = np.arange(n_tasks)

    for i, genotype in enumerate(genotypes):
        geno_df = df[df["genotype"] == genotype]
        offsets = x_positions + (i - n_geno / 2 + 0.5) * bar_width

        box_data = []
        for task in TASKS:
            task_vals = geno_df[geno_df["task"] == task]["final_gen_diversity"].values
            box_data.append(task_vals if len(task_vals) > 0 else np.array([]))

        ax.boxplot(
            box_data, positions=offsets, widths=bar_width * 0.9,
            patch_artist=True,
            boxprops=dict(facecolor=COLORS[genotype], alpha=0.7),
            medianprops=dict(color="black", linewidth=2),
            whiskerprops=dict(color=COLORS[genotype]),
            capprops=dict(color=COLORS[genotype]),
            flierprops=dict(markerfacecolor=COLORS[genotype], marker="o",
                            markersize=4, alpha=0.5),
            manage_ticks=False,
            label=GENOTYPE_LABELS[genotype],
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels([TASK_LABELS.get(t, t.capitalize()) for t in TASKS])
    apply_axis_style(ax, xlabel="Task", ylabel="Final-Generation Genome Diversity")
    apply_legend(ax)

    fig_path = os.path.join(out_dir, "genome_diversity_cross_task")
    save_figure(fig, out_dir, "genome_diversity_cross_task")

    # Analytical output
    print_analysis_header("Genome Diversity — Cross-Task")
    final_vals = {}
    for genotype in genotypes:
        geno_df = df[df["genotype"] == genotype]
        vals = geno_df["final_gen_diversity"].dropna().values
        if len(vals) > 0:
            final_vals[genotype] = vals
    print_summary_stats(final_vals, metric_label="final-gen genome diversity")
    print_stat_tests(final_vals)
    print_analysis_footer(f"{fig_path}.{{pdf,png}}")


# ── Public entry point ───────────────────────────────────────────────────────

def run(base_dir, tasks, genotypes, output_base=None):
    """Generate all genome diversity plots and analytical output.

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

    cross_task_data = []

    for task in tasks:
        genome_data_dict = {}
        pheno_data_dict = {}

        for genotype in genotypes:
            gdf = _load_genome_diversity(base_dir, task, genotype)
            genome_data_dict[genotype] = gdf

            pdf = _load_phenotypic_diversity(base_dir, task, genotype)
            pheno_data_dict[genotype] = pdf

        out_dir = figures_dir(base_dir, task=task, plotter_name=PLOTTER_NAME)

        if any(v is not None for v in genome_data_dict.values()):
            _plot_genome_diversity_over_generations(
                task, genome_data_dict, pheno_data_dict, genotypes, out_dir)

        _plot_genotypic_vs_phenotypic_diversity(
            task, genome_data_dict, pheno_data_dict, genotypes, out_dir)

        # Collect final-generation data for cross-task plot
        for genotype in genotypes:
            gdf = genome_data_dict.get(genotype)
            if gdf is None:
                continue
            max_gen = gdf["generation"].max()
            final_gen_df = gdf[gdf["generation"] == max_gen]
            for _, row in final_gen_df.iterrows():
                cross_task_data.append({
                    "task": task,
                    "genotype": genotype,
                    "run": row.get("run", None),
                    "final_gen_diversity": row["genome_diversity_mean"],
                })

    # Cross-task box plot
    cross_out = figures_dir(base_dir, task="cross_task", plotter_name=PLOTTER_NAME)
    _plot_genome_diversity_cross_task(cross_task_data, genotypes, cross_out)


if __name__ == "__main__":
    from experimentation.config import BASE_DIR, TASKS, GENOTYPES
    run(BASE_DIR, TASKS, GENOTYPES)
