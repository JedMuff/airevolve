"""Plotter for speciation dynamics data (NEAT experiments).

Produces per-task figures:
  - species_count_over_generations (line plot, mean +/- SE across runs)
  - species_composition (stacked area for representative run)
  - species_turnover (birth/extinction events over generations)
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from experimentation.config import (
    COLORS, GENOTYPE_LABELS, figures_dir,
)
from experimentation.collection_utils import (
    iter_experiments,
    iter_runs,
    load_evolution_csv,
)
from experimentation.plot_utils import (
    setup_style, save_figure, apply_axis_style,
    aggregate_by_generation, pad_generations,
    print_analysis_header, print_summary_stats, print_stat_tests,
    print_analysis_footer,
)

warnings.filterwarnings("ignore")

PLOTTER_NAME = "speciation_dynamics"


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_speciation_data(base_dir, task, genotype):
    """Load speciation_dynamics_data.csv."""
    path = os.path.join(base_dir, task, genotype, "speciation_dynamics_data.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


# ── Species count over generations ────────────────────────────────────────────

def _plot_species_count(data, genotypes, out_dir):
    """Line plot of species count over generations for all genotypes."""
    fig, ax = plt.subplots(figsize=(10, 6))

    final_vals = {}

    for genotype in genotypes:
        df = data.get(genotype)
        if df is None:
            continue

        max_gen = int(df["generation"].max()) + 1
        padded = pad_generations(df, max_generations=max_gen)
        mean_series, se_series = aggregate_by_generation(padded, "n_species")

        gens = mean_series.index.values
        means = mean_series.values
        ses = se_series.values

        ax.plot(gens, means, color=COLORS[genotype], linewidth=2,
                label=GENOTYPE_LABELS[genotype])
        ax.fill_between(gens, means - ses, means + ses,
                        color=COLORS[genotype], alpha=0.2)

        max_gen = df["generation"].max()
        final_vals[genotype] = df.loc[
            df["generation"] == max_gen, "n_species"
        ].dropna().values

    apply_axis_style(ax, xlabel="Generation", ylabel="Number of Species")
    ax.legend()
    fig.tight_layout()

    filename = "species_count_over_generations"
    save_figure(fig, out_dir, filename)

    print_analysis_header("Species Count Over Generations")
    print_summary_stats(final_vals, metric_label="Final Species Count")
    print_stat_tests(final_vals)
    print_analysis_footer(os.path.join(out_dir, filename))


# ── Species composition stacked area ─────────────────────────────────────────

def _plot_species_composition(base_dir, task, genotype, out_dir):
    """Stacked area chart of species sizes for a representative run.

    Picks the run with the median final species count.
    """
    experiment_dir = os.path.join(base_dir, task, genotype)
    runs = list(iter_runs(experiment_dir))
    if not runs:
        return

    # Find representative run (median final species count)
    run_species = []
    for run_name, run_path in runs:
        evo_df = load_evolution_csv(run_path)
        if evo_df is None or "species_id" not in evo_df.columns:
            continue
        pop_df = evo_df[evo_df["in_pop"] == True] if "in_pop" in evo_df.columns else evo_df  # noqa: E712
        max_gen = pop_df["generation"].max()
        final_n = pop_df[pop_df["generation"] == max_gen]["species_id"].nunique()
        run_species.append((run_name, run_path, final_n))

    if not run_species:
        return

    run_species.sort(key=lambda x: x[2])
    median_idx = len(run_species) // 2
    run_name, run_path, _ = run_species[median_idx]

    evo_df = load_evolution_csv(run_path)
    pop_df = evo_df[evo_df["in_pop"] == True] if "in_pop" in evo_df.columns else evo_df  # noqa: E712

    # Build generation x species size matrix
    generations = sorted(pop_df["generation"].unique())
    all_species = sorted(pop_df["species_id"].dropna().unique())
    species_map = {s: i for i, s in enumerate(all_species)}

    matrix = np.zeros((len(generations), len(all_species)))
    for i, gen in enumerate(generations):
        gen_df = pop_df[pop_df["generation"] == gen]
        counts = gen_df["species_id"].value_counts()
        for sid, count in counts.items():
            if sid in species_map:
                matrix[i, species_map[sid]] = count

    fig, ax = plt.subplots(figsize=(12, 6))

    # Use a colormap for species
    cmap = plt.cm.get_cmap("tab20", len(all_species))
    colors = [cmap(i) for i in range(len(all_species))]

    ax.stackplot(generations, matrix.T, colors=colors, alpha=0.8)

    apply_axis_style(ax, xlabel="Generation", ylabel="Population Count")
    fig.tight_layout()

    filename = f"species_composition_{genotype}"
    save_figure(fig, out_dir, filename)


# ── Species turnover ──────────────────────────────────────────────────────────

def _plot_species_turnover(data, genotypes, out_dir):
    """Line plot of species birth and extinction rates over generations."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    for genotype in genotypes:
        df = data.get(genotype)
        if df is None:
            continue

        max_gen = int(df["generation"].max()) + 1
        padded = pad_generations(df, max_generations=max_gen)

        mean_born, se_born = aggregate_by_generation(padded, "species_born")
        mean_ext, se_ext = aggregate_by_generation(padded, "species_extinct")

        gens = mean_born.index.values
        axes[0].plot(gens, mean_born.values, color=COLORS[genotype], linewidth=2,
                     label=GENOTYPE_LABELS[genotype])
        axes[0].fill_between(gens, mean_born.values - se_born.values,
                             mean_born.values + se_born.values,
                             color=COLORS[genotype], alpha=0.2)

        gens = mean_ext.index.values
        axes[1].plot(gens, mean_ext.values, color=COLORS[genotype], linewidth=2,
                     label=GENOTYPE_LABELS[genotype])
        axes[1].fill_between(gens, mean_ext.values - se_ext.values,
                             mean_ext.values + se_ext.values,
                             color=COLORS[genotype], alpha=0.2)

    apply_axis_style(axes[0], xlabel="Generation", ylabel="Species Born")
    axes[0].legend()

    apply_axis_style(axes[1], xlabel="Generation", ylabel="Species Extinct")
    axes[1].legend()

    fig.tight_layout()
    save_figure(fig, out_dir, "species_turnover")


# ── Public entry point ────────────────────────────────────────────────────────

def run(base_dir, tasks, genotypes, output_base=None):
    """Generate all speciation dynamics figures."""
    setup_style()

    out_base = output_base if output_base else base_dir

    for task in tasks:
        task_data = {}
        for genotype in genotypes:
            df = _load_speciation_data(base_dir, task, genotype)
            if df is not None:
                task_data[genotype] = df

        if not task_data:
            continue

        out_dir = figures_dir(out_base, task=task, plotter_name=PLOTTER_NAME)

        _plot_species_count(task_data, genotypes, out_dir)
        _plot_species_turnover(task_data, genotypes, out_dir)

        # Stacked area for each genotype (per-run visualization)
        for genotype in genotypes:
            if genotype in task_data:
                _plot_species_composition(base_dir, task, genotype, out_dir)


if __name__ == "__main__":
    from experimentation.config import BASE_DIR as _BASE_DIR, TASKS as _TASKS, GENOTYPES as _GENOTYPES
    run(_BASE_DIR, _TASKS, _GENOTYPES)
