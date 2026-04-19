"""Plotter for phenotype space coverage and density metrics.

Produces per-task figures:
  - mean_pairwise_distance: population spread over generations
  - mean_nn_distance: local sparsity over generations
  - coverage_ratio: sparsity relative to spread over generations
  - spread_vs_density: scatter of mean pairwise vs mean NN distance

Produces cross-task summary:
  - cross_task_coverage_ratio: final-generation coverage ratio comparison
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from experimentation.config import (
    BASE_DIR, TASKS, GENOTYPES, COLORS, GENOTYPE_LABELS, TASK_LABELS,
    Y_LIMITS, figures_dir,
)
from experimentation.plot_utils import (
    setup_style, save_figure, apply_axis_style,
    aggregate_by_generation, pad_generations,
    print_analysis_header, print_analysis_footer,
    print_summary_stats, print_stat_tests,
    print_trajectory_samples,
)

warnings.filterwarnings("ignore")

PLOTTER_NAME = "coverage_density"


# ── Data loading ─────────────────────────────────────────────────────────────

def _load_data(base_dir, task, genotype):
    """Load coverage_density_data.csv for one experiment."""
    path = os.path.join(base_dir, task, genotype, "coverage_density_data.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    return df if len(df) > 0 else None


# ── Trajectory plots ─────────────────────────────────────────────────────────

def _plot_trajectory(data_dict, genotypes, out_dir, col, ylabel, filename):
    """Generic trajectory plot: metric over generations with std shading."""
    fig, ax = plt.subplots(figsize=(10, 6))
    traj_data = {}

    for genotype in genotypes:
        if genotype not in data_dict or data_dict[genotype] is None:
            continue
        df = data_dict[genotype]
        mean, se = aggregate_by_generation(df, col)
        gens = mean.index.values
        color = COLORS[genotype]
        label = GENOTYPE_LABELS.get(genotype, genotype)
        ax.plot(gens, mean.values, label=label, color=color)
        ax.fill_between(gens, (mean - se).values, (mean + se).values,
                         alpha=0.2, color=color)
        traj_data[genotype] = df

    apply_axis_style(ax, xlabel="Generation", ylabel=ylabel)
    ylim_key = filename
    if ylim_key in Y_LIMITS:
        ax.set_ylim(0, Y_LIMITS[ylim_key])
    ax.legend()
    ax.grid(True)
    save_figure(fig, out_dir, filename)

    return traj_data


def _plot_spread_vs_density(data_dict, genotypes, out_dir):
    """Scatter of mean pairwise distance vs mean NN distance per generation.

    Each point is one generation of one run. Encodings with sparse coverage
    appear further from the diagonal (high NN relative to spread).
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    all_max_x = 0
    for genotype in genotypes:
        if genotype not in data_dict or data_dict[genotype] is None:
            continue
        df = data_dict[genotype]
        x = df["mean_pairwise_distance"].values
        y = df["mean_nn_distance"].values
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        if len(x) == 0:
            continue
        all_max_x = max(all_max_x, np.max(x))

        color = COLORS[genotype]
        label = GENOTYPE_LABELS.get(genotype, genotype)
        ax.scatter(x, y, color=color, alpha=0.25, s=15,
                   edgecolors="none", rasterized=True, label=label)

    # Reference line: y = x (maximum sparsity — NN equals mean pairwise)
    if all_max_x > 0:
        ref = np.linspace(0, all_max_x * 1.05, 100)
        ax.plot(ref, ref, "--", color="gray", linewidth=1, alpha=0.5,
                label="NN = spread")

    apply_axis_style(ax, xlabel="Mean Pairwise Distance (spread)",
                     ylabel="Mean NN Distance (sparsity)")
    ax.legend()
    ax.grid(True)
    save_figure(fig, out_dir, "spread_vs_density")


def _plot_spread_vs_density_by_task(all_task_data, genotypes, out_dir):
    """One panel per task showing spread vs density scatter."""
    tasks_with_data = [t for t in all_task_data if all_task_data[t]]
    if not tasks_with_data:
        return

    n = len(tasks_with_data)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6), squeeze=False)
    axes = axes[0]

    for idx, task in enumerate(tasks_with_data):
        ax = axes[idx]
        data_dict = all_task_data[task]
        all_max_x = 0

        for genotype in genotypes:
            if genotype not in data_dict or data_dict[genotype] is None:
                continue
            df = data_dict[genotype]
            x = df["mean_pairwise_distance"].values
            y = df["mean_nn_distance"].values
            mask = np.isfinite(x) & np.isfinite(y)
            x, y = x[mask], y[mask]
            if len(x) == 0:
                continue
            all_max_x = max(all_max_x, np.max(x))

            color = COLORS[genotype]
            label = GENOTYPE_LABELS.get(genotype, genotype)
            ax.scatter(x, y, color=color, alpha=0.25, s=15,
                       edgecolors="none", rasterized=True, label=label)

        if all_max_x > 0:
            ref = np.linspace(0, all_max_x * 1.05, 100)
            ax.plot(ref, ref, "--", color="gray", linewidth=1, alpha=0.5)

        apply_axis_style(
            ax,
            xlabel="Mean Pairwise Distance",
            ylabel="Mean NN Distance" if idx == 0 else None,
        )
        ax.set_title(TASK_LABELS.get(task, task), fontsize=18)
        ax.legend()
        ax.grid(True)

    fig.tight_layout()
    save_figure(fig, out_dir, "cross_task_spread_vs_density")


# ── Analytical output ────────────────────────────────────────────────────────

def _print_final_gen_stats(data_dict, genotypes, task):
    """Print final-generation summary stats and significance tests."""
    final_vals = {}
    for genotype in genotypes:
        if genotype not in data_dict or data_dict[genotype] is None:
            continue
        df = data_dict[genotype]
        final_gen = df["generation"].max()
        final = df[df["generation"] == final_gen]
        final_vals[genotype] = final["coverage_ratio"].dropna().values

    if not final_vals:
        return

    print_analysis_header(f"Coverage Ratio (Final Generation) — {task}")
    print_summary_stats(final_vals, metric_label="coverage ratio (NN/spread)")
    print_stat_tests(final_vals)

    # Also print NN and spread separately
    for metric, label in [("mean_nn_distance", "NN distance"),
                          ("mean_pairwise_distance", "pairwise distance")]:
        vals = {}
        for genotype in genotypes:
            if genotype not in data_dict or data_dict[genotype] is None:
                continue
            df = data_dict[genotype]
            final_gen = df["generation"].max()
            final = df[df["generation"] == final_gen]
            vals[genotype] = final[metric].dropna().values
        if vals:
            print_summary_stats(vals, metric_label=label)
            print_stat_tests(vals)

    print_analysis_footer(task)


# ── Cross-task summary table ─────────────────────────────────────────────────

def _collect_final_gen_data(all_task_data, tasks, genotypes):
    """Return dict keyed by (task, genotype, metric_col) → np.array of final-gen values."""
    metrics = [
        ("coverage_ratio",         "Cov. Ratio"),
        ("mean_pairwise_distance", "Pairwise Dist."),
        ("mean_nn_distance",       "NN Dist."),
    ]
    data = {}
    for task in tasks:
        if task not in all_task_data:
            continue
        for genotype in genotypes:
            df = all_task_data[task].get(genotype)
            if df is None:
                continue
            final_gen = df["generation"].max()
            final = df[df["generation"] == final_gen]
            for col, _ in metrics:
                data[(task, genotype, col)] = final[col].dropna().values
    return data, metrics


def _generate_cross_task_table(all_task_data, tasks, genotypes, base_dir):
    """Generate a LaTeX table of final-generation coverage/density stats across tasks.

    Rows: tasks × genotypes. Best value per task row is bolded.
    Followed by a condensed 3-row table averaged over all tasks.
    """
    data, metrics = _collect_final_gen_data(all_task_data, tasks, genotypes)
    if not data:
        return

    col_spec = "ll" + " r@{$\\,\\pm\\,$}l" * len(metrics)
    metric_headers = " & ".join(
        f"\\multicolumn{{2}}{{c}}{{\\textbf{{{label}}}}}"
        for _, label in metrics
    )

    # ── Per-task table ────────────────────────────────────────────────────────
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(
        "\\caption{Final-generation coverage and density statistics "
        "(mean $\\pm$ std across runs). Bold indicates best value per task row.}"
    )
    lines.append("\\label{tab:coverage_density_stats}")
    lines.append("\\small")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    lines.append(f"\\textbf{{Task}} & \\textbf{{Genotype}} & {metric_headers} \\\\")
    lines.append("\\midrule")

    for task in tasks:
        if task not in all_task_data:
            continue
        task_label = TASK_LABELS.get(task, task)

        best_mean = {}
        for col, _ in metrics:
            vals_by_g = {g: data[(task, g, col)] for g in genotypes
                         if (task, g, col) in data and len(data[(task, g, col)]) > 0}
            if vals_by_g:
                best_mean[col] = max(np.mean(v) for v in vals_by_g.values())

        first = True
        for genotype in genotypes:
            if not any((task, genotype, col) in data for col, _ in metrics):
                continue
            task_cell = task_label if first else ""
            first = False
            cells = []
            for col, _ in metrics:
                vals = data.get((task, genotype, col), np.array([]))
                if len(vals) == 0:
                    cells.append("-- & --")
                    continue
                m, s = np.mean(vals), np.std(vals)
                if col in best_mean and np.isclose(m, best_mean[col]):
                    cells.append(f"\\textbf{{{m:.3f}}} & \\textbf{{{s:.3f}}}")
                else:
                    cells.append(f"{m:.3f} & {s:.3f}")
            lines.append(
                f"{task_cell} & {GENOTYPE_LABELS.get(genotype, genotype)} & "
                + " & ".join(cells) + " \\\\"
            )
        lines.append("\\midrule")

    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    # ── Condensed table (3 rows, averaged over all tasks) ────────────────────
    lines.append("")
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(
        "\\caption{Coverage and density statistics averaged across all tasks "
        "(mean $\\pm$ std pooled over tasks and runs). Bold indicates best value.}"
    )
    lines.append("\\label{tab:coverage_density_condensed}")
    lines.append("\\small")
    col_spec_condensed = "l" + " r@{$\\,\\pm\\,$}l" * len(metrics)
    lines.append(f"\\begin{{tabular}}{{{col_spec_condensed}}}")
    lines.append("\\toprule")
    lines.append(f"\\textbf{{Genotype}} & {metric_headers} \\\\")
    lines.append("\\midrule")

    # Pool values across all tasks for each (genotype, metric)
    pooled = {}
    for genotype in genotypes:
        pooled[genotype] = {}
        for col, _ in metrics:
            all_vals = np.concatenate([
                data[(task, genotype, col)]
                for task in tasks
                if (task, genotype, col) in data and len(data[(task, genotype, col)]) > 0
            ]) if any(
                (task, genotype, col) in data and len(data[(task, genotype, col)]) > 0
                for task in tasks
            ) else np.array([])
            pooled[genotype][col] = all_vals

    best_pooled = {}
    for col, _ in metrics:
        means = {g: np.mean(pooled[g][col]) for g in genotypes
                 if len(pooled[g][col]) > 0}
        if means:
            best_pooled[col] = max(means.values())

    for genotype in genotypes:
        if not any(len(pooled[genotype][col]) > 0 for col, _ in metrics):
            continue
        cells = []
        for col, _ in metrics:
            vals = pooled[genotype][col]
            if len(vals) == 0:
                cells.append("-- & --")
                continue
            m, s = np.mean(vals), np.std(vals)
            if col in best_pooled and np.isclose(m, best_pooled[col]):
                cells.append(f"\\textbf{{{m:.3f}}} & \\textbf{{{s:.3f}}}")
            else:
                cells.append(f"{m:.3f} & {s:.3f}")
        lines.append(
            f"{GENOTYPE_LABELS.get(genotype, genotype)} & "
            + " & ".join(cells) + " \\\\"
        )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    latex = "\n".join(lines)

    out_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            "coverage_density_table.tex")
    with open(out_path, "w") as f:
        f.write(latex)
    print(f"\n  Cross-task coverage/density table saved to: {out_path}")
    print()
    print(latex)


# ── Main entry point ────────────────────────────────────────────────────────

def run(base_dir, tasks, genotypes, output_base=None):
    """Run all coverage/density plots and analyses.

    Args:
        base_dir: root data directory
        tasks: list of task names
        genotypes: list of genotype names
        output_base: override for output root
    """
    setup_style()

    all_task_data = {}

    for task in tasks:
        print(f"\n{'=' * 60}")
        print(f"Coverage/Density: {TASK_LABELS.get(task, task)}")
        print(f"{'=' * 60}")

        data_dict = {}
        for genotype in genotypes:
            data_dict[genotype] = _load_data(base_dir, task, genotype)

        if not any(v is not None for v in data_dict.values()):
            continue

        all_task_data[task] = data_dict
        out_dir = figures_dir(base_dir, task=task, plotter_name=PLOTTER_NAME)

        # Trajectory plots
        _plot_trajectory(data_dict, genotypes, out_dir,
                         "mean_pairwise_distance", "Mean Pairwise Distance",
                         "mean_pairwise_distance")

        _plot_trajectory(data_dict, genotypes, out_dir,
                         "mean_nn_distance", "Mean NN Distance",
                         "mean_nn_distance")

        traj_data = _plot_trajectory(data_dict, genotypes, out_dir,
                                     "coverage_ratio", "Coverage Ratio (NN / Spread)",
                                     "coverage_ratio")

        # Trajectory samples
        if traj_data:
            print_analysis_header(f"Coverage Ratio Trajectory — {task}")
            print_trajectory_samples(traj_data, value_col="coverage_ratio")
            print_analysis_footer(os.path.join(out_dir, "coverage_ratio"))

        # Spread vs density scatter
        _plot_spread_vs_density(data_dict, genotypes, out_dir)

        # Final-generation stats
        _print_final_gen_stats(data_dict, genotypes, task)

    # Cross-task spread vs density and summary table
    if len(all_task_data) > 1:
        cross_dir = figures_dir(base_dir, task="cross_task",
                                plotter_name=PLOTTER_NAME)
        _plot_spread_vs_density_by_task(all_task_data, genotypes, cross_dir)

    _generate_cross_task_table(all_task_data, tasks, genotypes, base_dir)


if __name__ == "__main__":
    run(BASE_DIR, TASKS, GENOTYPES)
