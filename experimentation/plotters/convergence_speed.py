"""Plotter for convergence speed analysis.

Produces per-task figures:
  - gen_to_threshold: grouped box plot of generation-to-{50,75,90}% fitness
  - auc_normalized: box plot of normalized area under the fitness curve
  - early_fitness: grouped box plot of fitness at early generations

Produces cross-task figures:
  - cross_task_auc: grouped box of normalized AUC by task
  - cross_task_gen_to_75pct: grouped box of gen-to-75% by task
  - cross-task significance summary with Fisher's combined p-values
"""

import os
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, combine_pvalues

from experimentation.config import (
    BASE_DIR, TASKS, GENOTYPES, COLORS, GENOTYPE_LABELS, TASK_LABELS,
    figures_dir,
)
from experimentation.plot_utils import (
    setup_style, save_figure, apply_axis_style,
    print_analysis_header, print_summary_stats, print_stat_tests,
    print_analysis_footer, rank_biserial, _effect_size_label,
)

warnings.filterwarnings("ignore")

PLOTTER_NAME = "convergence_speed"


# ── Data loading ─────────────────────────────────────────────────────────────

def _load_data(base_dir, task, genotypes):
    """Load convergence_speed_data.csv for each genotype, return dict of DataFrames."""
    data = {}
    for genotype in genotypes:
        csv_path = os.path.join(base_dir, task, genotype, "convergence_speed_data.csv")
        if os.path.exists(csv_path):
            data[genotype] = pd.read_csv(csv_path)
    return data


# ── Grouped box plot helper ──────────────────────────────────────────────────

def _grouped_boxplot(data_by_genotype, group_metrics, group_labels, genotypes,
                     ylabel, out_dir, filename, title=None):
    """Draw a grouped box plot with genotypes side-by-side within each group."""
    n_groups = len(group_metrics)
    n_genotypes = len(genotypes)
    width = 0.7 / n_genotypes
    fig, ax = plt.subplots(figsize=(10, 6))

    positions_map = {}
    for gi, genotype in enumerate(genotypes):
        if genotype not in data_by_genotype:
            continue
        df = data_by_genotype[genotype]
        box_data = []
        positions = []
        for mi, metric in enumerate(group_metrics):
            vals = df[metric].dropna().values
            box_data.append(vals)
            pos = mi + (gi - (n_genotypes - 1) / 2) * width
            positions.append(pos)
        if box_data:
            bp = ax.boxplot(box_data, positions=positions, widths=width * 0.85,
                            patch_artist=True, manage_ticks=False)
            color = COLORS.get(genotype, "#333333")
            for patch in bp["boxes"]:
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            for element in ("whiskers", "caps", "medians"):
                for line in bp[element]:
                    line.set_color("black")
            positions_map[genotype] = positions

    ax.set_xticks(range(n_groups))
    ax.set_xticklabels(group_labels)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    # Legend
    handles = []
    for genotype in genotypes:
        if genotype in data_by_genotype:
            handles.append(plt.Rectangle((0, 0), 1, 1,
                           facecolor=COLORS.get(genotype, "#333"),
                           alpha=0.7,
                           label=GENOTYPE_LABELS.get(genotype, genotype)))
    if handles:
        ax.legend(handles=handles, loc="best")

    apply_axis_style(ax, xlabel="", ylabel=ylabel)
    save_figure(fig, out_dir, filename)


def _single_boxplot(data_by_genotype, metric, genotypes, ylabel, out_dir,
                    filename, title=None):
    """Draw a single-metric box plot with one box per genotype."""
    fig, ax = plt.subplots(figsize=(8, 6))
    box_data = []
    labels = []
    colors_list = []
    for genotype in genotypes:
        if genotype not in data_by_genotype:
            continue
        vals = data_by_genotype[genotype][metric].dropna().values
        box_data.append(vals)
        labels.append(GENOTYPE_LABELS.get(genotype, genotype))
        colors_list.append(COLORS.get(genotype, "#333333"))

    if not box_data:
        plt.close(fig)
        return

    bp = ax.boxplot(box_data, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticklabels(labels)
    if title:
        ax.set_title(title)
    apply_axis_style(ax, xlabel="", ylabel=ylabel)
    save_figure(fig, out_dir, filename)


# ── Cross-task grouped box plot ──────────────────────────────────────────────

def _cross_task_grouped_boxplot(all_data, tasks, genotypes, metric, ylabel,
                                out_dir, filename):
    """Box plot: tasks on x-axis, genotypes side-by-side, y = metric."""
    n_tasks = len(tasks)
    n_genotypes = len(genotypes)
    width = 0.7 / n_genotypes
    fig, ax = plt.subplots(figsize=(10, 6))

    for gi, genotype in enumerate(genotypes):
        box_data = []
        positions = []
        for ti, task in enumerate(tasks):
            task_data = all_data.get(task, {})
            if genotype in task_data:
                vals = task_data[genotype][metric].dropna().values
            else:
                vals = np.array([])
            box_data.append(vals)
            positions.append(ti + (gi - (n_genotypes - 1) / 2) * width)

        if any(len(d) > 0 for d in box_data):
            bp = ax.boxplot(box_data, positions=positions, widths=width * 0.85,
                            patch_artist=True, manage_ticks=False)
            color = COLORS.get(genotype, "#333333")
            for patch in bp["boxes"]:
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            for element in ("whiskers", "caps", "medians"):
                for line in bp[element]:
                    line.set_color("black")

    ax.set_xticks(range(n_tasks))
    ax.set_xticklabels([TASK_LABELS.get(t, t) for t in tasks])

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=COLORS.get(g, "#333"),
               alpha=0.7, label=GENOTYPE_LABELS.get(g, g))
               for g in genotypes if any(g in all_data.get(t, {}) for t in tasks)]
    if handles:
        ax.legend(handles=handles, loc="best")

    apply_axis_style(ax, xlabel="", ylabel=ylabel)
    save_figure(fig, out_dir, filename)


# ── Significance helpers ─────────────────────────────────────────────────────

def _pairwise_tests(data_by_genotype, metric, genotypes):
    """Run pairwise Mann-Whitney U with Bonferroni correction for a metric."""
    available = [g for g in genotypes if g in data_by_genotype]
    pairs = list(combinations(available, 2))
    n_tests = len(pairs)
    results = []
    for a, b in pairs:
        va = data_by_genotype[a][metric].dropna().values
        vb = data_by_genotype[b][metric].dropna().values
        if len(va) < 2 or len(vb) < 2:
            results.append({"genome_a": a, "genome_b": b,
                           "p_value": np.nan, "p_corrected": np.nan,
                           "significant": False, "effect_size": np.nan,
                           "effect_label": ""})
            continue
        u_stat, p_value = mannwhitneyu(va, vb, alternative="two-sided")
        p_corr = min(p_value * n_tests, 1.0)
        r = rank_biserial(u_stat, len(va), len(vb))
        results.append({
            "genome_a": a, "genome_b": b,
            "median_a": np.median(va), "median_b": np.median(vb),
            "u_stat": u_stat, "p_value": p_value,
            "p_corrected": p_corr, "significant": p_corr < 0.05,
            "effect_size": r, "effect_label": _effect_size_label(r),
        })
    return results


def _print_metric_analysis(data_by_genotype, metric, metric_label, genotypes,
                           figure_path):
    """Print full analysis block for a single metric."""
    print_analysis_header(f"{metric_label}")
    vals_dict = {}
    for g in genotypes:
        if g in data_by_genotype:
            vals_dict[g] = data_by_genotype[g][metric].dropna().values
    print_summary_stats(vals_dict, metric_label=metric_label)
    print_stat_tests(vals_dict)
    print_analysis_footer(figure_path)


def _print_cross_task_significance(all_task_results, metric, metric_label,
                                   genotypes, tasks):
    """Print cross-task combined p-values with effect sizes for a metric."""
    available = genotypes
    pairs = list(combinations(available, 2))

    pair_pvals = {(a, b): [] for a, b in pairs}
    pair_effects = {(a, b): [] for a, b in pairs}

    for task in tasks:
        if task not in all_task_results:
            continue
        for r in all_task_results[task]:
            a, b = r["genome_a"], r["genome_b"]
            if not np.isnan(r["p_value"]):
                pair_pvals[(a, b)].append(r["p_value"])
            eff = r.get("effect_size")
            if eff is not None and not np.isnan(eff):
                pair_effects[(a, b)].append(eff)

    print(f"\nCOMBINED P-VALUES (Fisher's method) - {metric_label}")
    print("-" * 100)
    print(f"{'Comparison':<30} | {'Combined p-value':>16} | {'Significant':>11} | "
          f"{'Mean Effect r':>14} | {'Effect Size':>12}")
    print("-" * 100)
    for a, b in pairs:
        pvals = pair_pvals[(a, b)]
        if len(pvals) >= 2:
            _, cp = combine_pvalues(pvals, method="fisher")
        elif len(pvals) == 1:
            cp = pvals[0]
        else:
            cp = np.nan
        effs = pair_effects[(a, b)]
        mean_eff = np.mean(effs) if effs else np.nan
        eff_label = _effect_size_label(mean_eff) if not np.isnan(mean_eff) else ""
        sig = "Yes" if cp < 0.05 else "No"
        print(f"{a} vs {b:<27} | {cp:>16.6f} | {sig:>11} | "
              f"{mean_eff:>+14.3f} | {eff_label:>12}")
    print("-" * 100)


# ── LaTeX helpers ────────────────────────────────────────────────────────────

def _save_significance_latex(all_results, metrics_labels, save_path):
    """Save significance results as LaTeX tables."""
    lines = []
    for metric, label in metrics_labels.items():
        if metric not in all_results:
            continue
        suffix = metric.replace(" ", "_")
        lines.append("\\begin{table}[htbp]")
        lines.append("\\centering")
        lines.append(f"\\caption{{Convergence Speed - {label}}}")
        lines.append(f"\\label{{tab:convergence_{suffix}}}")
        lines.append("\\begin{tabular}{|l|c|c|c|c|c|}")
        lines.append("\\hline")
        lines.append("\\textbf{Comparison} & \\textbf{Median A} & \\textbf{Median B} & "
                     "\\textbf{p-corrected} & \\textbf{Sig.} & \\textbf{Effect $r$} \\\\")
        lines.append("\\hline")
        for r in all_results[metric]:
            a = r["genome_a"].replace("_", "\\_")
            b = r["genome_b"].replace("_", "\\_")
            sig = "Yes" if r["significant"] else "No"
            eff_r = r.get("effect_size", np.nan)
            eff_label = r.get("effect_label", "")
            med_a = r.get("median_a", np.nan)
            med_b = r.get("median_b", np.nan)
            lines.append(f"{a} vs {b} & {med_a:.4f} & {med_b:.4f} & "
                        f"{r['p_corrected']:.4f} & {sig} & "
                        f"{eff_r:+.3f} ({eff_label}) \\\\")
            lines.append("\\hline")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")
        lines.append("")

    filepath = os.path.join(save_path, "convergence_significance.tex")
    os.makedirs(save_path, exist_ok=True)
    with open(filepath, "w") as f:
        f.write("\n".join(lines))
    print(f"  LaTeX significance table saved to: {filepath}")


# ── Main entry point ─────────────────────────────────────────────────────────

def run(base_dir, tasks, genotypes):
    """Run convergence speed plots and analysis."""
    setup_style()

    all_task_data = {}      # task -> {genotype: DataFrame}
    all_task_results = {}   # metric -> {task -> [pairwise results]}

    # Key metrics for analysis
    metrics_labels = {
        "gen_to_50pct": "Generation to 50% Fitness",
        "gen_to_75pct": "Generation to 75% Fitness",
        "gen_to_90pct": "Generation to 90% Fitness",
        "auc_normalized": "Normalized AUC",
        "fitness_gen_5": "Fitness at Generation 5",
        "fitness_gen_10": "Fitness at Generation 10",
        "fitness_gen_15": "Fitness at Generation 15",
        "fitness_gen_25": "Fitness at Generation 25",
    }

    for task in tasks:
        data = _load_data(base_dir, task, genotypes)
        if not data:
            continue

        all_task_data[task] = data
        task_label = TASK_LABELS.get(task, task)
        out_dir = figures_dir(base_dir, task=task, plotter_name=PLOTTER_NAME)

        print(f"\n{'=' * 60}")
        print(f"Convergence Speed: {task_label}")
        print(f"{'=' * 60}")

        # --- Generation-to-threshold grouped box plot ---
        _grouped_boxplot(
            data, ["gen_to_50pct", "gen_to_75pct", "gen_to_90pct"],
            ["50%", "75%", "90%"], genotypes,
            ylabel="Generation", out_dir=out_dir,
            filename="gen_to_threshold",
        )

        # --- Normalized AUC box plot ---
        _single_boxplot(
            data, "auc_normalized", genotypes,
            ylabel="Normalized AUC", out_dir=out_dir,
            filename="auc_normalized",
        )

        # --- Early fitness grouped box plot ---
        _grouped_boxplot(
            data, ["fitness_gen_5", "fitness_gen_10", "fitness_gen_15", "fitness_gen_25"],
            ["Gen 5", "Gen 10", "Gen 15", "Gen 25"], genotypes,
            ylabel="Max Fitness", out_dir=out_dir,
            filename="early_fitness",
        )

        # --- Statistical analysis for each metric ---
        task_pairwise = {}
        for metric, label in metrics_labels.items():
            _print_metric_analysis(data, metric, f"{label} — {task_label}",
                                   genotypes, out_dir)
            results = _pairwise_tests(data, metric, genotypes)
            task_pairwise[metric] = results
            if metric not in all_task_results:
                all_task_results[metric] = {}
            all_task_results[metric][task] = results

        # Save per-task LaTeX
        _save_significance_latex(task_pairwise, metrics_labels, out_dir)

    # ── Cross-task analysis ──────────────────────────────────────────────────

    if len(all_task_data) < 2:
        return

    cross_dir = figures_dir(base_dir, task="cross_task", plotter_name=PLOTTER_NAME)

    print(f"\n{'=' * 60}")
    print("CROSS-TASK CONVERGENCE SPEED SUMMARY")
    print(f"{'=' * 60}")

    # Cross-task AUC plot
    _cross_task_grouped_boxplot(
        all_task_data, tasks, genotypes, "auc_normalized",
        ylabel="Normalized AUC", out_dir=cross_dir,
        filename="cross_task_auc",
    )

    # Cross-task gen-to-75% plot
    _cross_task_grouped_boxplot(
        all_task_data, tasks, genotypes, "gen_to_75pct",
        ylabel="Generation to 75% Fitness", out_dir=cross_dir,
        filename="cross_task_gen_to_75pct",
    )

    # Cross-task significance for key metrics
    for metric in ("auc_normalized", "gen_to_75pct", "fitness_gen_10"):
        if metric in all_task_results:
            label = metrics_labels[metric]
            _print_cross_task_significance(
                all_task_results[metric], metric, label,
                genotypes, tasks,
            )

    # Cross-task LaTeX
    cross_results = {}
    for metric, task_results in all_task_results.items():
        # Aggregate all task results for this metric
        all_pairs = []
        for task_res in task_results.values():
            all_pairs.extend(task_res)
        cross_results[metric] = all_pairs
    _save_significance_latex(cross_results, metrics_labels, cross_dir)


if __name__ == "__main__":
    run(BASE_DIR, TASKS, GENOTYPES)
