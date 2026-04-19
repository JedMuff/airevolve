"""Plotter for fitness and diversity data.

Produces per-task figures:
  - max_fitness: mean of per-run max fitness over generations (all genotypes)
  - mean_fitness: mean of per-run mean fitness over generations (all genotypes)
  - diversity: diversity over generations (all genotypes)
  - median_max_fitness: median max fitness with IQR shading (all genotypes)
  - median_mean_fitness: median mean fitness with IQR shading (all genotypes)

Produces cross-task figures:
  - cross_task win matrix and Fisher's combined p-values (printed)
  - significance LaTeX tables
"""

import os
import warnings
from itertools import combinations
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, combine_pvalues

from experimentation.config import (
    BASE_DIR, TASKS, GENOTYPES, COLORS, GENOTYPE_LABELS, TASK_LABELS,
    Y_LIMITS, figures_dir, csv_dir,
)
from experimentation.plot_utils import (
    setup_style, save_figure, apply_axis_style, apply_legend,
    set_tick_intervals,
    aggregate_by_generation, pad_generations,
    print_analysis_header, print_summary_stats, print_trajectory_samples,
    print_stat_tests, print_analysis_footer,
    rank_biserial, _effect_size_label,
)

warnings.filterwarnings("ignore")

PLOTTER_NAME = "fitness_diversity"


# ── Data helpers ─────────────────────────────────────────────────────────────

def _load_task_data(base_dir, task, genotypes):
    """Load fitness/diversity CSVs for each genotype under a task."""
    data = {}
    for genotype in genotypes:
        csv_path = os.path.join(base_dir, task, genotype, "fitness_diversity_data.csv")
        if os.path.exists(csv_path):
            data[genotype] = pd.read_csv(csv_path)
    return data


def _aggregate_runs(df):
    """Aggregate fitness/diversity across runs per generation.

    Returns dict with keys max_fitness, mean_fitness, std_fitness, diversity
    each as 2-D arrays (generations x runs).
    """
    grouped = df.groupby("generation")
    max_runs = max(len(g) for _, g in grouped)
    max_gen = df["generation"].max()

    arrays = {
        k: np.full((max_gen + 1, max_runs), np.nan)
        for k in ("max_fitness", "mean_fitness", "std_fitness", "diversity")
    }

    for gen, group in grouped:
        for idx, (_, row) in enumerate(group.iterrows()):
            if idx < max_runs:
                for k in arrays:
                    arrays[k][gen, idx] = row[k]
    return arrays


def _clean_inf(arr):
    """Replace +/-inf with NaN in-place."""
    arr[np.isinf(arr)] = np.nan
    return arr


# ── Significance tests ──────────────────────────────────────────────────────

def _run_significance_tests(experiment_data, genotypes):
    """Pairwise Mann-Whitney U on final-generation fitness values."""
    final_values = {}
    for name in genotypes:
        if name not in experiment_data:
            continue
        df = experiment_data[name]
        final_gen = df["generation"].max()
        final = df[df["generation"] == final_gen]
        final_values[name] = {
            "max_fitness": final["max_fitness"].values,
            "mean_fitness": final["mean_fitness"].values,
        }

    available = [n for n in genotypes if n in final_values]
    pairs = list(combinations(available, 2))
    n_comparisons = len(pairs)

    results = {}
    for metric in ("max_fitness", "mean_fitness"):
        results[metric] = []
        for a, b in pairs:
            va = final_values[a][metric]
            vb = final_values[b][metric]
            va = va[~np.isnan(va)]
            vb = vb[~np.isnan(vb)]
            if len(va) < 2 or len(vb) < 2:
                results[metric].append({
                    "genome_a": a, "genome_b": b,
                    "median_a": np.nan, "median_b": np.nan,
                    "u_stat": np.nan, "p_value": np.nan,
                    "p_corrected": np.nan, "significant": False,
                })
                continue
            u_stat, p_value = mannwhitneyu(va, vb, alternative="two-sided")
            p_corr = min(p_value * n_comparisons, 1.0)
            r_rb = rank_biserial(u_stat, len(va), len(vb))
            results[metric].append({
                "genome_a": a, "genome_b": b,
                "median_a": np.median(va), "median_b": np.median(vb),
                "u_stat": u_stat, "p_value": p_value,
                "p_corrected": p_corr, "significant": p_corr < 0.05,
                "effect_size": r_rb,
                "effect_label": _effect_size_label(r_rb),
            })
    return results


def _print_significance(results, task_name=""):
    """Print formatted significance test results with effect sizes."""
    prefix = f" - {task_name}" if task_name else ""
    for metric in ("max_fitness", "mean_fitness"):
        label = metric.replace("_", " ").title()
        print(f"\nSIGNIFICANCE TESTS{prefix} - {label} (Final Generation)")
        print("-" * 146)
        print(f"{'Comparison':<30} | {'Median A':>10} | {'Median B':>10} | "
              f"{'U-stat':>10} | {'p-value':>10} | {'p-corrected':>12} | {'Significant':>11} | "
              f"{'Effect r':>9} | {'Effect Size':>12}")
        print("-" * 146)
        for r in results[metric]:
            sig = "Yes" if r["significant"] else "No"
            comp = f"{r['genome_a']} vs {r['genome_b']}"
            eff_r = r.get("effect_size", np.nan)
            eff_label = r.get("effect_label", "")
            print(f"{comp:<30} | {r['median_a']:>10.4f} | {r['median_b']:>10.4f} | "
                  f"{r['u_stat']:>10.1f} | {r['p_value']:>10.4f} | "
                  f"{r['p_corrected']:>12.4f} | {sig:>11} | "
                  f"{eff_r:>+9.3f} | {eff_label:>12}")
        print("-" * 146)


def _generate_significance_latex(results, save_path, task_name=""):
    """Save significance results as LaTeX tables (with effect sizes)."""
    lines = []
    for metric in ("max_fitness", "mean_fitness"):
        label = metric.replace("_", " ").title()
        suffix = metric.replace("_", "")
        task_suffix = f"_{task_name}" if task_name else ""
        lines.append("\\begin{table}[htbp]")
        lines.append("\\centering")
        lines.append(f"\\caption{{Significance Tests - {label}"
                     f"{'(' + task_name + ')' if task_name else ''}}}")
        lines.append(f"\\label{{tab:significance_{suffix}{task_suffix}}}")
        lines.append("\\begin{tabular}{|l|c|c|c|c|c|c|c|}")
        lines.append("\\hline")
        lines.append("\\textbf{Comparison} & \\textbf{Median A} & \\textbf{Median B} & "
                     "\\textbf{U-stat} & \\textbf{p-value} & \\textbf{p-corrected} & "
                     "\\textbf{Sig.} & \\textbf{Effect $r$} \\\\")
        lines.append("\\hline")
        for r in results[metric]:
            a = r["genome_a"].replace("_", "\\_")
            b = r["genome_b"].replace("_", "\\_")
            sig = "Yes" if r["significant"] else "No"
            eff_r = r.get("effect_size", np.nan)
            eff_label = r.get("effect_label", "")
            lines.append(f"{a} vs {b} & {r['median_a']:.4f} & {r['median_b']:.4f} & "
                        f"{r['u_stat']:.1f} & {r['p_value']:.4f} & "
                        f"{r['p_corrected']:.4f} & {sig} & "
                        f"{eff_r:+.3f} ({eff_label}) \\\\")
            lines.append("\\hline")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")
        lines.append("")

    filepath = os.path.join(save_path, "significance_table.tex")
    os.makedirs(save_path, exist_ok=True)
    with open(filepath, "w") as f:
        f.write("\n".join(lines))
    print(f"  LaTeX significance table saved to: {filepath}")


def _print_cross_task_summary(all_task_results, genotypes, save_path=None):
    """Print cross-task win matrix and Fisher combined p-values."""
    available = genotypes
    pairs = list(combinations(available, 2))
    tasks = list(all_task_results.keys())
    n_tasks = len(tasks)

    for metric in ("max_fitness", "mean_fitness"):
        label = metric.replace("_", " ").title()
        wins = {g: {g2: 0 for g2 in available} for g in available}
        pair_pvals = {(a, b): [] for a, b in pairs}

        for task_name, results in all_task_results.items():
            for r in results[metric]:
                a, b = r["genome_a"], r["genome_b"]
                if not np.isnan(r["p_value"]):
                    pair_pvals[(a, b)].append(r["p_value"])
                if r["significant"]:
                    if r["median_a"] > r["median_b"]:
                        wins[a][b] += 1
                    elif r["median_b"] > r["median_a"]:
                        wins[b][a] += 1

        print(f"\nCROSS-TASK PAIRWISE WIN MATRIX - {label}")
        print("-" * (25 + 15 * len(available)))
        header = f"{'':>20}" + "".join(f"{g:>15}" for g in available)
        print(header)
        for a in available:
            row = f"{a:>20}"
            for b in available:
                if a == b:
                    row += f"{'---':>15}"
                else:
                    row += f"{wins[a][b]}/{n_tasks}".rjust(15)
            print(row)
        print("-" * (25 + 15 * len(available)))

        # Collect effect sizes across tasks
        pair_effects = {(a, b): [] for a, b in pairs}
        for task_name, results in all_task_results.items():
            for r in results[metric]:
                a, b = r["genome_a"], r["genome_b"]
                eff = r.get("effect_size")
                if eff is not None and not np.isnan(eff):
                    pair_effects[(a, b)].append(eff)

        print(f"\nCOMBINED P-VALUES (Fisher's method) - {label}")
        print("-" * 100)
        print(f"{'Comparison':<30} | {'Combined p-value':>16} | {'Significant':>11} | "
              f"{'Mean Effect r':>14} | {'Effect Size':>12}")
        print("-" * 100)
        fisher = []
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
            fisher.append({"genome_a": a, "genome_b": b,
                          "combined_p": cp, "significant": cp < 0.05,
                          "mean_effect": mean_eff, "effect_label": eff_label})
        print("-" * 100)

        if save_path:
            _save_cross_task_latex(wins, fisher, available, n_tasks, metric, label, save_path)


def _save_cross_task_latex(wins, fisher, available, n_tasks, metric, label, save_path):
    """Save cross-task significance results as LaTeX."""
    lines = []
    suffix = metric.replace("_", "")

    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(f"\\caption{{Cross-Task Pairwise Win Matrix - {label}}}")
    lines.append(f"\\label{{tab:win_matrix_{suffix}}}")
    col_spec = "|l|" + "c|" * len(available)
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\hline")
    hdr = " & ".join([""] + [g.replace("_", "\\_") for g in available]) + " \\\\"
    lines.append(hdr)
    lines.append("\\hline")
    for a in available:
        cells = [a.replace("_", "\\_")]
        for b in available:
            cells.append("---" if a == b else f"{wins[a][b]}/{n_tasks}")
        lines.append(" & ".join(cells) + " \\\\")
        lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")

    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(f"\\caption{{Combined P-Values (Fisher's Method) - {label}}}")
    lines.append(f"\\label{{tab:fisher_{suffix}}}")
    lines.append("\\begin{tabular}{|l|c|c|}")
    lines.append("\\hline")
    lines.append("\\textbf{Comparison} & \\textbf{Combined p-value} & \\textbf{Significant} \\\\")
    lines.append("\\hline")
    for r in fisher:
        a = r["genome_a"].replace("_", "\\_")
        b = r["genome_b"].replace("_", "\\_")
        sig = "Yes" if r["significant"] else "No"
        lines.append(f"{a} vs {b} & {r['combined_p']:.6f} & {sig} \\\\")
        lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    filepath = os.path.join(save_path, "cross_task_significance.tex")
    os.makedirs(save_path, exist_ok=True)
    with open(filepath, "w") as f:
        f.write("\n".join(lines))
    print(f"  LaTeX cross-task significance tables saved to: {filepath}")


# ── Per-task plot functions ──────────────────────────────────────────────────

def _plot_max_fitness(experiment_data, genotypes, out_dir):
    """Max fitness over generations (all genotypes on one figure)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    final_vals = {}

    for genotype in genotypes:
        if genotype not in experiment_data:
            continue
        agg = _aggregate_runs(experiment_data[genotype])
        data = _clean_inf(agg["max_fitness"])
        means = np.nanmean(data, axis=1)
        stds = np.nanstd(data, axis=1)
        gens = np.arange(len(means))
        color = COLORS[genotype]
        label = GENOTYPE_LABELS.get(genotype, genotype)
        ax.plot(gens, means, label=label, color=color)
        ax.fill_between(gens, means - stds, means + stds, alpha=0.2, color=color)

        final_gen = experiment_data[genotype]["generation"].max()
        final = experiment_data[genotype][experiment_data[genotype]["generation"] == final_gen]
        final_vals[genotype] = final["max_fitness"].dropna().values

    apply_axis_style(ax, "Generation", "Max Fitness")
    if "max_fitness" in Y_LIMITS:
        ax.set_ylim(0, Y_LIMITS["max_fitness"])
    set_tick_intervals(ax, x_interval=10, y_interval=5)
    apply_legend(ax)
    ax.grid(True)
    save_figure(fig, out_dir, "max_fitness")

    # Analytical output
    if final_vals:
        print_analysis_header("Max Fitness (Final Generation)")
        print_summary_stats(final_vals, "max fitness")
        print_stat_tests(final_vals)
        print_analysis_footer(os.path.join(out_dir, "max_fitness"))


def _plot_mean_fitness(experiment_data, genotypes, out_dir):
    """Mean fitness over generations (all genotypes on one figure)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    final_vals = {}

    for genotype in genotypes:
        if genotype not in experiment_data:
            continue
        agg = _aggregate_runs(experiment_data[genotype])
        data = _clean_inf(agg["mean_fitness"])
        means = np.nanmean(data, axis=1)
        stds = np.nanstd(data, axis=1)
        gens = np.arange(len(means))
        color = COLORS[genotype]
        label = GENOTYPE_LABELS.get(genotype, genotype)
        ax.plot(gens, means, label=label, color=color)
        ax.fill_between(gens, means - stds, means + stds, alpha=0.2, color=color)

        final_gen = experiment_data[genotype]["generation"].max()
        final = experiment_data[genotype][experiment_data[genotype]["generation"] == final_gen]
        final_vals[genotype] = final["mean_fitness"].dropna().values

    apply_axis_style(ax, "Generation", "Mean Fitness")
    if "mean_fitness" in Y_LIMITS:
        ax.set_ylim(0, Y_LIMITS["mean_fitness"])
    set_tick_intervals(ax, x_interval=10, y_interval=5)
    apply_legend(ax)
    ax.grid(True)
    save_figure(fig, out_dir, "mean_fitness")

    if final_vals:
        print_analysis_header("Mean Fitness (Final Generation)")
        print_summary_stats(final_vals, "mean fitness")
        print_stat_tests(final_vals)
        print_analysis_footer(os.path.join(out_dir, "mean_fitness"))


def _plot_diversity(experiment_data, genotypes, out_dir):
    """Diversity over generations (all genotypes on one figure)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    final_vals = {}

    for genotype in genotypes:
        if genotype not in experiment_data:
            continue
        agg = _aggregate_runs(experiment_data[genotype])
        data = agg["diversity"]
        means = np.nanmean(data, axis=1)
        stds = np.nanstd(data, axis=1)
        valid = ~np.isnan(means)
        gens = np.arange(len(means))
        color = COLORS[genotype]
        label = GENOTYPE_LABELS.get(genotype, genotype)
        if np.any(valid):
            ax.plot(gens[valid], means[valid], label=label, color=color)
            ax.fill_between(gens[valid], (means - stds)[valid],
                           (means + stds)[valid], alpha=0.2, color=color)

        final_gen = experiment_data[genotype]["generation"].max()
        final = experiment_data[genotype][experiment_data[genotype]["generation"] == final_gen]
        final_vals[genotype] = final["diversity"].dropna().values

    apply_axis_style(ax, "Generation", "Diversity")
    if "diversity" in Y_LIMITS:
        ax.set_ylim(0, Y_LIMITS["diversity"])
    set_tick_intervals(ax, x_interval=10)
    apply_legend(ax)
    ax.grid(True)
    save_figure(fig, out_dir, "diversity")

    if final_vals:
        print_analysis_header("Diversity (Final Generation)")
        print_summary_stats(final_vals, "diversity")
        print_stat_tests(final_vals)
        print_analysis_footer(os.path.join(out_dir, "diversity"))


def _plot_median_max_fitness(experiment_data, genotypes, out_dir):
    """Median max fitness with IQR shading (all genotypes)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for genotype in genotypes:
        if genotype not in experiment_data:
            continue
        agg = _aggregate_runs(experiment_data[genotype])
        data = _clean_inf(agg["max_fitness"])
        median = np.nanmedian(data, axis=1)
        q25 = np.nanpercentile(data, 25, axis=1)
        q75 = np.nanpercentile(data, 75, axis=1)
        gens = np.arange(len(median))
        color = COLORS[genotype]
        label = GENOTYPE_LABELS.get(genotype, genotype)
        ax.plot(gens, median, label=label, color=color)
        ax.fill_between(gens, q25, q75, alpha=0.2, color=color)

    apply_axis_style(ax, "Generation", "Max Fitness")
    if "median_max_fitness" in Y_LIMITS:
        ax.set_ylim(0, Y_LIMITS["median_max_fitness"])
    set_tick_intervals(ax, x_interval=10, y_interval=5)
    apply_legend(ax)
    ax.grid(True)
    save_figure(fig, out_dir, "median_max_fitness")


def _plot_median_mean_fitness(experiment_data, genotypes, out_dir):
    """Median mean fitness with IQR shading (all genotypes)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for genotype in genotypes:
        if genotype not in experiment_data:
            continue
        agg = _aggregate_runs(experiment_data[genotype])
        data = _clean_inf(agg["mean_fitness"])
        median = np.nanmedian(data, axis=1)
        q25 = np.nanpercentile(data, 25, axis=1)
        q75 = np.nanpercentile(data, 75, axis=1)
        gens = np.arange(len(median))
        color = COLORS[genotype]
        label = GENOTYPE_LABELS.get(genotype, genotype)
        ax.plot(gens, median, label=label, color=color)
        ax.fill_between(gens, q25, q75, alpha=0.2, color=color)

    apply_axis_style(ax, "Generation", "Mean Fitness")
    if "median_mean_fitness" in Y_LIMITS:
        ax.set_ylim(0, Y_LIMITS["median_mean_fitness"])
    set_tick_intervals(ax, x_interval=10, y_interval=5)
    apply_legend(ax)
    ax.grid(True)
    save_figure(fig, out_dir, "median_mean_fitness")


# ── Fitness improvement summary ─────────────────────────────────────────────

def _print_fitness_improvement(experiment_data, task_name):
    """Print improvement from first to last generation."""
    print_analysis_header(f"Fitness Improvement - {TASK_LABELS.get(task_name, task_name)}")
    print(f"{'Genotype':<20} {'First Gen':>12} {'Last Gen':>12} {'Improvement':>12} {'Pct':>10}")
    print("-" * 70)
    for geno, df in experiment_data.items():
        first = df[df["generation"] == 0]["max_fitness"].mean()
        last_gen = df["generation"].max()
        last = df[df["generation"] == last_gen]["max_fitness"].mean()
        imp = last - first
        if first != 0:
            pct = ((last - first) / abs(first)) * 100
            pct_str = f"{pct:.1f}%"
        else:
            pct_str = "inf"
        label = GENOTYPE_LABELS.get(geno, geno)
        print(f"{label:<20} {first:>12.4f} {last:>12.4f} {imp:>12.4f} {pct_str:>10}")
    print("-" * 70)


# ── Main entry point ────────────────────────────────────────────────────────

def run(base_dir, tasks, genotypes, output_base=None):
    """Run all fitness/diversity plots and statistical analyses.

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

    all_task_significance = {}

    for task in tasks:
        print(f"\n{'=' * 60}")
        print(f"Fitness/Diversity: {TASK_LABELS.get(task, task)}")
        print(f"{'=' * 60}")

        experiment_data = _load_task_data(base_dir, task, genotypes)
        if not experiment_data:
            continue

        out_dir = figures_dir(base_dir, task=task, plotter_name=PLOTTER_NAME)

        # Per-task plots (all genotypes compared)
        _plot_max_fitness(experiment_data, genotypes, out_dir)
        _plot_mean_fitness(experiment_data, genotypes, out_dir)
        _plot_diversity(experiment_data, genotypes, out_dir)
        _plot_median_max_fitness(experiment_data, genotypes, out_dir)
        _plot_median_mean_fitness(experiment_data, genotypes, out_dir)

        # Fitness improvement summary
        _print_fitness_improvement(experiment_data, task)

        # Significance tests
        sig_results = _run_significance_tests(experiment_data, genotypes)
        _print_significance(sig_results, task_name=task)
        _generate_significance_latex(sig_results, out_dir, task_name=task)
        all_task_significance[task] = sig_results

    # Cross-task summary
    if len(all_task_significance) > 1:
        print(f"\n{'=' * 60}")
        print("CROSS-TASK SIGNIFICANCE SUMMARY")
        print(f"{'=' * 60}")
        cross_dir = figures_dir(base_dir, task="cross_task", plotter_name=PLOTTER_NAME)
        _print_cross_task_summary(all_task_significance, genotypes, save_path=cross_dir)


if __name__ == "__main__":
    run(BASE_DIR, TASKS, GENOTYPES)
