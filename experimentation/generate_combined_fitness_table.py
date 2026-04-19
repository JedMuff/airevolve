"""Generate a combined LaTeX table of last-generation fitness statistics
from both MuPlusLambda and NEAT experiment configs.

Produces a single table with algorithm as section headers, best values bolded,
and superscript letters indicating which other genotypes each entry is
significantly better than (Bonferroni-corrected Mann-Whitney U, p < 0.05).

Fitness statistics are computed only on converged runs (max fitness > threshold).
A separate convergence rate column shows what fraction of runs converged.
"""

import os
import sys
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, fisher_exact

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experimentation.config import load_config, GENOTYPE_LABELS, TASK_LABELS
from experimentation.plotters.fitness_diversity import _load_task_data, _aggregate_runs, _clean_inf
from experimentation.plot_utils import rank_biserial, _effect_size_label

# Short superscript letters for each genotype label
_SUPERSCRIPT_LETTERS = {
    "Direct": "D",
    "CPPN": "C",
    "Hybrid": "H",
}

# Runs with final-generation max fitness below this are considered failed
CONVERGENCE_THRESHOLD = 5.0


def _last_gen_stats(experiment_data, genotypes):
    """Extract last-generation per-run fitness arrays per genotype.

    Returns both full arrays and converged-only arrays.
    """
    stats = {}
    for genotype in genotypes:
        if genotype not in experiment_data:
            continue
        df = experiment_data[genotype]
        final_gen = df["generation"].max()
        final = df[df["generation"] == final_gen]
        max_vals = final["max_fitness"].dropna().values
        mean_vals = final["mean_fitness"].dropna().values

        converged_mask = max_vals > CONVERGENCE_THRESHOLD
        n_total = len(max_vals)
        n_converged = int(converged_mask.sum())

        stats[genotype] = {
            "max_fitness": max_vals,
            "mean_fitness": mean_vals,
            "max_fitness_converged": max_vals[converged_mask],
            "mean_fitness_converged": mean_vals[converged_mask],
            "n_total": n_total,
            "n_converged": n_converged,
            "convergence_rate": n_converged / n_total if n_total > 0 else 0.0,
        }
    return stats


def _pairwise_significance(stats, genotypes):
    """Pairwise significance on converged-only fitness + convergence rate.

    - Mann-Whitney U (Bonferroni-corrected) on converged runs' fitness.
    - Fisher's exact test on convergence counts.
    """
    available = [g for g in genotypes if g in stats]
    pairs = list(combinations(available, 2))
    n_comparisons = len(pairs)

    results = {}
    # Fitness significance (converged runs only)
    for metric in ("max_fitness_converged", "mean_fitness_converged"):
        results[metric] = {}
        for a, b in pairs:
            va = stats[a][metric]
            vb = stats[b][metric]
            va = va[~np.isnan(va)]
            vb = vb[~np.isnan(vb)]
            if len(va) < 2 or len(vb) < 2:
                results[metric][(a, b)] = {
                    "p_corrected": np.nan, "significant": False,
                    "median_a": np.nan, "median_b": np.nan,
                }
                continue
            u_stat, p_value = mannwhitneyu(va, vb, alternative="two-sided")
            p_corr = min(p_value * n_comparisons, 1.0)
            results[metric][(a, b)] = {
                "p_corrected": p_corr,
                "significant": p_corr < 0.05,
                "median_a": np.median(va),
                "median_b": np.median(vb),
            }

    # Convergence rate significance (Fisher's exact test)
    results["convergence"] = {}
    for a, b in pairs:
        sa, sb = stats[a], stats[b]
        # 2x2 table: [[converged_a, failed_a], [converged_b, failed_b]]
        table = [
            [sa["n_converged"], sa["n_total"] - sa["n_converged"]],
            [sb["n_converged"], sb["n_total"] - sb["n_converged"]],
        ]
        _, p_value = fisher_exact(table)
        p_corr = min(p_value * n_comparisons, 1.0)
        results["convergence"][(a, b)] = {
            "p_corrected": p_corr,
            "significant": p_corr < 0.05,
            "rate_a": sa["convergence_rate"],
            "rate_b": sb["convergence_rate"],
        }

    return results


def _sig_superscripts(algo, task, genotype, metric, all_significance):
    """Build superscript string for genotypes that this genotype significantly beats."""
    key = (algo, task)
    if key not in all_significance:
        return ""
    sig = all_significance[key][metric]
    beaten = []
    for (a, b), res in sig.items():
        if not res["significant"]:
            continue
        if metric == "convergence":
            val_a, val_b = res["rate_a"], res["rate_b"]
        else:
            val_a, val_b = res["median_a"], res["median_b"]
        if genotype == a and val_a > val_b:
            label_b = GENOTYPE_LABELS.get(b, b)
            beaten.append(_SUPERSCRIPT_LETTERS.get(label_b, label_b[0]))
        elif genotype == b and val_b > val_a:
            label_a = GENOTYPE_LABELS.get(a, a)
            beaten.append(_SUPERSCRIPT_LETTERS.get(label_a, label_a[0]))
    return ",".join(sorted(beaten))


def _fmt_val(value, std, is_best, superscripts=""):
    """Format a median +/- std cell with optional bold and superscripts."""
    sup = f"$^{{{superscripts}}}$" if superscripts else ""
    if is_best:
        return f"\\textbf{{{value:.2f}}}{sup} & \\textbf{{{std:.2f}}}"
    return f"{value:.2f}{sup} & {std:.2f}"


def _fmt_rate(n_converged, n_total, is_best, superscripts=""):
    """Format a convergence rate cell as fraction with optional bold and superscripts."""
    sup = f"$^{{{superscripts}}}$" if superscripts else ""
    frac = f"{n_converged}/{n_total}"
    if is_best:
        return f"\\textbf{{{frac}}}{sup}"
    return f"{frac}{sup}"


def generate_table(config_paths):
    """Load both configs and generate a combined LaTeX table."""
    configs = []
    for path in config_paths:
        cfg = load_config(path)
        configs.append(cfg)

    # Collect data
    rows = []
    all_significance = {}

    for cfg in configs:
        base_dir = cfg["base_dir"]
        tasks = cfg["tasks"]
        genotypes = cfg["genotypes"]
        algo_label = cfg["figures_subdir"].replace("figures_", "")

        for task in tasks:
            experiment_data = _load_task_data(base_dir, task, genotypes)
            if not experiment_data:
                continue

            stats = _last_gen_stats(experiment_data, genotypes)
            sig = _pairwise_significance(stats, genotypes)
            all_significance[(algo_label, task)] = sig

            for genotype in genotypes:
                if genotype not in stats:
                    continue
                s = stats[genotype]
                conv = s["max_fitness_converged"]
                conv_mean = s["mean_fitness_converged"]
                rows.append({
                    "algorithm": algo_label,
                    "task": task,
                    "genotype": genotype,
                    "label": GENOTYPE_LABELS.get(genotype, genotype),
                    "median_max": np.median(conv) if len(conv) > 0 else np.nan,
                    "std_max": np.std(conv) if len(conv) > 0 else np.nan,
                    "median_mean": np.median(conv_mean) if len(conv_mean) > 0 else np.nan,
                    "std_mean": np.std(conv_mean) if len(conv_mean) > 0 else np.nan,
                    "n_converged": s["n_converged"],
                    "n_total": s["n_total"],
                    "convergence_rate": s["convergence_rate"],
                })

    if not rows:
        print("No data found!")
        return

    df = pd.DataFrame(rows)

    # Ordered lists
    tasks_ordered = []
    seen = set()
    for cfg in configs:
        for t in cfg["tasks"]:
            if t not in seen:
                tasks_ordered.append(t)
                seen.add(t)

    algos_ordered = []
    seen = set()
    for cfg in configs:
        label = cfg["figures_subdir"].replace("figures_", "")
        if label not in seen:
            algos_ordered.append((label, cfg["genotypes"]))
            seen.add(label)

    # Find best values per (algorithm, task) for bolding
    best = {}
    for algo, _ in algos_ordered:
        algo_rows = df[df["algorithm"] == algo]
        for task in tasks_ordered:
            task_rows = algo_rows[algo_rows["task"] == task]
            if task_rows.empty:
                continue
            best[(algo, task)] = {
                "median_max": task_rows["median_max"].max(),
                "median_mean": task_rows["median_mean"].max(),
                "convergence_rate": task_rows["convergence_rate"].max(),
            }

    # ── Build table ──
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Final generation fitness statistics (converged runs only, "
                  f"max fitness $> {CONVERGENCE_THRESHOLD:.0f}$). "
                  "Bold indicates best per group. "
                  "Superscripts denote genotypes significantly outperformed "
                  "(Bonferroni-corrected): "
                  "\\textsuperscript{D}\\,=\\,Direct, "
                  "\\textsuperscript{C}\\,=\\,CPPN, "
                  "\\textsuperscript{H}\\,=\\,Hybrid. "
                  "Fitness: Mann-Whitney U ($p < 0.05$). "
                  "Conv.: Fisher's exact test ($p < 0.05$).}")
    lines.append("\\label{tab:combined_fitness_stats}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{ll c r@{$\\,\\pm\\,$}l r@{$\\,\\pm\\,$}l}")
    lines.append("\\toprule")
    lines.append("\\textbf{Task} & \\textbf{Genotype} & \\textbf{Conv.} & "
                  "\\multicolumn{2}{c}{\\textbf{Median Max}} & "
                  "\\multicolumn{2}{c}{\\textbf{Median Mean}} \\\\")

    for algo, genotypes in algos_ordered:
        algo_rows = df[df["algorithm"] == algo]
        lines.append("\\midrule")
        lines.append(f"\\multicolumn{{7}}{{l}}{{\\textbf{{{algo}}}}} \\\\")
        lines.append("\\midrule")

        for task in tasks_ordered:
            task_rows = algo_rows[algo_rows["task"] == task]
            if task_rows.empty:
                continue

            best_max = best[(algo, task)]["median_max"]
            best_mean = best[(algo, task)]["median_mean"]
            best_conv = best[(algo, task)]["convergence_rate"]

            first_task = True
            for _, row in task_rows.iterrows():
                task_cell = TASK_LABELS.get(task, task) if first_task else ""

                sup_max = _sig_superscripts(
                    algo, task, row["genotype"], "max_fitness_converged",
                    all_significance)
                sup_mean = _sig_superscripts(
                    algo, task, row["genotype"], "mean_fitness_converged",
                    all_significance)
                sup_conv = _sig_superscripts(
                    algo, task, row["genotype"], "convergence",
                    all_significance)

                conv_str = _fmt_rate(row["n_converged"], row["n_total"],
                                     row["convergence_rate"] == best_conv,
                                     sup_conv)
                max_str = _fmt_val(row["median_max"], row["std_max"],
                                   row["median_max"] == best_max, sup_max)
                mean_str = _fmt_val(row["median_mean"], row["std_mean"],
                                    row["median_mean"] == best_mean, sup_mean)

                lines.append(
                    f"{task_cell} & {row['label']} & {conv_str} & "
                    f"{max_str} & {mean_str} \\\\")
                first_task = False

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    latex = "\n".join(lines)

    # Save
    out_path = os.path.join(os.path.dirname(__file__), "combined_fitness_table.tex")
    with open(out_path, "w") as f:
        f.write(latex)
    print(f"Combined LaTeX table saved to: {out_path}")
    print()
    print(latex)


if __name__ == "__main__":
    config_dir = os.path.dirname(os.path.abspath(__file__))
    config_paths = [
        os.path.join(config_dir, "experiment_config.yaml"),
        os.path.join(config_dir, "experiment_config_neat.yaml"),
    ]
    generate_table(config_paths)
