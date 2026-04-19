"""Generate a combined LaTeX table of last-generation genome diversity statistics
from both MuPlusLambda and NEAT experiment configs.

Same format as the fitness table: algorithm headers, bold best values,
superscript letters for significant pairwise differences.
"""

import os
import sys
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experimentation.config import load_config, GENOTYPE_LABELS, TASK_LABELS
from experimentation.plotters.genome_diversity import _load_genome_diversity, _load_phenotypic_diversity
from experimentation.plot_utils import aggregate_by_generation

_SUPERSCRIPT_LETTERS = {
    "Direct": "D",
    "CPPN": "C",
    "Hybrid": "H",
}


def _last_gen_diversity(base_dir, task, genotypes):
    """Extract last-generation genome diversity per-run values per genotype."""
    stats = {}
    for genotype in genotypes:
        gdf = _load_genome_diversity(base_dir, task, genotype)
        if gdf is None:
            continue
        max_gen = gdf["generation"].max()
        final = gdf[gdf["generation"] == max_gen]
        vals = final["genome_diversity_mean"].dropna().values
        stats[genotype] = vals
    return stats


def _last_gen_phenotypic_diversity(base_dir, task, genotypes):
    """Extract last-generation phenotypic diversity per-run values per genotype."""
    stats = {}
    for genotype in genotypes:
        pdf = _load_phenotypic_diversity(base_dir, task, genotype)
        if pdf is None:
            continue
        max_gen = pdf["generation"].max()
        final = pdf[pdf["generation"] == max_gen]
        vals = final["diversity_col"].dropna().values
        stats[genotype] = vals
    return stats


def _pairwise_significance(stats, genotypes):
    """Pairwise Mann-Whitney U with Bonferroni correction."""
    available = [g for g in genotypes if g in stats]
    pairs = list(combinations(available, 2))
    n_comparisons = len(pairs)

    results = {}
    for a, b in pairs:
        va = stats[a][~np.isnan(stats[a])]
        vb = stats[b][~np.isnan(stats[b])]
        if len(va) < 2 or len(vb) < 2:
            results[(a, b)] = {
                "p_corrected": np.nan, "significant": False,
                "median_a": np.nan, "median_b": np.nan,
            }
            continue
        u_stat, p_value = mannwhitneyu(va, vb, alternative="two-sided")
        p_corr = min(p_value * n_comparisons, 1.0)
        results[(a, b)] = {
            "p_corrected": p_corr,
            "significant": p_corr < 0.05,
            "median_a": np.median(va),
            "median_b": np.median(vb),
        }
    return results


def _sig_superscripts(genotype, sig_results):
    """Build superscript string for genotypes this one significantly exceeds."""
    beaten = []
    for (a, b), res in sig_results.items():
        if not res["significant"]:
            continue
        if genotype == a and res["median_a"] > res["median_b"]:
            label_b = GENOTYPE_LABELS.get(b, b)
            beaten.append(_SUPERSCRIPT_LETTERS.get(label_b, label_b[0]))
        elif genotype == b and res["median_b"] > res["median_a"]:
            label_a = GENOTYPE_LABELS.get(a, a)
            beaten.append(_SUPERSCRIPT_LETTERS.get(label_a, label_a[0]))
    return ",".join(sorted(beaten))


def _fmt_val(value, std, is_best, superscripts=""):
    """Format a median +/- std cell with optional bold and superscripts."""
    sup = f"$^{{{superscripts}}}$" if superscripts else ""
    if is_best:
        return f"\\textbf{{{value:.2f}}}{sup} & \\textbf{{{std:.2f}}}"
    return f"{value:.2f}{sup} & {std:.2f}"


def generate_table(config_paths):
    """Load both configs and generate a combined genome diversity LaTeX table."""
    configs = []
    for path in config_paths:
        cfg = load_config(path)
        configs.append(cfg)

    # Collect data
    rows = []
    all_genomic_sig = {}
    all_phenotypic_sig = {}

    for cfg in configs:
        base_dir = cfg["base_dir"]
        tasks = cfg["tasks"]
        genotypes = cfg["genotypes"]
        algo_label = cfg["figures_subdir"].replace("figures_", "")

        for task in tasks:
            genomic_stats = _last_gen_diversity(base_dir, task, genotypes)
            phenotypic_stats = _last_gen_phenotypic_diversity(base_dir, task, genotypes)

            if genomic_stats:
                genomic_sig = _pairwise_significance(genomic_stats, genotypes)
                all_genomic_sig[(algo_label, task)] = genomic_sig

            if phenotypic_stats:
                phenotypic_sig = _pairwise_significance(phenotypic_stats, genotypes)
                all_phenotypic_sig[(algo_label, task)] = phenotypic_sig

            for genotype in genotypes:
                gvals = genomic_stats.get(genotype, np.array([]))
                pvals = phenotypic_stats.get(genotype, np.array([]))
                if len(gvals) == 0 and len(pvals) == 0:
                    continue
                rows.append({
                    "algorithm": algo_label,
                    "task": task,
                    "genotype": genotype,
                    "label": GENOTYPE_LABELS.get(genotype, genotype),
                    "median_genomic": np.median(gvals) if len(gvals) > 0 else np.nan,
                    "std_genomic": np.std(gvals) if len(gvals) > 0 else np.nan,
                    "median_phenotypic": np.median(pvals) if len(pvals) > 0 else np.nan,
                    "std_phenotypic": np.std(pvals) if len(pvals) > 0 else np.nan,
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
                "median_genomic": task_rows["median_genomic"].max(),
                "median_phenotypic": task_rows["median_phenotypic"].max(),
            }

    # Check if phenotypic data exists at all
    has_phenotypic = df["median_phenotypic"].notna().any()

    # ── Build table ──
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")

    caption = ("Final generation diversity statistics. "
               "Bold indicates highest per group. "
               "Superscripts denote genotypes significantly exceeded "
               "(Bonferroni-corrected Mann-Whitney U, $p < 0.05$): "
               "\\textsuperscript{D}\\,=\\,Direct, "
               "\\textsuperscript{C}\\,=\\,CPPN, "
               "\\textsuperscript{H}\\,=\\,Hybrid.")
    lines.append(f"\\caption{{{caption}}}")
    lines.append("\\label{tab:combined_diversity_stats}")
    lines.append("\\small")

    if has_phenotypic:
        n_cols = 8
        lines.append("\\begin{tabular}{ll r@{$\\,\\pm\\,$}l r@{$\\,\\pm\\,$}l}")
        lines.append("\\toprule")
        lines.append("\\textbf{Task} & \\textbf{Genotype} & "
                      "\\multicolumn{2}{c}{\\textbf{Genomic}} & "
                      "\\multicolumn{2}{c}{\\textbf{Phenotypic}} \\\\")
    else:
        n_cols = 4
        lines.append("\\begin{tabular}{ll r@{$\\,\\pm\\,$}l}")
        lines.append("\\toprule")
        lines.append("\\textbf{Task} & \\textbf{Genotype} & "
                      "\\multicolumn{2}{c}{\\textbf{Genomic Diversity}} \\\\")

    n_total_cols = 6 if has_phenotypic else 4

    for algo, genotypes in algos_ordered:
        algo_rows = df[df["algorithm"] == algo]
        lines.append("\\midrule")
        lines.append(f"\\multicolumn{{{n_total_cols}}}{{l}}{{\\textbf{{{algo}}}}} \\\\")
        lines.append("\\midrule")

        for task in tasks_ordered:
            task_rows = algo_rows[algo_rows["task"] == task]
            if task_rows.empty:
                continue

            best_genomic = best[(algo, task)]["median_genomic"]
            best_phenotypic = best[(algo, task)]["median_phenotypic"]

            first_task = True
            for _, row in task_rows.iterrows():
                task_cell = TASK_LABELS.get(task, task) if first_task else ""

                # Genomic significance
                key = (algo, task)
                genomic_sig = all_genomic_sig.get(key, {})
                sup_genomic = _sig_superscripts(row["genotype"], genomic_sig)

                genomic_str = _fmt_val(row["median_genomic"], row["std_genomic"],
                                       row["median_genomic"] == best_genomic,
                                       sup_genomic)

                if has_phenotypic:
                    phenotypic_sig = all_phenotypic_sig.get(key, {})
                    sup_phenotypic = _sig_superscripts(row["genotype"], phenotypic_sig)

                    if np.isnan(row["median_phenotypic"]):
                        pheno_str = "--- & ---"
                    else:
                        pheno_str = _fmt_val(row["median_phenotypic"],
                                             row["std_phenotypic"],
                                             row["median_phenotypic"] == best_phenotypic,
                                             sup_phenotypic)

                    lines.append(
                        f"{task_cell} & {row['label']} & {genomic_str} & {pheno_str} \\\\")
                else:
                    lines.append(
                        f"{task_cell} & {row['label']} & {genomic_str} \\\\")
                first_task = False

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    latex = "\n".join(lines)

    out_path = os.path.join(os.path.dirname(__file__), "combined_genome_diversity_table.tex")
    with open(out_path, "w") as f:
        f.write(latex)
    print(f"Combined genome diversity LaTeX table saved to: {out_path}")
    print()
    print(latex)


def generate_cross_algorithm_table(config_paths):
    """Generate a table comparing MuPlusLambda vs NEAT per genotype and task.

    For each (task, genotype label) pair, compares last-generation genomic and
    phenotypic diversity between the two algorithms using Mann-Whitney U.
    """
    configs = []
    for path in config_paths:
        cfg = load_config(path)
        configs.append(cfg)

    # Collect raw per-run values keyed by (algo_label, task, genotype_label)
    raw_genomic = {}   # -> np.array of per-run values
    raw_phenotypic = {}

    for cfg in configs:
        base_dir = cfg["base_dir"]
        tasks = cfg["tasks"]
        genotypes = cfg["genotypes"]
        algo_label = cfg["figures_subdir"].replace("figures_", "")

        for task in tasks:
            genomic_stats = _last_gen_diversity(base_dir, task, genotypes)
            phenotypic_stats = _last_gen_phenotypic_diversity(base_dir, task, genotypes)

            for genotype in genotypes:
                geno_label = GENOTYPE_LABELS.get(genotype, genotype)
                if genotype in genomic_stats:
                    raw_genomic[(algo_label, task, geno_label)] = genomic_stats[genotype]
                if genotype in phenotypic_stats:
                    raw_phenotypic[(algo_label, task, geno_label)] = phenotypic_stats[genotype]

    # Determine ordering
    tasks_ordered = []
    seen = set()
    for cfg in configs:
        for t in cfg["tasks"]:
            if t not in seen:
                tasks_ordered.append(t)
                seen.add(t)

    algo_labels = []
    seen = set()
    for cfg in configs:
        label = cfg["figures_subdir"].replace("figures_", "")
        if label not in seen:
            algo_labels.append(label)
            seen.add(label)

    geno_labels_ordered = []
    seen = set()
    for cfg in configs:
        for g in cfg["genotypes"]:
            gl = GENOTYPE_LABELS.get(g, g)
            if gl not in seen:
                geno_labels_ordered.append(gl)
                seen.add(gl)

    if len(algo_labels) < 2:
        print("Need at least two algorithms for cross-algorithm comparison!")
        return

    algo_a, algo_b = algo_labels[0], algo_labels[1]

    # Build table rows
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Cross-algorithm diversity comparison ("
                 f"{algo_a} vs {algo_b}). "
                 "Each cell shows median $\\pm$ std. "
                 "Bold indicates significantly higher (Mann-Whitney U, $p < 0.05$). "
                 "Comparisons are per genotype label across algorithms.}")
    lines.append("\\label{tab:cross_algo_diversity}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{ll"
                 " r@{$\\,\\pm\\,$}l r@{$\\,\\pm\\,$}l c"
                 " r@{$\\,\\pm\\,$}l r@{$\\,\\pm\\,$}l c}")
    lines.append("\\toprule")
    lines.append(f"& & \\multicolumn{{5}}{{c}}{{\\textbf{{Genomic Diversity}}}}"
                 f" & \\multicolumn{{5}}{{c}}{{\\textbf{{Phenotypic Diversity}}}} \\\\")
    lines.append("\\cmidrule(lr){3-7} \\cmidrule(lr){8-12}")
    lines.append(f"\\textbf{{Task}} & \\textbf{{Genotype}}"
                 f" & \\multicolumn{{2}}{{c}}{{{algo_a}}}"
                 f" & \\multicolumn{{2}}{{c}}{{{algo_b}}}"
                 f" & \\textbf{{Sig.}}"
                 f" & \\multicolumn{{2}}{{c}}{{{algo_a}}}"
                 f" & \\multicolumn{{2}}{{c}}{{{algo_b}}}"
                 f" & \\textbf{{Sig.}} \\\\")
    lines.append("\\midrule")

    for task in tasks_ordered:
        first_task = True
        for geno_label in geno_labels_ordered:
            task_cell = TASK_LABELS.get(task, task) if first_task else ""

            # Genomic comparison
            ga = raw_genomic.get((algo_a, task, geno_label))
            gb = raw_genomic.get((algo_b, task, geno_label))
            genomic_str = _cross_algo_cells(ga, gb)

            # Phenotypic comparison
            pa = raw_phenotypic.get((algo_a, task, geno_label))
            pb = raw_phenotypic.get((algo_b, task, geno_label))
            pheno_str = _cross_algo_cells(pa, pb)

            lines.append(f"{task_cell} & {geno_label} & {genomic_str} & {pheno_str} \\\\")
            first_task = False
        lines.append("\\midrule")

    # Replace last \midrule with \bottomrule
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    latex = "\n".join(lines)

    out_path = os.path.join(os.path.dirname(__file__), "cross_algorithm_diversity_table.tex")
    with open(out_path, "w") as f:
        f.write(latex)
    print(f"Cross-algorithm diversity table saved to: {out_path}")
    print()
    print(latex)


def _cross_algo_cells(vals_a, vals_b):
    """Format two algorithm cells + significance marker for one metric.

    Returns string for 5 columns: median_a ± std_a, median_b ± std_b, sig_marker.
    The significantly higher value is bolded.
    """
    if vals_a is None or vals_b is None:
        return "--- & --- & --- & --- & ---"

    va = vals_a[~np.isnan(vals_a)]
    vb = vals_b[~np.isnan(vals_b)]

    if len(va) < 2 or len(vb) < 2:
        med_a = np.median(va) if len(va) > 0 else float("nan")
        std_a = np.std(va) if len(va) > 0 else float("nan")
        med_b = np.median(vb) if len(vb) > 0 else float("nan")
        std_b = np.std(vb) if len(vb) > 0 else float("nan")
        return (f"{med_a:.2f} & {std_a:.2f} & {med_b:.2f} & {std_b:.2f} & ---")

    med_a, std_a = np.median(va), np.std(va)
    med_b, std_b = np.median(vb), np.std(vb)

    u_stat, p_value = mannwhitneyu(va, vb, alternative="two-sided")
    significant = p_value < 0.05

    if significant:
        a_wins = med_a > med_b
        if a_wins:
            a_str = f"\\textbf{{{med_a:.2f}}} & \\textbf{{{std_a:.2f}}}"
            b_str = f"{med_b:.2f} & {std_b:.2f}"
        else:
            a_str = f"{med_a:.2f} & {std_a:.2f}"
            b_str = f"\\textbf{{{med_b:.2f}}} & \\textbf{{{std_b:.2f}}}"

        if p_value < 0.001:
            marker = "$\\ast\\ast\\ast$"
        elif p_value < 0.01:
            marker = "$\\ast\\ast$"
        else:
            marker = "$\\ast$"
    else:
        a_str = f"{med_a:.2f} & {std_a:.2f}"
        b_str = f"{med_b:.2f} & {std_b:.2f}"
        marker = ""

    return f"{a_str} & {b_str} & {marker}"


def generate_divergence_table(config_paths):
    """Generate a table showing median diversity for both algorithms + delta.

    For each (task, genotype label), shows:
      - MuPlusLambda median
      - NEAT median
      - Delta (NEAT - MuPlusLambda)
    Bold on the higher median of the two algorithms.
    """
    configs = []
    for path in config_paths:
        cfg = load_config(path)
        configs.append(cfg)

    # Collect raw per-run values keyed by (algo_label, task, genotype_label)
    raw_genomic = {}
    raw_phenotypic = {}

    for cfg in configs:
        base_dir = cfg["base_dir"]
        tasks = cfg["tasks"]
        genotypes = cfg["genotypes"]
        algo_label = cfg["figures_subdir"].replace("figures_", "")

        for task in tasks:
            genomic_stats = _last_gen_diversity(base_dir, task, genotypes)
            phenotypic_stats = _last_gen_phenotypic_diversity(base_dir, task, genotypes)

            for genotype in genotypes:
                geno_label = GENOTYPE_LABELS.get(genotype, genotype)
                if genotype in genomic_stats:
                    raw_genomic[(algo_label, task, geno_label)] = genomic_stats[genotype]
                if genotype in phenotypic_stats:
                    raw_phenotypic[(algo_label, task, geno_label)] = phenotypic_stats[genotype]

    # Determine ordering
    tasks_ordered = []
    seen = set()
    for cfg in configs:
        for t in cfg["tasks"]:
            if t not in seen:
                tasks_ordered.append(t)
                seen.add(t)

    algo_labels = []
    seen = set()
    for cfg in configs:
        label = cfg["figures_subdir"].replace("figures_", "")
        if label not in seen:
            algo_labels.append(label)
            seen.add(label)

    geno_labels_ordered = []
    seen = set()
    for cfg in configs:
        for g in cfg["genotypes"]:
            gl = GENOTYPE_LABELS.get(g, g)
            if gl not in seen:
                geno_labels_ordered.append(gl)
                seen.add(gl)

    if len(algo_labels) < 2:
        print("Need at least two algorithms for divergence table!")
        return

    algo_a, algo_b = algo_labels[0], algo_labels[1]

    def _comparison_cells(vals_a, vals_b):
        """Format: median_a & median_b & delta for one metric. Bold the higher."""
        if vals_a is None or vals_b is None:
            return "--- & --- & ---"

        va = vals_a[~np.isnan(vals_a)]
        vb = vals_b[~np.isnan(vals_b)]

        if len(va) == 0 or len(vb) == 0:
            return "--- & --- & ---"

        med_a, med_b = np.median(va), np.median(vb)
        diff = med_b - med_a
        sign = "+" if diff > 0 else ""

        if med_a > med_b:
            a_str = f"\\textbf{{{med_a:.2f}}}"
            b_str = f"{med_b:.2f}"
        elif med_b > med_a:
            a_str = f"{med_a:.2f}"
            b_str = f"\\textbf{{{med_b:.2f}}}"
        else:
            a_str = f"{med_a:.2f}"
            b_str = f"{med_b:.2f}"

        return f"{a_str} & {b_str} & {sign}{diff:.2f}"

    # Build table
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(f"\\caption{{Cross-algorithm diversity comparison. "
                 f"Median final-generation diversity for {algo_a} and {algo_b}, "
                 f"with $\\Delta = $ {algo_b} $-$ {algo_a}. "
                 f"Bold indicates the higher value.}}")
    lines.append("\\label{tab:diversity_divergence}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{ll ccc ccc}")
    lines.append("\\toprule")
    lines.append("& & \\multicolumn{3}{c}{\\textbf{Genomic Diversity}}"
                 " & \\multicolumn{3}{c}{\\textbf{Phenotypic Diversity}} \\\\")
    lines.append("\\cmidrule(lr){3-5} \\cmidrule(lr){6-8}")
    lines.append(f"\\textbf{{Task}} & \\textbf{{Genotype}}"
                 f" & {algo_a} & {algo_b} & $\\Delta$"
                 f" & {algo_a} & {algo_b} & $\\Delta$ \\\\")
    lines.append("\\midrule")

    for task in tasks_ordered:
        first_task = True
        for geno_label in geno_labels_ordered:
            task_cell = TASK_LABELS.get(task, task) if first_task else ""

            ga = raw_genomic.get((algo_a, task, geno_label))
            gb = raw_genomic.get((algo_b, task, geno_label))
            genomic_str = _comparison_cells(ga, gb)

            pa = raw_phenotypic.get((algo_a, task, geno_label))
            pb = raw_phenotypic.get((algo_b, task, geno_label))
            pheno_str = _comparison_cells(pa, pb)

            lines.append(f"{task_cell} & {geno_label} & {genomic_str} & {pheno_str} \\\\")
            first_task = False
        lines.append("\\midrule")

    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    latex = "\n".join(lines)

    out_path = os.path.join(os.path.dirname(__file__), "cross_algorithm_diversity_table.tex")
    # Append to the same file as the cross-algorithm table
    with open(out_path, "a") as f:
        f.write("\n\n")
        f.write(latex)
    print(f"Diversity divergence table appended to: {out_path}")
    print()
    print(latex)


if __name__ == "__main__":
    config_dir = os.path.dirname(os.path.abspath(__file__))
    config_paths = [
        os.path.join(config_dir, "experiment_config.yaml"),
        os.path.join(config_dir, "experiment_config_neat.yaml"),
    ]
    generate_table(config_paths)
    print("\n" + "=" * 80 + "\n")
    generate_cross_algorithm_table(config_paths)
    print("\n" + "=" * 80 + "\n")
    generate_divergence_table(config_paths)
