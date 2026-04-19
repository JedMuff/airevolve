"""Generate the paper's combined fitness table.

Layout: Task+Encoding as rows, Non-speciated / Speciated column groups
(each with Med. Max and Med. Mean). Converged-only (max fitness > 5).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experimentation.config import load_config, GENOTYPE_LABELS
from experimentation.plotters.fitness_diversity import _load_task_data
from experimentation.generate_combined_fitness_table import (
    _last_gen_stats,
    _pairwise_significance,
    _SUPERSCRIPT_LETTERS,
    CONVERGENCE_THRESHOLD,
)


TASK_SHORT = {
    "backandforth": "Shuttle",
    "figure8": "Fig.\\ 8",
    "circle": "Circle",
}

CANONICAL = ["spherical", "cppn", "hybrid_cppn"]
ENC_LABEL = {"spherical": "Direct", "cppn": "CPPN", "hybrid_cppn": "Hybrid"}
TASKS = ["backandforth", "figure8", "circle"]


def _sup(sig, genotype, metric):
    beaten = []
    for (a, b), res in sig[metric].items():
        if not res["significant"]:
            continue
        va, vb = res["median_a"], res["median_b"]
        if genotype == a and va > vb:
            lbl = GENOTYPE_LABELS.get(b, b)
            beaten.append(_SUPERSCRIPT_LETTERS.get(lbl, lbl[0]))
        elif genotype == b and vb > va:
            lbl = GENOTYPE_LABELS.get(a, a)
            beaten.append(_SUPERSCRIPT_LETTERS.get(lbl, lbl[0]))
    return ",".join(sorted(beaten))


def _cell(med, std, is_best, sups):
    sup = f"$^{{{sups}}}$" if sups else ""
    if is_best:
        return f"\\textbf{{{med:.2f}}}{sup}$\\pm$\\textbf{{{std:.2f}}}"
    return f"{med:.2f}{sup}$\\pm${std:.2f}"


def _algo_genotype(algo_label, canon):
    return canon if algo_label == "Non-speciated" else f"neat_{canon}"


def main():
    cfg_dir = os.path.dirname(os.path.abspath(__file__))
    mpl = load_config(os.path.join(cfg_dir, "experiment_config.yaml"))
    neat = load_config(os.path.join(cfg_dir, "experiment_config_neat.yaml"))
    algos = [("Non-speciated", mpl), ("Speciated", neat)]

    data = {}
    for algo_label, cfg in algos:
        for task in TASKS:
            exp = _load_task_data(cfg["base_dir"], task, cfg["genotypes"])
            stats = _last_gen_stats(exp, cfg["genotypes"])
            sig = _pairwise_significance(stats, cfg["genotypes"])
            data[(algo_label, task)] = (stats, sig)

    lines = []
    lines.append(r"\begin{table}[hbt!]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Final generation fitness statistics (converged runs only, "
        rf"max fitness $> {CONVERGENCE_THRESHOLD:.0f}$). "
        r"Bold indicates best per group. Superscripts denote encodings "
        r"significantly outperformed (Bonferroni-corrected, Mann-Whitney U, "
        r"$p < 0.05$): \textsuperscript{D}\,=\,Direct, "
        r"\textsuperscript{C}\,=\,CPPN, \textsuperscript{H}\,=\,Hybrid.}"
    )
    lines.append(r"\label{tab:combined_fitness_stats}")
    lines.append(r"\footnotesize")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.0}")
    lines.append(r"\begin{tabular}{ll cc cc}")
    lines.append(r"\toprule")
    lines.append(
        r" & & \multicolumn{2}{c}{\textbf{Non-speciated}} "
        r"& \multicolumn{2}{c}{\textbf{Speciated}} \\"
    )
    lines.append(r"\cmidrule(lr){3-4} \cmidrule(lr){5-6}")
    lines.append(
        r"\textbf{Task} & \textbf{Enc.} & \textbf{Med. Max} & \textbf{Med. Mean} "
        r"& \textbf{Med. Max} & \textbf{Med. Mean} \\"
    )
    lines.append(r"\midrule")

    for ti, task in enumerate(TASKS):
        if ti > 0:
            lines.append(r"\addlinespace")

        bests = {}
        for algo_label, _ in algos:
            stats, _sig = data[(algo_label, task)]
            maxes, means = [], []
            for canon in CANONICAL:
                g = _algo_genotype(algo_label, canon)
                s = stats.get(g)
                if s is None or len(s["max_fitness_converged"]) == 0:
                    maxes.append(-np.inf)
                    means.append(-np.inf)
                    continue
                maxes.append(np.median(s["max_fitness_converged"]))
                means.append(np.median(s["mean_fitness_converged"]))
            bests[(algo_label, "max")] = max(maxes)
            bests[(algo_label, "mean")] = max(means)

        for ci, canon in enumerate(CANONICAL):
            task_cell = TASK_SHORT[task] if ci == 0 else ""
            row = [task_cell, ENC_LABEL[canon]]
            for algo_label, _ in algos:
                stats, sig = data[(algo_label, task)]
                g = _algo_genotype(algo_label, canon)
                s = stats.get(g)
                if s is None or len(s["max_fitness_converged"]) == 0:
                    row.extend(["---", "---"])
                    continue
                med_max = np.median(s["max_fitness_converged"])
                std_max = np.std(s["max_fitness_converged"])
                med_mean = np.median(s["mean_fitness_converged"])
                std_mean = np.std(s["mean_fitness_converged"])
                sup_max = _sup(sig, g, "max_fitness_converged")
                sup_mean = _sup(sig, g, "mean_fitness_converged")
                row.append(_cell(med_max, std_max,
                                 med_max == bests[(algo_label, "max")], sup_max))
                row.append(_cell(med_mean, std_mean,
                                 med_mean == bests[(algo_label, "mean")], sup_mean))
            lines.append(" & ".join(row) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    latex = "\n".join(lines)
    out_path = os.path.join(cfg_dir, "paper_fitness_table.tex")
    with open(out_path, "w") as f:
        f.write(latex)
    print(f"Saved to: {out_path}\n")
    print(latex)


if __name__ == "__main__":
    main()
