"""Shared plotting utilities — uniform style, save helpers, analytical output, and deduplicated drawing functions."""

import os
import sys
import types
from itertools import combinations

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from experimentation.config import COLORS, GENOTYPE_LABELS, DEFAULT_STYLE


# ── Style ─────────────────────────────────────────────────────────────────────

STYLE_NAME = "seaborn-v0_8"

# Module-level style settings (overridden by setup_style)
_style = dict(DEFAULT_STYLE)
_last_config = None


def setup_style(config=None):
    """Apply uniform matplotlib style. Call once at script start.

    Args:
        config: dict with optional 'style' sub-dict to override defaults.
            If None, reuses the most recent explicit config (so plotters
            that call ``setup_style()`` with no args don't wipe out the
            ``run_plotting`` config).
    """
    global _style, _last_config
    if config is None:
        config = _last_config
    else:
        _last_config = config

    _style = dict(DEFAULT_STYLE)
    if config and "style" in config:
        _style.update(config["style"])

    plt.style.use(STYLE_NAME)
    plt.rcParams.update({
        "axes.labelsize": _style["axis_label_fontsize"],
        "xtick.labelsize": _style["tick_label_fontsize"],
        "ytick.labelsize": _style["tick_label_fontsize"],
        "legend.fontsize": _style["legend_fontsize"],
        "figure.dpi": _style["dpi"],
        "savefig.dpi": _style["dpi"],
        "savefig.bbox": "tight",
    })


def save_figure(fig, output_dir, filename):
    """Save figure as PDF and PNG. Strips any existing title. Prints save path."""
    fig.suptitle("")
    for ax in fig.get_axes():
        ax.set_title("")

    os.makedirs(output_dir, exist_ok=True)
    for fmt in ("pdf", "png"):
        path = os.path.join(output_dir, f"{filename}.{fmt}")
        fig.savefig(path, dpi=_style["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.join(output_dir, filename)}.{{pdf,png}}")


def apply_axis_style(ax, xlabel=None, ylabel=None):
    """Apply uniform font sizes to axis labels and ticks."""
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=_style["axis_label_fontsize"])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=_style["axis_label_fontsize"])
    ax.tick_params(labelsize=_style["tick_label_fontsize"])


def set_tick_intervals(ax, x_interval=None, y_interval=None):
    """Force fixed major-tick spacing on the given axes."""
    from matplotlib.ticker import MultipleLocator
    if x_interval is not None:
        ax.xaxis.set_major_locator(MultipleLocator(x_interval))
    if y_interval is not None:
        ax.yaxis.set_major_locator(MultipleLocator(y_interval))


def apply_legend(ax, *args, **kwargs):
    """Create a legend with fontsize from the configured style.

    Forces ``fontsize=_style["legend_fontsize"]`` so plotters don't depend
    on the rcParams fallback, which is unreliable under some matplotlib
    styles.
    """
    kwargs.setdefault("fontsize", _style["legend_fontsize"])
    return ax.legend(*args, **kwargs)


def genotype_legend(ax, genotypes=None):
    """Add standard genotype legend with correct colors and labels."""
    from experimentation.config import GENOTYPES
    genotypes = genotypes or GENOTYPES
    handles = []
    for g in genotypes:
        if g in COLORS:
            handles.append(plt.Line2D([0], [0], color=COLORS[g],
                                      label=GENOTYPE_LABELS.get(g, g), lw=2))
    ax.legend(handles=handles, fontsize=_style["legend_fontsize"])


# ── Data Helpers ──────────────────────────────────────────────────────────────

def aggregate_by_generation(df, value_col):
    """Aggregate a column by generation, returning (mean Series, SE Series)."""
    grouped = df.groupby("generation")[value_col]
    mean = grouped.mean()
    se = grouped.std() / np.sqrt(grouped.count())
    return mean, se


def pad_generations(df, max_generations=40):
    """Pad generation data so all runs span the same number of generations.

    Repeats the last generation's data forward for runs that ended early.
    """
    padded_data = []
    for run_id in df["run"].unique():
        run_data = df[df["run"] == run_id].copy()
        max_gen = run_data["generation"].max()
        if max_gen < max_generations - 1:
            last_gen_data = run_data[run_data["generation"] == max_gen].copy()
            for gen in range(max_gen + 1, max_generations):
                padding = last_gen_data.copy()
                padding["generation"] = gen
                padded_data.append(padding)
        padded_data.append(run_data)
    return pd.concat(padded_data, ignore_index=True)


# ── Analytical Output ─────────────────────────────────────────────────────────

_SEP = "=" * 80


def print_analysis_header(title):
    """Print analysis section header."""
    print(f"\n{_SEP}")
    print(f"ANALYSIS: {title}")
    print(_SEP)


def print_summary_stats(data_by_genotype, metric_label="value"):
    """Print summary statistics table (mean, std, Q1, median, Q3, IQR).

    Args:
        data_by_genotype: dict of {genotype_name: array of values}
        metric_label: label for the metric being summarised
    """
    header = f"{'':>15} {'mean':>8} {'std':>8} {'Q1':>8} {'median':>8} {'Q3':>8} {'IQR':>8}"
    print(f"\nSummary Statistics ({metric_label}):")
    print(header)
    for geno, vals in data_by_genotype.items():
        vals = np.asarray(vals)
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            print(f"{GENOTYPE_LABELS.get(geno, geno):>15} {'—':>8}")
            continue
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        print(f"{GENOTYPE_LABELS.get(geno, geno):>15} "
              f"{np.mean(vals):>8.3f} {np.std(vals):>8.3f} "
              f"{q1:>8.3f} {med:>8.3f} {q3:>8.3f} {q3 - q1:>8.3f}")


def print_trajectory_samples(data_by_genotype, generation_col="generation",
                             value_col="value", n_samples=5):
    """Print trajectory samples at evenly-spaced points along generations.

    Args:
        data_by_genotype: dict of {genotype_name: DataFrame with generation_col and value_col}
        generation_col: name of generation column
        value_col: name of value column
        n_samples: number of sample points along the trajectory
    """
    # Find global generation range
    all_gens = []
    for df in data_by_genotype.values():
        if df is not None and len(df) > 0:
            all_gens.extend(df[generation_col].unique())
    if not all_gens:
        return

    min_gen, max_gen = int(min(all_gens)), int(max(all_gens))
    sample_gens = np.linspace(min_gen, max_gen, n_samples, dtype=int)
    sample_gens = np.unique(sample_gens)

    header = f"{'':>15}" + "".join(f"  gen_{g:<5}" for g in sample_gens)
    print(f"\nTrajectory Samples (mean across runs):")
    print(header)

    for geno, df in data_by_genotype.items():
        if df is None or len(df) == 0:
            print(f"{GENOTYPE_LABELS.get(geno, geno):>15} {'—':>8}")
            continue
        row = f"{GENOTYPE_LABELS.get(geno, geno):>15}"
        gen_means = df.groupby(generation_col)[value_col].mean()
        for g in sample_gens:
            if g in gen_means.index:
                row += f"  {gen_means[g]:>8.3f}"
            else:
                # Find nearest generation
                nearest = gen_means.index[np.argmin(np.abs(gen_means.index - g))]
                row += f"  {gen_means[nearest]:>8.3f}"
        print(row)


def rank_biserial(u_stat, n1, n2):
    """Compute rank-biserial correlation (effect size for Mann-Whitney U).

    Returns a value in [-1, 1]:
      - +1: all values in group 1 exceed all values in group 2
      - -1: all values in group 2 exceed all values in group 1
      -  0: no difference

    Interpretation (Vargha & Delaney A = (r + 1) / 2):
      |r| < 0.2:  negligible
      |r| 0.2–0.5: small
      |r| 0.5–0.8: medium
      |r| > 0.8:  large
    """
    return 2.0 * u_stat / (n1 * n2) - 1.0


def _effect_size_label(r):
    """Return human-readable effect size category from rank-biserial r."""
    ar = abs(r)
    if ar < 0.2:
        return "negligible"
    elif ar < 0.5:
        return "small"
    elif ar < 0.8:
        return "medium"
    else:
        return "large"


def print_stat_tests(data_by_genotype, test_name="Mann-Whitney U"):
    """Print pairwise statistical test results with effect sizes.

    Args:
        data_by_genotype: dict of {genotype_name: array of values}
        test_name: name of the test for display
    """
    genotypes = list(data_by_genotype.keys())
    if len(genotypes) < 2:
        return

    n_tests = len(genotypes) * (len(genotypes) - 1) // 2

    print(f"\nStatistical Tests ({test_name}, Bonferroni corrected, n={n_tests}):")
    for g1, g2 in combinations(genotypes, 2):
        v1 = np.asarray(data_by_genotype[g1])
        v2 = np.asarray(data_by_genotype[g2])
        v1 = v1[~np.isnan(v1)]
        v2 = v2[~np.isnan(v2)]
        if len(v1) < 2 or len(v2) < 2:
            print(f"  {GENOTYPE_LABELS.get(g1, g1)} vs {GENOTYPE_LABELS.get(g2, g2)}: insufficient data")
            continue
        stat, p = mannwhitneyu(v1, v2, alternative="two-sided")
        p_corrected = min(p * n_tests, 1.0)
        sig = " *" if p_corrected < 0.05 else ""
        r = rank_biserial(stat, len(v1), len(v2))
        eff = _effect_size_label(r)
        l1 = GENOTYPE_LABELS.get(g1, g1)
        l2 = GENOTYPE_LABELS.get(g2, g2)
        print(f"  {l1} vs {l2}: p={p_corrected:.4f}{sig}  r={r:+.3f} ({eff})")


def print_analysis_footer(figure_path):
    """Print analysis section footer with figure path."""
    print(f"\nFigure: {figure_path}")
    print(_SEP)


# ── CPPN Visualization ───────────────────────────────────────────────────────

def _lazy_import_cppn_types():
    """Lazy import CPPN types to avoid fcl dependency at module load."""
    # Mock fcl if not available
    if "fcl" not in sys.modules:
        sys.modules["fcl"] = types.ModuleType("fcl")
    from airevolve.evolution_tools.genome_handlers.cppn.network import (
        CPPNNetwork, NodeType, ActivationFunction,
    )
    return CPPNNetwork, NodeType, ActivationFunction


def get_activation_colors():
    """Return ACTIVATION_COLORS dict (lazy-loaded to avoid import at module level)."""
    _, _, ActivationFunction = _lazy_import_cppn_types()
    return {
        ActivationFunction.IDENTITY: "#808080",
        ActivationFunction.SIGMOID: "#FF6B6B",
        ActivationFunction.TANH: "#4ECDC4",
        ActivationFunction.SIN: "#45B7D1",
        ActivationFunction.COS: "#96CEB4",
        ActivationFunction.GAUSSIAN: "#FFEAA7",
        ActivationFunction.ABS: "#DDA0DD",
        ActivationFunction.RELU: "#FF8C00",
        ActivationFunction.STEP: "#8B4513",
    }


def draw_cppn_graph(cppn, ax, output_labels=None):
    """Draw a CPPN network graph on a matplotlib Axes.

    Layered layout: inputs at bottom, outputs at top, hidden in between.
    Nodes colored by activation function, edges colored by weight sign.

    Args:
        cppn: CPPNNetwork object
        ax: matplotlib Axes
        output_labels: list of output node labels (defaults to full 7-output set)
    """
    _, NodeType, _ = _lazy_import_cppn_types()
    activation_colors = get_activation_colors()

    if output_labels is None:
        output_labels = [
            "arm_present", "magnitude", "arm_yaw", "arm_pitch",
            "motor_yaw", "motor_pitch", "direction",
        ]

    G = nx.DiGraph()

    input_nodes = cppn.get_input_nodes()
    output_nodes = cppn.get_output_nodes()
    hidden_nodes = cppn.get_hidden_nodes()
    all_nodes = list(cppn.nodes.values())

    for node in all_nodes:
        label = node.activation.value
        if node.input_label:
            label = node.input_label
        elif node.node_type == NodeType.OUTPUT and node.output_index is not None:
            if node.output_index < len(output_labels):
                label = output_labels[node.output_index]
        G.add_node(node.node_id, label=label, activation=node.activation,
                   node_type=node.node_type)

    for conn in cppn.get_enabled_connections():
        G.add_edge(conn.source_id, conn.target_id, weight=conn.weight)

    # Layered positioning
    pos = {}
    input_ids = [n.node_id for n in input_nodes]
    output_ids = [n.node_id for n in output_nodes]
    hidden_ids = [n.node_id for n in hidden_nodes]

    for i, nid in enumerate(input_ids):
        pos[nid] = (i / max(len(input_ids) - 1, 1), 0.0)
    for i, nid in enumerate(output_ids):
        pos[nid] = (i / max(len(output_ids) - 1, 1), 1.0)
    if hidden_ids:
        n_hidden = len(hidden_ids)
        n_layers = min(n_hidden, 3)
        for i, nid in enumerate(hidden_ids):
            layer = (i % n_layers) + 1
            y = layer / (n_layers + 1)
            col_in_layer = i // n_layers
            total_in_layer = (n_hidden + n_layers - 1) // n_layers
            x = col_in_layer / max(total_in_layer - 1, 1)
            pos[nid] = (x, y)

    # Draw edges
    for u, v, data in G.edges(data=True):
        w = data["weight"]
        color = "#2196F3" if w >= 0 else "#F44336"
        width = min(abs(w) * 0.8, 4.0)
        ax.annotate(
            "", xy=pos[v], xytext=pos[u],
            arrowprops=dict(
                arrowstyle="->", color=color, lw=max(width, 0.5),
                connectionstyle="arc3,rad=0.1", alpha=0.7,
            ),
        )

    # Draw nodes
    for node in all_nodes:
        nid = node.node_id
        if nid not in pos:
            continue
        x, y = pos[nid]
        color = activation_colors.get(node.activation, "#808080")
        circle = plt.Circle((x, y), 0.04, color=color, ec="black", lw=1.0, zorder=5)
        ax.add_patch(circle)
        label = G.nodes[nid].get("label", str(nid))
        ax.text(x, y - 0.07, label, ha="center", va="top", fontsize=6, zorder=6)

    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.15, 1.15)
    ax.set_aspect("equal")
    ax.axis("off")
