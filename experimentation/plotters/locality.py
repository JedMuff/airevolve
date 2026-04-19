"""Locality plotter — scatter plots of parent-child distance vs fitness difference.

Locality measures whether small changes in genotype/phenotype produce small
changes in fitness. Produces per-task scatter plots with regression lines and
a cross-task R-squared comparison.
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, spearmanr

from experimentation.config import (
    BASE_DIR, TASKS, GENOTYPES, COLORS, GENOTYPE_LABELS, TASK_LABELS,
    Y_LIMITS, figures_dir,
)
from experimentation.plot_utils import (
    setup_style, save_figure, apply_axis_style,
    print_analysis_header, print_analysis_footer,
    print_summary_stats, print_stat_tests,
)

warnings.filterwarnings("ignore")

PLOTTER_NAME = "locality"


# ── Data loading ─────────────────────────────────────────────────────────────

def _load_locality_data(base_dir, task, genotype):
    """Load locality_data.csv for one experiment."""
    path = os.path.join(base_dir, task, genotype, "locality_data.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if len(df) == 0:
        return None
    return df


# ── Plotting helpers ─────────────────────────────────────────────────────────

def _bin_and_weight(x, y, n_bins=20):
    """Bin data by x, returning bin centers, mean y, and counts per bin.

    Uses equal-width bins across the x range.
    """
    bin_edges = np.linspace(x.min(), x.max(), n_bins + 1)
    bin_indices = np.digitize(x, bin_edges, right=False)
    # Clamp to valid range (digitize can return n_bins+1 for x == x.max())
    bin_indices = np.clip(bin_indices, 1, n_bins)

    centers = []
    means = []
    counts = []
    for b in range(1, n_bins + 1):
        mask = bin_indices == b
        if mask.sum() < 3:
            continue
        centers.append((bin_edges[b - 1] + bin_edges[b]) / 2)
        means.append(np.mean(y[mask]))
        counts.append(mask.sum())

    return np.array(centers), np.array(means), np.array(counts)


def _scatter_with_trend(ax, x, y, color, label, alpha=0.15):
    """Draw scatter with weighted linear regression trend line and annotation.

    The regression is performed on binned data weighted by sample count per bin,
    so dense low-distance regions don't dominate the fit over sparse high-distance
    regions.
    """
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 10:
        return None

    ax.scatter(x, y, color=color, alpha=alpha, s=8, edgecolors="none", rasterized=True)

    # Bin data and fit weighted regression
    centers, means, counts = _bin_and_weight(x, y)
    if len(centers) < 3:
        return None

    weights = counts.astype(float)
    # Weighted least squares: minimize sum(w_i * (y_i - (a + b*x_i))^2)
    W = np.sum(weights)
    Wx = np.sum(weights * centers)
    Wy = np.sum(weights * means)
    Wxx = np.sum(weights * centers ** 2)
    Wxy = np.sum(weights * centers * means)

    denom = W * Wxx - Wx ** 2
    if abs(denom) < 1e-12:
        return None

    slope = (W * Wxy - Wx * Wy) / denom
    intercept = (Wy - slope * Wx) / W

    # Weighted R^2
    y_pred = slope * centers + intercept
    ss_res = np.sum(weights * (means - y_pred) ** 2)
    ss_tot = np.sum(weights * (means - Wy / W) ** 2)
    r_sq = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

    # Unweighted Spearman on raw data for comparison
    rho, sp_p = spearmanr(x, y)

    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, slope * x_line + intercept, color="black", linewidth=2,
            zorder=5, label="Weighted fit")

    ax.annotate(
        f"Weighted $R^2$={r_sq:.3f}\nslope={slope:.3f}\n"
        f"Spearman $\\rho$={rho:.3f}  n={len(x)}",
        xy=(0.03, 0.97), xycoords="axes fraction", fontsize=10, va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
    )

    return {"r_sq": r_sq, "slope": slope, "intercept": intercept,
            "spearman_rho": rho, "spearman_p": sp_p, "n": len(x)}


def _density_with_trend(ax, x, y, color, n_grid=100):
    """Draw filled KDE contour plot with weighted regression trend line and annotation."""
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 10:
        return None

    # KDE on a regular grid
    kde = gaussian_kde(np.vstack([x, y]))
    xi = np.linspace(x.min(), x.max(), n_grid)
    yi = np.linspace(y.min(), y.max(), n_grid)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = kde(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)

    # Power-law transform to compress dense peaks and reveal structure
    Zi_t = Zi ** 0.3
    ax.contourf(Xi, Yi, Zi_t, levels=15, cmap="magma", alpha=0.85)
    ax.contour(Xi, Yi, Zi_t, levels=15, colors="white", linewidths=0.3, alpha=0.5)

    # Bin data and fit weighted regression
    centers, means, counts = _bin_and_weight(x, y)
    if len(centers) < 3:
        return None

    weights = counts.astype(float)
    W = np.sum(weights)
    Wx = np.sum(weights * centers)
    Wy = np.sum(weights * means)
    Wxx = np.sum(weights * centers ** 2)
    Wxy = np.sum(weights * centers * means)

    denom = W * Wxx - Wx ** 2
    if abs(denom) < 1e-12:
        return None

    slope = (W * Wxy - Wx * Wy) / denom
    intercept = (Wy - slope * Wx) / W

    y_pred = slope * centers + intercept
    ss_res = np.sum(weights * (means - y_pred) ** 2)
    ss_tot = np.sum(weights * (means - Wy / W) ** 2)
    r_sq = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

    rho, sp_p = spearmanr(x, y)

    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, slope * x_line + intercept, color="black", linewidth=3.5, zorder=5)
    ax.plot(x_line, slope * x_line + intercept, color="white", linewidth=2, zorder=6)

    ax.annotate(
        f"Weighted $R^2$={r_sq:.3f}\nslope={slope:.3f}\n"
        f"Spearman $\\rho$={rho:.3f}  n={len(x)}",
        xy=(0.03, 0.97), xycoords="axes fraction", fontsize=10, va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
    )

    return {"r_sq": r_sq, "slope": slope, "intercept": intercept,
            "spearman_rho": rho, "spearman_p": sp_p, "n": len(x)}


def _plot_locality_density_panels(task, data_dict, genotypes, out_dir,
                                   distance_col, y_col, xlabel, ylabel, filename):
    """Create a multi-panel density plot (one panel per genotype)."""
    active_genotypes = [g for g in genotypes if g in data_dict and data_dict[g] is not None]
    if not active_genotypes:
        return {}

    n = len(active_genotypes)
    fig, axes = plt.subplots(1, n, figsize=(8 * n, 6), squeeze=False)
    axes = axes[0]

    stats = {}
    for i, genotype in enumerate(active_genotypes):
        ax = axes[i]
        df = data_dict[genotype]
        x = df[distance_col].values.astype(float)
        y = df[y_col].values.astype(float)

        result = _density_with_trend(ax, x, y, COLORS[genotype])
        if result:
            stats[genotype] = result

        apply_axis_style(ax, xlabel=xlabel, ylabel=ylabel if i == 0 else None)
        ax.set_title(GENOTYPE_LABELS[genotype], fontsize=18)

    fig.tight_layout()
    save_figure(fig, out_dir, filename)

    print_analysis_header(f"Locality Density ({filename}) — {task}")
    for genotype, s in stats.items():
        label = GENOTYPE_LABELS.get(genotype, genotype)
        print(f"  {label}: Weighted R^2={s['r_sq']:.4f}  slope={s['slope']:.4f}"
              f"  Spearman rho={s['spearman_rho']:.4f}  n={s['n']}")
    print_analysis_footer(os.path.join(out_dir, filename))

    return stats


def _plot_locality_panels(task, data_dict, genotypes, out_dir,
                          distance_col, y_col, xlabel, ylabel, filename):
    """Create a multi-panel scatter plot (one panel per genotype)."""
    active_genotypes = [g for g in genotypes if g in data_dict and data_dict[g] is not None]
    if not active_genotypes:
        return {}

    n = len(active_genotypes)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6), squeeze=False)
    axes = axes[0]

    stats = {}
    for i, genotype in enumerate(active_genotypes):
        ax = axes[i]
        df = data_dict[genotype]
        x = df[distance_col].values.astype(float)
        y = df[y_col].values.astype(float)

        result = _scatter_with_trend(ax, x, y, COLORS[genotype],
                                     GENOTYPE_LABELS[genotype])
        if result:
            stats[genotype] = result

        apply_axis_style(ax, xlabel=xlabel, ylabel=ylabel if i == 0 else None)
        ax.set_title(GENOTYPE_LABELS[genotype], fontsize=18)

    fig.tight_layout()
    save_figure(fig, out_dir, filename)

    # Analytical output
    print_analysis_header(f"Locality ({filename}) — {task}")
    for genotype, s in stats.items():
        label = GENOTYPE_LABELS.get(genotype, genotype)
        print(f"  {label}: Weighted R^2={s['r_sq']:.4f}  slope={s['slope']:.4f}"
              f"  Spearman rho={s['spearman_rho']:.4f}  n={s['n']}")
    print_analysis_footer(os.path.join(out_dir, filename))

    return stats


# ── Distance distribution ────────────────────────────────────────────────────

def _plot_distance_distribution(task, data_dict, genotypes, out_dir):
    """Violin + box plot of parent-child phenotypic distances per genotype.

    Shows that direct encodings produce children closer to parents.
    """
    active_genotypes = [g for g in genotypes if g in data_dict and data_dict[g] is not None]
    if not active_genotypes:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    plot_data = []
    labels = []
    colors = []
    dist_by_geno = {}

    for genotype in active_genotypes:
        df = data_dict[genotype]
        vals = df["phenotypic_distance"].dropna().values
        if len(vals) == 0:
            continue
        plot_data.append(vals)
        labels.append(GENOTYPE_LABELS[genotype])
        colors.append(COLORS[genotype])
        dist_by_geno[genotype] = vals

    if not plot_data:
        plt.close(fig)
        return

    positions = np.arange(1, len(plot_data) + 1)

    parts = ax.violinplot(plot_data, positions=positions, showmedians=True)
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(colors[i])
        body.set_alpha(0.7)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.grid(True, axis="y", alpha=0.3)
    if "phenotypic_distance" in Y_LIMITS:
        ax.set_ylim(0, Y_LIMITS["phenotypic_distance"])
    apply_axis_style(ax, xlabel="Encoding", ylabel="Parent-Child Phenotypic Distance")

    save_figure(fig, out_dir, "distance_distribution")

    # Analytical output
    print_analysis_header(f"Parent-Child Distance Distribution — {task}")
    print_summary_stats(dist_by_geno, metric_label="phenotypic distance")
    print_stat_tests(dist_by_geno)
    print_analysis_footer(os.path.join(out_dir, "distance_distribution"))


def _plot_distance_distribution_boxplot(task, data_dict, genotypes, out_dir):
    """Boxplot-only version of parent-child phenotypic distance per genotype."""
    active_genotypes = [g for g in genotypes if g in data_dict and data_dict[g] is not None]
    if not active_genotypes:
        return

    plot_data = []
    labels = []
    colors = []

    for genotype in active_genotypes:
        df = data_dict[genotype]
        vals = df["phenotypic_distance"].dropna().values
        if len(vals) == 0:
            continue
        plot_data.append(vals)
        labels.append(GENOTYPE_LABELS[genotype])
        colors.append(COLORS[genotype])

    if not plot_data:
        return

    positions = np.arange(1, len(plot_data) + 1)
    fig, ax = plt.subplots(figsize=(10, 6))

    bp = ax.boxplot(
        plot_data, positions=positions,
        patch_artist=True, showfliers=False,
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.grid(True, axis="y", alpha=0.3)
    if "phenotypic_distance" in Y_LIMITS:
        ax.set_ylim(0, Y_LIMITS["phenotypic_distance"])
    apply_axis_style(ax, xlabel="Encoding", ylabel="Parent-Child Phenotypic Distance")

    save_figure(fig, out_dir, "distance_distribution_boxplot")


# ── Fitness improvement by distance bin ──────────────────────────────────────

def _pool_locality_data(data_dict, active_genotypes):
    """Pool distance and fitness diff data across encodings.

    Returns pooled arrays, per-genotype distances, and per-genotype fitness diffs.
    """
    all_dist = []
    all_fdiff = []
    per_geno_dist = {}
    per_geno_fdiff = {}
    for genotype in active_genotypes:
        df = data_dict[genotype]
        dist = df["phenotypic_distance"].values.astype(float)
        fdiff = df["fitness_diff"].values.astype(float)
        mask = np.isfinite(dist) & np.isfinite(fdiff)
        dist, fdiff = dist[mask], fdiff[mask]
        all_dist.append(dist)
        all_fdiff.append(fdiff)
        per_geno_dist[genotype] = dist
        per_geno_fdiff[genotype] = fdiff

    return np.concatenate(all_dist), np.concatenate(all_fdiff), per_geno_dist, per_geno_fdiff


def _draw_encoding_boxplots(ax_box, active_genotypes, per_geno_dist):
    """Draw horizontal box plots showing each encoding's distance distribution."""
    box_data = []
    box_labels = []
    box_colors = []
    for genotype in active_genotypes:
        if genotype in per_geno_dist and len(per_geno_dist[genotype]) > 0:
            box_data.append(per_geno_dist[genotype])
            box_labels.append(GENOTYPE_LABELS[genotype])
            box_colors.append(COLORS[genotype])

    if box_data:
        positions = np.arange(1, len(box_data) + 1)
        bp = ax_box.boxplot(
            box_data, positions=positions, vert=False, widths=0.6,
            patch_artist=True, showfliers=False,
            medianprops=dict(color="black", linewidth=2),
            whiskerprops=dict(linewidth=1.5),
            capprops=dict(linewidth=1.5),
        )
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax_box.set_yticks(positions)
        ax_box.set_yticklabels(box_labels)
        apply_axis_style(ax_box, xlabel="Phenotypic Distance")


def _bin_data(dist, fdiff, bin_edges):
    """Bin data into equal-width bins. Returns (centers, values_per_bin) list."""
    n_bins = len(bin_edges) - 1
    result = []
    for j in range(n_bins):
        lo, hi = bin_edges[j], bin_edges[j + 1]
        in_bin = (dist >= lo) & (dist < hi) if j < n_bins - 1 \
            else (dist >= lo) & (dist <= hi)
        center = (lo + hi) / 2
        result.append((center, fdiff[in_bin]))
    return result


def _plot_improvement_by_distance(task, data_dict, genotypes, out_dir,
                                  filename="improvement_by_distance"):
    """Pooled improvement rate vs phenotypic distance with per-encoding curves
    and per-encoding box plots."""
    active_genotypes = [g for g in genotypes if g in data_dict and data_dict[g] is not None]
    if not active_genotypes:
        return

    n_bins = 15
    all_dist, all_fdiff, per_geno_dist, per_geno_fdiff = _pool_locality_data(
        data_dict, active_genotypes)
    if len(all_dist) < 20:
        return

    bin_edges = np.linspace(all_dist.min(), all_dist.max(), n_bins + 1)

    # Pooled curve
    pooled_bins = _bin_data(all_dist, all_fdiff, bin_edges)
    bin_centers = []
    improvement_rates = []
    improvement_ses = []
    for center, bin_fdiff in pooled_bins:
        if len(bin_fdiff) < 5:
            continue
        p = np.mean(bin_fdiff > 0)
        bin_centers.append(center)
        improvement_rates.append(p)
        improvement_ses.append(np.sqrt(p * (1 - p) / len(bin_fdiff)))

    if not bin_centers:
        return

    bin_centers = np.array(bin_centers)
    improvement_rates = np.array(improvement_rates)
    improvement_ses = np.array(improvement_ses)

    fig, (ax_rate, ax_box) = plt.subplots(
        2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )

    # Per-encoding curves (transparent, behind pooled)
    for genotype in active_genotypes:
        geno_bins = _bin_data(per_geno_dist[genotype], per_geno_fdiff[genotype], bin_edges)
        gc, gr = [], []
        for center, bin_fdiff in geno_bins:
            if len(bin_fdiff) < 5:
                continue
            gc.append(center)
            gr.append(np.mean(bin_fdiff > 0))
        if gc:
            ax_rate.plot(gc, gr, "o-", color=COLORS[genotype], linewidth=1.5,
                         markersize=4, alpha=0.4, label=GENOTYPE_LABELS[genotype])

    # Pooled curve (solid black, on top)
    ax_rate.plot(bin_centers, improvement_rates, "o-", color="black",
                 linewidth=2.5, markersize=5, label="All pooled", zorder=5)
    ax_rate.fill_between(bin_centers,
                         improvement_rates - 1.96 * improvement_ses,
                         improvement_rates + 1.96 * improvement_ses,
                         color="black", alpha=0.15, zorder=4)
    ax_rate.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    apply_axis_style(ax_rate, ylabel="Improvement Rate")
    ax_rate.legend()

    _draw_encoding_boxplots(ax_box, active_genotypes, per_geno_dist)

    fig.tight_layout()
    save_figure(fig, out_dir, filename)

    print_analysis_header(f"Fitness Improvement by Distance ({filename}) — {task}")
    print(f"  Pooled curve: {len(all_dist)} parent-child pairs across all encodings")
    for genotype in active_genotypes:
        df = data_dict.get(genotype)
        if df is None:
            continue
        label = GENOTYPE_LABELS.get(genotype, genotype)
        fdiff = df["fitness_diff"].dropna().values
        dist_vals = per_geno_dist.get(genotype, np.array([]))
        rate = np.mean(fdiff > 0) if len(fdiff) > 0 else 0
        print(f"  {label}: improvement rate = {rate:.3f}"
              f"  median distance = {np.median(dist_vals):.3f}"
              f"  mean distance = {np.mean(dist_vals):.3f}  n={len(fdiff)}")
    print_analysis_footer(os.path.join(out_dir, filename))


def _plot_fitness_change_by_distance(task, data_dict, genotypes, out_dir,
                                     filename="fitness_change_by_distance"):
    """Pooled mean fitness change vs phenotypic distance with per-encoding curves
    and per-encoding box plots."""
    active_genotypes = [g for g in genotypes if g in data_dict and data_dict[g] is not None]
    if not active_genotypes:
        return

    n_bins = 15
    all_dist, all_fdiff, per_geno_dist, per_geno_fdiff = _pool_locality_data(
        data_dict, active_genotypes)
    if len(all_dist) < 20:
        return

    bin_edges = np.linspace(all_dist.min(), all_dist.max(), n_bins + 1)

    # Pooled curve
    pooled_bins = _bin_data(all_dist, all_fdiff, bin_edges)
    bin_centers = []
    mean_diffs = []
    se_diffs = []
    for center, bin_fdiff in pooled_bins:
        if len(bin_fdiff) < 5:
            continue
        bin_centers.append(center)
        mean_diffs.append(np.mean(bin_fdiff))
        se_diffs.append(np.std(bin_fdiff) / np.sqrt(len(bin_fdiff)))

    if not bin_centers:
        return

    bin_centers = np.array(bin_centers)
    mean_diffs = np.array(mean_diffs)
    se_diffs = np.array(se_diffs)

    fig, (ax_mean, ax_box) = plt.subplots(
        2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )

    # Per-encoding curves (transparent, behind pooled)
    for genotype in active_genotypes:
        geno_bins = _bin_data(per_geno_dist[genotype], per_geno_fdiff[genotype], bin_edges)
        gc, gm = [], []
        for center, bin_fdiff in geno_bins:
            if len(bin_fdiff) < 5:
                continue
            gc.append(center)
            gm.append(np.mean(bin_fdiff))
        if gc:
            ax_mean.plot(gc, gm, "o-", color=COLORS[genotype], linewidth=1.5,
                         markersize=4, alpha=0.4, label=GENOTYPE_LABELS[genotype])

    # Pooled curve (solid black, on top)
    ax_mean.plot(bin_centers, mean_diffs, "o-", color="black",
                 linewidth=2.5, markersize=5, label="All pooled", zorder=5)
    ax_mean.fill_between(bin_centers,
                         mean_diffs - 1.96 * se_diffs,
                         mean_diffs + 1.96 * se_diffs,
                         color="black", alpha=0.15, zorder=4)
    ax_mean.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    apply_axis_style(ax_mean, ylabel="Mean Fitness Change")
    ax_mean.legend()

    _draw_encoding_boxplots(ax_box, active_genotypes, per_geno_dist)

    fig.tight_layout()
    save_figure(fig, out_dir, filename)

    print_analysis_header(f"Mean Fitness Change by Distance ({filename}) — {task}")
    print(f"  Pooled curve: {len(all_dist)} parent-child pairs across all encodings")
    for genotype in active_genotypes:
        df = data_dict.get(genotype)
        if df is None:
            continue
        label = GENOTYPE_LABELS.get(genotype, genotype)
        fdiff = per_geno_fdiff.get(genotype, np.array([]))
        dist_vals = per_geno_dist.get(genotype, np.array([]))
        print(f"  {label}: mean fitness change = {np.mean(fdiff):.4f}"
              f"  median distance = {np.median(dist_vals):.3f}"
              f"  mean distance = {np.mean(dist_vals):.3f}  n={len(fdiff)}")
    print_analysis_footer(os.path.join(out_dir, filename))


# ── Improvement/worsening magnitude ──────────────────────────────────────────

def _filter_data_dict(data_dict, condition_fn):
    """Return a copy of data_dict with rows filtered by condition_fn(df)."""
    filtered = {}
    for genotype, df in data_dict.items():
        if df is None:
            filtered[genotype] = None
            continue
        sub = df[condition_fn(df)]
        filtered[genotype] = sub if len(sub) > 0 else None
    return filtered


def _plot_magnitude_by_distance(task, data_dict, genotypes, out_dir,
                                 ylabel, filename):
    """Binned mean fitness change as % of parent fitness vs phenotypic distance.

    Expects data_dict already filtered to improving or worsening pairs.
    """
    active_genotypes = [g for g in genotypes if g in data_dict and data_dict[g] is not None]
    if not active_genotypes:
        return

    n_bins = 15

    # Compute percentage change and pool
    all_dist = []
    all_pct = []
    per_geno_dist = {}
    per_geno_pct = {}
    for genotype in active_genotypes:
        df = data_dict[genotype]
        dist = df["phenotypic_distance"].values.astype(float)
        fdiff = df["fitness_diff"].values.astype(float)
        parent_fit = df["parent_fitness"].values.astype(float)
        # Avoid division by zero
        valid = np.isfinite(dist) & np.isfinite(fdiff) & np.isfinite(parent_fit) & (np.abs(parent_fit) > 1e-12)
        dist, fdiff, parent_fit = dist[valid], fdiff[valid], parent_fit[valid]
        pct = fdiff / np.abs(parent_fit) * 100
        all_dist.append(dist)
        all_pct.append(pct)
        per_geno_dist[genotype] = dist
        per_geno_pct[genotype] = pct

    all_dist = np.concatenate(all_dist)
    all_pct = np.concatenate(all_pct)
    if len(all_dist) < 20:
        return

    bin_edges = np.linspace(all_dist.min(), all_dist.max(), n_bins + 1)

    # Pooled curve
    pooled_bins = _bin_data(all_dist, all_pct, bin_edges)
    bin_centers = []
    mean_pcts = []
    se_pcts = []
    for center, bin_vals in pooled_bins:
        if len(bin_vals) < 5:
            continue
        bin_centers.append(center)
        mean_pcts.append(np.mean(bin_vals))
        se_pcts.append(np.std(bin_vals) / np.sqrt(len(bin_vals)))

    if not bin_centers:
        return

    bin_centers = np.array(bin_centers)
    mean_pcts = np.array(mean_pcts)
    se_pcts = np.array(se_pcts)

    fig, (ax_main, ax_box) = plt.subplots(
        2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )

    # Per-encoding curves
    for genotype in active_genotypes:
        geno_bins = _bin_data(per_geno_dist[genotype], per_geno_pct[genotype], bin_edges)
        gc, gm = [], []
        for center, bin_vals in geno_bins:
            if len(bin_vals) < 5:
                continue
            gc.append(center)
            gm.append(np.mean(bin_vals))
        if gc:
            ax_main.plot(gc, gm, "o-", color=COLORS[genotype], linewidth=1.5,
                         markersize=4, alpha=0.4, label=GENOTYPE_LABELS[genotype])

    # Pooled curve
    ax_main.plot(bin_centers, mean_pcts, "o-", color="black",
                 linewidth=2.5, markersize=5, label="All pooled", zorder=5)
    ax_main.fill_between(bin_centers,
                         mean_pcts - 1.96 * se_pcts,
                         mean_pcts + 1.96 * se_pcts,
                         color="black", alpha=0.15, zorder=4)
    ax_main.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    apply_axis_style(ax_main, ylabel=ylabel)
    ax_main.legend()

    _draw_encoding_boxplots(ax_box, active_genotypes, per_geno_dist)

    fig.tight_layout()
    save_figure(fig, out_dir, filename)

    print_analysis_header(f"Magnitude by Distance ({filename}) — {task}")
    print(f"  Pooled: {len(all_dist)} parent-child pairs")
    for genotype in active_genotypes:
        label = GENOTYPE_LABELS.get(genotype, genotype)
        pct = per_geno_pct.get(genotype, np.array([]))
        dist_vals = per_geno_dist.get(genotype, np.array([]))
        print(f"  {label}: mean pct change = {np.mean(pct):.2f}%"
              f"  median distance = {np.median(dist_vals):.3f}  n={len(pct)}")
    print_analysis_footer(os.path.join(out_dir, filename))


def _plot_magnitude_combined(task, data_dict, genotypes, out_dir,
                              filename="magnitude_combined_by_distance"):
    """Combined improving/worsening magnitude as % of parent fitness vs distance.

    Draws two pooled curves (improving in green, worsening in red) on the same axes.
    """
    active_genotypes = [g for g in genotypes if g in data_dict and data_dict[g] is not None]
    if not active_genotypes:
        return

    n_bins = 15

    # Compute percentage change for all data
    all_dist = []
    all_pct = []
    for genotype in active_genotypes:
        df = data_dict[genotype]
        dist = df["phenotypic_distance"].values.astype(float)
        fdiff = df["fitness_diff"].values.astype(float)
        parent_fit = df["parent_fitness"].values.astype(float)
        valid = np.isfinite(dist) & np.isfinite(fdiff) & np.isfinite(parent_fit) & (np.abs(parent_fit) > 1e-12)
        all_dist.append(dist[valid])
        all_pct.append((fdiff[valid] / np.abs(parent_fit[valid])) * 100)

    all_dist = np.concatenate(all_dist)
    all_pct = np.concatenate(all_pct)
    if len(all_dist) < 20:
        return

    bin_edges = np.linspace(all_dist.min(), all_dist.max(), n_bins + 1)

    # Split into improving / worsening
    imp_mask = all_pct > 0
    wor_mask = all_pct < 0

    def _bin_subset(dist, pct):
        bins = _bin_data(dist, pct, bin_edges)
        centers, means, ses = [], [], []
        for center, bin_vals in bins:
            if len(bin_vals) < 5:
                continue
            centers.append(center)
            means.append(np.mean(bin_vals))
            ses.append(np.std(bin_vals) / np.sqrt(len(bin_vals)))
        return np.array(centers), np.array(means), np.array(ses)

    imp_c, imp_m, imp_se = _bin_subset(all_dist[imp_mask], all_pct[imp_mask])
    wor_c, wor_m, wor_se = _bin_subset(all_dist[wor_mask], all_pct[wor_mask])

    if len(imp_c) == 0 and len(wor_c) == 0:
        return

    # Collect per-genotype distances for boxplots
    per_geno_dist = {}
    for genotype in active_genotypes:
        df = data_dict[genotype]
        d = df["phenotypic_distance"].dropna().values
        per_geno_dist[genotype] = d

    fig, (ax_main, ax_box) = plt.subplots(
        2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )

    if len(imp_c) > 0:
        ax_main.plot(imp_c, imp_m, "o-", color="#2ca02c", linewidth=2.5,
                     markersize=5, label="Improving", zorder=5)
        ax_main.fill_between(imp_c, imp_m - 1.96 * imp_se, imp_m + 1.96 * imp_se,
                             color="#2ca02c", alpha=0.15, zorder=4)

    if len(wor_c) > 0:
        ax_main.plot(wor_c, wor_m, "o-", color="#d62728", linewidth=2.5,
                     markersize=5, label="Worsening", zorder=5)
        ax_main.fill_between(wor_c, wor_m - 1.96 * wor_se, wor_m + 1.96 * wor_se,
                             color="#d62728", alpha=0.15, zorder=4)

    ax_main.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    apply_axis_style(ax_main, ylabel="Mean Fitness Change (%)")
    ax_main.legend()

    _draw_encoding_boxplots(ax_box, active_genotypes, per_geno_dist)

    fig.tight_layout()
    save_figure(fig, out_dir, filename)

    print_analysis_header(f"Combined Magnitude by Distance — {task}")
    n_imp = imp_mask.sum()
    n_wor = wor_mask.sum()
    print(f"  Improving: {n_imp} pairs, mean = {np.mean(all_pct[imp_mask]):.2f}%" if n_imp > 0 else "  Improving: 0 pairs")
    print(f"  Worsening: {n_wor} pairs, mean = {np.mean(all_pct[wor_mask]):.2f}%" if n_wor > 0 else "  Worsening: 0 pairs")
    print_analysis_footer(os.path.join(out_dir, filename))


# ── Cross-task comparison ────────────────────────────────────────────────────

def _plot_locality_comparison(all_stats, genotypes, out_dir):
    """Grouped bar chart of R-squared values per genotype per task (signed phenotypic)."""
    if not all_stats:
        return

    tasks = list(all_stats.keys())
    active_genotypes = sorted({
        g for t in tasks
        for g in all_stats[t].get("phenotypic_signed", {})
    })
    active_genotypes = [g for g in genotypes if g in active_genotypes]
    if not active_genotypes:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    n_tasks = len(tasks)
    n_geno = len(active_genotypes)
    group_width = 0.8
    bar_width = group_width / max(n_geno, 1)
    x_positions = np.arange(n_tasks)

    for i, genotype in enumerate(active_genotypes):
        r_sq_vals = []
        for task in tasks:
            task_stats = all_stats[task].get("phenotypic_signed", {})
            s = task_stats.get(genotype)
            r_sq_vals.append(s["r_sq"] if s else 0.0)

        offset = x_positions + (i - n_geno / 2 + 0.5) * bar_width
        ax.bar(offset, r_sq_vals, width=bar_width * 0.9,
               color=COLORS[genotype], label=GENOTYPE_LABELS[genotype],
               alpha=0.8)

    ax.set_xticks(x_positions)
    ax.set_xticklabels([TASK_LABELS.get(t, t.capitalize()) for t in tasks])
    apply_axis_style(ax, xlabel="Task", ylabel="$R^2$")
    ax.set_title("Phenotypic Locality (Signed)", fontsize=18)
    ax.legend()

    fig.tight_layout()
    save_figure(fig, out_dir, "locality_comparison")


# ── Public entry point ───────────────────────────────────────────────────────

def run(base_dir, tasks, genotypes, output_base=None):
    """Generate all locality plots and analytical output."""
    if output_base is None:
        from experimentation.config import output_dir
        output_base = output_dir(base_dir)

    all_stats = {}

    for task in tasks:
        data_dict = {}
        for genotype in genotypes:
            data_dict[genotype] = _load_locality_data(base_dir, task, genotype)

        if not any(v is not None for v in data_dict.values()):
            continue

        out_dir = figures_dir(base_dir, task=task, plotter_name=PLOTTER_NAME)
        all_stats[task] = {}

        # Signed phenotypic locality scatter
        signed_stats = _plot_locality_panels(
            task, data_dict, genotypes, out_dir,
            distance_col="phenotypic_distance",
            y_col="fitness_diff",
            xlabel="Phenotypic Distance",
            ylabel="Fitness Difference (child - parent)",
            filename="phenotypic_locality_signed",
        )
        all_stats[task]["phenotypic_signed"] = signed_stats

        # Density version of signed scatter
        _plot_locality_density_panels(
            task, data_dict, genotypes, out_dir,
            distance_col="phenotypic_distance",
            y_col="fitness_diff",
            xlabel="Phenotypic Distance",
            ylabel="Fitness Difference (child - parent)",
            filename="phenotypic_locality_signed_density",
        )

        # Distance distribution: shows direct encoding children are closer to parents
        _plot_distance_distribution(task, data_dict, genotypes, out_dir)
        _plot_distance_distribution_boxplot(task, data_dict, genotypes, out_dir)

        # Improvement by distance bin: shows closer children → better fitness outcomes
        _plot_improvement_by_distance(task, data_dict, genotypes, out_dir)

        # Mean fitness change by distance bin
        _plot_fitness_change_by_distance(task, data_dict, genotypes, out_dir)

        # Improvement/worsening magnitude as % of parent fitness
        improving_dict = _filter_data_dict(data_dict, lambda df: df["fitness_diff"] > 0)
        _plot_magnitude_by_distance(
            task, improving_dict, genotypes, out_dir,
            ylabel="Mean Fitness Improvement (%)",
            filename="improvement_magnitude_by_distance",
        )

        worsening_dict = _filter_data_dict(data_dict, lambda df: df["fitness_diff"] < 0)
        _plot_magnitude_by_distance(
            task, worsening_dict, genotypes, out_dir,
            ylabel="Mean Fitness Worsening (%)",
            filename="worsening_magnitude_by_distance",
        )

        _plot_magnitude_combined(task, data_dict, genotypes, out_dir)

        # Selected-only variants (children that survived into population)
        # Only meaningful for mu+lambda where in_pop distinguishes survivors;
        # NEAT logs all offspring as in_pop=True so this is a no-op there.
        selected_dict = {}
        for genotype, df in data_dict.items():
            if df is not None and "in_pop" in df.columns:
                sel = df[df["in_pop"] == True]
                # Only produce selected plots if filtering actually removes rows
                if len(sel) < len(df) and len(sel) > 0:
                    selected_dict[genotype] = sel
                else:
                    selected_dict[genotype] = None
            else:
                selected_dict[genotype] = None

        if any(v is not None for v in selected_dict.values()):
            _plot_improvement_by_distance(
                task, selected_dict, genotypes, out_dir,
                filename="improvement_by_distance_selected")
            _plot_fitness_change_by_distance(
                task, selected_dict, genotypes, out_dir,
                filename="fitness_change_by_distance_selected")

    # Cross-task comparison
    cross_out = figures_dir(base_dir, task="cross_task", plotter_name=PLOTTER_NAME)
    _plot_locality_comparison(all_stats, genotypes, cross_out)

    # Cross-task pooled density: one panel per genotype with all tasks combined
    pooled_by_geno = {}
    for task in tasks:
        for genotype in genotypes:
            df = _load_locality_data(base_dir, task, genotype)
            if df is not None:
                pooled_by_geno.setdefault(genotype, []).append(df)
    pooled_dict = {}
    for genotype, dfs in pooled_by_geno.items():
        pooled_dict[genotype] = pd.concat(dfs, ignore_index=True)
    _plot_locality_density_panels(
        "cross_task", pooled_dict, genotypes, cross_out,
        distance_col="phenotypic_distance",
        y_col="fitness_diff",
        xlabel="Phenotypic Distance",
        ylabel="Fitness Difference (child - parent)",
        filename="phenotypic_locality_pooled_density",
    )
    _plot_distance_distribution("cross_task", pooled_dict, genotypes, cross_out)
    _plot_distance_distribution_boxplot("cross_task", pooled_dict, genotypes, cross_out)


if __name__ == "__main__":
    from experimentation.config import BASE_DIR, TASKS, GENOTYPES
    run(BASE_DIR, TASKS, GENOTYPES)
