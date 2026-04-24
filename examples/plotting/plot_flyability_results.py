"""
Flyability Results Plotter

Loads stored results from a flyability experiment and generates visualizations.

Usage:
    python examples/plotting/plot_flyability_results.py .data/flyability/<experiment_dir> --show
    python examples/plotting/plot_flyability_results.py .data/flyability/<experiment_dir> --dpi 150
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np


def parse_arguments():
    parser = argparse.ArgumentParser(description="Plot flyability experiment results")
    parser.add_argument("experiment_dir", help="Path to experiment output directory")
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for saved images (default: 300)")
    return parser.parse_args()


def load_data(experiment_dir):
    """Load summary and per-drone data from experiment directory."""
    summary_path = os.path.join(experiment_dir, "summary.json")
    all_drones_path = os.path.join(experiment_dir, "all_drones.json")

    if not os.path.exists(summary_path):
        print(f"Error: summary.json not found in {experiment_dir}")
        sys.exit(1)
    if not os.path.exists(all_drones_path):
        print(f"Error: all_drones.json not found in {experiment_dir}")
        sys.exit(1)

    with open(summary_path) as f:
        summary = json.load(f)
    with open(all_drones_path) as f:
        all_drones = json.load(f)

    return summary, all_drones


# ---------------------------------------------------------------------------
# Individual plots
# ---------------------------------------------------------------------------

def plot_funnel(ax, summary):
    """Funnel bar chart: N sampled -> hover -> repair -> initial fly -> tuned flyable."""
    stages = ["Sampled", "Hover\nPassed", "Repair\nPassed", "Initial\nFly", "Tuned\nFlyable"]
    counts = [
        summary["n_sampled"],
        summary["n_hover_passed"],
        summary["n_repair_passed"],
        summary["n_initial_fly"],
        summary["n_tuned_flyable"],
    ]

    colors = ["#4c72b0", "#55a868", "#c44e52", "#8172b2", "#ccb974"]
    bars = ax.bar(stages, counts, color=colors, edgecolor="white", linewidth=0.5)

    n_sampled = summary["n_sampled"]
    for bar, count in zip(bars, counts):
        pct = count / n_sampled * 100 if n_sampled > 0 else 0
        label = f"{count}\n({pct:.1f}%)"
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(),
            label, ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    ax.set_ylabel("Count")
    ax.set_title("Flyability Funnel")
    ax.set_ylim(0, max(counts) * 1.25 if max(counts) > 0 else 1)


def plot_gates_distribution(ax, tuned_drones, gates_threshold):
    """Histogram of gates_passed for all tuned drones."""
    if not tuned_drones:
        ax.text(0.5, 0.5, "No tuned drones", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Gates Passed Distribution (Tuned)")
        return

    gates = [d["tuning"]["gates_passed"] for d in tuned_drones]
    max_gates = max(gates) if gates else 0
    bins = np.arange(-0.5, max_gates + 1.5, 1)

    colors = ["#c44e52" if g < gates_threshold else "#55a868" for g in sorted(set(gates))]
    color_map = {g: "#55a868" if g >= gates_threshold else "#c44e52" for g in range(max_gates + 1)}
    bar_colors = [color_map.get(int(g), "#c44e52") for g in gates]

    n, bin_edges, patches = ax.hist(gates, bins=bins, edgecolor="white", linewidth=0.5)
    for patch, left_edge in zip(patches, bin_edges[:-1]):
        gate_val = int(left_edge + 0.5)
        patch.set_facecolor(color_map.get(gate_val, "#c44e52"))

    ax.axvline(gates_threshold - 0.5, color="black", linestyle="--", linewidth=1, label=f"Threshold ({gates_threshold})")
    ax.legend(fontsize=8)
    ax.set_xlabel("Gates Passed")
    ax.set_ylabel("Count")
    ax.set_title("Gates Passed Distribution (Tuned)")


def plot_evaluations_distribution(ax, successful_drones):
    """Histogram of n_evaluations for drones that met the gates threshold."""
    if not successful_drones:
        ax.text(0.5, 0.5, "No successful drones", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Evaluations to Threshold")
        return

    evals = [d["tuning"]["n_evaluations"] for d in successful_drones]
    ax.hist(evals, bins=20, color="#4c72b0", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Number of Evaluations")
    ax.set_ylabel("Count")
    ax.set_title("Evaluations to Threshold")

    mean_evals = np.mean(evals)
    ax.axvline(mean_evals, color="#c44e52", linestyle="--", linewidth=1, label=f"Mean ({mean_evals:.0f})")
    ax.legend(fontsize=8)


def plot_evaluations_vs_gates(ax, tuned_drones):
    """Scatter: n_evaluations vs gates_passed, colored by early_stopped."""
    if not tuned_drones:
        ax.text(0.5, 0.5, "No tuned drones", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Evaluations vs Gates Passed")
        return

    evals = [d["tuning"]["n_evaluations"] for d in tuned_drones]
    gates = [d["tuning"]["gates_passed"] for d in tuned_drones]
    early = [d["tuning"]["early_stopped"] for d in tuned_drones]

    colors = ["#55a868" if e else "#c44e52" for e in early]
    ax.scatter(evals, gates, c=colors, alpha=0.7, edgecolors="white", linewidth=0.5, s=50)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#55a868", markersize=8, label="Early stopped"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#c44e52", markersize=8, label="Ran full budget"),
    ]
    ax.legend(handles=legend_elements, fontsize=8)
    ax.set_xlabel("Number of Evaluations")
    ax.set_ylabel("Gates Passed")
    ax.set_title("Evaluations vs Gates Passed")


def plot_gains_distribution(axes, tuned_drones, gates_threshold):
    """Box plots comparing gains for successful vs unsuccessful drones."""
    if not tuned_drones:
        for ax in axes:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    gain_names = ["pos_P", "vel_P", "att_P", "rate_P"]

    successful = [d for d in tuned_drones if d["tuning"]["gates_passed"] >= gates_threshold]
    unsuccessful = [d for d in tuned_drones if d["tuning"]["gates_passed"] < gates_threshold]

    for ax, gain_name in zip(axes, gain_names):
        data = []
        labels = []

        if successful:
            vals = [d["tuning"]["best_gains"][gain_name] for d in successful if d["tuning"]["best_gains"]]
            if vals:
                data.append(vals)
                labels.append(f"Pass\n(n={len(vals)})")

        if unsuccessful:
            vals = [d["tuning"]["best_gains"][gain_name] for d in unsuccessful if d["tuning"]["best_gains"]]
            if vals:
                data.append(vals)
                labels.append(f"Fail\n(n={len(vals)})")

        if data:
            bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.5)
            colors = ["#55a868", "#c44e52"]
            for patch, color in zip(bp["boxes"], colors[:len(data)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)

        ax.set_title(gain_name, fontsize=10)
        ax.set_ylabel("Gain value")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_arguments()
    summary, all_drones = load_data(args.experiment_dir)

    gates_threshold = summary["experiment_config"].get("gates_threshold", 9)

    # Filter subsets
    tuned_drones = [d for d in all_drones if "tuning" in d]
    successful_drones = [d for d in tuned_drones if d["tuning"]["gates_passed"] >= gates_threshold]

    plots_dir = os.path.join(args.experiment_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    print(f"Generating plots for: {args.experiment_dir}")
    print(f"  Tuned drones: {len(tuned_drones)}")
    print(f"  Successful:   {len(successful_drones)}")
    print(f"  Plots dir:    {plots_dir}")

    # --- 1. Funnel chart ---
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    plot_funnel(ax1, summary)
    fig1.tight_layout()
    fig1.savefig(os.path.join(plots_dir, "funnel_chart.png"), dpi=args.dpi, bbox_inches="tight")

    # --- 2. Gates distribution ---
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    plot_gates_distribution(ax2, tuned_drones, gates_threshold)
    fig2.tight_layout()
    fig2.savefig(os.path.join(plots_dir, "gates_distribution.png"), dpi=args.dpi, bbox_inches="tight")

    # --- 3. Evaluations distribution ---
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    plot_evaluations_distribution(ax3, successful_drones)
    fig3.tight_layout()
    fig3.savefig(os.path.join(plots_dir, "evaluations_distribution.png"), dpi=args.dpi, bbox_inches="tight")

    # --- 4. Evaluations vs gates scatter ---
    fig4, ax4 = plt.subplots(figsize=(8, 5))
    plot_evaluations_vs_gates(ax4, tuned_drones)
    fig4.tight_layout()
    fig4.savefig(os.path.join(plots_dir, "evaluations_vs_gates.png"), dpi=args.dpi, bbox_inches="tight")

    # --- 5. Gains box plots ---
    fig5, axes5 = plt.subplots(1, 4, figsize=(14, 4))
    plot_gains_distribution(axes5, tuned_drones, gates_threshold)
    fig5.suptitle("Controller Gains Distribution", fontsize=12, y=1.02)
    fig5.tight_layout()
    fig5.savefig(os.path.join(plots_dir, "gains_distribution.png"), dpi=args.dpi, bbox_inches="tight")

    # --- Combined summary figure ---
    fig_all, axes_all = plt.subplots(2, 3, figsize=(18, 10))

    plot_funnel(axes_all[0, 0], summary)
    plot_gates_distribution(axes_all[0, 1], tuned_drones, gates_threshold)
    plot_evaluations_distribution(axes_all[0, 2], successful_drones)
    plot_evaluations_vs_gates(axes_all[1, 0], tuned_drones)

    # Gains in bottom-center and bottom-right (split into 2x2 inset)
    axes_all[1, 1].remove()
    axes_all[1, 2].remove()

    gain_names = ["pos_P", "vel_P", "att_P", "rate_P"]
    inset_axes = []
    for i, gain_name in enumerate(gain_names):
        row = i // 2
        col = i % 2
        left = 0.58 + col * 0.2
        bottom = 0.05 + (1 - row) * 0.22
        ax_inset = fig_all.add_axes([left, bottom, 0.17, 0.18])
        inset_axes.append(ax_inset)

    plot_gains_distribution(inset_axes, tuned_drones, gates_threshold)

    fig_all.suptitle("Flyability Experiment Summary", fontsize=14, fontweight="bold")
    fig_all.tight_layout(rect=[0, 0, 0.55, 0.95])
    fig_all.savefig(os.path.join(plots_dir, "flyability_summary.png"), dpi=args.dpi, bbox_inches="tight")

    print(f"\nSaved {6} plots to {plots_dir}")

    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
