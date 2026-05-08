"""Post-processing Pareto-front visualisation for NSGA-II bi-objective runs.

Reads one or more evolution_data.csv files (produced by run_energy_ablation.py
or run_nsga2_evolution.py) and generates a 2-D scatter plot:

  X-axis : Total Energy Consumed (J)   [Minimise]
  Y-axis : Waypoints / Gates Passed    [Maximise]

Each front-0 (Pareto-optimal) solution is highlighted.  When multiple CSV
files are supplied they are overlaid with distinct colours so experiment
results can be compared directly.

CLI usage
---------
  # Single experiment
  python plot_pareto_front.py results/exp_baseline_w_0.0/evolution_data.csv

  # Compare three experiments
  python plot_pareto_front.py \\
      results/exp_baseline_w_0.0/evolution_data.csv \\
      results/exp_dense_w_1e-05/evolution_data.csv \\
      results/exp_sparse_w_0.001/evolution_data.csv \\
      --labels "Baseline" "Dense 1e-5" "Sparse 1e-3" \\
      --output pareto_comparison.png

  # Show only the last-generation Pareto front (rank==0)
  python plot_pareto_front.py results/exp_baseline_w_0.0/evolution_data.csv --pareto-only

Module usage
------------
  from airevolve.evolution_tools.inspection_tools.plot_pareto_front import (
      plot_pareto_front, plot_pareto_comparison
  )
  fig = plot_pareto_front("results/exp_baseline_w_0.0/evolution_data.csv")
  fig.savefig("out.png")
"""

from __future__ import annotations

import argparse
import ast
import os
from pathlib import Path
from typing import List, Optional, Sequence

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd

# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_csv(path: str) -> pd.DataFrame:
    """Load evolution_data.csv and ensure waypoints/total_energy_j columns exist."""
    df = pd.read_csv(path)

    # Columns may be pre-expanded (waypoints, total_energy_j) or still packed
    # as a string representation of a tuple in 'fitness'.
    if "waypoints" not in df.columns or "total_energy_j" not in df.columns:
        if "fitness" not in df.columns:
            raise ValueError(
                f"{path}: need 'fitness' column or ('waypoints','total_energy_j') columns"
            )
        parsed = df["fitness"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
        df["waypoints"]      = parsed.apply(lambda t: float(t[0]))
        df["total_energy_j"] = parsed.apply(lambda t: float(t[1]))

    # Ensure numeric
    df["waypoints"]      = pd.to_numeric(df["waypoints"],      errors="coerce")
    df["total_energy_j"] = pd.to_numeric(df["total_energy_j"], errors="coerce")

    # Ensure rank column exists (might be missing for very old CSVs)
    if "rank" not in df.columns:
        df["rank"] = np.nan

    return df


def _last_gen_pop(df: pd.DataFrame) -> pd.DataFrame:
    """Return individuals from the last recorded generation."""
    last_gen = df["generation"].max()
    return df[df["generation"] == last_gen].copy()


# ── Single-experiment plot ────────────────────────────────────────────────────

def plot_pareto_front(
    csv_path: str,
    *,
    generation: Optional[int] = None,
    pareto_only: bool = False,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    color_by_rank: bool = True,
    max_ranks_shown: int = 5,
) -> plt.Figure:
    """Plot the Pareto scatter for a single CSV.

    Parameters
    ----------
    csv_path      : path to evolution_data.csv.
    generation    : generation to visualise (default: last available).
    pareto_only   : if True, plot only rank-0 (Pareto-optimal) individuals.
    ax            : optional existing Axes; creates a new figure if None.
    title         : plot title override.
    color_by_rank : colour points by Pareto front index.
    max_ranks_shown : fronts beyond this index are grouped as "other".
    """
    df = _load_csv(csv_path)
    pop = _last_gen_pop(df) if generation is None else df[df["generation"] == generation]

    # Drop failed individuals (energy sentinel) for cleaner plots
    finite_mask = np.isfinite(pop["total_energy_j"]) & np.isfinite(pop["waypoints"])
    pop = pop[finite_mask]

    if pareto_only:
        pop = pop[pop["rank"] == 0] if "rank" in pop.columns and pop["rank"].notna().any() else pop

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(9, 6))
    else:
        fig = ax.get_figure()

    cmap = cm.get_cmap("tab10", max_ranks_shown)

    if color_by_rank and "rank" in pop.columns and pop["rank"].notna().any():
        for r in range(max_ranks_shown + 1):
            mask = (pop["rank"] == r) if r < max_ranks_shown else (pop["rank"] >= max_ranks_shown)
            sub  = pop[mask]
            if sub.empty:
                continue
            colour = cmap(r) if r < max_ranks_shown else "lightgrey"
            label  = f"Front {r}" if r < max_ranks_shown else f"Front ≥{max_ranks_shown}"
            ax.scatter(
                sub["total_energy_j"], sub["waypoints"],
                c=[colour], label=label,
                edgecolors="k", linewidths=0.5, s=70, zorder=3,
            )
    else:
        ax.scatter(
            pop["total_energy_j"], pop["waypoints"],
            c="steelblue", edgecolors="k", linewidths=0.5, s=70, zorder=3,
        )

    _style_axes(ax, title or Path(csv_path).parent.name)

    if standalone:
        fig.tight_layout()
    return fig


# ── Multi-experiment comparison plot ─────────────────────────────────────────

def plot_pareto_comparison(
    csv_paths: Sequence[str],
    labels: Optional[Sequence[str]] = None,
    *,
    generation: Optional[int] = None,
    pareto_only: bool = True,
    output: Optional[str] = None,
    title: str = "Pareto-Front Comparison",
) -> plt.Figure:
    """Overlay Pareto fronts from multiple experiments.

    Parameters
    ----------
    csv_paths  : paths to evolution_data.csv files (one per experiment).
    labels     : legend labels (default: CSV parent directory names).
    generation : generation to compare (default: last in each file).
    pareto_only: if True, plot only rank-0 individuals per experiment.
    output     : if given, save the figure to this path.
    title      : overall figure title.
    """
    labels = list(labels) if labels else [Path(p).parent.name for p in csv_paths]
    cmap   = cm.get_cmap("tab10", len(csv_paths))

    fig, ax = plt.subplots(figsize=(10, 7))

    for idx, (path, label) in enumerate(zip(csv_paths, labels)):
        df  = _load_csv(path)
        pop = _last_gen_pop(df) if generation is None else df[df["generation"] == generation]

        finite_mask = np.isfinite(pop["total_energy_j"]) & np.isfinite(pop["waypoints"])
        pop = pop[finite_mask]

        if pareto_only and "rank" in pop.columns and pop["rank"].notna().any():
            pop = pop[pop["rank"] == 0]

        colour = cmap(idx)
        ax.scatter(
            pop["total_energy_j"], pop["waypoints"],
            c=[colour], label=label,
            edgecolors="k", linewidths=0.5, s=80, alpha=0.85, zorder=3,
        )

        # Draw a step-function Pareto curve through front-0 points
        if not pop.empty:
            srt = pop.sort_values("total_energy_j")
            ax.step(
                srt["total_energy_j"], srt["waypoints"],
                where="post", color=colour, linewidth=1.2, alpha=0.6,
            )

    _style_axes(ax, title)
    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=200, bbox_inches="tight")
        print(f"Saved comparison plot → {output}")

    return fig


# ── Shared axis styling ───────────────────────────────────────────────────────

def _style_axes(ax: plt.Axes, title: str) -> None:
    ax.set_xlabel("Total Energy Consumed (J)  ← Minimize", fontsize=12)
    ax.set_ylabel("Gates Passed  ↑ Maximize",               fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Plot NSGA-II Pareto front(s)")
    p.add_argument("csv_files", nargs="+", help="evolution_data.csv paths")
    p.add_argument("--labels",      nargs="*", default=None,
                   help="Legend labels (one per CSV file)")
    p.add_argument("--output",      default=None,
                   help="Save path for the figure (e.g. out.png)")
    p.add_argument("--generation",  type=int, default=None,
                   help="Generation to visualise (default: last)")
    p.add_argument("--pareto-only", action="store_true",
                   help="Show only Pareto-optimal (rank-0) individuals")
    p.add_argument("--title",       default=None)
    return p.parse_args()


def main():
    args = _parse_args()

    if len(args.csv_files) == 1:
        fig = plot_pareto_front(
            args.csv_files[0],
            generation=args.generation,
            pareto_only=args.pareto_only,
            title=args.title,
        )
    else:
        fig = plot_pareto_comparison(
            args.csv_files,
            labels=args.labels,
            generation=args.generation,
            pareto_only=args.pareto_only,
            output=args.output,
            title=args.title or "Pareto-Front Comparison",
        )

    if args.output and len(args.csv_files) == 1:
        fig.savefig(args.output, dpi=200, bbox_inches="tight")
        print(f"Saved → {args.output}")
    elif not args.output:
        plt.show()


if __name__ == "__main__":
    main()
