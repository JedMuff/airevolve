import argparse
import ast
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation

def _load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "waypoints" not in df.columns or "total_energy_j" not in df.columns:
        if "fitness" not in df.columns:
            raise ValueError(f"{path}: need 'fitness' column or ('waypoints','total_energy_j') columns")
        parsed = df["fitness"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
        df["waypoints"] = parsed.apply(lambda t: float(t[0]))
        df["total_energy_j"] = parsed.apply(lambda t: float(t[1]))
    df["waypoints"] = pd.to_numeric(df["waypoints"], errors="coerce")
    df["total_energy_j"] = pd.to_numeric(df["total_energy_j"], errors="coerce")
    return df

def main():
    parser = argparse.ArgumentParser(description="Plot all pareto fronts over generations.")
    parser.add_argument("csv_file", help="Path to evolution_data.csv")
    args = parser.parse_args()

    csv_path = args.csv_file
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    out_dir = os.path.dirname(os.path.abspath(csv_path))
    df = _load_csv(csv_path)

    finite_mask = np.isfinite(df["total_energy_j"]) & np.isfinite(df["waypoints"])
    df = df[finite_mask]

    # Drop sentinel energy values (1e9 = failed / non-hoverable morphologies) because otherwise it blows up the x-axis scale and hide real differences.
    _SENTINEL_THRESHOLD = 1e8
    df = df[df["total_energy_j"] < _SENTINEL_THRESHOLD]

    gens = sorted(df["generation"].unique())
    if not gens:
        print("No valid generation data found.")
        return

    # --- 1. Static Plot with Colormap ---
    fig, ax = plt.subplots(figsize=(10, 7))
    cmap = plt.get_cmap("viridis_r", len(gens))

    for i, g in enumerate(gens):
        pop = df[df["generation"] == g]
        if "rank" in pop.columns and pop["rank"].notna().any():
            pop = pop[pop["rank"] == 0]
        
        if pop.empty:
            continue
            
        srt = pop.sort_values("total_energy_j")
        
        # Plot connected data points for Pareto front
        ax.plot(srt["total_energy_j"], srt["waypoints"], color=cmap(i), alpha=0.5, linewidth=1.5, marker='o', markersize=4)
        
        # Highlight points for first and last generations to avoid clutter
        if i == len(gens) - 1:
            ax.scatter(srt["total_energy_j"], srt["waypoints"], c=[cmap(i)], 
                       label=f"Gen {int(g)} (Final)", edgecolors="k", zorder=5, s=60)
        elif i == 0:
            ax.scatter(srt["total_energy_j"], srt["waypoints"], c=[cmap(i)], 
                       label=f"Gen {int(g)} (Initial)", edgecolors="k", zorder=4, s=30)

    ax.set_xlabel("Total Energy Consumed (J)  ← Minimize", fontsize=12)
    ax.set_ylabel("Gates Passed  ↑ Maximize", fontsize=12)
    ax.set_title("Pareto Front Evolution Over All Generations", fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.invert_xaxis()

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(gens), vmax=max(gens)))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Generation")
    ax.legend(loc="best")

    fig.tight_layout()
    out_png = os.path.join(out_dir, "pareto_evolution_all.png")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"Saved static colormap plot -> {out_png}")
    plt.close(fig)

    # --- 2. Animation (MP4) ---
    print("Generating MP4 animation...")
    fig_anim, ax_anim = plt.subplots(figsize=(10, 7))
    ax_anim.set_xlabel("Total Energy Consumed (J)  ← Minimize", fontsize=12)
    ax_anim.set_ylabel("Gates Passed  ↑ Maximize", fontsize=12)
    ax_anim.set_title("Pareto Front Evolution", fontsize=13)
    ax_anim.grid(True, linestyle="--", alpha=0.4)
    
    ax_anim.set_xlim(df["total_energy_j"].max() * 1.05, df["total_energy_j"].min() * 0.95)
    ax_anim.set_ylim(df["waypoints"].min() - 1, df["waypoints"].max() + 1)

    line, = ax_anim.plot([], [], color="blue", linewidth=2.5, marker='o', markersize=6, zorder=10)
    scatter = ax_anim.scatter([], [], c="blue", edgecolors="k", s=80, zorder=11)
    title_text = ax_anim.text(0.05, 0.95, "", transform=ax_anim.transAxes, 
                              fontsize=14, fontweight="bold", verticalalignment='top')

    traces = []

    def init():
        line.set_data([], [])
        scatter.set_offsets(np.empty((0, 2)))
        title_text.set_text("")
        return line, scatter, title_text

    def update(frame):
        g = gens[frame]
        pop = df[df["generation"] == g]
        if "rank" in pop.columns and pop["rank"].notna().any():
            pop = pop[pop["rank"] == 0]
            
        srt = pop.sort_values("total_energy_j")
        
        # Add the previous generation as a faint gray trace
        if frame > 0:
            prev_g = gens[frame-1]
            prev_pop = df[df["generation"] == prev_g]
            if "rank" in prev_pop.columns and prev_pop["rank"].notna().any():
                prev_pop = prev_pop[prev_pop["rank"] == 0]
                prev_srt = prev_pop.sort_values("total_energy_j")
                tr, = ax_anim.plot(prev_srt["total_energy_j"], prev_srt["waypoints"], 
                                   color="gray", alpha=0.25, linewidth=1, marker='o', markersize=3, zorder=1)
                traces.append(tr)
                
        line.set_data(srt["total_energy_j"], srt["waypoints"])
        # Fix: handle 0 points correctly
        if len(srt) > 0:
            scatter.set_offsets(np.c_[srt["total_energy_j"], srt["waypoints"]])
        else:
            scatter.set_offsets(np.empty((0, 2)))
            
        title_text.set_text(f"Generation {int(g)}")
        return line, scatter, title_text

    ani = FuncAnimation(fig_anim, update, frames=len(gens), init_func=init, blit=False, repeat=False)
    out_mp4 = os.path.join(out_dir, "pareto_evolution.mp4")
    try:
        ani.save(out_mp4, fps=5, extra_args=['-vcodec', 'libx264'])
        print(f"Saved MP4 animation -> {out_mp4}")
    except Exception as e:
        print(f"Error saving MP4 (ffmpeg may not be installed): {e}")
        # Fallback to GIF
        out_gif = os.path.join(out_dir, "pareto_evolution.gif")
        try:
            ani.save(out_gif, fps=5, writer="pillow")
            print(f"Saved GIF animation instead -> {out_gif}")
        except Exception as e2:
            print(f"Error saving GIF: {e2}")
            
    plt.close(fig_anim)

if __name__ == "__main__":
    main()
