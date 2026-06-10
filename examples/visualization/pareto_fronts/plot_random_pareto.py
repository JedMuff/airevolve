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
from scipy.interpolate import interp1d

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

def get_pareto_front(pop):
    if "rank" in pop.columns and pop["rank"].notna().any():
        return pop[pop["rank"] == 0]
    
    sorted_pop = pop.sort_values(["total_energy_j", "waypoints"], ascending=[True, False])
    pareto_front = []
    best_waypoints = -np.inf
    
    for _, row in sorted_pop.iterrows():
        if row["waypoints"] > best_waypoints:
            pareto_front.append(row)
            best_waypoints = row["waypoints"]
            
    return pd.DataFrame(pareto_front)

def plot_smooth_lines(df, gens, out_dir):
    fig, ax = plt.subplots(figsize=(10, 7))
    cmap = plt.get_cmap("viridis", len(gens))
    
    for i, g in enumerate(gens):
        pop = df[df["generation"] == g]
        pf = get_pareto_front(pop)
        if pf.empty:
            continue
            
        srt = pf.sort_values("total_energy_j")
        
        if len(srt) > 2:
            try:
                # Smooth interpolation
                x = srt["total_energy_j"].values
                y = srt["waypoints"].values
                
                # Make strictly increasing for interpolation
                _, idx = np.unique(x, return_index=True)
                x = x[idx]
                y = y[idx]
                
                if len(x) > 3:
                    f = interp1d(x, y, kind='quadratic', fill_value="extrapolate")
                    x_new = np.linspace(x.min(), x.max(), 100)
                    y_new = f(x_new)
                    ax.plot(x_new, y_new, color=cmap(i), alpha=0.6, linewidth=1.5)
                else:
                    ax.plot(x, y, color=cmap(i), alpha=0.6, linewidth=1.5)
            except Exception:
                ax.plot(srt["total_energy_j"], srt["waypoints"], color=cmap(i), alpha=0.6, linewidth=1.5)
        else:
            ax.plot(srt["total_energy_j"], srt["waypoints"], color=cmap(i), alpha=0.6, linewidth=1.5)

    ax.set_xlabel("Total Energy Consumed (J)  ← Minimize", fontsize=12)
    ax.set_ylabel("Gates Passed  ↑ Maximize", fontsize=12)
    ax.set_title("Pareto Front Evolution: Smooth Interpolation", fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.4)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(gens), vmax=max(gens)))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Generation")
    
    out_png = os.path.join(out_dir, "pareto_plot1_smooth_lines.png")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

def plot_density(df, gens, out_dir):
    fig, ax = plt.subplots(figsize=(10, 7))
    
    all_pf_x = []
    all_pf_y = []
    
    for g in gens:
        pop = df[df["generation"] == g]
        pf = get_pareto_front(pop)
        if pf.empty:
            continue
        all_pf_x.extend(pf["total_energy_j"].values)
        all_pf_y.extend(pf["waypoints"].values)
        
    if not all_pf_x:
        return
        
    hb = ax.hexbin(all_pf_x, all_pf_y, gridsize=30, cmap='inferno', bins='log', mincnt=1)
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('log10(N)')
    
    ax.set_xlabel("Total Energy Consumed (J)  ← Minimize", fontsize=12)
    ax.set_ylabel("Gates Passed  ↑ Maximize", fontsize=12)
    ax.set_title("Density of Pareto Points Across All Generations", fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.2)
    
    out_png = os.path.join(out_dir, "pareto_plot2_density.png")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

def plot_filled_areas(df, gens, out_dir):
    fig, ax = plt.subplots(figsize=(10, 7))
    
    if len(gens) >= 3:
        sel_gens = [gens[0], gens[len(gens)//2], gens[-1]]
        labels = ["Initial", "Middle", "Final"]
    else:
        sel_gens = gens
        labels = [f"Gen {g}" for g in gens]
        
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    
    for g, label, color in zip(sel_gens, labels, colors[:len(sel_gens)]):
        pop = df[df["generation"] == g]
        pf = get_pareto_front(pop)
        if pf.empty:
            continue
            
        srt = pf.sort_values("total_energy_j")
        x = srt["total_energy_j"].values
        y = srt["waypoints"].values
        
        min_y = df["waypoints"].min() - 1
        ax.fill_between(x, min_y, y, color=color, alpha=0.4, label=f"{label} (Gen {int(g)})", step="post")
        ax.step(x, y, where="post", color=color, linewidth=2)

    ax.set_xlabel("Total Energy Consumed (J)  ← Minimize", fontsize=12)
    ax.set_ylabel("Gates Passed  ↑ Maximize", fontsize=12)
    ax.set_title("Dominated Area Progression", fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left")
    
    out_png = os.path.join(out_dir, "pareto_plot3_filled_areas.png")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

def plot_3d_surface(df, gens, out_dir):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    for i, g in enumerate(gens):
        pop = df[df["generation"] == g]
        pf = get_pareto_front(pop)
        if pf.empty:
            continue
            
        srt = pf.sort_values("total_energy_j")
        x = srt["total_energy_j"].values
        y = srt["waypoints"].values
        z = np.full_like(x, g)
        
        ax.plot(x, z, y, color=cm.viridis(i / max(1, len(gens)-1)), linewidth=2, alpha=0.8)

    ax.set_xlabel("Energy Consumed (J)")
    ax.set_ylabel("Generation")
    ax.set_zlabel("Gates Passed")
    ax.set_title("3D Pareto Front Evolution")
    
    ax.view_init(elev=20., azim=-45)
    
    out_png = os.path.join(out_dir, "pareto_plot4_3d.png")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Plot 4 different views of pareto fronts.")
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

    _SENTINEL_THRESHOLD = 1e8
    df = df[df["total_energy_j"] < _SENTINEL_THRESHOLD]

    gens = sorted(df["generation"].unique())
    if not gens:
        print("No valid generation data found.")
        return

    print("Generating Plot 1: Smooth Lines...")
    plot_smooth_lines(df, gens, out_dir)
    
    print("Generating Plot 2: Density...")
    plot_density(df, gens, out_dir)
    
    print("Generating Plot 3: Filled Areas...")
    plot_filled_areas(df, gens, out_dir)
    
    print("Generating Plot 4: 3D Evolution...")
    plot_3d_surface(df, gens, out_dir)
    
    print("All plots generated successfully!")

if __name__ == "__main__":
    main()
