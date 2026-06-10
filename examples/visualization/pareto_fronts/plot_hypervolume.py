import argparse
import ast
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

def calculate_hypervolume(pf, ref_x, ref_y):
    # pf is assumed to be sorted by total_energy_j ascending
    if pf.empty:
        return 0.0
        
    x = pf["total_energy_j"].values
    y = pf["waypoints"].values
    
    hv = 0.0
    for i in range(len(x)):
        # Calculate width to the next point, or to the reference point if it's the last one
        width = (x[i+1] if i + 1 < len(x) else ref_x) - x[i]
        # Calculate height from the reference point
        height = y[i] - ref_y
        
        # Only add positive areas (if a point is worse than reference, it doesn't contribute)
        if width > 0 and height > 0:
            hv += width * height
            
    return hv

def main():
    parser = argparse.ArgumentParser(description="Plot hypervolume over generations.")
    parser.add_argument("csv_file", help="Path to evolution_data.csv")
    parser.add_argument("--ref_x", type=float, default=None, help="Reference X (Energy). Default is max valid energy * 1.05")
    parser.add_argument("--ref_y", type=float, default=None, help="Reference Y (Waypoints). Default is min valid waypoints")
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
        
    # Determine reference points automatically if not provided
    ref_x = args.ref_x if args.ref_x is not None else df["total_energy_j"].max() * 1.05
    ref_y = args.ref_y if args.ref_y is not None else min(0, df["waypoints"].min())

    print(f"Using reference point: Energy={ref_x:.2f}, Waypoints={ref_y:.2f}")

    hvs = []
    valid_gens = []

    for g in gens:
        pop = df[df["generation"] == g]
        pf = get_pareto_front(pop)
        
        if pf.empty:
            continue
            
        # Sort ascending by energy
        pf = pf.sort_values("total_energy_j")
        
        hv = calculate_hypervolume(pf, ref_x, ref_y)
        hvs.append(hv)
        valid_gens.append(g)

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(valid_gens, hvs, marker='o', linestyle='-', color='b', linewidth=2, markersize=6)
    
    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel(f"Hypervolume\n(Ref: {ref_x:.1f} J, {ref_y:.1f} gates)", fontsize=12)
    ax.set_title("Hypervolume Evolution Over Generations", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.7)
    
    # Add a trendline if we have enough points
    if len(valid_gens) > 2:
        z = np.polyfit(valid_gens, hvs, 1)
        p = np.poly1d(z)
        ax.plot(valid_gens, p(valid_gens), "r--", alpha=0.8, label=f"Trend: {z[0]:.2e}/gen")
        ax.legend()
    
    fig.tight_layout()
    out_png = os.path.join(out_dir, "hypervolume_evolution.png")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"Saved hypervolume plot -> {out_png}")
    plt.close(fig)

if __name__ == "__main__":
    main()
