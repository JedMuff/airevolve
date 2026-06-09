import argparse
import ast
import os

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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

def main():
    parser = argparse.ArgumentParser(description="Plot 3D Pareto Fronts over generations.")
    parser.add_argument("csv_file", help="Path to evolution_data.csv")
    parser.add_argument("--interactive", action="store_true", help="Show interactive plot instead of saving to PNG (requires GUI).")
    args = parser.parse_args()

    if not args.interactive:
        matplotlib.use("Agg")
    else:
        # Use MacOS compatible interactive backend if running locally
        try:
            matplotlib.use("MacOSX")
        except:
            matplotlib.use("TkAgg")

    csv_path = args.csv_file
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    out_dir = os.path.dirname(os.path.abspath(csv_path))
    df = _load_csv(csv_path)

    finite_mask = np.isfinite(df["total_energy_j"]) & np.isfinite(df["waypoints"])
    df = df[finite_mask]

    # Drop sentinel energy values (1e9 = failed)
    _SENTINEL_THRESHOLD = 1e8
    df = df[df["total_energy_j"] < _SENTINEL_THRESHOLD]

    gens = sorted(df["generation"].unique())
    if not gens:
        print("No valid generation data found.")
        return

    fig = plt.figure(figsize=(20, 16))
    cmap = plt.get_cmap("viridis", len(gens))
    
    angles = [0, 90, 180, 270]
    
    for idx, angle in enumerate(angles, start=1):
        ax = fig.add_subplot(2, 2, idx, projection='3d')
        
        for i, g in enumerate(gens):
            pop = df[df["generation"] == g]
            # Plot only the Pareto front (rank 0) if rank info exists
            if "rank" in pop.columns and pop["rank"].notna().any():
                pop = pop[pop["rank"] == 0]

            if pop.empty:
                continue

            x_gen = np.full(len(pop), g)
            y_energy = pop["total_energy_j"]
            z_gates = pop["waypoints"]

            # Plot scatter points for the generation
            ax.scatter(x_gen, y_energy, z_gates, color=cmap(i), alpha=0.8, s=40, edgecolors='k', linewidth=0.5)

        ax.set_xlabel("Generation")
        ax.set_ylabel("Total Energy (J)")
        ax.set_zlabel("Gates Passed")
        ax.set_title(f"Rotation: {angle}°")

        # View angle: look from different sides
        ax.view_init(elev=20, azim=angle)

    fig.suptitle("3D Evolution of Pareto Fronts (4 Views)", fontsize=20)

    # Add a single colorbar for the whole figure
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(gens), vmax=max(gens)))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Generation", fontsize=14)

    if args.interactive:
        print("Displaying interactive 3D plots...")
        plt.show()
    else:
        out_png = os.path.join(out_dir, "pareto_evolution_3d_4views.png")
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        print(f"Saved 3D plot -> {out_png}")

if __name__ == "__main__":
    main()
