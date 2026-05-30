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

def _load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Handle older CSVs that used 'waypoints' instead of 'gates_passed'
    if "gates_passed" not in df.columns and "waypoints" in df.columns:
        df = df.rename(columns={"waypoints": "gates_passed"})

    if "gates_passed" not in df.columns or "total_energy_j" not in df.columns:
        if "fitness" not in df.columns:
            raise ValueError(f"{path}: need 'fitness' column or ('gates_passed','total_energy_j') columns")
        parsed = df["fitness"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
        df["gates_passed"] = parsed.apply(lambda t: float(t[0]))
        df["total_energy_j"] = parsed.apply(lambda t: float(t[1]))
        
    df["gates_passed"] = pd.to_numeric(df["gates_passed"], errors="coerce")
    df["total_energy_j"] = pd.to_numeric(df["total_energy_j"], errors="coerce")
    return df

def main():
    parser = argparse.ArgumentParser(description="Plot every unique individual across all generations.")
    parser.add_argument("csv_file", help="Path to evolution_data.csv")
    args = parser.parse_args()

    csv_path = args.csv_file
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    out_dir = os.path.dirname(os.path.abspath(csv_path))
    df = _load_csv(csv_path)

    finite_mask = np.isfinite(df["total_energy_j"]) & np.isfinite(df["gates_passed"])
    df = df[finite_mask]
    
    # 1. Drop duplicates so each individual ID is only plotted once. 
    # Since they survive with the same ID, their fitness doesn't change.
    df_unique = df.drop_duplicates(subset=["id"], keep="first")

    gens = sorted(df_unique["generation"].unique())
    if not gens:
        print("No valid generation data found.")
        return

    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 2. Use a colormap to show time (the generation they were born in)
    scatter = ax.scatter(
        df_unique["total_energy_j"], 
        df_unique["gates_passed"], 
        c=df_unique["generation"], 
        cmap="viridis", 
        alpha=0.6, 
        s=30, 
        edgecolors="none",
        zorder=2
    )

    # 3. Highlight the absolute best (Rank 0 of the final generation)
    # We look at the original dataframe for the final generation to find the final Pareto front
    final_gen = df["generation"].max()
    final_pop = df[df["generation"] == final_gen]
    if "rank" in final_pop.columns:
        pareto_front = final_pop[final_pop["rank"] == 0]
        ax.scatter(
            pareto_front["total_energy_j"], 
            pareto_front["gates_passed"], 
            facecolors="none", 
            edgecolors="red", 
            s=80, 
            linewidth=1.5,
            label=f"Final Pareto Front (Gen {int(final_gen)})",
            zorder=3
        )

    ax.set_xlabel("Total Energy Consumed (J)  ← Minimize", fontsize=12)
    ax.set_ylabel("Gates Passed  ↑ Maximize", fontsize=12)
    ax.set_title("Full Swarm Evolution: Every Unique Individual Explored", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.4, zorder=1)

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Generation Born", fontsize=11)
    
    ax.legend(loc="best")

    fig.tight_layout()
    out_png = os.path.join(out_dir, "swarm_evolution_all.png")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"Saved full swarm plot -> {out_png}")
    plt.close(fig)

if __name__ == "__main__":
    main()
