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

def main():
    parser = argparse.ArgumentParser(description="Plot diversity evolution over generations.")
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

    # Drop sentinel energy values (1e9 = failed)
    _SENTINEL_THRESHOLD = 1e8
    df = df[df["total_energy_j"] < _SENTINEL_THRESHOLD]

    gens = sorted(df["generation"].unique())
    if not gens:
        print("No valid generation data found.")
        return

    # Normalize objectives for fair diversity calculation
    w_min, w_max = df["waypoints"].min(), df["waypoints"].max()
    e_min, e_max = df["total_energy_j"].min(), df["total_energy_j"].max()

    w_range = max(w_max - w_min, 1e-6)
    e_range = max(e_max - e_min, 1e-6)

    df["w_norm"] = (df["waypoints"] - w_min) / w_range
    df["e_norm"] = (df["total_energy_j"] - e_min) / e_range

    diversities = []
    for g in gens:
        pop = df[df["generation"] == g]
        # Calculate diversity of the Pareto front if rank info is present
        if "rank" in pop.columns and pop["rank"].notna().any():
            pop = pop[pop["rank"] == 0]

        if len(pop) < 2:
            diversities.append(0.0)
            continue

        # Diversity = Average distance of points to the centroid in normalized objective space
        w_centroid = pop["w_norm"].mean()
        e_centroid = pop["e_norm"].mean()

        distances = np.sqrt((pop["w_norm"] - w_centroid)**2 + (pop["e_norm"] - e_centroid)**2)
        diversities.append(distances.mean())

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(gens, diversities, marker='o', linestyle='-', color='indigo', linewidth=2)
    
    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Diversity (Average distance to centroid)", fontsize=12)
    ax.set_title("Pareto Front Diversity Over Generations", fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.6)

    fig.tight_layout()
    out_png = os.path.join(out_dir, "pareto_diversity_evolution.png")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"Saved diversity plot -> {out_png}")
    plt.close(fig)

if __name__ == "__main__":
    main()
