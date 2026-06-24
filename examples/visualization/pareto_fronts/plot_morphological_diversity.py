import argparse
import ast
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def _load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    
    # Safely parse the genome column from string back into numeric arrays
    def parse_genome(g_str):
        if isinstance(g_str, str):
            try:
                # Use regex to extract all numbers (handles standard lists, numpy strings, and object reprs)
                floats = re.findall(r"-?\d+\.\d*(?:[eE][-+]?\d+)?|-?\.\d+|-?\d+", g_str)
                if not floats:
                    return np.nan
                return np.array([float(x) for x in floats])
            except Exception:
                return np.nan
        return g_str
        
    if "genome" not in df.columns:
        raise ValueError(f"{path}: missing 'genome' column required for morphological diversity.")
        
    df["genome_parsed"] = df["genome"].apply(parse_genome)
    # Drop rows where genome parsing failed
    df = df.dropna(subset=["genome_parsed"])
    
    return df

def main():
    parser = argparse.ArgumentParser(description="Plot morphological diversity over generations.")
    parser.add_argument("csv_file", help="Path to evolution_data.csv")
    args = parser.parse_args()

    csv_path = args.csv_file
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    out_dir = os.path.dirname(os.path.abspath(csv_path))
    df = _load_csv(csv_path)

    gens = sorted(df["generation"].unique())
    if not gens:
        print("No valid generation data found.")
        return

    valid_gens = []
    mean_diversities = []
    std_diversities = []

    for g in gens:
        pop = df[df["generation"] == g]
        
        # Stack all genomes into a 2D matrix (individuals x genes)
        try:
            pop_genomes = np.stack(pop["genome_parsed"].values)
        except ValueError:
            # Skip if genomes are not uniform length (rare, but possible if corrupted)
            continue
            
        if len(pop_genomes) < 2:
            continue

        # Morphological Diversity calculation:
        # Calculate the centroid (average genome) of the population
        centroid = np.mean(pop_genomes, axis=0)
        
        # Calculate Euclidean distance of each drone's physical body to the centroid
        distances = np.linalg.norm(pop_genomes - centroid, axis=1)
        
        mean_div = np.mean(distances)
        std_div = np.std(distances)
        
        valid_gens.append(g)
        mean_diversities.append(mean_div)
        std_diversities.append(std_div)

    gens = np.array(valid_gens)
    mean_diversities = np.array(mean_diversities)
    std_diversities = np.array(std_diversities)

    # Recreate the style of the supervisor's plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot the mean diversity line in red
    ax.plot(gens, mean_diversities, color='red', linewidth=2, label="Mean Morphological Diversity")
    
    # Add shaded region for standard deviation
    ax.fill_between(gens, 
                    mean_diversities - std_diversities, 
                    mean_diversities + std_diversities, 
                    color='red', alpha=0.2, label="± 1 Std Dev")
    
    ax.set_xlabel("Generation", fontsize=14)
    ax.set_ylabel("Diversity (Euclidean Distance)", fontsize=14)
    ax.set_title("Morphological Diversity Over Generations", fontsize=15)
    
    # Format grid and ticks similar to the requested style
    ax.grid(True, linestyle="-", alpha=0.7)
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # Optional: adjust y-limit if the bottom hits 0
    y_min = max(0, min(mean_diversities - std_diversities) * 0.9)
    y_max = max(mean_diversities + std_diversities) * 1.1
    ax.set_ylim([y_min, y_max])

    fig.tight_layout()
    out_png = os.path.join(out_dir, "morphological_diversity.png")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"Saved Morphological Diversity plot -> {out_png}")
    plt.close(fig)

if __name__ == "__main__":
    main()
