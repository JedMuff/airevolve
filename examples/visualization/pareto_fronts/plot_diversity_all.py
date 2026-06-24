import os
import re
import argparse
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
    parser = argparse.ArgumentParser(description="Plot morphological diversity for all runs of a task.")
    parser.add_argument("--base_dir", required=True, help="Base results directory")
    parser.add_argument("--task_name", required=True, help="Name of the task (e.g. figure8)")
    parser.add_argument("--run_dirs", nargs='+', required=True, help="List of run directories")
    args = parser.parse_args()
    
    all_run_distances = {} # gen -> list of all distances
    
    for run in args.run_dirs:
        csv_path = os.path.join(run, "evolution_data_repaired.csv")
        if not os.path.exists(csv_path):
            print(f"File not found: {csv_path}")
            continue
            
        print(f"Processing {os.path.basename(run)}...")
        df = _load_csv(csv_path)
        gens = sorted(df["generation"].unique())
        
        for g in gens:
            pop = df[df["generation"] == g]
            
            # Stack all genomes into a 2D matrix (individuals x genes)
            try:
                pop_genomes = np.stack(pop["genome_parsed"].values)
            except ValueError:
                continue
                
            if len(pop_genomes) < 2:
                continue
                
            # Morphological Diversity calculation:
            # Calculate the centroid (average genome) of the population
            centroid = np.mean(pop_genomes, axis=0)
            
            # Calculate Euclidean distance of each drone's physical body to the centroid
            distances = np.linalg.norm(pop_genomes - centroid, axis=1)
            
            if g not in all_run_distances:
                all_run_distances[g] = []
            all_run_distances[g].extend(distances.tolist())
            
    gens = sorted(all_run_distances.keys())
    if not gens:
        print("No valid generation data found across runs.")
        return
        
    mean_of_means = []
    std_of_means = []
    
    for g in gens:
        dists = all_run_distances[g]
        mean_of_means.append(np.mean(dists))
        std_of_means.append(np.std(dists))
        
    gens = np.array(gens)
    mean_of_means = np.array(mean_of_means)
    std_of_means = np.array(std_of_means)
    
    # Recreate the style of the supervisor's plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot the mean diversity line in red
    ax.plot(gens, mean_of_means, color='red', linewidth=2, label="Mean Morphological Diversity")
    
    # Add shaded region for standard deviation
    ax.fill_between(gens, 
                    mean_of_means - std_of_means, 
                    mean_of_means + std_of_means, 
                    color='red', alpha=0.2, label="± 1 Std Dev")
    
    ax.set_xlabel("Generation", fontsize=14)
    ax.set_ylabel("Diversity (Euclidean Distance)", fontsize=14)
    ax.set_title(f"Morphological Diversity Over Generations ({len(args.run_dirs)} Runs - {args.task_name})", fontsize=15)
    
    # Format grid and ticks similar to the requested style
    ax.grid(True, linestyle="-", alpha=0.7)
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # Optional: adjust y-limit if the bottom hits 0
    y_min = max(0, min(mean_of_means - std_of_means) * 0.9)
    y_max = max(mean_of_means + std_of_means) * 1.1
    ax.set_ylim([y_min, y_max])

    fig.tight_layout()
    out_path = os.path.join(args.base_dir, f"morphological_diversity_all_{args.task_name}.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved Morphological Diversity plot -> {out_path}")
    plt.close(fig)

if __name__ == "__main__":
    main()
