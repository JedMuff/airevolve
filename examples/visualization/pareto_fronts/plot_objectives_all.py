import os
import ast
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Plot objectives for all runs of a task.")
    parser.add_argument("--base_dir", required=True, help="Base results directory")
    parser.add_argument("--task_name", required=True, help="Name of the task (e.g. figure8)")
    parser.add_argument("--run_dirs", nargs='+', required=True, help="List of run directories")
    args = parser.parse_args()

    # Per-generation aggregates across runs
    all_max_g = {}        # gen -> [max_gates per run]  (for dashed line)
    all_mean_g = {}       # gen -> [mean_gates per run] (for solid line)
    all_gates_pooled = {} # gen -> [ALL individual gates from ALL runs] (for population std)
    all_min_e = {}        # gen -> [min_energy per run]  (for dashed line)
    all_mean_e = {}       # gen -> [mean_energy per run] (for solid line)
    all_energy_pooled = {} # gen -> [ALL individual energies from ALL runs] (for population std)
    
    for run in args.run_dirs:
        csv_path = os.path.join(run, "evolution_data_repaired.csv")
        if not os.path.exists(csv_path):
            print(f"File not found: {csv_path}")
            continue
            
        print(f"Processing {os.path.basename(run)}...")
        df = pd.read_csv(csv_path)
        
        # Parse fitness if needed
        if "gates_passed" not in df.columns or "total_energy_j" not in df.columns:
            if "fitness" in df.columns:
                def parse_fitness(f, idx):
                    try:
                        val = ast.literal_eval(f) if isinstance(f, str) else f
                        return val[idx]
                    except Exception:
                        return np.nan
                        
                df["gates_passed"] = df["fitness"].apply(lambda f: parse_fitness(f, 0))
                df["total_energy_j"] = df["fitness"].apply(lambda f: parse_fitness(f, 1))
            else:
                print(f"Missing fitness/gates/energy columns in {csv_path}")
                continue
                
        gens = sorted(df["generation"].unique())
        for g in gens:
            pop = df[df["generation"] == g]
            
            # Max/Min per run (for dashed lines)
            max_gates = pop["gates_passed"].max()
            energy_vals = pop["total_energy_j"].apply(lambda x: x if np.isfinite(x) and x < 1e8 else np.nan)
            min_energy = energy_vals.min()
            
            # Mean per run (for solid lines showing variance across runs instead of pooled population)
            mean_gates = pop["gates_passed"].mean()
            mean_energy = energy_vals.mean()
            
            if g not in all_max_g:
                all_max_g[g] = []
                all_mean_g[g] = []
                all_gates_pooled[g] = []
                all_min_e[g] = []
                all_mean_e[g] = []
                all_energy_pooled[g] = []
                
            all_max_g[g].append(max_gates)
            all_min_e[g].append(min_energy)
            all_mean_g[g].append(mean_gates)
            all_mean_e[g].append(mean_energy)
            
            # Pool ALL individual values (for population std)
            all_gates_pooled[g].extend(pop["gates_passed"].dropna().tolist())
            all_energy_pooled[g].extend(energy_vals.dropna().tolist())
            
    gens = sorted(all_max_g.keys())
    if not gens:
        print("No valid generation data found across runs.")
        return
        
    # Max/Min: mean and std across the 5 runs (for dashed lines, run-to-run variance)
    mean_of_max_g = np.array([np.mean(all_max_g[g]) for g in gens])
    std_of_max_g  = np.array([np.std(all_max_g[g]) for g in gens])
    
    mean_of_min_e = np.array([np.nanmean(all_min_e[g]) for g in gens])
    std_of_min_e  = np.array([np.nanstd(all_min_e[g]) for g in gens])
    
    # Mean: computed over the run-means
    mean_of_mean_g = np.array([np.mean(all_mean_g[g]) for g in gens])
    mean_of_mean_e = np.array([np.nanmean(all_mean_e[g]) for g in gens])
    
    # Std: computed over ALL pooled individuals to show population spread
    std_of_mean_g  = np.array([np.std(all_gates_pooled[g]) for g in gens])
    std_of_mean_e  = np.array([np.nanstd(all_energy_pooled[g]) for g in gens])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    color_gates = "tab:red"
    color_energy = "tab:red"
    
    # ── Gates Passed ──
    # Mean (solid line + darker shading)
    ax1.plot(gens, mean_of_mean_g, linewidth=1.5, color=color_gates, linestyle="-", label="Mean")
    ax1.fill_between(gens, mean_of_mean_g - std_of_mean_g, mean_of_mean_g + std_of_mean_g,
                     color=color_gates, alpha=0.15)
    
    # Max (dashed line + lighter shading)
    ax1.plot(gens, mean_of_max_g, linewidth=1.5, color=color_gates, linestyle="--", label="Max")
    ax1.fill_between(gens, mean_of_max_g - std_of_max_g, mean_of_max_g + std_of_max_g,
                     color=color_gates, alpha=0.10)
    
    ax1.set_xlabel("Generation", fontsize=12)
    ax1.set_ylabel("Number of Waypoints Passed", fontsize=12)
    ax1.set_title(f"Task Performance over Generations ({len(args.run_dirs)} Runs)", fontsize=13)
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc="lower right", fontsize=10)

    # ── Power Efficiency ──
    # Mean (solid line + darker shading)
    ax2.plot(gens, mean_of_mean_e, linewidth=1.5, color=color_energy, linestyle="-", label="Mean")
    ax2.fill_between(gens, mean_of_mean_e - std_of_mean_e, mean_of_mean_e + std_of_mean_e,
                     color=color_energy, alpha=0.15)
    
    # Min (dashed line + lighter shading)
    ax2.plot(gens, mean_of_min_e, linewidth=1.5, color=color_energy, linestyle="--", label="Best (Min)")
    ax2.fill_between(gens, mean_of_min_e - std_of_min_e, mean_of_min_e + std_of_min_e,
                     color=color_energy, alpha=0.10)
    
    ax2.set_xlabel("Generation", fontsize=12)
    ax2.set_ylabel("Total Energy (J)", fontsize=12)
    ax2.set_title(f"Power Efficiency over Generations ({len(args.run_dirs)} Runs)", fontsize=13)
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc="upper right", fontsize=10)

    fig.suptitle(f"Standard-PPO + Power-Aware NSGA-II ({args.task_name})", fontsize=11)
    fig.tight_layout()
    out_path = os.path.join(args.base_dir, f"objectives_over_generations_all_{args.task_name}.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Objectives plot -> {out_path}")

if __name__ == "__main__":
    main()
