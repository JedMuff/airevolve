import os
import ast
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def main():
    base_dir = "/Users/mikolajduchlinski/Desktop/results_final/results"
    run_names = [f"exp_lamarckian_ppo_power_ea_rep{i}" for i in range(1, 6)]
    
    all_max_g = {}
    all_min_e = {}
    
    for run in run_names:
        csv_path = os.path.join(base_dir, run, "evolution_data.csv")
        if not os.path.exists(csv_path):
            print(f"File not found: {csv_path}")
            continue
            
        print(f"Processing {run}...")
        df = pd.read_csv(csv_path)
        
        # Some scripts might not have "gates_passed" and "total_energy_j" already parsed
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
            max_gates = pop["gates_passed"].max()
            min_energy = pop["total_energy_j"].apply(lambda x: x if np.isfinite(x) else np.nan).min()
            
            if g not in all_max_g:
                all_max_g[g] = []
                all_min_e[g] = []
                
            all_max_g[g].append(max_gates)
            all_min_e[g].append(min_energy)
            
    gens = sorted(all_max_g.keys())
    if not gens:
        print("No valid generation data found across runs.")
        return
        
    # Calculate means and standard deviations
    mean_max_g = np.array([np.mean(all_max_g[g]) for g in gens])
    std_max_g = np.array([np.std(all_max_g[g]) for g in gens])
    
    mean_min_e = np.array([np.mean(all_min_e[g]) for g in gens])
    std_min_e = np.array([np.std(all_min_e[g]) for g in gens])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot Max Gates Passed
    ax1.plot(gens, mean_max_g, marker="o", markersize=3, linewidth=1.5, color="steelblue", label="Mean Max Gates")
    ax1.fill_between(gens, mean_max_g - std_max_g, mean_max_g + std_max_g, color="steelblue", alpha=0.2, label="± 1 Std Dev")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Max Gates Passed")
    ax1.set_title("Task Performance over Generations (5 Runs)")
    ax1.grid(True, linestyle="--", alpha=0.4)

    # Plot Min Energy
    ax2.plot(gens, mean_min_e, marker="o", markersize=3, linewidth=1.5, color="darkorange", label="Mean Min Energy")
    ax2.fill_between(gens, mean_min_e - std_min_e, mean_min_e + std_min_e, color="darkorange", alpha=0.2, label="± 1 Std Dev")
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Min Energy (J)")
    ax2.set_title("Best Power Efficiency over Generations (5 Runs)")
    ax2.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle("Lamarckian Standard-PPO + Power-Aware NSGA-II", fontsize=11)
    fig.tight_layout()
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "objectives_over_generations_all.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Objectives plot -> {out_path}")

if __name__ == "__main__":
    main()
