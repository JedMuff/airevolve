import os
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20, 'axes.titlesize': 24, 'axes.labelsize': 22, 'xtick.labelsize': 18, 'ytick.labelsize': 18, 'legend.fontsize': 18})
from scipy.optimize import linear_sum_assignment

def parse_genome(genome_str):
    start = genome_str.find('[[')
    end = genome_str.find(']]', start) + 2
    if start == -1 or end == 1:
        return None
    arms_str = genome_str[start:end]
    numbers = re.findall(r'[-+]?\d*\.\d+|[-+]?\d+', arms_str)
    if not numbers:
        return None
    try:
        arms = np.array([float(n) for n in numbers]).reshape(-1, 6)
        return arms
    except ValueError:
        return None

def angular_diff(a, b):
    return np.abs((a - b + np.pi) % (2 * np.pi) - np.pi)

def calc_central_asymmetry(arms):
    # Using Standard Deviation in centimeters for better readability
    # Arms[:, 0] is in meters, so multiply by 100
    return np.std(arms[:, 0])

def calc_bilateral_asymmetry(arms):
    n_arms = len(arms)
    if n_arms == 0:
        return 0.0
    
    reflected_arms = np.copy(arms)
    # Mirror across X-axis: negate azimuth angles
    reflected_arms[:, 1] = -reflected_arms[:, 1]  # Arm Azimuth
    reflected_arms[:, 3] = -reflected_arms[:, 3]  # Motor Azimuth
    
    cost_matrix = np.zeros((n_arms, n_arms))
    for i in range(n_arms):
        for j in range(n_arms):
            diff = 0
            # Arm Azimuth, Arm Polar, Motor Azimuth, Motor Polar
            for k in [1, 2, 3, 4]:
                diff += angular_diff(arms[i, k], reflected_arms[j, k])
            # Average the difference across the 4 parameters
            cost_matrix[i, j] = diff / 4.0
            
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return cost_matrix[row_ind, col_ind].sum() / n_arms

def process_run(csv_path):
    df = pd.read_csv(csv_path)
    generations = []
    central_asym_means = []
    bilateral_asym_means = []
    
    for gen, group in df.groupby('generation'):
        central_asyms = []
        bilateral_asyms = []
        for genome_str in group['genome']:
            arms = parse_genome(str(genome_str))
            if arms is not None:
                central_asyms.append(calc_central_asymmetry(arms))
                bilateral_asyms.append(calc_bilateral_asymmetry(arms))
        
        if central_asyms:
            generations.append(gen)
            central_asym_means.append(np.mean(central_asyms))
            bilateral_asym_means.append(np.mean(bilateral_asyms))
        
    return generations, central_asym_means, bilateral_asym_means

def main():
    parser = argparse.ArgumentParser(description="Plot symmetry metrics for all runs of a task.")
    parser.add_argument("--base_dir", required=True, help="Base results directory")
    parser.add_argument("--task_name", required=True, help="Name of the task (e.g. figure8)")
    parser.add_argument("--run_dirs", nargs='+', required=True, help="List of run directories")
    args = parser.parse_args()
    
    all_central = {}
    all_bilateral = {}
    
    for run_dir in args.run_dirs:
        csv_path = os.path.join(run_dir, "evolution_data_repaired.csv")
        run_name = os.path.basename(run_dir)
        if os.path.exists(csv_path):
            print(f"Processing {run_name}...")
            gens, c_means, b_means = process_run(csv_path)
            all_central[run_name] = pd.Series(c_means, index=gens)
            all_bilateral[run_name] = pd.Series(b_means, index=gens)
        else:
            print(f"File not found: {csv_path}")
            
    df_central = pd.DataFrame(all_central)
    df_bilateral = pd.DataFrame(all_bilateral)
    
    # Plot 1: Central Asymmetry
    plt.figure(figsize=(10, 6))
    for rep in df_central.columns:
        plt.plot(df_central.index, df_central[rep], alpha=0.3, color='#4c72b0')
    plt.plot(df_central.index, df_central.mean(axis=1), linewidth=2, color='#4c72b0', label='Mean')
    plt.fill_between(df_central.index, 
                     df_central.mean(axis=1) - df_central.std(axis=1),
                     df_central.mean(axis=1) + df_central.std(axis=1), 
                     color='#4c72b0', alpha=0.2, label='Std Dev')
    plt.title(f'Central Asymmetry over Generations ({args.task_name})', fontweight='bold')
    plt.xlabel('Generations')
    plt.ylabel('Central Asymmetry')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    out_path_central = os.path.join(args.base_dir, f'central_asymmetry_{args.task_name}.png')
    plt.savefig(out_path_central, dpi=300)
    plt.close()
    
    # Plot 2: Bilateral Asymmetry
    plt.figure(figsize=(10, 6))
    for rep in df_bilateral.columns:
        plt.plot(df_bilateral.index, df_bilateral[rep], alpha=0.3, color='#c44e52')
    plt.plot(df_bilateral.index, df_bilateral.mean(axis=1), linewidth=2, color='#c44e52', label='Mean')
    plt.fill_between(df_bilateral.index, 
                     df_bilateral.mean(axis=1) - df_bilateral.std(axis=1),
                     df_bilateral.mean(axis=1) + df_bilateral.std(axis=1), 
                     color='#c44e52', alpha=0.2, label='Std Dev')
    plt.title(f'Bilateral Asymmetry over Generations ({args.task_name})', fontweight='bold')
    plt.xlabel('Generations')
    plt.ylabel('Bilateral Asymmetry')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    out_path_bilateral = os.path.join(args.base_dir, f'bilateral_asymmetry_{args.task_name}.png')
    plt.savefig(out_path_bilateral, dpi=300)
    plt.close()
    
    print(f"Plots saved as {out_path_central} and {out_path_bilateral}")

if __name__ == "__main__":
    main()
