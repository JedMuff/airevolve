import os
import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    # Directory containing the training results
    results_dir = "results_training/figure8"
    
    # Find all monitor.csv files in subdirectories
    # This will match results_training/figure8/*/monitor.csv
    monitor_files = glob.glob(os.path.join(results_dir, "*", "monitor.csv"))
    
    if not monitor_files:
        print(f"No monitor.csv files found in {results_dir}/*/")
        return
        
    # Sort files so the legend looks organized
    monitor_files = sorted(monitor_files)
    
    plt.figure(figsize=(12, 8))
    
    # Set up a colormap to have different colors for each line
    cmap = plt.get_cmap('tab10')
    colors = cmap(np.linspace(0, 1, len(monitor_files)))
    
    for idx, csv_path in enumerate(monitor_files):
        # Extract the directory name (e.g., 'individual_0001' or 'standard_hexa')
        dir_name = os.path.basename(os.path.dirname(csv_path))
        
        try:
            # Read monitor.csv (stable baselines format usually has comments at the top)
            df = pd.read_csv(csv_path, comment='#')
            
            if len(df) == 0 or 'r' not in df.columns or 'l' not in df.columns:
                print(f"Skipping {dir_name}: Invalid or empty monitor.csv")
                continue
                
            r = df['r'].values
            l = df['l'].values
            
            # Use cumulative timesteps for X axis (common in SB3), or just episodes
            # The prompt mentions "reward per episode" so we plot the episode reward (Y) 
            # against episodes (X). If you prefer timesteps, uncomment the next line and change x below.
            # x = np.cumsum(l)
            x = np.arange(len(r))
            
            # Smooth the reward to get "average value lines" instead of noisy raw data
            window_size = min(100, max(1, len(r) // 10)) # Adaptive window size
            r_smooth = pd.Series(r).rolling(window=100, min_periods=1).mean().values
            
            # Plot only the smoothed average line (no std deviation area)
            plt.plot(x, r_smooth, label=dir_name, color=colors[idx], linewidth=2)
            
        except Exception as e:
            print(f"Error processing {dir_name}: {e}")
            
    plt.title("Reward per Episode for All Drones", fontsize=15, fontweight='bold')
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Smoothed Reward", fontsize=12)
    
    # Add a legend on the side
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., title="Drones")
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the plot
    out_file = "combined_rewards.png"
    plt.savefig(out_file, dpi=300)
    print(f"Successfully saved combined plot to {out_file}")

if __name__ == "__main__":
    main()
