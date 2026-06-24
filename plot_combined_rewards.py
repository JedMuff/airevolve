import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def load_data(file_path):
    # skip the first row (comment with json)
    df = pd.read_csv(file_path, skiprows=1)
    return df['r'].values

def plot_combined():
    dir1 = 'results_training/figure8/standard_hexa'
    dir2 = 'results_training/figure8/individual_1272'

    r1 = load_data(os.path.join(dir1, 'monitor.csv'))
    r2 = load_data(os.path.join(dir2, 'monitor.csv'))

    min_len = min(len(r1), len(r2))
    r1 = r1[:min_len]
    r2 = r2[:min_len]

    window_size = 1000
    
    def moving_avg_std(data, window):
        df = pd.Series(data)
        ma = df.rolling(window=window).mean().values
        std = df.rolling(window=window).std().values
        return ma, std

    ma1, std1 = moving_avg_std(r1, window_size)
    ma2, std2 = moving_avg_std(r2, window_size)

    # Plotting
    plt.figure(figsize=(10, 6))
    
    x1 = np.arange(len(r1))
    plt.plot(x1, ma1, label='Standard Hexa', color='blue')
    plt.fill_between(x1, ma1 - std1, ma1 + std1, color='blue', alpha=0.2)

    x2 = np.arange(len(r2))
    plt.plot(x2, ma2, label='Best evolved drone', color='orange')
    plt.fill_between(x2, ma2 - std2, ma2 + std2, color='orange', alpha=0.2)

    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title('Combined Reward Function')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('combined_rewards_plot.png')
    
    print(f"Max reward for Standard Hexa: {np.max(r1):.2f}")
    print(f"Max reward for Individual 1272: {np.max(r2):.2f}")

if __name__ == '__main__':
    plot_combined()
