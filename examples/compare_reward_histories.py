#!/usr/bin/env python3
"""
Compare reward histories between baseline and optimized training runs.
Creates a side-by-side comparison plot.
"""

import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np

def smooth_curve(data, window=100):
    """Apply moving average smoothing"""
    return pd.Series(data).rolling(window=window, min_periods=1).mean()

def compare_reward_plots(baseline_dir, optimized_dir, output_dir=None):
    """
    Compare reward histories between baseline and optimized runs
    
    Args:
        baseline_dir: Directory containing baseline monitor.csv
        optimized_dir: Directory containing optimized monitor.csv
        output_dir: Directory to save comparison plot (default: optimized_dir)
    """
    if output_dir is None:
        output_dir = optimized_dir
    
    # Load baseline data
    baseline_file = os.path.join(baseline_dir, "monitor.csv")
    if not os.path.exists(baseline_file):
        raise FileNotFoundError(f"monitor.csv not found in {baseline_dir}")
    baseline_data = pd.read_csv(baseline_file, skiprows=1)
    
    # Load optimized data
    optimized_file = os.path.join(optimized_dir, "monitor.csv")
    if not os.path.exists(optimized_file):
        raise FileNotFoundError(f"monitor.csv not found in {optimized_dir}")
    optimized_data = pd.read_csv(optimized_file, skiprows=1)
    
    # Extract data
    baseline_rewards = baseline_data["r"]
    baseline_episode_lengths = baseline_data["l"]
    optimized_rewards = optimized_data["r"]
    optimized_episode_lengths = optimized_data["l"]
    
    # Calculate cumulative simulation timesteps
    baseline_cumulative_steps = baseline_episode_lengths.cumsum()
    optimized_cumulative_steps = optimized_episode_lengths.cumsum()
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Raw rewards
    axes[0].plot(baseline_cumulative_steps, baseline_rewards, label="Baseline", alpha=0.3, color='blue')
    axes[0].plot(optimized_cumulative_steps, optimized_rewards, label="Optimized", alpha=0.3, color='red')
    axes[0].plot(baseline_cumulative_steps, smooth_curve(baseline_rewards, 100), 
                 label="Baseline (100-episode avg)", linewidth=2, color='blue')
    axes[0].plot(optimized_cumulative_steps, smooth_curve(optimized_rewards, 100), 
                 label="Optimized (100-episode avg)", linewidth=2, color='red')
    axes[0].set_xlabel("Simulation Timesteps")
    axes[0].set_ylabel("Episode Reward")
    axes[0].set_title("Reward History Comparison: Baseline vs Optimized Hyperparameters")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Smoothed comparison only
    axes[1].plot(baseline_cumulative_steps, smooth_curve(baseline_rewards, 500), 
                 label="Baseline (500-episode avg)", linewidth=2.5, color='blue')
    axes[1].plot(optimized_cumulative_steps, smooth_curve(optimized_rewards, 500), 
                 label="Optimized (500-episode avg)", linewidth=2.5, color='red')
    axes[1].set_xlabel("Simulation Timesteps")
    axes[1].set_ylabel("Average Episode Reward")
    axes[1].set_title("Smoothed Reward Comparison (500-episode moving average)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, "comparison_plot.pdf")
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    print(f"Comparison plot saved to: {output_path}")
    plt.close()
    
    # Print comparison statistics
    print(f"\n{'='*60}")
    print(f"Training Comparison Summary")
    print(f"{'='*60}")
    
    print(f"\nBaseline ({baseline_dir}):")
    print(f"  Total episodes: {len(baseline_rewards)}")
    print(f"  Final reward: {baseline_rewards.iloc[-1]:.2f}")
    print(f"  Mean reward (last 100 episodes): {baseline_rewards.iloc[-100:].mean():.2f}")
    print(f"  Std reward (last 100 episodes): {baseline_rewards.iloc[-100:].std():.2f}")
    print(f"  Max reward: {baseline_rewards.max():.2f}")
    
    print(f"\nOptimized ({optimized_dir}):")
    print(f"  Total episodes: {len(optimized_rewards)}")
    print(f"  Final reward: {optimized_rewards.iloc[-1]:.2f}")
    print(f"  Mean reward (last 100 episodes): {optimized_rewards.iloc[-100:].mean():.2f}")
    print(f"  Std reward (last 100 episodes): {optimized_rewards.iloc[-100:].std():.2f}")
    print(f"  Max reward: {optimized_rewards.max():.2f}")
    
    # Calculate improvement
    baseline_mean = baseline_rewards.iloc[-100:].mean()
    optimized_mean = optimized_rewards.iloc[-100:].mean()
    improvement = ((optimized_mean - baseline_mean) / abs(baseline_mean)) * 100
    
    baseline_std = baseline_rewards.iloc[-100:].std()
    optimized_std = optimized_rewards.iloc[-100:].std()
    stability_improvement = ((baseline_std - optimized_std) / baseline_std) * 100
    
    print(f"\n{'='*60}")
    print(f"Improvement Analysis")
    print(f"{'='*60}")
    print(f"  Mean reward improvement: {improvement:+.1f}%")
    print(f"  Stability improvement: {stability_improvement:+.1f}%")
    
    if improvement > 0:
        print(f"  ✓ Optimized hyperparameters achieved better performance!")
    else:
        print(f"  ✗ Baseline performed better (optimization may need more trials)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare baseline vs optimized reward histories")
    parser.add_argument("--baseline", type=str, required=True,
                        help="Baseline results directory")
    parser.add_argument("--optimized", type=str, required=True,
                        help="Optimized results directory")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for comparison plot (default: optimized dir)")
    
    args = parser.parse_args()
    
    compare_reward_plots(args.baseline, args.optimized, args.output_dir)
