#!/usr/bin/env python3
"""
Generate reward history plot from monitor.csv file.
This script creates the same figure.png that would be generated during training.
"""

import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def generate_reward_plot(results_dir, output_name="figure.pdf"):
    """
    Generate reward history plot from monitor.csv
    
    Args:
        results_dir: Directory containing monitor.csv
        output_name: Name of output plot file
    """
    monitor_file = os.path.join(results_dir, "monitor.csv")
    
    if not os.path.exists(monitor_file):
        raise FileNotFoundError(f"monitor.csv not found in {results_dir}")
    
    # Load training data
    data = pd.read_csv(monitor_file, skiprows=1)  # Skip the first row (comments)
    episode_rewards = data["r"]  # Rewards per episode
    episode_lengths = data["l"]  # Episode lengths in simulation steps
    cumulative_timesteps = episode_lengths.cumsum()  # Cumulative simulation timesteps
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_timesteps, episode_rewards, label="Episode Reward", alpha=0.7)
    plt.xlabel("Simulation Timesteps")
    plt.ylabel("Reward")
    plt.title("Reward per Episode")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    output_path = os.path.join(results_dir, output_name)
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close()
    
    # Print summary statistics
    print(f"\nTraining Summary:")
    print(f"  Total episodes: {len(episode_rewards)}")
    print(f"  Final reward: {episode_rewards.iloc[-1]:.2f}")
    print(f"  Mean reward (last 100 episodes): {episode_rewards.iloc[-100:].mean():.2f}")
    print(f"  Std reward (last 100 episodes): {episode_rewards.iloc[-100:].std():.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate reward history plot from monitor.csv")
    parser.add_argument("--results-dir", type=str, required=True,
                        help="Directory containing monitor.csv")
    parser.add_argument("--output-name", type=str, default="figure.pdf",
                        help="Name of output plot file (default: figure.pdf)")
    
    args = parser.parse_args()
    
    generate_reward_plot(args.results_dir, args.output_name)
