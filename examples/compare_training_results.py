#!/usr/bin/env python3
"""
Compare training results before and after hyperparameter optimization.
Creates side-by-side plots and statistical comparisons.
"""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from typing import List, Dict, Any
import seaborn as sns

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")


def load_training_data(result_dir: str) -> Dict[str, Any]:
    """Load training data from a result directory."""
    result_dir = Path(result_dir)
    
    data = {
        'directory': str(result_dir),
        'monitor_data': None,
        'final_fitness': None,
        'metadata': None,
        'results': None
    }
    
    # Load monitor data (training curve)
    monitor_files = list(result_dir.glob("*monitor.csv"))
    if monitor_files:
        try:
            monitor_data = pd.read_csv(monitor_files[0], skiprows=1)
            data['monitor_data'] = monitor_data
        except Exception as e:
            print(f"Warning: Could not load monitor data from {result_dir}: {e}")
    
    # Load final fitness
    fitness_file = result_dir / "fitness.txt"
    if fitness_file.exists():
        try:
            with open(fitness_file, 'r') as f:
                data['final_fitness'] = float(f.read().strip())
        except Exception as e:
            print(f"Warning: Could not load fitness from {result_dir}: {e}")
    
    # Load metadata
    metadata_file = result_dir / "metadata.pkl"
    if metadata_file.exists():
        try:
            with open(metadata_file, 'rb') as f:
                data['metadata'] = pickle.load(f)
        except Exception as e:
            print(f"Warning: Could not load metadata from {result_dir}: {e}")
    
    # Load results
    results_file = result_dir / "results.pkl"
    if results_file.exists():
        try:
            with open(results_file, 'rb') as f:
                data['results'] = pickle.load(f)
        except Exception as e:
            print(f"Warning: Could not load results from {result_dir}: {e}")
    
    return data


def calculate_statistics(monitor_data: pd.DataFrame) -> Dict[str, float]:
    """Calculate training statistics from monitor data."""
    if monitor_data is None or len(monitor_data) == 0:
        return {}
    
    rewards = monitor_data['r'].values
    timesteps = monitor_data['t'].values
    
    stats = {
        'final_reward_mean': np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards),
        'final_reward_std': np.std(rewards[-100:]) if len(rewards) >= 100 else np.std(rewards),
        'max_reward': np.max(rewards),
        'total_episodes': len(rewards),
        'total_timesteps': timesteps[-1] if len(timesteps) > 0 else 0,
        'reward_improvement': rewards[-1] - rewards[0] if len(rewards) > 1 else 0,
    }
    
    # Calculate convergence timestep (when reward first reaches 90% of max)
    if len(rewards) > 10:
        target_reward = 0.9 * stats['max_reward']
        convergence_idx = np.where(rewards >= target_reward)[0]
        if len(convergence_idx) > 0:
            stats['convergence_timestep'] = timesteps[convergence_idx[0]]
            stats['convergence_episode'] = convergence_idx[0]
        else:
            stats['convergence_timestep'] = timesteps[-1]
            stats['convergence_episode'] = len(rewards)
    
    # Fitness progression metrics
    cumulative_best = pd.Series(rewards).cummax()
    stats['best_fitness_ever'] = cumulative_best.iloc[-1]
    stats['episodes_to_first_gate'] = np.where(rewards >= 1.0)[0][0] if any(rewards >= 1.0) else len(rewards)
    stats['episodes_to_max_fitness'] = cumulative_best.idxmax() if stats['max_reward'] > 0 else len(rewards)
    
    # Learning stability metrics
    if len(rewards) >= 200:
        # Compare first half vs second half performance
        mid_point = len(rewards) // 2
        first_half_mean = np.mean(rewards[:mid_point])
        second_half_mean = np.mean(rewards[mid_point:])
        stats['learning_improvement'] = second_half_mean - first_half_mean
        stats['learning_stability'] = np.std(rewards[mid_point:]) / max(np.std(rewards[:mid_point]), 0.01)
    
    return stats


def plot_training_comparison(baseline_data: Dict, optimized_data: List[Dict], output_dir: str):
    """Create comparison plots of training curves."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle('Training Comparison: Baseline vs Optimized', fontsize=16, fontweight='bold')
    
    # Plot 1: Training curves (Episode Rewards)
    ax1 = axes[0, 0]
    
    # Baseline
    if baseline_data['monitor_data'] is not None:
        baseline_rewards = baseline_data['monitor_data']['r']
        baseline_episodes = range(len(baseline_rewards))
        # Smooth with rolling average
        baseline_smooth = baseline_rewards.rolling(window=min(50, len(baseline_rewards)//10), center=True).mean()
        ax1.plot(baseline_episodes, baseline_smooth, label='Baseline', color='red', linewidth=2)
        ax1.fill_between(baseline_episodes, 
                        baseline_rewards.rolling(window=min(50, len(baseline_rewards)//10)).min(),
                        baseline_rewards.rolling(window=min(50, len(baseline_rewards)//10)).max(),
                        alpha=0.2, color='red')
    
    # Optimized (multiple runs if provided)
    colors = ['blue', 'green', 'orange', 'purple', 'brown']
    for i, opt_data in enumerate(optimized_data[:5]):  # Max 5 runs for readability
        if opt_data['monitor_data'] is not None:
            opt_rewards = opt_data['monitor_data']['r']
            opt_episodes = range(len(opt_rewards))
            opt_smooth = opt_rewards.rolling(window=min(50, len(opt_rewards)//10), center=True).mean()
            
            label = f'Optimized' if i == 0 else f'Optimized Run {i+1}'
            ax1.plot(opt_episodes, opt_smooth, label=label, color=colors[i], linewidth=2)
            ax1.fill_between(opt_episodes,
                            opt_rewards.rolling(window=min(50, len(opt_rewards)//10)).min(),
                            opt_rewards.rolling(window=min(50, len(opt_rewards)//10)).max(),
                            alpha=0.2, color=colors[i])
    
    ax1.set_xlabel('Episode Number')
    ax1.set_ylabel('Episode Reward')
    ax1.set_title('Training Curves (Episode Rewards)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Fitness History Comparison (NEW!)
    ax2 = axes[0, 1]
    
    # Calculate cumulative best fitness over episodes
    if baseline_data['monitor_data'] is not None:
        baseline_rewards = baseline_data['monitor_data']['r']
        baseline_cumulative_best = baseline_rewards.cummax()
        baseline_episodes = range(len(baseline_rewards))
        ax2.plot(baseline_episodes, baseline_cumulative_best, 
                label='Baseline (Best So Far)', color='red', linewidth=3, linestyle='-')
        # Also plot recent performance (last 100 episodes average)
        baseline_recent = baseline_rewards.rolling(window=min(100, len(baseline_rewards)//5)).mean()
        ax2.plot(baseline_episodes, baseline_recent, 
                label='Baseline (Recent Avg)', color='red', linewidth=2, linestyle='--', alpha=0.7)
    
    for i, opt_data in enumerate(optimized_data[:3]):  # Max 3 for readability
        if opt_data['monitor_data'] is not None:
            opt_rewards = opt_data['monitor_data']['r']
            opt_cumulative_best = opt_rewards.cummax()
            opt_episodes = range(len(opt_rewards))
            
            label_best = f'Optimized (Best So Far)' if i == 0 else f'Opt {i+1} (Best)'
            label_recent = f'Optimized (Recent Avg)' if i == 0 else f'Opt {i+1} (Recent)'
            
            ax2.plot(opt_episodes, opt_cumulative_best, 
                    label=label_best, color=colors[i], linewidth=3, linestyle='-')
            
            opt_recent = opt_rewards.rolling(window=min(100, len(opt_rewards)//5)).mean()
            ax2.plot(opt_episodes, opt_recent, 
                    label=label_recent, color=colors[i], linewidth=2, linestyle='--', alpha=0.7)
    
    ax2.set_xlabel('Episode Number')
    ax2.set_ylabel('Fitness (Gates Passed)')
    ax2.set_title('Fitness History Progression')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Final performance comparison
    ax3 = axes[0, 2]
    
    final_rewards = []
    labels = []
    colors_bar = []
    
    if baseline_data['final_fitness'] is not None:
        final_rewards.append(baseline_data['final_fitness'])
        labels.append('Baseline')
        colors_bar.append('red')
    
    for i, opt_data in enumerate(optimized_data):
        if opt_data['final_fitness'] is not None:
            final_rewards.append(opt_data['final_fitness'])
            labels.append(f'Optimized {i+1}' if len(optimized_data) > 1 else 'Optimized')
            colors_bar.append(colors[i % len(colors)])
    
    if final_rewards:
        bars = ax3.bar(labels, final_rewards, color=colors_bar, alpha=0.7)
        ax3.set_ylabel('Final Fitness (Gates Passed)')
        ax3.set_title('Final Performance Comparison')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, final_rewards):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Learning Speed Comparison (NEW!)
    ax4 = axes[1, 0]
    
    # Calculate episodes to reach certain fitness thresholds
    thresholds = [1, 2, 3, 4, 5]  # Gates passed
    
    def episodes_to_threshold(rewards, threshold):
        """Find first episode where cumulative best reaches threshold."""
        cummax = rewards.cummax()
        indices = cummax[cummax >= threshold].index
        return indices[0] if len(indices) > 0 else len(rewards)
    
    if baseline_data['monitor_data'] is not None:
        baseline_rewards = baseline_data['monitor_data']['r']
        baseline_episodes_to_thresh = [episodes_to_threshold(baseline_rewards, t) for t in thresholds]
        ax4.plot(thresholds, baseline_episodes_to_thresh, 'o-', color='red', linewidth=2, 
                markersize=8, label='Baseline')
    
    for i, opt_data in enumerate(optimized_data[:3]):
        if opt_data['monitor_data'] is not None:
            opt_rewards = opt_data['monitor_data']['r']
            opt_episodes_to_thresh = [episodes_to_threshold(opt_rewards, t) for t in thresholds]
            label = f'Optimized' if i == 0 else f'Optimized {i+1}'
            ax4.plot(thresholds, opt_episodes_to_thresh, 'o-', color=colors[i], 
                    linewidth=2, markersize=8, label=label)
    
    ax4.set_xlabel('Fitness Threshold (Gates Passed)')
    ax4.set_ylabel('Episodes to Reach Threshold')
    ax4.set_title('Learning Speed Comparison')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')  # Log scale since episodes can vary widely
    
    # Plot 5: Training statistics comparison
    ax5 = axes[1, 1]
    
    stats_baseline = calculate_statistics(baseline_data.get('monitor_data'))
    stats_optimized = [calculate_statistics(opt_data.get('monitor_data')) for opt_data in optimized_data]
    
    if stats_baseline and any(stats_optimized):
        metrics = ['final_reward_mean', 'max_reward', 'total_episodes']
        metric_labels = ['Final Reward (Mean)', 'Max Reward', 'Total Episodes']
        
        baseline_values = [stats_baseline.get(m, 0) for m in metrics]
        optimized_values = [np.mean([s.get(m, 0) for s in stats_optimized if s]) for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ax5.bar(x - width/2, baseline_values, width, label='Baseline', color='red', alpha=0.7)
        ax5.bar(x + width/2, optimized_values, width, label='Optimized (Mean)', color='blue', alpha=0.7)
        
        ax5.set_xlabel('Metrics')
        ax5.set_ylabel('Values')
        ax5.set_title('Training Statistics Comparison')
        ax5.set_xticks(x)
        ax5.set_xticklabels(metric_labels, rotation=45, ha='right')
        ax5.legend()
        ax5.grid(True, alpha=0.3, axis='y')
    
    # Plot 6: Learning efficiency (reward vs timesteps)
    ax6 = axes[1, 2]
    
    if baseline_data['monitor_data'] is not None:
        baseline_df = baseline_data['monitor_data']
        ax6.scatter(baseline_df['t'], baseline_df['r'], alpha=0.5, color='red', s=10, label='Baseline')
    
    for i, opt_data in enumerate(optimized_data[:3]):  # Max 3 for readability
        if opt_data['monitor_data'] is not None:
            opt_df = opt_data['monitor_data']
            ax6.scatter(opt_df['t'], opt_df['r'], alpha=0.5, color=colors[i], s=10, 
                       label=f'Optimized {i+1}' if len(optimized_data) > 1 else 'Optimized')
    
    ax6.set_xlabel('Timesteps')
    ax6.set_ylabel('Episode Reward')
    ax6.set_title('Learning Efficiency')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_comparison.pdf'), bbox_inches='tight')
    plt.close()


def generate_comparison_report(baseline_data: Dict, optimized_data: List[Dict], output_dir: str):
    """Generate a text report comparing the results."""
    report_path = os.path.join(output_dir, 'comparison_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("TRAINING COMPARISON REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        # Basic info
        f.write("CONFIGURATION:\n")
        f.write(f"Baseline directory: {baseline_data['directory']}\n")
        f.write(f"Optimized directories: {[d['directory'] for d in optimized_data]}\n\n")
        
        # Final performance
        f.write("FINAL PERFORMANCE:\n")
        f.write("-" * 20 + "\n")
        
        baseline_fitness = baseline_data.get('final_fitness', 'N/A')
        f.write(f"Baseline final fitness: {baseline_fitness}\n")
        
        optimized_fitnesses = [d.get('final_fitness') for d in optimized_data if d.get('final_fitness') is not None]
        if optimized_fitnesses:
            opt_mean = np.mean(optimized_fitnesses)
            opt_std = np.std(optimized_fitnesses) if len(optimized_fitnesses) > 1 else 0
            f.write(f"Optimized final fitness: {opt_mean:.3f} ± {opt_std:.3f}\n")
            
            if isinstance(baseline_fitness, (int, float)) and baseline_fitness > 0:
                improvement = ((opt_mean - baseline_fitness) / baseline_fitness) * 100
                f.write(f"Improvement: {improvement:+.1f}%\n")
        
        f.write("\n")
        
        # Training statistics
        f.write("TRAINING STATISTICS:\n")
        f.write("-" * 20 + "\n")
        
        baseline_stats = calculate_statistics(baseline_data.get('monitor_data'))
        optimized_stats_list = [calculate_statistics(d.get('monitor_data')) for d in optimized_data]
        optimized_stats_list = [s for s in optimized_stats_list if s]  # Remove empty dicts
        
        if baseline_stats and optimized_stats_list:
            for metric in ['final_reward_mean', 'final_reward_std', 'max_reward', 'total_episodes']:
                if metric in baseline_stats:
                    baseline_val = baseline_stats[metric]
                    optimized_vals = [s.get(metric, 0) for s in optimized_stats_list]
                    opt_mean = np.mean(optimized_vals)
                    
                    f.write(f"{metric}: Baseline={baseline_val:.3f}, Optimized={opt_mean:.3f}\n")
        
        f.write("\n")
        
        # Fitness progression analysis
        f.write("FITNESS PROGRESSION ANALYSIS:\n")
        f.write("-" * 30 + "\n")
        
        if baseline_stats and optimized_stats_list:
            # Best fitness comparison
            baseline_best = baseline_stats.get('best_fitness_ever', 0)
            optimized_best_vals = [s.get('best_fitness_ever', 0) for s in optimized_stats_list]
            opt_best_mean = np.mean(optimized_best_vals)
            
            f.write(f"Best fitness ever achieved:\n")
            f.write(f"  Baseline: {baseline_best:.3f} gates\n")
            f.write(f"  Optimized: {opt_best_mean:.3f} gates (avg)\n")
            
            # Learning speed comparison
            baseline_first_gate = baseline_stats.get('episodes_to_first_gate', float('inf'))
            opt_first_gate_vals = [s.get('episodes_to_first_gate', float('inf')) for s in optimized_stats_list]
            opt_first_gate_mean = np.mean([v for v in opt_first_gate_vals if v != float('inf')])
            
            if baseline_first_gate != float('inf'):
                f.write(f"\nEpisodes to first gate passed:\n")
                f.write(f"  Baseline: {baseline_first_gate} episodes\n")
                if opt_first_gate_mean != float('inf'):
                    f.write(f"  Optimized: {opt_first_gate_mean:.1f} episodes (avg)\n")
                    speed_improvement = ((baseline_first_gate - opt_first_gate_mean) / baseline_first_gate) * 100
                    f.write(f"  Speed improvement: {speed_improvement:+.1f}%\n")
            
            # Convergence analysis
            baseline_convergence = baseline_stats.get('convergence_episode', float('inf'))
            opt_convergence_vals = [s.get('convergence_episode', float('inf')) for s in optimized_stats_list]
            opt_convergence_mean = np.mean([v for v in opt_convergence_vals if v != float('inf')])
            
            if baseline_convergence != float('inf') and opt_convergence_mean != float('inf'):
                f.write(f"\nEpisodes to convergence (90% of max performance):\n")
                f.write(f"  Baseline: {baseline_convergence} episodes\n")
                f.write(f"  Optimized: {opt_convergence_mean:.1f} episodes (avg)\n")
                convergence_improvement = ((baseline_convergence - opt_convergence_mean) / baseline_convergence) * 100
                f.write(f"  Convergence improvement: {convergence_improvement:+.1f}%\n")
            
            # Learning stability
            if 'learning_stability' in baseline_stats:
                baseline_stability = baseline_stats['learning_stability']
                opt_stability_vals = [s.get('learning_stability', 1.0) for s in optimized_stats_list]
                opt_stability_mean = np.mean(opt_stability_vals)
                
                f.write(f"\nLearning stability (lower = more stable):\n")
                f.write(f"  Baseline: {baseline_stability:.3f}\n")
                f.write(f"  Optimized: {opt_stability_mean:.3f}\n")
                stability_improvement = ((baseline_stability - opt_stability_mean) / baseline_stability) * 100
                f.write(f"  Stability improvement: {stability_improvement:+.1f}%\n")
        
        f.write("\n")
        
        # Metadata comparison
        f.write("TRAINING CONFIGURATION:\n")
        f.write("-" * 25 + "\n")
        
        baseline_meta = baseline_data.get('metadata', {})
        if baseline_meta:
            f.write("Baseline configuration:\n")
            for key, value in baseline_meta.items():
                f.write(f"  {key}: {value}\n")
        
        f.write("\n")
        
        # Summary
        f.write("SUMMARY:\n")
        f.write("-" * 10 + "\n")
        
        if optimized_fitnesses and isinstance(baseline_fitness, (int, float)):
            if opt_mean > baseline_fitness:
                f.write("✓ Hyperparameter optimization was SUCCESSFUL!\n")
                f.write(f"  - Performance improved by {improvement:+.1f}%\n")
                f.write(f"  - Baseline: {baseline_fitness:.3f} gates\n")
                f.write(f"  - Optimized: {opt_mean:.3f} ± {opt_std:.3f} gates\n")
                
                # Add learning speed insights if available
                if baseline_stats and optimized_stats_list:
                    baseline_first_gate = baseline_stats.get('episodes_to_first_gate', float('inf'))
                    opt_first_gate_vals = [s.get('episodes_to_first_gate', float('inf')) for s in optimized_stats_list]
                    opt_first_gate_mean = np.mean([v for v in opt_first_gate_vals if v != float('inf')])
                    
                    if baseline_first_gate != float('inf') and opt_first_gate_mean != float('inf'):
                        if opt_first_gate_mean < baseline_first_gate:
                            f.write(f"  - Learning speed also improved: {baseline_first_gate} → {opt_first_gate_mean:.1f} episodes to first gate\n")
                        
                    # Convergence insights
                    baseline_convergence = baseline_stats.get('convergence_episode', float('inf'))
                    opt_convergence_vals = [s.get('convergence_episode', float('inf')) for s in optimized_stats_list]
                    opt_convergence_mean = np.mean([v for v in opt_convergence_vals if v != float('inf')])
                    
                    if baseline_convergence != float('inf') and opt_convergence_mean != float('inf'):
                        if opt_convergence_mean < baseline_convergence:
                            f.write(f"  - Faster convergence: {baseline_convergence} → {opt_convergence_mean:.1f} episodes\n")
                
            else:
                f.write("⚠ Hyperparameter optimization showed no improvement.\n")
                f.write("  - Consider running more Optuna trials\n")
                f.write("  - Try different hyperparameter ranges\n")
                f.write("  - Increase training timesteps per trial\n")
                
                # Check if learning speed improved even if final performance didn't
                if baseline_stats and optimized_stats_list:
                    baseline_first_gate = baseline_stats.get('episodes_to_first_gate', float('inf'))
                    opt_first_gate_vals = [s.get('episodes_to_first_gate', float('inf')) for s in optimized_stats_list]
                    opt_first_gate_mean = np.mean([v for v in opt_first_gate_vals if v != float('inf')])
                    
                    if baseline_first_gate != float('inf') and opt_first_gate_mean != float('inf'):
                        if opt_first_gate_mean < baseline_first_gate * 0.8:  # 20% improvement
                            f.write("  + However, learning speed did improve significantly!\n")
                            f.write(f"    Episodes to first gate: {baseline_first_gate} → {opt_first_gate_mean:.1f}\n")
                        
        else:
            f.write("⚠ Could not determine improvement due to missing data.\n")
        
        f.write("\n")
        f.write("INTERPRETATION GUIDE:\n")
        f.write("-" * 20 + "\n")
        f.write("✓ Success indicators:\n")
        f.write("  - Higher final fitness scores\n")
        f.write("  - Faster convergence (fewer episodes to reach good performance)\n")
        f.write("  - More stable learning (lower variance in later episodes)\n")
        f.write("  - Quicker first gate passage\n")
        f.write("⚠ Areas for further optimization:\n")
        f.write("  - If final performance didn't improve: try more trials or longer training\n")
        f.write("  - If learning is unstable: focus on stability-related hyperparameters\n")
        f.write("  - If convergence is slow: adjust learning rate and network architecture\n")

    print(f"Comparison report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Compare training results before and after optimization')
    parser.add_argument('--baseline', type=str, nargs='+', required=True,
                       help='Directory(ies) containing baseline training results')
    parser.add_argument('--optimized', type=str, nargs='+', required=True,
                       help='Directory(ies) containing optimized training results')
    parser.add_argument('--output', type=str, default='comparison_results',
                       help='Output directory for comparison results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    print("Loading training data...")
    
    # Load baseline data (now supports multiple)
    baseline_data_list = []
    for baseline_dir in args.baseline:
        baseline_data = load_training_data(baseline_dir)
        baseline_data_list.append(baseline_data)
        print(f"Loaded baseline from: {baseline_dir}")
    
    # Load optimized data
    optimized_data = []
    for opt_dir in args.optimized:
        opt_data = load_training_data(opt_dir)
        optimized_data.append(opt_data)
        print(f"Loaded optimized from: {opt_dir}")
    
    print("\nGenerating comparison plots...")
    plot_training_comparison(baseline_data_list, optimized_data, args.output)
    
    print("Generating comparison report...")
    generate_comparison_report(baseline_data_list, optimized_data, args.output)
    
    print(f"\nComparison complete! Results saved to: {args.output}")
    print("Files generated:")
    print(f"  - {args.output}/training_comparison.pdf")
    print(f"  - {args.output}/comparison_report.txt")


if __name__ == "__main__":
    main()