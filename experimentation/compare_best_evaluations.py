#!/usr/bin/env python3
"""
Script to analyze and summarize drone evaluation results.
Provides comprehensive analysis of all experiments with optional lap time and reward analysis.
"""

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import torch

# Import functions from the first file
from stable_baselines3 import PPO
try:
    from airevolve.evolution_tools.inspection_tools.morphological_descriptors.hovering_info import get_sim
    from airevolve.evolution_tools.evaluators.drone_gate_env import DroneGateEnv
    from airevolve.evolution_tools.evaluators.gate_train import backandforth, circle, slalom, figure8
    SIMULATION_AVAILABLE = True
except ImportError:
    print("Warning: Simulation modules not available. --detailed analysis will be disabled.")
    SIMULATION_AVAILABLE = False

# Gate configurations for lap time calculation
GATE_COUNTS = {
    'circle': 4,
    'slalom': 4,
    'backandforth': 4,  # shuttlerun renamed to backandforth
    'figure8': 8
}

# Design categories for LaTeX table generation
TRADITIONAL_DESIGNS = ['spiderhex']  # Add traditional design names here

# Evolved designs mapped by task specialization
EVOLVED_DESIGNS_BY_TASK = {
    'circle': ['circle_design_rank_01'],
    'slalom': ['slalom_design_rank_02'], 
    'backandforth': ['shuttlerun_design_rank_01'],  # shuttlerun specialized design
    'figure8': ['design_rank_01']  # general design used for figure8
}

# Task name mapping for LaTeX
TASK_NAME_MAPPING = {
    'circle': 'Circle',
    'slalom': 'Slalom', 
    'backandforth': 'Shuttle run',
    'figure8': 'Figure8'
}

def calculate_stats(gate_times_sec, n_gates):
    """Calculate lap times, segment times, and their statistics."""
    segment_durations = np.diff(gate_times_sec)
    segment_indices = np.arange(len(segment_durations)) % n_gates

    avg_times = []
    std_times = []

    for i in range(n_gates):
        segment_times = segment_durations[segment_indices == i]
        avg_times.append(np.mean(segment_times))
        std_times.append(np.std(segment_times))

    n_laps = len(gate_times_sec) // n_gates
    lap_times = []
    for lap in range(n_laps):
        start_idx = lap * n_gates
        end_idx = start_idx + n_gates - 1
        lap_time = gate_times_sec[end_idx] - gate_times_sec[start_idx]
        lap_times.append(lap_time)

    lap_times = np.array(lap_times)

    return {
        "avg_segment_times": avg_times,
        "std_segment_times": std_times,
        "lap_times": lap_times,
        "avg_lap_time": np.mean(lap_times),
        "std_lap_time": np.std(lap_times),
        "total_laps": n_laps,
    }
    
def extract_simulation_data(individual, policy_file, gate_cfg, device):
    """
    Extract simulation data for a given individual and policy file.

    Args:
        individual (np.ndarray): The individual configuration.
        policy_file (str): Path to the policy file.
        gate_cfg (str): Gate configuration (e.g., "circle", "slalom").
        device (str): Device to run the simulation on (e.g., "cuda:0" or "cpu").

    Returns:
        dict: A dictionary containing positions, velocities, angular velocities, gate passes, and actions.
    """
    # Initialize the simulator
    sim = get_sim(individual)
    sim.compute_hover(verbose=False)
    Bf = sim.Bf
    Bm = sim.Bm

    # Define the environment based on the gate configuration
    gate_configs = {
        "backandforth": backandforth,
        "circle": circle,
        "slalom": slalom,
        "figure8": figure8
    }
    if gate_cfg not in gate_configs:
        raise ValueError("Invalid gate configuration")

    gate_config = gate_configs[gate_cfg]

    env = DroneGateEnv(
        num_envs=1,
        Bf=Bf,
        Bm=Bm,
        gates_pos=gate_config.gate_pos,
        gate_yaw=gate_config.gate_yaw,
        start_pos=gate_config.starting_pos,
        x_bounds=gate_config.x_bounds,
        y_bounds=gate_config.y_bounds,
        z_bounds=gate_config.z_bounds,
        initialize_at_random_gates=False,
        gates_ahead=1,
        num_state_history=0,
        num_action_history=0,
        history_step_size=1,
        render_mode=None,
        device=device
    )
    
    # Load the policy
    policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=[dict(pi=[64, 64, 64], vf=[64, 64, 64])], log_std_init=0)
    model = PPO("MlpPolicy", env,
                policy_kwargs=policy_kwargs,
                verbose=0,
                n_steps=1000,
                batch_size=5000,
                n_epochs=10,
                gamma=0.999,
                device=device)
    
    try:
        model = PPO.load(policy_file)
    except:
        model = PPO.load(policy_file[:-4])

    # Run the simulation
    env.reset()
    positions, velocities, angular_velocities, gate_passes, actions = [], [], [], [], []

    for _ in range(1200):
        action, _ = model.predict(env.states, deterministic=True)
        states, rewards, dones, infos = env.step(action)

        positions.append(env.world_states[0, 0:3])
        velocities.append(env.world_states[0, 3:6])
        angular_velocities.append(env.world_states[0, 9:12])
        gate_passes.append(infos[0]["gate_passed"])
        # Normalize actions to be between 0 and 1
        a = action[0] * 0.5 + 0.5
        actions.append(a)

    # Convert lists to numpy arrays
    positions = np.array(positions[:-1])
    velocities = np.array(velocities[:-1])
    angular_velocities = np.array(angular_velocities[:-1])
    gate_passes = np.array(gate_passes[:-1])
    actions = np.array(actions[:-1])

    return {
        "positions": positions,
        "velocities": velocities,
        "angular_velocities": angular_velocities,
        "gate_passes": gate_passes,
        "actions": actions
    }

def read_monitor_file(monitor_file):
    """Read training monitor file to extract episode rewards and timesteps."""
    try:
        data = pd.read_csv(monitor_file, skiprows=1)  # Skip the first row (comments)
        episode_rewards = data["r"]  # Rewards per episode
        time_steps = data["t"]  # Timesteps at each episode

        # remove the last episode if it is not finished
        if isinstance(episode_rewards.iloc[-1], str):
            episode_rewards = episode_rewards[:-1]
            time_steps = time_steps[:-1]

        episode_rewards = np.array(episode_rewards, dtype=float)
        time_steps = np.array(time_steps, dtype=int)

        return episode_rewards, time_steps
    except Exception as e:
        print(f"Error reading monitor file {monitor_file}: {e}")
        return None, None

def asymptotic_performance(episode_rewards):
    """Calculate the maximum reward achieved during training."""
    if episode_rewards is not None and len(episode_rewards) > 0:
        return np.max(episode_rewards)
    return None

def analyze_detailed_performance(exp_dir, design_name, task_name, device="cpu"):
    """
    Analyze detailed performance including lap times and max rewards.
    
    Args:
        exp_dir (Path): Directory containing experiment results
        design_name (str): Name of the design
        task_name (str): Name of the task
        device (str): Device for simulation
    
    Returns:
        dict: Dictionary containing detailed analysis results
    """
    detailed_results = {
        'avg_lap_time': None,
        'std_lap_time': None,
        'total_laps': None,
        'max_reward': None
    }
    
    if not SIMULATION_AVAILABLE:
        print("Simulation modules not available, skipping detailed analysis.")
        return detailed_results
    
    try:
        # Look for policy file
        policy_files = list(exp_dir.glob("*.zip")) + list(exp_dir.glob("*.pkl"))
        if not policy_files:
            print(f"No policy file found in {exp_dir}, skipping detailed analysis.")
            return detailed_results
        
        policy_file = str(policy_files[0])
        
        # Look for individual/design file
        individual_files = list(exp_dir.glob("individual.npy"))
        if not individual_files:
            print(f"No individual/design file found in {exp_dir}, skipping detailed analysis.")
            return detailed_results
            
        # Load individual
        individual = np.load(individual_files[0])
        
        # Extract simulation data
        sim_data = extract_simulation_data(individual, policy_file, task_name, device)
        
        # Calculate lap times if we have gate passes
        gate_passes = sim_data['gate_passes']
        n_gates = GATE_COUNTS.get(task_name, 4)
        
        # Find gate pass times
        gate_times = []
        for i, passed in enumerate(gate_passes):
            if passed:
                gate_times.append(i * 0.01)  # Assuming 100Hz simulation
        
        if len(gate_times) >= n_gates:
            gate_times_sec = np.array(gate_times)
            lap_stats = calculate_stats(gate_times_sec, n_gates)
            detailed_results.update({
                'avg_lap_time': lap_stats['avg_lap_time'],
                'std_lap_time': lap_stats['std_lap_time'],
                'total_laps': lap_stats['total_laps']
            })
        
        # Look for monitor file to get max reward
        monitor_files = list(exp_dir.glob("*monitor.csv")) + list(exp_dir.glob("monitor.csv"))
        if monitor_files:
            episode_rewards, _ = read_monitor_file(monitor_files[0])
            if episode_rewards is not None:
                detailed_results['max_reward'] = asymptotic_performance(episode_rewards)
        
    except Exception as e:
        print(f"Error in detailed analysis for {design_name}/{task_name}: {e}")
    
    return detailed_results

def load_results(base_dir="drone_evaluations", detailed=False, device="cpu"):
    """Load all results from the evaluation directory"""
    results = []
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"Results directory {base_dir} does not exist!")
        return pd.DataFrame()
    
    # Walk through directory structure: design/task/rep_X
    for design_dir in base_path.iterdir():
        if not design_dir.is_dir():
            continue
            
        design_name = design_dir.name
        
        for task_dir in design_dir.iterdir():
            if not task_dir.is_dir():
                continue
                
            task_name = task_dir.name
            
            for rep_dir in task_dir.iterdir():
                if not rep_dir.is_dir() or not rep_dir.name.startswith('rep_'):
                    continue
                
                rep_num = int(rep_dir.name.split('_')[1])
                
                # Load results
                results_file = rep_dir / "results.pkl"
                metadata_file = rep_dir / "metadata.pkl"
                fitness_file = rep_dir / "fitness.txt"
                
                result_entry = {
                    'design': design_name,
                    'task': task_name,
                    'repetition': rep_num,
                    'completed': False,
                    'fitness': None,
                    'duration': None,
                    'error': None,
                    'avg_lap_time': None,
                    'std_lap_time': None,
                    'total_laps': None,
                    'max_reward': None
                }
                
                # Try to load results
                if results_file.exists():
                    try:
                        with open(results_file, 'rb') as f:
                            result_data = pickle.load(f)
                        
                        result_entry.update({
                            'completed': result_data.get('success', False),
                            'fitness': result_data.get('fitness'),
                            'duration': result_data.get('duration'),
                            'error': result_data.get('error')
                        })
                    except Exception as e:
                        result_entry['error'] = f"Error loading results: {e}"
                
                # Try to get fitness from text file if pickle failed
                elif fitness_file.exists():
                    try:
                        with open(fitness_file, 'r') as f:
                            fitness_value = float(f.read().strip())
                        result_entry['fitness'] = fitness_value
                        result_entry['completed'] = True
                    except Exception as e:
                        result_entry['error'] = f"Error loading fitness: {e}"
                
                else:
                    print(f"No results found for {design_name}/{task_name}/rep_{rep_num}")
                    result_entry['error'] = "No results found"

                # Perform detailed analysis if requested and experiment completed
                if detailed and result_entry['completed']:
                    print(f"Analyzing detailed performance for {design_name}/{task_name}/rep_{rep_num}...")
                    detailed_results = analyze_detailed_performance(rep_dir, design_name, task_name, device)
                    print(f"  Detailed results: {detailed_results}")
                    result_entry.update(detailed_results)
                
                results.append(result_entry)
    
    return pd.DataFrame(results)

def print_summary(df, detailed=False):
    """Print comprehensive summary of results"""
    if df.empty:
        print("No results found!")
        return
    
    total_experiments = len(df)
    completed = df['completed'].sum()
    failed = total_experiments - completed
    
    print("="*80)
    print("DRONE EVALUATION RESULTS SUMMARY")
    print("="*80)
    
    print(f"\nOverall Progress:")
    print(f"  Total experiments: {total_experiments}")
    print(f"  Completed: {completed} ({completed/total_experiments*100:.1f}%)")
    print(f"  Failed/Pending: {failed} ({failed/total_experiments*100:.1f}%)")
    
    if completed > 0:
        avg_duration = df[df['completed']]['duration'].mean()
        print(f"  Average duration: {avg_duration:.1f} seconds")
    
    print(f"\nProgress by Design:")
    design_progress = df.groupby('design').agg({
        'completed': ['count', 'sum'],
        'fitness': 'count'
    }).round(2)
    design_progress.columns = ['total', 'completed', 'fitness_available']
    design_progress['completion_rate'] = (design_progress['completed'] / design_progress['total'] * 100).round(1)
    print(design_progress)
    
    print(f"\nProgress by Task:")
    task_progress = df.groupby('task').agg({
        'completed': ['count', 'sum'],
        'fitness': 'count'
    }).round(2)
    task_progress.columns = ['total', 'completed', 'fitness_available']  
    task_progress['completion_rate'] = (task_progress['completed'] / task_progress['total'] * 100).round(1)
    print(task_progress)
    
    # Show fitness statistics for completed experiments
    completed_df = df[df['completed'] & df['fitness'].notna()]
    if len(completed_df) > 0:
        print(f"\nFitness Statistics (n={len(completed_df)}):")
        fitness_stats = completed_df.groupby(['design', 'task'])['fitness'].agg(['count', 'mean', 'std', 'min', 'max']).round(2)
        print(fitness_stats)
        
        print(f"\nOverall Fitness by Design:")
        design_fitness = completed_df.groupby('design')['fitness'].agg(['count', 'mean', 'std', 'min', 'max']).round(2)
        print(design_fitness)
        
        print(f"\nOverall Fitness by Task:")
        task_fitness = completed_df.groupby('task')['fitness'].agg(['count', 'mean', 'std', 'min', 'max']).round(2)
        print(task_fitness)
    
    # Show detailed analysis if available
    if detailed:
        detailed_df = df[df['completed'] & df['avg_lap_time'].notna()]
        if len(detailed_df) > 0:
            print(f"\nLap Time Statistics (n={len(detailed_df)}):")
            lap_stats = detailed_df.groupby(['design', 'task'])['avg_lap_time'].agg(['count', 'mean', 'std', 'min', 'max']).round(2)
            print(lap_stats)
            
            print(f"\nOverall Lap Times by Design:")
            design_lap_times = detailed_df.groupby('design')['avg_lap_time'].agg(['count', 'mean', 'std', 'min', 'max']).round(2)
            print(design_lap_times)
        
        reward_df = df[df['completed'] & df['max_reward'].notna()]
        if len(reward_df) > 0:
            print(f"\nMaximum Reward Statistics (n={len(reward_df)}):")
            reward_stats = reward_df.groupby(['design', 'task'])['max_reward'].agg(['count', 'mean', 'std', 'min', 'max']).round(2)
            print(reward_stats)
            
            print(f"\nOverall Max Rewards by Design:")
            design_rewards = reward_df.groupby('design')['max_reward'].agg(['count', 'mean', 'std', 'min', 'max']).round(2)
            print(design_rewards)
    
    # Show failed experiments
    failed_df = df[~df['completed']]
    if len(failed_df) > 0:
        print(f"\nFailed/Pending Experiments ({len(failed_df)}):")
        failed_summary = failed_df.groupby(['design', 'task']).size().reset_index(name='count')
        print(failed_summary)
        
        if failed_df['error'].notna().any():
            print(f"\nError Messages:")
            error_counts = failed_df['error'].value_counts()
            for error, count in error_counts.items():
                if error and error != 'nan':
                    print(f"  {error}: {count} occurrences")

def create_visualizations(df, output_dir="plots", detailed=False):
    """Create visualization plots of the results"""
    if df.empty:
        print("No results to visualize")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    completed_df = df[df['completed'] & df['fitness'].notna()]
    
    if len(completed_df) == 0:
        print("No completed experiments to visualize")
        return
    
    # 1. Progress heatmap
    plt.figure(figsize=(12, 8))
    progress_matrix = df.pivot_table(values='completed', index='design', columns='task', aggfunc='sum', fill_value=0)
    sns.heatmap(progress_matrix, annot=True, fmt='d', cmap='RdYlGn', 
                cbar_kws={'label': 'Completed Experiments'})
    plt.title('Completion Progress by Design and Task')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/progress_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Fitness by design (box plot)
    if len(completed_df) > 0:
        plt.figure(figsize=(14, 8))
        sns.boxplot(data=completed_df, x='design', y='fitness')
        plt.xticks(rotation=45, ha='right')
        plt.title('Fitness Distribution by Design')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/fitness_by_design.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Fitness by task (box plot)
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=completed_df, x='task', y='fitness')
        plt.title('Fitness Distribution by Task')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/fitness_by_task.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Fitness heatmap (mean values)
        plt.figure(figsize=(12, 8))
        fitness_matrix = completed_df.pivot_table(values='fitness', index='design', columns='task', aggfunc='mean')
        sns.heatmap(fitness_matrix, annot=True, fmt='.2f', cmap='viridis', 
                    cbar_kws={'label': 'Mean Fitness'})
        plt.title('Mean Fitness by Design and Task')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/fitness_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Duration analysis if available
        if 'duration' in completed_df.columns and completed_df['duration'].notna().any():
            plt.figure(figsize=(12, 8))
            sns.boxplot(data=completed_df, x='design', y='duration')
            plt.xticks(rotation=45, ha='right')
            plt.ylabel('Duration (seconds)')
            plt.title('Evaluation Duration by Design')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/duration_by_design.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    # Additional plots for detailed analysis
    if detailed:
        # Lap time visualizations
        lap_df = df[df['completed'] & df['avg_lap_time'].notna()]
        if len(lap_df) > 0:
            plt.figure(figsize=(14, 8))
            sns.boxplot(data=lap_df, x='design', y='avg_lap_time')
            plt.xticks(rotation=45, ha='right')
            plt.ylabel('Average Lap Time (seconds)')
            plt.title('Lap Time Distribution by Design')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/lap_time_by_design.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            plt.figure(figsize=(12, 8))
            sns.boxplot(data=lap_df, x='task', y='avg_lap_time')
            plt.ylabel('Average Lap Time (seconds)')
            plt.title('Lap Time Distribution by Task')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/lap_time_by_task.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # Max reward visualizations
        reward_df = df[df['completed'] & df['max_reward'].notna()]
        if len(reward_df) > 0:
            plt.figure(figsize=(14, 8))
            sns.boxplot(data=reward_df, x='design', y='max_reward')
            plt.xticks(rotation=45, ha='right')
            plt.ylabel('Maximum Reward')
            plt.title('Maximum Reward Distribution by Design')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/max_reward_by_design.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            plt.figure(figsize=(12, 8))
            sns.boxplot(data=reward_df, x='task', y='max_reward')
            plt.ylabel('Maximum Reward')
            plt.title('Maximum Reward Distribution by Task')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/max_reward_by_task.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    print(f"Visualizations saved to {output_dir}/")

def export_raw_data(df, output_file="raw_data.csv"):
    """Export all raw experimental data to CSV"""
    if df.empty:
        print("No data to export")
        return
    
    # Create a clean version of the dataframe for export
    export_df = df.copy()
    
    # Reorder columns for better readability
    column_order = [
        'design', 'task', 'repetition', 'completed', 'fitness', 
        'max_reward', 'avg_lap_time', 'std_lap_time', 'total_laps',
        'duration', 'error'
    ]
    
    # Only include columns that exist in the dataframe
    available_columns = [col for col in column_order if col in export_df.columns]
    remaining_columns = [col for col in export_df.columns if col not in available_columns]
    final_columns = available_columns + remaining_columns
    
    export_df = export_df[final_columns]
    
    # Sort by design, task, then repetition for organized output
    export_df = export_df.sort_values(['design', 'task', 'repetition'])
    
    export_df.to_csv(output_file, index=False)
    print(f"Raw data exported to {output_file}")
    print(f"  Total experiments: {len(export_df)}")
    print(f"  Completed experiments: {export_df['completed'].sum()}")
    
    # Print some basic stats about the export
    if 'fitness' in export_df.columns:
        completed_with_fitness = export_df[export_df['completed'] & export_df['fitness'].notna()]
        if len(completed_with_fitness) > 0:
            print(f"  Experiments with fitness data: {len(completed_with_fitness)}")
    
    if 'avg_lap_time' in export_df.columns:
        with_lap_times = export_df[export_df['avg_lap_time'].notna()]
        if len(with_lap_times) > 0:
            print(f"  Experiments with lap time data: {len(with_lap_times)}")
    
    if 'max_reward' in export_df.columns:
        with_rewards = export_df[export_df['max_reward'].notna()]
        if len(with_rewards) > 0:
            print(f"  Experiments with reward data: {len(with_rewards)}")

def export_summary_csv(df, output_file="evaluation_summary.csv"):
    """Export summary statistics to CSV"""
    if df.empty:
        return
    
    # Create comprehensive summary
    summary_data = []
    
    for design in df['design'].unique():
        for task in df['task'].unique():
            subset = df[(df['design'] == design) & (df['task'] == task)]
            completed_subset = subset[subset['completed'] & subset['fitness'].notna()]
            
            summary_row = {
                'design': design,
                'task': task,
                'total_reps': len(subset),
                'completed_reps': len(completed_subset),
                'completion_rate': len(completed_subset) / len(subset) * 100 if len(subset) > 0 else 0,
                'mean_fitness': completed_subset['fitness'].mean() if len(completed_subset) > 0 else None,
                'std_fitness': completed_subset['fitness'].std() if len(completed_subset) > 0 else None,
                'min_fitness': completed_subset['fitness'].min() if len(completed_subset) > 0 else None,
                'max_fitness': completed_subset['fitness'].max() if len(completed_subset) > 0 else None,
                'mean_duration': completed_subset['duration'].mean() if len(completed_subset) > 0 else None
            }
            
            # Add detailed metrics if available
            lap_subset = completed_subset[completed_subset['avg_lap_time'].notna()]
            if len(lap_subset) > 0:
                summary_row.update({
                    'mean_avg_lap_time': lap_subset['avg_lap_time'].mean(),
                    'std_avg_lap_time': lap_subset['avg_lap_time'].std(),
                    'min_avg_lap_time': lap_subset['avg_lap_time'].min(),
                    'max_avg_lap_time': lap_subset['avg_lap_time'].max()
                })
            
            reward_subset = completed_subset[completed_subset['max_reward'].notna()]
            if len(reward_subset) > 0:
                summary_row.update({
                    'mean_max_reward': reward_subset['max_reward'].mean(),
                    'std_max_reward': reward_subset['max_reward'].std(),
                    'min_max_reward': reward_subset['max_reward'].min(),
                    'max_max_reward': reward_subset['max_reward'].max()
                })
            
            summary_data.append(summary_row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_file, index=False)
    print(f"Summary exported to {output_file}")

def find_missing_experiments(df):
    """Find which experiments are missing or failed"""
    designs = ['slalom_design_rank_02', 'shuttlerun_design_rank_01', 'circle_design_rank_01', 
               'design_rank_01', 'spiderhex']
    tasks = ['backandforth', 'figure8', 'circle', 'slalom']
    reps = list(range(10))
    
    expected = set()
    for design in designs:
        for task in tasks:
            for rep in reps:
                expected.add((design, task, rep))
    
    actual = set()
    for _, row in df.iterrows():
        actual.add((row['design'], row['task'], row['repetition']))
    
    missing = expected - actual
    failed = set()
    
    for _, row in df.iterrows():
        if not row['completed']:
            failed.add((row['design'], row['task'], row['repetition']))
    
    return missing, failed

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze drone evaluation results')
    parser.add_argument('--base_dir', type=str, default='drone_evaluations', 
                        help='Base directory containing results')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    parser.add_argument('--export', action='store_true', help='Export summary CSV')
    parser.add_argument('--export_raw', action='store_true', help='Export raw data CSV with all individual experiments')
    parser.add_argument('--missing', action='store_true', help='Show missing experiments')
    parser.add_argument('--detailed', action='store_true', 
                        help='Perform detailed analysis including lap times and max rewards (slower)')
    parser.add_argument('--device', type=str, default='cuda:0', 
                        help='Device for simulation (cpu, cuda:0, etc.)')
    parser.add_argument('--latex', action='store_true', 
                        help='Generate LaTeX table comparing traditional vs evolved designs')
    parser.add_argument('--traditional_designs', type=str, nargs='+', default=None,
                        help='List of traditional design names for LaTeX table')
    parser.add_argument('--evolved_designs', type=str, nargs='+', default=None,
                        help='List of evolved design names for LaTeX table')
    
    args = parser.parse_args()
    
    if args.detailed and not SIMULATION_AVAILABLE:
        print("Warning: --detailed analysis requested but simulation modules not available!")
        print("Install required packages or run without --detailed flag.")
        return
    
    if args.latex and args.detailed and not SIMULATION_AVAILABLE:
        print("Warning: --latex with complete metrics requires --detailed analysis, but simulation modules not available!")
        print("LaTeX table will be generated with fitness data only.")
    
    print("Loading results...")
    df = load_results(args.base_dir, detailed=args.detailed, device=args.device)
    
    if df.empty:
        print(f"No results found in {args.base_dir}")
        return
    
    print_summary(df, detailed=args.detailed)
    
    if args.missing:
        print("\n" + "="*60)
        print("MISSING/FAILED EXPERIMENTS")
        print("="*60)
        
        missing, failed = find_missing_experiments(df)
        
        if missing:
            print(f"\nMissing experiments ({len(missing)}):")
            missing_by_design = defaultdict(list)
            for design, task, rep in sorted(missing):
                missing_by_design[design].append(f"{task}/rep_{rep}")
            
            for design, experiments in missing_by_design.items():
                print(f"  {design}: {len(experiments)} missing")
                for exp in experiments[:5]:  # Show first 5
                    print(f"    - {exp}")
                if len(experiments) > 5:
                    print(f"    ... and {len(experiments) - 5} more")
        
        if failed:
            print(f"\nFailed experiments ({len(failed)}):")
            failed_by_design = defaultdict(list)
            for design, task, rep in sorted(failed):
                failed_by_design[design].append(f"{task}/rep_{rep}")
            
            for design, experiments in failed_by_design.items():
                print(f"  {design}: {len(experiments)} failed")
                for exp in experiments[:5]:  # Show first 5
                    print(f"    - {exp}")
                if len(experiments) > 5:
                    print(f"    ... and {len(experiments) - 5} more")
    
    if args.plot:
        print("\nGenerating visualizations...")
        create_visualizations(df, detailed=args.detailed)
    
    if args.export:
        print("\nExporting summary...")
        export_summary_csv(df)
    
    if args.export_raw:
        print("\nExporting raw data...")
        export_raw_data(df)
        
    # Keep the existing export behavior for backward compatibility
    if args.export and not args.export_raw:
        # Also export raw results with the old name
        df.to_csv("raw_results.csv", index=False)
        print("Raw results also exported to raw_results.csv")
    
    if args.latex:
        print("\nGenerating LaTeX table...")
        generate_latex_table(df, 
                            traditional_designs=args.traditional_designs,
                            evolved_designs_by_task=args.evolved_designs)
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)
    
    # Give some recommendations
    completed_df = df[df['completed'] & df['fitness'].notna()]
    if len(completed_df) > 0:
        best_overall = completed_df.loc[completed_df['fitness'].idxmin()]
        print(f"\nBest performing experiment:")
        print(f"  Design: {best_overall['design']}")
        print(f"  Task: {best_overall['task']}")  
        print(f"  Fitness: {best_overall['fitness']:.2f}")
        
        if args.detailed:
            if pd.notna(best_overall['avg_lap_time']):
                print(f"  Avg Lap Time: {best_overall['avg_lap_time']:.2f}s")
            if pd.notna(best_overall['max_reward']):
                print(f"  Max Reward: {best_overall['max_reward']:.2f}")
        
        print(f"\nBest design by average fitness:")
        design_avg = completed_df.groupby('design')['fitness'].mean().sort_values()
        print(f"  {design_avg.index[0]}: {design_avg.iloc[0]:.2f}")
        
        print(f"\nEasiest task by average fitness:")
        task_avg = completed_df.groupby('task')['fitness'].mean().sort_values()
        print(f"  {task_avg.index[0]}: {task_avg.iloc[0]:.2f}")
        
        if args.detailed:
            # Additional detailed recommendations
            lap_df = df[df['completed'] & df['avg_lap_time'].notna()]
            if len(lap_df) > 0:
                fastest_lap = lap_df.loc[lap_df['avg_lap_time'].idxmin()]
                print(f"\nFastest average lap time:")
                print(f"  Design: {fastest_lap['design']}")
                print(f"  Task: {fastest_lap['task']}")
                print(f"  Lap Time: {fastest_lap['avg_lap_time']:.2f}s")
            
            reward_df = df[df['completed'] & df['max_reward'].notna()]
            if len(reward_df) > 0:
                best_reward = reward_df.loc[reward_df['max_reward'].idxmax()]
                print(f"\nHighest maximum reward:")
                print(f"  Design: {best_reward['design']}")
                print(f"  Task: {best_reward['task']}")
                print(f"  Max Reward: {best_reward['max_reward']:.2f}")

if __name__ == "__main__":
    main()