"""
Analysis and visualization script for morphological descriptors data.
Plots metrics over generations and against fitness, with statistical analysis.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import warnings
import re
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def convert_str_to_nparray(s):
    """Convert string representation of numpy array back to numpy array."""
    if pd.isna(s) or s == 'nan':
        return np.array([])
    
    s = str(s)
    s = s.replace("nan", "np.nan")
    s = s.replace("0. ", "0.0")
    s = re.sub(r"(-?\d|np\.nan)\s+(?=-?\d|np\.nan)", r"\1, ", s)
    s = re.sub(r"(\d\.)\s+(?=\d)", r"\1, ", s)
    s = re.sub(r"\]\s+\[", "], [", s)
    s = s.replace(". ", " ")
    s = s.replace(".,", ",")
    
    try:
        arr = np.array(eval(s))
        return arr
    except:
        return np.array([])

def load_experiment_data(experiment_dirs):
    """Load morphological descriptors data from all experiment directories."""
    all_data = {}
    
    for exp_name, exp_dir in experiment_dirs.items():
        data_path = os.path.join(exp_dir, "morphological_descriptors_data.csv")
        
        if os.path.exists(data_path):
            print(f"Loading data from {exp_name}...")
            df = pd.read_csv(data_path)
            
            # Try to load individual morphology data from evolution_data.csv files
            print(f"  Loading individual morphology data...")
            df = load_individual_morphologies(df, exp_dir)
            
            all_data[exp_name] = df
            print(f"  Loaded {len(df)} records from {df['run'].nunique()} runs")
            print(f"  Generations: {df['generation'].min()} to {df['generation'].max()}")
        else:
            print(f"Warning: No data file found for {exp_name} at {data_path}")
    
    return all_data

def load_individual_morphologies(md_df, exp_dir):
    """Load individual morphology data and merge with morphological descriptors."""
    # Get all run directories
    run_dirs = [d for d in os.listdir(exp_dir) if d != ".DS_Store" and os.path.isdir(os.path.join(exp_dir, d))]
    
    # Dictionary to store individual morphologies
    morphology_data = {}
    
    for run_idx, run_dir in enumerate(run_dirs):
        run_path = os.path.join(exp_dir, run_dir)
        evolution_data_path = os.path.join(run_path, "evolution_data.csv")
        
        if os.path.exists(evolution_data_path):
            try:
                evo_data = pd.read_csv(evolution_data_path)
                
                # Process each row to extract individual morphologies
                for idx, row in evo_data.iterrows():
                    generation = row['generation']
                    offspring_str = row['offspring']
                    
                    # Convert offspring string to array
                    try:
                        offspring_array = convert_str_to_nparray(offspring_str)
                        key = (run_idx, generation, idx % 24)  # Assuming max 24 individuals per generation
                        morphology_data[key] = offspring_array
                    except:
                        continue
                        
            except Exception as e:
                print(f"    Warning: Could not load evolution data from {run_path}: {e}")
                continue
    
    # Add morphology data to the morphological descriptors dataframe
    offspring_list = []
    for idx, row in md_df.iterrows():
        run_id = row['run']
        generation = row['generation'] 
        individual_id = row['individual']
        
        key = (run_id, generation, individual_id)
        if key in morphology_data:
            offspring_list.append(str(morphology_data[key].tolist()))
        else:
            offspring_list.append(np.nan)
    
    md_df['offspring'] = offspring_list
    return md_df

def pad_generations(df, max_generations=40):
    """Pad generation data to ensure all runs have the same number of generations."""
    padded_data = []
    
    for run_id in df['run'].unique():
        run_data = df[df['run'] == run_id].copy()
        
        # Get the last generation for this run
        max_gen = run_data['generation'].max()
        
        if max_gen < max_generations - 1:  # 0-indexed generations
            # Create padding data by repeating the last generation's data
            last_gen_data = run_data[run_data['generation'] == max_gen].copy()
            
            for gen in range(max_gen + 1, max_generations):
                padding_data = last_gen_data.copy()
                padding_data['generation'] = gen
                padded_data.append(padding_data)
        
        padded_data.append(run_data)
    
    return pd.concat(padded_data, ignore_index=True)

def plot_metrics_over_generations(all_data, save_dir=None):
    """Plot how metrics change over generations for all experiments."""
    metrics = ['Central_Asymmetry', 'Bilateral_Asymmetry']
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        for exp_name, df in all_data.items():
            # Pad generations
            padded_df = pad_generations(df)
            
            # Calculate mean and std for each generation
            gen_stats = padded_df.groupby('generation')[metric].agg(['mean', 'std', 'count']).reset_index()
            
            # Plot mean with error bars
            ax.errorbar(gen_stats['generation'], gen_stats['mean'], 
                       yerr=gen_stats['std'], label=exp_name, alpha=0.8, capsize=3)
        
        ax.set_xlabel('Generation')
        ax.set_ylabel(metric.replace('_', ' '))
        ax.set_title(f'{metric.replace("_", " ")} Over Generations')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'metrics_over_generations.png'), dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_metrics_vs_fitness(all_data, save_dir=None):
    """Plot metrics against fitness for each experiment separately with fitness-threshold filtered examples."""
    metrics = ['Central_Asymmetry', 'Bilateral_Asymmetry']
    
    for exp_name, df in all_data.items():
        print(f"\nCreating plots for {exp_name}...")
        
        # Calculate fitness threshold for this experiment (e.g., 75th percentile)
        valid_fitness = df['fitness'].dropna()
        if len(valid_fitness) > 0:
            fitness_threshold = np.percentile(valid_fitness, 75)
            print(f"  Fitness threshold (75th percentile): {fitness_threshold:.3f}")
        else:
            fitness_threshold = 0
            print("  Warning: No valid fitness values found")
        
        for metric in metrics:
            # Create figure with subplots for main plot and morphology examples
            fig = plt.figure(figsize=(16, 6))
            
            # Main density plot
            ax_main = plt.subplot2grid((2, 4), (0, 0), colspan=2, rowspan=2)
            
            # Morphology subplots
            ax_high = plt.subplot2grid((2, 4), (0, 2), projection='3d')
            ax_low = plt.subplot2grid((2, 4), (1, 2), projection='3d')
            
            # Remove NaN values for this specific experiment and metric
            valid_data = df.dropna(subset=[metric, 'fitness'])
            
            if len(valid_data) > 0:
                # Create density plot background
                if len(valid_data) > 50:
                    try:
                        hist, x_edges, y_edges = np.histogram2d(
                            valid_data['fitness'], valid_data[metric], bins=30, density=True
                        )
                        extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
                        ax_main.imshow(hist.T, origin='lower', extent=extent, 
                                     aspect='auto', alpha=0.3, cmap='Blues')
                    except:
                        pass
                
                # Create scatter plot with transparency
                ax_main.scatter(valid_data['fitness'], valid_data[metric], 
                              alpha=0.4, color='steelblue', s=15)
                
                # Add trend line
                if len(valid_data) > 10:
                    z = np.polyfit(valid_data['fitness'], valid_data[metric], 1)
                    p = np.poly1d(z)
                    x_trend = np.linspace(valid_data['fitness'].min(), valid_data['fitness'].max(), 100)
                    ax_main.plot(x_trend, p(x_trend), '--', color='red', alpha=0.8, linewidth=2)
                
                # Add fitness threshold line
                ax_main.axvline(x=fitness_threshold, color='orange', linestyle=':', 
                              linewidth=2, alpha=0.7, label=f'Fitness threshold ({fitness_threshold:.2f})')
                
                # Find high-fitness individuals for morphology examples
                high_fitness_data = valid_data[valid_data['fitness'] >= fitness_threshold]
                
                if len(high_fitness_data) >= 2:
                    # Get individuals with morphology data
                    high_fitness_individuals = []
                    for idx, row in high_fitness_data.iterrows():
                        if 'offspring' in df.columns and pd.notna(df.loc[idx, 'offspring']):
                            try:
                                individual = convert_str_to_nparray(df.loc[idx, 'offspring'])
                                if len(individual) > 0:
                                    high_fitness_individuals.append({
                                        'individual': individual,
                                        'metric_value': row[metric],
                                        'fitness': row['fitness'],
                                        'generation': row['generation']
                                    })
                            except:
                                continue
                    
                    if len(high_fitness_individuals) >= 2:
                        # Sort by metric value and get extremes
                        sorted_individuals = sorted(high_fitness_individuals, key=lambda x: x['metric_value'])
                        
                        low_example = sorted_individuals[0]
                        high_example = sorted_individuals[-1]
                        
                        # Plot low metric value example (high fitness)
                        try:
                            represent_morphology_3d(ax_low, low_example['individual'], 
                                                  fitness=low_example['fitness'],
                                                  generation=low_example['generation'],
                                                  elev=0, azim=90, axis_labels=False, 
                                                  show_axis_ticks=False, fontsize=8)
                            ax_low.set_title(f'Low {metric.replace("_", " ")}\nMetric: {low_example["metric_value"]:.3f}\nFitness: {low_example["fitness"]:.3f}', 
                                           fontsize=9)
                        except Exception as e:
                            ax_low.text(0.5, 0.5, 0.5, 'Visualization\nError', 
                                       ha='center', va='center')
                            print(f"    Error visualizing low example: {e}")
                        
                        # Plot high metric value example (high fitness)  
                        try:
                            represent_morphology_3d(ax_high, high_example['individual'], 
                                                  fitness=high_example['fitness'],
                                                  generation=high_example['generation'],
                                                  elev=0, azim=90, axis_labels=False, 
                                                  show_axis_ticks=False, fontsize=8)
                            ax_high.set_title(f'High {metric.replace("_", " ")}\nMetric: {high_example["metric_value"]:.3f} Fitness: {high_example["fitness"]:.3f}', 
                                            fontsize=9)
                        except Exception as e:
                            ax_high.text(0.5, 0.5, 0.5, 'Visualization\nError', 
                                        ha='center', va='center')
                            print(f"    Error visualizing high example: {e}")
                        
                        # Highlight the selected examples on the main plot
                        ax_main.scatter([low_example['fitness']], [low_example['metric_value']], 
                                      color='red', s=100, marker='o', edgecolors='black', linewidth=2, 
                                      label='Low metric example', zorder=5)
                        ax_main.scatter([high_example['fitness']], [high_example['metric_value']], 
                                      color='green', s=100, marker='s', edgecolors='black', linewidth=2, 
                                      label='High metric example', zorder=5)
                    
                    else:
                        ax_low.text(0.5, 0.5, 0.5, f'No high-fitness\nindividuals with\nmorphology data\n(n={len(high_fitness_individuals)})', 
                                   ha='center', va='center', transform=ax_low.transAxes)
                        ax_high.text(0.5, 0.5, 0.5, f'No high-fitness\nindividuals with\nmorphology data\n(n={len(high_fitness_individuals)})', 
                                    ha='center', va='center', transform=ax_high.transAxes)
                else:
                    ax_low.text(0.5, 0.5, 0.5, f'Insufficient high-fitness\ndata (n={len(high_fitness_data)})\nThreshold: {fitness_threshold:.2f}', 
                               ha='center', va='center', transform=ax_low.transAxes)
                    ax_high.text(0.5, 0.5, 0.5, f'Insufficient high-fitness\ndata (n={len(high_fitness_data)})\nThreshold: {fitness_threshold:.2f}', 
                                ha='center', va='center', transform=ax_high.transAxes)
            
            # Configure main plot
            ax_main.set_xlabel('Fitness', fontsize=12)
            ax_main.set_ylabel(metric.replace('_', ' '), fontsize=12)
            # ax_main.set_title(f'{exp_name}: {metric.replace("_", " ")} vs Fitness', fontsize=14)
            ax_main.legend(fontsize=9)
            ax_main.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_dir:
                filename = f'{exp_name}_{metric.lower()}_vs_fitness_with_morphology.png'
                plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
            
            plt.show()

def represent_morphology_3d(ax, individual, elev=30, azim=30, fitness=None, generation=None,
                            circle_radius=0.0762, axis_labels=True, show_axis=True, show_axis_ticks=True, fontsize=10,
                            include_motor_orientation=True):
    """
    Plot axes at given positions with specified rotations and draw lines from each point to the origin.
    Represent the direction of spin with circles and directional arrows.

    :param individual: List of lists (x, y, z, rx, ry, rz, dir) representing the positions, rotations and direction of motor rotation.
    """
    import numpy as np
    
    # Convert to numpy array and handle NaN values
    if isinstance(individual, str):
        individual = convert_str_to_nparray(individual)
    
    individual = np.array(individual)
    # Remove any rows that are all NaN
    valid_rows = ~np.isnan(individual).all(axis=1)
    individual = individual[valid_rows]
    
    if len(individual) == 0:
        ax.text(0.5, 0.5, 0.5, 'No valid\nindividual data', 
               ha='center', va='center', transform=ax.transData)
        return None, None

    # Utility function for converting spherical to cartesian coordinates
    def convert_to_cartesian(mag, aya, api):
        x = mag * np.sin(api) * np.cos(aya)
        y = mag * np.sin(api) * np.sin(aya)
        z = mag * np.cos(api)
        return x, y, z

    # start at 0 to include origin
    mx = 0

    for arm in individual:
        if len(arm) < 6 or np.isnan(arm).any():
            continue
            
        mag, aya, api, mya, mpi, dir = arm[:6]
        x, y, z = convert_to_cartesian(mag, aya, api)
        
        # Rotation matrices
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(0), -np.sin(0)],
                       [0, np.sin(0), np.cos(0)]])
        
        Ry = np.array([[np.cos(-mpi), 0, np.sin(-mpi)],
                       [0, 1, 0],
                       [-np.sin(-mpi), 0, np.cos(-mpi)]])
        
        Rz = np.array([[np.cos(mya), -np.sin(mya), 0],
                       [np.sin(mya), np.cos(mya), 0],
                       [0, 0, 1]])
        
        # Combined rotation matrix
        R = Rz @ Ry @ Rx
        
        # Axes vectors
        axis_length = circle_radius*2
        z_axis = R @ np.array([0, 0, axis_length])
        
        if include_motor_orientation:
            ax.plot([x, x+z_axis[0]], [y, y+z_axis[1]], [z, z+z_axis[2]], 'r')
        
        # Draw line to the origin
        ax.plot([0, x], [0, y], [0, z], 'k--')

        # Draw rotation circle
        num_points = 100
        circle = np.array([[circle_radius * np.cos(t), circle_radius * np.sin(t), 0] for t in np.linspace(0, 2 * np.pi, num_points)])
        
        # Rotate circle to match the given rotations
        rotated_circle = (R @ circle.T).T + np.array([x, y, z])
        ax.plot(rotated_circle[:, 0], rotated_circle[:, 1], rotated_circle[:, 2], 'm')

        # Draw directional arrow on the circle
        if dir == 1:
            arrow_start = rotated_circle[-15,:]
            arrow_end = rotated_circle[-1,:]
        else:  # 'ccw'
            arrow_start = rotated_circle[15,:]
            arrow_end = rotated_circle[0,:]
        
        new_max = np.max(np.abs([x,y,z]))
        if new_max > mx:
            mx = new_max
        
        try:
            ax.quiver(arrow_start[0], arrow_start[1], arrow_start[2],
                      arrow_end[0] - arrow_start[0], arrow_end[1] - arrow_start[1], arrow_end[2] - arrow_start[2],
                      color='m', arrow_length_ratio=1)
        except:
            pass  # Skip arrow if it fails

    # Set plot limits
    scale_factor = 1.2
    if mx > 0:
        ax.set_xlim([-mx*scale_factor, mx*scale_factor])
        ax.set_ylim([-mx*scale_factor, mx*scale_factor])
        ax.set_zlim([-mx*scale_factor, mx*scale_factor])
    
    # Labels
    if axis_labels:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    if not show_axis:
        ax.set_axis_off()
    if not show_axis_ticks:
        ax.tick_params(labelbottom=False, labeltop=False, labelleft=False, labelright=False)
    
    if generation is not None and fitness is not None:
        ax.set_title(f'G: {generation}, F: {np.round(fitness,2)}', fontsize=fontsize)
    elif generation is not None and fitness is None:
        ax.set_title(f'G: {generation}', fontsize=fontsize)
    if fitness is not None and generation is None:
        ax.set_title(f'F: {np.round(fitness,2)}', fontsize=fontsize)

    # Set perspective
    ax.view_init(elev=elev, azim=azim)
    return None, None

def statistical_analysis(all_data, save_dir=None):
    """Perform statistical analysis on the relationships between metrics and fitness."""
    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS")
    print("="*60)
    
    metrics = ['Central_Asymmetry', 'Bilateral_Asymmetry']
    results = []
    
    for exp_name, df in all_data.items():
        print(f"\n{exp_name.upper()}:")
        print("-" * 40)
        
        for metric in metrics:
            # Remove NaN values
            valid_data = df.dropna(subset=[metric, 'fitness'])
            
            if len(valid_data) > 10:
                # Pearson correlation (linear relationship)
                pearson_corr, pearson_p = pearsonr(valid_data['fitness'], valid_data[metric])
                
                # Spearman correlation (monotonic relationship)
                spearman_corr, spearman_p = spearmanr(valid_data['fitness'], valid_data[metric])
                
                print(f"{metric}:")
                print(f"  Pearson correlation: {pearson_corr:.4f} (p={pearson_p:.4f})")
                print(f"  Spearman correlation: {spearman_corr:.4f} (p={spearman_p:.4f})")
                
                # Store results
                results.append({
                    'experiment': exp_name,
                    'metric': metric,
                    'pearson_r': pearson_corr,
                    'pearson_p': pearson_p,
                    'spearman_r': spearman_corr,
                    'spearman_p': spearman_p,
                    'n_samples': len(valid_data)
                })
                
                # Interpretation
                if abs(pearson_corr) > 0.7:
                    strength = "strong"
                elif abs(pearson_corr) > 0.3:
                    strength = "moderate"
                else:
                    strength = "weak"
                
                direction = "positive" if pearson_corr > 0 else "negative"
                significance = "significant" if pearson_p < 0.05 else "not significant"
                
                print(f"  → {strength} {direction} correlation ({significance})")
            else:
                print(f"{metric}: Insufficient data (n={len(valid_data)})")
    
    # Create summary table
    results_df = pd.DataFrame(results)
    
    # Save results if directory provided
    if save_dir:
        results_df.to_csv(os.path.join(save_dir, 'correlation_analysis.csv'), index=False)
    
    # Summary of significant correlations
    print(f"\n{'='*60}")
    print("SUMMARY OF SIGNIFICANT CORRELATIONS (p < 0.05)")
    print("="*60)
    
    significant_results = results_df[results_df['pearson_p'] < 0.05]
    
    if len(significant_results) > 0:
        for _, row in significant_results.iterrows():
            direction = "positive" if row['pearson_r'] > 0 else "negative"
            print(f"{row['experiment']} - {row['metric']}: "
                  f"{row['pearson_r']:.3f} ({direction}, p={row['pearson_p']:.4f})")
    else:
        print("No significant correlations found.")
    
    return results_df

def plot_combined_metrics_vs_fitness_with_dominance(all_data, save_dir=None, 
                                                   main_fontsize=30, title_fontsize=14, 
                                                   morph_title_fontsize=15, explanation_fontsize=8,
                                                   tick_fontsize=25):
    """Plot combined metrics vs fitness with task dominance based on fitness levels and morphology examples."""
    metrics = ['Central_Asymmetry', 'Bilateral_Asymmetry']
    
    # Task name mapping
    task_name_mapping = {
        "asym_circle": "Circle",
        "asym_slalom": "Slalom", 
        "asym_backnforth": "Shuttlerun",
        "asym_figure8": "Figure 8"
    }
    
    for metric in metrics:
        # Create figure with main plot and morphology subplots
        fig = plt.figure(figsize=(20, 8))
        
        # Main plot takes up most of the space
        ax_main = plt.subplot2grid((2, 6), (0, 0), colspan=4, rowspan=2)
        
        # Morphology subplots on the right
        ax_morph1 = plt.subplot2grid((2, 6), (0, 4), projection='3d')
        ax_morph2 = plt.subplot2grid((2, 6), (1, 4), projection='3d') 
        # ax_morph3 = plt.subplot2grid((2, 6), (0, 5), projection='3d')
        # ax_morph4 = plt.subplot2grid((2, 6), (1, 5), projection='3d')
        
        morph_axes = [ax_morph2, ax_morph1]
        
        # Collect data and find max fitness for each task
        task_data = {}
        max_fitnesses = {}
        
        for exp_name, df in all_data.items():
            valid_data = df.dropna(subset=[metric, 'fitness'])
            if len(valid_data) > 0:
                mapped_name = task_name_mapping.get(exp_name, exp_name)
                task_data[mapped_name] = valid_data
                max_fitnesses[mapped_name] = valid_data['fitness'].max()
        
        if not task_data:
            print(f"No valid data for {metric}")
            continue
        
        # Sort tasks by max fitness (ascending order)
        sorted_tasks = sorted(max_fitnesses.items(), key=lambda x: x[1])
        print(f"\n{metric} - Task max fitnesses (sorted):")
        for task, max_fit in sorted_tasks:
            print(f"  {task}: {max_fit:.3f}")
        
        # Define colors for each task
        task_colors = {
            sorted_tasks[0][0]: 'red',
            sorted_tasks[1][0]: 'blue', 
            sorted_tasks[2][0]: 'green',
            sorted_tasks[3][0]: 'orange'
        }
        
        # Create fitness ranges for dominance
        fitness_ranges = []
        for i, (task, max_fit) in enumerate(sorted_tasks):
            start_fit = sorted_tasks[i-1][1] if i > 0 else 0
            fitness_ranges.append((task, start_fit, max_fit))
        
        # Plot data with dynamic emphasis
        all_fitness_values = []
        for task_name, data in task_data.items():
            all_fitness_values.extend(data['fitness'].tolist())
        
        fitness_min, fitness_max = min(all_fitness_values), max(all_fitness_values)
        
        # Store morphology examples - only for figure8 task
        task_morphology_examples = {}
        
        # For each fitness level, determine which task should be emphasized
        for task_idx, (task_name, data) in enumerate(task_data.items()):
            fitness_vals = data['fitness'].values
            metric_vals = data[metric].values
            
            # Create arrays to store emphasized and faded points
            emphasized_fitness = []
            emphasized_metric = []
            faded_fitness = []
            faded_metric = []
            
            for f, m in zip(fitness_vals, metric_vals):
                # Determine which task should dominate at this fitness level
                dominant_task = None
                for range_task, start_f, end_f in fitness_ranges:
                    if start_f <= f <= end_f:
                        dominant_task = range_task
                        break
                
                # If no dominant task found (above all max), use the highest max task
                if dominant_task is None:
                    dominant_task = sorted_tasks[-1][0]
                
                if task_name == dominant_task:
                    emphasized_fitness.append(f)
                    emphasized_metric.append(m)
                else:
                    faded_fitness.append(f)
                    faded_metric.append(m)
            
            # Plot faded points first (lower z-order)
            if faded_fitness:
                ax_main.scatter(faded_fitness, faded_metric, 
                          color=task_colors[task_name], alpha=0.2, s=8)
            
            # Plot emphasized points
            if emphasized_fitness:
                ax_main.scatter(emphasized_fitness, emphasized_metric, 
                          color=task_colors[task_name], alpha=0.8, s=25, zorder=5)
            
            # Add trend line for ALL data points (not just emphasized)
            if len(fitness_vals) > 5:
                z = np.polyfit(fitness_vals, metric_vals, 1)
                p = np.poly1d(z)
                fit_range = np.linspace(fitness_vals.min(), fitness_vals.max(), 100)
                ax_main.plot(fit_range, p(fit_range), '--', 
                           color=task_colors[task_name], alpha=0.9, linewidth=2)
            
            # Collect morphology examples ONLY for figure8 task
            if task_name == 'Figure 8':
                # Calculate fitness threshold (75th percentile) for high-fitness examples
                fitness_threshold = np.percentile(data['fitness'], 75)
                high_fitness_data = data[data['fitness'] >= fitness_threshold]
                
                high_fitness_individuals = []
                for idx, row in high_fitness_data.iterrows():
                    if 'offspring' in data.columns and pd.notna(data.loc[idx, 'offspring']):
                        try:
                            individual = convert_str_to_nparray(data.loc[idx, 'offspring'])
                            if len(individual) > 0:
                                high_fitness_individuals.append({
                                    'individual': individual,
                                    'metric_value': row[metric],
                                    'fitness': row['fitness'],
                                    'generation': row['generation'],
                                    'task': task_name
                                })
                        except:
                            continue
                
                if len(high_fitness_individuals) >= 2:
                    # Sort by metric value and get extremes
                    sorted_individuals = sorted(high_fitness_individuals, key=lambda x: x['metric_value'])
                    
                    # Store low and high examples for this task
                    task_morphology_examples[task_name] = {
                        'low': sorted_individuals[0],
                        'high': sorted_individuals[-1]
                    }
        
        # Plot morphology examples (only figure8)
        example_count = 0
        for task_idx, (task_name, examples) in enumerate(task_morphology_examples.items()):
            if example_count >= len(morph_axes):
                break
                
            # Plot low metric value example
            if example_count < len(morph_axes):
                ax = morph_axes[example_count]
                example = examples['low']
                
                try:
                    represent_morphology_3d(ax, example['individual'], 
                                          fitness=example['fitness'],
                                          generation=example['generation'],
                                          elev=0, azim=90, axis_labels=False, 
                                          show_axis_ticks=False, fontsize=morph_title_fontsize)
                    ax.set_title(f'{task_name} Low {metric.replace("_", " ")}\nMetric: {example["metric_value"]:.3f} Fitness: {example["fitness"]:.3f}', 
                               fontsize=morph_title_fontsize)
                    
                    # Highlight this example on main plot
                    ax_main.scatter([example['fitness']], [example['metric_value']], 
                                  color=task_colors[task_name], s=150, marker='v', 
                                  edgecolors='black', linewidth=2, zorder=10)
                    
                except Exception as e:
                    ax.text(0.5, 0.5, 0.5, 'Visualization\nError', 
                           ha='center', va='center')
                    print(f"    Error visualizing {task_name} low example: {e}")
                
                example_count += 1
            
            # Plot high metric value example
            if example_count < len(morph_axes):
                ax = morph_axes[example_count]
                example = examples['high']
                
                try:
                    represent_morphology_3d(ax, example['individual'], 
                                          fitness=example['fitness'],
                                          generation=example['generation'],
                                          elev=0, azim=90, axis_labels=False, 
                                          show_axis_ticks=False, fontsize=morph_title_fontsize)
                    ax.set_title(f'{task_name} High {metric.replace("_", " ")}\nMetric: {example["metric_value"]:.3f} Fitness: {example["fitness"]:.3f}', 
                               fontsize=morph_title_fontsize)
                    
                    # Highlight this example on main plot
                    ax_main.scatter([example['fitness']], [example['metric_value']], 
                                  color=task_colors[task_name], s=150, marker='^', 
                                  edgecolors='black', linewidth=2, zorder=10)
                    
                except Exception as e:
                    ax.text(0.5, 0.5, 0.5, 'Visualization\nError', 
                           ha='center', va='center')
                    print(f"    Error visualizing {task_name} high example: {e}")
                
                example_count += 1
        
        # Fill remaining morphology subplots with placeholder text
        for i in range(example_count, len(morph_axes)):
            morph_axes[i].text(0.5, 0.5, 0.5, 'No morphology\ndata available', 
                              ha='center', va='center', transform=morph_axes[i].transAxes,
                              fontsize=morph_title_fontsize)
            morph_axes[i].set_axis_off()
        
        # Draw vertical lines at max fitness levels
        for i, (task, max_fit) in enumerate(sorted_tasks):
            line_style = '-' if i < len(sorted_tasks) - 1 else ':'
            ax_main.axvline(x=max_fit, color=task_colors[task], linestyle=line_style, 
                      linewidth=2, alpha=0.7)
        
        # Add density background for all data combined
        all_fitness = []
        all_metric_vals = []
        for data in task_data.values():
            all_fitness.extend(data['fitness'].tolist())
            all_metric_vals.extend(data[metric].tolist())
        
        if len(all_fitness) > 100:
            try:
                hist, x_edges, y_edges = np.histogram2d(
                    all_fitness, all_metric_vals, bins=50, density=True
                )
                extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
                ax_main.imshow(hist.T, origin='lower', extent=extent, 
                         aspect='auto', alpha=0.15, cmap='Greys', zorder=0)
            except:
                pass
        
        # Formatting
        ax_main.set_xlabel('Fitness', fontsize=main_fontsize)
        ax_main.set_ylabel(metric.replace('_', ' '), fontsize=main_fontsize)
        ax_main.tick_params(axis='both', which='major', labelsize=tick_fontsize)

        # ax_main.set_title(f'{metric.replace("_", " ")} vs Fitness - Task Dominance by Fitness Level', fontsize=title_fontsize)
        
        # NO LEGENDS on main plot
        
        ax_main.grid(True, alpha=0.3)
        
        # Add explanation text
        # explanation = ("Tasks emphasized when fitness is in their dominance range. Trend lines use ALL data per task.\n"
        #               "Triangles: ▲ = High metric, ▼ = Low metric examples (figure8 only). Morphologies shown on right.")
        # ax_main.text(0.02, 0.98, explanation, transform=ax_main.transAxes, fontsize=explanation_fontsize, 
        #        verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", alpha=0.8, facecolor='white'))
        
        plt.tight_layout()
        
        if save_dir:
            filename = f'combined_{metric.lower()}_vs_fitness_task_dominance_with_morphology.png'
            plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
                
        # Create separate legend figure
        create_legend_figure(task_colors, sorted_tasks, save_dir, metric, main_fontsize)

def create_legend_figure(task_colors, sorted_tasks, save_dir=None, metric='', legend_fontsize=12):
    """Create a separate figure containing only the legend."""
    fig_legend, ax_legend = plt.subplots(figsize=(10, 6))
    ax_legend.axis('off')
    
    # Create legend elements
    legend_elements = []
    
    # Task colors and emphasis
    for task_name, color in task_colors.items():
        from matplotlib.patches import Patch
        legend_elements.append(Patch(facecolor=color, alpha=0.8, label=f'{task_name} (emphasized)'))
        legend_elements.append(Patch(facecolor=color, alpha=0.2, label=f'{task_name} (faded)'))
    
    # Trend lines
    from matplotlib.lines import Line2D
    for task_name, color in task_colors.items():
        legend_elements.append(Line2D([0], [0], color=color, linestyle='--', 
                                     alpha=0.9, linewidth=2, label=f'{task_name} trend'))
    
    # Max fitness lines
    for i, (task, max_fit) in enumerate(sorted_tasks):
        line_style = '-' if i < len(sorted_tasks) - 1 else ':'
        legend_elements.append(Line2D([0], [0], color=task_colors[task], 
                                     linestyle=line_style, linewidth=2, alpha=0.7,
                                     label=f'{task} max: {max_fit:.2f}'))
    
    # Morphology markers (only for figure8)
    if 'figure8' in task_colors:
        legend_elements.append(Line2D([0], [0], marker='v', color='w', 
                                     markerfacecolor=task_colors['figure8'], 
                                     markersize=10, markeredgecolor='black',
                                     label='Low metric example (figure8)', linestyle='None'))
        legend_elements.append(Line2D([0], [0], marker='^', color='w', 
                                     markerfacecolor=task_colors['figure8'], 
                                     markersize=10, markeredgecolor='black',
                                     label='High metric example (figure8)', linestyle='None'))
    
    # Create the legend with multiple columns
    legend = ax_legend.legend(handles=legend_elements, loc='center', 
                            ncol=3, fontsize=legend_fontsize,
                            title=f'Legend for {metric.replace("_", " ")} vs Fitness Plot',
                            title_fontsize=legend_fontsize+2)
    
    plt.tight_layout()
    
    if save_dir:
        filename = f'legend_{metric.lower()}_vs_fitness_task_dominance.png'
        plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    
    
def create_correlation_heatmap(all_data, save_dir=None):
    """Create a heatmap showing correlations between metrics and fitness for each experiment."""
    metrics = ['Central_Asymmetry', 'Bilateral_Asymmetry','fitness']
    
    n_experiments = len(all_data)
    fig, axes = plt.subplots(1, n_experiments, figsize=(5*n_experiments, 4))
    
    if n_experiments == 1:
        axes = [axes]
    
    for i, (exp_name, df) in enumerate(all_data.items()):
        # Calculate correlation matrix
        valid_data = df[metrics].dropna()
        corr_matrix = valid_data.corr()
        
        # Create heatmap
        sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, 
                   square=True, ax=axes[i], cbar_kws={'shrink': 0.8})
        axes[i].set_title(f'{exp_name}')
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'correlation_heatmap.png'), dpi=300, bbox_inches='tight')
    
    plt.show()

def main():
    """Main analysis function."""
    # Define experiment directories (modify these paths as needed)
    experiment_dirs = {
        "asym_figure8": "data_backup/asym_figure8/",
        "asym_circle": "data_backup/asym_circle/",
        "asym_slalom": "data_backup/asym_slalom/",
        "asym_backnforth": "data_backup/asym_backnforth/",
    }
    
    # Set output directory for saving plots and results
    output_dir = "./analysis_results/"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading experiment data...")
    all_data = load_experiment_data(experiment_dirs)
    
    if not all_data:
        print("No data loaded. Check your file paths.")
        return
    
    print(f"\nLoaded data from {len(all_data)} experiments")
    
    # Create visualizations
    print("\nGenerating plots...")
    # plot_metrics_over_generations(all_data, output_dir)
    # plot_metrics_vs_fitness(all_data, output_dir)  # Individual task plots
    plot_combined_metrics_vs_fitness_with_dominance(all_data, output_dir)  # Combined smart plots
    create_correlation_heatmap(all_data, output_dir)
    
    # Perform statistical analysis
    correlation_results = statistical_analysis(all_data, output_dir)
    
    print(f"\nAnalysis complete! Results saved to: {output_dir}")

if __name__ == "__main__":
    main()