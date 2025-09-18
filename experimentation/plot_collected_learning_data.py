import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from scipy import stats

def calculate_learning_speed(df):
    """
    Calculate learning speed as max_reward / (convergence_start_time - burnin_end_time).
    Handles NaN values and division by zero cases.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the learning data
    
    Returns:
    --------
    pandas.Series
        Series containing learning speed values
    """
    # Calculate the denominator (time difference)
    time_diff = df['burnin_end_time']
    
    # Calculate learning speed, handling NaN and division by zero
    learning_speed = np.where(
        (pd.isna(df['max_reward']) | pd.isna(time_diff) | (time_diff == 0)),
        np.nan,
        df['max_reward'] / time_diff
    )
    
    return pd.Series(learning_speed, index=df.index)

def calculate_confidence_interval(data, confidence_level=0.95):
    """
    Calculate confidence interval for the mean of data.
    
    Parameters:
    -----------
    data : array-like
        Data points
    confidence_level : float
        Confidence level (default: 0.95 for 95% CI)
    
    Returns:
    --------
    tuple
        (mean, lower_bound, upper_bound, sem)
    """
    # Remove NaN values
    clean_data = data[~np.isnan(data)]
    
    if len(clean_data) == 0:
        return np.nan, np.nan, np.nan, np.nan
    
    if len(clean_data) == 1:
        # Single data point - no confidence interval
        return clean_data[0], clean_data[0], clean_data[0], 0.0
    
    # Calculate mean and standard error
    mean = np.mean(clean_data)
    sem = stats.sem(clean_data)  # Standard error of the mean
    
    # Calculate confidence interval using t-distribution
    df_freedom = len(clean_data) - 1
    t_critical = stats.t.ppf((1 + confidence_level) / 2, df_freedom)
    margin_of_error = t_critical * sem
    
    lower_bound = mean - margin_of_error
    upper_bound = mean + margin_of_error
    
    return mean, lower_bound, upper_bound, sem

def process_directory_collection(base_dir, fontsize=12, tick_fontsize=10, figsize=(15, 12), 
                                output_base_dir=None, summary_only=False, use_ci=False, 
                                confidence_level=0.95):
    """
    Process a collection of directories containing learning_data.csv files.
    
    Parameters:
    -----------
    base_dir : str
        Path to directory containing subdirectories with learning_data.csv files
    fontsize : int
        Font size for labels
    tick_fontsize : int
        Font size for tick labels
    figsize : tuple
        Figure size (width, height)
    output_base_dir : str, optional
        Base directory for saving plots. If None, saves in base_dir/plots/
    summary_only : bool
        If True, create only summary plots
    use_ci : bool
        If True, use confidence intervals instead of standard deviation
    confidence_level : float
        Confidence level for confidence intervals (default: 0.95)
    """
    
    if output_base_dir is None:
        output_base_dir = os.path.join(base_dir, 'plots')
    
    # Create base output directory
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Find all subdirectories with learning_data.csv files
    subdirs_with_data = []
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            csv_path = os.path.join(item_path, 'learning_data.csv')
            if os.path.exists(csv_path):
                subdirs_with_data.append((item, csv_path))
    
    if not subdirs_with_data:
        print(f"No subdirectories with learning_data.csv found in {base_dir}")
        return
    
    print(f"Found {len(subdirs_with_data)} directories with learning data:")
    for dirname, _ in subdirs_with_data:
        print(f"  - {dirname}")
    
    # Process each directory and collect stats for combined plots
    all_task_stats = {}  # Will store stats for each task/directory
    
    for dirname, csv_path in subdirs_with_data:
        print(f"\nProcessing {dirname}...")
        
        # Create output directory for this experiment
        experiment_output_dir = os.path.join(output_base_dir, dirname)
        os.makedirs(experiment_output_dir, exist_ok=True)
        
        # Plot metrics for this directory
        stats_df = None  # Initialize to handle variable scope
        try:
            if summary_only:
                figures = plot_metric_summary(csv_path, fontsize, tick_fontsize, 
                                            experiment_output_dir, use_ci, confidence_level)
                print(f"  Completed {dirname}: {len(figures)} summary plots saved")
            else:
                plot_result = plot_learning_metrics(csv_path, fontsize, tick_fontsize, figsize, 
                                                  experiment_output_dir, use_ci, confidence_level)
                if plot_result is not None:
                    figures, stats_df = plot_result
                    # summary_figures = plot_metric_summary(csv_path, fontsize, tick_fontsize, 
                    #                                     experiment_output_dir, use_ci, confidence_level)
                    print(f"  Completed {dirname}: {len(figures)} detailed plots + {len(summary_figures)} summary plots saved")
                else:
                    print(f"  Warning: plot_learning_metrics returned None for {dirname}")
                    continue
            
            # Store stats for combined plotting (always calculate stats even for summary_only)
            if summary_only or stats_df is None:
                # Calculate stats if we only did summary plots or if stats_df is None
                df = pd.read_csv(csv_path)
                df['learning_speed'] = calculate_learning_speed(df)
                generations = sorted(df['generation'].unique())
                
                # Get all metrics for combined plotting
                metric_columns = [
                    'burnin_start_time', 'burnin_end_time', 'burnin_start_performance', 
                    'burnin_end_performance', 'burnin_speed',
                    'convergence_start_time', 'convergence_end_time', 'convergence_start_performance', 
                    'convergence_end_performance', 'convergence_speed',
                    'intermediate_time', 'intermediate_performance', 'intermediate_speed',
                    'max_reward', 'volatility', 'learning_speed'
                ]
                
                generation_stats = []
                for gen in generations:
                    gen_data = df[df['generation'] == gen]
                    gen_stats = {'generation': gen}
                    
                    for metric in metric_columns:
                        if use_ci:
                            mean, lower, upper, sem = calculate_confidence_interval(
                                gen_data[metric].values, confidence_level)
                            gen_stats[f'{metric}_mean'] = mean
                            gen_stats[f'{metric}_lower'] = lower
                            gen_stats[f'{metric}_upper'] = upper
                            gen_stats[f'{metric}_sem'] = sem
                        else:
                            gen_stats[f'{metric}_mean'] = np.nanmean(gen_data[metric])
                            gen_stats[f'{metric}_std'] = np.nanstd(gen_data[metric])
                    
                    generation_stats.append(gen_stats)
                
                all_task_stats[dirname] = pd.DataFrame(generation_stats)
            else:
                # Use the stats_df from plot_learning_metrics
                all_task_stats[dirname] = stats_df
                
        except Exception as e:
            print(f"  Error processing {dirname}: {e}")
    
    # Create combined plots for all tasks
    print(f"all_task_stats keys: {list(all_task_stats.keys())}")
    if len(all_task_stats) > 1:
        print(f"\nCreating combined plots for all {len(all_task_stats)} tasks...")
        combined_figures = plot_all_tasks_combined(all_task_stats, fontsize, tick_fontsize, 
                                                 output_base_dir, use_ci, confidence_level)
        print(f"Combined plots saved: {len(combined_figures)} plots in {output_base_dir}")
    
    print(f"\nAll plots saved in: {output_base_dir}")

def plot_learning_metrics(csv_path, fontsize=12, tick_fontsize=10, figsize=(15, 12), 
                         output_dir=None, use_ci=False, confidence_level=0.95):
    """
    Plot learning metrics over generations from saved CSV data.
    
    Parameters:
    -----------
    csv_path : str
        Path to the learning_data.csv file
    fontsize : int
        Font size for labels
    tick_fontsize : int
        Font size for tick labels
    figsize : tuple
        Figure size (width, height)
    output_dir : str, optional
        Directory to save plots. If None, saves in same directory as CSV
    use_ci : bool
        If True, use confidence intervals instead of standard deviation
    confidence_level : float
        Confidence level for confidence intervals (default: 0.95)
    """
    
    # Read the CSV data
    df = pd.read_csv(csv_path)
    
    # Calculate learning speed
    df['learning_speed'] = calculate_learning_speed(df)
    
    # Get the metrics to plot (excluding ID columns)
    metric_columns = [
        'burnin_start_time', 'burnin_end_time', 'burnin_start_performance', 
        'burnin_end_performance', 'burnin_speed',
        'convergence_start_time', 'convergence_end_time', 'convergence_start_performance', 
        'convergence_end_performance', 'convergence_speed',
        'intermediate_time', 'intermediate_performance', 'intermediate_speed',
        'max_reward', 'volatility', 'learning_speed'
    ]
    
    # Create nice labels for plotting
    metric_labels = [
        'Burnin Start Time', 'Time of Max Learning Rate', 'Burnin Start Performance', 
        'Burnin End Performance', 'Burnin Speed',
        'Convergence Start Time', 'Convergence End Time', 'Convergence Start Performance', 
        'Convergence End Performance', 'Convergence Speed',
        'Intermediate Time', 'Intermediate Performance', 'Intermediate Speed',
        'Max Reward', 'Stability', 'Learning Speed'
    ]
    
    # Get unique generations and sort them
    generations = sorted(df['generation'].unique())
    
    # Calculate mean and confidence intervals or std for each metric across generations
    generation_stats = []
    for gen in generations:
        gen_data = df[df['generation'] == gen]
        gen_stats = {}
        gen_stats['generation'] = gen
        
        for metric in metric_columns:
            if use_ci:
                mean, lower, upper, sem = calculate_confidence_interval(
                    gen_data[metric].values, confidence_level)
                gen_stats[f'{metric}_mean'] = mean
                gen_stats[f'{metric}_lower'] = lower
                gen_stats[f'{metric}_upper'] = upper
                gen_stats[f'{metric}_sem'] = sem
            else:
                # Use nanmean and nanstd to ignore NaN values
                gen_stats[f'{metric}_mean'] = np.nanmean(gen_data[metric])
                gen_stats[f'{metric}_std'] = np.nanstd(gen_data[metric])
        
        generation_stats.append(gen_stats)
    
    stats_df = pd.DataFrame(generation_stats)
    
    # Create separate figures for each metric
    figures = []
    
    if output_dir is None:
        output_dir = os.path.dirname(csv_path)
    
    # Determine legend labels and filename suffix
    if use_ci:
        interval_label = f'{int(confidence_level*100)}% CI'
        filename_suffix = f'_{int(confidence_level*100)}ci'
    else:
        interval_label = '±1 Std'
        filename_suffix = '_std'
    
    for i, (metric, label) in enumerate(zip(metric_columns, metric_labels)):
        # Create a new figure for each metric
        fig, ax = plt.subplots(figsize=(8, 6))
        
        mean_col = f'{metric}_mean'
        
        if use_ci:
            lower_col = f'{metric}_lower'
            upper_col = f'{metric}_upper'
            
            # Get data, filtering out NaN values for plotting
            valid_mask = ~(np.isnan(stats_df[mean_col]) | 
                          np.isnan(stats_df[lower_col]) | 
                          np.isnan(stats_df[upper_col]))
            plot_generations = stats_df.loc[valid_mask, 'generation']
            plot_means = stats_df.loc[valid_mask, mean_col]
            plot_lower = stats_df.loc[valid_mask, lower_col]
            plot_upper = stats_df.loc[valid_mask, upper_col]
            
            if len(plot_generations) > 0:
                # Plot mean line in red
                ax.plot(plot_generations, plot_means, color='red', linewidth=2, label='Mean')
                
                # Plot confidence interval as shaded area
                ax.fill_between(plot_generations, plot_lower, plot_upper, 
                              color='red', alpha=0.3, label=interval_label)
        else:
            std_col = f'{metric}_std'
            
            # Get data, filtering out NaN values for plotting
            valid_mask = ~(np.isnan(stats_df[mean_col]) | np.isnan(stats_df[std_col]))
            plot_generations = stats_df.loc[valid_mask, 'generation']
            plot_means = stats_df.loc[valid_mask, mean_col]
            plot_stds = stats_df.loc[valid_mask, std_col]
            
            if len(plot_generations) > 0:
                # Plot mean line in red
                ax.plot(plot_generations, plot_means, color='red', linewidth=2, label='Mean')
                
                # Plot standard deviation as shaded area
                ax.fill_between(plot_generations, 
                              plot_means - plot_stds, 
                              plot_means + plot_stds, 
                              color='red', alpha=0.3, label=interval_label)
        
        # Customize appearance
        ax.set_xlabel('Generation', fontsize=fontsize)
        ax.set_ylabel(label, fontsize=fontsize)
        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
        ax.grid(True, alpha=1.0)
        # ax.legend(fontsize=fontsize-2)
        
        plt.tight_layout()
        
        # Save individual plot
        safe_filename = metric.replace('_', '-')
        output_path = os.path.join(output_dir, f'{safe_filename}_over_generations{filename_suffix}.png')
        if f'{safe_filename}_over_generations{filename_suffix}.png' in ['burnin-end-time_over_generations_95ci.png',
                                                                        'volatility_over_generations_95ci.png',
                                                                        'learning-speed_over_generations_95ci.png',
                                                                        'max-reward_over_generations_95ci.png']:
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {output_path}")
        
        figures.append(fig)
        plt.show()
    
    # Also save the aggregated statistics
    stats_output_path = os.path.join(output_dir, f'generation_statistics{filename_suffix}.csv')
    stats_df.to_csv(stats_output_path, index=False)
    print(f"Generation statistics saved to: {stats_output_path}")
    return figures, stats_df
    
def plot_all_tasks_combined(all_task_stats, fontsize=12, tick_fontsize=10, output_dir=None, 
                          use_ci=False, confidence_level=0.95):
    """
    Create combined plots showing all tasks on the same graph for each metric.
    
    Parameters:
    -----------
    all_task_stats : dict
        Dictionary mapping task names to their statistics DataFrames
    fontsize : int
        Font size for labels
    tick_fontsize : int
        Font size for tick labels
    output_dir : str
        Directory to save combined plots
    use_ci : bool
        If True, use confidence intervals instead of standard deviation
    confidence_level : float
        Confidence level for confidence intervals
    """
    
    if not all_task_stats:
        print("No task statistics available for combined plotting")
        return []
    
    # Get all metrics from the first task's DataFrame
    first_task_stats = list(all_task_stats.values())[0]
    
    # Extract metric names from column names (remove '_mean', '_std', '_lower', '_upper' suffixes)
    metric_names = set()
    for col in first_task_stats.columns:
        if col.endswith('_mean'):
            metric_names.add(col[:-5])  # Remove '_mean'
    
    # Remove 'generation' if it somehow got included
    metric_names.discard('generation')
    
    metric_names = sorted(list(metric_names))
    
    # Create nice labels for plotting
    metric_labels = {
        'burnin_start_time': 'Burnin Start Time',
        'burnin_end_time': 'Burnin End Time',
        'burnin_start_performance': 'Burnin Start Performance',
        'burnin_end_performance': 'Burnin End Performance',
        'burnin_speed': 'Burnin Speed',
        'convergence_start_time': 'Convergence Start Time',
        'convergence_end_time': 'Convergence End Time',
        'convergence_start_performance': 'Convergence Start Performance',
        'convergence_end_performance': 'Convergence End Performance',
        'convergence_speed': 'Convergence Speed',
        'intermediate_time': 'Intermediate Time',
        'intermediate_performance': 'Intermediate Performance',
        'intermediate_speed': 'Intermediate Speed',
        'max_reward': 'Max Reward',
        'volatility': 'Volatility',
        'learning_speed': 'Learning Speed'
    }
    
    # Define colors for different tasks
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_task_stats)))
    
    figures = []
    
    # Determine filename suffix
    filename_suffix = f'_{int(confidence_level*100)}ci' if use_ci else '_std'
    
    for metric in metric_names:
        # Create a new figure for each metric
        fig, ax = plt.subplots(figsize=(10, 6))
        
        mean_col = f'{metric}_mean'
        
        # Plot each task on the same graph
        for i, (task_name, stats_df) in enumerate(all_task_stats.items()):
            if mean_col in stats_df.columns:
                color = colors[i]
                
                if use_ci:
                    lower_col = f'{metric}_lower'
                    upper_col = f'{metric}_upper'
                    
                    if lower_col in stats_df.columns and upper_col in stats_df.columns:
                        # Get data, filtering out NaN values for plotting
                        valid_mask = ~(np.isnan(stats_df[mean_col]) | 
                                      np.isnan(stats_df[lower_col]) | 
                                      np.isnan(stats_df[upper_col]))
                        plot_generations = stats_df.loc[valid_mask, 'generation']
                        plot_means = stats_df.loc[valid_mask, mean_col]
                        plot_lower = stats_df.loc[valid_mask, lower_col]
                        plot_upper = stats_df.loc[valid_mask, upper_col]
                        
                        if len(plot_generations) > 0:
                            # Plot mean line
                            ax.plot(plot_generations, plot_means, color=color, linewidth=2, 
                                   label=f'{task_name}', alpha=0.8)
                            
                            # Plot confidence interval as shaded area
                            ax.fill_between(plot_generations, plot_lower, plot_upper,
                                          color=color, alpha=0.2)
                else:
                    std_col = f'{metric}_std'
                    
                    if std_col in stats_df.columns:
                        # Get data, filtering out NaN values for plotting
                        valid_mask = ~(np.isnan(stats_df[mean_col]) | np.isnan(stats_df[std_col]))
                        plot_generations = stats_df.loc[valid_mask, 'generation']
                        plot_means = stats_df.loc[valid_mask, mean_col]
                        plot_stds = stats_df.loc[valid_mask, std_col]
                        
                        if len(plot_generations) > 0:
                            # Plot mean line
                            ax.plot(plot_generations, plot_means, color=color, linewidth=2, 
                                   label=f'{task_name}', alpha=0.8)
                            
                            # Plot standard deviation as shaded area
                            ax.fill_between(plot_generations, 
                                          plot_means - plot_stds, 
                                          plot_means + plot_stds, 
                                          color=color, alpha=0.2)
        
        # Customize appearance
        label = metric_labels.get(metric, metric.replace('_', ' ').title())
        ax.set_xlabel('Generation', fontsize=fontsize)
        ax.set_ylabel(label, fontsize=fontsize)
        
        # Add CI/std info to title
        interval_info = f'{int(confidence_level*100)}% CI' if use_ci else '±1 Std'
        ax.set_title(f'{label} - All Tasks Comparison ({interval_info})', 
                    fontsize=fontsize+2, fontweight='bold')
        
        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
        ax.grid(True, alpha=1.0)
        # ax.legend(fontsize=fontsize-2, loc='best')
        
        plt.tight_layout()
        
        # Save combined plot
        safe_filename = metric.replace('_', '-')
        output_path = os.path.join(output_dir, f'{safe_filename}_all_tasks_combined{filename_suffix}.png')
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Combined plot saved to: {output_path}")
        
        figures.append(fig)
        plt.show()
    
    return figures

def plot_metric_summary(csv_path, fontsize=12, tick_fontsize=10, output_dir=None, 
                       use_ci=False, confidence_level=0.95):
    """
    Create a summary plot showing key metrics evolution over generations.
    
    Parameters:
    -----------
    csv_path : str
        Path to the learning_data.csv file
    fontsize : int
        Font size for labels
    tick_fontsize : int
        Font size for tick labels
    output_dir : str, optional
        Directory to save plots
    use_ci : bool
        If True, use confidence intervals instead of standard deviation
    confidence_level : float
        Confidence level for confidence intervals
    """
    
    df = pd.read_csv(csv_path)
    
    # Calculate learning speed
    df['learning_speed'] = calculate_learning_speed(df)
    
    generations = sorted(df['generation'].unique())
    
    # Select key metrics for summary plot (including learning speed)
    key_metrics = ['max_reward', 'volatility', 'burnin_speed', 'convergence_speed', 'learning_speed']
    key_labels = ['Max Reward', 'Volatility', 'Burnin Speed', 'Convergence Speed', 'Learning Speed']
    
    # Calculate stats
    generation_stats = []
    for gen in generations:
        gen_data = df[df['generation'] == gen]
        gen_stats = {'generation': gen}
        
        for metric in key_metrics:
            if use_ci:
                mean, lower, upper, sem = calculate_confidence_interval(
                    gen_data[metric].values, confidence_level)
                gen_stats[f'{metric}_mean'] = mean
                gen_stats[f'{metric}_lower'] = lower
                gen_stats[f'{metric}_upper'] = upper
                gen_stats[f'{metric}_sem'] = sem
            else:
                gen_stats[f'{metric}_mean'] = np.nanmean(gen_data[metric])
                gen_stats[f'{metric}_std'] = np.nanstd(gen_data[metric])
        
        generation_stats.append(gen_stats)
    
    stats_df = pd.DataFrame(generation_stats)
    
    # Create separate figures for each metric
    figures = []
    
    if output_dir is None:
        output_dir = os.path.dirname(csv_path)
    
    # Determine legend labels and filename suffix
    if use_ci:
        interval_label = f'{int(confidence_level*100)}% CI'
        filename_suffix = f'_{int(confidence_level*100)}ci'
    else:
        interval_label = '±1 Std'
        filename_suffix = '_std'
    
    for i, (metric, label) in enumerate(zip(key_metrics, key_labels)):
        # Create a new figure for each metric
        fig, ax = plt.subplots(figsize=(8, 6))
        
        mean_col = f'{metric}_mean'
        
        if use_ci:
            lower_col = f'{metric}_lower'
            upper_col = f'{metric}_upper'
            
            valid_mask = ~(np.isnan(stats_df[mean_col]) | 
                          np.isnan(stats_df[lower_col]) | 
                          np.isnan(stats_df[upper_col]))
            plot_generations = stats_df.loc[valid_mask, 'generation']
            plot_means = stats_df.loc[valid_mask, mean_col]
            plot_lower = stats_df.loc[valid_mask, lower_col]
            plot_upper = stats_df.loc[valid_mask, upper_col]
            
            if len(plot_generations) > 0:
                ax.plot(plot_generations, plot_means, color='red', linewidth=2, label='Mean')
                ax.fill_between(plot_generations, plot_lower, plot_upper,
                              color='red', alpha=0.3, label=interval_label)
        else:
            std_col = f'{metric}_std'
            
            valid_mask = ~(np.isnan(stats_df[mean_col]) | np.isnan(stats_df[std_col]))
            plot_generations = stats_df.loc[valid_mask, 'generation']
            plot_means = stats_df.loc[valid_mask, mean_col]
            plot_stds = stats_df.loc[valid_mask, std_col]
            
            if len(plot_generations) > 0:
                ax.plot(plot_generations, plot_means, color='red', linewidth=2, label='Mean')
                ax.fill_between(plot_generations, 
                              plot_means - plot_stds, 
                              plot_means + plot_stds, 
                              color='red', alpha=0.3, label=interval_label)
        
        ax.set_xlabel('Generation', fontsize=fontsize)
        ax.set_ylabel(label, fontsize=fontsize)
        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
        ax.grid(True, alpha=1.0)
        # ax.legend(fontsize=fontsize-2)
        
        plt.tight_layout()
        
        # Save individual plot
        safe_filename = metric.replace('_', '-')
        output_path = os.path.join(output_dir, f'{safe_filename}_summary{filename_suffix}.png')
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Summary plot saved to: {output_path}")
        
        figures.append(fig)
        plt.show()
    
    return figures

def plot_learning_speed_only(csv_path, fontsize=12, tick_fontsize=10, output_dir=None, 
                            use_ci=False, confidence_level=0.95):
    """
    Create a standalone plot for learning speed metric only.
    
    Parameters:
    -----------
    csv_path : str
        Path to the learning_data.csv file
    fontsize : int
        Font size for labels
    tick_fontsize : int
        Font size for tick labels
    output_dir : str, optional
        Directory to save plots
    use_ci : bool
        If True, use confidence intervals instead of standard deviation
    confidence_level : float
        Confidence level for confidence intervals
    """
    
    df = pd.read_csv(csv_path)
    
    # Calculate learning speed
    df['learning_speed'] = calculate_learning_speed(df)
    
    generations = sorted(df['generation'].unique())
    
    # Calculate stats for learning speed
    generation_stats = []
    for gen in generations:
        gen_data = df[df['generation'] == gen]
        
        if use_ci:
            mean, lower, upper, sem = calculate_confidence_interval(
                gen_data['learning_speed'].values, confidence_level)
            gen_stats = {
                'generation': gen,
                'learning_speed_mean': mean,
                'learning_speed_lower': lower,
                'learning_speed_upper': upper,
                'learning_speed_sem': sem
            }
        else:
            gen_stats = {
                'generation': gen,
                'learning_speed_mean': np.nanmean(gen_data['learning_speed']),
                'learning_speed_std': np.nanstd(gen_data['learning_speed'])
            }
        
        generation_stats.append(gen_stats)
    
    stats_df = pd.DataFrame(generation_stats)
    
    if output_dir is None:
        output_dir = os.path.dirname(csv_path)
    
    # Create figure for learning speed
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Determine legend labels and filename suffix
    if use_ci:
        interval_label = f'{int(confidence_level*100)}% CI'
        filename_suffix = f'_{int(confidence_level*100)}ci'
    else:
        interval_label = '±1 Std'
        filename_suffix = '_std'
    
    if use_ci:
        valid_mask = ~(np.isnan(stats_df['learning_speed_mean']) | 
                      np.isnan(stats_df['learning_speed_lower']) | 
                      np.isnan(stats_df['learning_speed_upper']))
        plot_generations = stats_df.loc[valid_mask, 'generation']
        plot_means = stats_df.loc[valid_mask, 'learning_speed_mean']
        plot_lower = stats_df.loc[valid_mask, 'learning_speed_lower']
        plot_upper = stats_df.loc[valid_mask, 'learning_speed_upper']
        
        if len(plot_generations) > 0:
            ax.plot(plot_generations, plot_means, color='red', linewidth=2, label='Mean')
            ax.fill_between(plot_generations, plot_lower, plot_upper,
                          color='red', alpha=0.3, label=interval_label)
    else:
        valid_mask = ~(np.isnan(stats_df['learning_speed_mean']) | 
                      np.isnan(stats_df['learning_speed_std']))
        plot_generations = stats_df.loc[valid_mask, 'generation']
        plot_means = stats_df.loc[valid_mask, 'learning_speed_mean']
        plot_stds = stats_df.loc[valid_mask, 'learning_speed_std']
        
        if len(plot_generations) > 0:
            ax.plot(plot_generations, plot_means, color='red', linewidth=2, label='Mean')
            ax.fill_between(plot_generations, 
                          plot_means - plot_stds, 
                          plot_means + plot_stds, 
                          color='red', alpha=0.3, label=interval_label)
    
    ax.set_xlabel('Generation', fontsize=fontsize)
    ax.set_ylabel('Learning Speed', fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    ax.grid(True, alpha=1.0)
    # ax.legend(fontsize=fontsize-2)
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, f'learning-speed_over_generations{filename_suffix}.png')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Learning speed plot saved to: {output_path}")
    
    plt.show()
    
    return fig

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot learning metrics from saved CSV data.")
    
    # Create mutually exclusive group for csv_path vs collection_dir
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--csv_path', type=str,
                           help='Path to a single learning_data.csv file')
    input_group.add_argument('--collection_dir', type=str,
                           help='Path to directory containing subdirectories with learning_data.csv files')
    
    parser.add_argument('--output_base_dir', type=str, default=None,
                       help='Base output directory for plots (for collection_dir mode)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for plots (for single csv_path mode, default: same as CSV)')
    parser.add_argument('--fontsize', type=int, default=17,
                       help='Font size for labels')
    parser.add_argument('--tick_fontsize', type=int, default=17,
                       help='Font size for tick labels')
    parser.add_argument('--width', type=int, default=15,
                       help='Figure width')
    parser.add_argument('--height', type=int, default=12,
                       help='Figure height')
    parser.add_argument('--summary_only', action='store_true',
                       help='Create only the summary plot with key metrics')
    parser.add_argument('--learning_speed_only', action='store_true',
                       help='Create only the learning speed plot')
    
    # New confidence interval arguments
    parser.add_argument('--use_ci', action='store_true',
                       help='Use confidence intervals instead of standard deviation')
    parser.add_argument('--confidence_level', type=float, default=0.95,
                       help='Confidence level for confidence intervals (default: 0.95)')
    
    args = parser.parse_args()
    
    # Validate confidence level
    if not 0.5 <= args.confidence_level <= 0.999:
        print(f"Error: Confidence level must be between 0.5 and 0.999, got {args.confidence_level}")
        exit(1)
    
    figsize = (args.width, args.height)
    
    if args.csv_path:
        # Single CSV file mode
        if not os.path.exists(args.csv_path):
            print(f"Error: CSV file {args.csv_path} does not exist")
            exit(1)
        
        if args.learning_speed_only:
            # Create only the learning speed plot
            plot_learning_speed_only(args.csv_path, args.fontsize, args.tick_fontsize, 
                                    args.output_dir, args.use_ci, args.confidence_level)
        elif args.summary_only:
            # Create only the summary plot
            plot_metric_summary(args.csv_path, args.fontsize, args.tick_fontsize, 
                              args.output_dir, args.use_ci, args.confidence_level)
        else:
            # Create full detailed plots
            plot_learning_metrics(args.csv_path, args.fontsize, args.tick_fontsize, figsize, 
                                args.output_dir, args.use_ci, args.confidence_level)
            
            # Also create summary plot
            # plot_metric_summary(args.csv_path, args.fontsize, args.tick_fontsize, 
            #                   args.output_dir, args.use_ci, args.confidence_level)
    
    elif args.collection_dir:
        # Collection directory mode
        if not os.path.exists(args.collection_dir):
            print(f"Error: Collection directory {args.collection_dir} does not exist")
            exit(1)
        
        if not os.path.isdir(args.collection_dir):
            print(f"Error: {args.collection_dir} is not a directory")
            exit(1)
        
        # Process the entire collection
        process_directory_collection(
            base_dir=args.collection_dir,
            fontsize=args.fontsize,
            tick_fontsize=args.tick_fontsize,
            figsize=figsize,
            output_base_dir=args.output_base_dir,
            summary_only=args.summary_only,
            use_ci=args.use_ci,
            confidence_level=args.confidence_level
        )

# Example usage:
# 
# Single CSV file with 95% confidence intervals:
# python plot_learning_metrics.py --csv_path /path/to/learning_data.csv --use_ci --confidence_level 0.95
#
# Single CSV file with 99% confidence intervals:
# python plot_learning_metrics.py --csv_path /path/to/learning_data.csv --use_ci --confidence_level 0.99
#
# Collection of directories with confidence intervals:
# python plot_learning_metrics.py --collection_dir /path/to/experiments/ --use_ci --confidence_level 0.95
#
# Summary plots only with confidence intervals:
# python plot_learning_metrics.py --csv_path /path/to/learning_data.csv --summary_only --use_ci
#
# Learning speed plot only with confidence intervals:
# python plot_learning_metrics.py --csv_path /path/to/learning_data.csv --learning_speed_only --use_ci --confidence_level 0.90
#
# Standard deviation plots (original behavior):
# python plot_learning_metrics.py --csv_path /path/to/learning_data.csv
#
# Mixed usage - some experiments might benefit from CI, others from std:
# python plot_learning_metrics.py --collection_dir /path/to/experiments/ --use_ci --confidence_level 0.95 --summary_only