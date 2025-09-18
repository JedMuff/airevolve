import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from math import pi

# ==================== FONT SIZE SETTINGS ====================
# Customize these font sizes as needed
FONT_SIZES = {
    'title': 20,           # Plot titles
    'axis_labels': 30,     # X and Y axis labels
    'x_tick_labels': 27,   # X-axis tick labels
    'y_tick_labels': 27,   # Y-axis tick labels  
    'legend': 14,          # Legend text
    'grid': 12,            # Grid labels
    'radar_ticks': 12      # Radar plot radial tick labels
}

# Apply global font size settings
plt.rcParams.update({
    'font.size': FONT_SIZES['x_tick_labels'],  # Default font size
    'axes.titlesize': FONT_SIZES['title'],
    'axes.labelsize': FONT_SIZES['axis_labels'],
    'xtick.labelsize': FONT_SIZES['x_tick_labels'],
    'ytick.labelsize': FONT_SIZES['y_tick_labels'],
    'legend.fontsize': FONT_SIZES['legend']
})

# Load the data
df = pd.read_csv('raw_data.csv')

# Design name mapping - customize these names as needed
design_name_mapping = {
    'circle_design_rank_01': 'Circle Design',
    'crosshexcopter': 'Standard Hexcopter', 
    'design_rank_01': 'Figure8 Design',
    'shuttlerun_design_rank_01': 'Shuttlerun Design',
    'slalom_design_rank_02': 'Slalom Design',
    'spiderhex': 'Spider Hex'
}

task_name_mapping = {
    'backandforth': 'Shuttlerun',
    'circle': 'Circle',
    'figure8': 'Figure 8',
    'slalom': 'Slalom'
}

# CONSISTENT COLOR MAPPING - Define colors for each design
def get_design_colors(designs):
    """Create consistent color mapping for designs."""
    # Define ALL possible designs that could appear in either script
    # This ensures consistent colors even if some designs are filtered out
    all_possible_designs = [
        'Circle Design',
        'Figure8 Design', 
        'Shuttlerun Design',
        'Slalom Design',
        'Spider Hex',
        'Standard Hexcopter'
    ]
    
    # Create color mapping for all possible designs
    colors = [
        '#E69F00',  # Orange
        '#56B4E9',  # Sky Blue
        '#009E73',  # Bluish Green
        '#F0E442',  # Yellow
        '#0072B2',  # Blue
        '#D55E00',  # Vermillion
    ]
    color_dict = {design: colors[i] for i, design in enumerate(all_possible_designs)}
    
    # Return only colors for designs that actually exist in current data
    return {design: color_dict[design] for design in designs if design in color_dict}

# Apply the name mapping
df['design'] = df['design'].map(design_name_mapping).fillna(df['design'])
df['task'] = df['task'].map(task_name_mapping).fillna(df['task'])

# ==================== LOG LAP TIMES ====================
# Apply logarithmic transformation to lap times for better visualization
df['log_avg_lap_time'] = np.log10(df['avg_lap_time'].replace(0, np.nan))

# Clean the data - remove rows with missing values for the metrics we need
df_clean = df.dropna(subset=['fitness', 'max_reward', 'avg_lap_time'])

# Additional filtering: only include design-task combinations with sufficient data points
min_data_points = 3  # Minimum number of data points required for a meaningful box plot

def filter_sufficient_data(df, metric_column):
    """Filter to only include design-task combinations with enough data points"""
    sufficient_data = []
    
    for design in df['design'].unique():
        for task in df['task'].unique():
            subset = df[(df['design'] == design) & (df['task'] == task)]
            # Only include if we have enough non-null data points
            valid_data = subset.dropna(subset=[metric_column])
            if len(valid_data) >= min_data_points:
                sufficient_data.append(valid_data)
    
    if sufficient_data:
        return pd.concat(sufficient_data, ignore_index=True)
    else:
        return pd.DataFrame()  # Return empty dataframe if no sufficient data

# Set style for better looking plots
plt.style.use('default')
sns.set_palette("husl")

# Get unique tasks and designs from the cleaned data
df_fitness_clean = filter_sufficient_data(df_clean, 'fitness')
tasks = df_fitness_clean['task'].unique() if not df_fitness_clean.empty else []
designs_unordered = df_fitness_clean['design'].unique() if not df_fitness_clean.empty else []

# Ensure consistent ordering of designs across both scripts
design_order = [
    'Circle Design',
    'Figure8 Design', 
    'Shuttlerun Design',
    'Slalom Design',
    'Spider Hex',
    'Standard Hexcopter'
]

# Filter and order designs based on what's actually present in the data
designs = [design for design in design_order if design in designs_unordered]

print(f"After filtering for sufficient data:")
print(f"Tasks with sufficient data: {list(tasks)}")
print(f"Designs with sufficient data (ordered): {designs}")

# Create consistent color mapping for designs
design_colors = get_design_colors(designs)
print(f"Design color mapping: {list(design_colors.keys())}")

# Figure 1: Fitness Box Plot - All designs on each task
if not df_fitness_clean.empty:
    fig1, ax1 = plt.subplots(figsize=(18, 8))
    fitness_data = []
    task_labels = []
    task_positions = []
    all_positions = []

    pos = 1
    task_spacing = 0.1  # Larger spacing between task groups
    design_spacing = 0.6  # Small spacing between designs within a task

    for task_idx, task in enumerate(tasks):
        task_start_pos = pos
        task_positions_current = []
        
        for i, design in enumerate(designs):  # Use ordered designs list
            design_task_data = df_fitness_clean[(df_fitness_clean['task'] == task) & (df_fitness_clean['design'] == design)]
            if not design_task_data.empty:
                fitness_values = design_task_data['fitness'].values
                if len(fitness_values) >= min_data_points:  # Double-check we have enough data
                    fitness_data.append(fitness_values)
                    current_pos = pos + i * design_spacing
                    all_positions.append(current_pos)
                    task_positions_current.append(current_pos)
        
        # Calculate center position for task label
        if task_positions_current:
            task_center = (min(task_positions_current) + max(task_positions_current)) / 2
            task_labels.append(task_center)
            task_positions.append(task)
        
        # Move to next task group with larger spacing
        pos += len(designs) * design_spacing + task_spacing

    if fitness_data:  # Only create plot if we have data
        box1 = ax1.boxplot(fitness_data, positions=all_positions, widths=0.6, patch_artist=True)
        for median in box1['medians']:
            median.set(color='black', linewidth=2)
        ax1.set_ylabel('Fitness', fontsize=FONT_SIZES['axis_labels'])
        ax1.locator_params(axis='y', nbins=10)
        ax1.grid(True, alpha=0.3)

        # Set x-axis labels for tasks at center of each group
        ax1.set_xticks(task_labels)
        ax1.set_xticklabels(task_positions, rotation=0, fontsize=FONT_SIZES['x_tick_labels'])
        
        # Set y-axis tick label font size
        ax1.tick_params(axis='y', labelsize=FONT_SIZES['y_tick_labels'])

        # Add vertical lines to separate task groups
        for i in range(1, len(task_labels)):
            separator_pos = task_labels[i-1] + (task_labels[i] - task_labels[i-1]) / 2
            ax1.axvline(x=separator_pos, color='gray', linestyle='--', alpha=0.5)

        # Color boxes by design using consistent color mapping
        box_idx = 0
        for task in tasks:
            for design in designs:  # Use ordered designs list
                design_task_data = df_fitness_clean[(df_fitness_clean['task'] == task) & (df_fitness_clean['design'] == design)]
                if not design_task_data.empty and len(design_task_data) >= min_data_points and box_idx < len(box1['boxes']):
                    box1['boxes'][box_idx].set_facecolor(design_colors[design])
                    box_idx += 1

        # Create legend elements for separate legend
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=design_colors[design], label=design) 
                          for design in designs]  # Use ordered designs list

        plt.tight_layout()
        plt.savefig('test_plots/fitness_boxplot.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Fitness boxplot saved successfully")
        
        # Store legend elements for separate legend figure
        global_legend_elements = legend_elements
    else:
        print("No sufficient data for fitness boxplot")
else:
    print("No data available for fitness boxplot after cleaning")

# Figure 2: Average Lap Time Box Plot - All designs on each task  
df_laptime_clean = filter_sufficient_data(df_clean, 'avg_lap_time')

if not df_laptime_clean.empty:
    fig2, ax2 = plt.subplots(figsize=(18, 8))
    laptime_data = []
    all_positions2 = []

    pos = 1
    for task_idx, task in enumerate(tasks):
        task_positions_current = []
        
        for i, design in enumerate(designs):  # Use ordered designs list
            design_task_data = df_laptime_clean[(df_laptime_clean['task'] == task) & (df_laptime_clean['design'] == design)]
            if not design_task_data.empty and len(design_task_data) >= min_data_points:
                laptime_values = design_task_data['avg_lap_time'].values
                laptime_data.append(laptime_values)
                current_pos = pos + i * design_spacing
                all_positions2.append(current_pos)
                task_positions_current.append(current_pos)
        
        pos += len(designs) * design_spacing + task_spacing

    if laptime_data:  # Only create plot if we have data
        box2 = ax2.boxplot(laptime_data, positions=all_positions2, widths=0.6, patch_artist=True)
        for median in box2['medians']:
            median.set(color='black', linewidth=2)
        ax2.set_ylabel('Average Lap Time', fontsize=FONT_SIZES['axis_labels'])
        ax1.locator_params(axis='y', nbins=10)
        ax2.grid(True, alpha=0.3)

        # Set x-axis labels for tasks
        ax2.set_xticks(task_labels)
        ax2.set_xticklabels(task_positions, rotation=0, fontsize=FONT_SIZES['x_tick_labels'])
        
        # Set y-axis tick label font size
        ax2.tick_params(axis='y', labelsize=FONT_SIZES['y_tick_labels'])

        # Add vertical lines to separate task groups
        for i in range(1, len(task_labels)):
            separator_pos = task_labels[i-1] + (task_labels[i] - task_labels[i-1]) / 2
            ax2.axvline(x=separator_pos, color='gray', linestyle='--', alpha=0.5)

        # Color boxes by design using consistent color mapping
        box_idx = 0
        for task in tasks:
            for design in designs:  # Use ordered designs list
                design_task_data = df_laptime_clean[(df_laptime_clean['task'] == task) & (df_laptime_clean['design'] == design)]
                if not design_task_data.empty and len(design_task_data) >= min_data_points and box_idx < len(box2['boxes']):
                    box2['boxes'][box_idx].set_facecolor(design_colors[design])
                    box_idx += 1

        plt.tight_layout()
        plt.savefig('test_plots/avg_lap_time_boxplot.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Average lap time boxplot saved successfully")
    else:
        print("No sufficient data for average lap time boxplot")
else:
    print("No data available for average lap time boxplot after cleaning")

# Figure 2b: LOGGED Average Lap Time Box Plot - All designs on each task  
df_log_laptime_clean = filter_sufficient_data(df_clean, 'log_avg_lap_time')

if not df_log_laptime_clean.empty:
    fig2b, ax2b = plt.subplots(figsize=(18, 8))
    log_laptime_data = []
    all_positions2b = []

    pos = 1
    for task_idx, task in enumerate(tasks):
        task_positions_current = []
        
        for i, design in enumerate(designs):  # Use ordered designs list
            design_task_data = df_log_laptime_clean[(df_log_laptime_clean['task'] == task) & (df_log_laptime_clean['design'] == design)]
            if not design_task_data.empty and len(design_task_data) >= min_data_points:
                log_laptime_values = design_task_data['log_avg_lap_time'].values
                log_laptime_data.append(log_laptime_values)
                current_pos = pos + i * design_spacing
                all_positions2b.append(current_pos)
                task_positions_current.append(current_pos)
        
        pos += len(designs) * design_spacing + task_spacing

    if log_laptime_data:  # Only create plot if we have data
        box2b = ax2b.boxplot(log_laptime_data, positions=all_positions2b, widths=0.6, patch_artist=True)
        ax2b.set_ylabel('Log₁₀(Average Lap Time)', fontsize=FONT_SIZES['axis_labels'])
        ax1.locator_params(axis='y', nbins=10)
        ax2b.grid(True, alpha=0.3)

        # Set x-axis labels for tasks
        ax2b.set_xticks(task_labels)
        ax2b.set_xticklabels(task_positions, rotation=0, fontsize=FONT_SIZES['x_tick_labels'])
        
        # Set y-axis tick label font size
        ax2b.tick_params(axis='y', labelsize=FONT_SIZES['y_tick_labels'])

        # Add vertical lines to separate task groups
        for i in range(1, len(task_labels)):
            separator_pos = task_labels[i-1] + (task_labels[i] - task_labels[i-1]) / 2
            ax2b.axvline(x=separator_pos, color='gray', linestyle='--', alpha=0.5)

        # Color boxes by design using consistent color mapping
        box_idx = 0
        for task in tasks:
            for design in designs:  # Use ordered designs list
                design_task_data = df_log_laptime_clean[(df_log_laptime_clean['task'] == task) & (df_log_laptime_clean['design'] == design)]
                if not design_task_data.empty and len(design_task_data) >= min_data_points and box_idx < len(box2b['boxes']):
                    box2b['boxes'][box_idx].set_facecolor(design_colors[design])
                    box_idx += 1

        plt.tight_layout()
        plt.savefig('test_plots/log_avg_lap_time_boxplot.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Logged average lap time boxplot saved successfully")
    else:
        print("No sufficient data for logged average lap time boxplot")
else:
    print("No data available for logged average lap time boxplot after cleaning")

# Figure 3: Max Reward Box Plot - All designs on each task
df_reward_clean = filter_sufficient_data(df_clean, 'max_reward')

if not df_reward_clean.empty:
    fig3, ax3 = plt.subplots(figsize=(18, 8))
    reward_data = []
    all_positions3 = []

    pos = 1
    for task_idx, task in enumerate(tasks):
        task_positions_current = []
        
        for i, design in enumerate(designs):  # Use ordered designs list
            design_task_data = df_reward_clean[(df_reward_clean['task'] == task) & (df_reward_clean['design'] == design)]
            if not design_task_data.empty and len(design_task_data) >= min_data_points:
                reward_values = design_task_data['max_reward'].values
                reward_data.append(reward_values)
                current_pos = pos + i * design_spacing
                all_positions3.append(current_pos)
                task_positions_current.append(current_pos)
        
        pos += len(designs) * design_spacing + task_spacing

    if reward_data:  # Only create plot if we have data
        box3 = ax3.boxplot(reward_data, positions=all_positions3, widths=0.6, patch_artist=True)
        for median in box3['medians']:
            median.set(color='black', linewidth=2)
        ax3.set_ylabel('Max Reward', fontsize=FONT_SIZES['axis_labels'])
        ax1.locator_params(axis='y', nbins=10)
        ax3.grid(True, alpha=0.3)

        # Set x-axis labels for tasks
        ax3.set_xticks(task_labels)
        ax3.set_xticklabels(task_positions, rotation=0, fontsize=FONT_SIZES['x_tick_labels'])
        
        # Set y-axis tick label font size
        ax3.tick_params(axis='y', labelsize=FONT_SIZES['y_tick_labels'])

        # Add vertical lines to separate task groups
        for i in range(1, len(task_labels)):
            separator_pos = task_labels[i-1] + (task_labels[i] - task_labels[i-1]) / 2
            ax3.axvline(x=separator_pos, color='gray', linestyle='--', alpha=0.5)

        # Color boxes by design using consistent color mapping
        box_idx = 0
        for task in tasks:
            for design in designs:  # Use ordered designs list
                design_task_data = df_reward_clean[(df_reward_clean['task'] == task) & (df_reward_clean['design'] == design)]
                if not design_task_data.empty and len(design_task_data) >= min_data_points and box_idx < len(box3['boxes']):
                    box3['boxes'][box_idx].set_facecolor(design_colors[design])
                    box_idx += 1

        plt.tight_layout()
        plt.savefig('test_plots/max_reward_boxplot.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Max reward boxplot saved successfully")
    else:
        print("No sufficient data for max reward boxplot")
else:
    print("No data available for max reward boxplot after cleaning")

# Create separate legend figure
if 'global_legend_elements' in globals() and global_legend_elements:
    fig_legend, ax_legend = plt.subplots(figsize=(3, len(designs) * 0.5))
    ax_legend.legend(handles=global_legend_elements, loc='center', frameon=False, 
                    fontsize=FONT_SIZES['legend'])
    ax_legend.axis('off')  # Hide axes
    plt.tight_layout()
    plt.savefig('test_plots/boxplot_legend.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    print("Boxplot legend saved as separate image")
else:
    print("No legend elements available to create separate legend")

# Figure 4: Radar Plot for Fitness by Design and Task
fig4, ax4 = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

# Calculate mean fitness for each design-task combination
radar_data = df_clean.groupby(['design', 'task'])['fitness'].max().reset_index()
pivot_data = radar_data.pivot(index='design', columns='task', values='fitness')

# Fill NaN values with 0 for designs that didn't complete certain tasks
pivot_data = pivot_data.fillna(0)

# Set up the radar chart
tasks_radar = list(pivot_data.columns)
N = len(tasks_radar)

# Compute angle for each axis
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]  # Complete the circle

# Plot each design using consistent colors
for design in designs:  # Use ordered designs list
    values = pivot_data.loc[design].values.flatten().tolist()
    values += values[:1]  # Complete the circle
    
    color = design_colors.get(design, 'black')  # Use consistent color mapping
    ax4.plot(angles, values, 'o-', linewidth=2, label=design, color=color)
    ax4.fill(angles, values, alpha=0.15, color=color)

# Add labels with custom font size
ax4.set_xticks(angles[:-1])
ax4.set_xticklabels(tasks_radar, fontsize=FONT_SIZES['x_tick_labels'])

# Set radial limits and ticks

max_value = 45 #max(pivot_data.max()) #* 1.1
ax4.set_ylim(0, max_value)

# Add more radial ticks with custom font size
num_ticks = 10  # Change this number to get more/fewer ticks
radial_ticks = np.round(np.linspace(0, max_value, num_ticks),0)
ax4.set_yticks(radial_ticks)
ax4.set_yticklabels([f'{tick:.1f}' for tick in radial_ticks], 
                   fontsize=FONT_SIZES['radar_ticks'])

# Optional: Add radial grid lines for better readability
# ax4.grid(True, alpha=0.3)

# Add legend with custom font size
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), 
          fontsize=FONT_SIZES['legend'])

plt.tight_layout()
plt.savefig('test_plots/fitness_radar_plot.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 5: Radar Plot for Avr Fitness by Design and Task
fig5, ax5 = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

# Calculate mean fitness for each design-task combination
radar_data = df_clean.groupby(['design', 'task'])['fitness'].median().reset_index()
pivot_data = radar_data.pivot(index='design', columns='task', values='fitness')

# Fill NaN values with 0 for designs that didn't complete certain tasks
pivot_data = pivot_data.fillna(0)

# Set up the radar chart
tasks_radar = list(pivot_data.columns)
N = len(tasks_radar)

# Compute angle for each axis
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]  # Complete the circle

# Plot each design using consistent colors
for design in designs:  # Use ordered designs list
    values = pivot_data.loc[design].values.flatten().tolist()
    values += values[:1]  # Complete the circle
    
    color = design_colors.get(design, 'black')  # Use consistent color mapping
    ax5.plot(angles, values, 'o-', linewidth=2, label=design, color=color)
    ax5.fill(angles, values, alpha=0.15, color=color)

# Add labels with custom font size
ax5.set_xticks(angles[:-1])
ax5.set_xticklabels(tasks_radar, fontsize=FONT_SIZES['x_tick_labels'])

# Set radial limits and ticks
max_value = 45# max(pivot_data.max()) #* 1.1
ax5.set_ylim(0, max_value)

# Add more radial ticks with custom font size
num_ticks = 10  # Change this number to get more/fewer ticks
radial_ticks = np.round(np.linspace(0, max_value, num_ticks),0)
ax5.set_yticks(radial_ticks)
ax5.set_yticklabels([f'{tick:.1f}' for tick in radial_ticks], 
                   fontsize=FONT_SIZES['radar_ticks'])

# Optional: Add radial grid lines for better readability
# ax5.grid(True, alpha=0.3)

# Add legend with custom font size
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), 
          fontsize=FONT_SIZES['legend'])

plt.tight_layout()
plt.savefig('test_plots/avr_fitness_radar_plot.png', dpi=300, bbox_inches='tight')
plt.close()

print("All plots have been saved:")
print("- fitness_boxplot.png (no legend)")
print("- avg_lap_time_boxplot.png (no legend)") 
print("- log_avg_lap_time_boxplot.png (no legend, NEW)")
print("- max_reward_boxplot.png (no legend)")
print("- boxplot_legend.png (separate legend)")
print("- fitness_radar_plot.png")

# Print summary statistics
print("\nSummary Statistics:")
print(f"Number of designs: {len(designs)}")
print(f"Number of tasks: {len(tasks)}")
print(f"Total data points: {len(df_clean)}")
print(f"\nDesigns: {list(designs)}")
print(f"Tasks: {list(tasks)}")

# Print font size settings being used
print(f"\nFont Size Settings:")
for setting, size in FONT_SIZES.items():
    print(f"{setting}: {size}px")

# Save design color mapping for use in other scripts
print(f"\nDesign color mapping for consistency:")
for design, color in design_colors.items():
    print(f"{design}: {color}")

# Print lap time statistics
print(f"\nLap Time Statistics:")
print(f"Original lap times - Min: {df_clean['avg_lap_time'].min():.3f}, Max: {df_clean['avg_lap_time'].max():.3f}")
print(f"Logged lap times - Min: {df_clean['log_avg_lap_time'].min():.3f}, Max: {df_clean['log_avg_lap_time'].max():.3f}")