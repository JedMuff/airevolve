"""
Script to load best individuals data and generate blueprint visualizations.
Reads CSV files created by the collection script and saves blueprint figures for each design.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import time

from airevolve.evolution_tools.inspection_tools.drone_visualizer import DroneVisualizer

def plot_blueprint(individual, title=None):
    """
    Plot blueprint with 4 different views of the morphology.
    Uses DroneVisualizer class for modern visualization.
    """
    # Plot a 2 by 2 grid with 4 perspectives of the morphology
    # 1. Top view, 2. Side view, 3. Isometric view, 4. Front view
    # Define the elevation and azimuth angles for each view
    elevations = [0,  0, 90, 30]  # Front, side, top, isometric
    azimuths =   [90, 0,  0, 45]

    fig = plt.figure(figsize=(12, 12))
    fig.subplots_adjust(wspace=0, hspace=0)
    
    visualizer = DroneVisualizer()
    
    for i in range(4):
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        visualizer.plot_3d(individual, ax=ax, elevation=elevations[i], azimuth=azimuths[i], include_motor_orientation=True)

    if title is not None:
        fig.suptitle(title, fontsize=16)

    return fig, ax


def load_individual_data(csv_path):
    """
    Load individual data from CSV file and convert individual_data back to numpy arrays.
    """
    df = pd.read_csv(csv_path)
    
    # Convert individual_data string representation back to numpy arrays
    for idx, row in df.iterrows():
        try:
            # Convert string representation back to list, then to numpy array
            individual_data = eval(row['individual_data'])
            df.at[idx, 'individual_data'] = np.array(individual_data)
        except Exception as e:
            print(f"Error converting individual data at index {idx}: {e}")
            df.at[idx, 'individual_data'] = None
    
    return df


def generate_blueprints_for_experiment(experiment_dir, output_subdir="blueprints", 
                                     max_individuals=None, dpi=300):
    """
    Generate blueprint figures for all best individuals in an experiment.
    
    Parameters:
    -----------
    experiment_dir : str
        Path to the experiment directory containing best_individual_data.csv
    output_subdir : str
        Subdirectory name within experiment_dir to save figures
    max_individuals : int, optional
        Maximum number of individuals to process (None = process all)
    dpi : int
        Resolution of saved figures
    """
    csv_path = os.path.join(experiment_dir, "best_individual_data.csv")
    
    if not os.path.exists(csv_path):
        print(f"No best_individual_data.csv found in {experiment_dir}")
        return
    
    print(f"Loading data from {csv_path}")
    df = load_individual_data(csv_path)
    
    if df.empty:
        print(f"No valid data found in {csv_path}")
        return
    
    # Create output directory
    output_dir = os.path.join(experiment_dir, output_subdir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Limit number of individuals if specified
    if max_individuals is not None:
        df = df.head(max_individuals)
    
    print(f"Generating blueprints for {len(df)} individuals...")
    
    successful_saves = 0
    
    for idx, row in df.iterrows():
        try:
            individual = row['individual_data']
            if individual is None:
                print(f"Skipping individual {idx}: invalid data")
                continue
            
            # Create descriptive title
            experiment_name = os.path.basename(experiment_dir)
            title = (f"{experiment_name} - Rank {row['rank']} - "
                    f"Fitness: {row['fitness']:.4f} - "
                    f"Gen: {row['generation']}")
            
            # Generate blueprint
            fig, ax = plot_blueprint(individual, title=title)
            
            # Create filename
            filename = (f"blueprint_rank_{row['rank']:02d}_"
                       f"fitness_{row['fitness']:.4f}_"
                       f"gen_{row['generation']}.png")
            filepath = os.path.join(output_dir, filename)
            
            # Save figure
            fig.savefig(filepath, dpi=dpi, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close(fig)  # Free memory
            
            successful_saves += 1
            
            if (idx + 1) % 5 == 0:
                print(f"Processed {idx + 1}/{len(df)} individuals...")
                
        except Exception as e:
            print(f"Error processing individual {idx}: {e}")
            continue
    
    print(f"Successfully saved {successful_saves} blueprint figures to {output_dir}")


def save_designs_as_numpy_arrays(experiment_dir, output_subdir="blueprints", max_individuals=None):
    """
    Save individual designs as numpy array text that can be copy-pasted into Python.
    
    Parameters:
    -----------
    experiment_dir : str
        Path to the experiment directory containing best_individual_data.csv
    output_subdir : str
        Subdirectory name within experiment_dir to save text file
    max_individuals : int, optional
        Maximum number of individuals to save (None = save all)
    """
    csv_path = os.path.join(experiment_dir, "best_individual_data.csv")
    
    if not os.path.exists(csv_path):
        print(f"No best_individual_data.csv found in {experiment_dir}")
        return
    
    df = load_individual_data(csv_path)
    if df.empty:
        print(f"No valid data found in {csv_path}")
        return
    
    # Create output directory
    output_dir = os.path.join(experiment_dir, output_subdir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Limit number of individuals if specified
    if max_individuals is not None:
        df = df.head(max_individuals)
    
    experiment_name = os.path.basename(experiment_dir)
    output_file = os.path.join(output_dir, "designs_as_numpy_arrays.txt")
    
    with open(output_file, 'w') as f:
        f.write(f"# Best Individual Designs - {experiment_name}\n")
        f.write(f"# Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Total individuals: {len(df)}\n\n")
        f.write("import numpy as np\n\n")
        
        for idx, row in df.iterrows():
            try:
                individual = row['individual_data']
                if individual is None:
                    f.write(f"# Rank {row['rank']}: Invalid data\n\n")
                    continue
                
                # Write header comment for this individual
                f.write(f"# Rank {row['rank']} - Fitness: {row['fitness']:.6f} - Generation: {row['generation']}\n")
                
                # Create variable name
                var_name = f"design_rank_{row['rank']:02d}"
                
                # Format the numpy array with proper formatting
                array_str = np.array2string(individual, 
                                          precision=6,
                                          separator=', ',
                                          suppress_small=True,
                                          max_line_width=100)
                
                # Write the array assignment
                f.write(f"{var_name} = np.array({array_str})\n\n")
                
            except Exception as e:
                f.write(f"# Rank {row['rank']}: Error formatting data - {e}\n\n")
                continue
        
        # Add a convenience dictionary at the end
        f.write("# Convenience dictionary with all designs\n")
        f.write("all_designs = {\n")
        for idx, row in df.iterrows():
            if row['individual_data'] is not None:
                var_name = f"design_rank_{row['rank']:02d}"
                fitness = row['fitness']
                f.write(f"    {row['rank']}: {{'design': {var_name}, 'fitness': {fitness:.6f}}},\n")
        f.write("}\n\n")
        
        # Add usage examples
        f.write("# Usage examples:\n")
        f.write("# individual = design_rank_01  # Get the best design\n")
        f.write("# individual = all_designs[1]['design']  # Alternative way\n")
        f.write("# fitness = all_designs[1]['fitness']  # Get fitness value\n")
    
    print(f"Saved {len([r for r in df.iterrows() if r[1]['individual_data'] is not None])} designs to {output_file}")


def generate_summary_figure(experiment_dir, top_n=6, output_subdir="blueprints"):
    """
    Generate a summary figure showing the top N individuals in a single plot.
    """
    csv_path = os.path.join(experiment_dir, "best_individual_data.csv")
    
    if not os.path.exists(csv_path):
        print(f"No best_individual_data.csv found in {experiment_dir}")
        return
    
    df = load_individual_data(csv_path)
    if df.empty or len(df) < top_n:
        print(f"Not enough valid individuals for summary figure")
        return
    
    # Take top N individuals
    top_df = df.head(top_n)
    
    # Create a large figure with subplots
    fig = plt.figure(figsize=(20, 12))
    experiment_name = os.path.basename(experiment_dir)
    fig.suptitle(f"Top {top_n} Individuals - {experiment_name}", fontsize=20)
    
    for i, (idx, row) in enumerate(top_df.iterrows()):
        try:
            individual = row['individual_data']
            if individual is None:
                continue
            
            # Create subplot for each individual (2 rows, top_n/2 columns)
            ax = fig.add_subplot(2, top_n//2, i+1, projection='3d')
            
            # Use isometric view for summary
            visualizer = DroneVisualizer()
            visualizer.plot_3d(individual, ax=ax, elevation=30, azimuth=45, include_motor_orientation=True)
            
            # Add title for this subplot
            subplot_title = f"Rank {row['rank']} - F: {row['fitness']:.3f}"
            ax.set_title(subplot_title, fontsize=12)
            
        except Exception as e:
            print(f"Error in summary figure for individual {i}: {e}")
            continue
    
    plt.tight_layout()
    
    # Save summary figure
    output_dir = os.path.join(experiment_dir, output_subdir)
    summary_path = os.path.join(output_dir, "summary_top_individuals.png")
    fig.savefig(summary_path, dpi=300, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    plt.close(fig)
    
    print(f"Summary figure saved to {summary_path}")


def main():
    """
    Main function to generate blueprints for all experiments.
    """
    
    # Define experiment directories (same as in your collection script)
    experiment_dirs = {
        # "asym_figure8": "data_backup/asym_figure8",
        # "asym_circle": "data_backup/asym_circle/",
        # "asym_slalom": "data_backup/asym_slalom/",
        # "asym_backnforth": "data_backup/asym_backnforth/",
        "sym_figure8": "data_backup/sym_figure8/",
        "sym_circle": "data_backup/sym_circle/",
        "sym_slalom": "data_backup/sym_slalom/",
        "sym_shuttlerun": "data_backup/sym_shuttlerun/",
    }
    
    # Parameters
    max_individuals_per_experiment = 15  # Limit to avoid too many files
    generate_summary = True
    dpi = 300  # High resolution for publication quality
    
    total_start_time = time.time()
    
    # Process each experiment
    for experiment_name, experiment_dir in experiment_dirs.items():
        if not os.path.exists(experiment_dir):
            print(f"Warning: Directory {experiment_dir} does not exist")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing experiment: {experiment_name}")
        print(f"Directory: {experiment_dir}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            # Generate individual blueprints
            generate_blueprints_for_experiment(
                experiment_dir, 
                max_individuals=max_individuals_per_experiment,
                dpi=dpi
            )
            
            # Generate summary figure
            if generate_summary:
                generate_summary_figure(experiment_dir, top_n=6)
            
            # Save designs as numpy arrays text file
            save_designs_as_numpy_arrays(
                experiment_dir,
                max_individuals=max_individuals_per_experiment
            )
            
        except Exception as e:
            print(f"Error processing experiment {experiment_name}: {e}")
            continue
        
        processing_time = time.time() - start_time
        print(f"Processing time for {experiment_name}: {processing_time:.2f} seconds")
    
    total_time = time.time() - total_start_time
    print(f"\nTotal processing time: {total_time:.2f} seconds")
    print("Blueprint generation completed!")


if __name__ == "__main__":
    print("\nStarting blueprint generation...")
    
    main()