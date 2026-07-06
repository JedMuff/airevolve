import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add the project root to path so we can import airevolve
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from airevolve.evolution_tools.genome_handlers.spherical_angular_genome_handler import SphericalAngularDroneGenomeHandler
from airevolve.evolution_tools.inspection_tools.drone_visualizer import DroneVisualizer

def main():
    file_path = "results_training/shuttlerun/shuttlerun_individual_1291/genome.npy"
    output_path = "results_training/shuttlerun/shuttlerun_individual_1291/morphology_4views.png"

    print(f"Loading genome from {file_path}...")
    
    # Load genome
    try:
        genome = np.load(file_path)
        genome = genome.astype(float)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        sys.exit(1)

    # Initialize the genome handler
    handler = SphericalAngularDroneGenomeHandler(
        genome=genome,
        min_max_narms=(3, 8),
        bilateral_plane_for_symmetry=None
    )

    arm_count = handler.get_arm_count()
    print(f"Loaded drone configuration with {arm_count} arms.")

    # Create visualizer
    visualizer = DroneVisualizer()

    # Plot the multi-view blueprint
    print("Generating 4-view blueprint...")
    fig_bp, axes = visualizer.plot_blueprint(
        handler.genome.arms,
        title=f"Drone Blueprint - {arm_count} Arms"
    )

    print(f"Saving to {output_path}...")
    fig_bp.savefig(output_path, dpi=300, bbox_inches='tight')
    print("Done!")

if __name__ == "__main__":
    main()
