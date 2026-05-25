import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Add the project root to path so we can import airevolve
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from airevolve.evolution_tools.genome_handlers.spherical_angular_genome_handler import SphericalAngularDroneGenomeHandler
from airevolve.evolution_tools.inspection_tools.drone_visualizer import DroneVisualizer

def main():
    parser = argparse.ArgumentParser(description="Visualize a drone genome from a .npy file.")
    parser.add_argument("file_path", type=str, help="Path to the genome .npy file")
    args = parser.parse_args()

    # Check if file exists
    if not os.path.exists(args.file_path):
        print(f"Error: File '{args.file_path}' does not exist.")
        sys.exit(1)

    print(f"Loading genome from {args.file_path}...")
    
    # Load genome
    try:
        genome = np.load(args.file_path)
        # Ensure genome is a float array (np.load may return object arrays which break np.isnan)
        genome = genome.astype(float)
    except Exception as e:
        print(f"Error loading {args.file_path}: {e}")
        sys.exit(1)

    # Initialize the genome handler (defaulting to SphericalAngularDroneGenomeHandler)
    handler = SphericalAngularDroneGenomeHandler(
        genome=genome,
        min_max_narms=(3, 8),
        bilateral_plane_for_symmetry=None
    )

    arm_count = handler.get_arm_count()
    print(f"Loaded drone configuration with {arm_count} arms.")

    # Display arm parameters
    valid_arms = handler.get_valid_arms()
    print("\nArm Parameters:")
    for i, arm in enumerate(valid_arms):
        print(f"  Arm {i+1}: magnitude={arm[0]:.3f}, arm_rotation={np.degrees(arm[1]):.1f}°, arm_pitch={np.degrees(arm[2]):.1f}°, motor_rotation={np.degrees(arm[3]):.1f}°, motor_pitch={np.degrees(arm[4]):.1f}°, direction={int(arm[5])}")
    print()

    # Create visualizer
    visualizer = DroneVisualizer()

    # Plot the multi-view blueprint
    print("Generating blueprint views...")
    fig_bp, _ = visualizer.plot_blueprint(
        handler.genome.arms,
        title=f"Drone Blueprint - {arm_count} Arms"
    )

    # Plot a standalone interactive 3D view
    print("Generating 3D interactive view...")
    fig_3d = plt.figure(figsize=(10, 8))
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    visualizer.plot_3d(handler.genome.arms, ax=ax_3d, title=f"3D Isometric View - {arm_count} Arms", elevation=30, azimuth=45)

    print("Opening plots... (Close the windows to exit)")
    plt.show()

if __name__ == "__main__":
    main()
