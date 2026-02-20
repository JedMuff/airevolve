"""
Example: Generate STL Files from Evolved Drone Genome

This script demonstrates how to use the phenotype_assembly module to generate
STL files from an evolved drone individual.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from airevolve.evolution_tools.genome_handlers.spherical_angular_genome_handler import (
    SphericalAngularDroneGenomeHandler
)
from airevolve.phenotype_assembly import generate_stl_files, AssemblyConfig
from airevolve.evolution_tools.inspection_tools.drone_visualizer import DroneVisualizer


def main():
    print("=" * 70)
    print("Generate STL Files from Evolved Drone Genome")
    print("=" * 70)

    # ========================================================================
    # OPTION 1: Generate a random individual (for demonstration)
    # ========================================================================

    print("\n[OPTION 1] Creating a random individual for demonstration...")

    # Create genome handler with desired parameters
    handler = SphericalAngularDroneGenomeHandler(
        min_max_narms=(3, 6),  # 3-6 arms
        bilateral_plane_for_symmetry=None,  # No symmetry constraint
        repair=True,  # Enable repair operations
        enable_collision_repair=False  # Disable collision repair for speed
    )

    # Generate a random individual
    population = handler.random_population(1)
    handler.genome = population[0]

    print(f"Generated random drone with {handler.get_arm_count()} arms")

    # Display arm parameters
    valid_arms = handler.get_valid_arms()
    print("\nArm parameters (first 3):")
    for i, arm in enumerate(valid_arms[:3]):
        print(f"  Arm {i+1}:")
        print(f"    magnitude: {arm[0]:.3f}")
        print(f"    arm_rotation: {np.degrees(arm[1]):.1f}°")
        print(f"    arm_pitch: {np.degrees(arm[2]):.1f}°")
        print(f"    motor_rotation: {np.degrees(arm[3]):.1f}°")
        print(f"    motor_pitch: {np.degrees(arm[4]):.1f}°")
        print(f"    direction: {int(arm[5])}")

    # ========================================================================
    # OPTION 2: Load an evolved individual from file (commented out)
    # ========================================================================

    # Uncomment and modify this section to load your evolved individual:
    """
    print("\n[OPTION 2] Loading evolved individual from file...")

    # Load genome from numpy file
    genome = np.load("path/to/your/evolved_genome.npy")

    # Create handler with the loaded genome
    handler = SphericalAngularDroneGenomeHandler(
        genome=genome,
        min_max_narms=(3, 8),
        bilateral_plane_for_symmetry=None
    )

    print(f"Loaded evolved drone with {handler.get_arm_count()} arms")
    """

    # ========================================================================
    # Generate STL Files
    # ========================================================================

    print("\n" + "=" * 70)
    print("Generating STL Files")
    print("=" * 70)

    # Generate STL files with all options
    result = generate_stl_files(
        genome_handler=handler,
        output_dir="./example_drone_stls",  # Output directory
        include_assembly=True,              # Generate full assembly STL
        include_landing_leg=False,          # Not yet implemented
        include_individual_parts=True,      # Generate individual arm STLs
        include_step_files=True,            # Generate STEP files for CAD editing
        distribute_arms_evenly=False,       # Use genome arm_rotation for placement
        magnitude_to_length_scale=100.0,    # 100 mm per unit magnitude
        assembly_config=AssemblyConfig(),   # Default physical dimensions
    )

    # ========================================================================
    # Display Results
    # ========================================================================

    print("\n" + "=" * 70)
    print("Generation Complete!")
    print("=" * 70)

    print(f"\nOutput directory: {result.output_dir}")

    if result.core_plate_file:
        print(f"\nCore plate:")
        print(f"  {result.core_plate_file}")

    if result.arm_files:
        print(f"\nIndividual arms ({len(result.arm_files)} files):")
        for arm_file in result.arm_files:
            print(f"  {arm_file.name}")

    if result.assembly_file:
        print(f"\nFull assembly:")
        print(f"  {result.assembly_file}")

    if result.step_files:
        print(f"\nSTEP files ({len(result.step_files)} files):")
        for step_file in result.step_files:
            print(f"  {step_file.name}")

    print("\n" + "=" * 70)
    print("You can now:")
    print("  1. Open the STL files in a 3D viewer (e.g., MeshLab, Blender)")
    print("  2. Open the STEP files in CAD software (e.g., FreeCAD, Fusion 360)")
    print("  3. Send the STL files to a 3D printer")
    print("=" * 70)

    # ========================================================================
    # Visualize the Drone Genome (as it will be fabricated)
    # ========================================================================

    print("\n" + "=" * 70)
    print("Visualizing Drone Genome (Fabrication Preview)")
    print("=" * 70)
    print("\nGenerating matplotlib visualization...")
    print("This shows the drone as it will be physically built from the STL files.")

    # Create a modified genome that reflects how the STL was actually built
    # The key difference: arm_rotation angles are replaced with evenly distributed angles
    valid_arms = handler.get_valid_arms()
    num_arms = len(valid_arms)

    print("\n" + "-" * 70)
    print("GENOME COMPARISON:")
    print("-" * 70)
    print("\nOriginal genome (as evolved):")
    for i, arm in enumerate(valid_arms):
        print(f"  Arm {i+1}: magnitude={arm[0]:.3f}, arm_rotation={np.degrees(arm[1]):.1f}°, "
              f"arm_pitch={np.degrees(arm[2]):.1f}°")

    print("\nFabrication genome (evenly distributed):")
    # Create visualization genome with evenly distributed arm rotations
    viz_genome = valid_arms.copy()
    for i in range(num_arms):
        # Replace arm_rotation (index 1) with evenly distributed angle
        evenly_distributed_angle = (2 * np.pi / num_arms) * i
        viz_genome[i, 1] = evenly_distributed_angle
        print(f"  Arm {i+1}: magnitude={viz_genome[i, 0]:.3f}, arm_rotation={np.degrees(viz_genome[i, 1]):.1f}°, "
              f"arm_pitch={np.degrees(viz_genome[i, 2]):.1f}°")

    print("\nSTL attachment angles (from CAD generation):")
    for i in range(num_arms):
        attachment_angle = (360.0 / num_arms) * i
        print(f"  Arm {i+1}: attachment_angle={attachment_angle:.1f}° (should match arm_rotation above)")
    print("-" * 70)

    # Create visualizer with realistic propeller size (6 inch = 152mm diameter)
    from airevolve.evolution_tools.inspection_tools.drone_visualizer import VisualizationConfig

    viz_config = VisualizationConfig(
        include_motor_orientation=True,
        motor_color='blue',
        orientation_color='red'
    )

    visualizer = DroneVisualizer(config=viz_config)

    # Create blueprint-style visualization (4 views) using the numpy array format
    fig, _ = visualizer.plot_blueprint(
        viz_genome,
        title=f"Evolved Drone - {handler.get_arm_count()} Arms - Fabrication Layout (6-inch props)"
    )

    # Save the visualization
    viz_path = result.output_dir / "drone_visualization_fabrication.png"
    fig.savefig(viz_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Fabrication visualization saved: {viz_path}")

    # Also create the original genome visualization for comparison
    print("\nGenerating original genome visualization for comparison...")
    fig2, _ = visualizer.plot_blueprint(
        handler,
        title=f"Original Genome - {handler.get_arm_count()} Arms (Simulation Scale)"
    )

    viz_path_orig = result.output_dir / "drone_visualization_original.png"
    fig2.savefig(viz_path_orig, dpi=150, bbox_inches='tight')
    print(f"✓ Original genome visualization saved: {viz_path_orig}")

    # Show the plots
    plt.show()

    print("\n" + "=" * 70)
    print("Compare the visualization with the generated STL files!")
    print("=" * 70)
    print("\nThe matplotlib visualization shows:")
    print("  - Arm positions and lengths")
    print("  - Motor orientations (red vectors)")
    print("  - Propeller rotation directions (arrows)")
    print("\nThe STL files can be opened in:")
    print("  - MeshLab, Blender (for 3D viewing)")
    print("  - FreeCAD, Fusion 360 (for CAD editing)")
    print("  - Your slicer software (for 3D printing)")
    print("=" * 70)


# ============================================================================
# Additional Examples
# ============================================================================

def example_quick_visualization():
    """Example: Quick single-file visualization."""
    from airevolve.phenotype_assembly import quick_visualize_genome

    print("\n" + "=" * 70)
    print("Quick Visualization Example")
    print("=" * 70)

    # Create a simple random individual
    handler = SphericalAngularDroneGenomeHandler(min_max_narms=(4, 4))
    population = handler.random_population(1)
    handler.genome = population[0]

    # Generate a single STL file for quick visualization
    output_file = quick_visualize_genome(handler, output_file="quick_drone.stl")

    print(f"\nQuick visualization saved to: {output_file}")


def example_custom_part_config():
    """Example: Generate STL with custom part configuration."""

    print("\n" + "=" * 70)
    print("Custom Part Configuration Example")
    print("=" * 70)

    # Create individual
    handler = SphericalAngularDroneGenomeHandler(min_max_narms=(4, 4))
    population = handler.random_population(1)
    handler.genome = population[0]

    # Custom assembly configuration
    custom_config = AssemblyConfig(
        sphere_radius=15,   # Larger sphere (default: 12)
        disc_diameter=25,   # Larger motor disc (default: 23)
        disc_thickness=4,   # Thicker disc (default: 3)
    )

    # Generate with custom config
    result = generate_stl_files(
        genome_handler=handler,
        output_dir="./custom_drone_stls",
        assembly_config=custom_config,
    )

    print(f"\nCustom drone saved to: {result.output_dir}")


def example_visualization_comparison():
    """Example: Create a detailed comparison visualization."""

    print("\n" + "=" * 70)
    print("Visualization Comparison Example")
    print("=" * 70)

    # Create a drone with 4 arms
    handler = SphericalAngularDroneGenomeHandler(min_max_narms=(4, 4))
    population = handler.random_population(1)
    handler.genome = population[0]

    print(f"\nCreated {handler.get_arm_count()}-arm drone")

    # Create visualizer
    visualizer = DroneVisualizer()

    # Create a figure with multiple views
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f"Drone Genome Analysis - {handler.get_arm_count()} Arms", fontsize=16)

    # 3D isometric view
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    visualizer.plot_3d(handler, ax=ax1, title="3D Isometric View")

    # Top view (2D)
    ax2 = fig.add_subplot(2, 3, 2)
    visualizer.plot_2d(handler, ax=ax2, title="Top-Down View (XY Plane)")

    # Front view
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    visualizer.plot_3d(handler, ax=ax3, title="Front View", elevation=0, azimuth=0)

    # Side view
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    visualizer.plot_3d(handler, ax=ax4, title="Side View", elevation=0, azimuth=90)

    # Top view (3D)
    ax5 = fig.add_subplot(2, 3, 5, projection='3d')
    visualizer.plot_3d(handler, ax=ax5, title="Top View (3D)", elevation=90, azimuth=0)

    # Another isometric angle
    ax6 = fig.add_subplot(2, 3, 6, projection='3d')
    visualizer.plot_3d(handler, ax=ax6, title="Alternate View", elevation=20, azimuth=60)

    plt.tight_layout()
    plt.savefig("drone_detailed_analysis.png", dpi=150, bbox_inches='tight')
    print("\n✓ Detailed visualization saved to: drone_detailed_analysis.png")
    plt.show()


if __name__ == "__main__":
    # Run main example
    main()

    # Uncomment to run additional examples:
    # example_quick_visualization()
    # example_custom_part_config()
    # example_visualization_comparison()
