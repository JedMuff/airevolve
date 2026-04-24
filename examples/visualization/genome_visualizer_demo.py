"""
Comprehensive demo of the genome visualization system.

This script demonstrates the new DroneVisualizer class and compares it with
the original visualization functions. Shows various genome formats, styling
options, and use cases.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any

# Add the project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from airevolve.evolution_tools.inspection_tools.drone_visualizer import DroneVisualizer, VisualizationConfig
from examples.sample_genomes import get_all_sample_genomes
import airevolve.evolution_tools.inspection_tools.utils as u

def demo_basic_visualization():
    """Demo basic 2D and 3D visualization with the new interface."""
    print("=== Basic Visualization Demo ===")
    
    # Get sample genomes
    samples = get_all_sample_genomes()
    
    # Create visualizer
    visualizer = DroneVisualizer()
    
    # Demo 1: Simple 3D plot
    print("Creating 3D visualization of quadcopter...")
    fig, ax = visualizer.plot_3d(
        samples['quadcopter_cartesian'], 
        title="Quadcopter - 3D View",
        fitness=0.95,
        generation=100
    )
    plt.savefig('demo_3d_quadcopter.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Demo 2: Simple 2D plot
    print("Creating 2D visualization of hexacopter...")
    fig, ax = visualizer.plot_2d(
        samples['hexacopter_cartesian'], 
        title="Hexacopter - Top View",
        fitness=0.87,
        generation=200
    )
    plt.savefig('demo_2d_hexacopter.png', dpi=150, bbox_inches='tight')
    plt.show()


def demo_coordinate_systems():
    """Demo visualization of different coordinate systems."""
    print("\n=== Coordinate Systems Demo ===")
    
    samples = get_all_sample_genomes()
    visualizer = DroneVisualizer()
    
    # Compare Cartesian vs Polar representations
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Cartesian quadcopter
    visualizer.plot_2d(
        samples['quadcopter_cartesian'], 
        ax=axes[0],
        title="Cartesian Coordinates"
    )
    
    # Polar quadcopter  
    visualizer.plot_2d(
        samples['quadcopter_polar'], 
        ax=axes[1],
        title="Polar Coordinates"
    )
    
    plt.tight_layout()
    plt.savefig('demo_coordinate_systems.png', dpi=150, bbox_inches='tight')
    plt.show()


def demo_blueprint_views():
    """Demo the blueprint visualization with multiple views."""
    print("\n=== Blueprint Views Demo ===")
    
    samples = get_all_sample_genomes()
    visualizer = DroneVisualizer()
    
    # Create blueprint of tilted motors configuration
    print("Creating blueprint of tilted motors drone...")
    fig, axes = visualizer.plot_blueprint(
        samples['tilted_motors_cartesian'],
        title="Tilted Motors Drone - Blueprint Views"
    )
    plt.savefig('demo_blueprint.png', dpi=150, bbox_inches='tight')
    plt.show()


def demo_styling_options():
    """Demo different styling and configuration options."""
    print("\n=== Styling Options Demo ===")
    
    samples = get_all_sample_genomes()
    
    # Create custom configurations
    configs = {
        'default': VisualizationConfig(),
        'large_motors': VisualizationConfig(
            circle_radius=0.15,
            motor_color='red',
            arm_color='blue'
        ),
        'minimal': VisualizationConfig(
            show_axis=False,
            show_axis_ticks=False,
            axis_labels=False,
            include_motor_orientation=False
        ),
        'detailed': VisualizationConfig(
            include_motor_orientation=True,
            include_motor_orientation_2d=2,  # arrows + labels
            show_limits=True,
            fontsize=12
        )
    }
    
    # Create subplot for each style
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, (name, config) in enumerate(configs.items()):
        visualizer = DroneVisualizer(config)
        visualizer.plot_2d(
            samples['evolved_example_polar'],
            ax=axes[i],
            title=f"Style: {name.title()}"
        )
    
    plt.tight_layout()
    plt.savefig('demo_styling.png', dpi=150, bbox_inches='tight')
    plt.show()


def demo_complex_geometries():
    """Demo visualization of complex, evolved geometries."""
    print("\n=== Complex Geometries Demo ===")
    
    samples = get_all_sample_genomes()
    visualizer = DroneVisualizer()
    
    # Show various complex configurations
    geometries = [
        ('3d_configuration_cartesian', '3D Configuration'),
        ('asymmetric_polar', 'Asymmetric Design'),
        ('evolved_example_polar', 'Evolved Example'),
        ('tilted_motors_cartesian', 'Tilted Motors')
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), subplot_kw={'projection': '3d'})
    axes = axes.flatten()
    
    for i, (genome_key, title) in enumerate(geometries):
        visualizer.plot_3d(
            samples[genome_key],
            ax=axes[i],
            title=title,
            elevation=45,
            azimuth=45
        )
    
    plt.tight_layout()
    plt.savefig('demo_complex_geometries.png', dpi=150, bbox_inches='tight')
    plt.show()

def demo_evolution_progression():
    """Demo visualization of evolution progression."""
    print("\n=== Evolution Progression Demo ===")
    
    # Create a mock evolution progression
    target = get_all_sample_genomes()['target_individual']
    
    # Simulate progression toward target
    generations = [0, 50, 100, 150, 200]
    fitnesses = [0.1, 0.3, 0.6, 0.8, 0.95]
    
    # Create noise that decreases over time
    np.random.seed(42)
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    visualizer = DroneVisualizer()
    
    for i, (gen, fitness) in enumerate(zip(generations, fitnesses)):
        # Add decreasing noise to target
        noise_scale = 0.3 * (1 - fitness)
        noise = np.random.normal(0, noise_scale, target.shape)
        noisy_genome = target + noise
        
        visualizer.plot_2d(
            noisy_genome,
            ax=axes[i],
            title=f"Gen {gen}",
            fitness=fitness,
            xlim=[-0.8, 0.8],
            ylim=[-0.8, 0.8]
        )
    
    fig.suptitle("Evolution Progression Toward Target", fontsize=16)
    plt.tight_layout()
    plt.savefig('demo_evolution_progression.png', dpi=150, bbox_inches='tight')
    plt.show()


def demo_interactive_features():
    """Demo interactive and advanced features."""
    print("\n=== Interactive Features Demo ===")
    
    samples = get_all_sample_genomes()
    visualizer = DroneVisualizer()
    
    # Demo: Multiple views of same genome
    genome = samples['evolved_example_polar']
    
    fig = plt.figure(figsize=(15, 10))
    
    # 3D view
    ax1 = plt.subplot(2, 3, 1, projection='3d')
    visualizer.plot_3d(genome, ax=ax1, title="3D View", elevation=30, azimuth=45)
    
    # Top view  
    ax2 = plt.subplot(2, 3, 2)
    visualizer.plot_2d(genome, ax=ax2, title="Top View")
    
    # Different 3D angle
    ax3 = plt.subplot(2, 3, 3, projection='3d')
    visualizer.plot_3d(genome, ax=ax3, title="Side View", elevation=0, azimuth=0)
    
    # With motor orientations
    ax4 = plt.subplot(2, 3, 4)
    config = VisualizationConfig(include_motor_orientation_2d=1)
    visualizer_detailed = DroneVisualizer(config)
    visualizer_detailed.plot_2d(genome, ax=ax4, title="With Motor Orientations")
    
    # Minimalist view
    ax5 = plt.subplot(2, 3, 5)
    config_minimal = VisualizationConfig(
        show_axis=False, 
        show_limits=False,
        include_motor_orientation=False
    )
    visualizer_minimal = DroneVisualizer(config_minimal)
    visualizer_minimal.plot_2d(genome, ax=ax5, title="Minimalist")
    
    # Blueprint thumbnail
    ax6 = plt.subplot(2, 3, 6, projection='3d')
    visualizer.plot_3d(genome, ax=ax6, title="Isometric", elevation=30, azimuth=45)
    
    plt.tight_layout()
    plt.savefig('demo_interactive_features.png', dpi=150, bbox_inches='tight')
    plt.show()


def demo_coordinate_conversion():
    """Demo the coordinate conversion utilities."""
    print("\n=== Coordinate Conversion Demo ===")
    
    # Test coordinate conversions
    print("Testing coordinate conversions...")
    
    # Test spherical to Cartesian
    mag, azi, pit = 1.0, np.pi/4, np.pi/6
    x, y, z = u.convert_to_cartesian(mag, azi, pit)
    print(f"Spherical ({mag:.2f}, {azi:.2f}, {pit:.2f}) -> Cartesian ({x:.2f}, {y:.2f}, {z:.2f})")
    
    # Test Cartesian to spherical
    mag2, azi2, pit2 = u.convert_to_spherical(x, y, z)
    print(f"Cartesian ({x:.2f}, {y:.2f}, {z:.2f}) -> Spherical ({mag2:.2f}, {azi2:.2f}, {pit2:.2f})")
    
    # Test auto-extraction
    samples = get_all_sample_genomes()
    
    print("\nTesting auto data extraction...")
    for name, genome in samples.items():
        try:
            data = u.auto_extract_genome_data(genome)
            print(f"{name}: {data['coordinate_system']} system, {len(data['positions'])} motors")
        except Exception as e:
            print(f"{name}: Error - {e}")


def run_all_demos():
    """Run all demonstration functions."""
    print("Running Genome Visualizer Comprehensive Demo")
    print("===========================================")
    
    demos = [
        demo_basic_visualization,
        demo_coordinate_systems, 
        demo_blueprint_views,
        demo_styling_options,
        demo_complex_geometries,
        demo_evolution_progression,
        demo_interactive_features,
        demo_coordinate_conversion
    ]
    
    for demo_func in demos:
        try:
            demo_func()
        except Exception as e:
            print(f"Error in {demo_func.__name__}: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "="*50 + "\n")
    
    print("Demo completed! Check the generated PNG files for visual results.")


if __name__ == "__main__":
    # Check if we should run all demos or just specific ones
    if len(sys.argv) > 1:
        demo_name = sys.argv[1]
        if demo_name == "basic":
            demo_basic_visualization()
        elif demo_name == "coordinates":
            demo_coordinate_systems()
        elif demo_name == "blueprint":
            demo_blueprint_views()
        elif demo_name == "styling":
            demo_styling_options()
        elif demo_name == "complex":
            demo_complex_geometries()
        elif demo_name == "evolution":
            demo_evolution_progression()
        elif demo_name == "interactive":
            demo_interactive_features()
        elif demo_name == "conversion":
            demo_coordinate_conversion()
        else:
            print(f"Unknown demo: {demo_name}")
            print("Available demos: basic, coordinates, blueprint, styling, complex, evolution, interactive, conversion")
    else:
        # Run all demos
        run_all_demos()