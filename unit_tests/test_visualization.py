#!/usr/bin/env python3
"""
Visualization tests for CartesianEulerDroneGenomeHandler.

This module contains tests for visual inspection capabilities including:
- 3D drone visualizations
- Uniformity distribution plots
- Population sampling visualizations
- Manual inspection tools
"""

import unittest
import numpy as np
import argparse
import sys
import os
from typing import List

# Visualization (optional)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from airevolve.evolution_tools.inspection_tools.drone_visualizer import DroneVisualizer

from airevolve.evolution_tools.genome_handlers.cartesian_euler_genome_handler import CartesianEulerDroneGenomeHandler
from airevolve.evolution_tools.genome_handlers.spherical_angular_genome_handler import SphericalAngularDroneGenomeHandler
from test_utilities import extract_positions, extract_orientations, extract_directions

# Collision repair imports
from airevolve.evolution_tools.genome_handlers.operators.particle_repair_operator import particle_repair_individual, are_there_cylinder_collisions
from airevolve.evolution_tools.genome_handlers.conversions.arm_conversions import (
    arms_to_cylinders_cartesian_euler, 
    arms_to_cylinders_polar_angular,
    cylinders_to_arms_cartesian_euler,
    cylinders_to_arms_polar_angular,
)
from airevolve.evolution_tools.inspection_tools.drone_visualizer import Cylinder, visualize_cylinders
import time
from unittest.mock import patch


class TestCartesianEulerVisualization(unittest.TestCase):
    """Test visualization capabilities for CartesianEulerDroneGenomeHandler."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.expected_narms = 4
        self.max_motor_pos = 1.0
        self.rnd = np.random.default_rng(42)  # Fixed seed for reproducibility
        self.min_max_narms = (4, 4)
        self.append_arm_chance = 0.0  # Must be 0 when min_narms == max_narms
        self.parameter_limits = np.array([
            [-1.0, 1.0],    # x position
            [-1.0, 1.0],    # y position  
            [-1.0, 1.0],    # z position
            [-np.pi, np.pi], # roll
            [-np.pi, np.pi], # pitch
            [-np.pi, np.pi], # yaw
            [0, 1]          # direction
        ])
        
    def _generate_visualization_population(self, size: int) -> List[CartesianEulerDroneGenomeHandler]:
        """Generate a population for visualization testing."""
        handler = CartesianEulerDroneGenomeHandler(
            min_max_narms=self.min_max_narms,
            parameter_limits=self.parameter_limits,
            append_arm_chance=self.append_arm_chance,
            rnd=self.rnd,
            repair=False  # No repair needed for random population
        )
        return handler.generate_random_population(size)
    
    
    def test_visual_inspection(self):
        """Generate visualizations for manual inspection of uniformity."""            
        # Skip if not in visual mode
        if not getattr(self, '_visual_mode', False):
            self.skipTest("Visual inspection only runs with --visual flag")
                    
        # Create output directory
        output_dir = "test_output"
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate smaller population for visualization
        vis_population = self._generate_visualization_population(100)
        
        # Extract data
        positions = extract_positions(vis_population)
        orientations = extract_orientations(vis_population)
        directions = extract_directions(vis_population)
        
        # 1. 3D scatter plot of positions
        fig = plt.figure(figsize=(12, 10))
        
        # 3D scatter plot
        ax1 = fig.add_subplot(221, projection='3d')
        ax1.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                   c=directions, cmap='viridis', alpha=0.6)
        ax1.set_xlabel('X Position')
        ax1.set_ylabel('Y Position')
        ax1.set_zlabel('Z Position')
        ax1.set_title('3D Motor Positions (colored by propeller direction)')
        
        # Set equal aspect ratio and bounds
        ax1.set_xlim([-self.max_motor_pos, self.max_motor_pos])
        ax1.set_ylim([-self.max_motor_pos, self.max_motor_pos])
        ax1.set_zlim([-self.max_motor_pos, self.max_motor_pos])
        
        # 2. 2D projections
        ax2 = fig.add_subplot(222)
        ax2.scatter(positions[:, 0], positions[:, 1], c=directions, cmap='viridis', alpha=0.6)
        ax2.set_xlabel('X Position')
        ax2.set_ylabel('Y Position')
        ax2.set_title('XY Projection')
        ax2.set_xlim([-self.max_motor_pos, self.max_motor_pos])
        ax2.set_ylim([-self.max_motor_pos, self.max_motor_pos])
        ax2.grid(True, alpha=0.3)
        
        # 3. Histogram of positions
        ax3 = fig.add_subplot(223)
        ax3.hist(positions[:, 0], bins=30, alpha=0.7, label='X', density=True)
        ax3.hist(positions[:, 1], bins=30, alpha=0.7, label='Y', density=True)
        ax3.hist(positions[:, 2], bins=30, alpha=0.7, label='Z', density=True)
        ax3.axhline(y=1/(2*self.max_motor_pos), color='r', linestyle='--', 
                   label=f'Expected uniform density')
        ax3.set_xlabel('Position Value')
        ax3.set_ylabel('Density')
        ax3.set_title('Position Distribution Histograms')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Histogram of orientations
        ax4 = fig.add_subplot(224)
        ax4.hist(orientations[:, 0], bins=30, alpha=0.7, label='Roll', density=True)
        ax4.hist(orientations[:, 1], bins=30, alpha=0.7, label='Pitch', density=True)
        ax4.hist(orientations[:, 2], bins=30, alpha=0.7, label='Yaw', density=True)
        ax4.axhline(y=1/(2*np.pi), color='r', linestyle='--', 
                   label=f'Expected uniform density')
        ax4.set_xlabel('Angle Value (radians)')
        ax4.set_ylabel('Density')
        ax4.set_title('Orientation Distribution Histograms')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'uniformity_analysis.png'), dpi=300, bbox_inches='tight')
        print(f"Saved uniformity analysis plot to {output_dir}/uniformity_analysis.png")
        
        # 5. Individual drone visualizations using DroneVisualizer
        visualizer = DroneVisualizer()
        
        # Select a few random drones for detailed visualization
        sample_indices = np.random.choice(len(vis_population), size=6, replace=False)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10), subplot_kw={'projection': '3d'})
        axes = axes.flatten()
        
        for i, idx in enumerate(sample_indices):
            drone = vis_population[idx]
            _, ax = visualizer.plot_3d(drone, ax=axes[i], 
                                     title=f'Drone {idx}')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'individual_drones.png'), dpi=300, bbox_inches='tight')
        print(f"Saved individual drone visualizations to {output_dir}/individual_drones.png")
        
        # Close all figures to prevent display
        plt.close('all')
        
        print(f"Visual inspection plots saved to {output_dir}/ directory")
        print("Manually inspect the plots to verify uniformity:")
        print("- 3D scatter should show even distribution in cube")
        print("- 2D projections should show even coverage")
        print("- Histograms should approximate uniform distributions")
        print("- Individual drones should show diverse configurations")

    def test_drone_visualizer_integration(self):
        """Test integration with DroneVisualizer class."""

        # Create a test drone
        handler = CartesianEulerDroneGenomeHandler(
            min_max_narms=self.min_max_narms,
            parameter_limits=self.parameter_limits,
            append_arm_chance=self.append_arm_chance,
            rnd=self.rnd
        )
        
        # Test that DroneVisualizer can handle the genome handler
        visualizer = DroneVisualizer()
        
        try:
            fig, ax = visualizer.plot_3d(handler, title="Test Drone")
            self.assertIsNotNone(fig)
            self.assertIsNotNone(ax)
            plt.close(fig)
        except Exception as e:
            self.fail(f"DroneVisualizer failed to plot genome handler: {e}")

    def test_population_diversity_visualization(self):
        """Test visualization of population diversity."""

        if not getattr(self, '_visual_mode', False):
            self.skipTest("Population diversity visualization only runs with --visual flag")
        
        # Create output directory
        output_dir = "test_output"
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate populations with different configurations
        populations = {}
        
        # Regular population
        populations['regular'] = self._generate_visualization_population(50)
        
        # Symmetric population
        handler_sym = CartesianEulerDroneGenomeHandler(
            min_max_narms=self.min_max_narms,
            parameter_limits=self.parameter_limits,
            append_arm_chance=self.append_arm_chance,
            bilateral_plane_for_symmetry="yz",
            rnd=self.rnd,
            repair=False  # No repair needed for random population
        )
        populations['symmetric'] = handler_sym.generate_random_population(50)
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        for i, (pop_name, population) in enumerate(populations.items()):
            positions = extract_positions(population)
            
            row = i // 2
            col = i % 2
            
            # 3D scatter plot
            if len(axes.shape) == 2:
                ax = axes[row, col]
            else:
                ax = axes[i]
            
            scatter = ax.scatter(positions[:, 0], positions[:, 1], c=positions[:, 2], 
                               cmap='viridis', alpha=0.6)
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
            ax.set_title(f'{pop_name.title()} Population (Z as color)')
            ax.grid(True, alpha=0.3)
            ax.set_xlim([-self.max_motor_pos, self.max_motor_pos])
            ax.set_ylim([-self.max_motor_pos, self.max_motor_pos])
            
            plt.colorbar(scatter, ax=ax, shrink=0.6)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'population_diversity.png'), dpi=300, bbox_inches='tight')
        print(f"Saved population diversity plots to {output_dir}/population_diversity.png")
        plt.close(fig)

    def test_symmetry_visualization(self):
        """Test visualization of symmetric genome features."""

        if not getattr(self, '_visual_mode', False):
            self.skipTest("Symmetry visualization only runs with --visual flag")
        
        # Create output directory
        output_dir = "test_output"
        os.makedirs(output_dir, exist_ok=True)
        
        # Create symmetric handler
        handler = CartesianEulerDroneGenomeHandler(
            min_max_narms=self.min_max_narms,
            parameter_limits=self.parameter_limits,
            append_arm_chance=self.append_arm_chance,
            bilateral_plane_for_symmetry="yz",
            rnd=self.rnd
        )
        
        # Generate a few symmetric drones
        symmetric_drones = handler.generate_random_population(4)
        
        # Visualize symmetric pairs
        visualizer = DroneVisualizer()
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), subplot_kw={'projection': '3d'})
        axes = axes.flatten()
        
        for i, drone in enumerate(symmetric_drones):
            _, ax = visualizer.plot_3d(drone, ax=axes[i], 
                                     title=f'Symmetric Drone {i+1}')
            
            # Highlight symmetry pairs
            pairs = drone.get_symmetry_pairs()
            positions = drone.get_motor_positions(include_nans=True)
            
            # Draw lines connecting symmetric pairs
            for pair in pairs:
                idx1, idx2 = pair
                pos1, pos2 = positions[idx1], positions[idx2]
                ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], [pos1[2], pos2[2]], 
                       'r--', alpha=0.5, linewidth=2, label='Symmetry pair' if i == 0 and pair == pairs[0] else "")
            
            if i == 0:
                ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'symmetric_drones.png'), dpi=300, bbox_inches='tight')
        print(f"Saved symmetric drone visualizations to {output_dir}/symmetric_drones.png")
        plt.close(fig)

    def test_mutation_effect_visualization(self):
        """Test visualization of mutation effects."""

        if not getattr(self, '_visual_mode', False):
            self.skipTest("Mutation effect visualization only runs with --visual flag")
        
        # Create output directory
        output_dir = "test_output"
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a base drone
        handler = CartesianEulerDroneGenomeHandler(
            min_max_narms=self.min_max_narms,
            parameter_limits=self.parameter_limits,
            append_arm_chance=self.append_arm_chance,
            mutation_probs=np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),  # High mutation
            rnd=self.rnd
        )
        
        original = handler.copy()
        
        # Create mutated versions
        mutated_versions = []
        for i in range(6):
            mutant = original.copy()
            mutant.mutate()
            mutated_versions.append(mutant)
        
        # Visualize original and mutations
        visualizer = DroneVisualizer()
        fig, axes = plt.subplots(2, 3, figsize=(15, 10), subplot_kw={'projection': '3d'})
        axes = axes.flatten()
        
        # Plot original in first subplot
        _, ax = visualizer.plot_3d(original, ax=axes[0], title='Original Drone')
        
        # Plot mutations
        for i, mutant in enumerate(mutated_versions[:5]):
            _, ax = visualizer.plot_3d(mutant, ax=axes[i+1], 
                                     title=f'Mutation {i+1}')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'mutation_effects.png'), dpi=300, bbox_inches='tight')
        print(f"Saved mutation effect visualizations to {output_dir}/mutation_effects.png")
        plt.close(fig)

    @patch('airevolve.evolution_tools.genome_handlers.operators.particle_repair_operator.are_there_cylinder_collisions', 
           side_effect=are_there_cylinder_collisions)
    def test_collision_repair_visualization(self, mock_collision_check):
        """Test visualization of collision repair process for both coordinate systems."""
        # Skip if not in visual mode
        if not getattr(self, '_visual_mode', False):
            self.skipTest("Collision repair visualization only runs with --visual flag")
                
        # Create output directory
        output_dir = "test_output"
        os.makedirs(output_dir, exist_ok=True)
        
        # Test parameters
        propeller_radius = 0.0762
        inner_boundary_radius = 0.09
        outer_boundary_radius = 0.4
        cylinder_height = 0.3048
        
        # Generate test cases
        test_cases = []
        
        # 1. Cartesian Asymmetric
        cart_handler = CartesianEulerDroneGenomeHandler(
            min_max_narms=(8, 8),
            parameter_limits=[[-0.4, 0.4], [-0.4, 0.4], [-0.4, 0.4],
                              [-np.pi, np.pi], [-np.pi, np.pi], [-np.pi, np.pi],
                              [0, 1]],
            append_arm_chance=0.0,
            repair=False,
            rnd=self.rnd
        )
        cart_handler.genome = cart_handler._generate_random_genome()
        test_cases.append(("Cartesian Asymmetric", cart_handler.copy(), arms_to_cylinders_cartesian_euler))
        
        # 2. Cartesian Symmetric
        cart_sym_handler = CartesianEulerDroneGenomeHandler(
            min_max_narms=(8, 8),
            parameter_limits=[[-0.4, 0.4],[-0.4, 0.4],[-0.4, 0.4],
                              [-np.pi, np.pi],[-np.pi, np.pi],[-np.pi, np.pi],
                              [0, 1]],
            bilateral_plane_for_symmetry="yz",
            append_arm_chance=0.0,
            repair=False,
            rnd=self.rnd
        )
        # Create symmetric genome
        cart_sym_handler.genome = cart_sym_handler._generate_random_genome()
        # cart_sym_handler.apply_bilateral_symmetry()
        test_cases.append(("Cartesian Symmetric", cart_sym_handler.copy(), arms_to_cylinders_cartesian_euler))
        
        # 3. Spherical Asymmetric
        sph_handler = SphericalAngularDroneGenomeHandler(
            min_max_narms=(8, 8),
            parameter_limits=[[0.0, 0.4],
                              [-np.pi, np.pi],[-np.pi, np.pi],
                              [-np.pi, np.pi],[-np.pi, np.pi],
                              [0, 1]],
            append_arm_chance=0.0,
            repair=False,
            rnd=self.rnd
        )
        sph_handler.genome = sph_handler._generate_random_genome()
        test_cases.append(("Spherical Asymmetric", sph_handler.copy(), arms_to_cylinders_polar_angular))
        
        # 4. Spherical Symmetric
        sph_sym_handler = SphericalAngularDroneGenomeHandler(
            min_max_narms=(8, 8),
            parameter_limits=[[0.0, 0.4],
                              [-np.pi, np.pi],[-np.pi, np.pi],
                              [-np.pi, np.pi],[-np.pi, np.pi],
                              [0, 1]],
            bilateral_plane_for_symmetry="yz",
            append_arm_chance=0.0,
            repair=False,
            rnd=self.rnd
        )
        # Create symmetric spherical genome
        sph_sym_handler.genome = sph_sym_handler._generate_random_genome()
        test_cases.append(("Spherical Symmetric", sph_sym_handler.copy(), arms_to_cylinders_polar_angular))
        
        # Create 4x2 subplot grid (4 test cases, before/after columns)
        fig, axes = plt.subplots(4, 2, figsize=(8, 10), subplot_kw={'projection': '3d'})
                
        for i, (case_name, genome_handler, conversion_func) in enumerate(test_cases):

            # Before repair
            cylinders_before = conversion_func(genome_handler.genome, propeller_radius, cylinder_height)
            collisions_before = are_there_cylinder_collisions(cylinders_before)
            
            # Plot before repair (left column)
            ax_before = axes[i, 0]
            self._plot_cylinder_visualization(ax_before, cylinders_before, inner_boundary_radius, 
                                            outer_boundary_radius, f"{case_name}\nBEFORE (Collisions: {'YES' if collisions_before else 'NO'})")
            
            # Repair process
            start_time = time.time()
            genome_handler.repair()
            repaired_genome = genome_handler.genome.copy()

            repair_time = time.time() - start_time
            
            # After repair
            valid_arms_after = ~np.isnan(repaired_genome).any(axis=-1)
            if np.sum(valid_arms_after) >= 2:
                valid_arm_params_after = repaired_genome[valid_arms_after]
                cylinders_after = conversion_func(valid_arm_params_after, propeller_radius, cylinder_height)
                collisions_after = are_there_cylinder_collisions(cylinders_after)

                # Plot after repair (right column)
                ax_after = axes[i, 1] 
                self._plot_cylinder_visualization(ax_after, cylinders_after, inner_boundary_radius,
                                                outer_boundary_radius, f"{case_name}\nAFTER (Collisions: {'YES' if collisions_after else 'NO'})")
            else:
                print(f"  Warning: No valid arms after repair")
                axes[i, 1].text(0.5, 0.5, 0.5, "Repair Failed\n(No valid arms)", 
                                ha='center', va='center', transform=axes[i, 1].transAxes)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'collision_repair_visualization.png'), dpi=300, bbox_inches='tight')
        plt.show()
        print(f"\nSaved collision repair visualization to {output_dir}/collision_repair_visualization.png")
        plt.close(fig)
    
    def _plot_cylinder_visualization(self, ax, cylinders, inner_radius, outer_radius, title):
        """Helper method to plot cylinders with boundary constraints."""
        # Draw boundary constraints as transparent spheres
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        
        # Inner boundary (green)
        x_inner = inner_radius * np.outer(np.cos(u), np.sin(v))
        y_inner = inner_radius * np.outer(np.sin(u), np.sin(v))
        z_inner = inner_radius * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x_inner, y_inner, z_inner, color='g', alpha=0.1)
        
        # Outer boundary (red)
        x_outer = outer_radius * np.outer(np.cos(u), np.sin(v))
        y_outer = outer_radius * np.outer(np.sin(u), np.sin(v))
        z_outer = outer_radius * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x_outer, y_outer, z_outer, color='r', alpha=0.1)
        
        # Check for collisions and assign colors
        collision_pairs = []
        for i in range(len(cylinders)):
            for j in range(i + 1, len(cylinders)):
                if are_there_cylinder_collisions([cylinders[i], cylinders[j]]):
                    collision_pairs.extend([i, j])
        
        # Draw cylinders with appropriate colors
        visualize_cylinders(cylinders)
        visualizer = DroneVisualizer()
        visualizer.plot_cylinders_3d(
            cylinders, title=title, ax=ax, 
        )
        # Set equal aspect ratio and limits
        max_range = outer_radius * 1.1
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        ax.set_zlim(-max_range, max_range)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title, fontsize=10)
        ax.view_init(elev=20, azim=45)


def setup_visual_mode():
    """Set up visual mode based on command line arguments."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test CartesianEulerDroneGenomeHandler Visualization')
    parser.add_argument('--visual', action='store_true', 
                       help='Enable visual inspection mode (generates plots)')
    
    args, unknown = parser.parse_known_args()
    
    # Set visual mode flag on the test class
    if args.visual:
        TestCartesianEulerVisualization._visual_mode = True
        print("Visual inspection mode enabled - plots will be generated")
    
    return unknown


if __name__ == '__main__':
    # Set up visual mode and get remaining arguments
    unknown_args = setup_visual_mode()
    
    # Prepare unittest arguments
    unittest_args = [sys.argv[0]] + unknown_args
    
    # Run the tests
    unittest.main(argv=unittest_args, verbosity=2)