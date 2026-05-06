#!/usr/bin/env python3
"""
Visualization tests for OptimizationBasedRepairOperator.

This module contains tests for visual inspection of the optimization-based repair operator:
- Side-by-side comparison of original and repaired genomes
- Collision detection visualization
- Repair effectiveness across different configurations
"""

import unittest
import numpy as np
import argparse
import sys
import os
from typing import List, Tuple

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Visualization
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from airevolve.evolution_tools.genome_handlers.spherical_angular_genome_handler import SphericalAngularDroneGenomeHandler
from airevolve.evolution_tools.genome_handlers.cartesian_euler_genome_handler import CartesianEulerDroneGenomeHandler
from airevolve.evolution_tools.genome_handlers.operators.optimization_repair_operator import (
    OptimizationBasedRepairOperator,
    OptimizationRepairConfig,
)
from airevolve.evolution_tools.genome_handlers.conversions.arm_conversions import (
    arms_to_cylinders_polar_angular,
    arms_to_cylinders_cartesian_euler,
)
from airevolve.evolution_tools.inspection_tools.drone_visualizer import DroneVisualizer
import time


class TestOptimizationRepairVisualization(unittest.TestCase):
    """Test visualization capabilities for OptimizationBasedRepairOperator."""

    def setUp(self):
        """Set up test fixtures."""
        self.rnd = np.random.default_rng(42)  # Fixed seed for reproducibility
        self.propeller_radius = 0.0254
        self.cylinder_height = 0.3048
        self.inner_boundary_radius = 0.09
        self.outer_boundary_radius = 0.4

    def _create_collision_genome_spherical(self) -> np.ndarray:
        """Create a test genome with intentional collisions (spherical coordinates)."""
        # Create 4 arms with intentional collisions
        genome = np.array([
            [0.15, 0.0, np.pi/4, 0.0, 0.0, 1],      # Arm 1
            [0.15, 0.1, np.pi/4, 0.0, 0.0, 1],      # Arm 2 - close to Arm 1 (collision!)
            [0.2, np.pi, np.pi/3, 0.0, 0.0, 1],     # Arm 3
            [0.2, -np.pi, np.pi/3, 0.0, 0.0, 1],    # Arm 4 - close to Arm 3 (collision!)
        ])
        return genome

    def _create_core_collision_genome_spherical(self) -> np.ndarray:
        """Create a test genome where arms pass through the central core."""
        genome = np.array([
            [0.3, 0.0, np.pi/6, 0.5, 0.0, 1],       # Arm angled toward core
            [0.3, np.pi/2, np.pi/6, 0.5, 0.0, 1],
            [0.3, np.pi, np.pi/6, 0.5, 0.0, 1],
            [0.3, -np.pi/2, np.pi/6, 0.5, 0.0, 1],
        ])
        return genome

    def _plot_cylinder_visualization(self, ax, cylinders, config, title, genome=None, highlight_collisions=True):
        """Helper method to plot cylinders with boundary constraints.

        Parameters:
        -----------
        ax : matplotlib 3D axis
            The axis to plot on
        cylinders : list
            List of Cylinder objects
        config : OptimizationRepairConfig
            Configuration object with disc and boundary parameters
        title : str
            Plot title
        genome : np.ndarray, optional
            Genome array to compute disc attachment points (shape: n_arms x 6)
        highlight_collisions : bool
            Whether to highlight collisions (not currently used)
        """
        # Draw boundary constraints as transparent spheres
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)

        # Core sphere (red)
        x_core = config.core_radius * np.outer(np.cos(u), np.sin(v))
        y_core = config.core_radius * np.outer(np.sin(u), np.sin(v))
        z_core = config.core_radius * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x_core, y_core, z_core, color='r', alpha=0.15, label='Core')

        # Inner boundary (green)
        x_inner = config.inner_boundary_radius * np.outer(np.cos(u), np.sin(v))
        y_inner = config.inner_boundary_radius * np.outer(np.sin(u), np.sin(v))
        z_inner = config.inner_boundary_radius * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x_inner, y_inner, z_inner, color='g', alpha=0.08)

        # Outer boundary (blue)
        x_outer = config.outer_boundary_radius * np.outer(np.cos(u), np.sin(v))
        y_outer = config.outer_boundary_radius * np.outer(np.sin(u), np.sin(v))
        z_outer = config.outer_boundary_radius * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x_outer, y_outer, z_outer, color='b', alpha=0.08)

        # Draw attachment disc (horizontal disc at z=disc_height)
        # Create a filled circle at the disc height
        theta_disc = np.linspace(0, 2 * np.pi, 50)
        x_disc = config.disc_radius * np.cos(theta_disc)
        y_disc = config.disc_radius * np.sin(theta_disc)
        z_disc = np.ones_like(theta_disc) * config.disc_height

        # Draw disc edge as a thick line
        ax.plot(x_disc, y_disc, z_disc, 'k-', linewidth=2, alpha=0.7, label='Attachment Disc')

        # Draw disc surface (semi-transparent)
        theta_mesh = np.linspace(0, 2 * np.pi, 20)
        r_mesh = np.linspace(0, config.disc_radius, 10)
        Theta_mesh, R_mesh = np.meshgrid(theta_mesh, r_mesh)
        X_disc_surf = R_mesh * np.cos(Theta_mesh)
        Y_disc_surf = R_mesh * np.sin(Theta_mesh)
        Z_disc_surf = np.ones_like(X_disc_surf) * config.disc_height
        ax.plot_surface(X_disc_surf, Y_disc_surf, Z_disc_surf, color='gray', alpha=0.15)

        # If genome is provided, draw attachment lines from disc edge to motors
        if genome is not None:
            from airevolve.evolution_tools.genome_handlers.operators.optimization_repair_operator import (
                compute_base_point
            )

            for i, cyl in enumerate(cylinders):
                # Get motor position from cylinder
                motor_pos = cyl.position

                # Compute base point on disc edge
                base_point = compute_base_point(
                    motor_pos,
                    config.disc_radius,
                    config.disc_height,
                    config.min_xy_projection
                )

                # Draw line from disc edge to motor
                ax.plot([base_point[0], motor_pos[0]],
                       [base_point[1], motor_pos[1]],
                       [base_point[2], motor_pos[2]],
                       'orange', linewidth=1.5, alpha=0.6, linestyle='--')

                # Draw a small sphere at the disc attachment point
                ax.scatter([base_point[0]], [base_point[1]], [base_point[2]],
                          c='orange', s=30, alpha=0.8, marker='o')

        # Draw cylinders
        visualizer = DroneVisualizer()
        visualizer.plot_cylinders_3d(cylinders, title=title, ax=ax)

        # Set equal aspect ratio and limits
        max_range = config.outer_boundary_radius * 1.2
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        ax.set_zlim(-max_range, max_range)

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(title, fontsize=10, weight='bold')
        ax.view_init(elev=20, azim=45)

    def test_optimization_repair_visualization(self):
        """Test side-by-side visualization of optimization repair for different cases."""
        # Skip if not in visual mode
        if not getattr(self, '_visual_mode', False):
            self.skipTest("Optimization repair visualization only runs with --visual flag")

        # Create output directory
        output_dir = "test_output"
        os.makedirs(output_dir, exist_ok=True)

        # Configuration for optimization repair
        config = OptimizationRepairConfig(
            disc_radius=0.15,
            disc_height=0.0,
            core_radius=0.05,
            propeller_radius=self.propeller_radius,
            propeller_tolerance=0.1,
            cylinder_height=self.cylinder_height,
            inner_boundary_radius=self.inner_boundary_radius,
            outer_boundary_radius=self.outer_boundary_radius,
            optimization_method='SLSQP',
            max_iterations=1000,
        )

        # Test cases with different collision scenarios
        test_cases = [
            ("Basic Collisions", self._create_collision_genome_spherical()),
            ("Core Collisions", self._create_core_collision_genome_spherical()),
        ]

        # Create repair operator
        repair_op = OptimizationBasedRepairOperator(
            optimization_config=config,
            coordinate_system='spherical',
            verbose=False
        )

        # Create figure with 2 rows (one per test case) and 2 columns (before/after)
        fig = plt.figure(figsize=(14, 8 * len(test_cases)))

        for case_idx, (case_name, original_genome) in enumerate(test_cases):
            print(f"\n{'='*70}")
            print(f"Processing: {case_name}")
            print(f"{'='*70}")

            # Check original validity
            is_valid_before = repair_op.validate(original_genome)
            print(f"Original genome valid: {is_valid_before}")

            # Get cylinders for original genome
            cylinders_before = arms_to_cylinders_polar_angular(
                original_genome,
                propeller_radius=config.propeller_radius,
                cylinder_height=config.cylinder_height
            )

            # Plot BEFORE repair (left column)
            ax_before = fig.add_subplot(len(test_cases), 2, case_idx * 2 + 1, projection='3d')
            self._plot_cylinder_visualization(
                ax_before, cylinders_before, config,
                f"{case_name}\nBEFORE Repair\n(Valid: {'✓' if is_valid_before else '✗'})",
                genome=original_genome
            )

            # Perform repair
            print(f"Repairing genome...")
            start_time = time.time()
            repaired_genome = repair_op.repair(original_genome)
            repair_time = time.time() - start_time
            print(f"Repair completed in {repair_time:.3f} seconds")

            # Check repaired validity
            is_valid_after = repair_op.validate(repaired_genome)
            print(f"Repaired genome valid: {is_valid_after}")

            # Calculate changes
            changes = np.abs(repaired_genome - original_genome)
            total_change = np.linalg.norm(changes)
            print(f"Total change (L2 norm): {total_change:.6f}")

            # Get cylinders for repaired genome
            cylinders_after = arms_to_cylinders_polar_angular(
                repaired_genome,
                propeller_radius=config.propeller_radius,
                cylinder_height=config.cylinder_height
            )

            # Plot AFTER repair (right column)
            ax_after = fig.add_subplot(len(test_cases), 2, case_idx * 2 + 2, projection='3d')
            self._plot_cylinder_visualization(
                ax_after, cylinders_after, config,
                f"{case_name}\nAFTER Repair\n(Valid: {'✓' if is_valid_after else '✗'}, Change: {total_change:.3f})",
                genome=repaired_genome
            )

            # Print summary
            print(f"Summary:")
            print(f"  - Repair succeeded: {is_valid_after}")
            print(f"  - Time taken: {repair_time:.3f}s")
            print(f"  - L2 change: {total_change:.6f}")

        plt.tight_layout()
        output_file = os.path.join(output_dir, 'optimization_repair_comparison.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\n{'='*70}")
        print(f"Saved visualization to {output_file}")
        print(f"{'='*70}\n")
        plt.show()
        plt.close(fig)

    def test_repair_operator_comparison(self):
        """Compare optimization repair with different configurations."""
        # Skip if not in visual mode
        if not getattr(self, '_visual_mode', False):
            self.skipTest("Repair operator comparison only runs with --visual flag")

        # Create output directory
        output_dir = "test_output"
        os.makedirs(output_dir, exist_ok=True)

        # Original genome with collisions
        original_genome = self._create_collision_genome_spherical()

        # Different configurations to test
        configs = [
            ("Default Config", OptimizationRepairConfig()),
            ("Tight Tolerance", OptimizationRepairConfig(
                propeller_tolerance=0.05,  # Tighter clearance
                constraint_tolerance=1e-8,
            )),
            ("Large Core", OptimizationRepairConfig(
                core_radius=0.08,  # Larger core
                disc_radius=0.18,
            )),
        ]

        # Create figure
        fig = plt.figure(figsize=(14, 6 * len(configs)))

        for idx, (config_name, config) in enumerate(configs):
            print(f"\n{'='*70}")
            print(f"Testing: {config_name}")
            print(f"{'='*70}")

            # Create repair operator
            repair_op = OptimizationBasedRepairOperator(
                optimization_config=config,
                coordinate_system='spherical',
                verbose=False
            )

            # Original cylinders
            cylinders_before = arms_to_cylinders_polar_angular(
                original_genome,
                propeller_radius=config.propeller_radius,
                cylinder_height=config.cylinder_height
            )

            # Plot BEFORE (left)
            ax_before = fig.add_subplot(len(configs), 2, idx * 2 + 1, projection='3d')
            is_valid_before = repair_op.validate(original_genome)
            self._plot_cylinder_visualization(
                ax_before, cylinders_before, config,
                f"{config_name}\nBEFORE\n(Valid: {'✓' if is_valid_before else '✗'})",
                genome=original_genome
            )

            # Repair
            start_time = time.time()
            repaired_genome = repair_op.repair(original_genome)
            repair_time = time.time() - start_time

            # Repaired cylinders
            cylinders_after = arms_to_cylinders_polar_angular(
                repaired_genome,
                propeller_radius=config.propeller_radius,
                cylinder_height=config.cylinder_height
            )

            # Plot AFTER (right)
            ax_after = fig.add_subplot(len(configs), 2, idx * 2 + 2, projection='3d')
            is_valid_after = repair_op.validate(repaired_genome)
            total_change = np.linalg.norm(repaired_genome - original_genome)
            self._plot_cylinder_visualization(
                ax_after, cylinders_after, config,
                f"{config_name}\nAFTER\n(Valid: {'✓' if is_valid_after else '✗'}, Time: {repair_time:.2f}s)",
                genome=repaired_genome
            )

            print(f"  Valid: {is_valid_after}, Time: {repair_time:.3f}s, Change: {total_change:.6f}")

        plt.tight_layout()
        output_file = os.path.join(output_dir, 'optimization_repair_configs.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\nSaved configuration comparison to {output_file}")
        plt.show()
        plt.close(fig)

    def test_random_population_repair(self):
        """Test repair on a random population of drones."""
        # Skip if not in visual mode
        if not getattr(self, '_visual_mode', False):
            self.skipTest("Random population repair only runs with --visual flag")

        # Create output directory
        output_dir = "test_output"
        os.makedirs(output_dir, exist_ok=True)

        # Create a spherical genome handler that generates collision-prone genomes
        handler = SphericalAngularDroneGenomeHandler(
            min_max_narms=(6, 6),
            parameter_limits=[
                [0.1, 0.25],  # Small r range (more collisions)
                [-np.pi, np.pi],
                [np.pi/4, 3*np.pi/4],  # Limited phi range
                [-np.pi, np.pi],
                [-np.pi, np.pi],
                [0, 1]
            ],
            append_arm_chance=0.0,
            repair=False,
            rnd=self.rnd
        )

        # Generate random genomes
        population = handler.generate_random_population(6)

        # Configuration
        config = OptimizationRepairConfig(
            propeller_radius=self.propeller_radius,
            cylinder_height=self.cylinder_height,
        )

        # Create repair operator
        repair_op = OptimizationBasedRepairOperator(
            optimization_config=config,
            coordinate_system='spherical',
            verbose=False
        )

        # Create figure with 3 rows and 4 columns
        fig = plt.figure(figsize=(16, 12))

        valid_count_before = 0
        valid_count_after = 0

        for idx, genome_handler in enumerate(population[:6]):
            original_genome = getattr(genome_handler.genome, 'arms', genome_handler.genome)

            # Check validity before
            is_valid_before = repair_op.validate(original_genome)
            if is_valid_before:
                valid_count_before += 1

            # Get cylinders before
            cylinders_before = arms_to_cylinders_polar_angular(
                original_genome,
                propeller_radius=config.propeller_radius,
                cylinder_height=config.cylinder_height
            )

            # Plot BEFORE (left side)
            ax_before = fig.add_subplot(3, 4, idx * 2 + 1, projection='3d')
            self._plot_cylinder_visualization(
                ax_before, cylinders_before, config,
                f"Drone {idx+1} - BEFORE\n({'Valid' if is_valid_before else 'Invalid'})",
                genome=original_genome
            )

            # Repair
            repaired_genome = repair_op.repair(original_genome)
            is_valid_after = repair_op.validate(repaired_genome)
            if is_valid_after:
                valid_count_after += 1

            # Get cylinders after
            cylinders_after = arms_to_cylinders_polar_angular(
                repaired_genome,
                propeller_radius=config.propeller_radius,
                cylinder_height=config.cylinder_height
            )

            # Plot AFTER (right side)
            ax_after = fig.add_subplot(3, 4, idx * 2 + 2, projection='3d')
            self._plot_cylinder_visualization(
                ax_after, cylinders_after, config,
                f"Drone {idx+1} - AFTER\n({'Valid' if is_valid_after else 'Invalid'})",
                genome=repaired_genome
            )

        plt.suptitle(f'Random Population Repair\nValid Before: {valid_count_before}/6, Valid After: {valid_count_after}/6',
                     fontsize=14, weight='bold')
        plt.tight_layout()
        output_file = os.path.join(output_dir, 'optimization_repair_population.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\nSaved population repair visualization to {output_file}")
        print(f"Valid genomes before repair: {valid_count_before}/6")
        print(f"Valid genomes after repair: {valid_count_after}/6")
        plt.show()
        plt.close(fig)


def setup_visual_mode():
    """Set up visual mode based on command line arguments."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test OptimizationBasedRepairOperator Visualization')
    parser.add_argument('--visual', action='store_true',
                       help='Enable visual inspection mode (generates plots)')

    args, unknown = parser.parse_known_args()

    # Set visual mode flag on the test class
    if args.visual:
        TestOptimizationRepairVisualization._visual_mode = True
        print("Visual inspection mode enabled - plots will be generated")

    return unknown


if __name__ == '__main__':
    # Set up visual mode and get remaining arguments
    unknown_args = setup_visual_mode()

    # Prepare unittest arguments
    unittest_args = [sys.argv[0]] + unknown_args

    # Run the tests
    unittest.main(argv=unittest_args, verbosity=2)
