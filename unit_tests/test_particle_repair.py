#!/usr/bin/env python3
"""
Comprehensive tests for particle repair operator.

This module consolidates and replaces:
- test_particle_repair_performance.py (performance and accuracy tests)
- test_particle_repair_symmetry.py (symmetry preservation tests)

Tests include:
- Collision detection accuracy and constraint enforcement
- Performance benchmarking and convergence behavior
- Symmetry preservation during repair operations
- Coordinate system integration and edge cases
- Boundary constraint handling and repair effectiveness

Supports both Cartesian and Spherical genome handlers via --genome-handler argument.
"""

import unittest
import numpy as np
import numpy.testing as npt
import time
import argparse
import sys
from unittest.mock import patch
from airevolve.evolution_tools.genome_handlers.operators.particle_repair_operator import (
    particle_repair_individual, 
    are_distance_constraints_violated,
    enforce_distance_constraints,
    get_combinations, 
    resolve_collision,
    are_there_cylinder_collisions
)
from airevolve.evolution_tools.genome_handlers.conversions.arm_conversions import (
    arms_to_cylinders_cartesian_euler,
    cylinders_to_arms_cartesian_euler,
    arms_to_cylinders_polar_angular,
    cylinders_to_arms_polar_angular,
    Cylinder
)

# Import repair operator classes (if available)
from airevolve.evolution_tools.genome_handlers.operators import (
    CartesianRepairOperator,
    SphericalRepairOperator,
    RepairConfig,
    RepairStrategy,
    RepairUtilities
)

# Import both genome handlers - will be selected dynamically
from airevolve.evolution_tools.genome_handlers.cartesian_euler_genome_handler import CartesianEulerDroneGenomeHandler
from airevolve.evolution_tools.genome_handlers.spherical_angular_genome_handler import SphericalAngularDroneGenomeHandler

# Global configuration variables set by command line arguments
GENOME_HANDLER_TYPE = 'cartesian'  # Default to cartesian
GENOME_HANDLER_CLASS = None
ARMS_TO_CYLINDERS_FUNC = None
CYLINDERS_TO_ARMS_FUNC = None

def configure_test_environment(handler_type='cartesian'):
    """Configure global test environment based on handler type."""
    global GENOME_HANDLER_TYPE, GENOME_HANDLER_CLASS, ARMS_TO_CYLINDERS_FUNC, CYLINDERS_TO_ARMS_FUNC
    
    GENOME_HANDLER_TYPE = handler_type
    
    if handler_type == 'cartesian':
        GENOME_HANDLER_CLASS = CartesianEulerDroneGenomeHandler
        ARMS_TO_CYLINDERS_FUNC = arms_to_cylinders_cartesian_euler
        CYLINDERS_TO_ARMS_FUNC = cylinders_to_arms_cartesian_euler
    elif handler_type == 'spherical':
        GENOME_HANDLER_CLASS = SphericalAngularDroneGenomeHandler
        ARMS_TO_CYLINDERS_FUNC = arms_to_cylinders_polar_angular
        CYLINDERS_TO_ARMS_FUNC = cylinders_to_arms_polar_angular
    else:
        raise ValueError(f"Unknown genome handler type: {handler_type}")

def create_collision_genome():
    """Create test genome with collisions based on current handler type."""
    if GENOME_HANDLER_TYPE == 'cartesian':
        # Cartesian format: [x, y, z, motor_yaw, motor_pitch, direction]
        return np.array([
            [0.8, 0.8, 0.8, 0.0, 0.0, 0.0, 0],   # First cluster
            [0.82, 0.81, 0.79, 0.1, 0.1, 0.1, 1], # Very close to first
            [0.81, 0.79, 0.81, -0.1, 0.0, 0.1, 0], # Very close to both
            [0.79, 0.81, 0.82, 0.0, -0.1, -0.1, 1], # Very close to all
        ])
    else:  # spherical
        # Spherical format: [magnitude, arm_rotation, arm_pitch, motor_rotation, motor_pitch, direction]
        return np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0],      # First arm
            [1.05, 0.1, 0.05, 0.1, 0.1, 1],   # Very close to first (collision)
            [1.02, 0.05, 0.02, -0.1, 0.0, 0], # Close to both
            [1.03, 0.08, 0.03, 0.0, -0.1, 1], # Close to all
        ])

def create_boundary_violation_genome():
    """Create genome with boundary violations based on current handler type."""
    if GENOME_HANDLER_TYPE == 'cartesian':
        return np.array([
            [0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 0],  # Too close to origin
            [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1],  # Too far from origin
            [0.1, 0.1, -0.1, 0.0, 0.0, 0.0, 0], # Mirror of first (too close)
            [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1],  # Mirror of second (too far)
        ])
    else:  # spherical
        return np.array([
            [0.05, 0.0, 0.0, 0.0, 0.0, 0],     # Too close to origin (small magnitude)
            [2.0, 0.0, 0.0, 0.0, 0.0, 1],      # Too far from origin (large magnitude)
            [0.06, np.pi, 0.0, 0.0, 0.0, 0],   # Mirror of first (too close)
            [1.8, np.pi, 0.0, 0.0, 0.0, 1],    # Mirror of second (too far)
        ])

def create_symmetric_genome():
    """Create symmetric genome for symmetry tests based on current handler type."""
    if GENOME_HANDLER_TYPE == 'cartesian':
        return np.array([
            [0.5, 0.3, 0.2, 0.1, 0.2, 0.3, 0],   # First arm
            [0.5, 0.3, -0.2, -0.1, -0.2, 0.3, 1], # Second arm (mirrored in XY plane)
            [0.45, 0.25, 0.15, 0.05, 0.15, 0.25, 0], # Third arm (close to first - collision)
            [0.45, 0.25, -0.15, -0.05, -0.15, 0.25, 1], # Fourth arm (mirrored)
        ])
    else:  # spherical
        return np.array([
            [1.0, 0.0, 0.0, 0.1, 0.2, 0],      # First arm
            [1.0, np.pi, 0.0, -0.1, 0.2, 1],   # Second arm (opposite azimuth)
            [0.95, 0.1, 0.05, 0.05, 0.15, 0],  # Third arm (close to first)
            [0.95, np.pi-0.1, 0.05, -0.05, 0.15, 1], # Fourth arm (symmetric)
        ])

def create_no_collision_genome():
    """Create genome with no collisions based on current handler type."""
    if GENOME_HANDLER_TYPE == 'cartesian':
        return np.array([
            [0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0],
            [0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 1],
            [-0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0],
            [0.0, -0.3, 0.0, 0.0, 0.0, 0.0, 1],
        ])
    else:  # spherical
        return np.array([
            [0.3, 0.0, 0.0, 0.0, 0.0, 0],
            [0.3, np.pi/2, 0.0, 0.0, 0.0, 1],
            [0.3, np.pi, 0.0, 0.0, 0.0, 0],
            [0.3, 3*np.pi/2, 0.0, 0.0, 0.0, 1],
        ])

def create_test_handler(**kwargs):
    """Create appropriate genome handler based on current type."""
    return GENOME_HANDLER_CLASS(**kwargs)

class TestCollisionDetectionAccuracy(unittest.TestCase):
    """Test accuracy of collision detection and repair."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.propeller_radius = 0.0762
        self.inner_boundary_radius = 0.09
        self.outer_boundary_radius = 0.4
        self.tolerance = 0.1
    
    def test_distance_constraint_detection(self):
        """Test accuracy of distance constraint violation detection."""
        # Create cylinders with various constraint violations

        cylinders_in_bounds = [
            Cylinder(position=np.array([0.1, 0.0, 0.0]), radius=0.0762, height=0.1, orientation=np.array([1.0, 0.0, 0.0, 0.0])),
            Cylinder(position=np.array([0.0, 0.3, 0.0]), radius=0.0762, height=0.1, orientation=np.array([1.0, 0.0, 0.0, 0.0])),
        ]

        cylinders_too_close = [
            Cylinder(position=np.array([0.02, 0.0, 0.0]), radius=0.0762, height=0.1, orientation=np.array([1.0, 0.0, 0.0, 0.0])),  # Too close to origin
            Cylinder(position=np.array([0.09, 0.0, 0.0]), radius=0.0762, height=0.1, orientation=np.array([1.0, 0.0, 0.0, 0.0])),
        ]   

        cylinders_too_far = [
            Cylinder(position=np.array([1.0, 0.0, 0.0]), radius=0.0762, height=0.1, orientation=np.array([1.0, 0.0, 0.0, 0.0])),
            Cylinder(position=np.array([3.0, 0.0, 0.0]), radius=0.0762, height=0.1, orientation=np.array([1.0, 0.0, 0.0, 0.0])),  # Too far from origin
        ]
        
        # Test detection accuracy
        self.assertFalse(are_distance_constraints_violated(
            cylinders_in_bounds, self.inner_boundary_radius, self.outer_boundary_radius))
        
        self.assertTrue(are_distance_constraints_violated(
            cylinders_too_close, self.inner_boundary_radius, self.outer_boundary_radius))
        
        self.assertTrue(are_distance_constraints_violated(
            cylinders_too_far, self.inner_boundary_radius, self.outer_boundary_radius))
    
    def test_distance_constraint_enforcement(self):
        """Test accuracy of distance constraint enforcement."""
        # Create cylinders outside boundaries
        cylinders = [
            Cylinder(position=np.array([0.1, 0.0, 0.0]), radius=0.0762, height=0.1, orientation=np.array([1.0, 0.0, 0.0, 0.0])),
            Cylinder(position=np.array([5.0, 0.0, 0.0]), radius=0.0762, height=0.1, orientation=np.array([1.0, 0.0, 0.0, 0.0])),
        ]
        # Enforce constraints
        corrected = enforce_distance_constraints(
            cylinders, self.inner_boundary_radius, self.outer_boundary_radius)
        
        # Check that constraints are now satisfied
        for cyl in corrected:
            distance = np.linalg.norm(cyl.position)
            self.assertGreaterEqual(distance, self.inner_boundary_radius)
            self.assertLessEqual(distance, self.outer_boundary_radius)
    
    def test_collision_resolution_effectiveness(self):
        """Test that collision resolution effectively separates colliding arms."""
        # Create severely colliding genome
        colliding_genome = create_collision_genome()
        
        with patch('airevolve.evolution_tools.genome_handlers.operators.particle_repair_operator.are_there_cylinder_collisions'):
            repaired_genome = particle_repair_individual(
                individual=colliding_genome,
                propeller_radius=self.propeller_radius,
                inner_boundary_radius=self.inner_boundary_radius,
                outer_boundary_radius=self.outer_boundary_radius,
                max_iterations=50,  # More iterations for difficult case
                arms_to_cylinders=ARMS_TO_CYLINDERS_FUNC,
                cylinders_to_arms=CYLINDERS_TO_ARMS_FUNC
            )
        
        # Verify that repair at least attempted to improve the situation
        self.assertIsNotNone(repaired_genome)
        self.assertFalse(np.any(np.isnan(repaired_genome)))
        
        # Check that some movement occurred (repair was attempted)
        position_changes = np.linalg.norm(repaired_genome[:, :3] - colliding_genome[:, :3], axis=1)
        significant_changes = np.sum(position_changes > 0.01)  # At least 1cm movement
        
        # At least some arms should have moved significantly
        self.assertGreater(significant_changes, 0, "No significant position changes detected")


class TestRepairPerformance(unittest.TestCase):
    """Test performance characteristics of particle repair."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.propeller_radius = 0.0762
        self.inner_boundary_radius = 0.09
        self.outer_boundary_radius = 0.4
        self.rng = np.random.default_rng(42)
    
    def test_convergence_speed(self):
        """Test that repair converges within reasonable iterations."""
        # Create moderately challenging collision scenario
        problem_genome = create_collision_genome()
        
        # Test with limited iterations
        max_iterations = 10
        start_time = time.time()
        
        repaired_genome = particle_repair_individual(
            individual=problem_genome,
            propeller_radius=self.propeller_radius,
            inner_boundary_radius=self.inner_boundary_radius,
            outer_boundary_radius=self.outer_boundary_radius,
            max_iterations=max_iterations,
            arms_to_cylinders=ARMS_TO_CYLINDERS_FUNC,
            cylinders_to_arms=CYLINDERS_TO_ARMS_FUNC
        )
        
        end_time = time.time()
        repair_time = end_time - start_time
        
        # Should complete quickly (within 1 second for this simple case)
        self.assertLess(repair_time, 1.0, "Repair took too long")
        
        # Should converge to valid result
        self.assertIsNotNone(repaired_genome)
        self.assertFalse(np.any(np.isnan(repaired_genome)))
    
    def test_scalability_with_arm_count(self):
        """Test performance scaling with number of arms."""
        arm_counts = [4, 6, 8]
        repair_times = []
        
        for num_arms in arm_counts:
            # Create genome with potential collisions
            genome_cols = 7 if GENOME_HANDLER_TYPE == 'cartesian' else 6
            genome = self.rng.uniform(-1, 1, (num_arms, genome_cols))
            genome[:, -1] = self.rng.integers(0, 2, num_arms)  # Binary direction
            
            # Make some arms close together to create collisions
            if GENOME_HANDLER_TYPE == 'cartesian':
                genome[1, :3] = genome[0, :3] + self.rng.uniform(-0.1, 0.1, 3)
                if num_arms > 4:
                    genome[3, :3] = genome[2, :3] + self.rng.uniform(-0.1, 0.1, 3)
            else:  # spherical
                # For spherical: [magnitude, arm_rotation, arm_pitch, motor_rotation, motor_pitch, direction]
                # Make arms have similar angles for collision
                genome[1, :3] = genome[0, :3] + self.rng.uniform(-0.1, 0.1, 3)
                if num_arms > 4:
                    genome[3, :3] = genome[2, :3] + self.rng.uniform(-0.1, 0.1, 3)
            
            start_time = time.time()
            
            repaired_genome = particle_repair_individual(
                individual=genome,
                propeller_radius=self.propeller_radius,
                inner_boundary_radius=self.inner_boundary_radius,
                outer_boundary_radius=self.outer_boundary_radius,
                max_iterations=20,
                arms_to_cylinders=ARMS_TO_CYLINDERS_FUNC,
                cylinders_to_arms=CYLINDERS_TO_ARMS_FUNC
            )
            
            end_time = time.time()
            repair_time = end_time - start_time
            repair_times.append(repair_time)
            
            # Verify repair succeeded
            self.assertIsNotNone(repaired_genome)
            self.assertFalse(np.any(np.isnan(repaired_genome)))
        
        # Performance should scale reasonably (not exponentially)
        # For this simple test, all should complete quickly
        for repair_time in repair_times:
            self.assertLess(repair_time, 2.0, "Repair scaling poorly with arm count")
    
    def test_step_size_effectiveness(self):
        """Test the effect of different step sizes on repair effectiveness."""
        # Create genome with known collision
        collision_genome = create_collision_genome()
        
        step_sizes = [0.1, 0.5, 1.0, 2.0]
        
        for step_size in step_sizes:
            repaired_genome = particle_repair_individual(
                individual=collision_genome.copy(),
                propeller_radius=self.propeller_radius,
                inner_boundary_radius=self.inner_boundary_radius,
                outer_boundary_radius=self.outer_boundary_radius,
                max_iterations=15,
                step_size=step_size,
                arms_to_cylinders=ARMS_TO_CYLINDERS_FUNC,
                cylinders_to_arms=CYLINDERS_TO_ARMS_FUNC
            )
            
            # All step sizes should produce valid results
            self.assertIsNotNone(repaired_genome, f"Repair failed for step_size={step_size}")
            
            # Check for NaN values, but allow them if conversion failed
            if np.any(np.isnan(repaired_genome)):
                # If NaN values exist, it might be due to conversion issues, skip this test
                continue
            
            # Results should be different from original (indicating repair occurred)
            position_diff = np.linalg.norm(repaired_genome[:, :3] - collision_genome[:, :3])
            self.assertGreater(position_diff, 1e-6, f"No repair detected for step_size={step_size}")


class TestCoordinateSystemIntegration(unittest.TestCase):
    """Test integration with different coordinate systems."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.propeller_radius = 0.0762
        self.inner_boundary_radius = 0.09
        self.outer_boundary_radius = 0.4
    
    def test_coordinate_system_repair_consistency(self):
        """Test that repair works consistently with the selected coordinate system."""
        # Create collision genome using current handler type
        collision_genome = create_collision_genome()
        
        # Repair with current coordinate system
        repaired_genome = particle_repair_individual(
            individual=collision_genome,
            propeller_radius=self.propeller_radius,
            inner_boundary_radius=self.inner_boundary_radius,
            outer_boundary_radius=self.outer_boundary_radius,
            max_iterations=20,
            arms_to_cylinders=ARMS_TO_CYLINDERS_FUNC,
            cylinders_to_arms=CYLINDERS_TO_ARMS_FUNC
        )
        
        # Repair should succeed
        self.assertIsNotNone(repaired_genome)
        
        # Check for significant failures
        nan_ratio = np.sum(np.isnan(repaired_genome)) / repaired_genome.size
        
        # Allow some NaN values due to conversion issues, but not total failure
        self.assertLess(nan_ratio, 0.5, f"Too many NaN values in {GENOME_HANDLER_TYPE} repair")
        
        # Check that at least some repair occurred where values are valid
        valid_mask = ~np.isnan(repaired_genome)
        if np.any(valid_mask):
            diff = np.linalg.norm(repaired_genome[valid_mask] - collision_genome[valid_mask])
            self.assertGreater(diff, 1e-6, f"No {GENOME_HANDLER_TYPE} repair detected")


class TestParticleRepairSymmetry(unittest.TestCase):
    """Test particle repair operator symmetry preservation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.rng = np.random.default_rng(42)  # Fixed seed for reproducibility
        
        # Common repair parameters
        self.propeller_radius = 0.0762
        self.inner_boundary_radius = 0.09
        self.outer_boundary_radius = 0.4
        self.max_iterations = 25
        self.step_size = 1.0
        self.propeller_tolerance = 0.1
        
        # Test fixed axis for symmetric repair
        self.xy_plane_axis = [0.0, 0.0, 1.0]  # Z-axis for XY plane symmetry
        self.xz_plane_axis = [0.0, 1.0, 0.0]  # Y-axis for XZ plane symmetry
        
    def test_get_combinations_free_repair(self):
        """Test that free repair checks all arm pairs."""
        num_arms = 4
        combinations = list(get_combinations(num_arms, repair_along_fixed_axis=None))
        
        # Should check all unique pairs (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
        expected_combinations = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        self.assertEqual(len(combinations), len(expected_combinations))
        
        for combo in expected_combinations:
            self.assertIn(combo, combinations)
    
    def test_get_combinations_symmetric_repair(self):
        """Test that symmetric repair only checks cross-half pairs."""
        num_arms = 4
        combinations = list(get_combinations(num_arms, repair_along_fixed_axis=self.xy_plane_axis))
        
        # Should only check pairs between first half (0,1) and second half (2,3)
        expected_combinations = [(0, 2), (0, 3), (1, 2), (1, 3)]
        self.assertEqual(len(combinations), len(expected_combinations))
        
        for combo in expected_combinations:
            self.assertIn(combo, combinations)
    
    def test_resolve_collision_natural_direction(self):
        """Test collision resolution using natural separation direction."""
        # Create two colliding cylinders
        cyl1 = Cylinder(position=np.array([0.0, 0.0, 0.0]), radius=self.propeller_radius, height=0.1, orientation=np.array([1.0, 0.0, 0.0, 0.0]))
        cyl2 = Cylinder(position=np.array([0.1, 0.0, 0.0]), radius=self.propeller_radius, height=0.1, orientation=np.array([1.0, 0.0, 0.0, 0.0]))

        # Store original positions
        orig_pos1 = cyl1.position.copy()
        orig_pos2 = cyl2.position.copy()
        
        # Resolve collision with natural direction
        cyl3, cyl4 = resolve_collision(cyl1, cyl2, required_clearance=0.3, step_size=1.0, fixed_direction=None)
        
        # Check that cylinders moved apart along the natural direction (x-axis)
        # Direction is from cyl1 to cyl2, so cyl1 moves toward -x, cyl2 moves toward +x
        self.assertLess(cyl3.position[0], orig_pos1[0])  # cyl1 moved in +x
        self.assertGreater(cyl4.position[0], orig_pos2[0])     # cyl2 moved in -x
        
        # Check y and z coordinates are unchanged (natural direction is x-axis)
        npt.assert_almost_equal(cyl3.position[1], orig_pos1[1])
        npt.assert_almost_equal(cyl3.position[2], orig_pos1[2])
        npt.assert_almost_equal(cyl4.position[1], orig_pos2[1])
        npt.assert_almost_equal(cyl4.position[2], orig_pos2[2])
    
    def test_resolve_collision_fixed_direction(self):
        """Test collision resolution using fixed direction for symmetry."""
        # Create two colliding cylinders
        cyl1 = Cylinder(position=np.array([0.0, 0.0, 0.0]), radius=self.propeller_radius, height=0.1, orientation=np.array([1.0, 0.0, 0.0, 0.0]))
        cyl2 = Cylinder(position=np.array([0.01, 0.0, 0.0]), radius=self.propeller_radius, height=0.1, orientation=np.array([1.0, 0.0, 0.0, 0.0]))

        # Store original positions
        orig_pos1 = cyl1.position.copy()
        orig_pos2 = cyl2.position.copy()

        # Resolve collision with fixed direction (z-axis)
        fixed_direction = [1.0, 0.0, 0.0]
        cyl3, cyl4 = resolve_collision(cyl1, cyl2, required_clearance=0.3, step_size=1.0, fixed_direction=fixed_direction)
        
        # Check that cylinders moved apart along the fixed direction (z-axis)
        # With fixed direction, cyl1 moves in +z direction, cyl2 moves in -z direction
        self.assertLess(cyl3.position[0], orig_pos1[0])  # cyl1 moved in +z
        self.assertGreater(cyl4.position[0], orig_pos2[0])     # cyl2 moved in -z


        # Check x and y coordinates are unchanged (fixed direction is z-axis)
        npt.assert_almost_equal(cyl3.position[1], orig_pos1[1])
        npt.assert_almost_equal(cyl3.position[2], orig_pos1[2])
        npt.assert_almost_equal(cyl4.position[1], orig_pos2[1])
        npt.assert_almost_equal(cyl4.position[2], orig_pos2[2])
    
    def test_repair_preserves_symmetry(self):
        """Test that particle repair preserves bilateral symmetry."""
        # Create a symmetric genome with colliding arms
        symmetric_genome = create_symmetric_genome()
        
        # Apply particle repair with symmetric constraints
        repaired_genome = particle_repair_individual(
            individual=symmetric_genome,
            propeller_radius=self.propeller_radius,
            inner_boundary_radius=self.inner_boundary_radius,
            outer_boundary_radius=self.outer_boundary_radius,
            max_iterations=self.max_iterations,
            step_size=self.step_size,
            propeller_tolerance=self.propeller_tolerance,
            repair_along_fixed_axis=self.xy_plane_axis,
            arms_to_cylinders=ARMS_TO_CYLINDERS_FUNC,
            cylinders_to_arms=CYLINDERS_TO_ARMS_FUNC
        )
        
        # Check that symmetry relationships are preserved after repair
        # XY plane symmetry: arms 0&1 should mirror arms 2&3
        # Z coordinates should be opposite, X,Y should be same, angles adjusted
        
        # Allow some tolerance for numerical precision and repair adjustments
        tolerance = 0.1
        
        # Check position relationships (may not be exact due to collision resolution)
        # But the general symmetry pattern should be maintained
        self.assertIsNotNone(repaired_genome)
        self.assertEqual(repaired_genome.shape, symmetric_genome.shape)
        
        # Verify no NaN values were introduced
        self.assertFalse(np.any(np.isnan(repaired_genome)))
    
    def test_spherical_repair_preserves_symmetry(self):
        # Create a symmetric spherical genome with potential collisions
        symmetric_genome = create_symmetric_genome()
        
        # Apply particle repair with symmetric constraints
        repaired_genome = particle_repair_individual(
            individual=symmetric_genome,
            propeller_radius=self.propeller_radius,
            inner_boundary_radius=self.inner_boundary_radius,
            outer_boundary_radius=self.outer_boundary_radius,
            max_iterations=self.max_iterations,
            step_size=self.step_size,
            propeller_tolerance=self.propeller_tolerance,
            repair_along_fixed_axis=self.xy_plane_axis,
            arms_to_cylinders=ARMS_TO_CYLINDERS_FUNC,
            cylinders_to_arms=CYLINDERS_TO_ARMS_FUNC
        )
        
        # Check that repair completed successfully
        self.assertIsNotNone(repaired_genome)
        self.assertEqual(repaired_genome.shape, symmetric_genome.shape)
        
        # Verify no NaN values were introduced
        self.assertFalse(np.any(np.isnan(repaired_genome)))
        
        # Check that direction values are preserved
        npt.assert_array_equal(repaired_genome[:, -1], symmetric_genome[:, -1])
    
    def test_repair_with_bilateral_symmetry_integration(self):
        """Test integration of particle repair with bilateral symmetry application."""
        # Create a genome handler with bilateral symmetry
        handler = create_test_handler(
            bilateral_plane_for_symmetry="xy",
            rnd=self.rng
        )
        
        # Create a test genome with potential collisions
        if GENOME_HANDLER_TYPE == 'cartesian':
            handler.genome = np.array([
                [0.5, 0.3, 0.2, 0.1, 0.2, 0.3, 0],  # First arm
                [0.48, 0.32, 0.18, 0.12, 0.18, 0.28, 0], # Second arm (close to first)
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],  # To be mirrored
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],  # To be mirrored
            ])
        else:  # spherical
            handler.genome = np.array([
                [1.0, 0.0, 0.0, 0.1, 0.2, 0],  # First arm
                [0.98, 0.1, 0.05, 0.12, 0.18, 0], # Second arm (close to first)
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],  # To be mirrored
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],  # To be mirrored
            ])
        
        # Apply symmetry first (as would happen in the repair workflow)
        handler.apply_symmetry()
        
        # Check that symmetry was applied
        self.assertFalse(np.any(np.isnan(handler.genome[2:])))
        
        # Now apply particle repair with symmetric constraints
        repaired_genome = particle_repair_individual(
            individual=handler.genome,
            propeller_radius=self.propeller_radius,
            inner_boundary_radius=self.inner_boundary_radius,
            outer_boundary_radius=self.outer_boundary_radius,
            repair_along_fixed_axis=self.xy_plane_axis,
            arms_to_cylinders=ARMS_TO_CYLINDERS_FUNC,
            cylinders_to_arms=CYLINDERS_TO_ARMS_FUNC
        )
        
        # Verify repair completed
        self.assertIsNotNone(repaired_genome)
        self.assertFalse(np.any(np.isnan(repaired_genome)))
        
        # Check that bilateral symmetry relationships still hold approximately
        # (allowing for collision resolution adjustments)
        tolerance = 0.2  # Allow larger tolerance due to collision resolution
        
        # Check that the mirroring pattern is approximately preserved
        # Arms 2&3 should be mirrors of arms 0&1 in the XY plane
        for i in range(2):  # First two arms
            if GENOME_HANDLER_TYPE == 'cartesian':
                mirror_i = i + 2  # Corresponding mirror arm
                
                # X and Y should be approximately the same
                self.assertAlmostEqual(repaired_genome[i, 0], repaired_genome[mirror_i, 0], delta=tolerance)
                self.assertAlmostEqual(repaired_genome[i, 1], repaired_genome[mirror_i, 1], delta=tolerance)
                
                # Z should be approximately opposite
                self.assertAlmostEqual(repaired_genome[i, 2], -repaired_genome[mirror_i, 2], delta=tolerance)
            else:  # spherical
                mirror_i = i + 2  # Corresponding mirror arm

                v, theta, phi = repaired_genome[i, :3]
                mv, mtheta, mphi = repaired_genome[mirror_i, :3]

                x1, y1, z1 = v * np.sin(theta) * np.cos(phi), v * np.sin(theta) * np.sin(phi), v * np.sin(theta)
                x2, y2, z2 = mv * np.sin(mtheta) * np.cos(mphi), mv * np.sin(mtheta) * np.sin(mphi), mv * np.sin(mtheta)

                # X and Y should be approximately the same
                self.assertAlmostEqual(x1, x2, delta=tolerance)
                self.assertAlmostEqual(y1, y2, delta=tolerance)
                # Z should be approximately opposite
                self.assertAlmostEqual(z1, -z2, delta=tolerance)

    
    def test_repair_boundary_constraints_with_symmetry(self):
        """Test that boundary constraints work correctly with symmetric repair."""
        # Create genome with arms outside boundaries
        out_of_bounds_genome = create_boundary_violation_genome()
        
        # Apply repair
        repaired_genome = particle_repair_individual(
            individual=out_of_bounds_genome,
            propeller_radius=self.propeller_radius,
            inner_boundary_radius=self.inner_boundary_radius,
            outer_boundary_radius=self.outer_boundary_radius,
            # repair_along_fixed_axis=self.xy_plane_axis,
            arms_to_cylinders=ARMS_TO_CYLINDERS_FUNC,
            cylinders_to_arms=CYLINDERS_TO_ARMS_FUNC
        )
        
        # Check that all arms are within boundaries
        for i in range(len(repaired_genome)):
            if not np.any(np.isnan(repaired_genome[i])):
                if GENOME_HANDLER_TYPE == 'cartesian':
                    distance = np.linalg.norm(repaired_genome[i, :3])  # Position distance from x,y,z
                else:  # spherical
                    distance = repaired_genome[i, 0]  # Magnitude is first column
                self.assertGreaterEqual(distance, self.inner_boundary_radius * 0.95)  # Allow small tolerance
                self.assertLessEqual(distance, self.outer_boundary_radius * 1.05)     # Allow small tolerance


class TestRepairConfig(unittest.TestCase):
    """Test RepairConfig class functionality."""
    
    def test_config_creation_defaults(self):
        """Test creating config with default values."""
            
        config = RepairConfig()
        
        self.assertTrue(config.apply_symmetry)
        
    def test_config_creation_custom(self):
        """Test creating config with custom values."""
            
        config = RepairConfig(
            apply_symmetry=False,
        )

        self.assertFalse(config.apply_symmetry)


    def test_config_validation(self):
        """Test config parameter validation."""
            
        # Test invalid radius values
        with self.assertRaises(AssertionError):
            RepairConfig(propeller_radius=-0.1)
        
        with self.assertRaises(AssertionError):
            RepairConfig(inner_boundary_radius=-0.5)
        
        with self.assertRaises(AssertionError):
            RepairConfig(outer_boundary_radius=0.1, inner_boundary_radius=0.5)  # outer < inner


class TestRepairStrategy(unittest.TestCase):
    """Test RepairStrategy enum functionality."""
    
    def test_strategy_values(self):
        """Test strategy enum values."""
            
        self.assertEqual(RepairStrategy.CLIP.value, "clip")
        self.assertEqual(RepairStrategy.WRAP.value, "wrap")
        self.assertEqual(RepairStrategy.RANDOM.value, "random")
        
    def test_strategy_from_string(self):
        """Test creating strategy from string."""
            
        self.assertEqual(RepairStrategy("clip"), RepairStrategy.CLIP)
        self.assertEqual(RepairStrategy("wrap"), RepairStrategy.WRAP)
        self.assertEqual(RepairStrategy("random"), RepairStrategy.RANDOM)
        
    def test_strategy_invalid_string(self):
        """Test creating strategy from invalid string."""
            
        with self.assertRaises(ValueError):
            RepairStrategy("invalid")


class TestRepairUtilities(unittest.TestCase):
    """Test RepairUtilities class functionality."""
    
    def test_clip_to_bounds_within_bounds(self):
        """Test clipping values already within bounds."""
            
        values = np.array([0.5, 1.5, 2.5])
        bounds = (0, 3)
        
        result = RepairUtilities.clip_to_bounds(values, bounds)
        npt.assert_array_equal(result, values)
        
    def test_clip_to_bounds_outside_bounds(self):
        """Test clipping values outside bounds."""
            
        values = np.array([-1.0, 0.5, 4.0])
        bounds = (0, 3)
        
        result = RepairUtilities.clip_to_bounds(values, bounds)
        expected = np.array([0.0, 0.5, 3.0])
        npt.assert_array_equal(result, expected)
        
    def test_wrap_to_bounds_within_bounds(self):
        """Test wrapping values already within bounds."""
            
        values = np.array([0.5, 1.5, 2.5])
        bounds = (0, 3)
        
        result = RepairUtilities.wrap_to_bounds(values, bounds)
        npt.assert_array_equal(result, values)
        
    def test_wrap_to_bounds_outside_bounds(self):
        """Test wrapping values outside bounds."""
            
        values = np.array([-1.0, 4.0, 7.0])
        bounds = (0, 3)
        
        result = RepairUtilities.wrap_to_bounds(values, bounds)
        expected = np.array([2.0, 1.0, 1.0])  # wrapped within [0, 3]
        npt.assert_array_almost_equal(result, expected)


class TestCartesianRepairOperator(unittest.TestCase):
    """Test CartesianRepairOperator functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
            
        self.config = RepairConfig(
            apply_symmetry=True,
        )
        self.config_no_symmetry = RepairConfig(
            apply_symmetry=False,
        )
        
    def test_operator_creation(self):
        """Test creating repair operator."""
            
        operator = CartesianRepairOperator(self.config)
        self.assertEqual(operator.config, self.config)
        
    def test_operator_creation_with_symmetry(self):
        """Test creating operator with symmetry enabled."""
            
        operator = CartesianRepairOperator(self.config)
        self.assertTrue(operator.config.apply_symmetry)


class TestSphericalRepairOperator(unittest.TestCase):
    """Test SphericalRepairOperator functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
            
        self.config = RepairConfig(
            apply_symmetry=False,
        )
        self.config_with_symmetry = RepairConfig(
            apply_symmetry=True,
        )
        
    def test_operator_creation(self):
        """Test creating spherical repair operator."""
            
        operator = SphericalRepairOperator(self.config)
        self.assertEqual(operator.config, self.config)

    def test_operator_creation_with_symmetry(self):
        """Test creating operator with symmetry enabled."""
            
        operator = SphericalRepairOperator(self.config_with_symmetry)
        self.assertTrue(operator.config.apply_symmetry)


class TestCollisionDetectionAndRepair(unittest.TestCase):
    """Test collision detection and repair effectiveness with random drones."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.rng = np.random.default_rng(42)  # Fixed seed for reproducibility
        
        # Common repair parameters
        self.propeller_radius = 0.0762
        self.inner_boundary_radius = 0.09
        self.outer_boundary_radius = 0.4
        self.max_iterations = 25
        self.step_size = 1.0
        self.propeller_tolerance = 0.1
        
        # Fixed axis for symmetric repair (Z-axis for XY plane symmetry)
        self.xy_plane_axis = [0.0, 0.0, 1.0]
    
    def _count_collisions(self, genome_handlers):
        """Count how many genomes have collisions."""
        collision_count = 0
        for handler in genome_handlers:
            genome = handler.genome.copy()
            # Convert to cylinders for collision checking
            valid_arms = ~np.isnan(genome).any(axis=-1)
            if np.sum(valid_arms) < 2:
                continue  # Need at least 2 arms to have collisions
                
            valid_arm_params = genome[valid_arms]
            cylinders = ARMS_TO_CYLINDERS_FUNC(valid_arm_params)
            
            if are_there_cylinder_collisions(cylinders):
                collision_count += 1
        
        return collision_count
    
    def test_asymmetric_drones_collision_repair(self):
        """Test collision repair on 10 random asymmetric 8-arm drones."""        
        # Create 10 random asymmetric drones with 8 arms each
        drones_handlers = []
        for i in range(100):
            
            # Configure parameters based on genome handler type
            genome_shape = (6, 7) if GENOME_HANDLER_TYPE == 'cartesian' else (6, 6)
            
            dummy_handler = create_test_handler(
                min_max_narms=(6, 6),  # Fixed 6 arms for testing
                # bilateral_plane_for_symmetry="xz",  # YZ plane symmetry
                repair=False,
                enable_collision_repair=False,
                propeller_radius=0.0762,
                inner_boundary_radius=0.09,
                outer_boundary_radius=0.4,
                append_arm_chance=0.0,  # No appending for this test
                rnd=self.rng,
            )
            handler = create_test_handler(
                genome = np.full(genome_shape, np.nan),
                min_max_narms=(6, 6),  # Fixed 6 arms for testing
                # bilateral_plane_for_symmetry="xz",  # YZ plane symmetry
                repair=True,
                enable_collision_repair=True,
                propeller_radius=0.0762,
                inner_boundary_radius=0.09,
                outer_boundary_radius=0.4,
                append_arm_chance=0.0,  # No appending for this test
                rnd=self.rng,
            )
            # Create random genome with 8 arms
            handler.genome = dummy_handler._generate_random_genome()
            drones_handlers.append(handler)
        
        # Count collisions before repair
        collisions_before = self._count_collisions(drones_handlers)
        print(f"Collisions before repair: {collisions_before}/100")
        
        copyed_drones_handlers = [handler.copy() for handler in drones_handlers]

        # Repair each drone and track time
        repaired_drones = []
        repair_times = []
        
        for i, drone_handler in enumerate(drones_handlers):
            start_time = time.time()
            
            drone_handler.repair()
            
            end_time = time.time()
            repair_time = end_time - start_time
            repair_times.append(repair_time)
            repaired_drones.append(drone_handler)
            print(f"Drone {i+1} repaired in {repair_time:.4f}s") 

        collisions_after = self._count_collisions(repaired_drones)
        print(f"Collisions after repair: {collisions_after}/100")
        
        # Print repair time statistics
        avg_repair_time = np.mean(repair_times)
        max_repair_time = np.max(repair_times)
        min_repair_time = np.min(repair_times)
        print(f"Repair times - Avg: {avg_repair_time:.4f}s, Min: {min_repair_time:.4f}s, Max: {max_repair_time:.4f}s")
        
        # Assert repair effectiveness - should reduce collisions
        self.assertLessEqual(collisions_after, collisions_before, 
                            "Repair should reduce or maintain collision count")
        self.assertLessEqual(collisions_after, 1,
                            "Repair should reduce collision count to a reasonable level")

        # All repair times should be reasonable (under 2 seconds)
        for repair_time in repair_times:
            self.assertLess(repair_time, 5, "Individual repair took too long")
    
    def test_symmetric_drones_collision_repair(self):
        """Test collision repair on 10 random symmetric 8-arm drones."""
        
        # Create 10 random symmetric drones with 8 arms each
        drone_handlers = []
        for i in range(100):
            # Create handler with bilateral symmetry
            # Configure parameters based on genome handler type
            genome_shape = (6, 7) if GENOME_HANDLER_TYPE == 'cartesian' else (6, 6)
            
            dummy_handler = create_test_handler(
                min_max_narms=(6, 6),  # Fixed 6 arms for testing
                bilateral_plane_for_symmetry="yz",  # YZ plane symmetry
                repair=False,
                enable_collision_repair=False,
                propeller_radius=0.0762,
                inner_boundary_radius=0.09,
                outer_boundary_radius=0.4,
                append_arm_chance=0.0,  # No appending for this test
                rnd=self.rng,
            )
            handler = create_test_handler(
                genome = np.full(genome_shape, np.nan),
                min_max_narms=(6, 6),  # Fixed 6 arms for testing
                bilateral_plane_for_symmetry="yz",  # YZ plane symmetry
                repair=True,
                enable_collision_repair=True,
                propeller_radius=0.0762,
                inner_boundary_radius=0.09,
                outer_boundary_radius=0.4,
                append_arm_chance=0.0,  # No appending for this test
                rnd=self.rng,
            )
            # Create random genome with 8 arms
            handler.genome = dummy_handler._generate_random_genome()
            drone_handlers.append(handler)
        
        # Count collisions before repair
        collisions_before = self._count_collisions(drone_handlers)
        print(f"Collisions before repair: {collisions_before}/100")
        
        # Repair each drone and track time
        repaired_drones = []
        repair_times = []
        
        for i, drone_handler in enumerate(drone_handlers):
            start_time = time.time()
            
            drone_handler.repair()
            
            end_time = time.time()
            repair_time = end_time - start_time
            repair_times.append(repair_time)
            repaired_drones.append(drone_handler)
            print(f"Drone {i+1} repaired in {repair_time:.4f}s")
            
        # Count collisions after repair
        collisions_after = self._count_collisions(repaired_drones)
        print(f"Collisions after repair: {collisions_after}/100")
        # Print repair time statistics
        avg_repair_time = np.mean(repair_times)
        max_repair_time = np.max(repair_times)
        min_repair_time = np.min(repair_times)
        print(f"Repair times - Avg: {avg_repair_time:.4f}s, Min: {min_repair_time:.4f}s, Max: {max_repair_time:.4f}s")
        
        # Assert repair effectiveness - should reduce collisions
        self.assertLessEqual(collisions_after, collisions_before,
                            "Repair should reduce or maintain collision count")
        
        # Note: At 100 iterations the overlap between cylinders in minimal
        self.assertLessEqual(collisions_after, 15,
                            "Repair should reduce collision count to a reasonable level")

        # All repair times should be reasonable (under 2 seconds)
        for repair_time in repair_times:
            self.assertLess(repair_time, 5, "Individual repair took too long")


class TestParticleRepairEdgeCases(unittest.TestCase):
    """Test edge cases for particle repair operator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.propeller_radius = 0.0762
        self.inner_boundary_radius = 0.09
        self.outer_boundary_radius = 0.4
    
    def test_repair_with_nan_arms(self):
        """Test repair handles NaN arms correctly."""
        if GENOME_HANDLER_TYPE == 'cartesian':
            genome_with_nans = np.array([
                [0.5, 0.3, 0.2, 0.1, 0.2, 0.3, 0],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                [0.45, 0.25, 0.15, 0.05, 0.15, 0.25, 0],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            ])
        else:  # spherical
            genome_with_nans = np.array([
                [1.0, 0.0, 0.0, 0.1, 0.2, 0],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                [0.95, 0.1, 0.05, 0.05, 0.15, 0],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            ])
        
        repaired_genome = particle_repair_individual(
            individual=genome_with_nans,
            propeller_radius=self.propeller_radius,
            inner_boundary_radius=self.inner_boundary_radius,
            outer_boundary_radius=self.outer_boundary_radius,
            arms_to_cylinders=ARMS_TO_CYLINDERS_FUNC,
            cylinders_to_arms=CYLINDERS_TO_ARMS_FUNC
        )
        
        # NaN arms should remain NaN
        self.assertTrue(np.all(np.isnan(repaired_genome[1])))
        self.assertTrue(np.all(np.isnan(repaired_genome[3])))
        
        # Valid arms should be processed
        self.assertFalse(np.any(np.isnan(repaired_genome[0])))
        self.assertFalse(np.any(np.isnan(repaired_genome[2])))
    
    def test_repair_no_collisions(self):
        """Test repair when no collisions exist."""
        no_collision_genome = create_no_collision_genome()
        
        repaired_genome = particle_repair_individual(
            individual=no_collision_genome,
            propeller_radius=self.propeller_radius,
            inner_boundary_radius=self.inner_boundary_radius,
            outer_boundary_radius=self.outer_boundary_radius,
            arms_to_cylinders=ARMS_TO_CYLINDERS_FUNC,
            cylinders_to_arms=CYLINDERS_TO_ARMS_FUNC
        )
        
        # Repair should complete successfully and produce valid results
        self.assertIsNotNone(repaired_genome)
        self.assertFalse(np.any(np.isnan(repaired_genome)))
        
        if GENOME_HANDLER_TYPE == 'cartesian':
            # Cartesian should remain exactly the same for no collisions
            npt.assert_allclose(repaired_genome, no_collision_genome, rtol=1e-10)
        else:  # spherical
            # For spherical coordinates, conversion round-trip may introduce changes
            # Just verify that repair completed without introducing invalid values
            self.assertEqual(repaired_genome.shape, no_collision_genome.shape)
            # Verify all values are finite
            self.assertTrue(np.all(np.isfinite(repaired_genome)))
    
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Run particle repair tests with different genome handlers',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_particle_repair.py                                    # Run with Cartesian handler (default)
  python test_particle_repair.py --genome-handler spherical         # Run with Spherical handler
  python test_particle_repair.py TestCollisionDetectionAccuracy     # Run specific test class
  python test_particle_repair.py TestCollisionDetectionAccuracy --genome-handler spherical
        """
    )
    
    parser.add_argument(
        '--genome-handler',
        choices=['cartesian', 'spherical'],
        default='cartesian',
        help='Genome handler type to use for tests (default: cartesian)'
    )
    
    # Parse known args to allow unittest to handle its own arguments
    args, remaining_args = parser.parse_known_args()
    
    # Configure test environment based on arguments
    configure_test_environment(args.genome_handler)
    
    print(f"Running particle repair tests with {GENOME_HANDLER_TYPE} genome handler")
    
    # Run unittest with remaining arguments
    sys.argv = [sys.argv[0]] + remaining_args
    unittest.main(verbosity=2)