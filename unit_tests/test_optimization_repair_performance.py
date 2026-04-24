#!/usr/bin/env python3
"""
Performance tests for optimization repair operator.

This module tests the performance of the optimization-based repair operator
and compares it with the particle repair operator.

Tests include:
- Repair time benchmarking
- Collision resolution effectiveness
- Scalability with arm count
- Comparison with particle repair operator

Supports both Cartesian and Spherical genome handlers via --genome-handler argument.
"""

import unittest
import numpy as np
import numpy.testing as npt
import time
import argparse
import sys
from typing import List, Tuple

from airevolve.evolution_tools.genome_handlers.operators.optimization_repair_operator import (
    optimization_repair_individual,
    OptimizationRepairConfig,
    genome_to_arm_cylinders_with_disc_base,
    _check_collisions
)
from airevolve.evolution_tools.genome_handlers.operators.particle_repair_operator import (
    particle_repair_individual,
    are_there_cylinder_collisions
)
from airevolve.evolution_tools.genome_handlers.conversions.arm_conversions import (
    arms_to_cylinders_cartesian_euler,
    cylinders_to_arms_cartesian_euler,
    arms_to_cylinders_polar_angular,
    cylinders_to_arms_polar_angular,
)

# Import both genome handlers
from airevolve.evolution_tools.genome_handlers.cartesian_euler_genome_handler import CartesianEulerDroneGenomeHandler
from airevolve.evolution_tools.genome_handlers.spherical_angular_genome_handler import SphericalAngularDroneGenomeHandler

# Global configuration variables
GENOME_HANDLER_TYPE = 'cartesian'
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


def create_test_handler(**kwargs):
    """Create appropriate genome handler based on current type."""
    return GENOME_HANDLER_CLASS(**kwargs)


def count_collisions_optimization(genome_handlers, config):
    """Count how many genomes have collisions using optimization check."""
    collision_count = 0
    for handler in genome_handlers:
        genome = handler.genome.copy()
        valid_arms = ~np.isnan(genome).any(axis=-1)
        if np.sum(valid_arms) < 2:
            continue

        valid_genome = genome[valid_arms]
        arm_cylinders = genome_to_arm_cylinders_with_disc_base(
            valid_genome, config, ARMS_TO_CYLINDERS_FUNC
        )

        if _check_collisions(arm_cylinders, config):
            collision_count += 1

    return collision_count


def count_collisions_particle(genome_handlers):
    """Count how many genomes have collisions using particle check."""
    collision_count = 0
    for handler in genome_handlers:
        genome = handler.genome.copy()
        valid_arms = ~np.isnan(genome).any(axis=-1)
        if np.sum(valid_arms) < 2:
            continue

        valid_genome = genome[valid_arms]
        cylinders = ARMS_TO_CYLINDERS_FUNC(valid_genome)

        if are_there_cylinder_collisions(cylinders):
            collision_count += 1

    return collision_count


def calculate_genome_edit_distance(genome_before, genome_after):
    """
    Calculate edit distance between two genomes.

    Uses Euclidean distance on valid (non-NaN) arm parameters.
    Only considers arms that are valid in both genomes.

    Args:
        genome_before: Genome array before repair
        genome_after: Genome array after repair

    Returns:
        float: Euclidean distance between the genomes
    """
    # Find arms that are valid in both genomes
    valid_before = ~np.isnan(genome_before).any(axis=-1)
    valid_after = ~np.isnan(genome_after).any(axis=-1)
    valid_both = valid_before & valid_after

    if not np.any(valid_both):
        # If no arms are valid in both, return NaN
        return np.nan

    # Calculate Euclidean distance on valid arms
    diff = genome_before[valid_both] - genome_after[valid_both]
    distance = np.sqrt(np.sum(diff ** 2))

    return distance


class TestOptimizationRepairPerformance(unittest.TestCase):
    """Test performance of optimization repair operator."""

    def setUp(self):
        """Set up test fixtures."""
        self.rng = np.random.default_rng(42)

        # Optimization repair configuration
        self.opt_config = OptimizationRepairConfig(
            propeller_radius=0.0254,
            propeller_tolerance=0.1,
            inner_boundary_radius=0.09,
            outer_boundary_radius=0.4,
            max_iterations=1000,
            optimization_method='SLSQP',
        )

        # Particle repair parameters
        self.particle_config = {
            'propeller_radius': 0.0254,
            'inner_boundary_radius': 0.09,
            'outer_boundary_radius': 0.4,
            'max_iterations': 25,
            'step_size': 1.0,
            'propeller_tolerance': 0.1,
        }

    def test_optimization_repair_asymmetric_drones(self):
        """Test optimization repair on 100 random asymmetric drones."""
        print(f"\n{'='*80}")
        print(f"Testing Optimization Repair on Asymmetric Drones ({GENOME_HANDLER_TYPE})")
        print(f"{'='*80}")

        # Create 100 random asymmetric drones
        drone_handlers = []
        for i in range(100):
            genome_shape = (6, 7) if GENOME_HANDLER_TYPE == 'cartesian' else (6, 6)

            dummy_handler = create_test_handler(
                min_max_narms=(6, 6),
                repair=False,
                enable_collision_repair=False,
                propeller_radius=0.0254,
                inner_boundary_radius=0.09,
                outer_boundary_radius=0.4,
                append_arm_chance=0.0,
                rnd=self.rng,
            )

            handler = create_test_handler(
                genome=np.full(genome_shape, np.nan),
                min_max_narms=(6, 6),
                repair=False,
                enable_collision_repair=False,
                propeller_radius=0.0254,
                inner_boundary_radius=0.09,
                outer_boundary_radius=0.4,
                append_arm_chance=0.0,
                rnd=self.rng,
            )

            handler.genome = dummy_handler._generate_random_genome()
            drone_handlers.append(handler)

        # Count collisions before repair
        collisions_before = count_collisions_optimization(drone_handlers, self.opt_config)
        print(f"\nCollisions before repair: {collisions_before}/100")

        # Repair each drone and track time and edit distance
        repaired_drones = []
        repair_times = []
        edit_distances = []
        success_count = 0

        for i, drone_handler in enumerate(drone_handlers):
            genome = drone_handler.genome.copy()

            start_time = time.time()
            repaired_genome = optimization_repair_individual(
                genome,
                self.opt_config,
                ARMS_TO_CYLINDERS_FUNC,
                CYLINDERS_TO_ARMS_FUNC,
                verbose=False
            )
            end_time = time.time()

            repair_time = end_time - start_time
            repair_times.append(repair_time)

            # Calculate edit distance
            edit_distance = calculate_genome_edit_distance(genome, repaired_genome)
            edit_distances.append(edit_distance)

            # Check if repair was successful
            valid_arms = ~np.isnan(repaired_genome).any(axis=-1)
            if np.sum(valid_arms) >= 2:
                valid_genome = repaired_genome[valid_arms]
                arm_cylinders = genome_to_arm_cylinders_with_disc_base(
                    valid_genome, self.opt_config, ARMS_TO_CYLINDERS_FUNC
                )
                if not _check_collisions(arm_cylinders, self.opt_config):
                    success_count += 1

            drone_handler.genome = repaired_genome
            repaired_drones.append(drone_handler)

            if (i + 1) % 10 == 0:
                print(f"Repaired {i+1}/100 drones (avg time: {np.mean(repair_times[-10:]):.4f}s)")

        # Count collisions after repair
        collisions_after = count_collisions_optimization(repaired_drones, self.opt_config)
        print(f"\nCollisions after repair: {collisions_after}/100")
        print(f"Successful repairs (collision-free): {success_count}/100")

        # Print repair time statistics
        avg_repair_time = np.mean(repair_times)
        max_repair_time = np.max(repair_times)
        min_repair_time = np.min(repair_times)
        std_repair_time = np.std(repair_times)

        print(f"\nRepair time statistics:")
        print(f"  Mean:   {avg_repair_time:.4f}s")
        print(f"  Median: {np.median(repair_times):.4f}s")
        print(f"  Std:    {std_repair_time:.4f}s")
        print(f"  Min:    {min_repair_time:.4f}s")
        print(f"  Max:    {max_repair_time:.4f}s")

        # Print edit distance statistics
        valid_edit_distances = [d for d in edit_distances if not np.isnan(d)]
        if valid_edit_distances:
            print(f"\nEdit distance statistics:")
            print(f"  Mean:   {np.mean(valid_edit_distances):.4f}")
            print(f"  Median: {np.median(valid_edit_distances):.4f}")
            print(f"  Std:    {np.std(valid_edit_distances):.4f}")
            print(f"  Min:    {np.min(valid_edit_distances):.4f}")
            print(f"  Max:    {np.max(valid_edit_distances):.4f}")

        # Assert repair effectiveness
        self.assertLessEqual(collisions_after, collisions_before,
                            "Repair should reduce or maintain collision count")

    def test_optimization_repair_symmetric_drones(self):
        """Test optimization repair on 100 random symmetric drones."""
        print(f"\n{'='*80}")
        print(f"Testing Optimization Repair on Symmetric Drones ({GENOME_HANDLER_TYPE})")
        print(f"{'='*80}")

        # Create 100 random symmetric drones
        drone_handlers = []
        for i in range(100):
            genome_shape = (6, 7) if GENOME_HANDLER_TYPE == 'cartesian' else (6, 6)

            dummy_handler = create_test_handler(
                min_max_narms=(6, 6),
                bilateral_plane_for_symmetry="yz",
                repair=False,
                enable_collision_repair=False,
                propeller_radius=0.0254,
                inner_boundary_radius=0.09,
                outer_boundary_radius=0.4,
                append_arm_chance=0.0,
                rnd=self.rng,
            )

            handler = create_test_handler(
                genome=np.full(genome_shape, np.nan),
                min_max_narms=(6, 6),
                bilateral_plane_for_symmetry="yz",
                repair=False,
                enable_collision_repair=False,
                propeller_radius=0.0254,
                inner_boundary_radius=0.09,
                outer_boundary_radius=0.4,
                append_arm_chance=0.0,
                rnd=self.rng,
            )

            handler.genome = dummy_handler._generate_random_genome()
            drone_handlers.append(handler)

        # Count collisions before repair
        collisions_before = count_collisions_optimization(drone_handlers, self.opt_config)
        print(f"\nCollisions before repair: {collisions_before}/100")

        # Repair each drone and track time and edit distance
        repaired_drones = []
        repair_times = []
        edit_distances = []
        success_count = 0

        for i, drone_handler in enumerate(drone_handlers):
            genome = drone_handler.genome.copy()

            start_time = time.time()
            repaired_genome = optimization_repair_individual(
                genome,
                self.opt_config,
                ARMS_TO_CYLINDERS_FUNC,
                CYLINDERS_TO_ARMS_FUNC,
                verbose=False
            )
            end_time = time.time()

            repair_time = end_time - start_time
            repair_times.append(repair_time)

            # Calculate edit distance
            edit_distance = calculate_genome_edit_distance(genome, repaired_genome)
            edit_distances.append(edit_distance)

            # Check if repair was successful
            valid_arms = ~np.isnan(repaired_genome).any(axis=-1)
            if np.sum(valid_arms) >= 2:
                valid_genome = repaired_genome[valid_arms]
                arm_cylinders = genome_to_arm_cylinders_with_disc_base(
                    valid_genome, self.opt_config, ARMS_TO_CYLINDERS_FUNC
                )
                if not _check_collisions(arm_cylinders, self.opt_config):
                    success_count += 1

            drone_handler.genome = repaired_genome
            repaired_drones.append(drone_handler)

            if (i + 1) % 10 == 0:
                print(f"Repaired {i+1}/100 drones (avg time: {np.mean(repair_times[-10:]):.4f}s)")

        # Count collisions after repair
        collisions_after = count_collisions_optimization(repaired_drones, self.opt_config)
        print(f"\nCollisions after repair: {collisions_after}/100")
        print(f"Successful repairs (collision-free): {success_count}/100")

        # Print repair time statistics
        avg_repair_time = np.mean(repair_times)
        max_repair_time = np.max(repair_times)
        min_repair_time = np.min(repair_times)
        std_repair_time = np.std(repair_times)

        print(f"\nRepair time statistics:")
        print(f"  Mean:   {avg_repair_time:.4f}s")
        print(f"  Median: {np.median(repair_times):.4f}s")
        print(f"  Std:    {std_repair_time:.4f}s")
        print(f"  Min:    {min_repair_time:.4f}s")
        print(f"  Max:    {max_repair_time:.4f}s")

        # Print edit distance statistics
        valid_edit_distances = [d for d in edit_distances if not np.isnan(d)]
        if valid_edit_distances:
            print(f"\nEdit distance statistics:")
            print(f"  Mean:   {np.mean(valid_edit_distances):.4f}")
            print(f"  Median: {np.median(valid_edit_distances):.4f}")
            print(f"  Std:    {np.std(valid_edit_distances):.4f}")
            print(f"  Min:    {np.min(valid_edit_distances):.4f}")
            print(f"  Max:    {np.max(valid_edit_distances):.4f}")

        # Assert repair effectiveness
        self.assertLessEqual(collisions_after, collisions_before,
                            "Repair should reduce or maintain collision count")


class TestComparisonWithParticleRepair(unittest.TestCase):
    """Compare optimization repair with particle repair operator."""

    def setUp(self):
        """Set up test fixtures."""
        self.rng = np.random.default_rng(42)

        # Optimization repair configuration
        self.opt_config = OptimizationRepairConfig(
            propeller_radius=0.0254,
            propeller_tolerance=0.1,
            inner_boundary_radius=0.09,
            outer_boundary_radius=0.4,
            max_iterations=1000,
            optimization_method='SLSQP',
        )

        # Particle repair parameters
        self.particle_config = {
            'propeller_radius': 0.0254,
            'inner_boundary_radius': 0.09,
            'outer_boundary_radius': 0.4,
            'max_iterations': 25,
            'step_size': 1.0,
            'propeller_tolerance': 0.1,
        }

    def test_compare_repair_methods(self):
        """Compare optimization vs particle repair on same test set."""
        print(f"\n{'='*80}")
        print(f"Comparing Optimization vs Particle Repair ({GENOME_HANDLER_TYPE})")
        print(f"{'='*80}")

        # Create 50 random test drones
        test_drones = []
        for i in range(50):
            genome_shape = (6, 7) if GENOME_HANDLER_TYPE == 'cartesian' else (6, 6)

            dummy_handler = create_test_handler(
                min_max_narms=(6, 6),
                repair=False,
                enable_collision_repair=False,
                propeller_radius=0.0254,
                inner_boundary_radius=0.09,
                outer_boundary_radius=0.4,
                append_arm_chance=0.0,
                rnd=self.rng,
            )

            handler = create_test_handler(
                genome=np.full(genome_shape, np.nan),
                min_max_narms=(6, 6),
                repair=False,
                enable_collision_repair=False,
                propeller_radius=0.0254,
                inner_boundary_radius=0.09,
                outer_boundary_radius=0.4,
                append_arm_chance=0.0,
                rnd=self.rng,
            )

            handler.genome = dummy_handler._generate_random_genome()
            test_drones.append(handler.genome.copy())

        print(f"\nCreated {len(test_drones)} test drones\n")

        # Test optimization repair
        print("Testing OPTIMIZATION REPAIR...")
        opt_times = []
        opt_edit_distances = []
        opt_successes = 0
        opt_collision_reductions = []

        for i, genome in enumerate(test_drones):
            # Check initial collisions
            valid_arms = ~np.isnan(genome).any(axis=-1)
            if np.sum(valid_arms) < 2:
                continue

            valid_genome = genome[valid_arms]
            arm_cylinders_before = genome_to_arm_cylinders_with_disc_base(
                valid_genome, self.opt_config, ARMS_TO_CYLINDERS_FUNC
            )
            had_collision = _check_collisions(arm_cylinders_before, self.opt_config)

            # Repair
            start_time = time.time()
            repaired = optimization_repair_individual(
                genome.copy(),
                self.opt_config,
                ARMS_TO_CYLINDERS_FUNC,
                CYLINDERS_TO_ARMS_FUNC,
                verbose=False
            )
            opt_times.append(time.time() - start_time)

            # Calculate edit distance
            edit_distance = calculate_genome_edit_distance(genome, repaired)
            opt_edit_distances.append(edit_distance)

            # Check final collisions
            valid_arms_after = ~np.isnan(repaired).any(axis=-1)
            if np.sum(valid_arms_after) >= 2:
                valid_genome_after = repaired[valid_arms_after]
                arm_cylinders_after = genome_to_arm_cylinders_with_disc_base(
                    valid_genome_after, self.opt_config, ARMS_TO_CYLINDERS_FUNC
                )
                has_collision = _check_collisions(arm_cylinders_after, self.opt_config)

                if not has_collision:
                    opt_successes += 1
                if had_collision and not has_collision:
                    opt_collision_reductions.append(1)

        print(f"  Optimization repair complete:")
        print(f"    Mean time: {np.mean(opt_times):.4f}s")
        print(f"    Success rate: {opt_successes}/{len(test_drones)} ({100*opt_successes/len(test_drones):.1f}%)")
        print(f"    Collision reductions: {len(opt_collision_reductions)}")

        # Test particle repair
        print("\nTesting PARTICLE REPAIR...")
        particle_times = []
        particle_edit_distances = []
        particle_successes = 0
        particle_collision_reductions = []

        for i, genome in enumerate(test_drones):
            # Check initial collisions
            valid_arms = ~np.isnan(genome).any(axis=-1)
            if np.sum(valid_arms) < 2:
                continue

            valid_genome = genome[valid_arms]
            cylinders_before = ARMS_TO_CYLINDERS_FUNC(valid_genome)
            had_collision = are_there_cylinder_collisions(cylinders_before)

            # Repair
            start_time = time.time()
            repaired = particle_repair_individual(
                genome.copy(),
                arms_to_cylinders=ARMS_TO_CYLINDERS_FUNC,
                cylinders_to_arms=CYLINDERS_TO_ARMS_FUNC,
                **self.particle_config
            )
            particle_times.append(time.time() - start_time)

            # Calculate edit distance
            edit_distance = calculate_genome_edit_distance(genome, repaired)
            particle_edit_distances.append(edit_distance)

            # Check final collisions
            valid_arms_after = ~np.isnan(repaired).any(axis=-1)
            if np.sum(valid_arms_after) >= 2:
                valid_genome_after = repaired[valid_arms_after]
                cylinders_after = ARMS_TO_CYLINDERS_FUNC(valid_genome_after)
                has_collision = are_there_cylinder_collisions(cylinders_after)

                if not has_collision:
                    particle_successes += 1
                if had_collision and not has_collision:
                    particle_collision_reductions.append(1)

        print(f"  Particle repair complete:")
        print(f"    Mean time: {np.mean(particle_times):.4f}s")
        print(f"    Success rate: {particle_successes}/{len(test_drones)} ({100*particle_successes/len(test_drones):.1f}%)")
        print(f"    Collision reductions: {len(particle_collision_reductions)}")

        # Compare results
        print(f"\n{'='*80}")
        print("COMPARISON SUMMARY:")
        print(f"{'='*80}")
        print(f"Speed:")
        print(f"  Optimization: {np.mean(opt_times):.4f}s ± {np.std(opt_times):.4f}s")
        print(f"  Particle:     {np.mean(particle_times):.4f}s ± {np.std(particle_times):.4f}s")
        print(f"  Speedup:      {np.mean(opt_times)/np.mean(particle_times):.2f}x {'(Particle faster)' if np.mean(opt_times) > np.mean(particle_times) else '(Optimization faster)'}")

        print(f"\nEffectiveness:")
        print(f"  Optimization success rate: {100*opt_successes/len(test_drones):.1f}%")
        print(f"  Particle success rate:     {100*particle_successes/len(test_drones):.1f}%")
        print(f"  Difference:                {100*(opt_successes - particle_successes)/len(test_drones):.1f}%")

        # Edit distance comparison
        valid_opt_edit_distances = [d for d in opt_edit_distances if not np.isnan(d)]
        valid_particle_edit_distances = [d for d in particle_edit_distances if not np.isnan(d)]

        if valid_opt_edit_distances and valid_particle_edit_distances:
            print(f"\nEdit Distance (genome modification):")
            print(f"  Optimization mean: {np.mean(valid_opt_edit_distances):.4f} ± {np.std(valid_opt_edit_distances):.4f}")
            print(f"  Particle mean:     {np.mean(valid_particle_edit_distances):.4f} ± {np.std(valid_particle_edit_distances):.4f}")
            print(f"  Ratio:             {np.mean(valid_opt_edit_distances)/np.mean(valid_particle_edit_distances):.2f}x {'(Optimization modifies more)' if np.mean(valid_opt_edit_distances) > np.mean(valid_particle_edit_distances) else '(Particle modifies more)'}")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Run optimization repair performance tests',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_optimization_repair_performance.py                        # Run with Cartesian handler (default)
  python test_optimization_repair_performance.py --genome-handler spherical  # Run with Spherical handler
        """
    )

    parser.add_argument(
        '--genome-handler',
        choices=['cartesian', 'spherical'],
        default='spherical',
        help='Genome handler type to use for tests (default: cartesian)'
    )

    # Parse known args
    args, remaining_args = parser.parse_known_args()

    # Configure test environment
    configure_test_environment(args.genome_handler)

    print(f"Running optimization repair tests with {GENOME_HANDLER_TYPE} genome handler")

    # Run unittest
    sys.argv = [sys.argv[0]] + remaining_args
    unittest.main(verbosity=2)
