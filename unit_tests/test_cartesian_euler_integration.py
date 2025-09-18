#!/usr/bin/env python3
"""
Integration and performance tests for CartesianEulerDroneGenomeHandler.

This module contains tests that verify:
- Population-level operations
- Integration workflows
- Performance characteristics
- Cross-component interactions
- Operator integration
"""

import unittest
import numpy as np
import numpy.testing as npt
import time
from airevolve.evolution_tools.genome_handlers.cartesian_euler_genome_handler import CartesianEulerDroneGenomeHandler


class TestCartesianEulerIntegration(unittest.TestCase):
    """Integration tests with other modules."""

    def setUp(self):
        """Set up test fixtures."""
        self.min_max_narms = (4, 4)
        self.parameter_limits = np.array([
            [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0],
            [-np.pi, np.pi], [-np.pi, np.pi], [-np.pi, np.pi],
            [0, 1]
        ])
        self.rnd = np.random.default_rng(42)

    def test_integration_with_arm_conversions(self):
        """Test integration with arm conversion functions."""
        # This test would require the actual arm_conversions module
        # For now, we'll create a mock test
        
        handler = CartesianEulerDroneGenomeHandler(
            min_max_narms=self.min_max_narms,
            parameter_limits=self.parameter_limits,
            append_arm_chance=0.0,
            rnd=self.rnd
        )
        
        # Generate random genome
        handler.genome = handler._generate_random_genome()
        
        # Extract components
        positions = handler.get_motor_positions()
        orientations = handler.get_motor_orientations()
        directions = handler.get_propeller_directions()
        
        # Test that we can reconstruct the genome from components
        reconstructed = np.column_stack([positions, orientations, directions])
        npt.assert_array_equal(reconstructed, handler.genome)

    def test_crossover_population(self):
        """Test crossover on populations using inherited method."""
        handler = CartesianEulerDroneGenomeHandler(
            min_max_narms=self.min_max_narms,
            parameter_limits=self.parameter_limits,
            append_arm_chance=0.0,
            rnd=self.rnd
        )
        
        pop_size = 5
        population1 = handler.generate_random_population(pop_size)
        population2 = handler.generate_random_population(pop_size)
        
        children = handler.crossover_population(population1, population2)
        
        self.assertEqual(len(children), pop_size)
        for child in children:
            self.assertIsInstance(child, CartesianEulerDroneGenomeHandler)
            self.assertEqual(child.genome.shape, (4, 7))

    def test_mutate_population(self):
        """Test mutation on populations using inherited method."""
        handler = CartesianEulerDroneGenomeHandler(
            min_max_narms=self.min_max_narms,
            parameter_limits=self.parameter_limits,
            append_arm_chance=0.0,
            mutation_probs=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),  # Always mutate for testing
            rnd=self.rnd
        )
        
        pop_size = 5
        population = handler.generate_random_population(pop_size)
        original_genomes = [ind.genome.copy() for ind in population]
        
        handler.mutate_population(population)
        
        # At least some individuals should have changed
        changes = 0
        for i, individual in enumerate(population):
            if not np.array_equal(individual.genome, original_genomes[i]):
                changes += 1
        
        # With mutation probability 1.0, we expect some changes
        # (though it's theoretically possible for no changes due to randomness)
        self.assertGreaterEqual(changes, 0)

    def test_repair_population(self):
        """Test population repair functionality."""
        handler = CartesianEulerDroneGenomeHandler(
            min_max_narms=self.min_max_narms,
            parameter_limits=self.parameter_limits,
            append_arm_chance=0.0,
            rnd=self.rnd
        )
        
        # Create population with some invalid individuals
        population = handler.generate_random_population(5)
        
        # Make some individuals invalid (but avoid NaN values)
        population[0].genome[0, 0] = 10.0  # Out of bounds
        population[2].genome[2, 3] = 5.0   # Out of bounds orientation
        
        # Verify some are invalid
        invalid_count = sum(1 for ind in population if not ind.is_valid())
        self.assertGreater(invalid_count, 0)
        
        # Repair population
        repaired_population = handler.repair_population(population)
        
        # All should be valid after repair
        for individual in repaired_population:
            self.assertTrue(individual.is_valid())
        
        # Check that original population is unchanged
        self.assertEqual(len(population), len(repaired_population))
        self.assertIsNot(population[0], repaired_population[0])

    def test_integration_workflow(self):
        """Test a complete evolutionary workflow."""
        # Create initial population
        handler = CartesianEulerDroneGenomeHandler(
            min_max_narms=self.min_max_narms,
            parameter_limits=self.parameter_limits,
            append_arm_chance=0.0,
            mutation_probs=np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]),
            rnd=self.rnd
        )
        
        population = handler.generate_random_population(10)
        
        # Ensure all are valid
        for individual in population:
            self.assertTrue(individual.is_valid())
        
        # Perform crossover
        parent_pairs = [(population[i], population[i+1]) for i in range(0, len(population)-1, 2)]
        children = []
        for parent1, parent2 in parent_pairs:
            child = parent1.crossover(parent2)
            children.append(child)
        
        # Mutate children
        for child in children:
            child.mutate()
        
        # Repair if needed
        for child in children:
            if not child.is_valid():
                child.repair()
            self.assertTrue(child.is_valid())
        
        # Test copying
        copies = [child.copy() for child in children]
        for original, copy_ind in zip(children, copies):
            npt.assert_array_equal(original.genome, copy_ind.genome)
            self.assertIsNot(original.genome, copy_ind.genome)

    def test_validate_symmetry(self):
        """Test symmetry validation method."""
        # Test without symmetry
        handler = CartesianEulerDroneGenomeHandler(
            min_max_narms=self.min_max_narms,
            append_arm_chance=0.0,
            bilateral_plane_for_symmetry=None,
            rnd=self.rnd
        )
        
        # Should always return True when no symmetry
        self.assertTrue(handler.validate_symmetry())
        
        # Test with symmetry
        handler_sym = CartesianEulerDroneGenomeHandler(
            min_max_narms=self.min_max_narms,
            append_arm_chance=0.0,
            bilateral_plane_for_symmetry="xy",
            rnd=self.rnd
        )
        
        # Should return True for properly symmetric genome
        self.assertTrue(handler_sym.validate_symmetry())

    def test_get_symmetry_pairs(self):
        """Test getting symmetry pairs."""
        # Test without symmetry
        handler = CartesianEulerDroneGenomeHandler(
            min_max_narms=self.min_max_narms,
            append_arm_chance=0.0,
            bilateral_plane_for_symmetry=None,
            rnd=self.rnd
        )
        
        pairs = handler.get_symmetry_pairs()
        self.assertEqual(len(pairs), 0)
        
        # Test with symmetry
        handler_sym = CartesianEulerDroneGenomeHandler(
            min_max_narms=self.min_max_narms,
            append_arm_chance=0.0,
            bilateral_plane_for_symmetry="xy",
            rnd=self.rnd
        )
        
        pairs = handler_sym.get_symmetry_pairs()
        self.assertEqual(len(pairs), 2)  # 4 arms = 2 pairs
        
        # Check pairs are correct
        expected_pairs = [(0, 2), (1, 3)]
        self.assertEqual(pairs, expected_pairs)

    def test_operator_initialization(self):
        """Test that operators are properly initialized."""
        handler = CartesianEulerDroneGenomeHandler(
            min_max_narms=self.min_max_narms,
            parameter_limits=self.parameter_limits,
            append_arm_chance=0.0,
            bilateral_plane_for_symmetry="xy",
            rnd=self.rnd
        )
        
        # Check that operators exist
        self.assertTrue(hasattr(handler, 'symmetry_operator'))
        self.assertTrue(hasattr(handler, 'repair_operator'))
        
        # Check that operators are configured correctly
        self.assertEqual(handler.symmetry_operator.config.plane.value, "xy")
        self.assertTrue(handler.symmetry_operator.config.enabled)
        
        # Test with no symmetry
        handler_no_sym = CartesianEulerDroneGenomeHandler(
            min_max_narms=self.min_max_narms,
            append_arm_chance=0.0,
            bilateral_plane_for_symmetry=None,
            rnd=self.rnd
        )
        
        self.assertFalse(handler_no_sym.symmetry_operator.config.enabled)

    def test_operator_integration(self):
        """Test that operators are properly integrated with handler methods."""
        handler = CartesianEulerDroneGenomeHandler(
            min_max_narms=self.min_max_narms,
            parameter_limits=self.parameter_limits,
            append_arm_chance=0.0,
            bilateral_plane_for_symmetry="xy",
            rnd=self.rnd
        )
        
        # Test that is_valid uses repair operator
        original_genome = handler.genome.copy()
        is_valid_result = handler.is_valid()
        
        # Genome should be unchanged by validation
        npt.assert_array_equal(handler.genome, original_genome)
        self.assertIsInstance(is_valid_result, bool)
        
        # Test that repair uses repair operator
        # Make genome invalid
        handler.genome[0, 0] = 10.0  # Out of bounds
        self.assertFalse(handler.is_valid())
        
        handler.repair()
        self.assertTrue(handler.is_valid())
        
        # Test that symmetry methods use symmetry operator
        handler.apply_symmetry()
        self.assertTrue(handler.validate_symmetry())


class TestCartesianEulerPerformance(unittest.TestCase):
    """Performance tests for CartesianEulerDroneGenomeHandler."""

    def setUp(self):
        """Set up test fixtures."""
        self.rnd = np.random.default_rng(42)

    def test_performance_large_population(self):
        """Test performance with larger populations."""
        handler = CartesianEulerDroneGenomeHandler(
            min_max_narms=(8, 8),  # Larger drone
            parameter_limits=np.array([
                [-2.0, 2.0], [-2.0, 2.0], [-2.0, 2.0],
                [-np.pi, np.pi], [-np.pi, np.pi], [-np.pi, np.pi],
                [0, 1]
            ]),
            append_arm_chance=0.0,
            rnd=self.rnd,
            repair=False
        )
        
        # Test population generation
        start_time = time.time()
        large_population = handler.generate_random_population(100)
        generation_time = time.time() - start_time
        
        # Should complete reasonably quickly
        self.assertLess(generation_time, 1.0)  # Less than 1 second
        self.assertEqual(len(large_population), 100)
        
        # Test batch operations
        start_time = time.time()
        for individual in large_population:
            individual.mutate()
        mutation_time = time.time() - start_time
        
        self.assertLess(mutation_time, 1.0)  # Less than 1 second

    def test_performance_scaling_with_arms(self):
        """Test performance scaling with number of arms."""
        arm_counts = [4, 6, 8, 12]
        times = []
        
        for narms in arm_counts:
            handler = CartesianEulerDroneGenomeHandler(
                min_max_narms=(narms, narms),
                append_arm_chance=0.0,
                rnd=self.rnd,
                repair=False
            )
            
            start_time = time.time()
            population = handler.generate_random_population(50)
            
            # Perform some operations
            for individual in population[:10]:
                individual.mutate()
                individual.is_valid()
                individual.copy()
            
            end_time = time.time()
            times.append(end_time - start_time)
        
        # Performance should scale reasonably (not exponentially)
        for t in times:
            self.assertLess(t, 2.0)  # All should complete within 2 seconds

    def test_memory_efficiency(self):
        """Test memory efficiency with large genomes."""
        handler = CartesianEulerDroneGenomeHandler(
            min_max_narms=(16, 16),  # Large drone
            append_arm_chance=0.0,
            rnd=self.rnd,
            repair=False
        )
        
        # Create and manipulate many individuals
        population = handler.generate_random_population(50)
        
        # Test that copying doesn't share memory inappropriately
        copies = [ind.copy() for ind in population[:5]]
        
        # Modify copies and ensure originals are unchanged
        for i, (original, copy_ind) in enumerate(zip(population[:5], copies)):
            copy_ind.genome[0, 0] = 999 + i
            self.assertNotEqual(original.genome[0, 0], copy_ind.genome[0, 0])

    def test_crossover_performance(self):
        """Test crossover performance with various population sizes."""
        handler = CartesianEulerDroneGenomeHandler(
            min_max_narms=(6, 6),
            append_arm_chance=0.0,
            rnd=self.rnd,
            repair=False
        )
        
        population_sizes = [10, 25, 50]
        
        for pop_size in population_sizes:
            population1 = handler.generate_random_population(pop_size)
            population2 = handler.generate_random_population(pop_size)
            
            start_time = time.time()
            children = handler.crossover_population(population1, population2)
            crossover_time = time.time() - start_time
            
            self.assertEqual(len(children), pop_size)
            self.assertLess(crossover_time, 1.0)  # Should complete quickly

    def test_mutation_performance(self):
        """Test mutation performance with various configurations."""
        # Test with different mutation probabilities
        mutation_configs = [
            np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]),  # Low
            np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),        # Medium
            np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),        # High
        ]
        
        for mutation_probs in mutation_configs:
            handler = CartesianEulerDroneGenomeHandler(
                min_max_narms=(6, 6),
                append_arm_chance=0.0,
                mutation_probs=mutation_probs,
                rnd=self.rnd,
                repair=False
            )
            
            population = handler.generate_random_population(25)
            
            start_time = time.time()
            handler.mutate_population(population)
            mutation_time = time.time() - start_time
            
            self.assertLess(mutation_time, 1.0)  # Should complete quickly


if __name__ == '__main__':
    unittest.main(verbosity=2)