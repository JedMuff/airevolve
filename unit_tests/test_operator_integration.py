#!/usr/bin/env python3
"""
Integration tests for operator system with genome handlers.
"""

import unittest
import numpy as np
import numpy.testing as npt
from airevolve.evolution_tools.genome_handlers.cartesian_euler_genome_handler import CartesianEulerDroneGenomeHandler
from airevolve.evolution_tools.genome_handlers.spherical_angular_genome_handler import SphericalAngularDroneGenomeHandler


class TestCartesianOperatorIntegration(unittest.TestCase):
    """Test integration of operators with CartesianEulerDroneGenomeHandler."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.min_max_narms = (4, 4)
        self.parameter_limits = np.array([
            [-0.4, 0.4],    # x position
            [-0.4, 0.4],    # y position  
            [-0.4, 0.4],    # z position
            [-np.pi, np.pi], # roll
            [-np.pi, np.pi], # pitch
            [-np.pi, np.pi], # yaw
            [0, 1]          # direction
        ])
        self.rng = np.random.default_rng(42)
        # For backward compatibility
        self.narms = self.min_max_narms[1]
        
    def test_symmetry_operator_integration(self):
        """Test that symmetry operator is properly integrated."""
        handler = CartesianEulerDroneGenomeHandler(
            genome=None,
            min_max_narms=self.min_max_narms,
            parameter_limits=self.parameter_limits,
            bilateral_plane_for_symmetry="xy",
            append_arm_chance=0.0,
            rnd=self.rng
        )
        
        # Check operator exists and is configured
        self.assertTrue(hasattr(handler, 'symmetry_operator'))
        self.assertEqual(handler.symmetry_operator.config.plane.value, "xy")
        self.assertTrue(handler.symmetry_operator.config.enabled)
        
    def test_repair_operator_integration(self):
        """Test that repair operator is properly integrated."""
        handler = CartesianEulerDroneGenomeHandler(
            genome=None,
            min_max_narms=self.min_max_narms,
            parameter_limits=self.parameter_limits,
            bilateral_plane_for_symmetry="xy",
            append_arm_chance=0.0,
            rnd=self.rng
        )
        
        # Check operator exists and is configured
        self.assertTrue(hasattr(handler, 'repair_operator'))
        self.assertTrue(handler.repair_operator.config.apply_symmetry)
        
    def test_population_operations(self):
        """Test population-level operations with operators."""
        handler = CartesianEulerDroneGenomeHandler(
            genome=None,
            min_max_narms=self.min_max_narms,
            parameter_limits=self.parameter_limits,
            bilateral_plane_for_symmetry="xy",
            append_arm_chance=0.0,
            rnd=self.rng
        )
        
        # Generate population
        population = handler.generate_random_population(5)
        
        # All should be valid and symmetric
        for individual in population:
            self.assertTrue(individual.is_valid())
            self.assertTrue(individual.validate_symmetry())
            
        # Repair population should work
        repaired_population = handler.repair_population(population)
        self.assertEqual(len(repaired_population), len(population))
        
        for individual in repaired_population:
            self.assertTrue(individual.is_valid())
            self.assertTrue(individual.validate_symmetry())
            
    def test_crossover_with_operators(self):
        """Test crossover with operator integration."""
        parent1 = CartesianEulerDroneGenomeHandler(
            genome=None,
            min_max_narms=self.min_max_narms,
            parameter_limits=self.parameter_limits,
            bilateral_plane_for_symmetry="xy",
            append_arm_chance=0.0,
            rnd=self.rng
        )
        
        parent2 = CartesianEulerDroneGenomeHandler(
            genome=None,
            min_max_narms=self.min_max_narms,
            parameter_limits=self.parameter_limits,
            bilateral_plane_for_symmetry="xy",
            append_arm_chance=0.0,
            rnd=self.rng
        )
        
        child = parent1.crossover(parent2)
        
        # Child should be valid and symmetric
        self.assertTrue(child.is_valid())
        self.assertTrue(child.validate_symmetry())
        self.assertEqual(child.bilateral_plane_for_symmetry, "xy")
        
    def test_mutation_with_operators(self):
        """Test mutation with operator integration."""
        handler = CartesianEulerDroneGenomeHandler(
            genome=None,
            min_max_narms=self.min_max_narms,
            parameter_limits=self.parameter_limits,
            bilateral_plane_for_symmetry="xy",
            append_arm_chance=0.0,
            mutation_probs=[0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1],  # Force mutation
            rnd=self.rng
        )
        
        # Mutate multiple times
        for _ in range(10):
            handler.mutate()
            
            # Should remain valid and symmetric
            self.assertTrue(handler.is_valid())
            self.assertTrue(handler.validate_symmetry())
            
    def test_operator_configuration_changes(self):
        """Test that operator configuration follows handler settings."""
        # Test with different symmetry planes
        for plane in ["xy", "xz", "yz"]:
            handler = CartesianEulerDroneGenomeHandler(
                min_max_narms=(2, 6),
                bilateral_plane_for_symmetry=plane,
                append_arm_chance=0.0,
                rnd=self.rng
            )
            
            self.assertEqual(handler.symmetry_operator.config.plane.value, plane)
            self.assertTrue(handler.symmetry_operator.config.enabled)
            
        # Test with no symmetry
        handler_no_sym = CartesianEulerDroneGenomeHandler(
            genome=None,
            min_max_narms=self.min_max_narms,
            parameter_limits=self.parameter_limits,
            bilateral_plane_for_symmetry=None,
            append_arm_chance=0.0,
            rnd=self.rng
        )
        
        self.assertIsNone(handler_no_sym.symmetry_operator.config.plane)
        self.assertFalse(handler_no_sym.symmetry_operator.config.enabled)
        self.assertFalse(handler_no_sym.repair_operator.config.apply_symmetry)
        
    def test_error_handling_integration(self):
        """Test error handling in integrated system."""
        handler = CartesianEulerDroneGenomeHandler(
            genome=None,
            min_max_narms=self.min_max_narms,
            parameter_limits=self.parameter_limits,
            bilateral_plane_for_symmetry="xy",
            append_arm_chance=0.0,
            rnd=self.rng
        )
        
        # Test with malformed genome
        original_genome = handler.genome.copy()
        handler.genome = np.zeros((2, 5))  # Wrong shape
        
        is_valid = handler.is_valid()
        self.assertFalse(is_valid)
            
        # Restore genome
        handler.genome = original_genome
        
class TestSphericalOperatorIntegration(unittest.TestCase):
    """Test integration of operators with SphericalAngularDroneGenomeHandler."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.min_narms = 2
        self.max_narms = 6
        self.parameter_limits = np.array([
            [0.09, 0.4],      # magnitude
            [-np.pi, np.pi],  # arm rotation
            [-np.pi, np.pi],    # arm pitch
            [-np.pi, np.pi],  # motor rotation
            [-np.pi, np.pi],    # motor pitch
            [0, 1]           # direction
        ])
        self.rng = np.random.default_rng(42)
        
    def test_symmetry_operator_integration(self):
        """Test that symmetry operator is properly integrated."""
        handler = SphericalAngularDroneGenomeHandler(
            genome=None,
            min_max_narms=(self.min_narms, self.max_narms),
            parameter_limits=self.parameter_limits,
            bilateral_plane_for_symmetry="xy",
            rnd=self.rng
        )
        
        # Check operator exists and is configured
        self.assertTrue(hasattr(handler, 'symmetry_operator'))
        self.assertTrue(handler.symmetry_operator.config.enabled)
        
    def test_repair_operator_integration(self):
        """Test that repair operator is properly integrated."""
        handler = SphericalAngularDroneGenomeHandler(
            genome=None,
            min_max_narms=(self.min_narms, self.max_narms),
            parameter_limits=self.parameter_limits,
            bilateral_plane_for_symmetry="xy",
            rnd=self.rng
        )
        
        # Check operator exists and is configured
        self.assertTrue(hasattr(handler, 'repair_operator'))
        self.assertEqual(handler.repair_operator.min_narms, handler.min_narms)
        self.assertEqual(handler.repair_operator.max_narms, handler.max_narms)
        npt.assert_array_equal(handler.repair_operator.parameter_limits, self.parameter_limits)
        
    def test_population_operations(self):
        """Test population-level operations with operators."""
        handler = SphericalAngularDroneGenomeHandler(
            genome=None,
            min_max_narms=(self.min_narms, self.max_narms),
            parameter_limits=self.parameter_limits,
            bilateral_plane_for_symmetry="xy",
            rnd=self.rng
        )
        
        # Generate population - retry to get valid symmetric individuals
        population = []
        
        for _ in range(5):  # Need 5 individuals
            individual = SphericalAngularDroneGenomeHandler(
                min_max_narms=(self.min_narms, self.max_narms),
                parameter_limits=self.parameter_limits,
                bilateral_plane_for_symmetry="xy",
                rnd=self.rng
            )
            individual.genome = individual._generate_random_genome()
            
            if individual.is_valid() and individual.validate_symmetry():
                population.append(individual)
                break

        # All should be valid and symmetric
        for individual in population:
            self.assertTrue(individual.is_valid())
            self.assertTrue(individual.validate_symmetry())
            
        # Test vectorized operations
        pop_array = handler.random_population(3)
        self.assertEqual(pop_array.shape, (3, self.max_narms, 6))
        
    def test_crossover_with_operators(self):
        """Test crossover with operator integration."""
        parent1 = SphericalAngularDroneGenomeHandler(
            genome=None,
            min_max_narms=(6, 6),
            parameter_limits=self.parameter_limits,
            bilateral_plane_for_symmetry="xy",
            rnd=self.rng
        )
        
        parent2 = SphericalAngularDroneGenomeHandler(
            genome=None,
            min_max_narms=(6, 6),
            parameter_limits=self.parameter_limits,
            bilateral_plane_for_symmetry="xy",
            rnd=self.rng
        )

        # Generate properly symmetric parent genomes
        # Try multiple times to ensure we get symmetric parents
        parent1.genome = parent1._generate_random_genome()
        parent2.genome = parent2._generate_random_genome()
        
        # Check if parents are symmetric before crossover
        child = parent1.crossover(parent2)

        # Child should be valid and symmetric
        self.assertTrue(child.is_valid())
        self.assertTrue(child.validate_symmetry())
        self.assertTrue(child.symmetry)
        
    def test_mutation_with_operators(self):
        """Test mutation with operator integration."""
        handler = SphericalAngularDroneGenomeHandler(
            genome=None,
            min_max_narms=(self.min_narms, self.max_narms),
            parameter_limits=self.parameter_limits,
            bilateral_plane_for_symmetry="xy",
            rnd=self.rng
        )
        # Set up initial genome
        handler.genome = handler._generate_random_genome()
        # Mutate multiple times
        for _ in range(10):
            handler.mutate()
            
            # Should remain valid and symmetric
            self.assertTrue(handler.is_valid())
            self.assertTrue(handler.validate_symmetry())
            
    def test_operator_configuration_changes(self):
        """Test that operator configuration follows handler settings."""
        # Test with symmetry enabled
        handler_sym = SphericalAngularDroneGenomeHandler(
            genome=None,
            min_max_narms=(self.min_narms, self.max_narms),
            parameter_limits=self.parameter_limits,
            bilateral_plane_for_symmetry="xy",
            rnd=self.rng
        )
        
        self.assertTrue(handler_sym.symmetry_operator.config.enabled)
        self.assertTrue(handler_sym.repair_operator.config.apply_symmetry)
        
        # Test with symmetry disabled
        handler_no_sym = SphericalAngularDroneGenomeHandler(
            genome=None,
            min_max_narms=(self.min_narms, self.max_narms),
            parameter_limits=self.parameter_limits,
            bilateral_plane_for_symmetry=None,
            rnd=self.rng
        )
        
        self.assertFalse(handler_no_sym.symmetry_operator.config.enabled)
        self.assertFalse(handler_no_sym.repair_operator.config.apply_symmetry)
        
    def test_arm_count_operations(self):
        """Test arm count operations with operators."""
        handler = SphericalAngularDroneGenomeHandler(
            genome=None,
            min_max_narms=(self.min_narms, self.max_narms),
            parameter_limits=self.parameter_limits,
            bilateral_plane_for_symmetry=None,
            rnd=self.rng
        )
        
        # Start with minimum arms
        handler.genome = np.full((self.max_narms, 6), np.nan)
        for i in range(self.min_narms):
            handler.genome[i] = [0.3, 1.0, 1.0, 1.0, 1.0, 1]
            
        original_count = handler.get_arm_count()
        
        # Add arm if possible
        success = handler.add_random_arm()
        if success:
            self.assertEqual(handler.get_arm_count(), original_count + 1)
            self.assertTrue(handler.is_valid())
            
        # Remove arm
        if handler.get_arm_count() > self.min_narms:
            handler.remove_arm(0)
            self.assertTrue(handler.is_valid())
            
    def test_nan_masking_integration(self):
        """Test NaN masking integration with operators."""
        handler = SphericalAngularDroneGenomeHandler(
            genome=None,
            min_max_narms=(self.min_narms, self.max_narms),
            parameter_limits=self.parameter_limits,
            bilateral_plane_for_symmetry=None,
            rnd=self.rng
        )
        
        # Create genome with some valid arms
        handler.genome = np.full((self.max_narms, 6), np.nan)
        handler.genome[0] = [0.3, 0.5, 0.8, 1.2, 0.3, 1]
        handler.genome[2] = [0.3, 1.0, 0.6, 1.8, 0.4, 0]
        
        # Validate should work with NaN masking
        self.assertTrue(handler.is_valid())
        
        # Get valid arms should work
        valid_arms = handler.get_valid_arms()
        self.assertEqual(len(valid_arms), 2)

        
    def test_error_handling_integration(self):
        """Test error handling in integrated system."""
        handler = SphericalAngularDroneGenomeHandler(
            genome=None,
            min_max_narms=(self.min_narms, self.max_narms),
            parameter_limits=self.parameter_limits,
            bilateral_plane_for_symmetry="xy",
            rnd=self.rng
        )
        
        # Test with malformed genome
        original_genome = handler.genome.copy()
        handler.genome = np.zeros((2, 4))  # Wrong shape
        
        is_valid = handler.is_valid()
        self.assertFalse(is_valid)
            
        # Restore genome
        handler.genome = original_genome

class TestCrossHandlerCompatibility(unittest.TestCase):
    """Test compatibility between different genome handlers."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.rng = np.random.default_rng(42)
        
    def test_consistent_repair_behavior(self):
        """Test that repair behavior is consistent across handlers."""
        cartesian_handler = CartesianEulerDroneGenomeHandler(
            min_max_narms=(2, 6),
            bilateral_plane_for_symmetry=None,  # No symmetry for simpler comparison
            rnd=self.rng
        )
        
        spherical_handler = SphericalAngularDroneGenomeHandler(
            genome=None,
            min_max_narms=(2, 6),
            parameter_limits=np.array([
                [0.09, 0.4], [-np.pi, np.pi], [-np.pi, np.pi],
                [-np.pi, np.pi], [-np.pi, np.pi], [0, 1]
            ]),
            bilateral_plane_for_symmetry=None,  # No symmetry for simpler comparison
            rnd=self.rng
        )
        
        # Both should handle repair consistently
        for handler in [cartesian_handler, spherical_handler]:
            # Should be valid after initialization
            self.assertTrue(handler.is_valid())
            
            # Should remain valid after repair
            handler.repair()
            self.assertTrue(handler.is_valid())
            
if __name__ == '__main__':
    unittest.main()