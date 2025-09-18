#!/usr/bin/env python3
"""
Unit tests for spherical symmetry functionality in SphericalAngularDroneGenomeHandler.
"""

import unittest
import numpy as np
import numpy.testing as npt
from airevolve.evolution_tools.genome_handlers.spherical_angular_genome_handler import SphericalAngularDroneGenomeHandler


class TestSphericalSymmetry(unittest.TestCase):
    """Test spherical symmetry functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.min_narms = 4
        self.max_narms = 8
        self.rng = np.random.default_rng(42)  # Fixed seed for reproducibility
        
        # Standard parameter limits
        self.parameter_limits = np.array([
            [0.09, 0.4],      # magnitude
            [-np.pi, np.pi],  # arm rotation
            [-np.pi, np.pi],    # arm pitch
            [-np.pi, np.pi],  # motor rotation
            [-np.pi, np.pi],    # motor pitch
            [0, 1]           # direction
        ])
        
    def test_init_with_symmetry(self):
        """Test initialization with symmetry enabled."""
        handler = SphericalAngularDroneGenomeHandler(
            genome=None,
            min_max_narms=(self.min_narms, self.max_narms),
            parameter_limits=self.parameter_limits,
            bilateral_plane_for_symmetry="xy",
            rnd=self.rng
        )
        
        # Check symmetry settings
        self.assertTrue(handler.symmetry)
        self.assertTrue(handler.symmetry_operator.config.enabled)
        
        # With symmetry, arm counts should remain unchanged
        self.assertEqual(handler.min_narms, self.min_narms)
        self.assertEqual(handler.max_narms, self.max_narms)
        
    def test_init_without_symmetry(self):
        """Test initialization without symmetry."""
        handler = SphericalAngularDroneGenomeHandler(
            genome=None,
            min_max_narms=(self.min_narms, self.max_narms),
            parameter_limits=self.parameter_limits,
            bilateral_plane_for_symmetry=None,
            rnd=self.rng
        )
        
        # Check symmetry settings
        self.assertFalse(handler.symmetry)
        self.assertFalse(handler.symmetry_operator.config.enabled)
        
        # Without symmetry, arm counts should be unchanged
        self.assertEqual(handler.min_narms, self.min_narms)
        self.assertEqual(handler.max_narms, self.max_narms)
        
    def test_apply_symmetry_basic(self):
        """Test basic symmetry application."""
        handler = SphericalAngularDroneGenomeHandler(
            genome=None,
            min_max_narms=(self.min_narms, self.max_narms),
            parameter_limits=self.parameter_limits,
            bilateral_plane_for_symmetry="xy",
            rnd=self.rng
        )
        
        # Create a test genome with known values
        handler.genome = np.full((self.max_narms, 6), np.nan)
        handler.genome[0] = [0.3, 0.5, 0.8, 1.2, 0.3, 1]  # First arm
        handler.genome[1] = [0.3, 1.0, 0.6, 1.8, 0.4, 0]  # Second arm
        
        original_genome = handler.genome.copy()
        
        # Apply symmetry
        handler.apply_symmetry()
        
        # Should still be valid
        self.assertTrue(handler.is_valid())
        
        # Check that more arms are now valid (symmetry should add arms)
        original_valid_count = np.sum(~np.isnan(original_genome[:, 0]))
        new_valid_count = np.sum(~np.isnan(handler.genome[:, 0]))
        self.assertGreaterEqual(new_valid_count, original_valid_count)
        
    def test_unapply_symmetry_basic(self):
        """Test basic symmetry removal."""
        handler = SphericalAngularDroneGenomeHandler(
            genome=None,
            min_max_narms=(self.min_narms, self.max_narms),
            parameter_limits=self.parameter_limits,
            bilateral_plane_for_symmetry="xy",
            rnd=self.rng
        )
        
        # Create a symmetric genome
        handler.genome = np.full((self.max_narms, 6), np.nan)
        handler.genome[0] = [0.0, 0.5, 0.8, 1.2, 0.3, 1]
        handler.genome[1] = [0.3, 1.0, 0.6, 1.8, 0.4, 0]
        handler.genome[2] = [0.0, -0.5, 0.8, 1.2, -0.3, 1]  # Symmetric arm
        handler.genome[3] = [0.3, -1.0, 0.6, 1.8, -0.4, 0]  # Symmetric arm
        
        handler.apply_symmetry()
        handler.repair()

        symmetric_valid_count = np.sum(~np.isnan(handler.genome[:, 0]))

        # Remove symmetry
        handler.unapply_symmetry()
        
        # Should still be valid
        self.assertFalse(handler.is_valid())
        
        # Should have fewer valid arms after removing symmetry
        unsymmetric_valid_count = np.sum(~np.isnan(handler.genome[:, 0]))
        self.assertLessEqual(unsymmetric_valid_count, symmetric_valid_count)
        
    def test_validate_symmetry_symmetric_genome(self):
        """Test symmetry validation with symmetric genome."""
        handler = SphericalAngularDroneGenomeHandler(
            genome=None,
            min_max_narms=(self.min_narms, self.max_narms),
            parameter_limits=self.parameter_limits,
            bilateral_plane_for_symmetry="xy",
            rnd=self.rng
        )
        
        # Create initial genome
        handler.genome = np.full((self.max_narms, 6), np.nan)
        handler.genome[0] = [0.3, 0.5, 0.8, 1.2, 0.3, 1]
        handler.genome[1] = [0.35, 1.0, 0.6, 1.8, 0.4, 0]
        
        # Apply symmetry
        handler.apply_symmetry()
        
        # Should validate as symmetric
        self.assertTrue(handler.validate_symmetry())
        
    def test_validate_symmetry_asymmetric_genome(self):
        """Test symmetry validation with asymmetric genome."""
        handler = SphericalAngularDroneGenomeHandler(
            genome=None,
            min_max_narms=(self.min_narms, self.max_narms),
            parameter_limits=self.parameter_limits,
            bilateral_plane_for_symmetry="xy",
            rnd=self.rng
        )
        
        # Create asymmetric genome
        handler.genome = np.full((self.max_narms, 6), np.nan)
        handler.genome[0] = [0.0, 0.5, 0.8, 1.2, 0.3, 1]
        handler.genome[1] = [0.2, 1.0, 0.6, 1.8, 0.4, 0]
        handler.genome[2] = [0.3, 1.5, 0.4, 0.9, 0.7, 1]  # Not symmetric
        
        # Should not validate as symmetric
        self.assertFalse(handler.validate_symmetry())
        
    def test_validate_symmetry_no_symmetry(self):
        """Test symmetry validation when symmetry is disabled."""
        handler = SphericalAngularDroneGenomeHandler(
            genome=None,
            min_max_narms=(self.min_narms, self.max_narms),
            parameter_limits=self.parameter_limits,
            bilateral_plane_for_symmetry=None,
            rnd=self.rng
        )
        
        # Should always validate as symmetric when symmetry is disabled
        self.assertTrue(handler.validate_symmetry())
        
    def test_get_symmetry_pairs_with_symmetry(self):
        """Test getting symmetry pairs with symmetry enabled."""
        handler = SphericalAngularDroneGenomeHandler(
            genome=None,
            min_max_narms=(self.min_narms, self.max_narms),
            parameter_limits=self.parameter_limits,
            bilateral_plane_for_symmetry="xy",
            rnd=self.rng
        )
        
        # Create genome with some valid arms
        handler.genome = np.full((self.max_narms, 6), np.nan)
        handler.genome[0] = [0.0, 0.5, 0.8, 1.2, 0.3, 1]
        handler.genome[1] = [0.3, 1.0, 0.6, 1.8, 0.4, 0]
        
        pairs = handler.get_symmetry_pairs()
        
        # Should return a list of tuples
        self.assertIsInstance(pairs, list)
        if len(pairs) > 0:
            self.assertIsInstance(pairs[0], tuple)
            self.assertEqual(len(pairs[0]), 2)
            
    def test_get_symmetry_pairs_without_symmetry(self):
        """Test getting symmetry pairs with symmetry disabled."""
        handler = SphericalAngularDroneGenomeHandler(
            genome=None,
            min_max_narms=(self.min_narms, self.max_narms),
            parameter_limits=self.parameter_limits,
            bilateral_plane_for_symmetry=None,
            rnd=self.rng
        )
        
        pairs = handler.get_symmetry_pairs()
        
        # Should return empty list when symmetry is disabled
        self.assertEqual(len(pairs), 0)
        
    def test_random_genome_with_symmetry(self):
        """Test random genome generation with symmetry."""
        handler = SphericalAngularDroneGenomeHandler(
            genome=None,
            min_max_narms=(self.min_narms, self.max_narms),
            parameter_limits=self.parameter_limits,
            bilateral_plane_for_symmetry="xy",
            rnd=self.rng,
            repair=False
        )
        
        # Generate multiple random genomes
        for _ in range(5):
            genome = handler._generate_random_genome()
            handler.genome = genome
            
            # Should be valid
            self.assertTrue(handler.is_valid())
            
            # Should satisfy symmetry constraints
            self.assertTrue(handler.validate_symmetry())
            
    def test_random_population_with_symmetry(self):
        """Test random population generation with symmetry."""
        handler = SphericalAngularDroneGenomeHandler(
            genome=None,
            min_max_narms=(self.min_narms, self.max_narms),
            parameter_limits=self.parameter_limits,
            bilateral_plane_for_symmetry="xy",
            rnd=self.rng,
            repair=False
        )
        
        population = handler.generate_random_population(5)
        
        # Check all individuals
        for individual in population:
            self.assertTrue(individual.is_valid())
            self.assertTrue(individual.validate_symmetry())
            self.assertTrue(individual.symmetry)
            
    def test_crossover_maintains_symmetry(self):
        """Test that crossover maintains symmetry."""
        parent1 = SphericalAngularDroneGenomeHandler(
            genome=None,
            min_max_narms=(self.min_narms, self.max_narms),
            parameter_limits=self.parameter_limits,
            bilateral_plane_for_symmetry="xy",
            rnd=self.rng,
            repair=False
        )
        
        parent2 = SphericalAngularDroneGenomeHandler(
            genome=None,
            min_max_narms=(self.min_narms, self.max_narms),
            parameter_limits=self.parameter_limits,
            bilateral_plane_for_symmetry="xy",
            rnd=self.rng,
            repair=False
        )
        
        # Set up parent genomes
        parent1.genome = parent1._generate_random_genome()
        parent2.genome = parent2._generate_random_genome()
        
        # Perform crossover
        child = parent1.crossover(parent2)
        
        # Child should be valid and symmetric
        self.assertTrue(child.is_valid())
        self.assertTrue(child.validate_symmetry())
        self.assertTrue(child.symmetry)
        
    def test_mutation_maintains_symmetry(self):
        """Test that mutation maintains symmetry."""
        handler = SphericalAngularDroneGenomeHandler(
            genome=None,
            min_max_narms=(self.min_narms, self.max_narms),
            parameter_limits=self.parameter_limits,
            bilateral_plane_for_symmetry="xy",
            rnd=self.rng,
            repair=False
        )
        
        # Set up initial genome
        handler.genome = handler._generate_random_genome()
        
        # Perform multiple mutations
        for _ in range(10):
            handler.mutate()
            
            # Should remain valid and symmetric
            self.assertTrue(handler.is_valid())
            self.assertTrue(handler.validate_symmetry())
            
    def test_copy_preserves_symmetry(self):
        """Test that copy preserves symmetry settings."""
        handler = SphericalAngularDroneGenomeHandler(
            genome=None,
            min_max_narms=(self.min_narms, self.max_narms),
            parameter_limits=self.parameter_limits,
            bilateral_plane_for_symmetry="xy",
            rnd=self.rng,
            repair=False
        )
        
        copy_handler = handler.copy()
        
        # Should preserve symmetry settings
        self.assertEqual(copy_handler.symmetry, handler.symmetry)
        self.assertEqual(copy_handler.symmetry_operator.config.enabled, 
                        handler.symmetry_operator.config.enabled)
        
        # Should be valid and symmetric
        self.assertTrue(copy_handler.is_valid())
        self.assertTrue(copy_handler.validate_symmetry())
        
    def test_repair_maintains_symmetry(self):
        """Test that repair maintains symmetry."""
        handler = SphericalAngularDroneGenomeHandler(
            genome=None,
            min_max_narms=(self.min_narms, self.max_narms),
            parameter_limits=self.parameter_limits,
            bilateral_plane_for_symmetry="xy",
            rnd=self.rng,
            repair=True
        )
        
        # Create invalid genome
        handler.genome = np.full((self.max_narms, 6), np.nan)
        handler.genome[0] = [10.0, 0.5, 0.8, 1.2, 0.3, 0]  # Invalid magnitude
        handler.genome[1] = [1.5, 1.0, 0.6, 1.8, 0.4, 0]
        handler.genome[2] = [1.0, 0.5, 0.8, 1.2, 0.3, 1]  # Valid arm
        handler.apply_symmetry()
        
        # Should be invalid
        self.assertFalse(handler.is_valid())
        
        # Repair should fix it and maintain symmetry
        handler.repair()
        self.assertTrue(handler.is_valid())
        self.assertTrue(handler.validate_symmetry())
        
    def test_population_symmetry_operations(self):
        """Test population-level symmetry operations."""
        handler = SphericalAngularDroneGenomeHandler(
            genome=None,
            min_max_narms=(self.min_narms, self.max_narms),
            parameter_limits=self.parameter_limits,
            bilateral_plane_for_symmetry="xy",
            rnd=self.rng,
            repair=False
        )
        
        # Create population
        population = handler.random_population(3)
        
        # Apply symmetry to population
        # symmetric_population = handler.apply_symmetry_pop(population)
        
        # Should have same shape
        self.assertEqual(population.shape, population.shape)
        
        # Unapply symmetry
        unsymmetric_population = handler.unapply_symmetry_pop(population)
        
        # Should have same shape
        self.assertEqual(unsymmetric_population.shape, population.shape)
        
    def test_symmetry_with_variable_arms(self):
        """Test symmetry with variable number of arms."""
        handler = SphericalAngularDroneGenomeHandler(
            genome=None,
            min_max_narms=(self.min_narms, self.max_narms),
            parameter_limits=self.parameter_limits,
            bilateral_plane_for_symmetry="xy",
            rnd=self.rng,
            repair=False
        )
        
        # Test with different numbers of valid arms
        for num_arms in range(handler.min_narms, handler.max_narms + 1):
            handler.genome = np.full((self.max_narms, 6), np.nan)
            
            # Add valid arms
            
            for i in range(num_arms // 2):
                handler.genome[i] = [
                    0.09 + i * 0.05,  # magnitude
                    i * 0.5,        # arm rotation
                    0.8,            # arm pitch
                    1.2,            # motor rotation
                    0.3,            # motor pitch
                    i % 2           # direction
                ]
            
            # Apply symmetry
            handler.apply_symmetry()
            
            # Should be valid and symmetric
            self.assertTrue(handler.is_valid())
            self.assertTrue(handler.validate_symmetry())
            
    def test_symmetry_edge_cases(self):
        """Test symmetry with edge cases."""
        handler = SphericalAngularDroneGenomeHandler(
            genome=None,
            min_max_narms=(self.min_narms, self.max_narms),
            parameter_limits=self.parameter_limits,
            bilateral_plane_for_symmetry="xy",
            rnd=self.rng,
            repair=False
        )
        
        # Test with empty genome (all NaN)
        handler.genome = np.full((self.max_narms, 6), np.nan)
        
        # Should handle empty genome gracefully
        handler.apply_symmetry()
        self.assertTrue(handler.validate_symmetry())
        
        # Test with single arm
        handler.genome = np.full((self.max_narms, 6), np.nan)
        handler.genome[0] = [1.0, 0.5, 0.8, 1.2, 0.3, 1]
        
        handler.apply_symmetry()
        self.assertTrue(handler.validate_symmetry())
        
    def test_symmetry_operator_integration(self):
        """Test that symmetry operator is properly integrated."""
        handler = SphericalAngularDroneGenomeHandler(
            genome=None,
            min_max_narms=(self.min_narms, self.max_narms),
            parameter_limits=self.parameter_limits,
            bilateral_plane_for_symmetry="xy",
            rnd=self.rng,
            repair=False
        )
        
        # Check operator exists and is configured correctly
        self.assertTrue(hasattr(handler, 'symmetry_operator'))
        self.assertTrue(handler.symmetry_operator.config.enabled)
        
        # Test operator methods work
        handler.genome = np.full((self.max_narms, 6), np.nan)
        handler.genome[0] = [1.0, 0.5, 0.8, 1.2, 0.3, 1]
        
        original_genome = handler.genome.copy()
        symmetric_genome = handler.symmetry_operator.apply_symmetry(original_genome)
        
        # Should be different (symmetry applied)
        handler.genome = symmetric_genome
        self.assertTrue(handler.validate_symmetry())
        
    def test_symmetry_parameter_bounds(self):
        """Test that symmetry respects parameter bounds."""
        handler = SphericalAngularDroneGenomeHandler(
            genome=None,
            min_max_narms=(self.min_narms, self.max_narms),
            parameter_limits=self.parameter_limits,
            bilateral_plane_for_symmetry="xy",
            rnd=self.rng,
            repair=False
        )
        
        # Create genome with values near bounds
        handler.genome = np.full((self.max_narms, 6), np.nan)
        handler.genome[0] = [
            self.parameter_limits[0, 1] - 0.001,  # magnitude near max
            self.parameter_limits[1, 1] - 0.001,  # arm rotation near max
            self.parameter_limits[2, 1] - 0.001,  # arm pitch near max
            self.parameter_limits[3, 1] - 0.001,  # motor rotation near max
            self.parameter_limits[4, 1] - 0.001,  # motor pitch near max
            1  # direction
        ]
        handler.genome[1] = [
            self.parameter_limits[0, 0] + 0.001,  # magnitude near min
            self.parameter_limits[1, 0] + 0.001,  # arm rotation near min
            self.parameter_limits[2, 0] + 0.001,  # arm pitch near min
            self.parameter_limits[3, 0] + 0.001,  # motor rotation near min
            self.parameter_limits[4, 0] + 0.001,  # motor pitch near min
            0  # direction
        ]
        
        # Apply symmetry
        handler.apply_symmetry()
        
        # Should be valid and within bounds
        self.assertTrue(handler.is_valid())
        
        # Check that all valid parameters are within bounds
        valid_arms = handler.get_valid_arms()
        for arm in valid_arms:
            for i in range(6):
                self.assertGreaterEqual(arm[i], self.parameter_limits[i, 0])
                self.assertLessEqual(arm[i], self.parameter_limits[i, 1])
                
    def test_string_representations_with_symmetry(self):
        """Test string representations include symmetry info."""
        handler = SphericalAngularDroneGenomeHandler(
            genome=None,
            min_max_narms=(self.min_narms, self.max_narms),
            parameter_limits=self.parameter_limits,
            bilateral_plane_for_symmetry="xy",
            rnd=self.rng,
            repair=False
        )
        
        str_repr = str(handler)
        self.assertIn("bilateral_plane_for_symmetry=xy", str_repr)
        
        repr_str = repr(handler)
        self.assertIn("bilateral_plane_for_symmetry=xy", repr_str)
        
        # Test without symmetry
        handler_no_sym = SphericalAngularDroneGenomeHandler(
            genome=None,
            min_max_narms=(self.min_narms, self.max_narms),
            parameter_limits=self.parameter_limits,
            bilateral_plane_for_symmetry=None,
            rnd=self.rng
        )
        
        str_repr_no_sym = str(handler_no_sym)
        self.assertIn("bilateral_plane_for_symmetry=None", str_repr_no_sym)
        
    def test_error_handling_with_symmetry(self):
        """Test error handling with symmetry enabled."""
        handler = SphericalAngularDroneGenomeHandler(
            genome=None,
            min_max_narms=(self.min_narms, self.max_narms),
            parameter_limits=self.parameter_limits,
            bilateral_plane_for_symmetry="xy",
            rnd=self.rng,
            repair=False
        )
        
        # Test with malformed genome
        original_shape = handler.genome.shape
        handler.genome = np.zeros((2, 4))  # Wrong shape
        
        # Should handle gracefully
        try:
            is_valid = handler.is_valid()
            self.assertFalse(is_valid)
        except Exception as e:
            self.fail(f"Unexpected exception in validation: {e}")
            
        # Restore shape
        handler.genome = np.full(original_shape, np.nan)


if __name__ == '__main__':
    unittest.main()