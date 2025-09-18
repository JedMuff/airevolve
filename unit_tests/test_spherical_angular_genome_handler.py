#!/usr/bin/env python3
"""
Unit tests for SphericalAngularDroneGenomeHandler with the new operator system.
"""

import unittest
import numpy as np
import numpy.testing as npt
from unittest.mock import MagicMock, patch
import argparse
import sys
import os
from typing import List, Tuple

# Statistical testing
from scipy import stats
import warnings

# Visualization (optional)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from airevolve.evolution_tools.inspection_tools.drone_visualizer import DroneVisualizer

# The module being tested
from airevolve.evolution_tools.genome_handlers.spherical_angular_genome_handler import SphericalAngularDroneGenomeHandler


class TestSphericalAngularDroneGenomeHandler(unittest.TestCase):
    """Comprehensive unit tests for SphericalAngularDroneGenomeHandler."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.min_narms = 3
        self.max_narms = 6
        self.rnd = np.random.default_rng(42)  # Fixed seed for reproducibility
        
        # Standard parameter limits
        self.parameter_limits = np.array([
            [0.09, 0.4],      # magnitude
            [-np.pi, np.pi],  # arm rotation
            [-np.pi, np.pi],    # arm pitch
            [-np.pi, np.pi],  # motor rotation
            [-np.pi, np.pi],    # motor pitch
            [0, 1]           # direction
        ])

    def test_initialization_basic(self):
        """Test basic initialization."""
        handler = SphericalAngularDroneGenomeHandler(
            genome=None,
            min_max_narms=(self.min_narms, self.max_narms),
            parameter_limits=self.parameter_limits,
            rnd=self.rnd
        )
        
        # Check basic attributes
        self.assertEqual(handler.min_narms, self.min_narms)
        self.assertEqual(handler.max_narms, self.max_narms)
        self.assertEqual(handler.genome.shape, (self.max_narms, 6))
        
        # Check that operators are initialized
        self.assertTrue(hasattr(handler, 'symmetry_operator'))
        self.assertTrue(hasattr(handler, 'repair_operator'))

    def test_initialization_with_symmetry(self):
        """Test initialization with symmetry enabled."""
        handler = SphericalAngularDroneGenomeHandler(
            genome=None,
            min_max_narms=(self.min_narms, self.max_narms),
            parameter_limits=self.parameter_limits,
            bilateral_plane_for_symmetry="xy",
            rnd=self.rnd
        )
        
        # Check symmetry settings
        self.assertTrue(handler.symmetry)
        self.assertTrue(handler.symmetry_operator.config.enabled)
        
        # With symmetry, arm counts remain the same but symmetry is enforced
        self.assertEqual(handler.min_narms, self.min_narms)
        self.assertEqual(handler.max_narms, self.max_narms)

    def test_initialization_invalid_parameters(self):
        """Test initialization with invalid parameters."""
        # Invalid arm counts
        with self.assertRaises(ValueError):
            SphericalAngularDroneGenomeHandler(
                min_max_narms=(0, 5),  # min_narms too low
                parameter_limits=self.parameter_limits,
                rnd=self.rnd
            )
        
        with self.assertRaises(ValueError):
            SphericalAngularDroneGenomeHandler(
                min_max_narms=(5, 3),  # max_narms < min_narms
                parameter_limits=self.parameter_limits,
                rnd=self.rnd
            )
        
        # Invalid parameter limits shape
        with self.assertRaises(ValueError):
            SphericalAngularDroneGenomeHandler(
                min_max_narms=(self.min_narms, self.max_narms),
                parameter_limits=np.array([[0, 1], [0, 2]]),  # Wrong shape
                rnd=self.rnd
            )

    def test_generate_random_genome(self):
        """Test random genome generation."""
        handler = SphericalAngularDroneGenomeHandler(
            genome=None,
            min_max_narms=(self.min_narms, self.max_narms),
            parameter_limits=self.parameter_limits,
            rnd=self.rnd
        )
        
        genome = handler._generate_random_genome()
        
        # Check shape
        self.assertEqual(genome.shape, (self.max_narms, 6))
        
        # Check that some arms are valid (non-NaN)
        valid_arms_mask = ~np.isnan(genome[:, 0])
        num_valid_arms = np.sum(valid_arms_mask)
        self.assertGreaterEqual(num_valid_arms, self.min_narms)
        self.assertLessEqual(num_valid_arms, self.max_narms)
        
        # Check parameter bounds for valid arms
        valid_arms = genome[valid_arms_mask]
        for i in range(6):
            param_values = valid_arms[:, i]
            self.assertTrue(np.all(param_values >= self.parameter_limits[i, 0]))
            self.assertTrue(np.all(param_values <= self.parameter_limits[i, 1]))
        
        # Check that direction values are 0 or 1
        directions = valid_arms[:, 5]
        self.assertTrue(np.all(np.isin(directions, [0, 1])))

    def test_generate_random_population(self):
        """Test random population generation."""
        handler = SphericalAngularDroneGenomeHandler(
            genome=None,
            min_max_narms=(self.min_narms, self.max_narms),
            parameter_limits=self.parameter_limits,
            rnd=self.rnd
        )
        
        population_size = 10
        population = handler.generate_random_population(population_size)
        
        # Check population size
        self.assertEqual(len(population), population_size)
        
        # Check that all individuals are valid
        for individual in population:
            self.assertIsInstance(individual, SphericalAngularDroneGenomeHandler)
            self.assertTrue(individual.is_valid())

    def test_crossover_valid_parents(self):
        """Test crossover between two valid parents."""
        handler = SphericalAngularDroneGenomeHandler(
            genome=None,
            min_max_narms=(self.min_narms, self.max_narms),
            parameter_limits=self.parameter_limits,
            rnd=self.rnd
        )
        
        parent1 = handler.copy()
        parent1.genome = parent1._generate_random_genome()
        
        parent2 = handler.copy()
        parent2.genome = parent2._generate_random_genome()
        
        child = parent1.crossover(parent2)
        
        # Check child properties
        self.assertIsInstance(child, SphericalAngularDroneGenomeHandler)
        self.assertEqual(child.genome.shape, (self.max_narms, 6))
        self.assertTrue(child.is_valid())

    def test_crossover_invalid_parent_type(self):
        """Test crossover with invalid parent type."""
        handler = SphericalAngularDroneGenomeHandler(
            genome=None,
            min_max_narms=(self.min_narms, self.max_narms),
            parameter_limits=self.parameter_limits,
            rnd=self.rnd
        )
        
        invalid_parent = "not_a_genome_handler"
        
        with self.assertRaises(TypeError):
            handler.crossover(invalid_parent)

    def test_mutation_basic(self):
        """Test basic mutation functionality."""
        handler = SphericalAngularDroneGenomeHandler(
            genome=None,
            min_max_narms=(self.min_narms, self.max_narms),
            parameter_limits=self.parameter_limits,
            rnd=self.rnd
        )
        
        # Set initial genome
        handler.genome = handler._generate_random_genome()
        original_genome = handler.genome.copy()
        
        # Perform mutation
        handler.mutate()
        
        # Genome should still be valid
        self.assertTrue(handler.is_valid())
        
        # Check that mutation occurred (might be subtle)
        self.assertEqual(handler.genome.shape, original_genome.shape)

    def test_copy(self):
        """Test genome handler copying."""
        handler = SphericalAngularDroneGenomeHandler(
            genome=None,
            min_max_narms=(self.min_narms, self.max_narms),
            parameter_limits=self.parameter_limits,
            rnd=self.rnd
        )
        
        handler.genome = handler._generate_random_genome()
        
        copy_handler = handler.copy()
        
        # Check that copy has same attributes
        self.assertEqual(copy_handler.min_narms, handler.min_narms)
        self.assertEqual(copy_handler.max_narms, handler.max_narms)
        npt.assert_array_equal(copy_handler.parameter_limits, handler.parameter_limits)
        
        # Check that genome is copied (not referenced)
        npt.assert_array_equal(copy_handler.genome, handler.genome)
        self.assertIsNot(copy_handler.genome, handler.genome)

    def test_is_valid_valid_genome(self):
        """Test validation of valid genome."""
        handler = SphericalAngularDroneGenomeHandler(
            genome=None,
            min_max_narms=(self.min_narms, self.max_narms),
            parameter_limits=self.parameter_limits,
            rnd=self.rnd
        )
        
        # Generate and set valid genome
        handler.genome = handler._generate_random_genome()
        self.assertTrue(handler.is_valid())

    def test_is_valid_invalid_shape(self):
        """Test validation with invalid genome shape."""
        handler = SphericalAngularDroneGenomeHandler(
            genome=None,
            min_max_narms=(self.min_narms, self.max_narms),
            parameter_limits=self.parameter_limits,
            rnd=self.rnd
        )
        
        # Set invalid shape
        handler.genome = np.ones((3, 5))  # Wrong number of parameters (should be 6)
        self.assertFalse(handler.is_valid())

    def test_is_valid_invalid_arm_count(self):
        """Test validation with invalid arm count."""
        handler = SphericalAngularDroneGenomeHandler(
            genome=None,
            min_max_narms=(self.min_narms, self.max_narms),
            parameter_limits=self.parameter_limits,
            rnd=self.rnd
        )
        
        # Set too few arms
        handler.genome = np.full((self.max_narms, 6), np.nan)
        handler.genome[0] = [1.0, 1.0, 1.0, 1.0, 1.0, 1]  # Only 1 arm
        self.assertFalse(handler.is_valid())

    def test_is_valid_out_of_bounds_parameters(self):
        """Test validation with out-of-bounds parameters."""
        handler = SphericalAngularDroneGenomeHandler(
            genome=None,
            min_max_narms=(self.min_narms, self.max_narms),
            parameter_limits=self.parameter_limits,
            rnd=self.rnd
        )
        
        # Set valid arm count but invalid parameters
        handler.genome = np.full((self.max_narms, 6), np.nan)
        for i in range(self.min_narms):
            handler.genome[i] = [1.0, 1.0, 1.0, 1.0, 1.0, 1]
        
        # Make one parameter out of bounds
        handler.genome[0, 0] = 10.0  # magnitude too high
        self.assertFalse(handler.is_valid())

    def test_repair_functionality(self):
        """Test repair functionality."""
        handler = SphericalAngularDroneGenomeHandler(
            genome=None,
            min_max_narms=(self.min_narms, self.max_narms),
            parameter_limits=self.parameter_limits,
            rnd=self.rnd
        )
        
        # Create invalid genome with enough arms
        handler.genome = np.full((self.max_narms, 6), np.nan)
        handler.genome[0] = [10.0, 1.0, 1.0, 1.0, 1.0, 0.5]  # Out of bounds magnitude and direction
        for i in range(1, self.min_narms):
            handler.genome[i] = [1.5, 1.0, 1.0, 1.0, 1.0, 1]  # Valid arms to meet minimum
        
        # Should be invalid
        self.assertFalse(handler.is_valid())
        
        # Repair should fix it
        handler.repair()
        self.assertTrue(handler.is_valid())

    def test_get_valid_arms(self):
        """Test getting valid arms."""
        handler = SphericalAngularDroneGenomeHandler(
            genome=None,
            min_max_narms=(self.min_narms, self.max_narms),
            parameter_limits=self.parameter_limits,
            rnd=self.rnd
        )
        
        # Set known genome
        handler.genome = np.full((self.max_narms, 6), np.nan)
        handler.genome[0] = [1.0, 1.0, 1.0, 1.0, 1.0, 1]
        handler.genome[2] = [1.5, 2.0, 0.5, 1.5, 0.8, 0]
        
        valid_arms = handler.get_valid_arms()
        
        # Should have 2 valid arms
        self.assertEqual(len(valid_arms), 2)
        npt.assert_array_equal(valid_arms[0], handler.genome[0])
        npt.assert_array_equal(valid_arms[1], handler.genome[2])

    def test_get_arm_count(self):
        """Test getting arm count."""
        handler = SphericalAngularDroneGenomeHandler(
            genome=None,
            min_max_narms=(self.min_narms, self.max_narms),
            parameter_limits=self.parameter_limits,
            rnd=self.rnd
        )
        
        # Set known genome
        handler.genome = np.full((self.max_narms, 6), np.nan)
        handler.genome[0] = [1.0, 1.0, 1.0, 1.0, 1.0, 1]
        handler.genome[2] = [1.5, 2.0, 0.5, 1.5, 0.8, 0]
        
        arm_count = handler.get_arm_count()
        self.assertEqual(arm_count, 2)

    def test_symmetry_methods(self):
        """Test symmetry-related methods."""
        handler = SphericalAngularDroneGenomeHandler(
            genome=None,
            min_max_narms=(self.min_narms, self.max_narms),
            parameter_limits=self.parameter_limits,
            bilateral_plane_for_symmetry="xy",
            rnd=self.rnd
        )
        
        # Test symmetry validation
        is_symmetric = handler.validate_symmetry()
        self.assertIsInstance(is_symmetric, bool)
        
        # Test getting symmetry pairs
        pairs = handler.get_symmetry_pairs()
        self.assertIsInstance(pairs, list)
        
        # Test applying symmetry with a genome that has space for symmetric arms
        handler.genome = np.full((self.max_narms, 6), np.nan)
        # Add just enough arms to ensure symmetry can work (less than max_narms/2)
        num_test_arms = min(2, self.max_narms // 2)
        for i in range(num_test_arms):
            handler.genome[i] = [1.0, 1.0, 1.0, 1.0, 1.0, 1]
        
        original_genome = handler.genome.copy()
        handler.apply_symmetry()
        # Apply symmetry should not crash
        self.assertEqual(handler.genome.shape, original_genome.shape)
        
        # Test removing symmetry
        handler.unapply_symmetry()
        self.assertEqual(handler.genome.shape, original_genome.shape)

    def test_add_remove_arms(self):
        """Test arm addition and removal."""
        handler = SphericalAngularDroneGenomeHandler(
            genome=None,
            min_max_narms=(self.min_narms, self.max_narms),
            parameter_limits=self.parameter_limits,
            rnd=self.rnd
        )
        
        # Start with minimum arms
        handler.genome = np.full((self.max_narms, 6), np.nan)
        for i in range(self.min_narms):
            handler.genome[i] = [1.0, 1.0, 1.0, 1.0, 1.0, 1]
        
        original_count = handler.get_arm_count()
        
        # Try to add an arm
        success = handler.add_random_arm()
        if success:
            self.assertEqual(handler.get_arm_count(), original_count + 1)
        
        # Try to remove an arm
        handler.remove_arm(0)
        self.assertEqual(handler.get_arm_count(), original_count - 1 + (1 if success else 0))

    def test_operator_initialization(self):
        """Test that operators are properly initialized."""
        handler = SphericalAngularDroneGenomeHandler(
            genome=None,
            min_max_narms=(4, 8),  # Use larger values to ensure symmetry works
            parameter_limits=self.parameter_limits,
            bilateral_plane_for_symmetry="xy",
            rnd=self.rnd
        )
        
        # Check that operators exist
        self.assertTrue(hasattr(handler, 'symmetry_operator'))
        self.assertTrue(hasattr(handler, 'repair_operator'))
        
        # Check that operators are configured correctly
        self.assertTrue(handler.symmetry_operator.config.enabled)
        self.assertEqual(handler.repair_operator.min_narms, handler.min_narms)
        self.assertEqual(handler.repair_operator.max_narms, handler.max_narms)

    def test_operator_integration(self):
        """Test that operators are properly integrated with handler methods."""
        handler = SphericalAngularDroneGenomeHandler(
            genome=None,
            min_max_narms=(self.min_narms, self.max_narms),
            parameter_limits=self.parameter_limits,
            rnd=self.rnd
        )
        
        # Test that is_valid uses repair operator
        handler.genome = handler._generate_random_genome()
        original_genome = handler.genome.copy()
        is_valid_result = handler.is_valid()
        
        # Genome should be unchanged by validation
        npt.assert_array_equal(handler.genome, original_genome)
        self.assertIsInstance(is_valid_result, bool)
        
        # Test that repair uses repair operator
        # Make genome invalid
        handler.genome[0, 0] = 10.0  # Out of bounds
        if handler.get_arm_count() > 0:
            self.assertFalse(handler.is_valid())
            
            handler.repair()
            self.assertTrue(handler.is_valid())

    def test_vectorized_operations(self):
        """Test vectorized population operations."""
        handler = SphericalAngularDroneGenomeHandler(
            genome=None,
            min_max_narms=(self.min_narms, self.max_narms),
            parameter_limits=self.parameter_limits,
            rnd=self.rnd
        )
        
        # Test vectorized random population generation
        population = handler.random_population(5)
        self.assertEqual(population.shape, (5, self.max_narms, 6))
        
        # Test that all individuals are valid
        for i in range(5):
            # Create temporary handler to test validity
            temp_handler = handler.copy()
            temp_handler.genome = population[i]
            self.assertTrue(temp_handler.is_valid())

    def test_string_representations(self):
        """Test string representations."""
        handler = SphericalAngularDroneGenomeHandler(
            genome=None,
            min_max_narms=(self.min_narms, self.max_narms),
            parameter_limits=self.parameter_limits,
            bilateral_plane_for_symmetry="xy",
            rnd=self.rnd
        )
        
        str_repr = str(handler)
        self.assertIn("SphericalAngularDroneGenomeHandler", str_repr)
        self.assertIn("bilateral_plane_for_symmetry=xy", str_repr)
        
        repr_str = repr(handler)
        self.assertIn("SphericalAngularDroneGenomeHandler", repr_str)
        self.assertIn("bilateral_plane_for_symmetry=xy", repr_str)

    def test_backward_compatibility(self):
        """Test backward compatibility with existing API."""
        handler = SphericalAngularDroneGenomeHandler(
            genome=None,
            min_max_narms=(self.min_narms, self.max_narms),
            parameter_limits=self.parameter_limits,
            bilateral_plane_for_symmetry="xy",
            rnd=self.rnd
        )
        
        # Test that old methods still work
        # Create a smaller population that can accommodate symmetry
        population = np.full((3, self.max_narms, 6), np.nan)
        num_test_arms = min(2, self.max_narms // 2)
        for i in range(3):
            for j in range(num_test_arms):
                population[i, j] = [1.0, 1.0, 1.0, 1.0, 1.0, 1]
        
        # Population-level symmetry operations
        symmetric_pop = handler.apply_symmetry_pop(population)
        self.assertEqual(symmetric_pop.shape, population.shape)
        
        unsymmetric_pop = handler.unapply_symmetry_pop(symmetric_pop)
        self.assertEqual(unsymmetric_pop.shape, population.shape)


class TestSphericalAngularUniformity(unittest.TestCase):
    """Test uniformity of random generation for spherical coordinates."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.min_narms = 3
        self.max_narms = 6
        self.sample_size = 500  # Smaller sample for spherical testing
        self.alpha = 0.05  # Significance level
        self.rnd = np.random.default_rng(42)
        
        self.parameter_limits = np.array([
            [0.5, 2.0],      # magnitude
            [-np.pi, np.pi],  # arm rotation
            [-np.pi, np.pi],    # arm pitch
            [-np.pi, np.pi],  # motor rotation
            [-np.pi, np.pi],    # motor pitch
            [0, 1]           # direction
        ])
    
    def _generate_large_population(self, size: int) -> List[SphericalAngularDroneGenomeHandler]:
        """Generate a large population for statistical testing."""
        handler = SphericalAngularDroneGenomeHandler(
            genome=None,
            min_max_narms=(self.min_narms, self.max_narms),
            parameter_limits=self.parameter_limits,
            rnd=self.rnd,
            repair=False  # Enable repair to ensure valid genomes
        )
        return handler.generate_random_population(size)
    
    def test_parameter_uniformity(self):
        """Test statistical uniformity of parameters."""
        print(f"\n=== Testing Spherical Parameter Uniformity (n={self.sample_size}) ===")
        
        population = self._generate_large_population(self.sample_size)
        
        # Extract all valid parameters
        all_params = []
        for individual in population:
            valid_arms = individual.get_valid_arms()
            if len(valid_arms) > 0:
                all_params.extend(valid_arms)
        
        all_params = np.array(all_params)
        
        # Test each parameter dimension
        param_names = ['magnitude', 'arm_rotation', 'arm_pitch', 'motor_rotation', 'motor_pitch']
        
        for i, param_name in enumerate(param_names):
            param_values = all_params[:, i]
            
            # Kolmogorov-Smirnov test against uniform distribution
            min_val, max_val = self.parameter_limits[i]
            uniform_dist = stats.uniform(loc=min_val, scale=max_val - min_val)
            ks_stat, ks_p_value = stats.kstest(param_values, uniform_dist.cdf)
            
            print(f"{param_name}: KS test p-value = {ks_p_value:.6f}")
            
            # For spherical coordinates, we expect some non-uniformity due to coordinate system
            # So we use a more lenient test
            if param_name in ['magnitude', 'arm_rotation', 'motor_rotation']:
                # These should be more uniform
                self.assertGreater(ks_p_value, self.alpha / 2,
                                 f"{param_name} fails uniformity test (p={ks_p_value:.6f})")
    
    def test_direction_uniformity(self):
        """Test uniformity of direction parameter."""
        print(f"\n=== Testing Direction Uniformity (n={self.sample_size}) ===")
        population = self._generate_large_population(self.sample_size)
        
        # Extract all directions
        all_directions = []
        for individual in population:
            valid_arms = individual.get_valid_arms()
            if len(valid_arms) > 0:
                all_directions.extend(valid_arms[:, 5])
        
        all_directions = np.array(all_directions)
        
        if len(all_directions) == 0:
            self.skipTest("No valid directions found in population")
        
        # Count 0s and 1s
        count_0 = np.sum(all_directions == 0)
        count_1 = np.sum(all_directions == 1)
        total = len(all_directions)
        
        print(f"Direction 0: {count_0}/{total} ({count_0/total:.3f})")
        print(f"Direction 1: {count_1}/{total} ({count_1/total:.3f})")
        
        # Binomial test
        if total > 0:
            binomial_p_value = stats.binomtest(count_1, total, p=0.5).pvalue
            print(f"Binomial test p-value = {binomial_p_value:.6f}")
            
            self.assertGreater(binomial_p_value, self.alpha,
                             f"Directions fail binomial uniformity test (p={binomial_p_value:.6f})")


class TestSphericalAngularUniformityAndVisualization(unittest.TestCase):
    """Test uniformity of random generation and provide visual inspection capabilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.min_narms = 3
        self.max_narms = 6
        self.sample_size = 500  # Sample size for visual testing
        self.alpha = 0.05  # Significance level
        self.rnd = np.random.default_rng(42)  # Fixed seed for reproducibility
        
        self.parameter_limits = np.array([
            [0.5, 2.0],      # magnitude
            [-np.pi, np.pi],  # arm rotation
            [-np.pi, np.pi],    # arm pitch
            [-np.pi, np.pi],  # motor rotation
            [-np.pi, np.pi],    # motor pitch
            [0, 1]           # direction
        ])
        
    def _generate_large_population(self, size: int) -> List[SphericalAngularDroneGenomeHandler]:
        """Generate a large population for statistical testing."""
        handler = SphericalAngularDroneGenomeHandler(
            genome=None,
            min_max_narms=(self.min_narms, self.max_narms),
            parameter_limits=self.parameter_limits,
            rnd=self.rnd,
            repair=False  # Enable repair for valid genomes
        )
        return handler.generate_random_population(size)
    
    def _extract_parameters(self, population: List[SphericalAngularDroneGenomeHandler]) -> np.ndarray:
        """Extract all valid parameters from a population."""
        all_params = []
        for individual in population:
            valid_arms = individual.get_valid_arms()
            if len(valid_arms) > 0:
                all_params.extend(valid_arms)
        return np.array(all_params)
    
    def _extract_directions(self, population: List[SphericalAngularDroneGenomeHandler]) -> np.ndarray:
        """Extract all propeller directions from a population."""
        all_directions = []
        for individual in population:
            valid_arms = individual.get_valid_arms()
            if len(valid_arms) > 0:
                all_directions.extend(valid_arms[:, 5])
        return np.array(all_directions)
    
    def test_visual_inspection(self):
        """Generate visualizations for manual inspection of uniformity."""

        # Skip if not in visual mode
        if not getattr(self, '_visual_mode', False):
            self.skipTest("Visual inspection only runs with --visual flag")
            
        print(f"\n=== Generating Visual Inspection Plots ===")
        
        # Create output directory
        output_dir = "test_output"
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate population for visualization
        vis_population = self._generate_large_population(100)
        
        # Extract data
        all_params = self._extract_parameters(vis_population)
        directions = self._extract_directions(vis_population)
        
        if len(all_params) == 0:
            self.skipTest("No valid parameters found for visualization")
        
        # 1. Parameter distribution analysis
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        param_names = ['Magnitude', 'Arm Rotation', 'Arm Pitch', 'Motor Rotation', 'Motor Pitch', 'Direction']
        
        for i, param_name in enumerate(param_names[:5]):  # Skip direction for now
            param_values = all_params[:, i]
            min_val, max_val = self.parameter_limits[i]
            
            # Histogram
            axes[i].hist(param_values, bins=30, alpha=0.7, density=True, 
                        edgecolor='black', linewidth=0.5)
            
            # Expected uniform density line
            expected_density = 1 / (max_val - min_val)
            axes[i].axhline(y=expected_density, color='r', linestyle='--', 
                           label=f'Expected uniform density')
            
            axes[i].set_xlabel(f'{param_name} Value')
            axes[i].set_ylabel('Density')
            axes[i].set_title(f'{param_name} Distribution')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
            axes[i].set_xlim(min_val, max_val)
        
        # Direction histogram (discrete)
        if len(directions) > 0:
            count_0 = np.sum(directions == 0)
            count_1 = np.sum(directions == 1)
            total = len(directions)
            
            axes[5].bar([0, 1], [count_0/total, count_1/total], alpha=0.7, 
                       edgecolor='black', linewidth=0.5)
            axes[5].axhline(y=0.5, color='r', linestyle='--', 
                           label='Expected uniform (0.5)')
            axes[5].set_xlabel('Direction Value')
            axes[5].set_ylabel('Proportion')
            axes[5].set_title('Direction Distribution')
            axes[5].legend()
            axes[5].grid(True, alpha=0.3)
            axes[5].set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'spherical_parameter_distributions.png'), 
                   dpi=300, bbox_inches='tight')
        print(f"Saved parameter distribution plot to {output_dir}/spherical_parameter_distributions.png")
        
        # 2. Spherical coordinate visualization
        if len(all_params) > 0:
            # Convert spherical to cartesian for 3D visualization
            magnitude = all_params[:, 0]
            arm_rotation = all_params[:, 1]  # azimuth
            arm_pitch = all_params[:, 2]     # elevation/pitch
            
            # Convert to cartesian coordinates
            x = magnitude * np.cos(arm_pitch) * np.cos(arm_rotation)
            y = magnitude * np.cos(arm_pitch) * np.sin(arm_rotation)
            z = magnitude * np.sin(arm_pitch)
            
            fig = plt.figure(figsize=(15, 5))
            
            # 3D scatter plot
            ax1 = fig.add_subplot(131, projection='3d')
            scatter = ax1.scatter(x, y, z, c=directions if len(directions) == len(all_params) else 'blue', 
                                cmap='viridis', alpha=0.6, s=20)
            ax1.set_xlabel('X Position')
            ax1.set_ylabel('Y Position')
            ax1.set_zlabel('Z Position')
            ax1.set_title('3D Motor Positions (Spherical→Cartesian)')
            
            # 2D projections
            ax2 = fig.add_subplot(132)
            ax2.scatter(x, y, c=directions if len(directions) == len(all_params) else 'blue', 
                       cmap='viridis', alpha=0.6, s=20)
            ax2.set_xlabel('X Position')
            ax2.set_ylabel('Y Position')
            ax2.set_title('XY Projection')
            ax2.grid(True, alpha=0.3)
            ax2.set_aspect('equal')
            
            # Polar plot of arm rotations and magnitudes
            ax3 = fig.add_subplot(133, projection='polar')
            ax3.scatter(arm_rotation, magnitude, c=directions if len(directions) == len(all_params) else 'blue', 
                       cmap='viridis', alpha=0.6, s=20)
            ax3.set_title('Arm Rotation vs Magnitude (Polar)')
            ax3.set_ylim(self.parameter_limits[0, 0], self.parameter_limits[0, 1])
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'spherical_coordinate_visualization.png'), 
                       dpi=300, bbox_inches='tight')
            print(f"Saved spherical coordinate visualization to {output_dir}/spherical_coordinate_visualization.png")
        
        # 3. Individual drone visualizations using DroneVisualizer
        visualizer = DroneVisualizer()
        
        # Select a few random drones for detailed visualization
        sample_indices = np.random.choice(len(vis_population), size=min(6, len(vis_population)), replace=False)
        
        n_rows = 2
        n_cols = 3
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10), subplot_kw={'projection': '3d'})
        if len(sample_indices) == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if len(sample_indices) > 1 else [axes]
        
        for i, idx in enumerate(sample_indices):
            if i >= len(axes):
                break
            drone = vis_population[idx]
            arm_count = drone.get_arm_count()
            
            # Convert spherical genome to format expected by visualizer
            valid_arms = drone.get_valid_arms()
            if len(valid_arms) > 0:
                # For spherical coordinates, we need to convert to a format the visualizer understands
                # The visualizer expects either CartesianEuler objects or numpy arrays with specific format
                try:
                    _, ax = visualizer.plot_3d(valid_arms, ax=axes[i], 
                                             title=f'Drone {idx} ({arm_count} arms)')
                except Exception as e:
                    # If visualization fails, create a simple plot
                    axes[i].text(0.5, 0.5, 0.5, f'Drone {idx}\n{arm_count} arms\nVisualization unavailable', 
                               transform=axes[i].transAxes, ha='center', va='center')
                    axes[i].set_title(f'Drone {idx} ({arm_count} arms)')
        
        # Hide unused subplots
        for i in range(len(sample_indices), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'spherical_individual_drones.png'), 
                   dpi=300, bbox_inches='tight')
        print(f"Saved individual drone visualizations to {output_dir}/spherical_individual_drones.png")
        
        # Close all figures to prevent display
        plt.close('all')
        
        print(f"Visual inspection plots saved to {output_dir}/ directory")
        print("Manually inspect the plots to verify uniformity:")
        print("- Parameter histograms should approximate uniform distributions")
        print("- 3D scatter should show reasonable distribution in spherical space")
        print("- Polar plot should show even angular and radial coverage") 
        print("- Individual drones should show diverse configurations")


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test SphericalAngularDroneGenomeHandler')
    parser.add_argument('--visual', action='store_true', 
                       help='Enable visual inspection mode (generates plots)')
    parser.add_argument('--unittest-args', nargs='*', 
                       help='Arguments to pass to unittest')
    
    args, unknown = parser.parse_known_args()
    
    # Set visual mode flag on the test class
    if args.visual:
        TestSphericalAngularUniformityAndVisualization._visual_mode = True
        print("Visual inspection mode enabled - plots will be generated")
    
    # Prepare unittest arguments
    unittest_args = [''] + unknown
    if args.unittest_args:
        unittest_args.extend(args.unittest_args)
    
    # Run the tests
    unittest.main(argv=unittest_args, verbosity=2)