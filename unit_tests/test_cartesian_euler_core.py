#!/usr/bin/env python3
"""
Core unit tests for CartesianEulerDroneGenomeHandler.

This module contains the fundamental functionality tests including:
- Initialization and configuration
- Random genome generation
- Basic genetic operations (crossover, mutation)
- Validation and repair
- Getter/setter methods
"""

import unittest
import numpy as np
import numpy.testing as npt
from unittest.mock import MagicMock
from airevolve.evolution_tools.genome_handlers.cartesian_euler_genome_handler import CartesianEulerDroneGenomeHandler

class TestCartesianEulerDroneGenomeHandler(unittest.TestCase):
    """Core unit tests for CartesianEulerDroneGenomeHandler."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.min_max_narms = (4, 4)
        self.parameter_limits = np.array([
            [-1.0, 1.0],    # x position
            [-1.0, 1.0],    # y position  
            [-1.0, 1.0],    # z position
            [-np.pi, np.pi], # roll
            [-np.pi, np.pi], # pitch
            [-np.pi, np.pi], # yaw
            [0, 1]          # direction
        ])
        self.mutation_probs = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
        self.mutation_scales_percentage = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0])
        self.append_arm_chance = 0.0  # Must be 0 when min_narms == max_narms
        self.rnd = np.random.default_rng(42)  # Fixed seed for reproducibility
        # For backward compatibility with existing tests  
        self.expected_narms = self.min_max_narms[1]
        self.max_motor_pos = self.parameter_limits[0, 1]

    def test_initialization_empty_genome(self):
        """Test initialization with empty genome (should generate random genome)."""
        handler = CartesianEulerDroneGenomeHandler(
            genome=None,
            min_max_narms=self.min_max_narms,
            parameter_limits=self.parameter_limits,
            append_arm_chance=self.append_arm_chance,
            rnd=self.rnd
        )
        
        # Check attributes
        self.assertEqual(handler.get_arm_count(), self.expected_narms)
        np.testing.assert_array_equal(handler.parameter_limits, self.parameter_limits)
        self.assertEqual(handler.min_narms, self.min_max_narms[0])
        self.assertEqual(handler.max_narms, self.min_max_narms[1])
        self.assertEqual(handler.genome.shape, (self.expected_narms, 7))
        
        # When genome=None, should generate a random genome (not zeros)
        self.assertFalse(np.allclose(handler.genome, np.zeros((self.expected_narms, 7))))
        
        # Validate the randomly generated genome
        self.assertTrue(handler.is_valid())
        
        # Check that positions are within bounds
        positions = handler.genome[:, :3]
        self.assertTrue(np.all(positions >= -self.max_motor_pos))
        self.assertTrue(np.all(positions <= self.max_motor_pos))
        
        # Check that orientations are within [-π, π]
        orientations = handler.genome[:, 3:6]
        self.assertTrue(np.all(orientations >= -np.pi))
        self.assertTrue(np.all(orientations <= np.pi))
        
        # Check that propeller directions are 0 or 1
        directions = handler.genome[:, 6]
        self.assertTrue(np.all(np.isin(directions, [0, 1])))

    def test_initialization_with_genome(self):
        """Test initialization with provided genome."""
        test_genome = np.random.rand(self.expected_narms, 7)
        handler = CartesianEulerDroneGenomeHandler(
            genome=test_genome,
            min_max_narms=self.min_max_narms,
            parameter_limits=self.parameter_limits,
            append_arm_chance=self.append_arm_chance,
            rnd=self.rnd
        )
        
        npt.assert_array_equal(handler.genome, test_genome)

    def test_initialization_default_mutation_scales(self):
        """Test that default mutation scales are set correctly."""
        handler = CartesianEulerDroneGenomeHandler(append_arm_chance=0.0, rnd=self.rnd)
        # Default mutation scales should be 10% of parameter ranges for first 6 params, 0 for direction
        expected_ranges = handler.parameter_limits[:, 1] - handler.parameter_limits[:, 0]
        expected_scales_percentages = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0])
        expected_scales = expected_ranges * expected_scales_percentages
        np.testing.assert_array_almost_equal(handler.mutation_scales, expected_scales)

    def test_initialization_invalid_mutation_probs(self):
        """Test that invalid mutation probs raise ValueError."""
        with self.assertRaises(ValueError):
            CartesianEulerDroneGenomeHandler(
                mutation_probs=[0.1, 0.1, 0.1],  # Too few elements
                append_arm_chance=0.0,
                rnd=self.rnd
            )

    def test_generate_random_genome(self):
        """Test random genome generation."""
        handler = CartesianEulerDroneGenomeHandler(
            min_max_narms=self.min_max_narms,
            parameter_limits=self.parameter_limits,
            append_arm_chance=self.append_arm_chance,
            rnd=self.rnd
        )
        
        random_genome = handler._generate_random_genome()
        
        # Check shape
        self.assertEqual(random_genome.shape, (self.expected_narms, 7))
        
        # Check position bounds
        positions = random_genome[:, :3]
        self.assertTrue(np.all(positions >= -self.max_motor_pos))
        self.assertTrue(np.all(positions <= self.max_motor_pos))
        
        # Check orientation bounds
        orientations = random_genome[:, 3:6]
        self.assertTrue(np.all(orientations >= -np.pi))
        self.assertTrue(np.all(orientations <= np.pi))
        
        # Check propeller directions
        directions = random_genome[:, 6]
        self.assertTrue(np.all(np.isin(directions, [0, 1])))

    def test_generate_random_population(self):
        """Test random population generation."""
        handler = CartesianEulerDroneGenomeHandler(
            min_max_narms=self.min_max_narms,
            parameter_limits=self.parameter_limits,
            append_arm_chance=self.append_arm_chance,
            rnd=self.rnd
        )
        
        population_size = 10
        population = handler.generate_random_population(population_size)
        
        # Check population size
        self.assertEqual(len(population), population_size)
        
        # Check that all individuals are valid
        for individual in population:
            self.assertIsInstance(individual, CartesianEulerDroneGenomeHandler)
            self.assertTrue(individual.is_valid())
            self.assertEqual(individual.genome.shape, (self.expected_narms, 7))

    def test_crossover_valid_parents(self):
        """Test crossover between two valid parents."""
        parent1 = CartesianEulerDroneGenomeHandler(
            min_max_narms=self.min_max_narms,
            parameter_limits=self.parameter_limits,
            append_arm_chance=self.append_arm_chance,
            rnd=self.rnd
        )
        parent1.genome = np.ones((self.expected_narms, 7))
        
        parent2 = CartesianEulerDroneGenomeHandler(
            min_max_narms=self.min_max_narms,
            parameter_limits=self.parameter_limits,
            append_arm_chance=self.append_arm_chance,
            rnd=self.rnd
        )
        parent2.genome = np.zeros((self.expected_narms, 7))
        
        child = parent1.crossover(parent2)
        
        # Check child properties
        self.assertIsInstance(child, CartesianEulerDroneGenomeHandler)
        self.assertEqual(child.genome.shape, (self.expected_narms, 7))
        self.assertEqual(child.get_arm_count(), self.expected_narms)
        
        # Child genome should contain values from both parents
        self.assertTrue(np.all(np.isin(child.genome, [0, 1])))

    def test_crossover_invalid_parent_type(self):
        """Test crossover with invalid parent type."""
        parent1 = CartesianEulerDroneGenomeHandler(append_arm_chance=0.0, rnd=self.rnd)
        invalid_parent = "not_a_genome_handler"
        
        with self.assertRaises(TypeError):
            parent1.crossover(invalid_parent)

    def test_crossover_mismatched_shapes(self):
        """Test crossover with mismatched genome shapes."""
        parent1 = CartesianEulerDroneGenomeHandler(min_max_narms=(4, 4), append_arm_chance=0.0, rnd=self.rnd)
        parent2 = CartesianEulerDroneGenomeHandler(min_max_narms=(6, 6), append_arm_chance=0.0, rnd=self.rnd)
        
        with self.assertRaises(ValueError):
            parent1.crossover(parent2)

    def test_mutate_position(self):
        """Test mutation of motor positions."""
        handler = CartesianEulerDroneGenomeHandler(
            min_max_narms=self.min_max_narms,
            parameter_limits=self.parameter_limits,
            append_arm_chance=self.append_arm_chance,
            mutation_probs=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),  # Always mutate
            rnd=self.rnd
        )
        
        # Set initial genome to zeros
        handler.genome = np.zeros((self.expected_narms, 7))
        original_genome = handler.genome.copy()
        
        # Create a mock for the rnd object with proper return values
        mock_rnd = MagicMock()
        # First call to choice will be for mutation type (2 = parameter 0), second call for arm selection (0)
        mock_rnd.choice.side_effect = [2, 0]  # Select parameter mutation for parameter 0, then arm 0
        mock_rnd.normal.return_value = 0.1  # Small perturbation
        
        # Replace handler's rnd with the mock
        handler.rnd = mock_rnd
        
        # Force mutation of position parameter
        handler.mutate()
        
        # Check that mutation occurred
        self.assertNotEqual(handler.genome[0, 0], original_genome[0, 0])

    def test_mutate_orientation(self):
        """Test mutation of motor orientations."""
        handler = CartesianEulerDroneGenomeHandler(
            min_max_narms=self.min_max_narms,
            parameter_limits=self.parameter_limits,
            append_arm_chance=self.append_arm_chance,
            mutation_probs=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),  # Always mutate
            rnd=self.rnd
        )
        
        # Set initial genome to zeros
        handler.genome = np.zeros((self.expected_narms, 7))
        original_genome = handler.genome.copy()
        
        # Replace the random generator with a mock object
        mock_rnd = MagicMock()
        # First call to choice will be for mutation type (5 = parameter 3), second call for arm selection (0)
        mock_rnd.choice.side_effect = [5, 0]  # Select parameter mutation for parameter 3 (roll), then arm 0
        mock_rnd.normal.return_value = 0.1  # Small perturbation
        handler.rnd = mock_rnd
        
        # Perform mutation
        handler.mutate()
        
        # Check that mutation occurred
        self.assertNotEqual(handler.genome[0, 3], original_genome[0, 3])
        
        # Check bounds are respected
        self.assertTrue(np.all(np.abs(handler.genome[:, 3:6]) <= np.pi))

    def test_mutate_propeller_direction(self):
        """Test mutation of propeller direction."""
        handler = CartesianEulerDroneGenomeHandler(
            min_max_narms=self.min_max_narms,
            parameter_limits=self.parameter_limits,
            append_arm_chance=self.append_arm_chance,
            mutation_probs=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),  # Always mutate
            rnd=self.rnd
        )
        
        # Set initial propeller direction to 0
        handler.genome = np.zeros((self.expected_narms, 7))
        
        # Mock the rnd object
        mock_rnd = MagicMock()
        # First call to choice will be for mutation type (8 = parameter 6), second call for arm selection (0)
        mock_rnd.choice.side_effect = [8, 0]  # Select parameter mutation for parameter 6 (direction), then arm 0
        
        # Replace handler's rnd with the mock
        handler.rnd = mock_rnd
        
        # Force mutation of propeller direction
        handler.mutate()
        
        # Check that direction flipped from 0 to 1
        self.assertEqual(handler.genome[0, 6], 1)

    def test_mutate_no_mutation(self):
        """Test that mutation doesn't occur when probability is low."""
        handler = CartesianEulerDroneGenomeHandler(
            min_max_narms=self.min_max_narms,
            parameter_limits=self.parameter_limits,
            append_arm_chance=0.0,  # No add/remove arm mutations  
            mutation_probs=np.array([1/7, 1/7, 1/7, 1/7, 1/7, 1/7, 1/7]),  # Equal but will be very unlikely due to mocking
            rnd=self.rnd
        )
        
        original_genome = handler.genome.copy()

        # Mock to ensure no actual mutation occurs
        mock_rnd = MagicMock()
        mock_rnd.choice.side_effect = [2, 0]  # Parameter mutation type, arm 0
        mock_rnd.normal.return_value = 0.0  # No perturbation
        handler.rnd = mock_rnd

        handler.mutate()
        
        # Genome should be unchanged
        npt.assert_allclose(handler.genome, original_genome)

    def test_copy(self):
        """Test genome handler copying."""
        original = CartesianEulerDroneGenomeHandler(
            min_max_narms=self.min_max_narms,
            parameter_limits=self.parameter_limits,
            append_arm_chance=self.append_arm_chance,
            mutation_scales_percentage=self.mutation_scales_percentage,
            rnd=self.rnd
        )
        original.genome = np.random.rand(self.expected_narms, 7)
        
        copy_handler = original.copy()
        
        # Check that copy has same attributes
        self.assertEqual(copy_handler.get_arm_count(), original.get_arm_count())
        np.testing.assert_array_almost_equal(copy_handler.parameter_limits, original.parameter_limits)
        np.testing.assert_array_almost_equal(copy_handler.mutation_scales, original.mutation_scales)
        
        # Check that genome is copied (not referenced)
        npt.assert_array_equal(copy_handler.genome, original.genome)
        self.assertIsNot(copy_handler.genome, original.genome)
        
        # Modify copy and ensure original is unchanged
        copy_handler.genome[0, 0] = 999
        self.assertNotEqual(copy_handler.genome[0, 0], original.genome[0, 0])

    def test_is_valid_valid_genome(self):
        """Test validation of valid genome."""
        handler = CartesianEulerDroneGenomeHandler(
            min_max_narms=self.min_max_narms,
            parameter_limits=self.parameter_limits,
            append_arm_chance=self.append_arm_chance,
            rnd=self.rnd
        )
        
        # Set valid genome
        handler.genome = np.array([
            [0.5, -0.3, 0.8, 0.1, -0.2, 1.0, 1],
            [-0.2, 0.7, -0.1, -0.5, 0.8, -1.2, 0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1],
            [1.0, -1.0, 0.5, 3.0, -2.5, 1.5, 0]
        ])
        
        self.assertTrue(handler.is_valid())

    def test_is_valid_invalid_shape(self):
        """Test validation with invalid genome shape."""
        handler = CartesianEulerDroneGenomeHandler(
            min_max_narms=self.min_max_narms,
            parameter_limits=self.parameter_limits,
            append_arm_chance=self.append_arm_chance,
            rnd=self.rnd
        )
        
        # Set invalid shape
        handler.genome = np.ones((3, 7))  # Wrong number of arms
        self.assertFalse(handler.is_valid())
        
        handler.genome = np.ones((self.expected_narms, 6))  # Wrong number of parameters
        self.assertFalse(handler.is_valid())

    def test_is_valid_nan_values(self):
        """Test validation with NaN values."""
        handler = CartesianEulerDroneGenomeHandler(
            min_max_narms=self.min_max_narms,
            parameter_limits=self.parameter_limits,
            append_arm_chance=self.append_arm_chance,
            rnd=self.rnd
        )
        
        handler.genome = np.ones((self.expected_narms, 7))
        handler.genome[0, 0] = np.nan
        self.assertFalse(handler.is_valid())

    def test_is_valid_out_of_bounds_position(self):
        """Test validation with out-of-bounds positions."""
        handler = CartesianEulerDroneGenomeHandler(
            min_max_narms=self.min_max_narms,
            parameter_limits=np.array([
                [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0],
                [-np.pi, np.pi], [-np.pi, np.pi], [-np.pi, np.pi],
                [0, 1]
            ]),
            append_arm_chance=self.append_arm_chance,
            rnd=self.rnd
        )
        
        handler.genome = np.zeros((self.expected_narms, 7))
        handler.genome[0, 0] = 2.0  # Exceeds max_motor_pos
        self.assertFalse(handler.is_valid())

    def test_is_valid_out_of_bounds_orientation(self):
        """Test validation with out-of-bounds orientations."""
        handler = CartesianEulerDroneGenomeHandler(
            min_max_narms=self.min_max_narms,
            parameter_limits=self.parameter_limits,
            append_arm_chance=self.append_arm_chance,
            rnd=self.rnd
        )
        
        handler.genome = np.zeros((self.expected_narms, 7))
        handler.genome[0, 3] = 4.0  # Exceeds π
        self.assertFalse(handler.is_valid())

    def test_is_valid_invalid_direction(self):
        """Test validation with invalid propeller direction."""
        handler = CartesianEulerDroneGenomeHandler(
            min_max_narms=self.min_max_narms,
            parameter_limits=self.parameter_limits,
            append_arm_chance=self.append_arm_chance,
            rnd=self.rnd
        )
        
        handler.genome = np.zeros((self.expected_narms, 7))
        handler.genome[0, 6] = 0.5  # Not 0 or 1
        self.assertFalse(handler.is_valid())

    def test_repair_out_of_bounds(self):
        """Test repair of out-of-bounds values."""
        handler = CartesianEulerDroneGenomeHandler(
            min_max_narms=self.min_max_narms,
            parameter_limits=np.array([
                [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0],
                [-np.pi, np.pi], [-np.pi, np.pi], [-np.pi, np.pi],
                [0, 1]
            ]),
            append_arm_chance=self.append_arm_chance,
            rnd=self.rnd
        )
        
        # Set out-of-bounds values
        handler.genome = np.array([
            [2.0, -3.0, 1.5, 4.0, -5.0, 2.0, 1],  # All out of bounds
            [0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 0],   # Some out of bounds
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1],   # Direction out of bounds
            [1.0, -1.0, 0.5, 3.0, -3.0, 1.5, 0] # Mixed
        ])
        
        handler.repair()
        # Check that all values are now within bounds
        self.assertTrue(handler.is_valid())
        
        # Check specific bounds
        self.assertTrue(np.all(np.abs(handler.genome[:, :3]) <= 1.0))
        self.assertTrue(np.all(np.abs(handler.genome[:, 3:6]) <= np.pi))
        self.assertTrue(np.all(np.isin(handler.genome[:, 6], [0, 1])))

    def test_repair_out_of_bounds_values(self):
        """Test repair of out-of-bounds values only (not NaN/inf values)."""
        handler = CartesianEulerDroneGenomeHandler(
            min_max_narms=self.min_max_narms,
            parameter_limits=self.parameter_limits,
            append_arm_chance=self.append_arm_chance,
            rnd=self.rnd
        )
        
        # Set some out-of-bounds values (but not NaN/inf)
        handler.genome = np.ones((self.expected_narms, 7))
        handler.genome[0, 0] = 5.0  # Out of position bounds
        handler.genome[1, 3] = 10.0  # Out of orientation bounds
        
        self.assertFalse(handler.is_valid())
        
        handler.repair()

        # Should be valid after repair
        self.assertTrue(handler.is_valid())
        # Check that all values are finite and within bounds
        self.assertTrue(np.isfinite(handler.genome).all())
        self.assertTrue(np.all(np.abs(handler.genome[:, :3]) <= 1.0))
        self.assertTrue(np.all(np.abs(handler.genome[:, 3:6]) <= np.pi))
        self.assertTrue(np.all(np.isin(handler.genome[:, 6], [0, 1])))

    def test_get_motor_positions(self):
        """Test getting motor positions."""
        handler = CartesianEulerDroneGenomeHandler(
            min_max_narms=self.min_max_narms,
            parameter_limits=self.parameter_limits,
            append_arm_chance=self.append_arm_chance,
            rnd=self.rnd
        )
        
        test_positions = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
            [1.0, -1.0, 0.0]
        ])
        handler.genome[:, :3] = test_positions
        
        positions = handler.get_motor_positions()
        npt.assert_array_equal(positions, test_positions)
        
        # Ensure it's a copy, not a reference
        positions[0, 0] = 999
        self.assertNotEqual(handler.genome[0, 0], 999)

    def test_get_motor_orientations(self):
        """Test getting motor orientations."""
        handler = CartesianEulerDroneGenomeHandler(
            min_max_narms=self.min_max_narms,
            parameter_limits=self.parameter_limits,
            append_arm_chance=self.append_arm_chance,
            rnd=self.rnd
        )
        
        test_orientations = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
            [1.0, -1.0, 0.0]
        ])
        handler.genome[:, 3:6] = test_orientations
        
        orientations = handler.get_motor_orientations()
        npt.assert_array_equal(orientations, test_orientations)
        
        # Ensure it's a copy, not a reference
        orientations[0, 0] = 999
        self.assertNotEqual(handler.genome[0, 3], 999)

    def test_get_propeller_directions(self):
        """Test getting propeller directions."""
        handler = CartesianEulerDroneGenomeHandler(
            min_max_narms=self.min_max_narms,
            parameter_limits=self.parameter_limits,
            append_arm_chance=self.append_arm_chance,
            rnd=self.rnd
        )
        
        test_directions = np.array([0, 1, 1, 0])
        handler.genome[:, 6] = test_directions
        
        directions = handler.get_propeller_directions()
        npt.assert_array_equal(directions, test_directions)
        
        # Ensure it's a copy, not a reference
        directions[0] = 999
        self.assertNotEqual(handler.genome[0, 6], 999)

    def test_set_motor_position(self):
        """Test setting motor position."""
        handler = CartesianEulerDroneGenomeHandler(
            min_max_narms=self.min_max_narms,
            parameter_limits=self.parameter_limits,
            append_arm_chance=self.append_arm_chance,
            rnd=self.rnd
        )
        
        new_position = np.array([0.1, 0.2, 0.3])
        handler.set_motor_position(0, new_position)
        
        npt.assert_array_equal(handler.genome[0, :3], new_position)

    def test_set_motor_position_invalid_index(self):
        """Test setting motor position with invalid arm index."""
        handler = CartesianEulerDroneGenomeHandler(
            min_max_narms=self.min_max_narms,
            parameter_limits=self.parameter_limits,
            append_arm_chance=self.append_arm_chance,
            rnd=self.rnd
        )
        
        with self.assertRaises(ValueError):
            handler.set_motor_position(self.expected_narms, np.array([0.1, 0.2, 0.3]))
        
        with self.assertRaises(ValueError):
            handler.set_motor_position(-1, np.array([0.1, 0.2, 0.3]))

    def test_set_motor_position_invalid_length(self):
        """Test setting motor position with invalid position length."""
        handler = CartesianEulerDroneGenomeHandler(
            min_max_narms=self.min_max_narms,
            parameter_limits=self.parameter_limits,
            append_arm_chance=self.append_arm_chance,
            rnd=self.rnd
        )
        
        with self.assertRaises(ValueError):
            handler.set_motor_position(0, np.array([0.1, 0.2]))  # Too short
        
        with self.assertRaises(ValueError):
            handler.set_motor_position(0, np.array([0.1, 0.2, 0.3, 0.4]))  # Too long

    def test_set_motor_orientation(self):
        """Test setting motor orientation."""
        handler = CartesianEulerDroneGenomeHandler(
            min_max_narms=self.min_max_narms,
            parameter_limits=self.parameter_limits,
            append_arm_chance=self.append_arm_chance,
            rnd=self.rnd
        )
        
        new_orientation = np.array([0.4, 0.5, 0.6])
        handler.set_motor_orientation(1, new_orientation)
        
        npt.assert_array_equal(handler.genome[1, 3:6], new_orientation)

    def test_set_motor_orientation_invalid_index(self):
        """Test setting motor orientation with invalid arm index."""
        handler = CartesianEulerDroneGenomeHandler(
            min_max_narms=self.min_max_narms,
            parameter_limits=self.parameter_limits,
            append_arm_chance=self.append_arm_chance,
            rnd=self.rnd
        )
        
        with self.assertRaises(ValueError):
            handler.set_motor_orientation(self.expected_narms, np.array([0.1, 0.2, 0.3]))
        
        with self.assertRaises(ValueError):
            handler.set_motor_orientation(-1, np.array([0.1, 0.2, 0.3]))

    def test_set_motor_orientation_invalid_length(self):
        """Test setting motor orientation with invalid orientation length."""
        handler = CartesianEulerDroneGenomeHandler(
            min_max_narms=self.min_max_narms,
            parameter_limits=self.parameter_limits,
            append_arm_chance=self.append_arm_chance,
            rnd=self.rnd
        )
        
        with self.assertRaises(ValueError):
            handler.set_motor_orientation(0, np.array([0.1, 0.2]))  # Too short
        
        with self.assertRaises(ValueError):
            handler.set_motor_orientation(0, np.array([0.1, 0.2, 0.3, 0.4]))  # Too long

    def test_set_propeller_direction(self):
        """Test setting propeller direction."""
        handler = CartesianEulerDroneGenomeHandler(
            min_max_narms=self.min_max_narms,
            parameter_limits=self.parameter_limits,
            append_arm_chance=self.append_arm_chance,
            rnd=self.rnd
        )
        
        handler.set_propeller_direction(2, 1)
        self.assertEqual(handler.genome[2, 6], 1)
        
        handler.set_propeller_direction(2, 0)
        self.assertEqual(handler.genome[2, 6], 0)

    def test_set_propeller_direction_invalid_index(self):
        """Test setting propeller direction with invalid arm index."""
        handler = CartesianEulerDroneGenomeHandler(
            min_max_narms=self.min_max_narms,
            parameter_limits=self.parameter_limits,
            append_arm_chance=self.append_arm_chance,
            rnd=self.rnd
        )
        
        with self.assertRaises(ValueError):
            handler.set_propeller_direction(self.expected_narms, 1)
        
        with self.assertRaises(ValueError):
            handler.set_propeller_direction(-1, 1)

    def test_set_propeller_direction_invalid_direction(self):
        """Test setting propeller direction with invalid direction value."""
        handler = CartesianEulerDroneGenomeHandler(
            min_max_narms=self.min_max_narms,
            parameter_limits=self.parameter_limits,
            append_arm_chance=self.append_arm_chance,
            rnd=self.rnd
        )
        
        with self.assertRaises(ValueError):
            handler.set_propeller_direction(0, 2)  # Invalid direction
        
        with self.assertRaises(ValueError):
            handler.set_propeller_direction(0, -1)  # Invalid direction

    def test_str_representation(self):
        """Test string representation."""
        handler = CartesianEulerDroneGenomeHandler(
            min_max_narms=self.min_max_narms,
            parameter_limits=self.parameter_limits,
            append_arm_chance=self.append_arm_chance,
            rnd=self.rnd
        )
        
        str_repr = str(handler)
        self.assertIn("CartesianEulerDroneGenomeHandler", str_repr)
        self.assertIn(f"max_narms={self.expected_narms}", str_repr)
        self.assertIn(f"shape=({self.expected_narms}, 7)", str_repr)

    def test_repr_representation(self):
        """Test detailed representation."""
        handler = CartesianEulerDroneGenomeHandler(
            min_max_narms=self.min_max_narms,
            parameter_limits=self.parameter_limits,
            append_arm_chance=self.append_arm_chance,
            mutation_probs=np.array([0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]),
            rnd=self.rnd
        )
        
        repr_str = repr(handler)
        self.assertIn("CartesianEulerDroneGenomeHandler", repr_str)
        self.assertIn(f"min_max_narms=({self.expected_narms}, {self.expected_narms})", repr_str)
        self.assertIn(f"genome_shape=({self.expected_narms}, 7)", repr_str)

    def test_get_arm_count(self):
        """Test getting arm count."""
        handler = CartesianEulerDroneGenomeHandler(
            min_max_narms=self.min_max_narms,
            parameter_limits=self.parameter_limits,
            append_arm_chance=self.append_arm_chance,
            rnd=self.rnd
        )
        
        self.assertEqual(handler.get_arm_count(), self.expected_narms)
        
        # Test with different arm counts
        handler_8 = CartesianEulerDroneGenomeHandler(min_max_narms=(8, 8), append_arm_chance=0.0, rnd=self.rnd)
        self.assertEqual(handler_8.get_arm_count(), 8)


if __name__ == '__main__':
    unittest.main(verbosity=2)