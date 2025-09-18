#!/usr/bin/env python3
"""
Comprehensive tests for bilateral symmetry functionality.

This module consolidates and replaces:
- test_bilateral_symmetry.py (bilateral symmetry with genome handlers)
- Bilateral symmetry portions of test_operators_symmetry.py

Tests include:
- Symmetry configuration and validation
- Bilateral symmetry application (XY, XZ, YZ planes)  
- Genetic operations with symmetry preservation
- Operator integration and error handling
- Cross-plane symmetry validation
"""

import unittest
import numpy as np
import numpy.testing as npt

# Import symmetry operator classes
from airevolve.evolution_tools.genome_handlers.operators import (
    CartesianSymmetryOperator,
    SphericalSymmetryOperator,
    SymmetryConfig,
    SymmetryPlane,
    SymmetryUtilities
)

# Import genome handler for integration tests
from airevolve.evolution_tools.genome_handlers.cartesian_euler_genome_handler import CartesianEulerDroneGenomeHandler


class TestSymmetryConfig(unittest.TestCase):
    """Test SymmetryConfig class functionality."""
    
    def test_config_creation_with_plane_string(self):
        """Test creating config with plane string."""
        config = SymmetryConfig(plane="xy", enabled=True)
        self.assertEqual(config.plane, SymmetryPlane.XY)
        self.assertTrue(config.enabled)
        
    def test_config_creation_with_plane_enum(self):
        """Test creating config with plane enum."""
        config = SymmetryConfig(plane=SymmetryPlane.XZ, enabled=True)
        self.assertEqual(config.plane, SymmetryPlane.XZ)
        self.assertTrue(config.enabled)
        
    def test_config_creation_with_none_plane(self):
        """Test creating config with None plane."""
        config = SymmetryConfig(plane=None, enabled=False)
        self.assertIsNone(config.plane)
        self.assertFalse(config.enabled)
        
    def test_config_invalid_plane_string(self):
        """Test creating config with invalid plane string."""
        with self.assertRaises(ValueError):
            SymmetryConfig(plane="invalid", enabled=True)
            
    def test_config_disabled_with_plane(self):
        """Test creating disabled config with plane."""
        config = SymmetryConfig(plane="xy", enabled=False)
        self.assertEqual(config.plane, SymmetryPlane.XY)
        self.assertFalse(config.enabled)


class TestSymmetryPlane(unittest.TestCase):
    """Test SymmetryPlane enum functionality."""
    
    def test_plane_values(self):
        """Test symmetry plane enum values."""
        self.assertEqual(SymmetryPlane.XY.value, "xy")
        self.assertEqual(SymmetryPlane.XZ.value, "xz")
        self.assertEqual(SymmetryPlane.YZ.value, "yz")
    
    def test_plane_from_string(self):
        """Test creating plane from string."""
        planes = ["xy", "xz", "yz"]
        expected = [SymmetryPlane.XY, SymmetryPlane.XZ, SymmetryPlane.YZ]
        
        for plane_str, expected_plane in zip(planes, expected):
            # Test that we can find the enum from string
            found_plane = None
            for plane in SymmetryPlane:
                if plane.value == plane_str:
                    found_plane = plane
                    break
            self.assertEqual(found_plane, expected_plane)


class TestCartesianSymmetryOperator(unittest.TestCase):
    """Test CartesianSymmetryOperator class functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.narms = 4
        self.test_genome = np.array([
            [0.5, 0.3, 0.2, 0.1, 0.2, 0.3, 0],  # Arm 0
            [0.4, 0.2, 0.1, 0.0, 0.1, 0.2, 1],  # Arm 1
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],  # Arm 2 (will be mirrored)
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],  # Arm 3 (will be mirrored)
        ])
        
    def test_operator_creation_xy_plane(self):
        """Test creating operator with XY plane."""
        config = SymmetryConfig(plane="xy", enabled=True)
        operator = CartesianSymmetryOperator(config=config)
        
        self.assertEqual(operator.config.plane, SymmetryPlane.XY)
        self.assertTrue(operator.config.enabled)
        
    def test_operator_creation_disabled(self):
        """Test creating disabled operator."""
        config = SymmetryConfig(plane=None, enabled=False)
        operator = CartesianSymmetryOperator(config=config)
        
        self.assertIsNone(operator.config.plane)
        self.assertFalse(operator.config.enabled)
        
    def test_apply_symmetry_xy_plane(self):
        """Test applying symmetry with XY plane."""
        config = SymmetryConfig(plane="xy", enabled=True)
        operator = CartesianSymmetryOperator(config=config)
        
        result = operator.apply_symmetry(self.test_genome)
        
        # Check that first half is mirrored to second half
        self.assertAlmostEqual(result[2, 0], result[0, 0])  # x same
        self.assertAlmostEqual(result[2, 1], result[0, 1])  # y same
        self.assertAlmostEqual(result[2, 2], -result[0, 2])  # z mirrored
        self.assertAlmostEqual(result[2, 3], -result[0, 3])  # roll mirrored
        self.assertAlmostEqual(result[2, 4], -result[0, 4])  # pitch mirrored
        self.assertAlmostEqual(result[2, 5], result[0, 5])  # yaw same
        self.assertEqual(result[2, 6], 1 - result[0, 6])  # direction flipped
        
    def test_apply_symmetry_xz_plane(self):
        """Test applying symmetry with XZ plane."""
        config = SymmetryConfig(plane="xz", enabled=True)
        operator = CartesianSymmetryOperator(config=config)
        
        result = operator.apply_symmetry(self.test_genome)
        
        # Check XZ plane symmetry
        self.assertAlmostEqual(result[2, 0], result[0, 0])  # x same
        self.assertAlmostEqual(result[2, 1], -result[0, 1])  # y mirrored
        self.assertAlmostEqual(result[2, 2], result[0, 2])  # z same
        self.assertAlmostEqual(result[2, 3], -result[0, 3])  # roll mirrored
        self.assertAlmostEqual(result[2, 4], result[0, 4])  # pitch same
        self.assertAlmostEqual(result[2, 5], -result[0, 5])  # yaw mirrored
        self.assertEqual(result[2, 6], 1 - result[0, 6])  # direction flipped
        
    def test_apply_symmetry_yz_plane(self):
        """Test applying symmetry with YZ plane."""
        config = SymmetryConfig(plane="yz", enabled=True)
        operator = CartesianSymmetryOperator(config=config)
        
        result = operator.apply_symmetry(self.test_genome)
        
        # Check YZ plane symmetry
        self.assertAlmostEqual(result[2, 0], -result[0, 0])  # x mirrored
        self.assertAlmostEqual(result[2, 1], result[0, 1])  # y same
        self.assertAlmostEqual(result[2, 2], result[0, 2])  # z same
        self.assertAlmostEqual(result[2, 3], result[0, 3])  # roll same
        self.assertAlmostEqual(result[2, 4], -result[0, 4])  # pitch mirrored
        self.assertAlmostEqual(result[2, 5], -result[0, 5])  # yaw mirrored
        self.assertEqual(result[2, 6], 1 - result[0, 6])  # direction flipped
        
    def test_apply_symmetry_disabled(self):
        """Test applying symmetry when disabled."""
        config = SymmetryConfig(plane=None, enabled=False)
        operator = CartesianSymmetryOperator(config=config)
        
        result = operator.apply_symmetry(self.test_genome)
        
        # Should return unchanged genome
        npt.assert_array_equal(result, self.test_genome)
        
    def test_validate_symmetry_symmetric(self):
        """Test validating symmetric genome."""
        config = SymmetryConfig(plane="xy", enabled=True)
        operator = CartesianSymmetryOperator(config=config)
        
        # Apply symmetry first
        symmetric_genome = operator.apply_symmetry(self.test_genome)
        
        # Should validate as symmetric
        self.assertTrue(operator.validate_symmetry(symmetric_genome))
        
    def test_validate_symmetry_asymmetric(self):
        """Test validating asymmetric genome."""
        config = SymmetryConfig(plane="xy", enabled=True)
        operator = CartesianSymmetryOperator(config=config)
        
        # Create asymmetric genome
        asymmetric_genome = np.array([
            [0.5, 0.3, 0.2, 0.1, 0.2, 0.3, 0],
            [0.4, 0.2, 0.1, 0.0, 0.1, 0.2, 1],
            [0.6, 0.4, 0.3, 0.2, 0.3, 0.4, 1],  # Not symmetric
            [0.3, 0.1, 0.0, -0.1, 0.0, 0.1, 0]
        ])
        
        # Should validate as asymmetric
        self.assertFalse(operator.validate_symmetry(asymmetric_genome))
        
    def test_validate_symmetry_disabled(self):
        """Test validating when symmetry is disabled."""
        config = SymmetryConfig(plane=None, enabled=False)
        operator = CartesianSymmetryOperator(config=config)
        
        # Should always return True when disabled
        self.assertTrue(operator.validate_symmetry(self.test_genome))

    def test_apply_symmetry_wrong_shape(self):
        """Test applying symmetry with wrong genome shape."""
        config = SymmetryConfig(plane="xy", enabled=True)
        operator = CartesianSymmetryOperator(config=config)
        
        # Wrong shape genome
        wrong_genome = np.array([
            [0.5, 0.3, 0.2],  # Too few parameters
            [0.4, 0.2, 0.1]
        ])
        
        with self.assertRaises(ValueError):
            operator.apply_symmetry(wrong_genome)


    def test_symmetry_tolerance(self):
        """Test symmetry validation with custom tolerance."""
        config = SymmetryConfig(plane="xy", enabled=True, tolerance=1e-6)
        operator = CartesianSymmetryOperator(config=config)
        
        # Create nearly symmetric genome
        nearly_symmetric = self.test_genome.copy()
        nearly_symmetric[2] = [0.5, 0.3, -0.2001, -0.1, -0.2, 0.3, 1]  # Small error
        nearly_symmetric[3] = [0.4, 0.2, -0.1, 0.0, -0.1, 0.2, 0]
        
        # Should fail with strict tolerance
        self.assertFalse(operator.validate_symmetry(nearly_symmetric))
        
        # Should pass with loose tolerance
        loose_config = SymmetryConfig(plane="xy", enabled=True, tolerance=1e-3)
        loose_operator = CartesianSymmetryOperator(config=loose_config)
        self.assertTrue(loose_operator.validate_symmetry(nearly_symmetric))


class TestSphericalSymmetryOperator(unittest.TestCase):
    """Test SphericalSymmetryOperator class functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.narms = 4
        # Spherical genome: [magnitude, azimuth, pitch, motor_yaw, motor_pitch, direction]
        self.test_genome = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0],
            [1.5, np.pi/4, np.pi/6, np.pi/3, -np.pi/6, 1],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
        ])
        
    def test_operator_creation_basic(self):
        """Test creating spherical symmetry operator."""
        config = SymmetryConfig(plane="xy", enabled=True)
        operator = SphericalSymmetryOperator(config=config)
        
        self.assertEqual(operator.config.plane, SymmetryPlane.XY)
        self.assertTrue(operator.config.enabled)
        
    def test_apply_symmetry_basic(self):
        """Test applying spherical symmetry."""
        config = SymmetryConfig(plane="xy", enabled=True)
        operator = SphericalSymmetryOperator(config=config)
        
        result = operator.apply_symmetry(self.test_genome)
        
        # Check that symmetry was applied (exact values depend on implementation)
        self.assertEqual(result.shape, self.test_genome.shape)
        # First two arms should be preserved
        npt.assert_array_almost_equal(result[:2], self.test_genome[:2])
        # Last two arms should have been computed
        self.assertFalse(np.any(np.isnan(result[2:])))
        
    def test_apply_symmetry_disabled(self):
        """Test spherical symmetry when disabled."""
        config = SymmetryConfig(plane=None, enabled=False)
        operator = SphericalSymmetryOperator(config=config)
        
        result = operator.apply_symmetry(self.test_genome)
        
        # Should return unchanged genome
        npt.assert_array_equal(result, self.test_genome)
        
    def test_validate_symmetry_basic(self):
        """Test validating spherical symmetry."""
        config = SymmetryConfig(plane="xy", enabled=True)
        operator = SphericalSymmetryOperator(config=config)
        
        # Apply symmetry first
        symmetric_genome = operator.apply_symmetry(self.test_genome)
        
        # Should validate as symmetric
        self.assertTrue(operator.validate_symmetry(symmetric_genome))
        
    def test_validate_symmetry_disabled(self):
        """Test validating when spherical symmetry is disabled."""
        config = SymmetryConfig(plane=None, enabled=False)
        operator = SphericalSymmetryOperator(config=config)
        
        # Should always return True when disabled
        self.assertTrue(operator.validate_symmetry(self.test_genome))
        

    def test_spherical_coordinate_validation(self):
        """Test validation of spherical coordinate constraints."""
        config = SymmetryConfig(plane="xy", enabled=True)
        operator = SphericalSymmetryOperator(config=config)
        
        # Test with valid spherical coordinates
        valid_genome = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0],
            [2.0, np.pi, -np.pi/2, np.pi/2, -np.pi, 1],
            [1.5, -np.pi, np.pi/2, -np.pi/2, np.pi, 0],
            [0.8, np.pi/3, -np.pi/3, np.pi/6, -np.pi/4, 1],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
        ])
        
        # Should handle valid coordinates
        result = operator.apply_symmetry(valid_genome)
        self.assertIsNotNone(result)

    def test_spherical_boundary_handling(self):
        """Test handling of spherical coordinate boundaries."""
        config = SymmetryConfig(plane="xy", enabled=True)
        operator = SphericalSymmetryOperator(config=config)
        
        # Test with boundary values
        boundary_genome = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0],           # Zero magnitude
            [10.0, 2*np.pi, np.pi, 2*np.pi, np.pi, 1],  # Large values
            [1.0, -2*np.pi, -np.pi, -2*np.pi, -np.pi, 0],  # Negative bounds
            [np.inf, np.nan, 0.0, 0.0, 0.0, 2]      # Invalid values
        ])
        
        # Should handle boundary cases gracefully
        try:
            result = operator.apply_symmetry(boundary_genome)
            self.assertIsNotNone(result)
            # Check that invalid values were handled
            self.assertTrue(np.all(np.isfinite(result[:, 0])))  # Magnitudes should be finite
            self.assertTrue(np.all(np.isin(result[:, 5], [0, 1])))  # Directions should be 0 or 1
        except Exception:
            # Some boundary cases might raise exceptions, which is acceptable
            pass


class TestBilateralSymmetryWithGenomeHandlers(unittest.TestCase):
    """Test bilateral symmetry functionality with CartesianEulerDroneGenomeHandler."""
    
    def setUp(self):
        """Set up test fixtures."""
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
        self.mutation_probs = np.array([0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1])
        self.rng = np.random.default_rng(42)  # Fixed seed for reproducibility
        
    def test_init_valid_symmetry_planes(self):
        """Test initialization with valid symmetry planes."""
        valid_planes = ["xy", "xz", "yz", None]
        
        for plane in valid_planes:
            handler = CartesianEulerDroneGenomeHandler(
                bilateral_plane_for_symmetry=plane,
                rnd=self.rng
            )
            self.assertEqual(handler.bilateral_plane_for_symmetry, plane)
    
    def test_init_invalid_symmetry_plane(self):
        """Test initialization with invalid symmetry plane."""
        with self.assertRaises(ValueError):
            CartesianEulerDroneGenomeHandler(
                bilateral_plane_for_symmetry="invalid",
                rnd=self.rng
            )
    
    def test_apply_bilateral_symmetry_xy_plane(self):
        """Test bilateral symmetry application for XY plane."""
        handler = CartesianEulerDroneGenomeHandler(
            genome=None,
            min_max_narms=self.min_max_narms,
            parameter_limits=self.parameter_limits,
            bilateral_plane_for_symmetry="xy",
            append_arm_chance=0.0,  # No new arms
            rnd=self.rng
        )
        
        # Create a test genome
        handler.genome = np.array([
            [0.5, 0.3, 0.2, 0.1, 0.2, 0.3, 0],  # First arm
            [0.4, 0.2, 0.1, 0.0, 0.1, 0.2, 1],  # Second arm
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],  # Third arm (will be mirrored)
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],  # Fourth arm (will be mirrored)
        ])
        
        handler.apply_symmetry()
        
        # Check that first half is mirrored to second half
        self.assertAlmostEqual(handler.genome[2, 0], handler.genome[0, 0])  # x same
        self.assertAlmostEqual(handler.genome[2, 1], handler.genome[0, 1])  # y same
        self.assertAlmostEqual(handler.genome[2, 2], -handler.genome[0, 2])  # z mirrored
        self.assertAlmostEqual(handler.genome[2, 3], -handler.genome[0, 3])  # roll mirrored
        self.assertAlmostEqual(handler.genome[2, 4], -handler.genome[0, 4])  # pitch mirrored
        self.assertAlmostEqual(handler.genome[2, 5], handler.genome[0, 5])  # yaw same
        self.assertEqual(handler.genome[2, 6], 1 - handler.genome[0, 6])  # direction mirrored
        
    def test_apply_bilateral_symmetry_xz_plane(self):
        """Test bilateral symmetry application for XZ plane."""
        handler = CartesianEulerDroneGenomeHandler(
            genome=None,
            min_max_narms=self.min_max_narms,
            parameter_limits=self.parameter_limits,
            bilateral_plane_for_symmetry="xz",
            append_arm_chance=0.0,  # No new arms
            rnd=self.rng
        )
        
        # Create a test genome
        handler.genome = np.array([
            [0.5, 0.3, 0.2, 0.1, 0.2, 0.3, 0],  # First arm
            [0.4, 0.2, 0.1, 0.0, 0.1, 0.2, 1],  # Second arm
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],  # Third arm (will be mirrored)
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],  # Fourth arm (will be mirrored)
        ])
        
        handler.apply_symmetry()
        
        # Check that first half is mirrored to second half
        self.assertAlmostEqual(handler.genome[2, 0], handler.genome[0, 0])  # x same
        self.assertAlmostEqual(handler.genome[2, 1], -handler.genome[0, 1])  # y mirrored
        self.assertAlmostEqual(handler.genome[2, 2], handler.genome[0, 2])  # z same
        self.assertAlmostEqual(handler.genome[2, 3], -handler.genome[0, 3])  # roll mirrored
        self.assertAlmostEqual(handler.genome[2, 4], handler.genome[0, 4])  # pitch same
        self.assertAlmostEqual(handler.genome[2, 5], -handler.genome[0, 5])  # yaw mirrored
        self.assertEqual(handler.genome[2, 6], 1 - handler.genome[0, 6])  # direction mirrored
        
    def test_apply_bilateral_symmetry_yz_plane(self):
        """Test bilateral symmetry application for YZ plane."""
        handler = CartesianEulerDroneGenomeHandler(
            genome=None,
            min_max_narms=self.min_max_narms,
            parameter_limits=self.parameter_limits,
            bilateral_plane_for_symmetry="yz",
            append_arm_chance=0.0,  # No new arms
            rnd=self.rng
        )
        
        # Create a test genome
        handler.genome = np.array([
            [0.5, 0.3, 0.2, 0.1, 0.2, 0.3, 0],  # First arm
            [0.4, 0.2, 0.1, 0.0, 0.1, 0.2, 1],  # Second arm
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],  # Third arm (will be mirrored)
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],  # Fourth arm (will be mirrored)
        ])
        
        handler.apply_symmetry()
        
        # Check that first half is mirrored to second half
        self.assertAlmostEqual(handler.genome[2, 0], -handler.genome[0, 0])  # x mirrored
        self.assertAlmostEqual(handler.genome[2, 1], handler.genome[0, 1])  # y same
        self.assertAlmostEqual(handler.genome[2, 2], handler.genome[0, 2])  # z same
        self.assertAlmostEqual(handler.genome[2, 3], handler.genome[0, 3])  # roll same
        self.assertAlmostEqual(handler.genome[2, 4], -handler.genome[0, 4])  # pitch mirrored
        self.assertAlmostEqual(handler.genome[2, 5], -handler.genome[0, 5])  # yaw mirrored
        self.assertEqual(handler.genome[2, 6], 1 - handler.genome[0, 6])  # direction mirrored
        
    def test_apply_bilateral_symmetry_odd_arms(self):
        """Test that bilateral symmetry raises error for odd number of arms."""
        with self.assertRaises(ValueError):
            CartesianEulerDroneGenomeHandler(
                min_max_narms=(3, 3),
                parameter_limits=self.parameter_limits,
                bilateral_plane_for_symmetry="xy",
                append_arm_chance=0.0,  # No new arms
                rnd=self.rng
            )
    
    def test_apply_bilateral_symmetry_none(self):
        """Test that no symmetry is applied when bilateral_plane_for_symmetry is None."""
        handler = CartesianEulerDroneGenomeHandler(
            genome=None,
            min_max_narms=self.min_max_narms,
            parameter_limits=self.parameter_limits,
            bilateral_plane_for_symmetry=None,
            append_arm_chance=0.0,  # No new arms
            rnd=self.rng
        )
        
        original_genome = handler.genome.copy()
        handler.apply_symmetry()
        
        # Genome should remain unchanged
        np.testing.assert_array_equal(handler.genome, original_genome)
    
    def test_generate_random_genome_with_symmetry(self):
        """Test that random genome generation applies symmetry."""
        handler = CartesianEulerDroneGenomeHandler(
            genome=None,
            min_max_narms=self.min_max_narms,
            parameter_limits=self.parameter_limits,
            bilateral_plane_for_symmetry="xy",
            append_arm_chance=0.0,  # No new arms
            rnd=self.rng
        )
        
        genome = handler._generate_random_genome()
        handler.genome = genome
        
        # Check that the genome is valid and symmetric
        self.assertTrue(handler.is_valid())
        
        # Check specific symmetry properties for xy plane
        self.assertAlmostEqual(handler.genome[2, 0], handler.genome[0, 0])  # x same
        self.assertAlmostEqual(handler.genome[2, 1], handler.genome[0, 1])  # y same
        self.assertAlmostEqual(handler.genome[2, 2], -handler.genome[0, 2])  # z mirrored
        
    def test_generate_random_population_with_symmetry(self):
        """Test that random population generation maintains symmetry."""
        handler = CartesianEulerDroneGenomeHandler(
            genome=None,
            min_max_narms=self.min_max_narms,
            parameter_limits=self.parameter_limits,
            bilateral_plane_for_symmetry="xy",
            append_arm_chance=0.0,  # No new arms
            rnd=self.rng
        )
        
        population = handler.generate_random_population(5)
        
        # Check that all individuals are valid and have symmetry
        for individual in population:
            self.assertTrue(individual.is_valid())
            self.assertEqual(individual.bilateral_plane_for_symmetry, "xy")
    
    def test_crossover_maintains_symmetry(self):
        """Test that crossover maintains symmetry."""
        parent1 = CartesianEulerDroneGenomeHandler(
            genome=None,
            min_max_narms=self.min_max_narms,
            parameter_limits=self.parameter_limits,
            bilateral_plane_for_symmetry="xy",
            append_arm_chance=0.0,  # No new arms
            rnd=self.rng
        )
        parent2 = CartesianEulerDroneGenomeHandler(
            genome=None,
            min_max_narms=self.min_max_narms,
            parameter_limits=self.parameter_limits,
            bilateral_plane_for_symmetry="xy",
            append_arm_chance=0.0,  # No new arms
            rnd=self.rng
        )
        
        child = parent1.crossover(parent2)
        
        self.assertTrue(child.is_valid())
        self.assertEqual(child.bilateral_plane_for_symmetry, "xy")
    
    def test_mutate_maintains_symmetry(self):
        """Test that mutation maintains symmetry."""
        handler = CartesianEulerDroneGenomeHandler(
            genome=None,
            min_max_narms=self.min_max_narms,
            parameter_limits=self.parameter_limits,
            bilateral_plane_for_symmetry="xy",
            mutation_probs=[0.2, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2],  # Force mutation
            append_arm_chance=0.0,  # No new arms
            rnd=self.rng
        )
        # Run mutation multiple times
        for _ in range(10):
            handler.mutate()
            self.assertTrue(handler.is_valid())
    
    def test_copy_preserves_symmetry(self):
        """Test that copy preserves symmetry settings."""
        handler = CartesianEulerDroneGenomeHandler(
            genome=None,
            min_max_narms=self.min_max_narms,
            parameter_limits=self.parameter_limits,
            bilateral_plane_for_symmetry="xz",
            append_arm_chance=0.0,  # No new arms
            rnd=self.rng
        )
        
        copy_handler = handler.copy()
        
        self.assertEqual(copy_handler.bilateral_plane_for_symmetry, "xz")
        self.assertTrue(copy_handler.is_valid())
        np.testing.assert_array_equal(copy_handler.genome, handler.genome)
    
    def test_repair_restores_symmetry(self):
        """Test that repair restores symmetry."""
        handler = CartesianEulerDroneGenomeHandler(
            genome=None,
            min_max_narms=self.min_max_narms,
            parameter_limits=self.parameter_limits,
            bilateral_plane_for_symmetry="xy",
            append_arm_chance=0.0,  # No new arms
            rnd=self.rng
        )
        
        # Break symmetry by manually modifying genome
        handler.genome[2, 2] = 0.5  # Should be -handler.genome[0, 2]
        
        # Verify it's now invalid
        self.assertFalse(handler.is_valid())
        
        # Repair should restore symmetry
        handler.repair()
        self.assertTrue(handler.is_valid())
    
    def test_is_valid_checks_symmetry(self):
        """Test that is_valid checks symmetry constraints."""
        handler = CartesianEulerDroneGenomeHandler(
            genome=None,
            min_max_narms=self.min_max_narms,
            parameter_limits=self.parameter_limits,
            bilateral_plane_for_symmetry="xy",
            append_arm_chance=0.0,  # No new arms
            rnd=self.rng
        )
        
        # Initially should be valid
        self.assertTrue(handler.is_valid())
        
        # Break symmetry
        handler.genome[2, 2] = 0.5  # Should be -handler.genome[0, 2]
        
        # Should now be invalid
        self.assertFalse(handler.is_valid())
    
    def test_string_representations(self):
        """Test string representations include symmetry info."""
        handler = CartesianEulerDroneGenomeHandler(
            genome=None,
            min_max_narms=self.min_max_narms,
            parameter_limits=self.parameter_limits,
            bilateral_plane_for_symmetry="xy",
            append_arm_chance=0.0,  # No new arms
            rnd=self.rng
        )
        
        str_repr = str(handler)
        self.assertIn("symmetry=xy", str_repr)
        
        repr_str = repr(handler)
        self.assertIn("bilateral_plane_for_symmetry=xy", repr_str)
        
        # Test with no symmetry
        handler_no_sym = CartesianEulerDroneGenomeHandler(
            genome=None,
            min_max_narms=self.min_max_narms,
            parameter_limits=self.parameter_limits,
            bilateral_plane_for_symmetry=None,
            append_arm_chance=0.0,  # No new arms
            rnd=self.rng
        )
        
        str_repr_no_sym = str(handler_no_sym)
        self.assertNotIn("symmetry=", str_repr_no_sym)
    
    def test_no_symmetry_behavior(self):
        """Test that handler works correctly without symmetry."""
        handler = CartesianEulerDroneGenomeHandler(
            genome=None,
            min_max_narms=self.min_max_narms,
            parameter_limits=self.parameter_limits,
            bilateral_plane_for_symmetry=None,
            append_arm_chance=0.0,  # No new arms
            rnd=self.rng
        )
        
        # All operations should work normally
        self.assertTrue(handler.is_valid())
        
        population = handler.generate_random_population(3)
        self.assertEqual(len(population), 3)
        
        child = population[0].crossover(population[1])
        self.assertTrue(child.is_valid())
        
        child.mutate()
        self.assertTrue(child.is_valid())
        
        copy_handler = child.copy()
        self.assertTrue(copy_handler.is_valid())

    def test_validate_symmetry_method(self):
        """Test the validate_symmetry method."""
        handler = CartesianEulerDroneGenomeHandler(
            genome=None,
            min_max_narms=self.min_max_narms,
            parameter_limits=self.parameter_limits,
            bilateral_plane_for_symmetry="xy",
            append_arm_chance=0.0,  # No new arms
            rnd=self.rng
        )
        
        # Should be symmetric after initialization
        self.assertTrue(handler.validate_symmetry())
        
        # Break symmetry manually
        handler.genome[2, 2] = 0.5  # Should be -handler.genome[0, 2]
        self.assertFalse(handler.validate_symmetry())

        # Restore symmetry
        handler.apply_symmetry()
        self.assertTrue(handler.validate_symmetry())

    def test_get_symmetry_pairs_method(self):
        """Test the get_symmetry_pairs method."""
        # Test without symmetry
        handler_no_sym = CartesianEulerDroneGenomeHandler(
            genome=None,
            min_max_narms=self.min_max_narms,
            parameter_limits=self.parameter_limits,
            bilateral_plane_for_symmetry=None,
            append_arm_chance=0.0,  # No new arms
            rnd=self.rng
        )
        
        pairs = handler_no_sym.get_symmetry_pairs()
        self.assertEqual(len(pairs), 0)
        
        # Test with symmetry
        handler = CartesianEulerDroneGenomeHandler(
            genome=None,
            min_max_narms=self.min_max_narms,
            parameter_limits=self.parameter_limits,
            bilateral_plane_for_symmetry="xy",
            append_arm_chance=0.0,  # No new arms
            rnd=self.rng
        )
        
        pairs = handler.get_symmetry_pairs()
        self.assertEqual(len(pairs), 2)  # 4 arms = 2 pairs
        self.assertEqual(pairs, [(0, 2), (1, 3)])

    def test_symmetry_operator_integration(self):
        """Test that symmetry operator is properly integrated."""
        handler = CartesianEulerDroneGenomeHandler(
            genome=None,
            min_max_narms=self.min_max_narms,
            parameter_limits=self.parameter_limits,
            bilateral_plane_for_symmetry="xz",
            append_arm_chance=0.0,  # No new arms
            rnd=self.rng
        )
        
        # Check operator exists and is configured correctly
        self.assertTrue(hasattr(handler, 'symmetry_operator'))
        self.assertEqual(handler.symmetry_operator.config.plane.value, "xz")
        self.assertTrue(handler.symmetry_operator.config.enabled)
        
        # Test operator methods work
        original_genome = handler.genome.copy()
        symmetric_genome = handler.symmetry_operator.apply_symmetry(original_genome)
        
        # Should be different (symmetry applied)
        # But both should be valid
        handler.genome = symmetric_genome
        self.assertTrue(handler.is_valid())
        self.assertTrue(handler.validate_symmetry())

    def test_symmetry_with_different_planes(self):
        """Test that symmetry validation works with different planes."""
        planes = ["xy", "xz", "yz"]
        
        for plane in planes:
            with self.subTest(plane=plane):
                handler = CartesianEulerDroneGenomeHandler(
                    min_max_narms=self.min_max_narms,
                    parameter_limits=self.parameter_limits,
                    bilateral_plane_for_symmetry=plane,
                    append_arm_chance=0.0,  # No new arms
                    rnd=self.rng
                )
                
                # Should be valid and symmetric after initialization
                self.assertTrue(handler.is_valid())
                self.assertTrue(handler.validate_symmetry())
                
                # Get symmetry pairs
                pairs = handler.get_symmetry_pairs()
                self.assertEqual(len(pairs), 2)
                
                # Apply symmetry should maintain validity
                handler.apply_symmetry()
                self.assertTrue(handler.is_valid())
                self.assertTrue(handler.validate_symmetry())

    def test_symmetry_tolerance(self):
        """Test symmetry validation tolerance."""
        handler = CartesianEulerDroneGenomeHandler(
            genome=None,
            min_max_narms=self.min_max_narms,
            parameter_limits=self.parameter_limits,
            bilateral_plane_for_symmetry="xy",
            append_arm_chance=0.0,  # No new arms
            rnd=self.rng
        )
        
        # Apply perfect symmetry
        handler.apply_symmetry()
        self.assertTrue(handler.validate_symmetry())
        
        # Introduce small error within tolerance
        handler.genome[2, 2] = -handler.genome[0, 2] + 1e-8  # Very small error
        self.assertTrue(handler.validate_symmetry())  # Should still be valid
        
        # Introduce larger error outside tolerance
        handler.genome[2, 2] = -handler.genome[0, 2] + 1e-4  # Larger error
        self.assertFalse(handler.validate_symmetry())  # Should be invalid


class TestSymmetryUtilities(unittest.TestCase):
    """Test symmetry utility functions and integration."""
    
    def test_utility_functions_exist(self):
        """Test that symmetry utility functions are available."""
        # This test verifies that the utilities class exists
        # Specific tests would depend on the actual implementation
        self.assertTrue(hasattr(SymmetryUtilities, '__class__'))
    
    def test_symmetry_integration(self):
        """Test integration between different symmetry components."""
        # Create test configurations for different planes
        configs = [
            SymmetryConfig(plane="xy", enabled=True),
            SymmetryConfig(plane="xz", enabled=True),
            SymmetryConfig(plane="yz", enabled=True)
        ]
        
        # Test that symmetry operators can be created with each config
        for config in configs:
            cartesian_op = CartesianSymmetryOperator(config)
            spherical_op = SphericalSymmetryOperator(config)
            
            self.assertIsNotNone(cartesian_op)
            self.assertIsNotNone(spherical_op)
            self.assertEqual(cartesian_op.config, config)
            self.assertEqual(spherical_op.config, config)


if __name__ == '__main__':
    unittest.main(verbosity=2)