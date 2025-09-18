#!/usr/bin/env python3
"""
Statistical uniformity tests for CartesianEulerDroneGenomeHandler.

This module contains rigorous statistical tests to verify that randomly generated 
drone genomes have uniformly distributed parameters including:
- Position uniformity in cube space
- Orientation uniformity in angular space  
- Propeller direction uniformity (binary distribution)
"""

import unittest
import numpy as np
from typing import List

# Statistical testing
from scipy import stats
import warnings

from airevolve.evolution_tools.genome_handlers.cartesian_euler_genome_handler import CartesianEulerDroneGenomeHandler
from test_utilities import extract_positions, extract_orientations, extract_directions


class TestCartesianEulerUniformity(unittest.TestCase):
    """Test uniformity of random generation using statistical methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.expected_narms = 4
        self.max_motor_pos = 1.0
        self.sample_size = 1000  # Large sample for statistical testing
        self.alpha = 0.05  # Significance level
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
        
    def _generate_large_population(self, size: int) -> List[CartesianEulerDroneGenomeHandler]:
        """Generate a large population for statistical testing."""
        handler = CartesianEulerDroneGenomeHandler(
            min_max_narms=self.min_max_narms,
            parameter_limits=self.parameter_limits,
            append_arm_chance=self.append_arm_chance,
            rnd=self.rnd,
            repair=False  # No repair needed for uniformity tests
        )
        return handler.generate_random_population(size)
    
    
    def test_position_uniformity_statistical(self):
        """Test statistical uniformity of motor positions in cube space."""
        print(f"\n=== Testing Position Uniformity (n={self.sample_size}) ===")
        
        population = self._generate_large_population(self.sample_size)
        positions = extract_positions(population)
        
        # Test each coordinate dimension
        coordinate_names = ['X', 'Y', 'Z']
        for i, coord_name in enumerate(coordinate_names):
            coord_values = positions[:, i]
            
            # Kolmogorov-Smirnov test against uniform distribution
            # scipy.stats.uniform uses loc=lower_bound, scale=range
            uniform_dist = stats.uniform(loc=-self.max_motor_pos, scale=2*self.max_motor_pos)
            ks_stat, ks_p_value = stats.kstest(coord_values, uniform_dist.cdf)
            
            print(f"{coord_name}-coordinate: KS test p-value = {ks_p_value:.6f}")
            
            # Test should NOT reject null hypothesis (uniform distribution)
            self.assertGreater(ks_p_value, self.alpha, 
                             f"{coord_name}-coordinate fails uniformity test (p={ks_p_value:.6f})")
            
            # Additional check: Chi-square goodness of fit test
            # Divide the range into bins and test for equal frequencies
            num_bins = 10
            bin_edges = np.linspace(-self.max_motor_pos, self.max_motor_pos, num_bins + 1)
            observed_freq, _ = np.histogram(coord_values, bins=bin_edges)
            expected_freq = len(coord_values) / num_bins
            
            # Chi-square test
            chi2_stat, chi2_p_value = stats.chisquare(observed_freq, [expected_freq] * num_bins)
            
            print(f"{coord_name}-coordinate: Chi-square test p-value = {chi2_p_value:.6f}")
            
            self.assertGreater(chi2_p_value, self.alpha,
                             f"{coord_name}-coordinate fails chi-square uniformity test (p={chi2_p_value:.6f})")
    
    def test_orientation_uniformity_statistical(self):
        """Test statistical uniformity of motor orientations in angular space."""
        print(f"\n=== Testing Orientation Uniformity (n={self.sample_size}) ===")
        
        population = self._generate_large_population(self.sample_size)
        orientations = extract_orientations(population)
        
        # Test each angular dimension
        angle_names = ['Roll', 'Pitch', 'Yaw']
        for i, angle_name in enumerate(angle_names):
            angle_values = orientations[:, i]
            
            # Kolmogorov-Smirnov test against uniform distribution on [-π, π]
            uniform_dist = stats.uniform(loc=-np.pi, scale=2*np.pi)
            ks_stat, ks_p_value = stats.kstest(angle_values, uniform_dist.cdf)
            
            print(f"{angle_name} angle: KS test p-value = {ks_p_value:.6f}")
            
            self.assertGreater(ks_p_value, self.alpha,
                             f"{angle_name} angle fails uniformity test (p={ks_p_value:.6f})")
            
            # Circular uniformity test (Rayleigh test)
            # Null hypothesis: uniform distribution on the circle
            rayleigh_stat, rayleigh_p_value = stats.rayleigh.fit(np.abs(angle_values))
            
            # For circular uniformity, we can also use the range test for circular data
            # For simplicity, we'll use the range test for circular data
            wrapped_angles = np.mod(angle_values + np.pi, 2*np.pi) - np.pi
            range_stat = np.max(wrapped_angles) - np.min(wrapped_angles)
            expected_range = 2*np.pi
            
            print(f"{angle_name} angle: range = {range_stat:.6f}, expected ≈ {expected_range:.6f}")
            
            # Range should be close to 2π for good coverage
            self.assertGreater(range_stat, 0.9 * expected_range,
                             f"{angle_name} angle has insufficient range coverage")
    
    def test_propeller_direction_uniformity(self):
        """Test uniformity of propeller direction (binary) distribution."""
        print(f"\n=== Testing Propeller Direction Uniformity (n={self.sample_size}) ===")
        
        population = self._generate_large_population(self.sample_size)
        directions = extract_directions(population)
        
        # Count 0s and 1s
        count_0 = np.sum(directions == 0)
        count_1 = np.sum(directions == 1)
        total = len(directions)
        
        print(f"Direction 0: {count_0}/{total} ({count_0/total:.3f})")
        print(f"Direction 1: {count_1}/{total} ({count_1/total:.3f})")
        
        # Binomial test: null hypothesis is p=0.5
        # Use two-tailed test
        binomial_p_value = stats.binomtest(count_1, total, p=0.5).pvalue
        
        print(f"Binomial test p-value = {binomial_p_value:.6f}")
        
        self.assertGreater(binomial_p_value, self.alpha,
                         f"Propeller directions fail binomial uniformity test (p={binomial_p_value:.6f})")
        
        # Chi-square goodness of fit test
        expected_freq = total / 2
        observed_freq = np.array([count_0, count_1])
        chi2_stat, chi2_p_value = stats.chisquare(observed_freq, [expected_freq, expected_freq])
        
        print(f"Chi-square test p-value = {chi2_p_value:.6f}")
        
        self.assertGreater(chi2_p_value, self.alpha,
                         f"Propeller directions fail chi-square uniformity test (p={chi2_p_value:.6f})")

    def test_position_variance_consistency(self):
        """Test that position variance is consistent across coordinates."""
        print(f"\n=== Testing Position Variance Consistency (n={self.sample_size}) ===")
        
        population = self._generate_large_population(self.sample_size)
        positions = extract_positions(population)
        
        # Calculate variance for each coordinate
        variances = np.var(positions, axis=0)
        
        # Expected variance for uniform distribution on [-a, a] is a²/3
        expected_variance = (self.max_motor_pos ** 2) / 3
        
        print(f"Expected variance: {expected_variance:.6f}")
        
        for i, coord_name in enumerate(['X', 'Y', 'Z']):
            observed_variance = variances[i]
            print(f"{coord_name}-coordinate variance: {observed_variance:.6f}")
            
            # Check that variance is within reasonable range of expected value
            # Allow for some statistical variation
            tolerance = 0.1 * expected_variance  # 10% tolerance
            self.assertAlmostEqual(observed_variance, expected_variance, delta=tolerance,
                                 msg=f"{coord_name}-coordinate variance deviates too much from expected")

    def test_orientation_variance_consistency(self):
        """Test that orientation variance is consistent across angles."""
        print(f"\n=== Testing Orientation Variance Consistency (n={self.sample_size}) ===")
        
        population = self._generate_large_population(self.sample_size)
        orientations = extract_orientations(population)
        
        # Calculate variance for each angle
        variances = np.var(orientations, axis=0)
        
        # Expected variance for uniform distribution on [-π, π] is (2π)²/12 = π²/3
        expected_variance = (np.pi ** 2) / 3
        
        print(f"Expected variance: {expected_variance:.6f}")
        
        for i, angle_name in enumerate(['Roll', 'Pitch', 'Yaw']):
            observed_variance = variances[i]
            print(f"{angle_name} variance: {observed_variance:.6f}")
            
            # Check that variance is within reasonable range of expected value
            tolerance = 0.1 * expected_variance  # 10% tolerance
            self.assertAlmostEqual(observed_variance, expected_variance, delta=tolerance,
                                 msg=f"{angle_name} variance deviates too much from expected")

    def test_independence_between_parameters(self):
        """Test that different parameters are statistically independent."""
        
        population = self._generate_large_population(self.sample_size)
        
        # Extract all genome data
        all_genomes = np.array([ind.genome for ind in population])
        flattened_data = all_genomes.reshape(-1, 7)
        
        # Test correlation between position coordinates
        positions = flattened_data[:, :3]
        position_corr = np.corrcoef(positions.T)
        
        print("Position correlation matrix:")
        print(position_corr)
        
        # Off-diagonal elements should be close to zero
        for i in range(3):
            for j in range(i+1, 3):
                correlation = position_corr[i, j]
                print(f"Position {i}-{j} correlation: {correlation:.6f}")
                self.assertLess(abs(correlation), 0.1, 
                              f"Positions {i} and {j} are too correlated: {correlation}")
        
        # Test correlation between orientation angles
        orientations = flattened_data[:, 3:6]
        orientation_corr = np.corrcoef(orientations.T)
        
        print("\nOrientation correlation matrix:")
        print(orientation_corr)
        
        # Off-diagonal elements should be close to zero
        for i in range(3):
            for j in range(i+1, 3):
                correlation = orientation_corr[i, j]
                print(f"Orientation {i}-{j} correlation: {correlation:.6f}")
                self.assertLess(abs(correlation), 0.1, 
                              f"Orientations {i} and {j} are too correlated: {correlation}")

    def test_cross_population_consistency(self):
        """Test that multiple populations have consistent statistical properties."""
        
        num_populations = 5
        population_size = 200  # Smaller populations for this test
        
        position_means = []
        orientation_means = []
        direction_proportions = []
        
        for pop_idx in range(num_populations):
            # Use different seeds for each population
            rnd = np.random.default_rng(42 + pop_idx)
            handler = CartesianEulerDroneGenomeHandler(
                min_max_narms=self.min_max_narms,
                parameter_limits=self.parameter_limits,
                append_arm_chance=self.append_arm_chance,
                rnd=rnd,
                repair=False  # No repair needed for uniformity tests
            )
            
            population = handler.generate_random_population(population_size)
            positions = extract_positions(population)
            orientations = extract_orientations(population)
            directions = extract_directions(population)
            
            position_means.append(np.mean(positions, axis=0))
            orientation_means.append(np.mean(orientations, axis=0))
            direction_proportions.append(np.mean(directions))
        
        # Test that means are consistent across populations
        position_means = np.array(position_means)
        orientation_means = np.array(orientation_means)
        direction_proportions = np.array(direction_proportions)
        
        print(f"Position means across populations: {position_means}")
        print(f"Orientation means across populations: {orientation_means}")
        print(f"Direction proportions across populations: {direction_proportions}")
        
        # Position means should be close to zero
        for i in range(3):
            coord_means = position_means[:, i]
            mean_of_means = np.mean(coord_means)
            std_of_means = np.std(coord_means)
            
            print(f"Position coordinate {i}: mean={mean_of_means:.6f}, std={std_of_means:.6f}")
            
            # Mean should be close to zero (expected value for uniform on [-1,1])
            self.assertLess(abs(mean_of_means), 0.1, 
                          f"Position coordinate {i} mean is biased: {mean_of_means}")
            
            # Standard deviation should be reasonable
            self.assertLess(std_of_means, 0.2, 
                          f"Position coordinate {i} has excessive variation: {std_of_means}")
        
        # Orientation means should be close to zero
        for i in range(3):
            angle_means = orientation_means[:, i]
            mean_of_means = np.mean(angle_means)
            std_of_means = np.std(angle_means)
            
            print(f"Orientation angle {i}: mean={mean_of_means:.6f}, std={std_of_means:.6f}")
            
            # Mean should be close to zero (expected value for uniform on [-π,π])
            self.assertLess(abs(mean_of_means), 0.3, 
                          f"Orientation angle {i} mean is biased: {mean_of_means}")
        
        # Direction proportions should be close to 0.5
        mean_proportion = np.mean(direction_proportions)
        std_proportion = np.std(direction_proportions)
        
        print(f"Direction proportions: mean={mean_proportion:.6f}, std={std_proportion:.6f}")
        
        self.assertAlmostEqual(mean_proportion, 0.5, delta=0.1,
                             msg=f"Direction proportion is biased: {mean_proportion}")
        self.assertLess(std_proportion, 0.1,
                       f"Direction proportion has excessive variation: {std_proportion}")


if __name__ == '__main__':
    unittest.main(verbosity=2)