# Unit Tests for AirEvolve2

This directory contains a comprehensive test suite for the AirEvolve2 evolutionary algorithm framework. The tests are organized by functionality and component to ensure clarity and maintainability.

## Test Organization

The test suite is organized into the following categories:

### Core Functionality Tests
- **`test_cartesian_euler_core.py`** - Basic genome handler functionality
- **`test_cartesian_euler_integration.py`** - Integration and performance tests
- **`test_spherical_angular_genome_handler.py`** - Spherical coordinate genome handler
- **`test_operator_integration.py`** - Integration tests for operator system with genome handlers

### Symmetry Tests
- **`test_bilateral_symmetry.py`** - Bilateral symmetry operations and validation
- **`test_spherical_symmetry.py`** - Spherical symmetry operations

### Repair Tests
- **`test_particle_repair.py`** - Comprehensive particle repair, collision detection, and performance tests

### Specialized Tests
- **`test_visualization.py`** - Visualization and plotting capabilities
- **`test_statistical_uniformity.py`** - Statistical validation of random generation

### Utilities
- **`test_utilities.py`** - Shared helper functions and utilities (not a test file, contains common functions used by other tests)
- **`extra_testing_repair.py`** - Debug tool for visualizing particle repair process (not included in test runner)

## Quick Start

### Run All Tests
```bash
# Run the automated test script
python unit_tests/run_all_tests.py

# Or run all tests manually
python -m unittest discover unit_tests/ -v
```

### Run Specific Test Categories
```bash
# Core functionality only
python unit_tests/test_cartesian_euler_core.py
python unit_tests/test_cartesian_euler_integration.py
python unit_tests/test_operator_integration.py

# Symmetry tests only
python unit_tests/test_bilateral_symmetry.py
python unit_tests/test_spherical_symmetry.py

# Repair tests only (default: Cartesian)
python unit_tests/test_particle_repair.py

# Repair tests with Spherical genome handler
python unit_tests/test_particle_repair.py --genome-handler spherical

# Specialized tests
python unit_tests/test_visualization.py --visual
python unit_tests/test_statistical_uniformity.py
```

### Run Tests with Visual Output
```bash
# Generate visualization plots
python unit_tests/test_visualization.py --visual

# Run all tests with visual inspection
python unit_tests/run_all_tests.py --visual
```

## Detailed Test Descriptions

### Core Functionality Tests

#### `test_cartesian_euler_core.py`
Tests fundamental genome handler operations:
- **Initialization** - Empty genome, provided genome, default parameters
- **Random Generation** - Single genomes and populations
- **Genetic Operations** - Crossover, mutation, copying
- **Validation & Repair** - Boundary checking, constraint enforcement
- **Accessors** - Get/set methods for positions, orientations, directions

**Key Test Classes:**
- `TestCartesianEulerDroneGenomeHandler` - Main functionality tests

**Example Usage:**
```bash
python unit_tests/test_cartesian_euler_core.py
```

#### `test_cartesian_euler_integration.py`
Tests integration workflows and performance:
- **Population Operations** - Batch crossover, mutation, repair
- **Evolutionary Workflows** - Complete generation cycles
- **Performance Testing** - Large populations, scaling behavior
- **Memory Efficiency** - Resource usage validation

**Key Test Classes:**
- `TestCartesianEulerIntegration` - Integration workflows
- `TestCartesianEulerPerformance` - Performance benchmarks

**Example Usage:**
```bash
python unit_tests/test_cartesian_euler_integration.py
```

#### `test_operator_integration.py`
Tests integration of operators with genome handlers:
- **Operator System** - Integration of repair, symmetry, and genetic operators
- **Multi-Handler Support** - Cartesian and spherical coordinate systems
- **Workflow Testing** - Complete operator chains and combinations
- **Performance Integration** - Operator performance with different genome types

**Key Test Classes:**
- `TestCartesianOperatorIntegration` - Cartesian genome handler operator integration
- `TestSphericalOperatorIntegration` - Spherical genome handler operator integration (if present)

**Example Usage:**
```bash
python unit_tests/test_operator_integration.py
```

### Operator Tests

#### `test_bilateral_symmetry.py`
Bilateral symmetry operations and validation:
- **Symmetry Configuration** - Plane specification and validation
- **Bilateral Operations** - XY, XZ, YZ plane symmetry
- **Cartesian Symmetry** - Position and orientation mirroring
- **Spherical Symmetry** - Angular coordinate mirroring
- **Symmetry Preservation** - Across genetic operations

**Key Test Classes:**
- `TestCartesianBilateralSymmetry` - Cartesian symmetry operations
- `TestCartesianSymmetryOperator` - Enhanced symmetry operator functionality
- `TestSphericalSymmetryOperator` - Spherical coordinate symmetry

**Example Usage:**
```bash
python unit_tests/test_bilateral_symmetry.py
```

#### `test_spherical_symmetry.py`
Spherical coordinate symmetry operations:
- **Angular Mirroring** - Azimuth and pitch transformations
- **Coordinate Conversion** - Spherical-Cartesian consistency
- **Boundary Handling** - Proper wrap-around behavior

**Example Usage:**
```bash
python unit_tests/test_spherical_symmetry.py
```

#### `test_particle_repair.py`
Comprehensive particle repair and collision detection supporting both coordinate systems:
- **Genome Handler Support** - Cartesian and Spherical coordinate systems
- **Collision Detection** - Accuracy and boundary constraint validation
- **Repair Performance** - Convergence speed and scalability
- **Coordinate Systems** - Cartesian vs spherical repair consistency
- **Symmetry Preservation** - Repair with symmetry constraints
- **Configuration** - Repair strategies and parameter validation
- **Edge Cases** - NaN handling, boundary conditions

**Key Test Classes:**
- `TestCollisionDetectionAccuracy` - Collision detection validation
- `TestRepairPerformance` - Performance benchmarks
- `TestCoordinateSystemIntegration` - Cross-system consistency
- `TestParticleRepairSymmetry` - Symmetry-aware repair
- `TestRepairConfig` - Configuration validation
- `TestRepairStrategy` - Strategy enumeration
- `TestRepairUtilities` - Utility function testing
- `TestCartesianRepairOperator` - Cartesian repair operations
- `TestSphericalRepairOperator` - Spherical repair operations
- `TestParticleRepairEdgeCases` - Edge case handling

**Example Usage:**
```bash
# Run with Cartesian genome handler (default)
python unit_tests/test_particle_repair.py

# Run with Spherical genome handler
python unit_tests/test_particle_repair.py --genome-handler spherical

# Run specific test class with spherical handler
python unit_tests/test_particle_repair.py TestCollisionDetectionAccuracy --genome-handler spherical
```

### Specialized Tests

#### `test_visualization.py`
Visual inspection and plotting capabilities:
- **3D Drone Visualization** - Individual drone rendering
- **Population Analysis** - Distribution and diversity plots
- **Symmetry Visualization** - Symmetric pair highlighting
- **Mutation Effects** - Before/after comparison plots

**Key Test Classes:**
- `TestCartesianEulerVisualization` - Main visualization tests

**Dependencies:** `matplotlib`, `airevolve.inspection_tools`

**Example Usage:**
```bash
# Generate plots for manual inspection
python unit_tests/test_visualization.py --visual

# Test visualization integration without plots
python unit_tests/test_visualization.py
```

**Output:** Saves plots to `test_output/` directory:
- `uniformity_analysis.png` - Distribution analysis
- `individual_drones.png` - Sample drone visualizations
- `population_diversity.png` - Population comparison
- `symmetric_drones.png` - Symmetry visualization
- `mutation_effects.png` - Mutation impact analysis

#### `test_statistical_uniformity.py`
Statistical validation of random generation:
- **Position Uniformity** - Kolmogorov-Smirnov and Chi-square tests
- **Orientation Uniformity** - Angular distribution validation
- **Direction Uniformity** - Binary distribution testing
- **Independence Testing** - Parameter correlation analysis
- **Cross-Population Consistency** - Multiple population validation

**Key Test Classes:**
- `TestCartesianEulerUniformity` - Statistical uniformity tests

**Dependencies:** `scipy.stats`

**Configuration:**
- Sample size: 1000 genomes
- Significance level: α = 0.05
- Test types: KS test, Chi-square, Binomial test

**Example Usage:**
```bash
python unit_tests/test_statistical_uniformity.py
```

**Sample Output:**
```
=== Testing Position Uniformity (n=1000) ===
X-coordinate: KS test p-value = 0.453077
X-coordinate: Chi-square test p-value = 0.076187
Y-coordinate: KS test p-value = 0.217195
...

=== Testing Orientation Uniformity (n=1000) ===
Roll angle: KS test p-value = 0.911214
Roll angle: range = 6.280821, expected ≈ 6.283185
...

=== Testing Propeller Direction Uniformity (n=1000) ===
Direction 0: 1977/4000 (0.494)
Direction 1: 2023/4000 (0.506)
Binomial test p-value = 0.476771
```

## Running Specific Tests

### Individual Test Methods
```bash
# Run specific test method
python unit_tests/test_cartesian_euler_core.py TestCartesianEulerDroneGenomeHandler.test_initialization_empty_genome

# Run specific test class
python unit_tests/test_statistical_uniformity.py TestCartesianEulerUniformity
```

### Test Discovery Patterns
```bash
# Run tests matching pattern
python -m unittest discover unit_tests/ -p "test_*core*" -v
python -m unittest discover unit_tests/ -p "test_*symmetry*" -v
python -m unittest discover unit_tests/ -p "test_*repair*" -v
python -m unittest discover unit_tests/ -p "test_*visual*" -v
```

### Performance Testing
```bash
# Run only performance tests
python unit_tests/test_cartesian_euler_integration.py TestCartesianEulerPerformance
python unit_tests/test_particle_repair.py TestRepairPerformance
```

## Dependencies

### Required Dependencies
- `numpy` - Numerical computations
- `scipy` - Statistical testing and optimization
- `unittest` - Python testing framework

### Optional Dependencies
- `matplotlib` - Visualization (for `--visual` mode)
- `airevolve.inspection_tools.drone_visualizer` - 3D drone visualization

### Install All Dependencies
```bash
# Install core dependencies
pip install numpy scipy

# Install visualization dependencies
pip install matplotlib

# Install project in development mode
pip install -e .
```

## Test Configuration

### Environment Variables
```bash
# Enable verbose output
export UNITTEST_VERBOSE=1

# Set custom test output directory
export TEST_OUTPUT_DIR=/path/to/output
```

### Test Parameters
Most tests use configurable parameters that can be modified in the test files:

**Statistical Tests:**
- `sample_size = 1000` - Number of genomes for statistical testing
- `alpha = 0.05` - Significance level for hypothesis tests

**Performance Tests:**
- Population sizes: 10, 25, 50, 100
- Arm counts: 4, 6, 8, 12, 16
- Time limits: 1-2 seconds per operation

**Visualization Tests:**
- Output resolution: 300 DPI
- Population samples: 50-100 individuals
- Random seed: 42 (for reproducibility)

## Troubleshooting

### Common Issues

#### ImportError: No module named 'airevolve'
```bash
# Install project in development mode
pip install -e .
```

#### ImportError: No module named 'matplotlib'
```bash
# Install visualization dependencies
pip install matplotlib
# Or run without --visual flag
python unit_tests/test_visualization.py
```

#### ImportError: No module named 'scipy'
```bash
# Install scipy for statistical tests
pip install scipy
```

#### Permission denied: test_output/
```bash
# Check write permissions
chmod 755 unit_tests/
mkdir -p unit_tests/test_output
```

### Test Failures

#### Statistical Test Failures
If statistical uniformity tests fail:
1. Check random seed configuration
2. Increase sample size in test
3. Verify genome handler bounds
4. Review mutation probability settings

#### Performance Test Failures
If performance tests fail:
1. Increase time limits in assertions
2. Check system resource availability
3. Reduce population sizes for slower systems
4. Profile specific operations

#### Visualization Test Failures
If visualization tests fail:
1. Ensure matplotlib backend is compatible
2. Check display environment (especially on headless systems)
3. Verify output directory permissions
4. Use `--visual` flag for plot generation

### Getting Help

For additional help:
1. Check the main project documentation in `CLAUDE.md`
2. Review specific test docstrings for detailed information
3. Run tests with `-v` flag for verbose output
4. Check the AirEvolve2 repository issues for known problems

## Contributing

When adding new tests:

1. **Follow naming conventions**: `test_<component>_<functionality>.py`
2. **Add comprehensive docstrings**: Describe test purpose and expected behavior
3. **Use appropriate test categories**: Core, operator, or specialized tests
4. **Include performance considerations**: Avoid overly long-running tests
5. **Update this README**: Add new test descriptions and usage examples
6. **Test dependencies**: Ensure tests handle missing optional dependencies gracefully

### Test Template
```python
#!/usr/bin/env python3
"""
Description of test module purpose.

Tests for [component] including:
- Test category 1
- Test category 2
- Test category 3
"""

import unittest
import numpy as np

class TestNewComponent(unittest.TestCase):
    """Test [component] functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        pass
    
    def test_basic_functionality(self):
        """Test basic [component] functionality."""
        pass

if __name__ == '__main__':
    unittest.main(verbosity=2)
```