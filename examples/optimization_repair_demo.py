"""
Demonstration of the optimization-based repair operator.

This example shows how to use the OptimizationBasedRepairOperator to repair
drone genomes with collisions using constrained optimization.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from airevolve.evolution_tools.genome_handlers.operators.optimization_repair_operator import (
    OptimizationBasedRepairOperator,
    OptimizationRepairConfig,
    optimization_repair_individual,
)
from airevolve.evolution_tools.genome_handlers.operators.repair_base import RepairConfig


def create_collision_genome():
    """
    Create a test genome with intentional collisions.

    Returns a 4-arm drone with arms positioned to collide.
    """
    # Create 4 arms in spherical coordinates: [r, theta, phi, pitch, yaw, direction]
    genome = np.array([
        [0.15, 0.0, np.pi/4, 0.0, 0.0, 1],      # Arm 1
        [0.15, 0.1, np.pi/4, 0.0, 0.0, 1],      # Arm 2 - close to Arm 1 (collision!)
        [0.2, np.pi, np.pi/3, 0.0, 0.0, 1],     # Arm 3
        [0.2, -np.pi, np.pi/3, 0.0, 0.0, 1],    # Arm 4 - close to Arm 3 (collision!)
    ])

    return genome


def create_core_collision_genome():
    """
    Create a test genome where arms pass through the central core.
    """
    # Create arms that would intersect the core sphere
    genome = np.array([
        [0.3, 0.0, np.pi/6, 0.5, 0.0, 1],       # Arm angled toward core
        [0.3, np.pi/2, np.pi/6, 0.5, 0.0, 1],
        [0.3, np.pi, np.pi/6, 0.5, 0.0, 1],
        [0.3, -np.pi/2, np.pi/6, 0.5, 0.0, 1],
    ])

    return genome


def demo_basic_repair():
    """Demonstrate basic repair of a collision genome."""
    print("=" * 70)
    print("DEMO 1: Basic Collision Repair")
    print("=" * 70)

    # Create collision genome
    genome = create_collision_genome()
    print("\nOriginal genome (with collisions):")
    print(genome)

    # Create repair operator
    repair_op = OptimizationBasedRepairOperator(
        coordinate_system='spherical',
        verbose=True
    )

    # Check if genome has collisions
    is_valid = repair_op.validate(genome)
    print(f"\nOriginal genome valid: {is_valid}")

    # Repair genome
    print("\nRepairing genome using optimization...")
    repaired_genome = repair_op.repair(genome)

    print("\nRepaired genome:")
    print(repaired_genome)

    # Validate repaired genome
    is_valid_after = repair_op.validate(repaired_genome)
    print(f"\nRepaired genome valid: {is_valid_after}")

    # Show changes
    changes = np.abs(repaired_genome - genome)
    print("\nAbsolute changes per parameter:")
    print(changes)
    print(f"\nTotal change (L2 norm): {np.linalg.norm(changes):.6f}")


def demo_custom_config():
    """Demonstrate repair with custom configuration."""
    print("\n\n" + "=" * 70)
    print("DEMO 2: Repair with Custom Configuration")
    print("=" * 70)

    # Create genome
    genome = create_core_collision_genome()
    print("\nOriginal genome:")
    print(genome)

    # Create custom optimization config
    opt_config = OptimizationRepairConfig(
        disc_radius=0.12,           # Smaller disc
        disc_height=0.0,
        core_radius=0.06,           # Larger core
        propeller_radius=0.0762,
        optimization_method='SLSQP',
        max_iterations=500,
        constraint_tolerance=1e-6,
    )

    # Create repair operator with custom config
    repair_op = OptimizationBasedRepairOperator(
        optimization_config=opt_config,
        coordinate_system='spherical',
        verbose=True
    )

    # Repair
    print("\nRepairing with custom configuration...")
    repaired_genome = repair_op.repair(genome)

    print("\nRepaired genome:")
    print(repaired_genome)

    is_valid = repair_op.validate(repaired_genome)
    print(f"\nRepaired genome valid: {is_valid}")


def demo_comparison_with_original():
    """Compare optimization repair with no repair."""
    print("\n\n" + "=" * 70)
    print("DEMO 3: Comparison - Before and After")
    print("=" * 70)

    # Create collision genome
    genome = create_collision_genome()

    # Create repair operator
    repair_op = OptimizationBasedRepairOperator(
        coordinate_system='spherical',
        verbose=False  # Suppress optimization output
    )

    # Repair
    repaired_genome = repair_op.repair(genome)

    # Display comparison
    print("\nParameter-by-parameter comparison:")
    print("-" * 70)
    param_names = ['r', 'theta', 'phi', 'pitch', 'yaw', 'direction']

    for arm_idx in range(len(genome)):
        print(f"\nArm {arm_idx + 1}:")
        for param_idx, param_name in enumerate(param_names):
            orig_val = genome[arm_idx, param_idx]
            repaired_val = repaired_genome[arm_idx, param_idx]
            change = repaired_val - orig_val

            print(f"  {param_name:10s}: {orig_val:8.4f} -> {repaired_val:8.4f} "
                  f"(change: {change:+8.4f})")

    # Summary statistics
    print("\n" + "-" * 70)
    print("Summary Statistics:")
    print(f"  Mean absolute change: {np.mean(np.abs(repaired_genome - genome)):.6f}")
    print(f"  Max absolute change:  {np.max(np.abs(repaired_genome - genome)):.6f}")
    print(f"  Total L2 change:      {np.linalg.norm(repaired_genome - genome):.6f}")


def demo_functional_api():
    """Demonstrate using the functional API directly."""
    print("\n\n" + "=" * 70)
    print("DEMO 4: Using Functional API")
    print("=" * 70)

    # Create genome
    genome = create_collision_genome()
    print("\nOriginal genome:")
    print(genome)

    # Create config
    config = OptimizationRepairConfig(
        disc_radius=0.15,
        core_radius=0.05,
        max_iterations=500,
    )

    # Use functional API directly
    print("\nRepairing using functional API...")
    repaired_genome = optimization_repair_individual(
        genome,
        config=config,
        verbose=True
    )

    print("\nRepaired genome:")
    print(repaired_genome)


if __name__ == "__main__":
    print("\n")
    print("*" * 70)
    print("OPTIMIZATION-BASED REPAIR OPERATOR DEMONSTRATION")
    print("*" * 70)

    # Run demos
    demo_basic_repair()
    demo_custom_config()
    demo_comparison_with_original()
    demo_functional_api()

    print("\n" + "*" * 70)
    print("DEMONSTRATION COMPLETE")
    print("*" * 70)
    print("\nKey takeaways:")
    print("1. The optimization repair minimizes changes to the original genome")
    print("2. It respects disc-based attachment geometry")
    print("3. It ensures no cylinder-cylinder or cylinder-core collisions")
    print("4. Configuration is flexible and customizable")
    print("5. Both class-based and functional APIs are available")
    print("\n")
