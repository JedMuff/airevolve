"""
Genome to CAD Parameter Adapter

This module provides conversion functions between the genome representation used
in SphericalAngularDroneGenomeHandler and the parameters needed for CAD generation.

Genome Format (SphericalAngularDroneGenomeHandler):
    - Shape: (max_narms, 6) with NaN masking for variable arm counts
    - Columns: [magnitude, arm_rotation, arm_pitch, motor_rotation, motor_pitch, direction]
    - Spherical coordinates + motor orientations
    - Angles in radians

CAD Parameter Format:
    - arm_tilt: Tilt angle for arm attachment (rotation around Y axis) in degrees
    - arm_azimuth: Azimuth angle for arm attachment (rotation around Z axis) in degrees
    - motor_tilt: Tilt angle for motor disc (rotation around Y axis) in degrees
    - motor_azimuth: Azimuth angle for motor disc (rotation around Z axis) in degrees
    - arm_attachment_angle: Angle around the disc edge (0-360°) where arm attaches
    - arm_length: Length of the arm cylinder in mm
"""

import numpy as np
import numpy.typing as npt
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass


@dataclass
class ArmCADParameters:
    """Parameters for a single arm needed by CAD generation."""
    arm_tilt: float  # degrees
    arm_azimuth: float  # degrees
    motor_tilt: float  # degrees
    motor_azimuth: float  # degrees
    arm_attachment_angle: float  # degrees, 0-360
    arm_length: float  # mm
    direction: int  # 0 or 1 for propeller rotation direction


@dataclass
class DroneCADParameters:
    """Complete set of parameters for drone CAD generation."""
    arms: List[ArmCADParameters]
    num_arms: int


def radians_to_degrees(angle_rad: float) -> float:
    """Convert radians to degrees."""
    return np.degrees(angle_rad)


def genome_to_cad_parameters(
    genome_handler,
    magnitude_to_length_scale: float = 100.0,  # mm per unit magnitude
    distribute_arms_evenly: bool = True
) -> DroneCADParameters:
    """
    Convert a SphericalAngularDroneGenomeHandler genome to CAD parameters.

    The mapping strategy:
    - magnitude → arm_length (scaled to mm)
    - arm_rotation (theta) → arm_azimuth (azimuth in XY plane)
    - arm_pitch (phi) → arm_tilt (angle from vertical)
    - motor_rotation → motor_azimuth
    - motor_pitch → motor_tilt
    - arm positions distributed evenly around the core plate if distribute_arms_evenly=True

    Args:
        genome_handler: SphericalAngularDroneGenomeHandler instance
        magnitude_to_length_scale: Scale factor to convert magnitude to mm
        distribute_arms_evenly: If True, distribute arms evenly around disc circumference
            ignoring the genome's arm_rotation values. If False, use arm_rotation.

    Returns:
        DroneCADParameters containing all parameters for CAD generation
    """
    # Get valid arms from genome
    valid_arms = genome_handler.get_valid_arms()
    num_arms = len(valid_arms)

    if num_arms == 0:
        return DroneCADParameters(arms=[], num_arms=0)

    arms_params = []

    for i, arm in enumerate(valid_arms):
        magnitude, arm_rotation, arm_pitch, motor_rotation, motor_pitch, direction = arm

        # Convert magnitude to arm length in mm
        arm_length = magnitude * magnitude_to_length_scale

        # Determine arm attachment angle around the disc
        if distribute_arms_evenly:
            # Distribute arms evenly in a circle (0, 360/n, 2*360/n, ...)
            arm_attachment_angle = (360.0 / num_arms) * i
        else:
            # Use the arm_rotation from genome (convert from radians)
            arm_attachment_angle = radians_to_degrees(arm_rotation) % 360.0

        # Convert spherical arm_pitch to tilt angle
        # arm_pitch in genome is polar angle (0 to π)
        # Convert to tilt: 0° = vertical up, 90° = horizontal, 180° = vertical down
        # Tilt angle interpretation: angle from Z-axis (vertical)
        arm_tilt = radians_to_degrees(arm_pitch) - 90.0  # Adjust to make 0 = up along arm

        # arm_rotation becomes azimuth (already handled above for attachment angle)
        # For the arm orientation itself, use 0 (arm points radially outward)
        arm_azimuth = 0.0

        # Convert motor orientation angles
        motor_tilt = radians_to_degrees(motor_pitch) - 90.0
        motor_azimuth = radians_to_degrees(motor_rotation)

        arm_params = ArmCADParameters(
            arm_tilt=arm_tilt,
            arm_azimuth=arm_azimuth,
            motor_tilt=motor_tilt,
            motor_azimuth=motor_azimuth,
            arm_attachment_angle=arm_attachment_angle,
            arm_length=arm_length,
            direction=int(direction)
        )
        arms_params.append(arm_params)

    return DroneCADParameters(arms=arms_params, num_arms=num_arms)


def cad_parameters_to_assembly_vector(cad_params: DroneCADParameters) -> List[List[float]]:
    """
    Convert DroneCADParameters to the assembly vector format used by full_drone_assembly.py.

    Assembly vector format:
        [arm_tilt, arm_azimuth, motor_tilt, motor_azimuth, arm_attachment_angle, arm_length]

    Args:
        cad_params: DroneCADParameters instance

    Returns:
        List of parameter vectors, one per arm
    """
    assembly_vector = []

    for arm in cad_params.arms:
        assembly_vector.append([
            arm.arm_tilt,
            arm.arm_azimuth,
            arm.motor_tilt,
            arm.motor_azimuth,
            arm.arm_attachment_angle,
            arm.arm_length
        ])

    return assembly_vector


def cad_parameters_to_individual_vector(cad_params: DroneCADParameters) -> List[List[float]]:
    """
    Convert DroneCADParameters to the individual parameter vector format.

    Individual vector format:
        [arm_tilt, arm_azimuth, motor_tilt, motor_azimuth]

    Args:
        cad_params: DroneCADParameters instance

    Returns:
        List of parameter vectors, one per arm
    """
    individual_vector = []

    for arm in cad_params.arms:
        individual_vector.append([
            arm.arm_tilt,
            arm.arm_azimuth,
            arm.motor_tilt,
            arm.motor_azimuth
        ])

    return individual_vector


# ============================================================================
# EXAMPLE AND TESTING
# ============================================================================

if __name__ == "__main__":
    import sys
    import os

    # Add parent directory to path for imports
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    from airevolve.evolution_tools.genome_handlers.spherical_angular_genome_handler import (
        SphericalAngularDroneGenomeHandler
    )

    print("=" * 70)
    print("Genome to CAD Parameter Adapter - Testing")
    print("=" * 70)

    # Create a test genome handler with 4 arms
    handler = SphericalAngularDroneGenomeHandler(
        min_max_narms=(3, 6),
        bilateral_plane_for_symmetry=None
    )

    # Generate a random individual
    print("\nGenerating random individual...")
    population = handler.random_population(1)
    handler.genome = population[0]

    print(f"Generated individual with {handler.get_arm_count()} arms")
    print("\nGenome (first 3 arms):")
    valid_arms = handler.get_valid_arms()
    for i, arm in enumerate(valid_arms[:3]):
        print(f"  Arm {i+1}: magnitude={arm[0]:.3f}, arm_rot={np.degrees(arm[1]):.1f}°, "
              f"arm_pitch={np.degrees(arm[2]):.1f}°, motor_rot={np.degrees(arm[3]):.1f}°, "
              f"motor_pitch={np.degrees(arm[4]):.1f}°, dir={int(arm[5])}")

    # Convert to CAD parameters
    print("\n" + "-" * 70)
    print("Converting to CAD parameters (evenly distributed)...")
    print("-" * 70)

    cad_params = genome_to_cad_parameters(handler, distribute_arms_evenly=True)

    print(f"\nCAD Parameters for {cad_params.num_arms} arms:")
    for i, arm in enumerate(cad_params.arms[:3]):
        print(f"\nArm {i+1}:")
        print(f"  arm_tilt: {arm.arm_tilt:.1f}°")
        print(f"  arm_azimuth: {arm.arm_azimuth:.1f}°")
        print(f"  motor_tilt: {arm.motor_tilt:.1f}°")
        print(f"  motor_azimuth: {arm.motor_azimuth:.1f}°")
        print(f"  arm_attachment_angle: {arm.arm_attachment_angle:.1f}°")
        print(f"  arm_length: {arm.arm_length:.1f}mm")
        print(f"  direction: {arm.direction}")

    # Convert to assembly vector
    print("\n" + "-" * 70)
    print("Converting to assembly vector format...")
    print("-" * 70)

    assembly_vector = cad_parameters_to_assembly_vector(cad_params)
    print(f"\nAssembly vector (first 3 arms):")
    for i, params in enumerate(assembly_vector[:3]):
        print(f"  Arm {i+1}: {params}")

    # Convert to individual vector
    print("\n" + "-" * 70)
    print("Converting to individual vector format...")
    print("-" * 70)

    individual_vector = cad_parameters_to_individual_vector(cad_params)
    print(f"\nIndividual vector (first 3 arms):")
    for i, params in enumerate(individual_vector[:3]):
        print(f"  Arm {i+1}: {params}")

    print("\n" + "=" * 70)
    print("Testing completed successfully!")
    print("=" * 70)
