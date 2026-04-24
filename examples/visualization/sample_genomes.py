"""
Sample genome data for demonstration and testing of visualization tools.

This module provides various example drone genomes in different formats
to showcase the capabilities of the visualization system.
"""

import numpy as np
from airevolve.evolution_tools.genome_handlers.cartesian_euler_genome_handler import CartesianEulerDroneGenomeHandler


def create_quadcopter_cartesian() -> CartesianEulerDroneGenomeHandler:
    """
    Create a standard quadcopter configuration using Cartesian coordinates.
    
    Returns:
        CartesianEulerDroneGenomeHandler with classic quadcopter layout
    """
    # Classic quadcopter: 4 arms in + configuration
    genome_data = np.array([
        # [x,    y,    z,   roll, pitch,  yaw,  direction]
        [ 0.3,  0.0,  0.0,   0.0,   0.0,  0.0,     1],  # Front
        [-0.3,  0.0,  0.0,   0.0,   0.0,  0.0,     1],  # Back  
        [ 0.0,  0.3,  0.0,   0.0,   0.0,  0.0,     0],  # Right
        [ 0.0, -0.3,  0.0,   0.0,   0.0,  0.0,     0],  # Left
    ])
    
    return CartesianEulerDroneGenomeHandler(genome=genome_data, min_max_narms=(4,4))


def create_hexacopter_cartesian() -> CartesianEulerDroneGenomeHandler:
    """
    Create a hexacopter configuration using Cartesian coordinates.
    
    Returns:
        CartesianEulerDroneGenomeHandler with hexacopter layout
    """
    # Hexacopter: 6 arms evenly spaced in circle
    angles = np.linspace(0, 2*np.pi, 7)[:-1]  # 6 evenly spaced angles
    radius = 0.35
    
    genome_data = []
    for i, angle in enumerate(angles):
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = 0.0
        direction = i % 2  # Alternating directions
        genome_data.append([x, y, z, 0.0, 0.0, 0.0, direction])
    
    return CartesianEulerDroneGenomeHandler(genome=np.array(genome_data), min_max_narms=(6,6))


def create_tilted_motors_cartesian() -> CartesianEulerDroneGenomeHandler:
    """
    Create a drone with tilted motors using Cartesian coordinates.
    
    Returns:
        CartesianEulerDroneGenomeHandler with tilted motor orientations
    """
    tilt_angle = np.pi / 6  # 30 degrees
    
    genome_data = np.array([
        # [x,    y,    z,   roll,       pitch,      yaw,  direction]
        [ 0.25, 0.25, 0.0,   0.0,  tilt_angle,    0.0,     1],
        [ 0.25,-0.25, 0.0,   0.0,  tilt_angle,    0.0,     1],
        [-0.25,-0.25, 0.0,   0.0, -tilt_angle,    0.0,     0],
        [-0.25, 0.25, 0.0,   0.0, -tilt_angle,    0.0,     0],
    ])
    
    return CartesianEulerDroneGenomeHandler(genome=genome_data, min_max_narms=(4,4))


def create_3d_configuration_cartesian() -> CartesianEulerDroneGenomeHandler:
    """
    Create a 3D drone configuration with motors at different heights.
    
    Returns:
        CartesianEulerDroneGenomeHandler with 3D motor placement
    """
    genome_data = np.array([
        # [x,    y,    z,    roll, pitch,  yaw,  direction]
        [ 0.2,  0.2,  0.1,   0.0,   0.0,  0.0,     1],
        [ 0.2, -0.2, -0.1,   0.0,   0.0,  0.0,     1],
        [-0.2, -0.2,  0.1,   0.0,   0.0,  0.0,     0],
        [-0.2,  0.2, -0.1,   0.0,   0.0,  0.0,     0],
        [ 0.0,  0.0,  0.2,   0.0,   0.0,  0.0,     1],  # Top motor
        [ 0.0,  0.0, -0.2,   0.0,   0.0,  0.0,     0],  # Bottom motor
    ])
    
    return CartesianEulerDroneGenomeHandler(genome=genome_data, min_max_narms=(6,6))


def create_quadcopter_polar() -> np.ndarray:
    """
    Create a standard quadcopter configuration using polar coordinates.
    
    Returns:
        Numpy array with polar coordinate genome data
        [magnitude, azimuth, pitch, motor_yaw, motor_pitch, direction]
    """
    genome_data = np.array([
        # [mag,  azimuth,  pitch, m_yaw, m_pitch, direction]
        [0.3,     0.0,     0.0,   0.0,    0.0,       1],  # Front
        [0.3,    np.pi,    0.0,   0.0,    0.0,       1],  # Back
        [0.3,   np.pi/2,   0.0,   0.0,    0.0,       0],  # Right
        [0.3,  -np.pi/2,   0.0,   0.0,    0.0,       0],  # Left
    ])
    
    return genome_data


def create_asymmetric_polar() -> np.ndarray:
    """
    Create an asymmetric drone configuration using polar coordinates.
    
    Returns:
        Numpy array with polar coordinate genome data
    """
    genome_data = np.array([
        # [mag,  azimuth,    pitch,   m_yaw,   m_pitch, direction]
        [0.4,     0.0,      0.0,     0.0,      0.0,       1],
        [0.2,    np.pi/3,   0.1,     0.2,      0.1,       0],
        [0.35,   np.pi,     0.0,     0.0,      0.0,       1],
        [0.25,  -np.pi/4,  -0.05,   -0.1,     0.0,       0],
        [0.15,   np.pi/2,   0.2,     0.3,      0.2,       1],
    ])
    
    return genome_data


def create_evolved_example_polar() -> np.ndarray:
    """
    Create an example that might result from evolution - more complex geometry.
    
    Returns:
        Numpy array with polar coordinate genome data
    """
    genome_data = np.array([
        # [mag,  azimuth,    pitch,   m_yaw,   m_pitch, direction]
        [0.38,   0.2,       0.05,    0.1,      0.15,      1],
        [0.42,   1.8,      -0.03,   -0.05,     0.12,      0],
        [0.35,   3.0,       0.08,    0.2,      -0.1,      1],
        [0.29,  -1.2,      -0.02,   -0.15,     0.08,      0],
        [0.31,   2.5,       0.12,    0.25,     -0.05,     1],
        [0.27,  -0.8,       0.06,   -0.1,      0.18,      0],
    ])
    
    return genome_data


def create_target_individual() -> np.ndarray:
    """
    Create the target individual from the evolution example.
    
    Returns:
        Numpy array representing the target configuration
    """
    return np.array([
        [ 0.5,  0.5, 0.0, 0.0, 0.0, 0.0, 1], 
        [ 0.5, -0.5, 0.0, 0.0, 0.0, 0.0, 1], 
        [-0.5, -0.5, 0.0, 0.0, 0.0, 0.0, 0], 
        [-0.5,  0.5, 0.0, 0.0, 0.0, 0.0, 0]
    ])


def get_all_sample_genomes() -> dict:
    """
    Get a dictionary of all sample genomes for easy access.
    
    Returns:
        Dictionary mapping names to genome data
    """
    return {
        # Cartesian examples (genome handlers)
        'quadcopter_cartesian': create_quadcopter_cartesian(),
        'hexacopter_cartesian': create_hexacopter_cartesian(),
        'tilted_motors_cartesian': create_tilted_motors_cartesian(),
        '3d_configuration_cartesian': create_3d_configuration_cartesian(),
        
        # Polar examples (numpy arrays)
        'quadcopter_polar': create_quadcopter_polar(),
        'asymmetric_polar': create_asymmetric_polar(),
        'evolved_example_polar': create_evolved_example_polar(),
        'target_individual': create_target_individual(),
    }


def print_genome_info(name: str, genome_data) -> None:
    """
    Print information about a genome for debugging.
    
    Args:
        name: Name of the genome
        genome_data: Genome handler or array
    """
    print(f"\n=== {name} ===")
    
    if hasattr(genome_data, 'genome'):
        print(f"Type: {type(genome_data).__name__}")
        print(f"Shape: {genome_data.genome.shape}")
        print(f"Data:\n{genome_data.genome}")
        
        if hasattr(genome_data, 'get_motor_positions'):
            print(f"Positions:\n{genome_data.get_motor_positions()}")
            print(f"Orientations:\n{genome_data.get_motor_orientations()}")
            print(f"Directions: {genome_data.get_propeller_directions()}")
    else:
        print(f"Type: {type(genome_data).__name__}")
        print(f"Shape: {genome_data.shape}")
        print(f"Data:\n{genome_data}")


if __name__ == "__main__":
    # Demo the sample genomes
    print("Sample Genome Data for Visualization Demos")
    print("==========================================")
    
    samples = get_all_sample_genomes()
    
    for name, genome in samples.items():
        print_genome_info(name, genome)