"""
Simulation package for drone dynamics and configuration.

This package contains the core simulation components including drone configuration,
physics simulation, and propeller data.
"""

from .drone_simulator import DroneSimulator, create_quadrotor, create_hexarotor, create_tricopter, create_octorotor
from .drone_configuration import DroneConfiguration
from .propeller_data import create_standard_propeller_config, GRAVITY
from .drone_interface import DroneInterface

__all__ = [
    'DroneSimulator',
    'DroneConfiguration',
    'DroneInterface',
    'create_standard_propeller_config',
    'create_quadrotor',
    'create_hexarotor',
    'create_tricopter',
    'create_octorotor',
    'GRAVITY'
]