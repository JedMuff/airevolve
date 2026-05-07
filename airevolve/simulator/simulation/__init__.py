"""
Simulation package for drone dynamics and configuration.

This package contains the core simulation components including drone configuration,
physics simulation, and propeller data.
"""

from .drone_simulator import DroneSimulator, create_quadrotor, create_hexarotor, create_tricopter, create_octorotor
from .drone_configuration import DroneConfiguration
from .propeller_data import create_standard_propeller_config, get_extended_prop_params, GRAVITY
from .drone_interface import DroneInterface
from .dynamics_params import derive_reference_params, W_MIN_N, W_MAX_N
from .battery_model import LiPoBatteryModel

__all__ = [
    'DroneSimulator',
    'DroneConfiguration',
    'DroneInterface',
    'LiPoBatteryModel',
    'create_standard_propeller_config',
    'create_quadrotor',
    'create_hexarotor',
    'create_tricopter',
    'create_octorotor',
    'derive_reference_params',
    'get_extended_prop_params',
    'GRAVITY',
    'W_MIN_N',
    'W_MAX_N',
]