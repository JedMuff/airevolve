"""
Simulation package for drone dynamics and configuration.

This package contains the core simulation components including drone configuration,
physics simulation, and propeller data.
"""

from .drone_simulator import DroneSimulator, create_quadrotor, create_hexarotor, create_tricopter, create_octorotor
from .drone_configuration import DroneConfiguration
from .propeller_data import create_standard_propeller_config, genome_to_propellers, get_extended_prop_params, GRAVITY
from .drone_interface import DroneInterface
from .dynamics_params import derive_reference_params, sysid_to_reference_params, W_MIN_N, W_MAX_N
from .randomization import make_randomization
from .sysid_presets import PARAMS_HEX3, real_hex3_propellers, calibrate_prop_constants

__all__ = [
    'DroneSimulator',
    'DroneConfiguration',
    'DroneInterface',
    'create_standard_propeller_config',
    'genome_to_propellers',
    'create_quadrotor',
    'create_hexarotor',
    'create_tricopter',
    'create_octorotor',
    'derive_reference_params',
    'sysid_to_reference_params',
    'make_randomization',
    'PARAMS_HEX3',
    'real_hex3_propellers',
    'calibrate_prop_constants',
    'get_extended_prop_params',
    'GRAVITY',
    'W_MIN_N',
    'W_MAX_N',
]