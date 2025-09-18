"""
Propeller Library and Physical Constants

This module contains propeller specifications including force/moment constants,
maximum RPM, and mass properties for different propeller sizes.

Data extracted from drone-hover package for integration with geometric control framework.
"""

import numpy as np

# Propeller library with specifications for different sizes
PROPELLER_LIBRARY = {
    "prop4": {
        "constants": [7.24e-07, 8.20e-09],  # [k_f, k_m] force and moment constants
        "wmax": 3927,                       # Maximum angular velocity (rad/s)
        "mass": 0.018                       # Propeller + motor mass (kg)
    },
    "prop5": {
        "constants": [1.08e-06, 1.22e-08],
        "wmax": 3142,
        "mass": 0.0196
    },
    "prop6": {
        "constants": [2.21e-06, 2.74e-08],
        "wmax": 2618,
        "mass": 0.0252
    },
    "prop7": {
        "constants": [4.65e-06, 6.62e-08],
        "wmax": 2244,
        "mass": 0.046
    },
    "prop8": {
        "constants": [7.60e-06, 1.14e-07],
        "wmax": 1963,
        "mass": 0.056
    }
}

# Physical constants
GRAVITY = 9.81  # m/s^2

# Material properties for mass/inertia calculations
CONTROLLER_MASS = 0.250  # kg, based on 4S 2200mAh LiPo + flight controller
BEAM_DENSITY = 1500 * 0.005 * 0.01  # kg/m, carbon fiber: density * thickness * width

def get_propeller_specs(prop_size):
    """
    Get propeller specifications for a given size.
    
    Args:
        prop_size (int): Propeller size in inches (4-8)
        
    Returns:
        dict: Propeller specifications including constants, wmax, and mass
        
    Raises:
        ValueError: If propeller size is not available
    """
    prop_key = f"prop{prop_size}"
    if prop_key not in PROPELLER_LIBRARY:
        available_sizes = [int(key[4:]) for key in PROPELLER_LIBRARY.keys()]
        raise ValueError(f"Propeller size {prop_size} not available. "
                        f"Available sizes: {available_sizes}")
    
    return PROPELLER_LIBRARY[prop_key].copy()

def validate_propeller_config(props):
    """
    Validate propeller configuration format.
    
    Args:
        props (list): List of propeller dictionaries
        
    Raises:
        KeyError: If required keys are missing
        ValueError: If values are invalid
    """
    required_keys = ["loc", "dir", "propsize"]
    
    for i, prop in enumerate(props):
        # Check required keys
        for key in required_keys:
            if key not in prop:
                raise KeyError(f"'{key}' is missing in propeller {i}")
        
        # Validate propeller size
        if prop["propsize"] not in [4, 5, 6, 7, 8]:
            raise ValueError(f"Invalid propeller size {prop['propsize']} in propeller {i}. "
                           f"Available sizes: [4, 5, 6, 7, 8]")
        
        # Validate direction format
        if len(prop["dir"]) != 4:
            raise ValueError(f"Direction must have 4 elements [x, y, z, rotation] for propeller {i}")
        
        if prop["dir"][-1] not in ["ccw", "cw"]:
            raise ValueError(f"Invalid rotation direction '{prop['dir'][-1]}' for propeller {i}. "
                           f"Use 'ccw' or 'cw'")
        
        # Validate location format
        if len(prop["loc"]) != 3:
            raise ValueError(f"Location must have 3 elements [x, y, z] for propeller {i}")

def create_standard_propeller_config(config_type, arm_length=0.11, prop_size=5):
    """
    Create standard propeller configurations for common drone types.
    
    Args:
        config_type (str): Type of configuration ('quad', 'hex', 'tri', 'octo')
        arm_length (float): Length of drone arms in meters
        prop_size (int): Propeller size in inches
        
    Returns:
        list: List of propeller dictionaries
    """
    if config_type == "quad" or config_type == "quadrotor":
        # Standard X configuration quadcopter
        return [
            {"loc": [arm_length * np.cos(np.pi/4), arm_length * np.sin(np.pi/4), 0], 
             "dir": [0, 0, -1, "ccw"], "propsize": prop_size},
            {"loc": [arm_length * np.cos(3*np.pi/4), arm_length * np.sin(3*np.pi/4), 0], 
             "dir": [0, 0, -1, "cw"], "propsize": prop_size},
            {"loc": [arm_length * np.cos(5*np.pi/4), arm_length * np.sin(5*np.pi/4), 0], 
             "dir": [0, 0, -1, "ccw"], "propsize": prop_size},
            {"loc": [arm_length * np.cos(7*np.pi/4), arm_length * np.sin(7*np.pi/4), 0], 
             "dir": [0, 0, -1, "cw"], "propsize": prop_size}
        ]
    
    elif config_type == "hex" or config_type == "hexarotor":
        # Standard hexacopter configuration
        return [
            {"loc": [arm_length, 0, 0], "dir": [0, 0, -1, "ccw"], "propsize": prop_size},
            {"loc": [arm_length * np.cos(np.pi/3), arm_length * np.sin(np.pi/3), 0], 
             "dir": [0, 0, -1, "cw"], "propsize": prop_size},
            {"loc": [arm_length * np.cos(2*np.pi/3), arm_length * np.sin(2*np.pi/3), 0], 
             "dir": [0, 0, -1, "ccw"], "propsize": prop_size},
            {"loc": [arm_length * np.cos(np.pi), arm_length * np.sin(np.pi), 0], 
             "dir": [0, 0, -1, "cw"], "propsize": prop_size},
            {"loc": [arm_length * np.cos(4*np.pi/3), arm_length * np.sin(4*np.pi/3), 0], 
             "dir": [0, 0, -1, "ccw"], "propsize": prop_size},
            {"loc": [arm_length * np.cos(5*np.pi/3), arm_length * np.sin(5*np.pi/3), 0], 
             "dir": [0, 0, -1, "cw"], "propsize": prop_size}
        ]
    
    elif config_type == "tri" or config_type == "tricopter":
        # Standard tricopter configuration (no tilt rotor)
        return [
            {"loc": [arm_length, 0, 0], "dir": [0, 0, -1, "ccw"], "propsize": prop_size},
            {"loc": [arm_length * np.cos(2*np.pi/3), arm_length * np.sin(2*np.pi/3), 0], 
             "dir": [0, 0, -1, "cw"], "propsize": prop_size},
            {"loc": [arm_length * np.cos(4*np.pi/3), arm_length * np.sin(4*np.pi/3), 0], 
             "dir": [0, 0, -1, "ccw"], "propsize": prop_size}
        ]
    
    elif config_type == "octo" or config_type == "octorotor":
        # Standard octacopter configuration
        angles = [i * np.pi/4 for i in range(8)]
        spin_dirs = ["ccw", "cw"] * 4  # Alternating pattern
        
        return [
            {"loc": [arm_length * np.cos(angle), arm_length * np.sin(angle), 0], 
             "dir": [0, 0, -1, spin_dir], "propsize": prop_size}
            for angle, spin_dir in zip(angles, spin_dirs)
        ]
    
    else:
        raise ValueError(f"Unknown configuration type: {config_type}. "
                        f"Available types: ['quad', 'hex', 'tri', 'octo']")