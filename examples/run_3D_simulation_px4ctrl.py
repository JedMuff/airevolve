# -*- coding: utf-8 -*-
"""
Configurable 3D Simulation for Multi-Rotor Drones with Generalized Controller

This simulation uses the GeneralizedControl class that works with arbitrary
drone configurations through configurable control allocation.

Based on original simulation by:
author: John Bass
email: john.bobzwik@gmail.com
license: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
import json
import sys
import os

from airevolve.controllers.trajectory_generation.trajectory import Trajectory
from airevolve.controllers.px4_ctrl.px4_based_ctrl import GeneralizedControl  # Use the new generalized controller
from airevolve.simulator.simulation import DroneInterface
from airevolve.simulator.simulation import create_standard_propeller_config
from airevolve.controllers.utils.wind_model import Wind
import airevolve.controllers.utils as utils

def quad_sim(t, Ts, quad, ctrl, wind, traj, orient="NED"):
    """
    Single simulation step for configurable drone.
    
    Args:
        t: Current time
        Ts: Time step
        quad: DroneInterface instance
        ctrl: GeneralizedControl instance
        wind: Wind model
        traj: Trajectory instance
        
    Returns:
        Updated time
    """
    # Dynamics (using last timestep's commands)
    quad.update(t, Ts, ctrl.w_cmd, wind)
    t += Ts

    # Trajectory for Desired States 
    sDes = traj.desiredState(t, Ts, quad)        

    # Generate Commands (for next iteration)
    ctrl.controller(sDes, quad, traj.ctrlType, Ts)

    # Debug: Check trajectory following performance and control signals
    if t > 0.1 and int(t * 200) % 100 == 0:  # Print every 0.5 seconds after 0.1s
        pos_error = np.linalg.norm(quad.pos - sDes[0:3])
        vel_error = np.linalg.norm(quad.vel - sDes[3:6])
        
        # Always print basic debug info to track the issue
        print(f"\nDEBUG t={t:.2f}s:")
        print(f"  Pos error: {pos_error:.3f}m | Vel error: {vel_error:.3f}m/s")
        print(f"  Current pos: [{quad.pos[0]:.2f}, {quad.pos[1]:.2f}, {quad.pos[2]:.2f}]")
        print(f"  Desired pos: [{sDes[0]:.2f}, {sDes[1]:.2f}, {sDes[2]:.2f}]")
        print(f"  Current vel: [{quad.vel[0]:.2f}, {quad.vel[1]:.2f}, {quad.vel[2]:.2f}]")
        print(f"  Desired vel: [{sDes[3]:.2f}, {sDes[4]:.2f}, {sDes[5]:.2f}]")
        print(f"  Current euler: [{quad.euler[0]*180/np.pi:.1f}, {quad.euler[1]*180/np.pi:.1f}, {quad.euler[2]*180/np.pi:.1f}]°")
        print(f"  Motor commands: [{ctrl.w_cmd[0]:.0f}, {ctrl.w_cmd[1]:.0f}, {ctrl.w_cmd[2]:.0f}, {ctrl.w_cmd[3]:.0f}] rad/s")
        print(f"  Motor thrusts: [{quad.thr[0]:.2f}, {quad.thr[1]:.2f}, {quad.thr[2]:.2f}, {quad.thr[3]:.2f}] N")
        
        # Debug coordinate frame and signs
        total_thrust = np.sum(quad.thr)
        gravity_force = quad.params["mB"] * quad.params["g"]
        print(f"  COORDINATE FRAME DEBUG:")
        print(f"    Total motor thrust: {total_thrust:.2f} N")
        print(f"    Gravity force: {gravity_force:.2f} N")
        print(f"    Net vertical force: {total_thrust - gravity_force:.2f} N")
        print(f"    Expected acceleration: {(total_thrust - gravity_force)/quad.params['mB']:.2f} m/s²")
        print(f"    Coordinate system: {orient}")
        print(f"    Z-axis direction: {'Down=+' if orient == 'NED' else 'Up=+'}")
        
        # Check for instability indicators
        if pos_error > 5.0 or vel_error > 10.0:
            print(f"  *** INSTABILITY DETECTED ***")
            if hasattr(ctrl, 'vel_I_term'):
                print(f"  Velocity I terms: [{ctrl.vel_I_term[0]:.3f}, {ctrl.vel_I_term[1]:.3f}, {ctrl.vel_I_term[2]:.3f}]")
            if t > 3.0:  # Stop early if severely unstable
                print(f"  Stopping simulation due to severe instability")
                return float('inf')  # Signal to stop

    return t

def create_drone_from_config(config_dict, Ti=0):
    """
    Create drone from configuration dictionary.
    
    Args:
        config_dict: Dictionary containing drone configuration
        Ti: Initial time

    Returns:
        DroneInterface instance
    """
    if "propellers" in config_dict:
        # Custom propeller configuration
        return DroneInterface(Ti, propellers=config_dict["propellers"])
    else:
        # Standard configuration
        drone_type = config_dict.get("type", "quad")
        arm_length = config_dict.get("arm_length", 0.11)
        prop_size = config_dict.get("prop_size", 5)
        
        # Handle "matched" type - create configuration equivalent to original framework
        if drone_type == "matched" or prop_size == "matched":
            # Create matched drone configuration
            propellers = [
                {"loc": [0.16, 0.16, 0], "dir": [0, 0, -1, "ccw"], "propsize": "matched"},
                {"loc": [-0.16, 0.16, 0], "dir": [0, 0, -1, "cw"], "propsize": "matched"},
                {"loc": [-0.16, -0.16, 0], "dir": [0, 0, -1, "ccw"], "propsize": "matched"},
                {"loc": [0.16, -0.16, 0], "dir": [0, 0, -1, "cw"], "propsize": "matched"}
            ]
            return DroneInterface(Ti, propellers=propellers)
        else:
            # Convert string prop_size to int if needed
            if isinstance(prop_size, str) and prop_size.isdigit():
                prop_size = int(prop_size)

            return DroneInterface(Ti,
                                        drone_type=drone_type,
                                        arm_length=arm_length,
                                        prop_size=prop_size)

def print_drone_info(quad, ctrl):
    """Print comprehensive drone configuration and control information."""
    config_info = quad.get_configuration_info()
    prop_info = quad.get_propeller_info()
    control_info = ctrl.get_control_info()
    
    print("\n" + "="*60)
    print("DRONE CONFIGURATION")
    print("="*60)
    
    print(f"Drone Type: {config_info.get('num_motors')}-rotor")
    print(f"Mass: {config_info['mass']:.3f} kg")
    print(f"Center of Gravity: [{config_info['center_of_gravity'][0]:.3f}, {config_info['center_of_gravity'][1]:.3f}, {config_info['center_of_gravity'][2]:.3f}] m")
    
    print(f"\nInertia Properties:")
    print(f"  Ix: {config_info['Ix']:.6f} kg⋅m²")
    print(f"  Iy: {config_info['Iy']:.6f} kg⋅m²") 
    print(f"  Iz: {config_info['Iz']:.6f} kg⋅m²")
    
    print(f"\nControl Properties:")
    print(f"  Controller: Generalized Control Allocation")
    print(f"  Motors: {control_info['num_motors']}")
    print(f"  Over-actuated: {config_info['is_over_actuated']}")
    print(f"  Force rank: {config_info['force_rank']}")
    print(f"  Moment rank: {config_info['moment_rank']}")
    print(f"  Combined rank: {config_info['combined_rank']}")
    
    print(f"\nPropeller Configuration:")
    for i, prop in enumerate(prop_info):
        print(f"  Motor {i+1}: {prop['propsize']}\" at [{prop['loc'][0]:.3f}, {prop['loc'][1]:.3f}, {prop['loc'][2]:.3f}] m, {prop['dir'][-1]} rotation")
    
    print("="*60)

def demo_different_configurations():
    """Demonstrate the controller working with different drone types."""
    
    configurations = [
        {"type": "quad", "arm_length": 0.11, "prop_size": 5, "name": "Quadrotor"},
        {"type": "hex", "arm_length": 0.10, "prop_size": 4, "name": "Hexarotor"},
        {"type": "octo", "arm_length": 0.09, "prop_size": 4, "name": "Octorotor"},
        {"type": "tri", "arm_length": 0.12, "prop_size": 6, "name": "Tricopter"}
    ]
    
    print("\n" + "="*60)
    print("GENERALIZED CONTROLLER DEMO")
    print("="*60)
    
    for config in configurations:
        print(f"\nTesting {config['name']}...")
        try:
            # Create drone
            quad = create_drone_from_config(config, Ti=0)
            
            # Create controller
            ctrl = GeneralizedControl(quad, yawType=1)
            
            # Test basic controller setup
            config_info = quad.get_configuration_info()
            control_info = ctrl.get_control_info()
            
            print(f"  ✓ {config['name']}: {config_info['num_motors']} motors, "
                  f"mass={config_info['mass']:.2f}kg, ")
            
        except Exception as e:
            print(f"  ✗ {config['name']}: Failed - {e}")
    
    print("\nAll configurations use the same GeneralizedControl class!")
    print("="*60)

def create_custom_configurations():
    """Create example custom drone configurations."""
    
    # Y6 Configuration (6 motors in Y shape)
    y6_config = {
        "propellers": [
            # Front arm - coaxial motors
            {"loc": [0.15, 0, 0], "dir": [0, 0, -1, "ccw"], "propsize": 5},
            {"loc": [0.15, 0, -0.05], "dir": [0, 0, -1, "cw"], "propsize": 5},
            # Left rear arm - coaxial motors  
            {"loc": [-0.075, -0.13, 0], "dir": [0, 0, -1, "ccw"], "propsize": 5},
            {"loc": [-0.075, -0.13, -0.05], "dir": [0, 0, -1, "cw"], "propsize": 5},
            # Right rear arm - coaxial motors
            {"loc": [-0.075, 0.13, 0], "dir": [0, 0, -1, "ccw"], "propsize": 5},
            {"loc": [-0.075, 0.13, -0.05], "dir": [0, 0, -1, "cw"], "propsize": 5}
        ],
        "name": "Y6 Coaxial"
    }
    
    # Tilted rotor configuration
    tilt_angle = np.pi/6  # 30 degrees
    tilted_config = {
        "propellers": [
            {"loc": [0.12, 0.12, 0], "dir": [np.sin(tilt_angle), 0, -np.cos(tilt_angle), "ccw"], "propsize": 5},
            {"loc": [-0.12, 0.12, 0], "dir": [-np.sin(tilt_angle), 0, -np.cos(tilt_angle), "cw"], "propsize": 5},
            {"loc": [-0.12, -0.12, 0], "dir": [-np.sin(tilt_angle), 0, -np.cos(tilt_angle), "ccw"], "propsize": 5},
            {"loc": [0.12, -0.12, 0], "dir": [np.sin(tilt_angle), 0, -np.cos(tilt_angle), "cw"], "propsize": 5}
        ],
        "name": "Tilted Quadrotor"
    }
    
    return [y6_config, tilted_config]

def main():
    """Main simulation function with generalized controller support."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generalized 3D Drone Simulation')
    parser.add_argument('--config', type=str, help='JSON configuration file')
    parser.add_argument('--type', type=str, choices=['quad', 'hex', 'tri', 'octo', 'matched'], 
                       default='matched', help='Standard drone type')
    parser.add_argument('--arm-length', type=float, default=0.16, 
                       help='Arm length in meters')
    parser.add_argument('--prop-size', type=str, choices=['4', '5', '6', '7', '8', 'matched'], 
                       default='matched', help='Propeller size in inches or "matched"')
    parser.add_argument('--orient', type=str, choices=['NED', 'ENU'], 
                       default='NED', help='Coordinate frame orientation')
    parser.add_argument('--time', type=float, default=20, 
                       help='Simulation time in seconds')
    parser.add_argument('--dt', type=float, default=0.005, 
                       help='Time step in seconds')
    parser.add_argument('--save', action='store_true', 
                       help='Save animation')
    parser.add_argument('--info', action='store_true',
                       help='Print detailed drone configuration')
    parser.add_argument('--demo', action='store_true',
                       help='Run demo of different configurations')
    parser.add_argument('--custom', action='store_true',
                       help='Test custom configurations')
    
    args = parser.parse_args()
    
    # Run demo if requested
    if args.demo:
        demo_different_configurations()
        return
    
    # Test custom configurations if requested
    if args.custom:
        print("Testing custom configurations...")
        custom_configs = create_custom_configurations()
        for config in custom_configs:
            print(f"\nTesting {config['name']}...")
            try:
                quad = DroneInterface(Ti=0, propellers=config["propellers"])
                ctrl = GeneralizedControl(quad, yawType=1)
                info = quad.get_configuration_info()
                print(f"  ✓ {config['name']}: {info['num_motors']} motors, mass={info['mass']:.2f}kg")
            except Exception as e:
                print(f"  ✗ {config['name']}: Failed - {e}")
        return
    
    start_time = time.time()

    # Simulation Setup
    Ti = 0
    Ts = args.dt
    Tf = args.time
    ifsave = 1 if args.save else 0

    # Choose trajectory settings
    ctrlOptions = ["xyz_pos", "xy_vel_z_pos", "xyz_vel"]
    trajSelect = np.zeros(3)

    # Select Control Type (0: xyz_pos, 1: xy_vel_z_pos, 2: xyz_vel)
    ctrlType = ctrlOptions[0]   
    
    # Trajectory configuration
    trajSelect[0] = 14    # minimum jerk trajectory
    trajSelect[1] = 3    # follow yaw
    trajSelect[2] = 1    # use average speed
    
    print("Control type: {}".format(ctrlType))

    # Create drone from configuration
    if args.config:
        # Load from JSON file
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        quad = create_drone_from_config(config_dict, Ti)
        print(f"Created drone from config file: {args.config}")
    else:
        # Create standard drone from command line arguments
        config_dict = {
            "type": args.type,
            "arm_length": args.arm_length,
            "prop_size": args.prop_size
        }
        quad = create_drone_from_config(config_dict, Ti)
        if args.type == "matched" or args.prop_size == "matched":
            print(f"Created matched drone (equivalent to original framework)")
        else:
            print(f"Created {args.type} with {args.arm_length}m arms and {args.prop_size}\" props")

    # Initialize Generalized Controller
    ctrl = GeneralizedControl(quad, yawType=1, orient=args.orient)  # Pass orient as parameter
    
    # Enable detailed control debugging
    ctrl._debug_control = True
    
    # Debug: Check the mixer matrix conditioning
    mixer_fm = np.zeros((4, quad.drone_sim.num_motors))
    for i in range(quad.drone_sim.num_motors):
        prop = quad.drone_sim.config.propellers[i]
        w_max = prop["wmax"]
        mixer_fm[0, i] = -quad.drone_sim.Bf[2, i] / (w_max**2)
        mixer_fm[1, i] = quad.drone_sim.Bm[0, i] / (w_max**2)
        mixer_fm[2, i] = quad.drone_sim.Bm[1, i] / (w_max**2)  
        mixer_fm[3, i] = quad.drone_sim.Bm[2, i] / (w_max**2)
    
    print(f"\nDEBUG: Mixer matrix analysis:")
    print(f"mixer_fm =\n{mixer_fm}")
    print(f"Matrix condition number: {np.linalg.cond(mixer_fm):.2e}")
    print(f"Matrix determinant: {np.linalg.det(mixer_fm):.2e}")
    print(f"mixerFMinv =\n{quad.params['mixerFMinv']}")
    print(f"mixerFMinv condition number: {np.linalg.cond(quad.params['mixerFMinv']):.2e}")
    
    # DEBUG: Check the scaling issue
    print(f"\nDEBUG: Scaling analysis:")
    w_max = quad.drone_sim.config.propellers[0]["wmax"]
    print(f"w_max: {w_max}")
    print(f"w_max^2: {w_max**2}")
    print(f"Bf matrix (thrust allocation):\n{quad.drone_sim.Bf}")
    print(f"Bm matrix (moment allocation):\n{quad.drone_sim.Bm}")
    
    # Check what happens if we don't divide by w_max^2
    mixer_fm_fixed = np.zeros((4, quad.drone_sim.num_motors))
    for i in range(quad.drone_sim.num_motors):
        mixer_fm_fixed[0, i] = -quad.drone_sim.Bf[2, i]  # No scaling
        mixer_fm_fixed[1, i] = quad.drone_sim.Bm[0, i]   # No scaling
        mixer_fm_fixed[2, i] = quad.drone_sim.Bm[1, i]   # No scaling
        mixer_fm_fixed[3, i] = quad.drone_sim.Bm[2, i]   # No scaling
    
    mixerFMinv_fixed = np.linalg.pinv(mixer_fm_fixed)
    print(f"\nDEBUG: Fixed scaling (no w_max^2 division):")
    print(f"mixer_fm_fixed =\n{mixer_fm_fixed}")
    print(f"mixerFMinv_fixed =\n{mixerFMinv_fixed}")
    print(f"Fixed matrix condition number: {np.linalg.cond(mixerFMinv_fixed):.2e}")
    
    # Initialize Wind and Trajectory  
    traj = Trajectory(quad, ctrlType, trajSelect)
    wind = Wind('None', 2.0, 90, -15)

    # Print drone information if requested
    if args.info:
        print_drone_info(quad, ctrl)

    # Trajectory for First Desired States
    sDes = traj.desiredState(0, Ts, quad)        

    # Generate First Commands
    ctrl.controller(sDes, quad, traj.ctrlType, Ts)
    
    # Initialize Result Matrices
    numTimeStep = int(Tf/Ts+1)
    
    # Get the actual number of motors for this drone
    actual_num_motors = quad.drone_sim.num_motors
    
    t_all          = np.zeros(numTimeStep)
    s_all          = np.zeros([numTimeStep, len(quad.state)])
    pos_all        = np.zeros([numTimeStep, len(quad.pos)])
    vel_all        = np.zeros([numTimeStep, len(quad.vel)])
    quat_all       = np.zeros([numTimeStep, len(quad.quat)])
    omega_all      = np.zeros([numTimeStep, len(quad.omega)])
    euler_all      = np.zeros([numTimeStep, len(quad.euler)])
    sDes_traj_all  = np.zeros([numTimeStep, len(traj.sDes)])
    sDes_calc_all  = np.zeros([numTimeStep, len(ctrl.sDesCalc)])
    w_cmd_all      = np.zeros([numTimeStep, len(ctrl.w_cmd)])
    wMotor_all     = np.zeros([numTimeStep, max(4, actual_num_motors)])
    thr_all        = np.zeros([numTimeStep, max(4, actual_num_motors)])
    tor_all        = np.zeros([numTimeStep, max(4, actual_num_motors)])

    # Store initial values
    t_all[0]            = Ti
    s_all[0,:]          = quad.state
    pos_all[0,:]        = quad.pos
    vel_all[0,:]        = quad.vel
    quat_all[0,:]       = quad.quat
    omega_all[0,:]      = quad.omega
    euler_all[0,:]      = quad.euler
    sDes_traj_all[0,:]  = traj.sDes
    sDes_calc_all[0,:]  = ctrl.sDesCalc
    w_cmd_all[0,:]      = ctrl.w_cmd
    wMotor_all[0,:len(quad.wMotor)] = quad.wMotor
    thr_all[0,:len(quad.thr)]       = quad.thr
    tor_all[0,:len(quad.tor)]       = quad.tor

    # Run Simulation
    print(f"\nRunning simulation for {Tf}s with {quad.drone_sim.num_motors} motors...")
    print(f"Using GeneralizedControl with configurable allocation matrices")
    t = Ti
    i = 1
    while round(t,3) < Tf:
        
        t_new = quad_sim(t, Ts, quad, ctrl, wind, traj, args.orient)
        
        # Check for early termination due to instability
        if t_new == float('inf'):
            print(f"Simulation terminated early at t={t:.2f}s due to instability")
            break
        
        t = t_new
        
        # Store results
        t_all[i]             = t
        s_all[i,:]           = quad.state
        pos_all[i,:]         = quad.pos
        vel_all[i,:]         = quad.vel
        quat_all[i,:]        = quad.quat
        omega_all[i,:]       = quad.omega
        euler_all[i,:]       = quad.euler
        sDes_traj_all[i,:]   = traj.sDes
        sDes_calc_all[i,:]   = ctrl.sDesCalc
        w_cmd_all[i,:]       = ctrl.w_cmd
        wMotor_all[i,:len(quad.wMotor)] = quad.wMotor
        thr_all[i,:len(quad.thr)]       = quad.thr
        tor_all[i,:len(quad.tor)]       = quad.tor
        
        i += 1
    
    end_time = time.time()
    print("Simulated {:.2f}s in {:.6f}s.".format(t, end_time - start_time))

    # View Results
    utils.makeFigures(quad.params, t_all, pos_all, vel_all, quat_all, omega_all, 
                     euler_all, w_cmd_all, wMotor_all, thr_all, tor_all, 
                     sDes_traj_all, sDes_calc_all)
    
    ani = utils.sameAxisAnimation(t_all, traj.wps, pos_all, quat_all, sDes_traj_all, 
                                 Ts, quad.params, traj.xyzType, traj.yawType, ifsave, args.orient)
    plt.show()

def create_example_configs():
    """Create example configuration files for different drone types."""
    
    # Quadrotor configuration
    quad_config = {
        "type": "quad",
        "arm_length": 0.11,
        "prop_size": 5,
        "description": "Standard quadrotor configuration"
    }
    
    # Hexarotor configuration  
    hex_config = {
        "type": "hex",
        "arm_length": 0.10,
        "prop_size": 4,
        "description": "Standard hexarotor configuration"
    }
    
    # Custom Y6 configuration
    y6_config = {
        "propellers": [
            {"loc": [0.15, 0, 0], "dir": [0, 0, -1, "ccw"], "propsize": 5},
            {"loc": [0.15, 0, -0.05], "dir": [0, 0, -1, "cw"], "propsize": 5},
            {"loc": [-0.075, -0.13, 0], "dir": [0, 0, -1, "ccw"], "propsize": 5},
            {"loc": [-0.075, -0.13, -0.05], "dir": [0, 0, -1, "cw"], "propsize": 5},
            {"loc": [-0.075, 0.13, 0], "dir": [0, 0, -1, "ccw"], "propsize": 5},
            {"loc": [-0.075, 0.13, -0.05], "dir": [0, 0, -1, "cw"], "propsize": 5}
        ],
        "description": "Y6 coaxial configuration with 6 motors"
    }
    
    # Save configuration files
    configs = {
        "quad_config.json": quad_config,
        "hex_config.json": hex_config, 
        "y6_config.json": y6_config
    }
    
    for filename, config in configs.items():
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Created example configuration: {filename}")

if __name__ == "__main__":
    # Check if we should create example configs
    if len(sys.argv) > 1 and sys.argv[1] == "--create-examples":
        create_example_configs()
        sys.exit(0)
        
    # Run main simulation
    main()