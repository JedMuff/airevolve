# -*- coding: utf-8 -*-
"""
3D Simulation with Lee Geometric Controller for Multi-Rotor Drones

This simulation demonstrates the Lee geometric controller working with arbitrary
drone configurations through configurable control allocation.

Based on the PX4 control example but using Lee geometric control algorithms.

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
from airevolve.controllers.lee_control.lee_controller import LeeGeometricControl  # Use Lee controller
from airevolve.simulator.simulation import DroneInterface
from airevolve.simulator.simulation import create_standard_propeller_config
from airevolve.controllers.utils.wind_model import Wind
import airevolve.controllers.utils as utils

def quad_sim(t, Ts, quad, ctrl, wind, traj, orient="NED"):
    """
    Single simulation step for configurable drone with Lee control.
    
    Args:
        t: Current time
        Ts: Time step
        quad: DroneInterface instance
        ctrl: LeeGeometricControl instance
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

    # Check for lap completion (for figure-8 trajectory)
    if hasattr(traj, 'lap_count') and hasattr(traj, 'lap_completion_times'):
        if not hasattr(traj, '_last_printed_lap'):
            traj._last_printed_lap = 0

        # Print each new lap completion
        if traj.lap_count > traj._last_printed_lap:
            lap_num = traj.lap_count
            lap_time = traj.lap_completion_times[-1]

            # Calculate lap duration (time since last lap or since ramp-up)
            if len(traj.lap_completion_times) > 1:
                lap_duration = lap_time - traj.lap_completion_times[-2]
                print(f"\n✓ LAP {lap_num} COMPLETED at t={lap_time:.3f}s (lap duration: {lap_duration:.3f}s)")
            else:
                print(f"\n✓ LAP {lap_num} COMPLETED at t={lap_time:.3f}s")

            traj._last_printed_lap = traj.lap_count

    # Generate Commands (for next iteration)
    ctrl.controller(sDes, quad, traj.ctrlType, Ts)

    # Debug: Check trajectory following performance and control signals
    if t > 0.1 and int(t * 200) % 100 == 0:  # Print every 0.5 seconds after 0.1s
        pos_error = np.linalg.norm(quad.pos - sDes[0:3])
        vel_error = np.linalg.norm(quad.vel - sDes[3:6])
        
        # Always print basic debug info to track the issue
        print(f"\nDEBUG t={t:.2f}s (Lee Control):")
        print(f"  Pos error: {pos_error:.3f}m | Vel error: {vel_error:.3f}m/s")
        print(f"  Current pos: [{quad.pos[0]:.2f}, {quad.pos[1]:.2f}, {quad.pos[2]:.2f}]")
        print(f"  Desired pos: [{sDes[0]:.2f}, {sDes[1]:.2f}, {sDes[2]:.2f}]")
        print(f"  Current vel: [{quad.vel[0]:.2f}, {quad.vel[1]:.2f}, {quad.vel[2]:.2f}]")
        print(f"  Desired vel: [{sDes[3]:.2f}, {sDes[4]:.2f}, {sDes[5]:.2f}]")
        print(f"  Current euler: [{quad.euler[0]*180/np.pi:.1f}, {quad.euler[1]*180/np.pi:.1f}, {quad.euler[2]*180/np.pi:.1f}]°")
        print(f"  Motor commands: [{ctrl.w_cmd[0]:.0f}, {ctrl.w_cmd[1]:.0f}, {ctrl.w_cmd[2]:.0f}, {ctrl.w_cmd[3]:.0f}] rad/s")
        print(f"  Motor thrusts: [{quad.thr[0]:.2f}, {quad.thr[1]:.2f}, {quad.thr[2]:.2f}, {quad.thr[3]:.2f}] N")
        
        # Lee control specific debug info
        print(f"  LEE CONTROL DEBUG:")
        print(f"    Force command: [{ctrl.wrench_command[0]:.2f}, {ctrl.wrench_command[1]:.2f}, {ctrl.wrench_command[2]:.2f}] N")
        print(f"    Moment command: [{ctrl.wrench_command[3]:.3f}, {ctrl.wrench_command[4]:.3f}, {ctrl.wrench_command[5]:.3f}] N⋅m")
        if hasattr(ctrl, 'desired_orientation') and ctrl.desired_orientation is not None:
            des_quat = np.array(ctrl.desired_orientation).flatten()
            if len(des_quat) >= 4:
                print(f"    Desired orientation (quat): [{des_quat[0]:.3f}, {des_quat[1]:.3f}, {des_quat[2]:.3f}, {des_quat[3]:.3f}]")
            else:
                print(f"    Desired orientation (quat): {des_quat}")
        else:
            print(f"    Desired orientation (quat): [not initialized]")
        
        # Debug coordinate frame and signs
        total_thrust = np.sum(quad.thr)
        gravity_force = quad.params["mB"] * quad.params["g"]
        print(f"    Total motor thrust: {total_thrust:.2f} N")
        print(f"    Gravity force: {gravity_force:.2f} N")
        print(f"    Net vertical force: {total_thrust - gravity_force:.2f} N")
        print(f"    Expected acceleration: {(total_thrust - gravity_force)/quad.params['mB']:.2f} m/s²")
        print(f"    Coordinate system: {orient}")
        
        # Check for instability indicators
        if pos_error > 5.0 or vel_error > 10.0:
            print(f"  *** INSTABILITY DETECTED ***")
            if hasattr(ctrl, 'position_controller'):
                print(f"    Position gains: [{ctrl.pos_P_gain[0]:.3f}, {ctrl.pos_P_gain[1]:.3f}, {ctrl.pos_P_gain[2]:.3f}]")
                print(f"    Velocity gains: [{ctrl.vel_P_gain[0]:.3f}, {ctrl.vel_P_gain[1]:.3f}, {ctrl.vel_P_gain[2]:.3f}]")
            if t > 3.0:  # Stop early if severely unstable
                print(f"    Stopping simulation due to severe instability")
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
    """Print comprehensive drone configuration and Lee control information."""
    config_info = quad.get_configuration_info()
    prop_info = quad.get_propeller_info()
    control_info = ctrl.get_control_info()
    
    print("\n" + "="*60)
    print("DRONE CONFIGURATION WITH LEE CONTROL")
    print("="*60)
    
    print(f"Drone Type: {config_info.get('num_motors')}-rotor")
    print(f"Mass: {config_info['mass']:.3f} kg")
    print(f"Center of Gravity: [{config_info['center_of_gravity'][0]:.3f}, {config_info['center_of_gravity'][1]:.3f}, {config_info['center_of_gravity'][2]:.3f}] m")
    
    print(f"\nInertia Properties:")
    print(f"  Ix: {config_info['Ix']:.6f} kg⋅m²")
    print(f"  Iy: {config_info['Iy']:.6f} kg⋅m²") 
    print(f"  Iz: {config_info['Iz']:.6f} kg⋅m²")
    
    print(f"\nControl Properties:")
    print(f"  Controller: {control_info['controller_type']}")
    print(f"  Coordinate Frame: {control_info['coordinate_frame']}")
    print(f"  Motors: {control_info['num_motors']}")
    print(f"  Over-actuated: {config_info['is_over_actuated']}")
    print(f"  Force rank: {config_info['force_rank']}")
    print(f"  Moment rank: {config_info['moment_rank']}")
    print(f"  Combined rank: {config_info['combined_rank']}")
    
    # Control gains analysis
    gains = control_info['control_gains']
    print(f"\nLee Control Gains:")
    print(f"  Position P: {gains['position_P']}")
    print(f"  Velocity P: {gains['velocity_P']}")
    print(f"  Attitude P: {gains['attitude_P']}")
    print(f"  Rate P: {gains['rate_P']}")
    
    print(f"\nPropeller Configuration:")
    for i, prop in enumerate(prop_info):
        print(f"  Motor {i+1}: {prop['propsize']}\" at [{prop['loc'][0]:.3f}, {prop['loc'][1]:.3f}, {prop['loc'][2]:.3f}] m, {prop['dir'][-1]} rotation")
    
    print("="*60)

def demo_different_configurations():
    """Demonstrate the Lee controller working with different drone types."""
    
    configurations = [
        {"type": "quad", "arm_length": 0.11, "prop_size": 5, "name": "Quadrotor"},
        {"type": "hex", "arm_length": 0.10, "prop_size": 4, "name": "Hexarotor"},
        {"type": "octo", "arm_length": 0.09, "prop_size": 4, "name": "Octorotor"},
        {"type": "tri", "arm_length": 0.12, "prop_size": 6, "name": "Tricopter"}
    ]
    
    print("\n" + "="*60)
    print("LEE GEOMETRIC CONTROLLER DEMO")
    print("="*60)
    
    for config in configurations:
        print(f"\nTesting {config['name']} with Lee Control...")
        try:
            # Create drone
            quad = create_drone_from_config(config, Ti=0)
            
            # Create Lee controller
            ctrl = LeeGeometricControl(quad, yawType=1)
            
            # Test basic controller setup
            config_info = quad.get_configuration_info()
            control_info = ctrl.get_control_info()
            
            print(f"  ✓ {config['name']}: {config_info['num_motors']} motors, "
                  f"mass={config_info['mass']:.2f}kg, "
                  f"controller={control_info['controller_type']}")
            
        except Exception as e:
            print(f"  ✗ {config['name']}: Failed - {e}")
    
    print("\nAll configurations use the same LeeGeometricControl class!")
    print("="*60)

def main():
    """Main simulation function with Lee geometric controller support."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Lee Geometric Controller 3D Drone Simulation')
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
    parser.add_argument('--gains', type=str, 
                       help='Lee control gains as "pos_kp,vel_kp,att_kp,rate_kp" (comma-separated)')
    parser.add_argument('--aggressive', type=float, default=1.0,
                       help='Aggressiveness factor for auto-scaling (0.5=conservative, 2.0=aggressive)')
    parser.add_argument('--traj-period', type=float, default=6.0,
                       help='Figure-8 period in seconds (default: 6.0)')
    parser.add_argument('--traj-amp-x', type=float, default=3.0,
                       help='Figure-8 amplitude in X direction in meters (default: 3.0)')
    parser.add_argument('--traj-amp-y', type=float, default=2.0,
                       help='Figure-8 amplitude in Y direction in meters (default: 2.0)')
    parser.add_argument('--traj-ramp', type=float, default=3.0,
                       help='Figure-8 ramp-up time in seconds (default: 3.0)')

    args = parser.parse_args()
    
    # Run demo if requested
    if args.demo:
        demo_different_configurations()
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
    trajSelect[0] = 14    # figure-8 trajectory
    trajSelect[1] = 3     # follow yaw
    trajSelect[2] = 1     # use average speed
    
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

    # Parse custom gains if provided
    lee_gains = {}
    if args.gains:
        try:
            gain_values = [float(x) for x in args.gains.split(',')]
            if len(gain_values) == 4:
                lee_gains = {
                    'pos_P_gain': np.array([gain_values[0]] * 3),
                    'vel_P_gain': np.array([gain_values[1]] * 3), 
                    'att_P_gain': np.array([gain_values[2]] * 3),
                    'rate_P_gain': np.array([gain_values[3]] * 3)
                }
                print(f"Using custom Lee gains: pos={gain_values[0]}, vel={gain_values[1]}, att={gain_values[2]}, rate={gain_values[3]}")
            else:
                print("Warning: Invalid gains format, using defaults")
        except:
            print("Warning: Failed to parse gains, using defaults")

    # Initialize Lee Geometric Controller
    ctrl = LeeGeometricControl(quad, yawType=1, orient=args.orient,
                               auto_scale_gains=True,       # Enable auto-scaling by default
                               aggressiveness=args.aggressive,
                               **lee_gains)  # Apply custom gains if provided

    # Initialize Wind and Trajectory with custom figure-8 parameters
    figure8_params = {
        'period': args.traj_period,
        'amplitude_x': args.traj_amp_x,
        'amplitude_y': args.traj_amp_y,
        'ramp_time': args.traj_ramp
    }
    traj = Trajectory(quad, ctrlType, trajSelect, figure8_params=figure8_params)
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
    print(f"\nRunning simulation for {Tf}s with Lee Geometric Control...")
    print(f"Using {quad.drone_sim.num_motors} motors with auto-scaling (aggressiveness={args.aggressive})")
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
    print("Simulated {:.2f}s in {:.6f}s using Lee Geometric Control.".format(t, end_time - start_time))

    # View Results
    utils.makeFigures(quad.params, t_all, pos_all, vel_all, quat_all, omega_all, 
                     euler_all, w_cmd_all, wMotor_all, thr_all, tor_all, 
                     sDes_traj_all, sDes_calc_all)
    
    ani = utils.sameAxisAnimation(t_all, traj.wps, pos_all, quat_all, sDes_traj_all, 
                                 Ts, quad.params, traj.xyzType, traj.yawType, ifsave, args.orient)
    plt.show()

if __name__ == "__main__":
    # Run main simulation
    main()