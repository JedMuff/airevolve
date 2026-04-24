# -*- coding: utf-8 -*-
"""
3D Simulation with Lee Geometric Controller - B-Spline Visualization Tool

This script is designed to visualize and test tuned Lee controllers with B-spline
gate trajectories on a 2-inch quad. It can load tuned configurations from the
curriculum-based tuner or run with default parameters for testing.

Primary use: Visualize results from curriculum-based tuning
Secondary use: Standalone testing with default B-spline trajectory

Usage:
    # Visualize tuned controller (primary use)
    python examples/simulation/run_3D_simulation_lee_ctrl.py \\
        --bspline-config tuning_results_gates/stage3_best.json \\
        --gates figure8 \\
        --time 20

    # Standalone with defaults
    python examples/simulation/run_3D_simulation_lee_ctrl.py \\
        --gates circle \\
        --time 15

    # Headless (no visualization)
    python examples/simulation/run_3D_simulation_lee_ctrl.py \\
        --bspline-config results.json --gates figure8 --no-viz
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
import json
import sys
import os

from airevolve.controllers.trajectory_generation.trajectory import Trajectory
from airevolve.controllers.trajectory_generation.bspline_gate_trajectory import BSplineGateTrajectory
from airevolve.controllers.lee_control.lee_controller import LeeGeometricControl
from airevolve.simulator.simulation import DroneInterface
from airevolve.controllers.utils.wind_model import Wind
from airevolve.controllers.utils.gate_configs import GATE_CONFIGS
import airevolve.controllers.utils as utils

# Import GateChecker from the tuning example, which lives as a sibling
# directory (examples/tuning/) after the examples/ reorg.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tuning'))
try:
    from tune_lee_controller_gates import GateChecker, ARM_LENGTH, PROP_SIZE, create_2inch_quad
except ImportError:
    print("Warning: Could not import from tune_lee_controller_gates.py")
    GateChecker = None
    ARM_LENGTH = 0.07
    PROP_SIZE = 2

    def create_2inch_quad():
        """Fallback: Create standard 2-inch prop quadrotor configuration."""
        propellers = [
            {"loc": [ARM_LENGTH, ARM_LENGTH, 0], "dir": [0, 0, -1, "ccw"], "propsize": PROP_SIZE},
            {"loc": [-ARM_LENGTH, ARM_LENGTH, 0], "dir": [0, 0, -1, "cw"], "propsize": PROP_SIZE},
            {"loc": [-ARM_LENGTH, -ARM_LENGTH, 0], "dir": [0, 0, -1, "ccw"], "propsize": PROP_SIZE},
            {"loc": [ARM_LENGTH, -ARM_LENGTH, 0], "dir": [0, 0, -1, "cw"], "propsize": PROP_SIZE}
        ]
        return DroneInterface(0, propellers=propellers)


def quad_sim(t, Ts, quad, ctrl, wind, traj, gate_checker=None):
    """
    Single simulation step for 2-inch quad with Lee control.

    Args:
        t: Current time
        Ts: Time step
        quad: DroneInterface instance
        ctrl: LeeGeometricControl instance
        wind: Wind model
        traj: Trajectory instance
        gate_checker: Optional GateChecker instance

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

    # Check for gate passing
    if gate_checker is not None:
        passed = gate_checker.check_gate_passing(quad.pos)
        if passed:
            print(f"\n*** GATE {gate_checker.gates_passed} PASSED at t={t:.3f}s ***")
            print(f"    Position: [{quad.pos[0]:.2f}, {quad.pos[1]:.2f}, {quad.pos[2]:.2f}]")

    return t


def main():
    """Main simulation function for B-spline trajectory visualization."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Lee Geometric Controller B-Spline Visualization for 2-inch Quad',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--gates', type=str, required=True,
                       choices=['figure8', 'circle', 'slalom', 'backandforth'],
                       help='Gate configuration (required)')
    parser.add_argument('--bspline-config', type=str,
                       help='Path to B-spline configuration JSON file from tuning')
    parser.add_argument('--time', type=float, default=20.0,
                       help='Simulation time in seconds (default: 20)')
    parser.add_argument('--dt', type=float, default=0.005,
                       help='Time step in seconds (default: 0.005)')
    parser.add_argument('--no-viz', action='store_true',
                       help='Disable visualization (for debugging)')
    parser.add_argument('--save', action='store_true',
                       help='Save animation')

    args = parser.parse_args()

    start_time = time.time()

    # Simulation Setup
    Ti = 0
    Ts = args.dt
    Tf = args.time
    ifsave = 1 if args.save else 0

    print("\n" + "="*70)
    print("LEE CONTROLLER B-SPLINE VISUALIZATION - 2-INCH QUAD")
    print("="*70)
    print(f"Gate configuration: {args.gates}")
    print(f"Drone: 2-inch quad (arm_length={ARM_LENGTH}m, prop_size={PROP_SIZE}\")")
    print(f"Simulation time: {Tf}s")
    if args.bspline_config:
        print(f"Loading config: {args.bspline_config}")
    else:
        print("Using default B-spline parameters")
    print("="*70 + "\n")

    # Create 2-inch quad
    quad = create_2inch_quad()

    # Get gate configuration
    gate_config = GATE_CONFIGS[args.gates]

    bspline_params = None
    lee_gains = {}
    use_auto_scaling = True

    # Load configuration if provided
    if args.bspline_config:
        print(f"Loading configuration from: {args.bspline_config}")
        with open(args.bspline_config, 'r') as f:
            config = json.load(f)

        # Extract B-spline parameters
        if 'bspline_params' in config:
            bspline_params = np.array(config['bspline_params'])
            print(f"  Loaded {len(bspline_params)} B-spline parameters")

        # Extract controller gains
        if 'gains' in config:
            gains = config['gains']
            lee_gains = {
                'pos_P_gain': np.array([gains['pos_P']] * 3),
                'vel_P_gain': np.array([gains['vel_P']] * 3),
                'att_P_gain': np.array([gains['att_P']] * 3),
                'rate_P_gain': np.array([gains['rate_P']] * 3)
            }
            use_auto_scaling = False  # Disable auto-scaling with tuned gains

            print(f"\n  Tuned controller gains:")
            print(f"    Position P: {gains['pos_P']:.3f}")
            print(f"    Velocity P: {gains['vel_P']:.3f}")
            print(f"    Attitude P: {gains['att_P']:.3f}")
            print(f"    Rate P: {gains['rate_P']:.4f}")

            if 'gates_passed' in config:
                print(f"  Gates passed during tuning: {config['gates_passed']}")
            if 'stage' in config:
                print(f"  Curriculum stage: {config['stage']}")

    # Create B-spline trajectory
    traj = Trajectory(quad, "xyz_pos", np.array([15, 3, 1]),
                     gate_config=gate_config)

    # Set B-spline parameters if loaded from config
    if bspline_params is not None:
        traj.bspline_trajectory.set_parameters(bspline_params)
        print("\n  B-spline trajectory loaded from config")

        # Debug: check initial trajectory state
        initial_pos, initial_vel, _ = traj.bspline_trajectory.evaluate(0.0)
        print(f"  Trajectory starts at: pos={initial_pos}")
    else:
        print("\n  Using default B-spline trajectory")

    # CRITICAL: Get start position AFTER setting parameters (parameters may override default start position)
    # Use evaluate(0.0) to get the actual trajectory position at t=0, not get_start_position()
    # which returns the control point that may differ slightly from the spline evaluation
    start_pos, start_vel, _ = traj.bspline_trajectory.evaluate(0.0)

    # Compute initial yaw from trajectory's initial heading direction
    # Use velocity at t=0.05s to get stable heading direction (startup phase uses quintic ramp)
    _, vel_050, _ = traj.bspline_trajectory.evaluate(0.05)
    if np.linalg.norm(vel_050[:2]) > 0.001:  # If horizontal velocity > 0.001 m/s
        initial_yaw = np.arctan2(vel_050[1], vel_050[0])
    else:
        # Fallback to first gate direction if velocity is too small
        initial_yaw = gate_config.gate_yaw[0]

    initial_euler = np.array([0.0, 0.0, initial_yaw])  # Start level, facing initial heading

    # Set drone state (attitude parameter uses euler angles, not quaternion)
    # Start from rest (zero velocity) as trajectory expects
    quad.drone_sim.set_state(position=start_pos, velocity=np.zeros(3),
                            attitude=initial_euler, angular_velocity=np.zeros(3))
    quad._update_state_variables()

    print(f"  Drone initial position: [{start_pos[0]:.2f}, {start_pos[1]:.2f}, {start_pos[2]:.2f}]")
    print(f"  Drone initial velocity: [0.00, 0.00, 0.00] m/s")
    print(f"  Drone initial yaw: {initial_yaw*180/np.pi:.2f}° (trajectory heading)\n")

    # Create Lee Geometric Controller
    ctrl = LeeGeometricControl(quad, yawType=1, orient='NED',
                               auto_scale_gains=use_auto_scaling,
                               **lee_gains)

    if use_auto_scaling:
        print("Using auto-scaled controller gains (no tuned config provided)\n")
    else:
        print("Using tuned controller gains (auto-scaling disabled)\n")

    # Initialize Wind (no wind)
    wind = Wind('None', 2.0, 90, -15)

    # Initialize gate checker
    gate_checker = None
    if GateChecker is not None:
        gate_checker = GateChecker(gate_config.gate_pos, gate_config.gate_yaw, gate_config.gate_size)
        print(f"Gate pass detection enabled ({gate_checker.num_gates} gates)\n")

    # Trajectory for First Desired States
    sDes = traj.desiredState(0, Ts, quad)

    # Generate First Commands
    ctrl.controller(sDes, quad, traj.ctrlType, Ts)

    # Initialize Result Matrices
    numTimeStep = int(Tf/Ts+1)

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
    wMotor_all     = np.zeros([numTimeStep, 4])  # 2-inch quad always has 4 motors
    thr_all        = np.zeros([numTimeStep, 4])
    tor_all        = np.zeros([numTimeStep, 4])

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
    wMotor_all[0,:]     = quad.wMotor
    thr_all[0,:]        = quad.thr
    tor_all[0,:]        = quad.tor

    # Run Simulation
    print(f"Running simulation for {Tf}s with Lee Geometric Control...")
    t = Ti
    i = 1
    while round(t,3) < Tf:

        t_new = quad_sim(t, Ts, quad, ctrl, wind, traj, gate_checker)

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
        wMotor_all[i,:]      = quad.wMotor
        thr_all[i,:]         = quad.thr
        tor_all[i,:]         = quad.tor

        i += 1

    end_time = time.time()
    print("Simulated {:.2f}s in {:.6f}s.".format(t, end_time - start_time))

    # Print gate pass summary
    if gate_checker is not None:
        print(f"\n{'='*70}")
        print(f"GATE PASS SUMMARY")
        print(f"{'='*70}")
        print(f"Total gates passed: {gate_checker.gates_passed} / {gate_checker.num_gates}")
        print(f"{'='*70}\n")

    # Visualization
    if not args.no_viz:
        # Generate plots
        utils.makeFigures(quad.params, t_all, pos_all, vel_all, quat_all, omega_all,
                         euler_all, w_cmd_all, wMotor_all, thr_all, tor_all,
                         sDes_traj_all, sDes_calc_all)

        # Create animation with gates and B-spline trajectory
        gate_pos = gate_config.gate_pos
        gate_yaw = gate_config.gate_yaw
        gate_size = gate_config.gate_size
        bspline_traj = traj.bspline_trajectory

        ani = utils.sameAxisAnimation(t_all, traj.wps, pos_all, quat_all, sDes_traj_all,
                                     Ts, quad.params, traj.xyzType, traj.yawType, ifsave, 'NED',
                                     gate_pos=gate_pos, gate_yaw=gate_yaw, gate_size=gate_size,
                                     bspline_traj=bspline_traj)
        plt.show()
    else:
        print("\nVisualization disabled (--no-viz flag)")


if __name__ == "__main__":
    main()
