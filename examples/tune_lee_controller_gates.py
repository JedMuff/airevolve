#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Curriculum-Based CMA-ES Optimization for Lee Controller with B-Spline Trajectories

This script implements a 3-stage curriculum learning approach to optimize both
controller gains and B-spline trajectory parameters for gate racing with a 2-inch quad.

Curriculum Stages:
    Stage 1: Optimize controller gains only (4 params)
    Stage 2: Optimize gains + timing parameters (7 params)
    Stage 3: Optimize gains + timing + gate offsets (7 + n_gates×3 params)

Usage:
    # Run all stages automatically (recommended)
    python tune_lee_controller_gates.py --stage all --gates figure8 --max-evals 200

    # Run all stages with different evaluations per stage
    python tune_lee_controller_gates.py --stage all --gates figure8 \\
        --max-evals-1 500 --max-evals-2 1000 --max-evals-3 2000

    # Or run individual stages:
    # Stage 1: Tune gains only
    python tune_lee_controller_gates.py --stage 1 --gates figure8 --max-evals 200

    # Stage 2: Tune gains + timing (load stage 1 results)
    python tune_lee_controller_gates.py --stage 2 --gates figure8 \\
        --load-prev tuning_results_gates/stage1_best.json --max-evals 300

    # Stage 3: Tune gains + timing + trajectory (load stage 2 results)
    python tune_lee_controller_gates.py --stage 3 --gates figure8 \\
        --load-prev tuning_results_gates/stage2_best.json --max-evals 500
"""

import numpy as np
import argparse
import json
import time
import os
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
import multiprocessing

# Import drone simulation components
from airevolve.controllers.trajectory_generation.bspline_gate_trajectory import BSplineGateTrajectory
from airevolve.controllers.lee_control.lee_controller import LeeGeometricControl
from airevolve.simulator.simulation import DroneInterface
from airevolve.controllers.utils.wind_model import Wind
from airevolve.controllers.utils.gate_configs import GATE_CONFIGS

# Try to import CMA-ES
try:
    import cma
    CMA_AVAILABLE = True
except ImportError:
    CMA_AVAILABLE = False
    print("ERROR: 'cma' package not found. Install with: pip install cma")
    import sys
    sys.exit(1)


# ============================================================================
# QUAD CONFIGURATION (FIXED)
# ============================================================================

# 2-inch quadrotor configuration (~0.08kg total mass)
ARM_LENGTH = 0.06  # 60mm arms (true 2-inch micro quad)
PROP_SIZE = 2  # 2-inch propellers

def create_2inch_quad():
    """Create 2-inch quadrotor configuration."""
    propellers = [
        {"loc": [ARM_LENGTH, ARM_LENGTH, 0], "dir": [0, 0, -1, "ccw"], "propsize": PROP_SIZE},
        {"loc": [-ARM_LENGTH, ARM_LENGTH, 0], "dir": [0, 0, -1, "cw"], "propsize": PROP_SIZE},
        {"loc": [-ARM_LENGTH, -ARM_LENGTH, 0], "dir": [0, 0, -1, "ccw"], "propsize": PROP_SIZE},
        {"loc": [ARM_LENGTH, -ARM_LENGTH, 0], "dir": [0, 0, -1, "cw"], "propsize": PROP_SIZE}
    ]
    return DroneInterface(0, propellers=propellers)


# ============================================================================
# GATE CHECKING LOGIC
# ============================================================================

class GateChecker:
    """Handles gate passing detection"""

    def __init__(self, gate_pos, gate_yaw, gate_size=2.0):
        """
        Initialize gate checker

        Args:
            gate_pos: Array of gate positions [N, 3]
            gate_yaw: Array of gate yaw angles [N]
            gate_size: Size of gates in meters
        """
        self.gate_pos = gate_pos
        self.gate_yaw = gate_yaw
        self.gate_size = gate_size
        self.num_gates = len(gate_pos)
        self.current_gate_idx = 0
        self.gates_passed = 0
        self.last_pos = None
        self.max_gate_distance = self._calculate_max_gate_distance()

    def reset(self):
        """Reset gate checker state"""
        self.current_gate_idx = 0
        self.gates_passed = 0
        self.last_pos = None

    def check_gate_passing(self, pos):
        """
        Check if drone passed through current gate

        Args:
            pos: Current position [x, y, z]

        Returns:
            True if gate was passed
        """
        if self.last_pos is None:
            self.last_pos = pos.copy()
            return False

        # Get current gate
        gate_idx = self.current_gate_idx % self.num_gates
        gate_pos = self.gate_pos[gate_idx]
        gate_yaw = self.gate_yaw[gate_idx]

        # Gate normal vector (direction perpendicular to gate plane)
        normal = np.array([np.cos(gate_yaw), np.sin(gate_yaw), 0.0])

        # Project positions onto normal direction
        pos_old_proj = np.dot(self.last_pos - gate_pos, normal)
        pos_new_proj = np.dot(pos - gate_pos, normal)

        # Check if crossed gate plane
        crossed_plane = ((pos_old_proj < 0) and (pos_new_proj > 0)) or \
                       ((pos_old_proj > 0) and (pos_new_proj < 0))

        if crossed_plane:
            # Find intersection point
            t = -pos_old_proj / (pos_new_proj - pos_old_proj)
            intersection = self.last_pos + t * (pos - self.last_pos)

            # Transform to gate's local frame
            rel_pos = intersection - gate_pos

            # Create gate's local coordinate system
            if abs(normal[2]) < 0.9:
                up = np.array([0.0, 0.0, 1.0])
            else:
                up = np.array([0.0, 1.0, 0.0])

            right = np.cross(normal, up)
            right = right / np.linalg.norm(right)
            actual_up = np.cross(right, normal)

            # Project onto gate plane
            lateral = np.dot(rel_pos, right)
            vertical = np.dot(rel_pos, actual_up)

            # Check if within gate opening
            half_size = self.gate_size / 2.0
            within_bounds = (abs(lateral) <= half_size) and (abs(vertical) <= half_size)

            if within_bounds:
                self.gates_passed += 1
                self.current_gate_idx += 1
                self.last_pos = pos.copy()
                return True

        self.last_pos = pos.copy()
        return False

    def _calculate_max_gate_distance(self):
        """
        Calculate the maximum distance between consecutive gates in the sequence

        Returns:
            Maximum distance between any two consecutive gates
        """
        max_dist = 0.0
        for i in range(self.num_gates):
            next_idx = (i + 1) % self.num_gates
            dist = np.linalg.norm(self.gate_pos[next_idx] - self.gate_pos[i])
            max_dist = max(max_dist, dist)
        return max_dist

    def get_normalized_distance_to_next_gate(self, pos):
        """
        Calculate normalized distance to the next gate (current target)

        Args:
            pos: Current position [x, y, z]

        Returns:
            Normalized distance bonus in [0, 1], where:
            - 1.0 means at the gate position
            - 0.0 means at or beyond max_gate_distance away
            - Values capped at 0.0 if distance > max_gate_distance
        """
        # Get the next gate to pass
        gate_idx = self.current_gate_idx % self.num_gates
        gate_pos = self.gate_pos[gate_idx]

        # Calculate distance to next gate
        distance = np.linalg.norm(pos - gate_pos)

        # Normalize: closer = higher score, capped at 0 if too far
        if distance >= self.max_gate_distance:
            return 0.0

        normalized = 1.0 - (distance / self.max_gate_distance)
        return max(0.0, min(1.0, normalized))


# ============================================================================
# SIMULATION
# ============================================================================

def simulate_bspline(pos_gain, vel_gain, att_gain, rate_gain, bspline_params,
                     gate_config, sim_time=20.0, dt=0.005, n_startup_points=1,
                     verbose=False):
    """
    Run simulation with Lee controller and B-spline gate trajectory

    Args:
        pos_gain, vel_gain, att_gain, rate_gain: Controller gains
        bspline_params: Array of B-spline trajectory parameters
        gate_config: Gate configuration class
        sim_time: Simulation time in seconds
        dt: Time step in seconds
        n_startup_points: Number of startup control points
        verbose: If True, show debug output

    Returns:
        Dictionary with gates_passed, crashed, completed, flight_time
    """
    try:
        # Create 2-inch quad
        quad = create_2inch_quad()

        # Create Lee controller with specified gains
        lee_gains = {
            'pos_P_gain': np.array([pos_gain] * 3),
            'vel_P_gain': np.array([vel_gain] * 3),
            'att_P_gain': np.array([att_gain] * 3),
            'rate_P_gain': np.array([rate_gain] * 3)
        }
        ctrl = LeeGeometricControl(quad, yawType=1, orient='NED',
                                   auto_scale_gains=False, **lee_gains)

        # Create B-spline trajectory
        # Use gate_offset_scale=0.5 to constrain offsets to ±half gate size (keeping control points within gates)
        bspline_traj = BSplineGateTrajectory(gate_config, n_startup_points=n_startup_points, gate_offset_scale=0.5)
        bspline_traj.set_parameters(bspline_params)

        # Set drone initial position
        start_pos = bspline_traj.get_start_position()
        quad.drone_sim.set_state(position=start_pos)

        # Create Trajectory wrapper (xyzType=15 for B-spline)
        from airevolve.controllers.trajectory_generation.trajectory import Trajectory
        traj = Trajectory(quad, "xyz_pos", np.array([15, 3, 1]),
                         gate_config=gate_config,
                         bspline_params={'n_startup_points': n_startup_points})
        traj.bspline_trajectory = bspline_traj

        # Create wind model (no wind)
        wind = Wind('None', 2.0, 90, -15)

        # Initialize gate checker
        gate_checker = GateChecker(gate_config.gate_pos, gate_config.gate_yaw,
                                  gate_config.gate_size)

        # Get initial desired state and command
        sDes = traj.desiredState(0, dt, quad)
        ctrl.controller(sDes, quad, "xyz_pos", dt)

        # Run simulation
        t = 0
        crashed = False
        num_steps = int(sim_time / dt)

        for step in range(num_steps):
            # Update dynamics
            quad.update(t, dt, ctrl.w_cmd, wind)
            t += dt

            # Get desired state
            sDes = traj.desiredState(t, dt, quad)

            # Generate control commands
            ctrl.controller(sDes, quad, "xyz_pos", dt)

            # Check gate passing
            gate_checker.check_gate_passing(quad.pos)

            # Check bounds
            if (quad.pos[0] < gate_config.x_bounds[0] or
                quad.pos[0] > gate_config.x_bounds[1] or
                quad.pos[1] < gate_config.y_bounds[0] or
                quad.pos[1] > gate_config.y_bounds[1] or
                quad.pos[2] < gate_config.z_bounds[0] or
                quad.pos[2] > gate_config.z_bounds[1]):
                crashed = True
                break

            # Check for excessive error (instability)
            pos_error = np.linalg.norm(quad.pos - sDes[0:3])
            if pos_error > 10.0:
                crashed = True
                break

        # Calculate normalized distance to next gate at end of simulation
        distance_bonus = gate_checker.get_normalized_distance_to_next_gate(quad.pos)

        return {
            'gates_passed': gate_checker.gates_passed,
            'distance_bonus': distance_bonus,
            'crashed': crashed,
            'completed': not crashed,
            'flight_time': t,
            'success': True
        }

    except Exception as e:
        if verbose:
            print(f"Simulation error: {e}")
        return {
            'gates_passed': 0,
            'distance_bonus': 0.0,
            'crashed': True,
            'completed': False,
            'flight_time': 0,
            'success': False,
            'error': str(e)
        }


# ============================================================================
# MULTIPROCESSING WRAPPER
# ============================================================================

def _evaluate_solution_wrapper(args):
    """Wrapper function for parallel evaluation"""
    (params, stage, gate_config, sim_time, dt, n_startup_points,
     fixed_bspline_params, fixed_gains, fixed_timing, fixed_gate_offsets,
     n_gate_offset_params, n_timing_params) = args

    # Extract parameters based on stage
    if stage == 1:
        # Stage 1: Optimize gains only
        pos_g, vel_g, att_g, rate_g = params[0:4]
        bspline_params = fixed_bspline_params

    elif stage == 2:
        # Stage 2: Optimize gains + timing
        pos_g, vel_g, att_g, rate_g = params[0:4]
        timing_params = params[4:7]

        # Reconstruct full B-spline params with new timing
        start_pos = fixed_bspline_params[0:3]
        n_startup_coords = n_startup_points * 3
        startup_points = fixed_bspline_params[3:3+n_startup_coords]
        gate_offsets = fixed_gate_offsets

        bspline_params = np.concatenate([start_pos, startup_points, gate_offsets, timing_params])

    elif stage == 3:
        # Stage 3: Optimize gains + timing + gate offsets
        pos_g, vel_g, att_g, rate_g = params[0:4]
        timing_params = params[4:7]
        gate_offsets = params[7:7+n_gate_offset_params]

        # Reconstruct full B-spline params
        start_pos = fixed_bspline_params[0:3]
        n_startup_coords = n_startup_points * 3
        startup_points = fixed_bspline_params[3:3+n_startup_coords]

        bspline_params = np.concatenate([start_pos, startup_points, gate_offsets, timing_params])

    else:
        raise ValueError(f"Invalid stage: {stage}")

    # Run simulation
    result = simulate_bspline(
        pos_g, vel_g, att_g, rate_g,
        bspline_params,
        gate_config, sim_time, dt,
        n_startup_points=n_startup_points,
        verbose=False
    )

    if result['success']:
        gates = result['gates_passed']
        distance_bonus = result['distance_bonus']
        penalty = 100 if result['crashed'] else 0

        # Fitness includes gates passed + normalized distance to next gate
        fitness = gates + distance_bonus
        score = -fitness + penalty

        result['score'] = score
        result['fitness'] = fitness
        result['gains'] = {
            'pos_P': pos_g,
            'vel_P': vel_g,
            'att_P': att_g,
            'rate_P': rate_g
        }
        result['bspline_params'] = bspline_params.tolist()

        return (score, result)
    else:
        return (1000.0, None)


# ============================================================================
# CURRICULUM TUNER
# ============================================================================

class CurriculumTuner:
    """Curriculum-based CMA-ES optimization for Lee controller with B-spline trajectories"""

    def __init__(self, gate_config, stage, sim_time=20.0, dt=0.005,
                 output_dir="tuning_results_gates", n_startup_points=1):
        """
        Initialize curriculum tuner

        Args:
            gate_config: Gate configuration class
            stage: Curriculum stage (1, 2, or 3)
            sim_time: Simulation time in seconds
            dt: Time step in seconds
            output_dir: Output directory for results
            n_startup_points: Number of startup control points
        """
        self.gate_config = gate_config
        self.stage = stage
        self.sim_time = sim_time
        self.dt = dt
        self.output_dir = output_dir
        self.n_startup_points = n_startup_points
        self.results = []
        self.best_score = -float('inf')
        self.best_params = None

        # Create B-spline template
        # Use gate_offset_scale=0.5 to constrain offsets to ±half gate size (keeping control points within gates)
        self.bspline_template = BSplineGateTrajectory(gate_config, n_startup_points=n_startup_points, gate_offset_scale=0.5)
        self.n_bspline_params = self.bspline_template.get_parameter_count()
        self.bspline_bounds = self.bspline_template.get_parameter_bounds()
        self.fixed_bspline_params = self.bspline_template.get_default_parameters()

        # Get parameter groups
        self.bspline_bounds_by_group = self.bspline_template.get_parameter_bounds_by_group()
        self.fixed_gate_offsets = self.bspline_template.get_gate_offset_parameters()
        self.n_gate_offset_params = self.bspline_template.get_gate_offset_count()
        self.n_timing_params = self.bspline_template.get_timing_parameter_count()

        # Fixed parameters (will be overridden if loading from previous stage)
        # Initial guess for 2-inch quad based on successful optimization runs
        self.fixed_gains = [2.0, 1.5, 0.6, -0.3]  # pos_P, vel_P, att_P, rate_P

        # Update default timing parameters to better values for 2-inch quads
        # Based on successful runs: faster total time, lower velocity scale, shorter startup
        self.fixed_timing = np.array([15.0, 0.85, 1.5])  # total_time, velocity_scale, startup_time

        # Update fixed_bspline_params to use the new timing defaults
        # Replace the timing parameters (last 3 values) in the default bspline params
        idx = 3 + self.n_startup_points * 3 + self.n_gate_offset_params  # Index where timing starts
        self.fixed_bspline_params[idx:idx+3] = self.fixed_timing

        # Create output directory
        Path(self.output_dir).mkdir(exist_ok=True)

    def load_previous_stage(self, config_path):
        """Load results from previous curriculum stage"""
        print(f"\nLoading previous stage from: {config_path}")
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Load gains
        if 'gains' in config:
            gains = config['gains']
            self.fixed_gains = [
                gains['pos_P'],
                gains['vel_P'],
                gains['att_P'],
                gains['rate_P']
            ]
            print(f"Loaded gains: pos={self.fixed_gains[0]:.3f}, vel={self.fixed_gains[1]:.3f}, "
                  f"att={self.fixed_gains[2]:.3f}, rate={self.fixed_gains[3]:.4f}")

        # Load B-spline params if available
        if 'bspline_params' in config:
            bspline_params = np.array(config['bspline_params'])
            self.fixed_bspline_params = bspline_params

            # Extract timing and gate offsets for stages 2 and 3
            idx = 3 + self.n_startup_points * 3  # After start_pos and startup_points
            self.fixed_gate_offsets = bspline_params[idx:idx+self.n_gate_offset_params]
            self.fixed_timing = bspline_params[idx+self.n_gate_offset_params:idx+self.n_gate_offset_params+3]

            print(f"Loaded B-spline parameters ({len(bspline_params)} params)")
            print(f"  Timing: total_time={self.fixed_timing[0]:.2f}s, "
                  f"vel_scale={self.fixed_timing[1]:.2f}, startup_time={self.fixed_timing[2]:.2f}s")

    def get_optimization_config(self):
        """Get optimization configuration for current stage"""
        if self.stage == 1:
            # Stage 1: Optimize gains only (4 params)
            # Bounds for 2-inch quads with varying arm lengths
            # Based on optimization data: working region pos~1-4, vel~0.8-3, att~0.3-1.5, rate~-0.8 to -0.05
            initial_guess = self.fixed_gains
            bounds = [
                [0.01, 10.0],     # pos_P (relaxed for different arm lengths)
                [0.01, 10.0],     # vel_P (relaxed for different arm lengths)
                [0.01, 10.0],     # att_P (wider range for different inertias)
                [-5.0, -0.01]   # rate_P (full range for different configurations)
            ]
            initial_std = 0.8  # Moderate search width
            param_count = 4
            param_description = "Controller gains (4 params)"

        elif self.stage == 2:
            # Stage 2: Optimize gains + timing (7 params)
            # Timing may need to be faster for more agile 2-inch quad
            initial_guess = self.fixed_gains + self.fixed_timing.tolist()
            bounds = [
                [0.01, 10.0],     # pos_P
                [0.01, 10.0],     # vel_P
                [0.01, 10.0],     # att_P
                [-5.0, -0.01],  # rate_P
                [3.0, 15.1],    # total_time (faster than matched quad)
                [0.1, 10.0],     # velocity_scale (may need higher for agile quad)
                [0.1, 2.0]      # startup_time (shorter for faster quad)
            ]
            initial_std = 0.8
            param_count = 7
            param_description = "Gains (4) + Timing (3)"

        elif self.stage == 3:
            # Stage 3: Optimize gains + timing + gate offsets (7 + n_gates×3 params)
            initial_guess = self.fixed_gains + self.fixed_timing.tolist() + self.fixed_gate_offsets.tolist()

            # Bounds for gains + timing (adjusted for 2-inch quad)
            bounds = [
                [0.01, 10.0],     # pos_P
                [0.01, 10.0],     # vel_P
                [0.01, 10.0],     # att_P
                [-5.0, -0.01],  # rate_P
                [3.0, 15.1],    # total_time (relaxed for 2-inch quad)
                [0.1, 10.0],     # velocity_scale (relaxed for 2-inch quad)
                [0.1, 2.0]      # startup_time (relaxed for 2-inch quad)
            ]

            # Add bounds for gate offsets
            gate_lower, gate_upper = self.bspline_bounds_by_group['gate_offsets']
            for i in range(len(gate_lower)):
                bounds.append([gate_lower[i], gate_upper[i]])

            initial_std = 0.3  # Smaller std to stay close to Stage 2's working solution
            param_count = 7 + self.n_gate_offset_params
            param_description = f"Gains (4) + Timing (3) + Gate Offsets ({self.n_gate_offset_params})"

        else:
            raise ValueError(f"Invalid stage: {self.stage}. Must be 1, 2, or 3.")

        return {
            'initial_guess': initial_guess,
            'bounds': bounds,
            'initial_std': initial_std,
            'param_count': param_count,
            'description': param_description
        }

    def objective_function(self, params):
        """Objective function for CMA-ES"""
        if self.stage == 1:
            pos_g, vel_g, att_g, rate_g = params[0:4]
            bspline_params = self.fixed_bspline_params

        elif self.stage == 2:
            pos_g, vel_g, att_g, rate_g = params[0:4]
            timing_params = params[4:7]

            # Reconstruct B-spline params
            start_pos = self.fixed_bspline_params[0:3]
            n_startup_coords = self.n_startup_points * 3
            startup_points = self.fixed_bspline_params[3:3+n_startup_coords]

            bspline_params = np.concatenate([start_pos, startup_points,
                                            self.fixed_gate_offsets, timing_params])

        elif self.stage == 3:
            pos_g, vel_g, att_g, rate_g = params[0:4]
            timing_params = params[4:7]
            gate_offsets = params[7:7+self.n_gate_offset_params]

            # Reconstruct B-spline params
            start_pos = self.fixed_bspline_params[0:3]
            n_startup_coords = self.n_startup_points * 3
            startup_points = self.fixed_bspline_params[3:3+n_startup_coords]

            bspline_params = np.concatenate([start_pos, startup_points,
                                            gate_offsets, timing_params])

        result = simulate_bspline(
            pos_g, vel_g, att_g, rate_g,
            bspline_params,
            self.gate_config, self.sim_time, self.dt,
            n_startup_points=self.n_startup_points,
            verbose=False
        )

        if result['success']:
            gates = result['gates_passed']
            distance_bonus = result['distance_bonus']
            penalty = 100 if result['crashed'] else 0

            # Fitness includes gates passed + normalized distance to next gate
            fitness = gates + distance_bonus
            score = -fitness + penalty

            result['score'] = score
            result['fitness'] = fitness
            result['gains'] = {
                'pos_P': pos_g,
                'vel_P': vel_g,
                'att_P': att_g,
                'rate_P': rate_g
            }
            result['bspline_params'] = bspline_params.tolist()

            self.results.append(result)

            # Track best
            if fitness > self.best_score and not result['crashed']:
                self.best_score = fitness
                self.best_params = result

            return score
        else:
            return 1000.0

    def run_optimization(self, max_evaluations=200, num_workers=None, timeout_per_eval=30.0):
        """Run CMA-ES optimization for current curriculum stage"""
        if not CMA_AVAILABLE:
            print("ERROR: CMA-ES requires the 'cma' package")
            return

        # Set up parallel workers
        if num_workers is None:
            num_workers = max(1, multiprocessing.cpu_count() // 2)

        # Get optimization configuration
        config = self.get_optimization_config()

        print(f"\n{'='*70}")
        print(f"CURRICULUM STAGE {self.stage} - LEE CONTROLLER CMA-ES OPTIMIZATION")
        print(f"{'='*70}")
        print(f"Gate configuration: {self.gate_config.__name__}")
        print(f"Number of gates: {len(self.gate_config.gate_pos)}")
        print(f"Drone: 2-inch quad (arm_length={ARM_LENGTH}m)")
        print(f"Optimization: {config['description']}")
        print(f"Total parameters: {config['param_count']}")
        print(f"Max evaluations: {max_evaluations}")
        print(f"Parallel workers: {num_workers}")
        print(f"Simulation time: {self.sim_time}s")
        print(f"{'='*70}\n")

        # CMA-ES options
        options = {
            'bounds': [list(b) for b in zip(*config['bounds'])],
            'maxfevals': max_evaluations,
            'verb_disp': 1,
            'verb_log': 0,
            'tolx': 1e-8,
            'tolfun': 1e-6,
            'tolfunhist': 1e-6,
        }

        # Run CMA-ES
        print("Starting CMA-ES optimization...\n")
        start_time = time.time()

        try:
            es = cma.CMAEvolutionStrategy(config['initial_guess'], config['initial_std'], options)

            iteration = 0

            # For Stages 2 and 3: Evaluate the exact initial guess (previous stage solution) first
            # This ensures we don't regress from the previous stage's performance
            if self.stage in [2, 3]:
                print(f"Evaluating Stage {self.stage - 1} solution (initial guess) before optimization...")
                initial_result = self.objective_function(config['initial_guess'])
                if initial_result < 1000.0:
                    print(f"Stage {self.stage - 1} solution baseline: {-initial_result:.2f} fitness")
                print()

            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                while not es.stop():
                    iteration += 1
                    solutions = es.ask()
                    print(f"Iteration {iteration}: Evaluating {len(solutions)} solutions...", flush=True)

                    # Parallel evaluation
                    if num_workers > 1:
                        eval_args = [
                            (sol, self.stage, self.gate_config, self.sim_time, self.dt,
                             self.n_startup_points, self.fixed_bspline_params, self.fixed_gains,
                             self.fixed_timing, self.fixed_gate_offsets, self.n_gate_offset_params,
                             self.n_timing_params)
                            for sol in solutions
                        ]

                        future_to_sol = {
                            executor.submit(_evaluate_solution_wrapper, args): args[0]
                            for args in eval_args
                        }

                        results_dict = {}
                        completed = 0
                        timeouts = 0
                        for future in as_completed(future_to_sol):
                            sol = future_to_sol[future]
                            try:
                                score, result = future.result(timeout=timeout_per_eval)
                                results_dict[tuple(sol)] = score

                                if result is not None:
                                    self.results.append(result)

                                    if result['fitness'] > self.best_score and not result['crashed']:
                                        self.best_score = result['fitness']
                                        self.best_params = result

                                completed += 1
                                print(f"  Completed {completed}/{len(solutions)}", end='\r', flush=True)
                            except TimeoutError:
                                print(f"\n  Timeout evaluating solution (>{timeout_per_eval}s)", flush=True)
                                results_dict[tuple(sol)] = 1000.0
                                completed += 1
                                timeouts += 1
                            except Exception as e:
                                print(f"\n  Error evaluating solution: {e}", flush=True)
                                results_dict[tuple(sol)] = 1000.0
                                completed += 1

                        print()
                        if timeouts > 0:
                            print(f"  Warning: {timeouts} evaluations timed out")
                        fitness_values = [results_dict[tuple(sol)] for sol in solutions]
                    else:
                        # Serial execution
                        fitness_values = [self.objective_function(x) for x in solutions]

                    es.tell(solutions, fitness_values)

                    # Print progress
                    best_idx = np.argmin(fitness_values)
                    best_solution = solutions[best_idx]
                    best_score = fitness_values[best_idx]

                    print(f"\n{'='*70}")
                    print(f"Iteration {iteration} | Evaluated: {es.result.evaluations}/{max_evaluations}")
                    print(f"{'='*70}")
                    print(f"Best in generation: Fitness={-best_score:.2f}")
                    print(f"  Gains: pos={best_solution[0]:.2f}, vel={best_solution[1]:.2f}, "
                          f"att={best_solution[2]:.2f}, rate={best_solution[3]:.3f}")

                    if self.stage >= 2 and len(best_solution) >= 7:
                        print(f"  Timing: total_time={best_solution[4]:.2f}s, "
                              f"vel_scale={best_solution[5]:.2f}, startup_time={best_solution[6]:.2f}s")

                    print(f"\nBest ever: {self.best_score:.2f} fitness")
                    if self.best_params:
                        gains = self.best_params['gains']
                        print(f"  Gains: pos={gains['pos_P']:.2f}, vel={gains['vel_P']:.2f}, "
                              f"att={gains['att_P']:.2f}, rate={gains['rate_P']:.3f}")
                    print(f"{'='*70}")

                    # Save intermediate results
                    if iteration % 5 == 0:
                        self._save_results()

            end_time = time.time()
            elapsed = end_time - start_time

            # Print final results
            print(f"\n{'='*70}")
            print(f"STAGE {self.stage} OPTIMIZATION COMPLETE")
            print(f"{'='*70}")
            print(f"Time elapsed: {elapsed/60:.1f} minutes")
            print(f"Total evaluations: {es.result.evaluations}")
            print(f"\nBest result: {self.best_score:.2f} fitness")

            if self.best_params:
                gates = self.best_params.get('gates_passed', 0)
                distance_bonus = self.best_params.get('distance_bonus', 0.0)
                print(f"  Gates passed: {gates}")
                print(f"  Distance bonus: {distance_bonus:.3f}")

                gains = self.best_params['gains']
                print(f"\nBest gains:")
                print(f"  Position: {gains['pos_P']:.3f}")
                print(f"  Velocity: {gains['vel_P']:.3f}")
                print(f"  Attitude: {gains['att_P']:.3f}")
                print(f"  Rate: {gains['rate_P']:.4f}")
                print(f"\nFlight time: {self.best_params['flight_time']:.2f}s")
                print(f"Crashed: {self.best_params['crashed']}")

            print(f"{'='*70}\n")

            # Save final results
            self._save_results()
            self._save_best_config()

        except KeyboardInterrupt:
            print("\n\nOptimization interrupted. Saving results...")
            self._save_results()
            self._save_best_config()
        except Exception as e:
            print(f"\n\nERROR: {e}")
            import traceback
            traceback.print_exc()
            self._save_results()

    def _save_results(self):
        """Save all results to JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.output_dir}/stage{self.stage}_results_{timestamp}.json"

        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"Results saved to: {filename}")

    def _save_best_config(self):
        """Save best configuration for current stage"""
        if self.best_params is None:
            return

        config = {
            'stage': self.stage,
            'timestamp': datetime.now().isoformat(),
            'fitness': self.best_score,
            'gates_passed': self.best_params.get('gates_passed', 0),
            'distance_bonus': self.best_params.get('distance_bonus', 0.0),
            'gains': self.best_params['gains'],
            'bspline_params': self.best_params['bspline_params'],
            'n_startup_points': self.n_startup_points,
            'flight_time': self.best_params['flight_time'],
            'crashed': self.best_params['crashed']
        }

        config_file = f"{self.output_dir}/stage{self.stage}_best.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"Best config saved to: {config_file}")

        # Print next steps
        print(f"\n{'='*70}")
        if self.stage < 3:
            print(f"TO CONTINUE TO STAGE {self.stage + 1}:")
            print(f"{'='*70}")
            print(f"python tune_lee_controller_gates.py --stage {self.stage + 1} \\")
            print(f"    --gates [gate_config] \\")
            print(f"    --load-prev {config_file} \\")
            print(f"    --max-evals [num_evals]")
        else:
            print("TO VISUALIZE FINAL RESULT:")
            print(f"{'='*70}")
            print(f"python examples/run_3D_simulation_lee_ctrl.py \\")
            print(f"    --bspline-config {config_file} \\")
            print(f"    --gates [gate_config] \\")
            print(f"    --time {self.sim_time:.1f}")
        print(f"{'='*70}\n")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Curriculum-Based CMA-ES Optimization for Lee Controller',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--stage', type=str, required=True,
                       help='Curriculum stage: 1=gains, 2=gains+timing, 3=gains+timing+trajectory, all=run all stages')
    parser.add_argument('--gates', type=str, required=True,
                       choices=['figure8', 'circle', 'slalom', 'backandforth'],
                       help='Gate configuration')
    parser.add_argument('--load-prev', type=str, default=None,
                       help='Load previous stage results (JSON file)')
    parser.add_argument('--max-evals', type=int, default=200,
                       help='Maximum evaluations per stage (default: 200)')
    parser.add_argument('--max-evals-1', type=int, default=None,
                       help='Maximum evaluations for Stage 1 (overrides --max-evals)')
    parser.add_argument('--max-evals-2', type=int, default=None,
                       help='Maximum evaluations for Stage 2 (overrides --max-evals)')
    parser.add_argument('--max-evals-3', type=int, default=None,
                       help='Maximum evaluations for Stage 3 (overrides --max-evals)')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers (default: CPU_count/2)')
    parser.add_argument('--time', type=float, default=20.0,
                       help='Simulation time in seconds (default: 20.0)')
    parser.add_argument('--dt', type=float, default=0.005,
                       help='Time step in seconds (default: 0.005)')
    parser.add_argument('--output', type=str, default='tuning_results_gates',
                       help='Output directory (default: tuning_results_gates)')
    parser.add_argument('--n-startup-points', type=int, default=1,
                       help='Number of startup control points (default: 1)')
    parser.add_argument('--timeout', type=float, default=30.0,
                       help='Timeout per evaluation in seconds (default: 30.0)')

    args = parser.parse_args()

    # Get gate configuration
    gate_config = GATE_CONFIGS[args.gates]

    # Handle 'all' stage option - run all stages sequentially
    if args.stage.lower() == 'all':
        # Determine max evaluations for each stage
        max_evals_1 = args.max_evals_1 if args.max_evals_1 is not None else args.max_evals
        max_evals_2 = args.max_evals_2 if args.max_evals_2 is not None else args.max_evals
        max_evals_3 = args.max_evals_3 if args.max_evals_3 is not None else args.max_evals

        print("\n" + "="*70)
        print("RUNNING ALL CURRICULUM STAGES SEQUENTIALLY")
        print("="*70)
        print("This will run Stage 1 → Stage 2 → Stage 3 automatically")
        print(f"Gate configuration: {args.gates}")
        print(f"Max evaluations:")
        print(f"  Stage 1 (gains):           {max_evals_1}")
        print(f"  Stage 2 (gains + timing):  {max_evals_2}")
        print(f"  Stage 3 (full):            {max_evals_3}")
        print("="*70 + "\n")

        # Run Stage 1
        print("\n" + "#"*70)
        print("# STARTING STAGE 1: GAINS OPTIMIZATION")
        print("#"*70 + "\n")

        tuner1 = CurriculumTuner(
            gate_config,
            stage=1,
            sim_time=args.time,
            dt=args.dt,
            output_dir=args.output,
            n_startup_points=args.n_startup_points
        )
        tuner1.run_optimization(
            max_evaluations=max_evals_1,
            num_workers=args.workers,
            timeout_per_eval=args.timeout
        )
        stage1_config = f"{args.output}/stage1_best.json"

        # Check if Stage 1 succeeded
        if not os.path.exists(stage1_config):
            print("\n" + "="*70)
            print("ERROR: Stage 1 failed to produce results!")
            print("="*70)
            print(f"Expected config file not found: {stage1_config}")
            print("Cannot continue to Stage 2 and 3.")
            print("="*70 + "\n")
            return

        # Run Stage 2 (load Stage 1 results)
        print("\n" + "#"*70)
        print("# STARTING STAGE 2: GAINS + TIMING OPTIMIZATION")
        print("#"*70 + "\n")

        tuner2 = CurriculumTuner(
            gate_config,
            stage=2,
            sim_time=args.time,
            dt=args.dt,
            output_dir=args.output,
            n_startup_points=args.n_startup_points
        )
        tuner2.load_previous_stage(stage1_config)
        tuner2.run_optimization(
            max_evaluations=max_evals_2,
            num_workers=args.workers,
            timeout_per_eval=args.timeout
        )
        stage2_config = f"{args.output}/stage2_best.json"

        # Check if Stage 2 succeeded
        if not os.path.exists(stage2_config):
            print("\n" + "="*70)
            print("ERROR: Stage 2 failed to produce results!")
            print("="*70)
            print(f"Expected config file not found: {stage2_config}")
            print("Cannot continue to Stage 3.")
            print(f"\nStage 1 results available: {stage1_config}")
            print("="*70 + "\n")
            return

        # Run Stage 3 (load Stage 2 results)
        print("\n" + "#"*70)
        print("# STARTING STAGE 3: FULL OPTIMIZATION (GAINS + TIMING + TRAJECTORY)")
        print("#"*70 + "\n")

        tuner3 = CurriculumTuner(
            gate_config,
            stage=3,
            sim_time=args.time,
            dt=args.dt,
            output_dir=args.output,
            n_startup_points=args.n_startup_points
        )
        tuner3.load_previous_stage(stage2_config)
        tuner3.run_optimization(
            max_evaluations=max_evals_3,
            num_workers=args.workers,
            timeout_per_eval=args.timeout
        )

        # Print final summary
        print("\n" + "="*70)
        print("ALL CURRICULUM STAGES COMPLETE!")
        print("="*70)
        print(f"\nStage 1 (Gains):              {tuner1.best_score:.2f} fitness")
        print(f"Stage 2 (Gains + Timing):     {tuner2.best_score:.2f} fitness")
        print(f"Stage 3 (Full Optimization):  {tuner3.best_score:.2f} fitness")
        print(f"\nFinal config saved to: {args.output}/stage3_best.json")
        print("\nTo visualize the result:")
        print(f"python examples/run_3D_simulation_lee_ctrl.py \\")
        print(f"    --bspline-config {args.output}/stage3_best.json \\")
        print(f"    --gates {args.gates} \\")
        print(f"    --time {args.time:.1f}")
        print("="*70 + "\n")

    else:
        # Single stage execution
        stage = int(args.stage)
        if stage not in [1, 2, 3]:
            parser.error(f"Invalid stage '{args.stage}'. Must be 1, 2, 3, or 'all'")

        # Create tuner
        tuner = CurriculumTuner(
            gate_config,
            stage=stage,
            sim_time=args.time,
            dt=args.dt,
            output_dir=args.output,
            n_startup_points=args.n_startup_points
        )

        # Load previous stage if specified
        if args.load_prev:
            tuner.load_previous_stage(args.load_prev)

        # Run optimization
        tuner.run_optimization(
            max_evaluations=args.max_evals,
            num_workers=args.workers,
            timeout_per_eval=args.timeout
        )


if __name__ == '__main__':
    main()
