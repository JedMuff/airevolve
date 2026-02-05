#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CMA-ES Optimization for Lee Controller with Gate-Based Fitness

This script uses CMA-ES to optimize both controller gains and trajectory
parameters, with fitness based on the number of gates passed in correct order.

Fitness Function:
- Number of gates passed through in correct order within time limit
- Penalties for crashes and out-of-bounds
- Higher is better

Usage:
    python tune_lee_controller_gates.py --max-evals 200 --gate-cfg figure8
    python tune_lee_controller_gates.py --max-evals 500 --workers 8
    python tune_lee_controller_gates.py --gate-cfg circle --time 20.0
"""

import numpy as np
import argparse
import json
import time
import os
import sys
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed, TimeoutError
import multiprocessing
import signal

# Import drone simulation components
from airevolve.controllers.trajectory_generation.trajectory import Trajectory
from airevolve.controllers.lee_control.lee_controller import LeeGeometricControl
from airevolve.simulator.simulation import DroneInterface
from airevolve.controllers.utils.wind_model import Wind

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
# GATE CONFIGURATIONS
# ============================================================================

class GateConfig:
    """Base class for gate configurations"""
    gate_pos = None
    gate_yaw = None
    gate_size = 2.0  # Gate size in meters
    x_bounds = [-10, 10]
    y_bounds = [-10, 10]
    z_bounds = [-2, 2]

class Figure8Gates(GateConfig):
    """Figure-8 gate configuration"""
    gate_pos = np.array([
        [-1.5,  1.5,  0.0],
        [ 0.0,  0.0,  0.0],
        [ 1.5, -1.5,  0.0],
        [ 3.0,  0.0,  0.0],
        [ 1.5,  1.5,  0.0],
        [ 0.0,  0.0,  0.0],
        [-1.5, -1.5,  0.0],
        [-3.0,  0.0,  0.0],
    ], dtype=np.float64)
    gate_yaw = np.array([0, -1, 0, 1, 2, -1, 2, 1], dtype=np.float64) * np.pi / 2
    x_bounds = np.array([-5, 5], dtype=np.float64)
    y_bounds = np.array([-3, 3], dtype=np.float64)
    z_bounds = np.array([-2, 2], dtype=np.float64)

class CircleGates(GateConfig):
    """Circular gate configuration"""
    gate_pos = np.array([
        [ 0.0, -2.0,  0.0],
        [ 2.0,  0.0,  0.0],
        [ 0.0,  2.0,  0.0],
        [-2.0,  0.0,  0.0]
    ], dtype=np.float64)
    gate_yaw = np.array([0, 1, 2, 3], dtype=np.float64) * np.pi / 2
    x_bounds = np.array([-4, 4], dtype=np.float64)
    y_bounds = np.array([-4, 4], dtype=np.float64)
    z_bounds = np.array([-2, 2], dtype=np.float64)

class SlalomGates(GateConfig):
    """Slalom gate configuration"""
    num_gates = 20
    gate_pos = np.array([[x, (i % 2) * (1 if i % 4 == 1 else -1), 0]
                         for i, x in enumerate(range(0, num_gates*2, 2))],
                        dtype=np.float64)
    gate_yaw = np.tile([1, 0, -1, 0], num_gates) * np.pi / 2
    x_bounds = np.array([-2, num_gates*2+2], dtype=np.float64)
    y_bounds = np.array([-3, 3], dtype=np.float64)
    z_bounds = np.array([-2, 2], dtype=np.float64)

class BackAndForthGates(GateConfig):
    """Back and forth gate configuration"""
    gate_pos = np.array([
        [ 2.0,  0.0,  0.0],
        [ 8.0,  0.0,  0.0],
        [ 8.0,  0.0,  0.0],
        [ 2.0,  0.0,  0.0],
    ], dtype=np.float64)
    gate_yaw = np.array([0, 0, 2, 2], dtype=np.float64) * np.pi / 2
    x_bounds = np.array([-1, 11], dtype=np.float64)
    y_bounds = np.array([-2, 2], dtype=np.float64)
    z_bounds = np.array([-2, 2], dtype=np.float64)


GATE_CONFIGS = {
    'figure8': Figure8Gates,
    'circle': CircleGates,
    'slalom': SlalomGates,
    'backandforth': BackAndForthGates
}


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

        # Check if crossed gate plane (either direction)
        crossed_plane = ((pos_old_proj < 0) and (pos_new_proj > 0)) or \
                       ((pos_old_proj > 0) and (pos_new_proj < 0))

        if crossed_plane:
            # Find the intersection point where the path crosses the gate plane
            # Linear interpolation: intersection = last_pos + t * (pos - last_pos)
            # where t is chosen so that the projection onto normal is 0
            t = -pos_old_proj / (pos_new_proj - pos_old_proj)
            intersection = self.last_pos + t * (pos - self.last_pos)

            # Transform intersection to gate's local frame
            # Gate frame: normal is X-axis, perpendicular vectors are Y and Z
            # We need perpendicular vectors in the gate plane
            rel_pos = intersection - gate_pos

            # Create gate's local coordinate system
            # Normal is already defined (forward direction)
            # Right vector (perpendicular to normal in XY plane, or use Z if normal is vertical)
            if abs(normal[2]) < 0.9:  # Not pointing mostly vertical
                up = np.array([0.0, 0.0, 1.0])
            else:
                up = np.array([0.0, 1.0, 0.0])

            right = np.cross(normal, up)
            right = right / np.linalg.norm(right)
            actual_up = np.cross(right, normal)

            # Project onto gate plane (perpendicular to normal)
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


# ============================================================================
# SIMULATION WITH GATES
# ============================================================================

def simulate_with_gates(pos_gain, vel_gain, att_gain, rate_gain,
                       traj_period=6.0, traj_amp_x=3.0, traj_amp_y=2.0, traj_ramp=3.0,
                       gate_config=Figure8Gates, sim_time=20.0, dt=0.005, verbose=False):
    """
    Run simulation with Lee controller and gate checking

    Args:
        pos_gain, vel_gain, att_gain, rate_gain: Controller gains
        traj_period, traj_amp_x, traj_amp_y, traj_ramp: Trajectory parameters
        gate_config: Gate configuration class
        sim_time: Simulation time in seconds
        dt: Time step in seconds
        verbose: If True, show debug output from simulation

    Returns:
        Dictionary with gates_passed, crashed, completed, flight_time
    """
    # Run simulation without output suppression
    try:
        # Create drone (matched quadrotor configuration)
        propellers = [
            {"loc": [0.16, 0.16, 0], "dir": [0, 0, -1, "ccw"], "propsize": "matched"},
            {"loc": [-0.16, 0.16, 0], "dir": [0, 0, -1, "cw"], "propsize": "matched"},
            {"loc": [-0.16, -0.16, 0], "dir": [0, 0, -1, "ccw"], "propsize": "matched"},
            {"loc": [0.16, -0.16, 0], "dir": [0, 0, -1, "cw"], "propsize": "matched"}
        ]
        quad = DroneInterface(0, propellers=propellers)

        # Create Lee controller with specified gains
        lee_gains = {
            'pos_P_gain': np.array([pos_gain] * 3),
            'vel_P_gain': np.array([vel_gain] * 3),
            'att_P_gain': np.array([att_gain] * 3),
            'rate_P_gain': np.array([rate_gain] * 3)
        }
        ctrl = LeeGeometricControl(quad, yawType=1, orient='NED',
                                   auto_scale_gains=False, **lee_gains)

        # Create trajectory with specified parameters
        figure8_params = {
            'period': traj_period,
            'amplitude_x': traj_amp_x,
            'amplitude_y': traj_amp_y,
            'ramp_time': traj_ramp,
            'center_z': 0.0  # Match gate height
        }
        traj = Trajectory(quad, "xyz_pos", np.array([14, 3, 1]),
                         figure8_params=figure8_params)

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

        return {
            'gates_passed': gate_checker.gates_passed,
            'crashed': crashed,
            'completed': not crashed,
            'flight_time': t,
            'success': True
        }

    except Exception as e:
        return {
            'gates_passed': 0,
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
    """
    Wrapper function for parallel evaluation (must be at module level for pickling)

    Args:
        args: Tuple of (params, tune_trajectory, gate_config, sim_time, dt)

    Returns:
        Tuple of (score, result_dict)
    """
    params, tune_trajectory, gate_config, sim_time, dt = args

    if tune_trajectory:
        pos_g, vel_g, att_g, rate_g, period, amp_x, amp_y, ramp = params
    else:
        pos_g, vel_g, att_g, rate_g = params
        period, amp_x, amp_y, ramp = 6.0, 3.0, 2.0, 3.0

    result = simulate_with_gates(
        pos_g, vel_g, att_g, rate_g,
        period, amp_x, amp_y, ramp,
        gate_config, sim_time, dt,
        verbose=False
    )

    if result['success']:
        gates = result['gates_passed']
        penalty = 100 if result['crashed'] else 0
        score = -gates + penalty

        result['score'] = score
        result['fitness'] = gates
        result['gains'] = {
            'pos_P': pos_g,
            'vel_P': vel_g,
            'att_P': att_g,
            'rate_P': rate_g
        }
        if tune_trajectory:
            result['trajectory'] = {
                'period': period,
                'amplitude_x': amp_x,
                'amplitude_y': amp_y,
                'ramp_time': ramp
            }

        return (score, result)
    else:
        return (1000.0, None)


# ============================================================================
# CMA-ES TUNER
# ============================================================================

class GateBasedTuner:
    """CMA-ES optimization for Lee controller with gate-based fitness"""

    def __init__(self, gate_config, sim_time=20.0, dt=0.005, output_dir="tuning_results_gates"):
        self.gate_config = gate_config
        self.sim_time = sim_time
        self.dt = dt
        self.output_dir = output_dir
        self.results = []
        self.best_score = -float('inf')
        self.best_params = None

        # Create output directory
        Path(self.output_dir).mkdir(exist_ok=True)

    def objective_function(self, params, tune_trajectory=True):
        """
        Objective function for CMA-ES

        Args:
            params: [pos_gain, vel_gain, att_gain, rate_gain, period, amp_x, amp_y, ramp]
                    or [pos_gain, vel_gain, att_gain, rate_gain] if not tuning trajectory
            tune_trajectory: Whether trajectory parameters are included

        Returns:
            Negative gates passed (CMA-ES minimizes, we want to maximize gates)
        """
        if tune_trajectory:
            pos_g, vel_g, att_g, rate_g, period, amp_x, amp_y, ramp = params
        else:
            pos_g, vel_g, att_g, rate_g = params
            period, amp_x, amp_y, ramp = 6.0, 3.0, 2.0, 3.0

        result = simulate_with_gates(
            pos_g, vel_g, att_g, rate_g,
            period, amp_x, amp_y, ramp,
            self.gate_config, self.sim_time, self.dt,
            verbose=False
        )

        if result['success']:
            # Fitness = negative gates passed (CMA-ES minimizes)
            # Add penalty for crashes
            gates = result['gates_passed']
            penalty = 100 if result['crashed'] else 0
            score = -gates + penalty

            result['score'] = score
            result['fitness'] = gates
            result['gains'] = {
                'pos_P': pos_g,
                'vel_P': vel_g,
                'att_P': att_g,
                'rate_P': rate_g
            }
            if tune_trajectory:
                result['trajectory'] = {
                    'period': period,
                    'amplitude_x': amp_x,
                    'amplitude_y': amp_y,
                    'ramp_time': ramp
                }

            self.results.append(result)

            # Track best
            if gates > self.best_score and not result['crashed']:
                self.best_score = gates
                self.best_params = result

            return score
        else:
            # Failed simulation - large penalty
            return 1000.0

    def run_cmaes_tuning(self, max_evaluations=200, initial_guess=None,
                        initial_std=None, num_workers=None, tune_trajectory=True,
                        timeout_per_eval=30.0):
        """
        Run CMA-ES optimization with parallel evaluation

        Args:
            max_evaluations: Maximum number of evaluations
            initial_guess: Starting point for optimization
            initial_std: Initial standard deviation
            num_workers: Number of parallel workers
            tune_trajectory: Whether to tune trajectory parameters
            timeout_per_eval: Timeout in seconds for each evaluation (default: 30s)
        """
        if not CMA_AVAILABLE:
            print("ERROR: CMA-ES requires the 'cma' package")
            return

        # Set up parallel workers
        if num_workers is None:
            num_workers = max(1, multiprocessing.cpu_count() // 2)

        print(f"\n{'='*70}")
        print(f"LEE CONTROLLER CMA-ES OPTIMIZATION WITH GATE FITNESS")
        print(f"{'='*70}")
        print(f"Gate configuration: {self.gate_config.__name__}")
        print(f"Number of gates: {len(self.gate_config.gate_pos)}")
        print(f"Max evaluations: {max_evaluations}")
        print(f"Parallel workers: {num_workers}")
        print(f"Tuning trajectory: {tune_trajectory}")
        print(f"Simulation time: {self.sim_time}s")
        print(f"Timeout per eval: {timeout_per_eval}s")
        print(f"{'='*70}\n")

        # Set up initial guess and bounds
        if initial_guess is None:
            initial_guess = [10.0, 9.0, 4.4, -0.3]
            if tune_trajectory:
                initial_guess.extend([6.0, 3.0, 2.0, 3.0])

        if initial_std is None:
            initial_std = [2.0, 2.0, 0.5, 0.1]
            if tune_trajectory:
                initial_std.extend([1.5, 0.8, 0.6, 0.8])

        # Bounds
        bounds = [
            [5.0, 20.0],    # pos_P
            [5.0, 20.0],    # vel_P
            [2.0, 8.0],     # att_P
            [-1.0, 0.1]     # rate_P
        ]

        if tune_trajectory:
            bounds.extend([
                [3.0, 12.0],    # period
                [1.0, 5.0],     # amplitude_x
                [1.0, 4.0],     # amplitude_y
                [1.0, 5.0]      # ramp_time
            ])

        # CMA-ES options
        options = {
            'bounds': [list(b) for b in zip(*bounds)],
            'maxfevals': max_evaluations,
            'verb_disp': 1,
            'verb_log': 0,
            'tolx': 1e-8,  # Relax parameter change tolerance
            'tolfun': 1e-6,  # Relax function value tolerance
            'tolfunhist': 1e-6,  # Relax function history tolerance
        }

        # Run CMA-ES
        print("Starting CMA-ES optimization...\n")
        start_time = time.time()

        try:
            es = cma.CMAEvolutionStrategy(initial_guess, initial_std[0], options)

            iteration = 0
            # Use ProcessPoolExecutor for true parallelism (not ThreadPoolExecutor)
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                while not es.stop():
                    iteration += 1
                    solutions = es.ask()
                    print(f"Iteration {iteration}: Evaluating {len(solutions)} solutions...", flush=True)

                    # Parallel evaluation
                    if num_workers > 1:
                        # Prepare arguments for each solution
                        eval_args = [
                            (sol, tune_trajectory, self.gate_config, self.sim_time, self.dt)
                            for sol in solutions
                        ]

                        # Submit all jobs
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
                                # Wait for result with timeout
                                score, result = future.result(timeout=timeout_per_eval)
                                results_dict[tuple(sol)] = score

                                # Store result if valid
                                if result is not None:
                                    self.results.append(result)

                                    # Track best
                                    if result['fitness'] > self.best_score and not result['crashed']:
                                        self.best_score = result['fitness']
                                        self.best_params = result

                                completed += 1
                                print(f"  Completed {completed}/{len(solutions)}", end='\r', flush=True)
                            except TimeoutError:
                                print(f"\n  Timeout evaluating solution (>{timeout_per_eval}s)", flush=True)
                                results_dict[tuple(sol)] = 1000.0  # Penalty for timeout
                                completed += 1
                                timeouts += 1
                            except Exception as e:
                                print(f"\n  Error evaluating solution: {e}", flush=True)
                                results_dict[tuple(sol)] = 1000.0
                                completed += 1

                        print()  # New line after progress
                        if timeouts > 0:
                            print(f"  Warning: {timeouts} evaluations timed out")
                        fitness_values = [results_dict[tuple(sol)] for sol in solutions]
                    else:
                        # Serial execution
                        fitness_values = [self.objective_function(x, tune_trajectory)
                                        for x in solutions]

                    es.tell(solutions, fitness_values)

                    # Print progress
                    best_idx = np.argmin(fitness_values)
                    best_solution = solutions[best_idx]
                    best_score = fitness_values[best_idx]

                    print(f"\n{'='*70}")
                    print(f"Iteration {iteration} | Evaluated: {es.result.evaluations}/{max_evaluations}")
                    print(f"{'='*70}")
                    print(f"Best in generation: Score={-best_score:.1f} gates")
                    print(f"  Gains: pos={best_solution[0]:.2f}, vel={best_solution[1]:.2f}, "
                          f"att={best_solution[2]:.2f}, rate={best_solution[3]:.3f}")
                    if tune_trajectory and len(best_solution) >= 8:
                        print(f"  Traj: period={best_solution[4]:.2f}s, "
                              f"amp_x={best_solution[5]:.2f}m, amp_y={best_solution[6]:.2f}m, "
                              f"ramp={best_solution[7]:.2f}s")

                    print(f"\nBest ever: {self.best_score:.0f} gates")
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
            print(f"OPTIMIZATION COMPLETE")
            print(f"{'='*70}")
            print(f"Time elapsed: {elapsed/60:.1f} minutes")
            print(f"Total evaluations: {es.result.evaluations}")
            print(f"\nBest result: {self.best_score:.0f} gates passed")

            if self.best_params:
                gains = self.best_params['gains']
                print(f"\nBest gains:")
                print(f"  Position: {gains['pos_P']:.3f}")
                print(f"  Velocity: {gains['vel_P']:.3f}")
                print(f"  Attitude: {gains['att_P']:.3f}")
                print(f"  Rate: {gains['rate_P']:.4f}")

                if 'trajectory' in self.best_params:
                    traj = self.best_params['trajectory']
                    print(f"\nBest trajectory:")
                    print(f"  Period: {traj['period']:.2f}s")
                    print(f"  Amplitude X: {traj['amplitude_x']:.2f}m")
                    print(f"  Amplitude Y: {traj['amplitude_y']:.2f}m")
                    print(f"  Ramp time: {traj['ramp_time']:.2f}s")

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
        filename = f"{self.output_dir}/results_{timestamp}.json"

        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"Results saved to: {filename}")

    def _save_best_config(self):
        """Save best configuration"""
        if self.best_params is None:
            return

        config = {
            'timestamp': datetime.now().isoformat(),
            'gates_passed': int(self.best_score),
            'gains': self.best_params['gains'],
            'flight_time': self.best_params['flight_time'],
            'crashed': self.best_params['crashed']
        }

        if 'trajectory' in self.best_params:
            config['trajectory'] = self.best_params['trajectory']

        config_file = f"{self.output_dir}/best_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"Best config saved to: {config_file}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='CMA-ES Optimization for Lee Controller with Gate Fitness',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--gate-cfg', type=str, default='figure8',
                       choices=['figure8', 'circle', 'slalom', 'backandforth'],
                       help='Gate configuration (default: figure8)')
    parser.add_argument('--max-evals', type=int, default=200,
                       help='Maximum evaluations (default: 200)')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers (default: CPU_count/2)')
    parser.add_argument('--time', type=float, default=20.0,
                       help='Simulation time in seconds (default: 20.0)')
    parser.add_argument('--dt', type=float, default=0.005,
                       help='Time step in seconds (default: 0.005)')
    parser.add_argument('--output', type=str, default='tuning_results_gates',
                       help='Output directory (default: tuning_results_gates)')
    parser.add_argument('--no-tune-trajectory', action='store_true',
                       help='Only tune gains, not trajectory parameters')
    parser.add_argument('--timeout', type=float, default=30.0,
                       help='Timeout per evaluation in seconds (default: 30.0)')

    args = parser.parse_args()

    # Get gate configuration
    gate_config = GATE_CONFIGS[args.gate_cfg]

    # Create tuner
    tuner = GateBasedTuner(gate_config, sim_time=args.time, dt=args.dt,
                          output_dir=args.output)

    # Run optimization
    tuner.run_cmaes_tuning(max_evaluations=args.max_evals,
                          num_workers=args.workers,
                          tune_trajectory=not args.no_tune_trajectory,
                          timeout_per_eval=args.timeout)


if __name__ == '__main__':
    main()
