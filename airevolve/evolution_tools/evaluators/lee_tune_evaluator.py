"""
Lee Controller Tuning Evaluator for Evolutionary Algorithm

This evaluator uses CMA-ES to optimize controller gains for each evolved morphology.
Each individual's fitness is determined by the maximum number of gates passed during
controller tuning (Stage 1 optimization: gains only).

If controller tuning fails to pass any gates, the individual receives fitness of 0.

Design:
- Takes an evolved drone morphology
- Converts it to DroneInterface
- Runs Stage 1 CMA-ES optimization (4 parameters: pos_P, vel_P, att_P, rate_P)
- Returns max gates passed as fitness

This integrates the tune_lee_controller_gates.py pipeline into the evolutionary loop.
"""

import numpy as np
import os
import sys
import time
import json
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
import multiprocessing

# Import drone simulation components
from airevolve.simulator.simulation import DroneInterface
from airevolve.controllers.trajectory_generation.bspline_gate_trajectory import BSplineGateTrajectory
from airevolve.controllers.lee_control.lee_controller import LeeGeometricControl
from airevolve.controllers.utils.wind_model import Wind
from airevolve.controllers.utils.gate_configs import GATE_CONFIGS

# Try to import CMA-ES
try:
    import cma
    CMA_AVAILABLE = True
except ImportError:
    CMA_AVAILABLE = False
    print("WARNING: 'cma' package not found. Install with: pip install cma")


# ============================================================================
# GATE CHECKING LOGIC (from tune_lee_controller_gates.py)
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
# SIMULATION (adapted from tune_lee_controller_gates.py)
# ============================================================================

def simulate_with_gains(individual, pos_gain, vel_gain, att_gain, rate_gain,
                        gate_config, sim_time=20.0, dt=0.005, n_startup_points=1,
                        gate_only_mode=False, verbose=False):
    """
    Run simulation with Lee controller for a given morphology and gains

    Args:
        individual: Evolved drone morphology (genome array)
        pos_gain, vel_gain, att_gain, rate_gain: Controller gains
        gate_config: Gate configuration class
        sim_time: Simulation time in seconds
        dt: Time step in seconds
        n_startup_points: Number of startup control points
        gate_only_mode: If True, use gate-only mode (pure racing loop)
        verbose: If True, show debug output

    Returns:
        Dictionary with gates_passed, crashed, completed, flight_time
    """
    try:
        # Convert genome to DroneInterface
        # Individual format: N x 6 array [r, theta, phi, pitch, yaw, direction]
        propellers = []
        for arm in individual:
            r, theta, phi, motor_pitch, motor_yaw, direction = arm

            # Convert spherical to Cartesian
            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = r * np.cos(phi)

            # Motor orientation
            rot = "ccw" if direction < 0.5 else "cw"

            # Estimate prop size from arm length (rough heuristic)
            prop_size = max(1, min(5, int(r * 20)))  # Scale to 1-5 inch props

            propellers.append({
                "loc": [x, y, z],
                "dir": [motor_pitch, motor_yaw, 0, rot],
                "propsize": prop_size
            })

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

        # Create B-spline trajectory
        bspline_traj = BSplineGateTrajectory(gate_config, n_startup_points=n_startup_points,
                                            gate_offset_scale=0.5, gate_only_mode=gate_only_mode)
        bspline_params = bspline_traj.get_default_parameters()
        bspline_traj.set_parameters(bspline_params)

        # Set drone initial position
        start_pos = bspline_traj.get_start_position()
        quad.drone_sim.set_state(position=start_pos)

        # Create Trajectory wrapper (xyzType=15 for B-spline)
        from airevolve.controllers.trajectory_generation.trajectory import Trajectory
        traj = Trajectory(quad, "xyz_pos", np.array([15, 3, 1]),
                         gate_config=gate_config,
                         bspline_params={'n_startup_points': n_startup_points,
                                       'gate_only_mode': gate_only_mode})
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
    (params, individual, gate_config, sim_time, dt, n_startup_points, gate_only_mode) = args

    pos_g, vel_g, att_g, rate_g = params[0:4]

    # Run simulation
    result = simulate_with_gains(
        individual, pos_g, vel_g, att_g, rate_g,
        gate_config, sim_time, dt,
        n_startup_points=n_startup_points,
        gate_only_mode=gate_only_mode,
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

        return (score, result)
    else:
        return (1000.0, None)


# ============================================================================
# STAGE 1 TUNER FOR SINGLE MORPHOLOGY
# ============================================================================

def optimize_controller_for_morphology(individual, gate_config, max_evaluations=100,
                                      num_workers=None, sim_time=20.0, dt=0.005,
                                      n_startup_points=1, gate_only_mode=False,
                                      timeout_per_eval=30.0, save_dir=None):
    """
    Run Stage 1 CMA-ES optimization to tune controller gains for a morphology.

    Args:
        individual: Evolved drone morphology (genome array)
        gate_config: Gate configuration class
        max_evaluations: Maximum CMA-ES evaluations
        num_workers: Number of parallel workers
        sim_time: Simulation time in seconds
        dt: Time step in seconds
        n_startup_points: Number of startup control points
        gate_only_mode: If True, use gate-only mode (pure racing loop)
        timeout_per_eval: Timeout per evaluation in seconds
        save_dir: Directory to save results (optional)

    Returns:
        dict with:
            - fitness: max gates passed (0 if failed)
            - best_gains: dict of best gains found
            - gates_passed: number of gates passed
            - success: True if optimization succeeded
    """
    if not CMA_AVAILABLE:
        print("ERROR: CMA-ES requires the 'cma' package")
        return {'fitness': 0, 'best_gains': None, 'gates_passed': 0, 'success': False}

    # Set up parallel workers
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() // 2)

    # Stage 1: Optimize gains only (4 params)
    # Bounds for gains (same as tune_lee_controller_gates.py)
    initial_guess = [2.0, 1.5, 0.6, -0.3]  # pos_P, vel_P, att_P, rate_P
    bounds = [
        [0.01, 10.0],     # pos_P
        [0.01, 10.0],     # vel_P
        [0.01, 10.0],     # att_P
        [-5.0, -0.01]     # rate_P
    ]
    initial_std = 0.8

    # CMA-ES options
    options = {
        'bounds': [list(b) for b in zip(*bounds)],
        'maxfevals': max_evaluations,
        'verb_disp': 0,  # Silent for evolution
        'verb_log': 0,
        'tolx': 1e-8,
        'tolfun': 1e-6,
        'tolfunhist': 1e-6,
    }

    # Track best result
    best_score = -float('inf')
    best_result = None
    all_results = []

    try:
        es = cma.CMAEvolutionStrategy(initial_guess, initial_std, options)

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            while not es.stop():
                solutions = es.ask()

                # Parallel evaluation
                if num_workers > 1:
                    eval_args = [
                        (sol, individual, gate_config, sim_time, dt, n_startup_points, gate_only_mode)
                        for sol in solutions
                    ]

                    future_to_sol = {
                        executor.submit(_evaluate_solution_wrapper, args): args[0]
                        for args in eval_args
                    }

                    results_dict = {}
                    for future in as_completed(future_to_sol):
                        sol = future_to_sol[future]
                        try:
                            score, result = future.result(timeout=timeout_per_eval)
                            results_dict[tuple(sol)] = score

                            if result is not None:
                                all_results.append(result)

                                if result['fitness'] > best_score and not result['crashed']:
                                    best_score = result['fitness']
                                    best_result = result

                        except TimeoutError:
                            results_dict[tuple(sol)] = 1000.0
                        except Exception as e:
                            results_dict[tuple(sol)] = 1000.0

                    fitness_values = [results_dict[tuple(sol)] for sol in solutions]
                else:
                    # Serial execution
                    fitness_values = []
                    for sol in solutions:
                        score, result = _evaluate_solution_wrapper(
                            (sol, individual, gate_config, sim_time, dt, n_startup_points, gate_only_mode)
                        )
                        fitness_values.append(score)

                        if result is not None:
                            all_results.append(result)
                            if result['fitness'] > best_score and not result['crashed']:
                                best_score = result['fitness']
                                best_result = result

                es.tell(solutions, fitness_values)

    except Exception as e:
        print(f"CMA-ES optimization error: {e}")
        return {'fitness': 0, 'best_gains': None, 'gates_passed': 0, 'success': False}

    # Save results if save_dir provided
    if save_dir is not None and best_result is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        config = {
            'timestamp': datetime.now().isoformat(),
            'fitness': best_score,
            'gates_passed': best_result.get('gates_passed', 0),
            'distance_bonus': best_result.get('distance_bonus', 0.0),
            'gains': best_result['gains'],
            'flight_time': best_result['flight_time'],
            'crashed': best_result['crashed']
        }

        config_file = os.path.join(save_dir, "tuning_results.json")
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

    # Return results
    if best_result is not None:
        return {
            'fitness': best_score,
            'best_gains': best_result['gains'],
            'gates_passed': best_result['gates_passed'],
            'distance_bonus': best_result.get('distance_bonus', 0.0),
            'success': True
        }
    else:
        return {
            'fitness': 0,
            'best_gains': None,
            'gates_passed': 0,
            'distance_bonus': 0.0,
            'success': False
        }


# ============================================================================
# EVALUATOR FOR EVOLUTIONARY ALGORITHM
# ============================================================================

def evaluate_individual_with_tuning(individual, ind_save_dir, gate_cfg='circle',
                                    max_evals=100, num_workers=4, sim_time=20.0,
                                    dt=0.005, n_startup_points=1, gate_only_mode=False,
                                    timeout=30.0, num=None):
    """
    Evaluate individual by tuning its controller gains via CMA-ES Stage 1.

    This is the main fitness function for evolution with Lee controller tuning.

    Args:
        individual: Evolved drone morphology (genome array)
        ind_save_dir: Directory to save results for this individual
        gate_cfg: Gate configuration ('circle', 'figure8', 'slalom', 'backandforth')
        max_evals: Maximum CMA-ES evaluations
        num_workers: Number of parallel workers for CMA-ES
        sim_time: Simulation time in seconds
        dt: Time step in seconds
        n_startup_points: Number of startup control points
        gate_only_mode: If True, use gate-only mode (pure racing loop)
        timeout: Timeout per evaluation in seconds
        num: Individual number (for logging)

    Returns:
        int: Number of gates passed (0 if tuning failed)
    """
    # Get gate configuration
    gate_config = GATE_CONFIGS[gate_cfg]

    # Run Stage 1 optimization
    result = optimize_controller_for_morphology(
        individual=individual,
        gate_config=gate_config,
        max_evaluations=max_evals,
        num_workers=num_workers,
        sim_time=sim_time,
        dt=dt,
        n_startup_points=n_startup_points,
        gate_only_mode=gate_only_mode,
        timeout_per_eval=timeout,
        save_dir=ind_save_dir
    )

    # Return integer fitness (gates passed)
    return int(result['gates_passed'])
