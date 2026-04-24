#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Curriculum-Based CMA-ES Optimization for Lee Controller with B-Spline Trajectories
MATCHED QUAD VERSION - For verification testing

This script uses the SAME drone configuration as Quadcopter_SimCon to verify
that the controller works when properly configured.

Usage:
    python tune_lee_controller_gates_matched.py --stage 1 --gates figure8 --max-evals 50
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
# QUAD CONFIGURATION - MATCHED (SAME AS QUADCOPTER_SIMCON)
# ============================================================================

# Matched quadrotor configuration (~1.5kg total mass)
ARM_LENGTH = 0.16  # 160mm arms
PROP_SIZE = "matched"  # Matched propellers

def create_matched_quad():
    """Create matched quadrotor configuration (same as Quadcopter_SimCon)."""
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

    def __init__(self, gate_pos, gate_yaw, gate_size=1.0):
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
                     gate_config, sim_time=20.0, dt=0.005, verbose=False):
    """
    Run simulation with Lee controller and B-spline gate trajectory

    Args:
        pos_gain, vel_gain, att_gain, rate_gain: Controller gains
        bspline_params: Array of B-spline trajectory parameters
        gate_config: Gate configuration class
        sim_time: Simulation time in seconds
        dt: Time step in seconds
        verbose: If True, show debug output

    Returns:
        Dictionary with gates_passed, crashed, completed, flight_time
    """
    try:
        # Create MATCHED quad (same as Quadcopter_SimCon)
        quad = create_matched_quad()

        # Create Lee controller with specified gains
        lee_gains = {
            'pos_P_gain': np.array([pos_gain] * 3),
            'vel_P_gain': np.array([vel_gain] * 3),
            'att_P_gain': np.array([att_gain] * 3),
            'rate_P_gain': np.array([rate_gain] * 3)
        }
        ctrl = LeeGeometricControl(quad, yawType=1, orient='NED', **lee_gains)

        # Create B-spline trajectory
        bspline_traj = BSplineGateTrajectory(gate_config, gate_offset_scale=0.5)
        bspline_traj.set_parameters(bspline_params)

        # Set drone initial position
        start_pos = bspline_traj.get_start_position()
        quad.drone_sim.set_state(position=start_pos)

        # Create Trajectory wrapper (xyzType=15 for B-spline)
        from airevolve.controllers.trajectory_generation.trajectory import Trajectory
        traj = Trajectory(quad, "xyz_pos", np.array([15, 3, 1]),
                         gate_config=gate_config)
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
# SIMPLE TUNER (STAGE 1 ONLY FOR VERIFICATION)
# ============================================================================

class SimpleTuner:
    """Simple CMA-ES optimization for gain tuning (Stage 1 only)"""

    def __init__(self, gate_config, sim_time=20.0, dt=0.005,
                 output_dir="tuning_results_gates_matched"):
        """
        Initialize simple tuner

        Args:
            gate_config: Gate configuration class
            sim_time: Simulation time in seconds
            dt: Time step in seconds
            output_dir: Output directory for results
        """
        self.gate_config = gate_config
        self.sim_time = sim_time
        self.dt = dt
        self.output_dir = output_dir
        self.results = []
        self.best_score = -float('inf')
        self.best_params = None

        # Create B-spline template
        self.bspline_template = BSplineGateTrajectory(gate_config, gate_offset_scale=0.5)
        self.fixed_bspline_params = self.bspline_template.get_default_parameters()

        # Create output directory
        Path(self.output_dir).mkdir(exist_ok=True)

    def run_optimization(self, max_evaluations=200, num_workers=None):
        """Run CMA-ES optimization for Stage 1 (gains only)"""
        if not CMA_AVAILABLE:
            print("ERROR: CMA-ES requires the 'cma' package")
            return

        # Set up parallel workers
        if num_workers is None:
            num_workers = max(1, multiprocessing.cpu_count() // 2)

        # Initial guess and bounds (SAME AS QUADCOPTER_SIMCON)
        initial_guess = [10.0, 9.0, 4.4, -0.3]  # pos, vel, att, rate
        bounds = [
            [5.0, 20.0],    # pos_P
            [5.0, 20.0],    # vel_P
            [2.0, 8.0],     # att_P
            [-1.0, 0.1]     # rate_P
        ]

        print(f"\n{'='*70}")
        print(f"LEE CONTROLLER CMA-ES OPTIMIZATION - MATCHED QUAD")
        print(f"{'='*70}")
        print(f"Gate configuration: {self.gate_config.__name__}")
        print(f"Number of gates: {len(self.gate_config.gate_pos)}")
        print(f"Drone: MATCHED quad (arm_length={ARM_LENGTH}m)")
        print(f"Optimization: Controller gains (4 params)")
        print(f"Max evaluations: {max_evaluations}")
        print(f"Parallel workers: {num_workers}")
        print(f"Initial guess: {initial_guess}")
        print(f"{'='*70}\n")

        # CMA-ES options
        options = {
            'bounds': [list(b) for b in zip(*bounds)],
            'maxfevals': max_evaluations,
            'verb_disp': 1,
            'verb_log': 0,
        }

        # Run CMA-ES
        print("Starting CMA-ES optimization...\n")
        start_time = time.time()

        try:
            es = cma.CMAEvolutionStrategy(initial_guess, 2.0, options)

            iteration = 0
            while not es.stop():
                iteration += 1
                solutions = es.ask()
                print(f"Iteration {iteration}: Evaluating {len(solutions)} solutions...", flush=True)

                # Evaluate solutions
                fitness_values = []
                for sol in solutions:
                    pos_g, vel_g, att_g, rate_g = sol

                    result = simulate_bspline(
                        pos_g, vel_g, att_g, rate_g,
                        self.fixed_bspline_params,
                        self.gate_config, self.sim_time, self.dt,
                        verbose=False
                    )

                    if result['success']:
                        gates = result['gates_passed']
                        distance_bonus = result['distance_bonus']
                        penalty = 100 if result['crashed'] else 0

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
                        result['bspline_params'] = self.fixed_bspline_params.tolist()

                        self.results.append(result)

                        if fitness > self.best_score and not result['crashed']:
                            self.best_score = fitness
                            self.best_params = result

                        fitness_values.append(score)
                    else:
                        fitness_values.append(1000.0)

                es.tell(solutions, fitness_values)

                # Print progress
                best_idx = np.argmin(fitness_values)
                best_solution = solutions[best_idx]

                print(f"\n{'='*70}")
                print(f"Iteration {iteration} | Evaluated: {es.result.evaluations}/{max_evaluations}")
                print(f"{'='*70}")
                print(f"Best in generation: Fitness={-fitness_values[best_idx]:.2f}")
                print(f"  Gains: pos={best_solution[0]:.2f}, vel={best_solution[1]:.2f}, "
                      f"att={best_solution[2]:.2f}, rate={best_solution[3]:.3f}")

                print(f"\nBest ever: {self.best_score:.2f} fitness")
                if self.best_params:
                    gains = self.best_params['gains']
                    print(f"  Gains: pos={gains['pos_P']:.2f}, vel={gains['vel_P']:.2f}, "
                          f"att={gains['att_P']:.2f}, rate={gains['rate_P']:.3f}")
                print(f"{'='*70}")

            end_time = time.time()
            elapsed = end_time - start_time

            # Print final results
            print(f"\n{'='*70}")
            print(f"OPTIMIZATION COMPLETE")
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

            print(f"{'='*70}\n")

            # Save results
            self._save_results()

        except KeyboardInterrupt:
            print("\n\nOptimization interrupted. Saving results...")
            self._save_results()
        except Exception as e:
            print(f"\n\nERROR: {e}")
            import traceback
            traceback.print_exc()

    def _save_results(self):
        """Save all results to JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.output_dir}/results_{timestamp}.json"

        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"Results saved to: {filename}")

        # Save best config
        if self.best_params:
            config = {
                'timestamp': datetime.now().isoformat(),
                'fitness': self.best_score,
                'gates_passed': self.best_params.get('gates_passed', 0),
                'distance_bonus': self.best_params.get('distance_bonus', 0.0),
                'gains': self.best_params['gains'],
                'bspline_params': self.best_params['bspline_params'],
                'flight_time': self.best_params['flight_time'],
                'crashed': self.best_params['crashed']
            }

            config_file = f"{self.output_dir}/best.json"
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)

            print(f"Best config saved to: {config_file}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Lee Controller Optimization - MATCHED QUAD (Verification)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--gates', type=str, required=True,
                       choices=['figure8', 'circle', 'slalom', 'backandforth'],
                       help='Gate configuration')
    parser.add_argument('--max-evals', type=int, default=50,
                       help='Maximum evaluations (default: 50)')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers (default: CPU_count/2)')
    parser.add_argument('--time', type=float, default=20.0,
                       help='Simulation time in seconds (default: 20.0)')
    parser.add_argument('--dt', type=float, default=0.005,
                       help='Time step in seconds (default: 0.005)')
    parser.add_argument('--output', type=str, default='tuning_results_gates_matched',
                       help='Output directory (default: tuning_results_gates_matched)')

    args = parser.parse_args()

    # Get gate configuration
    gate_config = GATE_CONFIGS[args.gates]

    # Create tuner
    tuner = SimpleTuner(
        gate_config,
        sim_time=args.time,
        dt=args.dt,
        output_dir=args.output
    )

    # Run optimization
    tuner.run_optimization(
        max_evaluations=args.max_evals,
        num_workers=args.workers
    )


if __name__ == '__main__':
    main()
