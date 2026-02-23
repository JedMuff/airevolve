# -*- coding: utf-8 -*-
"""
B-Spline Gate Trajectory Generator

This module provides automatic trajectory generation for gate-based racing
using B-spline curves. Control points are initialized at gate positions and
can be optimized to find the fastest path through gates.

Author: Generated for AirEvolve project
License: MIT
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
from .bspline_utils import BSplineCurve, generate_knot_vector
from ..utils.gate_configs import GateConfig


class BSplineGateTrajectory:
    """
    B-spline trajectory generator optimized for gate passing.

    This class creates a smooth trajectory that passes through or near gates
    in a racing configuration. The trajectory consists of:
    - Startup control points: Independent points for smooth takeoff
    - Gate control points: Points near gates (optimizable within gate bounds)
    - Closure: Periodic B-spline to create a continuous racing loop

    Parameters can be optimized using CMA-ES or other optimization algorithms.
    """

    def __init__(self, gate_config: GateConfig, n_startup_points: int = 1,
                 degree: int = 3, gate_offset_scale: float = 1.0):
        """
        Initialize B-spline gate trajectory.

        The trajectory consists of:
        1. Run-up phase: starting position → startup points → gate 0 (open/non-looping)
        2. Racing loop: gate 0 → gate 1 → ... → gate N-1 → back to gate 0 (periodic loop)

        Args:
            gate_config: Gate configuration with positions and yaws
            n_startup_points: Number of intermediate control points between start and gate 0 (default: 1, minimum 1 for cubic splines)
            degree: B-spline degree (default: 3 for cubic)
            gate_offset_scale: Scale factor for gate offset bounds (default: 1.0)
                              Offsets are bounded by ±gate_size * gate_offset_scale
        """
        self.gate_config = gate_config
        self.n_startup_points = n_startup_points
        self.degree = degree
        self.gate_offset_scale = gate_offset_scale

        # Extract gate information
        self.gate_positions = np.array(gate_config.gate_pos, dtype=np.float64)
        self.gate_yaws = np.array(gate_config.gate_yaw, dtype=np.float64)
        self.gate_size = gate_config.gate_size
        self.n_gates = len(self.gate_positions)

        # Control point structure:
        # Startup spline (open): starting_point, startup_points, gate[0]
        # Loop spline (periodic): gate control points with offsets
        self.n_startup_control_points = 1 + n_startup_points + 1  # start + intermediates + gate[0]
        self.n_loop_control_points = self.n_gates

        # Default parameters (will be overridden by set_parameters)
        self.starting_point = None
        self.startup_points = None
        self.gate_offsets = None
        self.total_time = 20.0
        self.velocity_scale = 1.0
        self.startup_time = 3.0  # Fixed time for startup phase (seconds)

        # Initialize with default values
        self._initialize_default_parameters()

        # B-spline curves (created when parameters are set)
        self.startup_spline = None  # Open spline for run-up
        self.loop_spline = None     # Periodic spline for racing loop
        self._rebuild_splines()

    def _initialize_default_parameters(self):
        """Initialize control points with default values."""
        # Get starting position
        if hasattr(self.gate_config, 'starting_pos') and self.gate_config.starting_pos is not None:
            self.starting_point = np.array(self.gate_config.starting_pos, dtype=np.float64)
        else:
            # Fallback: calculate start position behind gate 0 by 2 meters
            first_gate = self.gate_positions[0]
            first_gate_yaw = self.gate_yaws[0]
            distance_behind = 2.0  # meters
            self.starting_point = first_gate - distance_behind * np.array([
                np.cos(first_gate_yaw),
                np.sin(first_gate_yaw),
                0.0
            ])

        first_gate = self.gate_positions[0]

        # Startup points: intermediate points between start and gate 0
        if self.n_startup_points > 0:
            self.startup_points = np.zeros((self.n_startup_points, 3))
            for i in range(self.n_startup_points):
                alpha = (i + 1) / (self.n_startup_points + 1)
                self.startup_points[i] = (1 - alpha) * self.starting_point + alpha * first_gate
        else:
            self.startup_points = np.zeros((0, 3))

        # Gate offsets: initialize at zero (control points at gate positions)
        self.gate_offsets = np.zeros((self.n_gates, 3))

    def get_startup_control_points(self) -> np.ndarray:
        """
        Get control points for the startup (run-up) spline.

        Returns:
            Array of control points for startup: [starting_point, startup_points..., gate[0]]
        """
        first_gate_with_offset = self.gate_positions[0] + self.gate_offsets[0]

        control_points_list = [self.starting_point.reshape(1, 3)]

        if self.n_startup_points > 0:
            control_points_list.append(self.startup_points)

        control_points_list.append(first_gate_with_offset.reshape(1, 3))

        return np.vstack(control_points_list)

    def get_loop_control_points(self) -> np.ndarray:
        """
        Get control points for the racing loop spline.

        Returns:
            Array of gate control points with offsets
        """
        return self.gate_positions + self.gate_offsets

    def _rebuild_splines(self):
        """Rebuild both startup and loop B-spline curves."""
        # Startup spline: clamped (passes through start and gate[0])
        startup_cps = self.get_startup_control_points()

        # For startup spline, use lower degree if we don't have enough control points
        # Degree 3 (cubic) needs at least 4 points, degree 2 (quadratic) needs at least 3
        n_startup_cps = len(startup_cps)
        startup_degree = min(self.degree, n_startup_cps - 1)
        if startup_degree < 1:
            raise ValueError(f"Need at least 2 control points for startup spline, got {n_startup_cps}")

        self.startup_spline = BSplineCurve(startup_cps, degree=startup_degree, boundary='clamped')

        # Loop spline: periodic (closes on itself through gates)
        loop_cps = self.get_loop_control_points()
        self.loop_spline = BSplineCurve(loop_cps, degree=self.degree, boundary='periodic')

    def set_parameters(self, params: np.ndarray):
        """
        Set trajectory parameters from optimization vector.

        Parameter vector structure:
        [
            # Starting position (3)
            start_x, start_y, start_z,

            # Startup intermediate control points (n_startup × 3)
            sx0, sy0, sz0, sx1, sy1, sz1, ...

            # Gate position offsets (n_gates × 3)
            g0_dx, g0_dy, g0_dz, g1_dx, g1_dy, g1_dz, ...

            # Timing/velocity parameters (3)
            total_time, velocity_scale, startup_time
        ]

        Args:
            params: Parameter vector
        """
        expected_length = 3 + self.n_startup_points * 3 + self.n_gates * 3 + 3
        if len(params) != expected_length:
            raise ValueError(f"Expected {expected_length} parameters, got {len(params)}")

        idx = 0

        # Extract starting position
        self.starting_point = params[idx:idx + 3]
        idx += 3

        # Extract startup intermediate points
        if self.n_startup_points > 0:
            startup_flat = params[idx:idx + self.n_startup_points * 3]
            self.startup_points = startup_flat.reshape((self.n_startup_points, 3))
            idx += self.n_startup_points * 3

        # Extract gate offsets
        gate_offsets_flat = params[idx:idx + self.n_gates * 3]
        self.gate_offsets = gate_offsets_flat.reshape((self.n_gates, 3))
        idx += self.n_gates * 3

        # Extract timing parameters
        self.total_time = params[idx]
        self.velocity_scale = params[idx + 1]
        self.startup_time = params[idx + 2]

        # Rebuild splines with new parameters
        self._rebuild_splines()

    def get_default_parameters(self) -> np.ndarray:
        """
        Get default parameter vector for initialization.

        Returns:
            Default parameter vector
        """
        params = []

        # Starting position
        params.extend(self.starting_point)

        # Startup intermediate points
        if self.n_startup_points > 0:
            params.extend(self.startup_points.flatten())

        # Gate offsets (zeros)
        params.extend(self.gate_offsets.flatten())

        # Timing parameters
        params.extend([self.total_time, self.velocity_scale, self.startup_time])

        return np.array(params)

    def get_parameter_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get bounds for optimization parameters.

        Returns:
            Tuple of (lower_bounds, upper_bounds)
        """
        lower = []
        upper = []

        x_bounds = self.gate_config.x_bounds
        y_bounds = self.gate_config.y_bounds
        z_bounds = self.gate_config.z_bounds

        # Bounds for starting position (allow small movement around default start)
        start_offset = 0.5  # meters
        lower.extend([
            max(x_bounds[0], self.starting_point[0] - start_offset),
            max(y_bounds[0], self.starting_point[1] - start_offset),
            max(z_bounds[0], self.starting_point[2] - start_offset)
        ])
        upper.extend([
            min(x_bounds[1], self.starting_point[0] + start_offset),
            min(y_bounds[1], self.starting_point[1] + start_offset),
            min(z_bounds[1], self.starting_point[2] + start_offset)
        ])

        # Bounds for startup intermediate points (allow movement within workspace)
        for _ in range(self.n_startup_points):
            lower.extend([x_bounds[0], y_bounds[0], z_bounds[0]])
            upper.extend([x_bounds[1], y_bounds[1], z_bounds[1]])

        # Bounds for gate offsets (limited by gate size)
        max_offset = self.gate_size * self.gate_offset_scale
        for _ in range(self.n_gates):
            lower.extend([-max_offset, -max_offset, -max_offset])
            upper.extend([max_offset, max_offset, max_offset])

        # Bounds for timing parameters
        lower.extend([5.0, 0.3, 1.0])     # total_time, velocity_scale, startup_time
        upper.extend([30.0, 2.0, 10.0])   # total_time, velocity_scale, startup_time

        return np.array(lower), np.array(upper)

    def evaluate(self, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate trajectory at time t.

        For t < startup_time: use startup spline (run-up phase)
        For t >= startup_time: use loop spline (periodic racing loop)

        Args:
            t: Time in seconds

        Returns:
            Tuple of (position, velocity, acceleration)
            Each is a numpy array of shape (3,)
        """
        if self.startup_spline is None or self.loop_spline is None:
            raise RuntimeError("Splines not initialized. Call set_parameters first.")

        if t < self.startup_time:
            # Startup phase: use startup spline
            spline = self.startup_spline
            u, du_dt, d2u_dt2 = self._time_to_parameter_startup(t)
        else:
            # Racing loop phase: use loop spline
            spline = self.loop_spline
            u, du_dt, d2u_dt2 = self._time_to_parameter_loop(t - self.startup_time)

        # Evaluate spline
        position = spline.position(u)
        velocity = spline.velocity(u, du_dt)
        acceleration = spline.acceleration(u, du_dt, d2u_dt2)

        return position, velocity, acceleration

    def _time_to_parameter_startup(self, t: float) -> Tuple[float, float, float]:
        """
        Map time to B-spline parameter for startup phase.

        Args:
            t: Time in seconds (0 to startup_time)

        Returns:
            Tuple of (u, du/dt, d²u/dt²)
        """
        u_min = self.startup_spline.u_min
        u_max = self.startup_spline.u_max
        u_range = u_max - u_min

        # Linear mapping over startup time
        t_normalized = np.clip(t / self.startup_time, 0.0, 1.0)
        u = u_min + u_range * t_normalized
        du_dt = (u_range / self.startup_time) * self.velocity_scale
        d2u_dt2 = 0.0

        return u, du_dt, d2u_dt2

    def _time_to_parameter_loop(self, t_loop: float) -> Tuple[float, float, float]:
        """
        Map time to B-spline parameter for racing loop phase.

        Args:
            t_loop: Time since entering loop (t - startup_time)

        Returns:
            Tuple of (u, du/dt, d²u/dt²)
        """
        u_min = self.loop_spline.u_min
        u_max = self.loop_spline.u_max
        u_range = u_max - u_min

        # Periodic mapping over remaining time
        loop_time = self.total_time - self.startup_time
        t_normalized = (t_loop % loop_time) / loop_time
        u = u_min + u_range * t_normalized
        du_dt = (u_range / loop_time) * self.velocity_scale
        d2u_dt2 = 0.0

        # Clamp to valid range
        u = np.clip(u, u_min, u_max - 1e-10)

        return u, du_dt, d2u_dt2

    def sample_trajectory(self, dt: float = 0.01) -> Dict[str, np.ndarray]:
        """
        Sample the entire trajectory at regular time intervals.

        Args:
            dt: Time step in seconds

        Returns:
            Dictionary with keys:
            - 'time': Array of time values
            - 'position': Array of positions (N, 3)
            - 'velocity': Array of velocities (N, 3)
            - 'acceleration': Array of accelerations (N, 3)
        """
        n_samples = int(self.total_time / dt) + 1
        times = np.linspace(0, self.total_time, n_samples)

        positions = []
        velocities = []
        accelerations = []

        for t in times:
            pos, vel, acc = self.evaluate(t)
            positions.append(pos)
            velocities.append(vel)
            accelerations.append(acc)

        return {
            'time': times,
            'position': np.array(positions),
            'velocity': np.array(velocities),
            'acceleration': np.array(accelerations)
        }

    def check_gate_proximity(self, dt: float = 0.01) -> np.ndarray:
        """
        Check minimum distance from trajectory to each gate.

        Args:
            dt: Time step for sampling trajectory

        Returns:
            Array of minimum distances to each gate
        """
        trajectory = self.sample_trajectory(dt)
        positions = trajectory['position']

        min_distances = np.full(self.n_gates, np.inf)

        for gate_idx in range(self.n_gates):
            gate_pos = self.gate_positions[gate_idx]
            distances = np.linalg.norm(positions - gate_pos, axis=1)
            min_distances[gate_idx] = np.min(distances)

        return min_distances

    def get_parameter_count(self) -> int:
        """Get total number of parameters."""
        return 3 + self.n_startup_points * 3 + self.n_gates * 3 + 3

    def get_gate_offset_count(self) -> int:
        """Get number of gate offset parameters."""
        return self.n_gates * 3

    def get_timing_parameter_count(self) -> int:
        """Get number of timing parameters."""
        return 3

    def get_gate_offset_parameters(self) -> np.ndarray:
        """
        Get gate offset parameters as a flat array.

        Returns:
            Flattened array of gate offsets [g0_dx, g0_dy, g0_dz, g1_dx, ...]
        """
        return self.gate_offsets.flatten()

    def get_timing_parameters(self) -> np.ndarray:
        """
        Get timing/velocity parameters.

        Returns:
            Array of [total_time, velocity_scale, startup_time]
        """
        return np.array([self.total_time, self.velocity_scale, self.startup_time])

    def set_gate_offset_parameters(self, params: np.ndarray):
        """
        Set gate offset parameters and rebuild splines.

        Args:
            params: Flat array of gate offsets (n_gates * 3 elements)
        """
        expected_length = self.n_gates * 3
        if len(params) != expected_length:
            raise ValueError(f"Expected {expected_length} gate offset parameters, got {len(params)}")

        self.gate_offsets = params.reshape((self.n_gates, 3))
        self._rebuild_splines()

    def set_timing_parameters(self, params: np.ndarray):
        """
        Set timing/velocity parameters (no spline rebuild needed).

        Args:
            params: Array of [total_time, velocity_scale, startup_time]
        """
        if len(params) != 3:
            raise ValueError(f"Expected 3 timing parameters, got {len(params)}")

        self.total_time = params[0]
        self.velocity_scale = params[1]
        self.startup_time = params[2]

    def get_parameter_bounds_by_group(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Get parameter bounds organized by group.

        Returns:
            Dictionary with keys 'gate_offsets' and 'timing', each containing
            (lower_bounds, upper_bounds) tuples
        """
        # Gate offset bounds
        max_offset = self.gate_size * self.gate_offset_scale
        gate_lower = np.full(self.n_gates * 3, -max_offset)
        gate_upper = np.full(self.n_gates * 3, max_offset)

        # Timing parameter bounds
        timing_lower = np.array([5.0, 0.3, 1.0])     # total_time, velocity_scale, startup_time
        timing_upper = np.array([30.0, 2.0, 10.0])   # total_time, velocity_scale, startup_time

        return {
            'gate_offsets': (gate_lower, gate_upper),
            'timing': (timing_lower, timing_upper)
        }

    def get_start_position(self) -> np.ndarray:
        """
        Get the starting position for the trajectory.

        Uses the starting_pos from gate configuration if available,
        otherwise calculates position behind gate 0.

        Returns:
            Starting position [x, y, z]
        """
        # Use starting position from gate config if available
        if hasattr(self.gate_config, 'starting_pos') and self.gate_config.starting_pos is not None:
            return np.array(self.gate_config.starting_pos, dtype=np.float64)

        # Fallback: calculate start position behind gate 0 by 2 meters
        first_gate = self.gate_positions[0]
        first_gate_yaw = self.gate_yaws[0]

        distance_behind = 2.0  # meters
        start_pos = first_gate - distance_behind * np.array([
            np.cos(first_gate_yaw),
            np.sin(first_gate_yaw),
            0.0
        ])

        return start_pos

    def get_info(self) -> Dict[str, Any]:
        """
        Get trajectory information.

        Returns:
            Dictionary with trajectory configuration
        """
        return {
            'n_gates': self.n_gates,
            'n_startup_points': self.n_startup_points,
            'n_startup_control_points': self.n_startup_control_points,
            'n_loop_control_points': self.n_loop_control_points,
            'degree': self.degree,
            'total_time': self.total_time,
            'velocity_scale': self.velocity_scale,
            'startup_time': self.startup_time,
            'n_parameters': self.get_parameter_count(),
            'gate_size': self.gate_size,
            'gate_offset_scale': self.gate_offset_scale
        }


def create_gate_trajectory_from_config(gate_config_name: str,
                                       n_startup_points: int = 1) -> BSplineGateTrajectory:
    """
    Convenience function to create trajectory from gate configuration name.

    Args:
        gate_config_name: Name of gate configuration ('figure8', 'circle', etc.)
        n_startup_points: Number of intermediate control points between start and gate 0 (default: 1, minimum 1 for cubic splines)

    Returns:
        BSplineGateTrajectory instance

    Example:
        >>> from airevolve.controllers.utils.gate_configs import GATE_CONFIGS
        >>> traj = create_gate_trajectory_from_config('figure8', n_startup_points=1)
    """
    from ..utils.gate_configs import GATE_CONFIGS

    if gate_config_name not in GATE_CONFIGS:
        raise ValueError(f"Unknown gate configuration: {gate_config_name}")

    gate_config = GATE_CONFIGS[gate_config_name]
    return BSplineGateTrajectory(gate_config, n_startup_points=n_startup_points)
