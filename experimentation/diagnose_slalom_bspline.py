#!/usr/bin/env python3
"""
Diagnostic script for slalom B-spline trajectory.

Analyzes the B-spline trajectory for SlalomGates to verify:
1. Control point offsets are sensible (no explosion from periodic wraparound)
2. Trajectory starts near the desired start position
3. Trajectory passes close to all gates
4. 2D visualization of trajectory, gates, and control points
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '.')

from airevolve.controllers.trajectory_generation.bspline_gate_trajectory import BSplineGateTrajectory
from airevolve.controllers.utils.gate_configs import SlalomGates, Figure8Gates


def analyze_offsets(name, gate_config):
    """Print control point offsets for a gate config."""
    traj = BSplineGateTrajectory(gate_config)
    print(f"\n{'='*60}")
    print(f"Offset analysis: {name}")
    print(f"  periodic={getattr(gate_config, 'periodic', True)}")
    print(f"  n_gates={traj.n_gates}")
    print(f"{'='*60}")

    max_offset_norm = 0
    for i in range(traj.n_gates):
        offset = traj.gate_offsets[i]
        norm = np.linalg.norm(offset)
        max_offset_norm = max(max_offset_norm, norm)
        if i < 5 or i >= traj.n_gates - 2:
            print(f"  Gate {i:2d}: offset={offset}, |offset|={norm:.4f}")
        elif i == 5:
            print(f"  ...")

    print(f"  Max offset norm: {max_offset_norm:.4f}")
    return traj


def analyze_start(name, traj):
    """Analyze trajectory start position."""
    print(f"\nStart analysis: {name}")
    desired_start = traj.get_start_position()
    pos_at_t0, _, _ = traj.evaluate(0.0)
    dist = np.linalg.norm(pos_at_t0 - desired_start)
    print(f"  _u_start={traj._u_start:.4f}")
    print(f"  u_min={traj.spline.u_min:.4f}, u_max={traj.spline.u_max:.4f}")
    print(f"  Desired start: {desired_start}")
    print(f"  Trajectory at t=0: {pos_at_t0}")
    print(f"  Distance from desired start: {dist:.4f} m")
    return dist


def analyze_gate_proximity(name, traj):
    """Check minimum distance from trajectory to each gate."""
    print(f"\nGate proximity: {name}")
    min_dists = traj.check_gate_proximity(dt=0.01)
    for i, d in enumerate(min_dists):
        flag = " *** FAR" if d > 0.5 else ""
        if i < 5 or i >= traj.n_gates - 2 or d > 0.5:
            print(f"  Gate {i:2d}: min_dist={d:.4f} m{flag}")
        elif i == 5:
            print(f"  ...")
    print(f"  Max min-distance: {np.max(min_dists):.4f} m")
    print(f"  Gates within 0.5m: {np.sum(min_dists < 0.5)}/{traj.n_gates}")
    return min_dists


def plot_trajectory_2d(name, traj, filename):
    """Create top-down 2D plot of trajectory, gates, and control points."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 6))

    # Sample trajectory
    sample = traj.sample_trajectory(dt=0.01)
    positions = sample['position']

    # Plot trajectory
    ax.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=1, alpha=0.7, label='Trajectory')

    # Plot gates
    gate_pos = traj.gate_positions
    for i in range(traj.n_gates):
        color = 'red' if i == 0 else 'green'
        ax.plot(gate_pos[i, 0], gate_pos[i, 1], 'o', color=color, markersize=8)
        ax.annotate(str(i), (gate_pos[i, 0], gate_pos[i, 1]),
                    textcoords="offset points", xytext=(5, 5), fontsize=7)

        # Gate normal
        yaw = traj.gate_yaws[i]
        normal = np.array([np.cos(yaw), np.sin(yaw)]) * 0.3
        ax.arrow(gate_pos[i, 0], gate_pos[i, 1], normal[0], normal[1],
                 head_width=0.1, head_length=0.05, fc='gray', ec='gray', alpha=0.5)

    # Plot control points
    cps = traj.get_all_control_points()
    ax.plot(cps[:, 0], cps[:, 1], 'x', color='orange', markersize=6, label='Control points')

    # Plot start
    desired_start = traj.get_start_position()
    pos_at_t0, _, _ = traj.evaluate(0.0)
    ax.plot(desired_start[0], desired_start[1], '*', color='purple', markersize=15, label='Desired start')
    ax.plot(pos_at_t0[0], pos_at_t0[1], 's', color='cyan', markersize=10, label='Traj at t=0')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'{name} - Top-down view')
    ax.legend(loc='upper left', fontsize=8)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"\nSaved plot: {filename}")
    plt.close()


def main():
    print("=" * 60)
    print("Slalom B-Spline Trajectory Diagnostics")
    print("=" * 60)

    # Analyze SlalomGates
    traj_slalom = analyze_offsets("SlalomGates", SlalomGates)
    analyze_start("SlalomGates", traj_slalom)
    analyze_gate_proximity("SlalomGates", traj_slalom)
    plot_trajectory_2d("SlalomGates", traj_slalom, "experimentation/plots/slalom_trajectory.png")

    # Compare with Figure8Gates
    traj_fig8 = analyze_offsets("Figure8Gates", Figure8Gates)
    analyze_start("Figure8Gates", traj_fig8)
    analyze_gate_proximity("Figure8Gates", traj_fig8)
    plot_trajectory_2d("Figure8Gates", traj_fig8, "experimentation/plots/figure8_trajectory.png")


if __name__ == '__main__':
    main()
