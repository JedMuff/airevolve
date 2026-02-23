#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize Initial B-Spline Trajectory for Tuning

This script creates a 3D visualization of the initial (default) B-spline trajectory
before optimization. Useful for understanding the starting point and checking if
the trajectory setup makes sense.

Usage:
    python visualize_initial_bspline.py --gate-cfg figure8
    python visualize_initial_bspline.py --gate-cfg circle --n-startup-points 3
    python visualize_initial_bspline.py --gate-cfg slalom --samples 500
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import argparse

from airevolve.controllers.trajectory_generation.bspline_gate_trajectory import BSplineGateTrajectory
from airevolve.controllers.utils.gate_configs import GATE_CONFIGS


def visualize_bspline_trajectory(bspline_traj, gate_config, n_samples=200, orient="NED"):
    """
    Create 3D visualization of B-spline trajectory with gates, control points, and knots.

    Args:
        bspline_traj: BSplineGateTrajectory instance
        gate_config: Gate configuration
        n_samples: Number of samples along trajectory
        orient: Coordinate frame ("NED" or "ENU")
    """
    # Sample the trajectory
    trajectory_data = bspline_traj.sample_trajectory(dt=bspline_traj.total_time / n_samples)

    positions = trajectory_data['position']
    velocities = trajectory_data['velocity']
    times = trajectory_data['time']

    # Get control points from both splines
    startup_cps = bspline_traj.get_startup_control_points()
    loop_cps = bspline_traj.get_loop_control_points()

    # Create figure
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Extract coordinates for trajectory
    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]

    # Startup control points
    startup_x = startup_cps[:, 0]
    startup_y = startup_cps[:, 1]
    startup_z = startup_cps[:, 2]

    # Loop control points
    loop_x = loop_cps[:, 0]
    loop_y = loop_cps[:, 1]
    loop_z = loop_cps[:, 2]

    # Apply NED orientation if needed
    if orient == "NED":
        z = -z
        startup_z = -startup_z
        loop_z = -loop_z

    # Plot trajectory path (split into startup and loop phases)
    startup_end_idx = int(bspline_traj.startup_time / (bspline_traj.total_time / n_samples))
    ax.plot(x[:startup_end_idx], y[:startup_end_idx], z[:startup_end_idx],
            'g-', linewidth=2, label='Run-up Phase', alpha=0.8, zorder=3)
    ax.plot(x[startup_end_idx:], y[startup_end_idx:], z[startup_end_idx:],
            'b-', linewidth=2, label='Racing Loop', alpha=0.8, zorder=3)

    # Draw startup control points
    ax.scatter(startup_x, startup_y, startup_z, color='limegreen', marker='s', s=150,
               alpha=0.7, label='Run-up Control Points', edgecolors='darkgreen', linewidths=1.5, zorder=5)

    # Draw loop control points
    ax.scatter(loop_x, loop_y, loop_z, color='purple', marker='o', s=150,
               alpha=0.7, label='Loop Control Points', edgecolors='black', linewidths=1.5, zorder=5)

    # Mark start point
    ax.scatter([x[0]], [y[0]], [z[0]], color='green', marker='o', s=300,
               label='Start', edgecolors='darkgreen', linewidths=3, zorder=15)

    # Draw lines connecting startup control points
    ax.plot(startup_x, startup_y, startup_z, color='limegreen', linestyle='--',
            linewidth=1.5, alpha=0.4, label='Run-up Polygon')

    # Draw lines connecting loop control points (periodic)
    loop_x_closed = np.append(loop_x, loop_x[0])
    loop_y_closed = np.append(loop_y, loop_y[0])
    loop_z_closed = np.append(loop_z, loop_z[0])
    ax.plot(loop_x_closed, loop_y_closed, loop_z_closed, color='purple', linestyle='--',
            linewidth=1.5, alpha=0.4, label='Loop Polygon')

    # Add labels to startup control points
    for i, (cpx, cpy, cpz) in enumerate(zip(startup_x, startup_y, startup_z)):
        if i == 0:
            label = 'START'
        elif i < len(startup_cps) - 1:
            label = f'RU{i}'  # Run-up intermediate
        else:
            label = 'G0'  # Gate 0
        ax.text(cpx, cpy, cpz, f'  {label}', fontsize=8, color='darkgreen', weight='bold')

    # Add labels to loop control points
    for i, (cpx, cpy, cpz) in enumerate(zip(loop_x, loop_y, loop_z)):
        label = f'G{i}'  # Gate number
        ax.text(cpx, cpy, cpz, f'  {label}', fontsize=8, color='purple', weight='bold')

    # Draw gates
    gate_pos = np.array(gate_config.gate_pos)
    gate_yaw = np.array(gate_config.gate_yaw)
    gate_size = gate_config.gate_size

    for idx, (gpos, gyaw) in enumerate(zip(gate_pos, gate_yaw)):
        # Create gate corners (vertical square)
        half_size = gate_size / 2.0

        # Gate corners in local frame
        local_corners = np.array([
            [0, -half_size, -half_size],  # Bottom left
            [0,  half_size, -half_size],  # Bottom right
            [0,  half_size,  half_size],  # Top right
            [0, -half_size,  half_size]   # Top left
        ])

        # Rotation matrix for yaw
        cos_yaw = np.cos(gyaw)
        sin_yaw = np.sin(gyaw)
        R = np.array([
            [cos_yaw, -sin_yaw, 0],
            [sin_yaw,  cos_yaw, 0],
            [0,        0,       1]
        ])

        # Transform corners to world frame
        gate_corners_3d = []
        for corner in local_corners:
            rotated = R @ corner
            world_pos = rotated + gpos

            if orient == "NED":
                gate_corners_3d.append([world_pos[0], world_pos[1], -world_pos[2]])
            else:
                gate_corners_3d.append([world_pos[0], world_pos[1], world_pos[2]])

        # Draw gate as polygon (add label only on first gate for legend)
        if idx == 0:
            gate_poly = Poly3DCollection([gate_corners_3d], alpha=0.25,
                                         facecolor='orange', edgecolor='darkorange',
                                         linewidths=2.5, label='Gates')
        else:
            gate_poly = Poly3DCollection([gate_corners_3d], alpha=0.25,
                                         facecolor='orange', edgecolor='darkorange',
                                         linewidths=2.5)
        ax.add_collection3d(gate_poly)

        # Draw gate center and label with order number
        if orient == "NED":
            gate_z = -gpos[2]
        else:
            gate_z = gpos[2]

        # Draw center marker with order number inside
        ax.scatter([gpos[0]], [gpos[1]], [gate_z], color='darkorange',
                   marker='o', s=150, edgecolors='black', linewidths=2, zorder=6)

        # Add large order number on the gate (1-indexed)
        gate_num = idx + 1
        ax.text(gpos[0], gpos[1], gate_z, f'{gate_num}', fontsize=14,
                color='white', weight='bold', ha='center', va='center', zorder=7,
                bbox=dict(boxstyle='circle,pad=0.3', facecolor='darkorange',
                         edgecolor='black', linewidth=2))

        # Add gate label slightly offset
        offset = gate_size * 0.6
        ax.text(gpos[0], gpos[1] + offset, gate_z, f'Gate {gate_num}', fontsize=9,
                color='darkorange', weight='bold', ha='center')

    # Calculate and display gate proximity information
    min_distances = bspline_traj.check_gate_proximity(dt=bspline_traj.total_time / n_samples)

    # Set axis properties
    extraEachSide = 0.5
    maxRange = 0.5 * np.array([
        x.max() - x.min(),
        y.max() - y.min(),
        z.max() - z.min()
    ]).max() + extraEachSide

    mid_x = 0.5 * (x.max() + x.min())
    mid_y = 0.5 * (y.max() + y.min())
    mid_z = 0.5 * (z.max() + z.min())

    ax.set_xlim3d([mid_x - maxRange, mid_x + maxRange])
    ax.set_xlabel('X (m)', fontsize=10, weight='bold')

    if orient == "NED":
        ax.set_ylim3d([mid_y + maxRange, mid_y - maxRange])
    else:
        ax.set_ylim3d([mid_y - maxRange, mid_y + maxRange])
    ax.set_ylabel('Y (m)', fontsize=10, weight='bold')

    ax.set_zlim3d([mid_z - maxRange, mid_z + maxRange])
    ax.set_zlabel('Z (m) - Altitude', fontsize=10, weight='bold')

    # Title and legend
    ax.set_title(f'Initial B-Spline Trajectory\n'
                 f'Gate Config: {gate_config.__name__} | '
                 f'Startup Points: {bspline_traj.n_startup_points} | '
                 f'Degree: {bspline_traj.degree}',
                 fontsize=12, weight='bold', pad=20)

    # Place legend outside the plot to avoid overlap
    ax.legend(loc='upper left', bbox_to_anchor=(0.0, 0.95), fontsize=9,
              framealpha=0.95, edgecolor='black', fancybox=True, shadow=True)

    # Add grid
    ax.grid(True, alpha=0.3)

    # Add info text box
    info_text = f"Trajectory Info:\n"
    info_text += f"  Startup CPs: {bspline_traj.n_startup_control_points}\n"
    info_text += f"  Loop CPs: {bspline_traj.n_loop_control_points}\n"
    info_text += f"  Total Time: {bspline_traj.total_time:.1f}s\n"
    info_text += f"  Startup Time: {bspline_traj.startup_time:.1f}s\n"
    info_text += f"  Velocity Scale: {bspline_traj.velocity_scale:.2f}\n\n"
    info_text += f"Gate Proximity (min distance):\n"

    for i, dist in enumerate(min_distances):
        pass_status = "✓" if dist < gate_size / 2 else "✗"
        info_text += f"  Gate {i+1}: {dist:.2f}m {pass_status}\n"

    # Add text box
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text2D(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=9,
              verticalalignment='top', bbox=props, family='monospace')

    plt.tight_layout()

    return fig, ax


def plot_velocity_profile(bspline_traj, n_samples=200):
    """
    Plot velocity magnitude over time.

    Args:
        bspline_traj: BSplineGateTrajectory instance
        n_samples: Number of samples
    """
    trajectory_data = bspline_traj.sample_trajectory(dt=bspline_traj.total_time / n_samples)

    times = trajectory_data['time']
    velocities = trajectory_data['velocity']

    vel_mag = np.linalg.norm(velocities, axis=1)

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(times, vel_mag, 'b-', linewidth=2)
    ax.set_xlabel('Time (s)', fontsize=11, weight='bold')
    ax.set_ylabel('Velocity Magnitude (m/s)', fontsize=11, weight='bold')
    ax.set_title('Velocity Profile Along Trajectory', fontsize=12, weight='bold')
    ax.grid(True, alpha=0.3)

    # Add statistics
    stats_text = f"Velocity Statistics:\n"
    stats_text += f"  Mean: {np.mean(vel_mag):.2f} m/s\n"
    stats_text += f"  Max: {np.max(vel_mag):.2f} m/s\n"
    stats_text += f"  Min: {np.min(vel_mag):.2f} m/s"

    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right',
            bbox=props, family='monospace')

    plt.tight_layout()

    return fig, ax


def main():
    parser = argparse.ArgumentParser(
        description='Visualize Initial B-Spline Trajectory',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--gate-cfg', type=str, default='figure8',
                       choices=['figure8', 'circle', 'slalom', 'backandforth'],
                       help='Gate configuration (default: figure8)')
    parser.add_argument('--n-startup-points', type=int, default=2,
                       help='Number of startup control points (default: 2)')
    parser.add_argument('--samples', type=int, default=300,
                       help='Number of trajectory samples (default: 300)')
    parser.add_argument('--orient', type=str, choices=['NED', 'ENU'], default='NED',
                       help='Coordinate frame orientation (default: NED)')
    parser.add_argument('--save', type=str, default=None,
                       help='Save figure to file (e.g., "trajectory.png")')
    parser.add_argument('--show-velocity', action='store_true',
                       help='Also show velocity profile plot')

    args = parser.parse_args()

    # Get gate configuration
    gate_config = GATE_CONFIGS[args.gate_cfg]

    print(f"\n{'='*70}")
    print(f"INITIAL B-SPLINE TRAJECTORY VISUALIZATION")
    print(f"{'='*70}")
    print(f"Gate configuration: {args.gate_cfg}")
    print(f"Number of gates: {len(gate_config.gate_pos)}")
    print(f"Startup points: {args.n_startup_points}")
    print(f"Coordinate frame: {args.orient}")
    print(f"{'='*70}\n")

    # Create B-spline trajectory with default parameters
    bspline_traj = BSplineGateTrajectory(gate_config, n_startup_points=args.n_startup_points)

    # Print trajectory info
    info = bspline_traj.get_info()
    print("Trajectory Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    print(f"\nDefault Parameters:")
    default_params = bspline_traj.get_default_parameters()
    print(f"  Total parameters: {len(default_params)}")
    print(f"  Parameter vector shape: {default_params.shape}")

    # Get parameter bounds
    lower, upper = bspline_traj.get_parameter_bounds()
    print(f"\nParameter Bounds:")
    print(f"  Lower bounds shape: {lower.shape}")
    print(f"  Upper bounds shape: {upper.shape}")

    # Check gate proximity
    print(f"\n{'='*70}")
    print("Gate Proximity Analysis:")
    print(f"{'='*70}")
    min_distances = bspline_traj.check_gate_proximity(dt=bspline_traj.total_time / args.samples)

    for i, dist in enumerate(min_distances):
        gate_size = gate_config.gate_size
        within_gate = "PASS" if dist < gate_size / 2 else "MISS"
        print(f"  Gate {i+1}: min distance = {dist:.3f}m [{within_gate}]")

    passes = sum(1 for d in min_distances if d < gate_config.gate_size / 2)
    print(f"\nEstimated gates passed with default params: {passes}/{len(min_distances)}")
    print(f"{'='*70}\n")

    # Create visualizations
    print("Creating 3D visualization...")
    fig1, ax1 = visualize_bspline_trajectory(bspline_traj, gate_config,
                                             n_samples=args.samples, orient=args.orient)

    if args.show_velocity:
        print("Creating velocity profile plot...")
        fig2, ax2 = plot_velocity_profile(bspline_traj, n_samples=args.samples)

    # Save if requested
    if args.save:
        print(f"Saving figure to: {args.save}")
        fig1.savefig(args.save, dpi=150, bbox_inches='tight')

        if args.show_velocity:
            vel_filename = args.save.replace('.png', '_velocity.png')
            fig2.savefig(vel_filename, dpi=150, bbox_inches='tight')
            print(f"Velocity profile saved to: {vel_filename}")

    print("\nDisplaying visualization...")
    print("(Close the window to exit)\n")

    plt.show()


if __name__ == '__main__':
    main()
