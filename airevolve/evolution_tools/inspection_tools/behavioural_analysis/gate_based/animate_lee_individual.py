import json
import os
import numpy as np

from airevolve.simulator.visualization.animation import view
from airevolve.controllers.utils.gate_configs import GATE_CONFIGS
from airevolve.evolution_tools.evaluators.lee_tune_evaluator import simulate_with_gains
from airevolve.evolution_tools.inspection_tools.utils import convert_to_cartesian, ENU_to_NED
from airevolve.evolution_tools.inspection_tools.morphological_descriptors.hovering_info import orientation_to_unit_vector


def _genome_to_propellers(genome):
    """Convert a genome array to propeller dicts for the view() function."""
    propellers = []
    for arm in genome:
        r, theta, phi, motor_pitch, motor_yaw, direction = arm
        ex, ey, ez = convert_to_cartesian(r, theta, phi)
        x, y, z = ENU_to_NED(ex, ey, ez)
        rot = "ccw" if direction < 0.5 else "cw"
        motor_dir = orientation_to_unit_vector(0.0, motor_pitch, motor_yaw)
        propellers.append({
            "loc": [float(x), float(y), float(z)],
            "dir": [float(motor_dir[0]), float(motor_dir[1]), float(motor_dir[2]), rot],
            "propsize": 2,
        })
    return propellers


def animate_lee_individual(genome, tuning_results_path, gate_cfg,
                           save_dir, file_name,
                           sim_time=20.0, dt=0.005,
                           view_type='top', follow=True,
                           draw_forces=False, draw_path=True,
                           auto_play=True, record=True,
                           motor_colors=None, fps=100):
    """
    Pre-record a Lee-controller flight and replay it via view().

    Args:
        genome: Drone morphology array (N_arms, 6).
        tuning_results_path: Path to tuning_results.json.
        gate_cfg: Gate configuration name.
        save_dir: Directory to save the output video.
        file_name: Video filename (e.g. '/top_view.mp4').
        sim_time: Simulation duration in seconds.
        dt: Simulation timestep (default 0.005 s = 200 Hz).
        view_type: Camera view ('top' or 'iso').
        follow: Whether camera follows the drone.
        draw_forces: Whether to draw thrust vectors.
        draw_path: Whether to draw flight path.
        auto_play: Whether animation auto-plays.
        record: Whether to record video.
        motor_colors: List of colour names for motors.
        fps: Playback frames per second.
    """
    if motor_colors is None:
        motor_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']

    os.makedirs(save_dir, exist_ok=True)

    # Load tuning results
    with open(tuning_results_path, 'r') as f:
        tuning = json.load(f)

    gains = tuning['gains']
    bspline_timing = tuning.get('bspline_timing', None)
    gate_offsets = tuning.get('gate_offsets', None)

    gate_config = GATE_CONFIGS[gate_cfg]

    # Run simulation once with trajectory recording
    result = simulate_with_gains(
        individual=genome,
        pos_gain=gains['pos_P'],
        vel_gain=gains['vel_P'],
        att_gain=gains['att_P'],
        rate_gain=gains['rate_P'],
        gate_config=gate_config,
        sim_time=sim_time,
        dt=dt,
        bspline_timing=bspline_timing,
        gate_offsets=gate_offsets,
        record_trajectory=True,
    )

    traj = result['trajectory']
    positions = traj['positions']
    euler_angles = traj['euler_angles']
    motor_commands = traj['motor_commands']

    # Subsample from simulation rate (200 Hz) to playback rate (100 Hz)
    sim_hz = int(round(1.0 / dt))
    subsample = max(1, sim_hz // fps)
    positions = positions[::subsample]
    euler_angles = euler_angles[::subsample]
    motor_commands = motor_commands[::subsample]

    # Normalise motor commands for visualisation
    w_max = motor_commands.max()
    if w_max > 0:
        motor_commands = motor_commands / w_max

    num_motors = genome.shape[0]
    num_frames = len(positions)

    # Build propellers config for view()
    propellers = _genome_to_propellers(genome)

    # Index-based callback for view()
    frame_idx = [0]

    def get_drone_state():
        idx = min(frame_idx[0], num_frames - 1)
        frame_idx[0] += 1

        state = {
            'x': positions[idx, 0],
            'y': positions[idx, 1],
            'z': positions[idx, 2],
            'phi': euler_angles[idx, 0],
            'theta': euler_angles[idx, 1],
            'psi': euler_angles[idx, 2],
        }
        for m in range(num_motors):
            state[f'u{m+1}'] = motor_commands[idx, m]

        return state

    view(
        propellers,
        get_drone_state=get_drone_state,
        fps=fps,
        gate_pos=gate_config.gate_pos,
        gate_yaw=gate_config.gate_yaw,
        gate_size=gate_config.gate_size,
        record_steps=num_frames,
        record_file=save_dir + file_name,
        show_window=False,
        view_type=view_type,
        follow=follow,
        draw_forces=draw_forces,
        draw_path=draw_path,
        auto_play=auto_play,
        record=record,
        motor_colors=motor_colors,
    )
