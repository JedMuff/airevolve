"""
Create combined video visualisation for a Lee-controller-tuned drone individual.

Usage:
    python examples/make_lee_video.py \
        .data/lee_tuning_figure8_6arms_20260226_110350/generation_49/individual_1210 \
        --gate-cfg figure8
"""

import os
import sys
import argparse
import numpy as np

from airevolve.controllers.utils.gate_configs import GATE_CONFIGS
from airevolve.evolution_tools.inspection_tools.behavioural_analysis.gate_based.extract_lee_simulation_data import extract_lee_simulation_data
from airevolve.evolution_tools.inspection_tools.behavioural_analysis.gate_based.animate_lee_individual import animate_lee_individual
from airevolve.evolution_tools.inspection_tools.behavioural_analysis.gate_based.combine_videos import combine_videos_from_directory
from airevolve.evolution_tools.inspection_tools.behavioural_analysis.gate_based.plot_speed_actions import plot_speed_angspeed_actions
from airevolve.evolution_tools.inspection_tools.behavioural_analysis.gate_based.calculate_stats import calculate_stats


def process_lee_individual(individual_dir, gate_cfg="figure8", sim_time=20.0,
                           dt=0.005, fps=100, width=864, height=700, dpi=200,
                           gate_label_ylevel=11.0, fontsize=7, pad=0.05,
                           offset_val=0.5, gate_line_alpha=0.5, alpha=1.0,
                           motor_colors=None, color='blue',
                           iso_overlay_position='lower right',
                           overlay_text_scale=0.7,
                           action_smooth_ms=0.0,
                           draw_forces=False):
    """
    Process a Lee-controller-tuned individual to create visualisation videos.

    Expects the directory to contain:
        - genome.npy: drone morphology
        - tuning_results.json: Lee controller gains + trajectory parameters

    Produces in <individual_dir>/videos/:
        - speed_plot.png/mp4, angular_speed_plot.png/mp4, actions_plot.png/mp4
        - top_view.mp4, iso_view.mp4
        - combined_output.mp4

    Args:
        individual_dir: Path to the individual's data directory.
        gate_cfg: Gate configuration name.
        sim_time: Simulation duration (seconds).
        dt: Simulation timestep (seconds).
        fps: Frames per second for videos.
        width: Video/plot width in pixels.
        height: Video/plot height in pixels.
        dpi: DPI for plot figures.
        gate_label_ylevel: Y-level for gate labels on plots.
        fontsize: Font size for gate labels.
        pad: Padding for gate label circles.
        offset_val: Offset for alternating gate labels.
        gate_line_alpha: Alpha for gate passage lines.
        alpha: Alpha for plot lines.
        motor_colors: List of colours for motor visualisation.
        color: Primary colour for speed/angular-speed plots.

    Returns:
        dict: Performance statistics.
    """
    if motor_colors is None:
        motor_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']

    # Validate inputs
    if not os.path.exists(individual_dir):
        raise ValueError(f"Individual directory does not exist: {individual_dir}")

    genome_file = os.path.join(individual_dir, "genome.npy")
    tuning_file = os.path.join(individual_dir, "tuning_results.json")

    if not os.path.exists(genome_file):
        raise ValueError(f"Genome file not found: {genome_file}")
    if not os.path.exists(tuning_file):
        raise ValueError(f"Tuning results file not found: {tuning_file}")

    # Output directory
    vid_dir = os.path.join(individual_dir, "videos")
    os.makedirs(vid_dir, exist_ok=True)

    # Load genome
    genome = np.load(genome_file)

    print(f"Processing Lee individual from: {individual_dir}")
    print(f"Gate configuration: {gate_cfg}")
    print(f"Simulation: {sim_time}s at dt={dt}")

    # --- 1. Extract simulation data ---
    print("Extracting simulation data...")
    ind_data = extract_lee_simulation_data(
        genome, tuning_file, gate_cfg,
        sim_time=sim_time, dt=dt,
    )

    ind_speed = np.linalg.norm(ind_data["velocities"], axis=1)
    ind_angular_speed = np.linalg.norm(ind_data["angular_velocities"], axis=1)
    ind_timesteps = np.arange(len(ind_speed))
    ind_gate_passes = ind_data["gate_passes"]
    ind_actions = ind_data["actions"]

    # --- 2. Calculate stats ---
    gate_timesteps = ind_timesteps[ind_gate_passes]
    # Convert from simulation timesteps to seconds
    gate_times_sec = gate_timesteps * dt

    n_gates = len(GATE_CONFIGS[gate_cfg].gate_pos)
    stats = calculate_stats(gate_times_sec, n_gates)
    print("######")
    print(f"Statistics for {individual_dir}:")
    for key, value in stats.items():
        print(f"{key} : {np.round(np.array(value), 3)}")
    print("######")

    # --- 3. Plot speed / angular speed / actions ---
    # Subsample plotting data to match playback fps (sim runs at 1/dt Hz)
    sim_hz = int(round(1.0 / dt))
    subsample = max(1, sim_hz // fps)
    plot_speed = ind_speed[::subsample]
    plot_ang_speed = ind_angular_speed[::subsample]
    plot_timesteps = np.arange(len(plot_speed))
    plot_actions = ind_actions[::subsample]

    # Optional cosmetic smoothing on the action plot. Defaults to 0 (raw
    # motor commands) so the plot stays honest about controller chatter.
    # Pass e.g. action_smooth_ms=100 for a 0.1s causal moving average.
    if action_smooth_ms and action_smooth_ms > 0:
        window = max(3, int(round(action_smooth_ms * fps / 1000.0)))
        if plot_actions.shape[0] >= window:
            kernel = np.ones(window) / window
            padded = np.pad(plot_actions, ((window - 1, 0), (0, 0)), mode='edge')
            plot_actions = np.stack([
                np.convolve(padded[:, m], kernel, mode='valid')
                for m in range(plot_actions.shape[1])
            ], axis=1)

    # Gate passes are sparse booleans — point-sampling would miss events
    # that land on skipped timesteps. Use a rolling OR over each window.
    n = len(ind_gate_passes)
    n_out = len(plot_speed)
    plot_gate_passes = np.zeros(n_out, dtype=bool)
    for i in range(n_out):
        start = i * subsample
        end = min(start + subsample, n)
        plot_gate_passes[i] = np.any(ind_gate_passes[start:end])

    print("Creating static plots...")
    plot_speed_angspeed_actions(
        plot_timesteps, plot_speed, plot_ang_speed, plot_actions, plot_gate_passes,
        save_dir=vid_dir, motor_colors=motor_colors,
        fps=fps, width=width, height=height, dpi=dpi,
        gate_lines=True, gate_labels=True,
        gate_label_ylevel=gate_label_ylevel, fontsize=fontsize,
        pad=pad, offset_val=offset_val, gate_line_alpha=gate_line_alpha,
        alpha=alpha, color=color, animate=False,
    )

    print("Creating animated plots...")
    plot_speed_angspeed_actions(
        plot_timesteps, plot_speed, plot_ang_speed, plot_actions, plot_gate_passes,
        save_dir=vid_dir, motor_colors=motor_colors,
        fps=fps, width=width, height=height, dpi=dpi,
        gate_lines=True, gate_labels=False,
        gate_label_ylevel=gate_label_ylevel, fontsize=fontsize,
        pad=pad, offset_val=offset_val, gate_line_alpha=gate_line_alpha,
        alpha=alpha, color=color, animate=True,
    )

    # --- 4. Create 3D animation videos ---
    try:
        print("Creating trajectory animations...")

        animate_lee_individual(
            genome=genome,
            tuning_results_path=tuning_file,
            gate_cfg=gate_cfg,
            save_dir=vid_dir,
            file_name="/top_view.mp4",
            sim_time=sim_time, dt=dt,
            view_type='top', follow=True,
            draw_forces=draw_forces, draw_path=True,
            auto_play=True, record=True,
            motor_colors=motor_colors, fps=fps,
            overlay_text_position=None,  # no gates counter on the top view
            overlay_text_scale=overlay_text_scale,
        )

        animate_lee_individual(
            genome=genome,
            tuning_results_path=tuning_file,
            gate_cfg=gate_cfg,
            save_dir=vid_dir,
            file_name="/iso_view.mp4",
            sim_time=sim_time, dt=dt,
            view_type='iso', follow=True,
            draw_forces=draw_forces, draw_path=True,
            auto_play=True, record=True,
            motor_colors=motor_colors, fps=fps,
            overlay_text_position=iso_overlay_position,
            overlay_text_scale=overlay_text_scale,
        )

        print("Animations created successfully!")
    except Exception as e:
        print(f"Warning: Animation creation failed: {e}")
        print("Plots have been generated successfully in the videos directory.")

    # --- 5. Combine all videos ---
    try:
        print("Combining videos into final compilation...")
        combine_videos_from_directory(vid_dir)
        print("Combined video created successfully!")
    except Exception as e:
        print(f"Warning: Video combination failed: {e}")

    print(f"Video processing completed! Videos saved to: {vid_dir}")
    return stats


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Create visualisation videos for a Lee-controller-tuned drone individual.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "individual_dir",
        help="Directory containing genome.npy and tuning_results.json",
    )
    parser.add_argument(
        "--gate-cfg",
        choices=["slalom", "figure8", "circle", "backandforth"],
        default="figure8",
        help="Gate configuration used during tuning",
    )
    parser.add_argument("--sim-time", type=float, default=20.0,
                        help="Simulation duration in seconds")
    parser.add_argument("--dt", type=float, default=0.005,
                        help="Simulation timestep in seconds")
    parser.add_argument("--fps", type=int, default=100,
                        help="Frames per second for videos")
    parser.add_argument("--width", type=int, default=864,
                        help="Video width in pixels")
    parser.add_argument("--height", type=int, default=700,
                        help="Video height in pixels")
    parser.add_argument("--color", default="blue",
                        help="Primary colour for plots")
    parser.add_argument("--iso-overlay-pos",
                        choices=["upper left", "upper right", "lower left", "lower right"],
                        default="lower right",
                        help="Corner for the gates-passed counter on the iso panel")
    parser.add_argument("--overlay-scale", type=float, default=0.7,
                        help="Font scale for the gates-passed counter")
    parser.add_argument("--action-smooth-ms", type=float, default=0.0,
                        help="Cosmetic moving-average window (ms) for the action "
                             "plot. 0 = raw motor commands (honest).")
    parser.add_argument("--no-forces", action="store_true",
                        help="Hide per-motor thrust direction arrows in both views")

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    try:
        stats = process_lee_individual(
            individual_dir=args.individual_dir,
            gate_cfg=args.gate_cfg,
            sim_time=args.sim_time,
            dt=args.dt,
            fps=args.fps,
            width=args.width,
            height=args.height,
            color=args.color,
            iso_overlay_position=args.iso_overlay_pos,
            overlay_text_scale=args.overlay_scale,
            action_smooth_ms=args.action_smooth_ms,
            draw_forces=not args.no_forces,
        )
        print("Processing completed successfully!")
        return stats
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
