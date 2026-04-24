import os
import sys
import argparse
import numpy as np
import torch
from airevolve.evolution_tools.inspection_tools.behavioural_analysis.gate_based.animate_individual_with_gates import animate_individual
from airevolve.evolution_tools.inspection_tools.behavioural_analysis.gate_based.extract_simulation_data import extract_simulation_data
from airevolve.evolution_tools.inspection_tools.behavioural_analysis.gate_based.combine_videos import combine_videos_from_directory
from airevolve.evolution_tools.inspection_tools.behavioural_analysis.gate_based.plot_speed_actions import plot_speed_angspeed_actions
from airevolve.evolution_tools.inspection_tools.behavioural_analysis.gate_based.calculate_stats import calculate_stats

def process_individual(individual_dir, gate_cfg="slalom", device=None, fps=100, width=864, height=700, dpi=200, 
                       gate_label_ylevel=11.0, fontsize=7, pad=0.05, offset_val=0.5, gate_line_alpha=0.5, alpha=1.0, 
                       motor_colors=None, color='blue'):
    """
    Process an individual drone to create visualization videos and analysis plots.
    
    Args:
        individual_dir (str): Directory containing individual.npy and policy.zip
        gate_cfg (str): Gate configuration ('slalom', 'figure8', 'circle', 'backandforth')
        device (str): Device to use for computation ('cpu', 'cuda:0', etc.)
        fps (int): Frames per second for videos
        width (int): Video width in pixels
        height (int): Video height in pixels
        dpi (int): DPI for plots
        gate_label_ylevel (float): Y-level for gate labels
        fontsize (int): Font size for labels
        pad (float): Padding for plots
        offset_val (float): Offset value for plots
        gate_line_alpha (float): Alpha value for gate lines
        alpha (float): Alpha value for plots
        motor_colors (list): Colors for motors
        color (str): Primary color for plots
        
    Returns:
        dict: Statistics about the individual's performance
    """
    if motor_colors is None:
        motor_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']

    # Validate inputs
    if not os.path.exists(individual_dir):
        raise ValueError(f"Individual directory does not exist: {individual_dir}")
    
    ind_policy_file = os.path.join(individual_dir, "policy.zip")
    individual_body = os.path.join(individual_dir, "individual.npy")
    
    if not os.path.exists(ind_policy_file):
        raise ValueError(f"Policy file not found: {ind_policy_file}")
    if not os.path.exists(individual_body):
        raise ValueError(f"Individual file not found: {individual_body}")

    # Set device
    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    # Define directories and configurations
    vid_dir = os.path.join(individual_dir, "videos")
    if not os.path.exists(vid_dir):
        os.makedirs(vid_dir)

    # Load data for the individual
    individual = np.load(individual_body)
    
    print(f"Processing individual from: {individual_dir}")
    print(f"Gate configuration: {gate_cfg}")
    print(f"Using device: {device}")

    # Extract simulation data
    ind_data = extract_simulation_data(individual, ind_policy_file, gate_cfg, device)

    # Access extracted data
    ind_speed = np.linalg.norm(ind_data["velocities"], axis=1)
    ind_angular_speed = np.linalg.norm(ind_data["angular_velocities"], axis=1)
    ind_timesteps = np.arange(len(ind_speed))
    ind_gate_passes = ind_data["gate_passes"]
    ind_actions = ind_data["actions"]
    # Calculate the time at which is gate is got to
    gate_timesteps = ind_timesteps[ind_gate_passes]
    gate_times_sec = gate_timesteps / fps

    if gate_cfg == "figure8":
        n_gates = 8
    else:
        n_gates = 4 
    stats = calculate_stats(gate_times_sec, n_gates)
    print("######")
    print(f"Statistics for {individual_dir}:")
    for key, value in stats.items():
        print(f"{key} : {np.round(np.array(value),3)}")
    print("######")

    # Plot speed, angular speed, and actions
    plot_speed_angspeed_actions(ind_timesteps, ind_speed, ind_angular_speed, ind_actions, ind_gate_passes, save_dir=vid_dir,
                       fps=fps, width=width, height=height, dpi=dpi, gate_lines=True, gate_labels=True, 
                       gate_label_ylevel=gate_label_ylevel, fontsize=fontsize, pad=pad, offset_val=offset_val, 
                       gate_line_alpha=gate_line_alpha, alpha=alpha, color=color, animate=False)

    plot_speed_angspeed_actions(ind_timesteps, ind_speed, ind_angular_speed, ind_actions, ind_gate_passes, save_dir=vid_dir,
                       fps=fps, width=width, height=height, dpi=dpi, gate_lines=True, gate_labels=False, 
                       gate_label_ylevel=gate_label_ylevel, fontsize=fontsize, pad=pad, offset_val=offset_val, 
                       gate_line_alpha=gate_line_alpha, alpha=alpha, color=color, animate=True)

    # Create animation videos
    try:
        print("Creating trajectory animations...")
        
        # Create top view animation
        animate_individual(
            gate_cfg=gate_cfg,
            individual_dir=individual_dir,
            save_dir=vid_dir,
            file_name="/top_view.mp4",
            device=device,
            view_type='top',
            follow=True,
            draw_forces=False,
            draw_path=True,
            auto_play=True,
            record=True,
            motor_colors=motor_colors
        )
        
        # Create isometric view animation
        animate_individual(
            gate_cfg=gate_cfg,
            individual_dir=individual_dir,
            save_dir=vid_dir,
            file_name="/iso_view.mp4",
            device=device,
            view_type='iso',
            follow=True,
            draw_forces=False,
            draw_path=True,
            auto_play=True,
            record=True,
            motor_colors=motor_colors
        )
        
        print("Animations created successfully!")
    except Exception as e:
        print(f"Warning: Animation creation failed: {e}")
        print("Plots have been generated successfully in the videos directory.")
    
    # Combine all videos into a single comprehensive video
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
        description="Create visualization videos and analysis plots for a trained drone individual.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "individual_dir", 
        nargs='?',
        default="simple_evaluation",
        help="Directory containing individual.npy and policy.zip files"
    )
    
    parser.add_argument(
        "--gate-cfg", 
        choices=["slalom", "figure8", "circle", "backandforth"],
        default="figure8",
        help="Gate configuration used during training"
    )
    
    parser.add_argument(
        "--device",
        default=None,
        help="Device to use for computation (e.g., 'cpu', 'cuda:0'). Auto-detected if not specified."
    )
    
    parser.add_argument(
        "--fps",
        type=int,
        default=100,
        help="Frames per second for videos"
    )
    
    parser.add_argument(
        "--width",
        type=int,
        default=864,
        help="Video width in pixels"
    )
    
    parser.add_argument(
        "--height",
        type=int,
        default=700,
        help="Video height in pixels"
    )
    
    parser.add_argument(
        "--color",
        default="blue",
        help="Primary color for plots"
    )
    
    return parser.parse_args()


def main():
    """Main function for command line usage."""
    args = parse_args()
    
    try:
        stats = process_individual(
            individual_dir=args.individual_dir,
            gate_cfg=args.gate_cfg,
            device=args.device,
            fps=args.fps,
            width=args.width,
            height=args.height,
            color=args.color
        )
        print("Processing completed successfully!")
        return stats
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()