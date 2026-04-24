"""
Create visualization videos for hover learning evaluation.

This module provides functions to create video recordings and visualizations
of trained hover policies, showing the drone attempting to maintain a stable
hover at a target position.
"""

import os
import sys
import argparse
import numpy as np
import torch
from stable_baselines3 import PPO

# Import hover environment and visualization tools
from airevolve.evolution_tools.evaluators.drone_hover_env import DroneHoverEnv
from airevolve.evolution_tools.evaluators.hover_train import animate_hover_policy


def create_hover_video(individual_dir,
                      difficulty='easy',
                      device=None,
                      fps=100,
                      record_steps=1200,
                      view_type='iso',
                      follow=True,
                      draw_forces=False,
                      draw_path=True,
                      motor_colors=None,
                      video_filename='hover_video.mp4'):
    """
    Create a video of the trained hover policy.

    Args:
        individual_dir (str): Directory containing individual.npy and policy.zip
        difficulty (str): Difficulty level ('easy', 'medium', 'hard')
        device (str): Device to use for computation ('cpu', 'cuda:0', etc.)
        fps (int): Frames per second for video
        record_steps (int): Number of simulation steps to record
        view_type (str): Camera view ('top', 'iso', 'isometric')
        follow (bool): Whether camera should follow the drone
        draw_forces (bool): Whether to draw thrust force vectors
        draw_path (bool): Whether to draw drone trajectory path
        motor_colors (list): Colors for motors
        video_filename (str): Output video filename

    Returns:
        str: Path to created video file
    """
    if motor_colors is None:
        motor_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']

    # Validate inputs
    if not os.path.exists(individual_dir):
        raise ValueError(f"Individual directory does not exist: {individual_dir}")

    policy_file = os.path.join(individual_dir, "policy.zip")
    individual_file = os.path.join(individual_dir, "individual.npy")

    if not os.path.exists(policy_file):
        raise ValueError(f"Policy file not found: {policy_file}")
    if not os.path.exists(individual_file):
        raise ValueError(f"Individual file not found: {individual_file}")

    # Set device
    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create video directory
    vid_dir = os.path.join(individual_dir, "videos")
    os.makedirs(vid_dir, exist_ok=True)

    # Load individual and policy
    individual = np.load(individual_file)
    model = PPO.load(policy_file, device=device)

    print(f"Creating hover video from: {individual_dir}")
    print(f"Difficulty: {difficulty}")
    print(f"Using device: {device}")
    print(f"Recording {record_steps} steps at {fps} fps")

    # Configure difficulty settings (same as in hover_train.py)
    if difficulty == "easy":
        add_disturbances = False
        disturbance_strength = 0.0
        position_tolerance = 0.5
        velocity_tolerance = 0.3
        x_bounds = [-5, 5]
        y_bounds = [-5, 5]
        z_bounds = [-5, 0]
    elif difficulty == "medium":
        add_disturbances = True
        disturbance_strength = 0.05
        position_tolerance = 0.5
        velocity_tolerance = 0.3
        x_bounds = [-4, 4]
        y_bounds = [-4, 4]
        z_bounds = [-4, 0]
    elif difficulty == "hard":
        add_disturbances = True
        disturbance_strength = 0.1
        position_tolerance = 0.3
        velocity_tolerance = 0.2
        x_bounds = [-3, 3]
        y_bounds = [-3, 3]
        z_bounds = [-3, 0]
    else:
        raise ValueError("Invalid difficulty level. Choose 'easy', 'medium', or 'hard'")

    # Create environment for visualization
    env = DroneHoverEnv(
        num_envs=1,
        individual=individual,
        target_pos=np.array([0.0, 0.0, -1.5], dtype=np.float32),
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        z_bounds=z_bounds,
        position_tolerance=position_tolerance,
        velocity_tolerance=velocity_tolerance,
        num_state_history=0,
        num_action_history=0,
        history_step_size=1,
        render_mode='rgb_array',
        device=device,
        add_disturbances=add_disturbances,
        disturbance_strength=disturbance_strength
    )

    # Create video path
    video_path = os.path.join(vid_dir, video_filename)

    # Create the animation/video
    print(f"Recording animation to: {video_path}")
    animate_hover_policy(
        individual=individual,
        model=model,
        env=env,
        deterministic=True,
        view_type=view_type,
        motor_colors=motor_colors,
        follow=follow,
        draw_forces=draw_forces,
        draw_path=draw_path,
        auto_play=True,
        record=True,
        record_steps=record_steps,
        record_file=video_path,
        fps=fps,
        show_window=False  # Don't show window during recording
    )

    print(f"Video created successfully: {video_path}")
    return video_path


def process_individual(individual_dir, difficulty='easy', device=None):
    """
    Process an individual to create all hover visualization videos.

    Creates multiple views:
    - Top view with path tracing
    - Isometric view with path tracing
    - Isometric view with force vectors

    Args:
        individual_dir (str): Directory containing individual.npy and policy.zip
        difficulty (str): Difficulty level ('easy', 'medium', 'hard')
        device (str): Device to use for computation

    Returns:
        dict: Paths to created videos
    """
    videos = {}

    try:
        print("Creating top view video...")
        videos['top_view'] = create_hover_video(
            individual_dir=individual_dir,
            difficulty=difficulty,
            device=device,
            view_type='top',
            follow=True,
            draw_forces=False,
            draw_path=True,
            video_filename='hover_top_view.mp4'
        )
    except Exception as e:
        print(f"Warning: Top view video creation failed: {e}")

    try:
        print("Creating isometric view video...")
        videos['iso_view'] = create_hover_video(
            individual_dir=individual_dir,
            difficulty=difficulty,
            device=device,
            view_type='iso',
            follow=True,
            draw_forces=False,
            draw_path=True,
            video_filename='hover_iso_view.mp4'
        )
    except Exception as e:
        print(f"Warning: Isometric view video creation failed: {e}")

    try:
        print("Creating isometric view with forces...")
        videos['iso_forces'] = create_hover_video(
            individual_dir=individual_dir,
            difficulty=difficulty,
            device=device,
            view_type='iso',
            follow=True,
            draw_forces=True,
            draw_path=False,
            video_filename='hover_iso_forces.mp4'
        )
    except Exception as e:
        print(f"Warning: Isometric view with forces creation failed: {e}")

    print(f"\nVideo processing completed!")
    if videos:
        print("Created videos:")
        for name, path in videos.items():
            print(f"  {name}: {path}")

    return videos


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Create visualization videos for a trained hover policy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "individual_dir",
        nargs='?',
        default="hover_evaluation",
        help="Directory containing individual.npy and policy.zip files"
    )

    parser.add_argument(
        "--difficulty",
        choices=["easy", "medium", "hard"],
        default="easy",
        help="Difficulty level used during training"
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
        "--steps",
        type=int,
        default=1200,
        help="Number of simulation steps to record"
    )

    parser.add_argument(
        "--view",
        choices=["top", "iso", "isometric"],
        default="iso",
        help="Camera view type"
    )

    parser.add_argument(
        "--single",
        action="store_true",
        help="Create only a single video instead of all views"
    )

    return parser.parse_args()


def main():
    """Main function for command line usage."""
    args = parse_args()

    try:
        if args.single:
            # Create single video
            video_path = create_hover_video(
                individual_dir=args.individual_dir,
                difficulty=args.difficulty,
                device=args.device,
                fps=args.fps,
                record_steps=args.steps,
                view_type=args.view,
                video_filename=f'hover_{args.view}_view.mp4'
            )
            print(f"Video created: {video_path}")
        else:
            # Create all videos
            videos = process_individual(
                individual_dir=args.individual_dir,
                difficulty=args.difficulty,
                device=args.device
            )
            print(f"Created {len(videos)} videos")

        print("Processing completed successfully!")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
