#!/usr/bin/env python3
"""
Visualize the trained gate navigation policy from curriculum learning.
Creates a video showing the drone navigating through gates.
"""

import numpy as np
import os
import sys
from stable_baselines3 import PPO

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from airevolve.evolution_tools.evaluators.drone_gate_env import DroneGateEnv
from airevolve.evolution_tools.evaluators import gate_train


def visualize_gate_policy(curriculum_dir='curriculum_learning_example',
                          gate_cfg='figure8',
                          output_video='gate_navigation.mp4',
                          record_steps=1200,
                          view_type='iso',
                          device='cpu'):
    """
    Load and visualize the trained gate navigation policy.

    Args:
        curriculum_dir: Directory containing curriculum learning results
        gate_cfg: Gate configuration used during training
        output_video: Path to save output video
        record_steps: Number of steps to record
        view_type: Camera view ('top', 'iso', 'isometric')
        device: Device ('cpu' or 'cuda:0')
    """
    # Load the individual (drone design)
    individual_path = os.path.join(curriculum_dir, "individual.npy")
    individual = np.load(individual_path)
    print(f"Loaded drone design from: {individual_path}")

    # Load the trained gate policy
    policy_path = os.path.join(curriculum_dir, "stage2_gate", "policy.zip")
    model = PPO.load(policy_path)
    print(f"Loaded gate policy from: {policy_path}")

    # Load gate configuration
    if gate_cfg == "figure8":
        gate_pos = gate_train.figure8.gate_pos
        gate_yaw = gate_train.figure8.gate_yaw
        start_pos = gate_train.figure8.starting_pos
        x_bounds = gate_train.figure8.x_bounds
        y_bounds = gate_train.figure8.y_bounds
        z_bounds = gate_train.figure8.z_bounds
    elif gate_cfg == "circle":
        gate_pos = gate_train.circle.gate_pos
        gate_yaw = gate_train.circle.gate_yaw
        start_pos = gate_train.circle.starting_pos
        x_bounds = gate_train.circle.x_bounds
        y_bounds = gate_train.circle.y_bounds
        z_bounds = gate_train.circle.z_bounds
    elif gate_cfg == "backandforth":
        gate_pos = gate_train.backandforth.gate_pos
        gate_yaw = gate_train.backandforth.gate_yaw
        start_pos = gate_train.backandforth.starting_pos
        x_bounds = gate_train.backandforth.x_bounds
        y_bounds = gate_train.backandforth.y_bounds
        z_bounds = gate_train.backandforth.z_bounds
    elif gate_cfg == "slalom":
        gate_pos = gate_train.slalom.gate_pos
        gate_yaw = gate_train.slalom.gate_yaw
        start_pos = gate_train.slalom.starting_pos
        x_bounds = gate_train.slalom.x_bounds
        y_bounds = gate_train.slalom.y_bounds
        z_bounds = gate_train.slalom.z_bounds
    else:
        raise ValueError(f"Invalid gate configuration: {gate_cfg}")

    print(f"Gate configuration: {gate_cfg}")
    print(f"Number of gates: {len(gate_pos)}")

    # Create gate environment
    env = DroneGateEnv(
        num_envs=1,
        individual=individual,
        gates_pos=gate_pos,
        gate_yaw=gate_yaw,
        start_pos=start_pos,
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        z_bounds=z_bounds,
        gates_ahead=1,
        num_state_history=0,
        num_action_history=0,
        history_step_size=1,
        render_mode=None,
        device=device
    )

    print(f"\nCreating video...")
    print(f"  Steps: {record_steps}")
    print(f"  View: {view_type}")
    print(f"  Output: {output_video}")

    # Animate the policy
    gate_train.animate_policy(
        individual=individual,
        model=model,
        env=env,
        deterministic=True,
        view_type=view_type,
        record=True,
        record_steps=record_steps,
        record_file=output_video,
        show_window=False
    )

    print(f"\n✓ Video saved to: {output_video}")


def main():
    """Create a video of the trained gate navigation policy."""

    visualize_gate_policy(
        curriculum_dir='curriculum_learning_example',
        gate_cfg='figure8',
        output_video='curriculum_gate_navigation.mp4',
        record_steps=1200,  # 12 seconds at 100Hz
        view_type='iso',
        device='cpu'
    )


if __name__ == "__main__":
    main()
