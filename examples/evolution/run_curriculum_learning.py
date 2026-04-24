#!/usr/bin/env python3
"""
Curriculum learning script that trains a drone in two stages:
1. Hover training (basic stabilization)
2. Gate navigation training (figure8 course with policy transfer)
"""

import numpy as np
import os
import sys
import time
import pickle
import torch
import json
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from airevolve.evolution_tools.evaluators.hover_train import FullStatsCallback, ProgressBarCallback
from airevolve.evolution_tools.evaluators import gate_train
from airevolve.evolution_tools.evaluators.drone_gate_env import DroneGateEnv
from airevolve.evolution_tools.evaluators.drone_hover_env import DroneHoverEnv
from stable_baselines3.common.callbacks import BaseCallback
from collections import deque


class ThresholdStoppingCallback(BaseCallback):
    """
    Callback that monitors hover success rate during training and stops early when threshold is met.

    This is more efficient than stopping to run separate test episodes:
    - Uses actual training episodes (100s of episodes vs 10 test episodes)
    - No interruption to training
    - More accurate with larger sample size
    - Faster overall
    """

    def __init__(self, success_threshold=0.90, window_size=100, check_freq=1000, verbose=1):
        """
        Args:
            success_threshold: Success rate threshold to stop training (0.0-1.0)
            window_size: Number of recent episodes to track (default: 100)
            check_freq: Check threshold every N steps (default: 1000)
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.success_threshold = success_threshold
        self.window_size = window_size
        self.check_freq = check_freq
        self.success_rates = deque(maxlen=window_size)
        self.threshold_met = False
        self.last_check = 0

    def _on_step(self) -> bool:
        """
        Called at every training step.

        Collects hover_success_rate from episode completions and checks threshold periodically.

        Returns:
            True to continue training, False to stop
        """
        # Collect success rates from completed episodes
        if len(self.locals.get("infos", [])) > 0:
            for info in self.locals["infos"]:
                if "hover_success_rate" in info:
                    self.success_rates.append(info["hover_success_rate"])

        # Check threshold periodically
        if self.num_timesteps - self.last_check >= self.check_freq:
            self.last_check = self.num_timesteps

            if len(self.success_rates) >= min(10, self.window_size):  # Need at least 10 episodes
                mean_success_rate = np.mean(self.success_rates)

                if self.verbose >= 1:
                    print(f"\n  [{self.num_timesteps:,} steps] Rolling success rate ({len(self.success_rates)} episodes): {mean_success_rate:.2%}")

                if mean_success_rate >= self.success_threshold:
                    self.threshold_met = True
                    if self.verbose >= 1:
                        print(f"  ✓ Threshold met ({mean_success_rate:.2%} >= {self.success_threshold:.1%})! Stopping training early.")
                    return False  # Stop training

        return True  # Continue training


class BestModelCallback(BaseCallback):
    """
    Callback that saves the best model based on episode reward.

    Only starts saving after a certain fraction of total timesteps to avoid
    saving early models before the policy has properly learned.
    """

    def __init__(self, save_dir, total_timesteps, start_fraction=0.75, verbose=1):
        """
        Args:
            save_dir: Directory to save best models
            total_timesteps: Total training timesteps
            start_fraction: Fraction of training to wait before saving (default: 0.75 = last 25%)
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.save_dir = save_dir
        self.total_timesteps = total_timesteps
        self.start_fraction = start_fraction
        self.start_timestep = int(total_timesteps * start_fraction)
        self.best_reward = -np.inf
        self.best_model_path = None
        self.saving_active = False
        self.episode_rewards = []

        # Create directory for best models
        os.makedirs(save_dir, exist_ok=True)

    def _on_step(self) -> bool:
        """
        Called at every training step.

        Saves model when a new best reward is achieved after start_timestep.

        Returns:
            True to continue training
        """
        # Activate saving once we reach the start_timestep
        if not self.saving_active and self.num_timesteps >= self.start_timestep:
            self.saving_active = True
            if self.verbose >= 1:
                print(f"\n  [BestModelCallback] Activated at {self.num_timesteps:,} steps (will save best models from now on)")

        # Collect episode rewards
        if len(self.locals.get("infos", [])) > 0:
            for info in self.locals["infos"]:
                # Episode info is marked with "episode" key in stable-baselines3
                if "episode" in info:
                    episode_reward = info["episode"]["r"]
                    self.episode_rewards.append(episode_reward)

                    # Check if this is a new best (only if saving is active)
                    if self.saving_active and episode_reward > self.best_reward:
                        self.best_reward = episode_reward

                        # Save the model
                        model_name = f"best_model_reward_{episode_reward:.2f}_step_{self.num_timesteps}"
                        model_path = os.path.join(self.save_dir, model_name)
                        self.model.save(model_path)
                        self.best_model_path = model_path + ".zip"

                        if self.verbose >= 1:
                            print(f"\n  [BestModelCallback] New best reward: {episode_reward:.2f} at {self.num_timesteps:,} steps")
                            print(f"                      Saved to: {model_name}.zip")

        return True  # Continue training


class GateProgressCallback(BaseCallback):
    """
    Callback that monitors gate passing performance during training.

    Tracks how many gates are passed per episode to show learning progress.
    """

    def __init__(self, window_size=100, check_freq=10000, save_dir=None, verbose=1):
        """
        Args:
            window_size: Number of recent episodes to track (default: 100)
            check_freq: Report progress every N steps (default: 10000)
            save_dir: Directory to save gates_passed.csv (optional)
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.window_size = window_size
        self.check_freq = check_freq
        self.save_dir = save_dir
        self.gates_passed = deque(maxlen=window_size)
        self.last_check = 0

        # Store all gates passed data for saving
        self.all_gates_passed = []
        self.all_timesteps = []

        # Initialize CSV file if save_dir is provided
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)
            self.csv_path = os.path.join(self.save_dir, "gates_passed.csv")
            with open(self.csv_path, 'w') as f:
                f.write("timestep,gates_passed\n")

    def _on_step(self) -> bool:
        """
        Called at every training step.

        Collects num_gates_passed from episode completions and reports periodically.

        Returns:
            True to continue training
        """
        # Collect gates passed from completed episodes
        if len(self.locals.get("infos", [])) > 0:
            for info in self.locals["infos"]:
                if "num_gates_passed" in info:
                    # num_gates_passed is an array, get first element
                    gates = info["num_gates_passed"]
                    if hasattr(gates, '__iter__'):
                        gates_value = gates[0]
                    else:
                        gates_value = gates

                    self.gates_passed.append(gates_value)

                    # Save to persistent storage
                    self.all_gates_passed.append(gates_value)
                    self.all_timesteps.append(self.num_timesteps)

                    # Write to CSV file immediately if save_dir is set
                    if self.save_dir is not None:
                        with open(self.csv_path, 'a') as f:
                            f.write(f"{self.num_timesteps},{gates_value}\n")

        # Report progress periodically
        if self.num_timesteps - self.last_check >= self.check_freq:
            self.last_check = self.num_timesteps

            if len(self.gates_passed) >= 10:  # Need at least 10 episodes
                mean_gates = np.mean(self.gates_passed)
                max_gates = np.max(self.gates_passed)

                if self.verbose >= 1:
                    print(f"\n  [{self.num_timesteps:,} steps] Gates passed per episode ({len(self.gates_passed)} episodes): avg={mean_gates:.1f}, max={max_gates}")

        return True  # Continue training


# Stage 1: Hover Training with Threshold-based Early Stopping
def stage1_hover_training(individual, save_dir, max_timesteps, num_envs, device,
                         success_threshold=0.90, window_size=100, check_freq=10000):
    """
    Train hovering policy from scratch with early stopping based on performance threshold.

    Uses a rolling window of episode success rates during training to determine when to stop,
    which is faster and more accurate than separate test evaluations.

    Args:
        individual: Drone design array
        save_dir: Directory to save Stage 1 results
        max_timesteps: Maximum training timesteps for hover stage
        num_envs: Number of parallel environments
        device: Device ('cpu' or 'cuda:0')
        success_threshold: Hover success rate threshold to advance to stage 2 (default: 0.90)
        window_size: Number of recent episodes to track (default: 100)
        check_freq: Check threshold every N steps (default: 10000)

    Returns:
        hover_success_rate: Final performance metric from hover training
        model_path: Path to saved hover policy
        actual_timesteps: Actual timesteps trained (may be less than max if threshold met)
    """
    target_pos = np.array([0.0, 0.0, -1.5], dtype=np.float32)

    # Create hover environment
    train_env = DroneHoverEnv(
        num_envs=num_envs,
        individual=individual,
        target_pos=target_pos,
        x_bounds=[-5, 5],
        y_bounds=[-5, 5],
        z_bounds=[-5, 0],
        position_tolerance=0.5,
        velocity_tolerance=0.3,
        motor_limit=1.0,  # Limit to bottom half of thrust range for better learning
        num_state_history=0,
        num_action_history=0,
        history_step_size=1,
        render_mode=None,
        device=device,
        add_disturbances=False,
        disturbance_strength=0.0
    )

    # Wrap training environment in monitor
    monitor_file = save_dir
    train_env = VecMonitor(train_env, filename=monitor_file)

    # Create PPO model
    policy_kwargs = dict(
        activation_fn=torch.nn.ReLU,
        net_arch=dict(pi=[64, 64, 64], vf=[64, 64, 64]),
        log_std_init=0.041
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        policy_kwargs=policy_kwargs,
        verbose=0,
        tensorboard_log=save_dir,
        learning_rate=1.0e-4,
        n_steps=256,
        batch_size=128,
        n_epochs=22,
        gamma=0.9965,
        gae_lambda=0.894,
        clip_range=0.156,
        ent_coef=0.0044,
        vf_coef=0.55,
        max_grad_norm=3.19,
        device=device
    )

    # Setup callbacks
    progress_callback = ProgressBarCallback(max_timesteps, num_envs)
    threshold_callback = ThresholdStoppingCallback(
        success_threshold=success_threshold,
        window_size=window_size,
        check_freq=check_freq,
        verbose=1
    )
    stats_callback = FullStatsCallback()

    callbacks = [progress_callback, threshold_callback, stats_callback]

    print(f"  Threshold: {success_threshold:.1%} | Max steps: {max_timesteps:,}")
    print(f"  Monitoring: Rolling window of {window_size} episodes, checking every {check_freq:,} steps")

    # Train with automatic early stopping
    model.learn(
        total_timesteps=max_timesteps,
        reset_num_timesteps=True,
        log_interval=100,
        callback=callbacks
    )

    # Get actual timesteps trained (may be less than max if stopped early)
    actual_timesteps = model.num_timesteps

    # Calculate final success rate from the rolling window
    if len(threshold_callback.success_rates) > 0:
        hover_success_rate = np.mean(threshold_callback.success_rates)
    else:
        hover_success_rate = 0.0

    # Save final model
    model_path = os.path.join(save_dir, "policy.zip")
    model.save(os.path.join(save_dir, "policy"))

    # Save metrics
    metrics = {
        "hover_success_rate": hover_success_rate,
        "actual_timesteps": actual_timesteps,
        "max_timesteps": max_timesteps,
        "threshold_met": threshold_callback.threshold_met,
        "num_episodes_tracked": len(threshold_callback.success_rates)
    }
    np.save(os.path.join(save_dir, "hover_metrics.npy"), metrics)

    return hover_success_rate, model_path, actual_timesteps


# Stage 2: Gate Navigation Training (with transfer)
def stage2_gate_training(individual, hover_model_path, save_dir,
                         timesteps, num_envs, gate_cfg, device,
                         window_size=100, check_freq=10000):
    """
    Load hover policy and continue training on gate navigation.

    Args:
        individual: Drone design array
        hover_model_path: Path to trained hover policy
        save_dir: Directory to save Stage 2 results
        timesteps: Training timesteps for gate stage
        num_envs: Number of parallel environments
        gate_cfg: Gate configuration ('figure8', 'circle', etc.)
        device: Device ('cpu' or 'cuda:0')
        window_size: Number of recent episodes to track (default: 100)
        check_freq: Report progress every N steps (default: 10000)

    Returns:
        num_gates_passed: Number of gates passed in final evaluation
    """
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

    # Load the hover policy
    print(f"Loading hover policy from: {hover_model_path}")
    hover_model = PPO.load(hover_model_path)

    # Create gate environment
    env = DroneGateEnv(
        num_envs=num_envs,
        individual=individual,
        gates_pos=gate_pos,
        gate_yaw=gate_yaw,
        start_pos=start_pos,
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        z_bounds=z_bounds,
        gates_ahead=1,
        motor_limit=1.0,  # Limit to bottom half of thrust range for better learning
        num_state_history=0,
        num_action_history=0,
        history_step_size=1,
        render_mode=None,
        device=device
    )

    # Wrap in monitor
    monitor_file = save_dir
    env = VecMonitor(env, filename=monitor_file)

    # Create new model with same hyperparameters
    policy_kwargs = dict(
        activation_fn=torch.nn.ReLU,
        net_arch=dict(pi=[64, 64, 64], vf=[64, 64, 64]),
        log_std_init=0.041
    )

    gate_model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=0,
        tensorboard_log=save_dir,
        learning_rate=1.0e-4,
        n_steps=256,
        batch_size=128,
        n_epochs=22,
        gamma=0.9965,
        gae_lambda=0.894,
        clip_range=0.156,
        ent_coef=0.0044,
        vf_coef=0.55,
        max_grad_norm=3.19,
        device=device
    )

    # Transfer weights from hover policy (actor network only)
    # Value function (critic) and log_std (exploration) are reset for gate-specific learning
    gate_model.policy.action_net.load_state_dict(hover_model.policy.action_net.state_dict())
    print("Successfully transferred hover policy (actor) weights to gate model")
    print("Value function (critic) and exploration (log_std) reset for gate-specific learning")

    # Setup callbacks for gate training
    progress_callback = ProgressBarCallback(timesteps, num_envs)
    gate_progress_callback = GateProgressCallback(
        window_size=window_size,
        check_freq=check_freq,
        save_dir=save_dir,  # Save gates_passed data to CSV
        verbose=1
    )
    best_model_callback = BestModelCallback(
        save_dir=os.path.join(save_dir, "best_models"),
        total_timesteps=timesteps,
        start_fraction=0.75,  # Start saving in last 25% of training
        verbose=1
    )
    stats_callback = FullStatsCallback()

    callbacks = [progress_callback, gate_progress_callback, best_model_callback, stats_callback]

    print(f"  Monitoring: Rolling window of {window_size} episodes, checking every {check_freq:,} steps")
    print(f"  Best model saving: Will start at {int(timesteps * 0.75):,} steps (last 25% of training)")

    # Continue training on gate task
    gate_model.learn(
        total_timesteps=timesteps,
        reset_num_timesteps=False,
        log_interval=100,
        callback=callbacks
    )

    # Save the gate policy
    gate_model.save(os.path.join(save_dir, "policy"))

    # Get final performance metrics from training
    if len(gate_progress_callback.gates_passed) > 0:
        avg_gates_passed = np.mean(gate_progress_callback.gates_passed)
        max_gates_passed = np.max(gate_progress_callback.gates_passed)
    else:
        avg_gates_passed = 0.0
        max_gates_passed = 0

    # Evaluate performance - run test episodes
    test_env = DroneGateEnv(
        num_envs=1,
        individual=individual,
        gates_pos=gate_pos,
        gate_yaw=gate_yaw,
        start_pos=start_pos,
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        z_bounds=z_bounds,
        gates_ahead=1,
        motor_limit=1.0,  # Limit to bottom half of thrust range for better learning
        num_state_history=0,
        num_action_history=0,
        history_step_size=1,
        render_mode=None,
        device=device
    )

    # Run evaluation
    test_env.reset()
    for _ in range(1000):
        actions, _ = gate_model.predict(test_env.states, deterministic=True)
        _, _, _, infos = test_env.step(actions)

    final_gates_passed = infos[0]["num_gates_passed"][0]

    print(f"  Training performance: avg={avg_gates_passed:.1f} gates, max={max_gates_passed}")
    print(f"  Final evaluation: {final_gates_passed} gates passed")

    # Return results including best model info
    results = {
        "final_gates_passed": final_gates_passed,
        "avg_gates_passed": avg_gates_passed,
        "max_gates_passed": max_gates_passed,
        "best_model_reward": best_model_callback.best_reward,
        "best_model_path": best_model_callback.best_model_path
    }

    if best_model_callback.best_model_path is not None:
        print(f"  Best model saved: {os.path.basename(best_model_callback.best_model_path)}")
        print(f"  Best reward achieved: {best_model_callback.best_reward:.2f}")

    return results


def load_monitor_data(monitor_dir):
    """
    Load episode data from VecMonitor CSV file.

    Args:
        monitor_dir: Directory containing monitor.csv file

    Returns:
        DataFrame with episode rewards, lengths, and timestamps
    """
    monitor_file = os.path.join(monitor_dir, "monitor.csv")

    if not os.path.exists(monitor_file):
        print(f"Warning: Monitor file not found at {monitor_file}")
        return None

    # Read monitor CSV (skip first line which contains metadata)
    try:
        df = pd.read_csv(monitor_file, skiprows=1)
        return df
    except Exception as e:
        print(f"Error loading monitor data from {monitor_file}: {e}")
        return None


def plot_rewards(output_dir, stage1_dir, stage2_dir, stage1_timesteps):
    """
    Plot episode rewards over time for both training stages.

    Args:
        output_dir: Directory to save plots
        stage1_dir: Stage 1 (hover) training directory
        stage2_dir: Stage 2 (gate) training directory
        stage1_timesteps: Number of timesteps in stage 1
    """
    print("\n" + "=" * 60)
    print("GENERATING REWARD PLOTS")
    print("=" * 60)

    # Load data from both stages
    stage1_data = load_monitor_data(stage1_dir)
    stage2_data = load_monitor_data(stage2_dir)

    if stage1_data is None and stage2_data is None:
        print("No monitor data found. Skipping plots.")
        return

    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Plot Stage 1 (Hover Training)
    if stage1_data is not None and len(stage1_data) > 0:
        # Calculate cumulative timesteps
        stage1_data['cumulative_timesteps'] = stage1_data['l'].cumsum()

        ax = axes[0]
        ax.plot(stage1_data['cumulative_timesteps'], stage1_data['r'],
                alpha=0.3, linewidth=0.5, color='blue', label='Episode Reward')

        # Add rolling average
        window = min(100, len(stage1_data) // 10)
        if window > 0:
            rolling_mean = stage1_data['r'].rolling(window=window, min_periods=1).mean()
            ax.plot(stage1_data['cumulative_timesteps'], rolling_mean,
                   linewidth=2, color='darkblue', label=f'Rolling Mean ({window} episodes)')

        ax.set_xlabel('Timesteps')
        ax.set_ylabel('Episode Reward')
        ax.set_title('Stage 1: Hover Training')
        ax.legend()
        ax.grid(True, alpha=0.3)

        print(f"Stage 1: {len(stage1_data)} episodes, avg reward: {stage1_data['r'].mean():.2f}")
    else:
        axes[0].text(0.5, 0.5, 'No Stage 1 data available',
                    ha='center', va='center', transform=axes[0].transAxes)
        axes[0].set_title('Stage 1: Hover Training')

    # Plot Stage 2 (Gate Training)
    if stage2_data is not None and len(stage2_data) > 0:
        # Calculate cumulative timesteps (offset by stage1 timesteps)
        stage2_data['cumulative_timesteps'] = stage1_timesteps + stage2_data['l'].cumsum()

        ax = axes[1]
        ax.plot(stage2_data['cumulative_timesteps'], stage2_data['r'],
                alpha=0.3, linewidth=0.5, color='green', label='Episode Reward')

        # Add rolling average
        window = min(100, len(stage2_data) // 10)
        if window > 0:
            rolling_mean = stage2_data['r'].rolling(window=window, min_periods=1).mean()
            ax.plot(stage2_data['cumulative_timesteps'], rolling_mean,
                   linewidth=2, color='darkgreen', label=f'Rolling Mean ({window} episodes)')

        ax.set_xlabel('Timesteps')
        ax.set_ylabel('Episode Reward')
        ax.set_title('Stage 2: Gate Navigation Training')
        ax.legend()
        ax.grid(True, alpha=0.3)

        print(f"Stage 2: {len(stage2_data)} episodes, avg reward: {stage2_data['r'].mean():.2f}")
    else:
        axes[1].text(0.5, 0.5, 'No Stage 2 data available',
                    ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title('Stage 2: Gate Navigation Training')

    # Plot Combined View
    ax = axes[2]

    if stage1_data is not None and len(stage1_data) > 0:
        ax.plot(stage1_data['cumulative_timesteps'], stage1_data['r'],
                alpha=0.3, linewidth=0.5, color='blue', label='Stage 1 Episodes')

        window = min(100, len(stage1_data) // 10)
        if window > 0:
            rolling_mean = stage1_data['r'].rolling(window=window, min_periods=1).mean()
            ax.plot(stage1_data['cumulative_timesteps'], rolling_mean,
                   linewidth=2, color='darkblue', label='Stage 1 Mean')

    if stage2_data is not None and len(stage2_data) > 0:
        ax.plot(stage2_data['cumulative_timesteps'], stage2_data['r'],
                alpha=0.3, linewidth=0.5, color='green', label='Stage 2 Episodes')

        window = min(100, len(stage2_data) // 10)
        if window > 0:
            rolling_mean = stage2_data['r'].rolling(window=window, min_periods=1).mean()
            ax.plot(stage2_data['cumulative_timesteps'], rolling_mean,
                   linewidth=2, color='darkgreen', label='Stage 2 Mean')

        # Add vertical line at stage transition
        ax.axvline(x=stage1_timesteps, color='red', linestyle='--',
                  linewidth=2, alpha=0.7, label='Stage Transition')

    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Episode Reward')
    ax.set_title('Combined: Full Curriculum Learning')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Save plot
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'reward_curves.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved reward plot to: {plot_path}")
    plt.close()

    print("=" * 60)


def plot_gates_passed(output_dir, stage2_dir):
    """
    Plot gates passed over time during gate navigation training.

    Args:
        output_dir: Directory to save plots
        stage2_dir: Stage 2 (gate) training directory
    """
    print("\n" + "=" * 60)
    print("GENERATING GATES PASSED PLOT")
    print("=" * 60)

    gates_csv_path = os.path.join(stage2_dir, "gates_passed.csv")

    if not os.path.exists(gates_csv_path):
        print(f"No gates_passed.csv found at {gates_csv_path}. Skipping plot.")
        print("=" * 60)
        return

    # Load gates passed data
    try:
        gates_data = pd.read_csv(gates_csv_path)
        print(f"Loaded {len(gates_data)} episode records from gates_passed.csv")
    except Exception as e:
        print(f"Error loading gates data: {e}")
        print("=" * 60)
        return

    if len(gates_data) == 0:
        print("No data in gates_passed.csv. Skipping plot.")
        print("=" * 60)
        return

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # Plot raw episode data
    ax.plot(gates_data['timestep'], gates_data['gates_passed'],
            alpha=0.3, linewidth=0.5, color='green', label='Episode Gates Passed')

    # Add rolling average
    window = min(100, len(gates_data) // 10)
    if window > 0:
        rolling_mean = gates_data['gates_passed'].rolling(window=window, min_periods=1).mean()
        ax.plot(gates_data['timestep'], rolling_mean,
               linewidth=2, color='darkgreen', label=f'Rolling Mean ({window} episodes)')

    # Add maximum achieved line
    max_gates = gates_data['gates_passed'].max()
    ax.axhline(y=max_gates, color='red', linestyle='--',
              linewidth=1.5, alpha=0.7, label=f'Max Gates: {int(max_gates)}')

    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Gates Passed')
    ax.set_title('Gate Navigation Training Progress')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Save plot
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'gates_passed.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved gates passed plot to: {plot_path}")
    plt.close()

    # Print statistics
    mean_gates = gates_data['gates_passed'].mean()
    final_100_mean = gates_data['gates_passed'].tail(100).mean() if len(gates_data) >= 100 else mean_gates
    print(f"Overall average: {mean_gates:.2f} gates")
    print(f"Final 100 episodes average: {final_100_mean:.2f} gates")
    print(f"Maximum achieved: {int(max_gates)} gates")

    print("=" * 60)


# Main curriculum learning function
def run_curriculum_learning(individual,
                           output_dir='curriculum_learning_results',
                           hover_max_timesteps=int(1e7),
                           hover_success_threshold=0.90,
                           hover_window_size=100,
                           hover_check_freq=10000,
                           gate_timesteps=int(1e7),
                           gate_window_size=100,
                           gate_check_freq=10000,
                           num_envs=100,
                           gate_cfg='figure8',
                           device='cpu'):
    """
    Run full curriculum learning pipeline with threshold-based progression.

    Stage 1 uses a rolling window of training episode success rates to determine
    when to advance, which is faster and more accurate than separate test evaluations.

    Args:
        individual: Drone design array
        output_dir: Base directory for results
        hover_max_timesteps: Maximum training timesteps for Stage 1 (hover)
        hover_success_threshold: Success rate threshold to advance to Stage 2 (default: 0.90)
        hover_window_size: Number of recent episodes to track in Stage 1 (default: 100)
        hover_check_freq: Check threshold every N steps in Stage 1 (default: 10000)
        gate_timesteps: Training timesteps for Stage 2 (gate)
        gate_window_size: Number of recent episodes to track in Stage 2 (default: 100)
        gate_check_freq: Report progress every N steps in Stage 2 (default: 10000)
        num_envs: Number of parallel environments
        gate_cfg: Gate configuration for Stage 2
        device: Device ('cpu' or 'cuda:0')

    Returns:
        dict: Results from both stages
    """

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    stage1_dir = os.path.join(output_dir, 'stage1_hover')
    stage2_dir = os.path.join(output_dir, 'stage2_gate')
    os.makedirs(stage1_dir, exist_ok=True)
    os.makedirs(stage2_dir, exist_ok=True)

    # Save individual
    np.save(os.path.join(output_dir, "individual.npy"), individual)

    print("=" * 60)
    print("CURRICULUM LEARNING: Two-Stage Training")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Stage 1: Hover training (max {hover_max_timesteps:,} steps, threshold={hover_success_threshold:.1%})")
    print(f"Stage 2: Gate training ({gate_timesteps:,} steps)")
    print(f"Gate configuration: {gate_cfg}")
    print("=" * 60)

    # STAGE 1: HOVER TRAINING
    print("\n[STAGE 1] Starting hover training...")
    stage1_start = time.time()

    hover_success_rate, hover_model_path, actual_hover_timesteps = stage1_hover_training(
        individual=individual,
        save_dir=stage1_dir,
        max_timesteps=hover_max_timesteps,
        num_envs=num_envs,
        device=device,
        success_threshold=hover_success_threshold,
        window_size=hover_window_size,
        check_freq=hover_check_freq
    )

    stage1_duration = time.time() - stage1_start
    print(f"[STAGE 1] Completed in {stage1_duration:.2f}s")
    print(f"[STAGE 1] Trained for {actual_hover_timesteps:,} steps (saved {hover_max_timesteps - actual_hover_timesteps:,} steps)")
    print(f"[STAGE 1] Hover success rate: {hover_success_rate:.2%}")

    # STAGE 2: GATE TRAINING (with transfer)
    print("\n[STAGE 2] Starting gate training with policy transfer...")
    stage2_start = time.time()

    stage2_results = stage2_gate_training(
        individual=individual,
        hover_model_path=hover_model_path,
        save_dir=stage2_dir,
        timesteps=gate_timesteps,
        num_envs=num_envs,
        gate_cfg=gate_cfg,
        device=device,
        window_size=gate_window_size,
        check_freq=gate_check_freq
    )

    stage2_duration = time.time() - stage2_start
    print(f"[STAGE 2] Completed in {stage2_duration:.2f}s")
    print(f"[STAGE 2] Gates passed: {stage2_results['final_gates_passed']}")

    # Save overall results
    results = {
        'stage1_hover_success_rate': hover_success_rate,
        'stage1_actual_timesteps': actual_hover_timesteps,
        'stage1_max_timesteps': hover_max_timesteps,
        'stage1_threshold': hover_success_threshold,
        'stage1_threshold_met': hover_success_rate >= hover_success_threshold,
        'stage1_duration': stage1_duration,
        'stage2_gates_passed': stage2_results['final_gates_passed'],
        'stage2_avg_gates_passed': stage2_results['avg_gates_passed'],
        'stage2_max_gates_passed': stage2_results['max_gates_passed'],
        'stage2_best_model_reward': stage2_results['best_model_reward'],
        'stage2_best_model_path': stage2_results['best_model_path'],
        'stage2_timesteps': gate_timesteps,
        'stage2_duration': stage2_duration,
        'total_duration': stage1_duration + stage2_duration,
        'gate_cfg': gate_cfg
    }

    with open(os.path.join(output_dir, "curriculum_results.pkl"), 'wb') as f:
        pickle.dump(results, f)

    print("\n" + "=" * 60)
    print("CURRICULUM LEARNING COMPLETE")
    print("=" * 60)
    print(f"Total duration: {results['total_duration']:.2f}s")
    print(f"Stage 1 hover success: {hover_success_rate:.2%} ({actual_hover_timesteps:,} steps)")
    print(f"Stage 2 gates passed: {results['stage2_gates_passed']}")
    if results['stage2_best_model_path']:
        print(f"Best model: {os.path.basename(results['stage2_best_model_path'])} (reward: {results['stage2_best_model_reward']:.2f})")

    # Generate plots
    plot_rewards(output_dir, stage1_dir, stage2_dir, actual_hover_timesteps)
    plot_gates_passed(output_dir, stage2_dir)

    return results


def main():
    """Example usage with a simple quadcopter design."""

    # Example quadcopter in X configuration
    example_individual = np.array([
        [0.07,  np.pi/4,   0.0,  0.0,  0.0,  1.0],  # Front-right (CW)
        [0.07, -np.pi/4,   0.0,  0.0,  0.0,  0.0],  # Front-left (CCW)
        [0.07,  3*np.pi/4, 0.0,  0.0,  0.0,  0.0],  # Back-left (CCW)
        [0.07, -3*np.pi/4, 0.0,  0.0,  0.0,  1.0],  # Back-right (CW)
        # [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        # [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    ])

    # Run curriculum learning with threshold-based progression
    results = run_curriculum_learning(
        individual=example_individual,
        output_dir='curriculum_learning_example',
        hover_max_timesteps=int(10),#2e6),       # Max 2M steps for hover
        hover_success_threshold=0.90,            # Advance when 90% success rate achieved
        hover_window_size=500,                   # Track last 500 episodes (stage 1)
        hover_check_freq=50_000,                 # Check threshold every 50K steps (stage 1)
        gate_timesteps=int(1e7),               # 25M steps for gate
        gate_window_size=100,                    # Track last 100 episodes (stage 2)
        gate_check_freq=100_000,                  # Report every 50K steps (stage 2)
        num_envs=100,
        gate_cfg='figure8',
        device='cpu'
    )

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    if results['stage1_threshold_met']:
        print(f"Stage 1: Reached {results['stage1_hover_success_rate']:.1%} in {results['stage1_actual_timesteps']:,} steps")
        print(f"         (Stopped early, saved {results['stage1_max_timesteps'] - results['stage1_actual_timesteps']:,} steps)")
    else:
        print(f"Stage 1: Reached {results['stage1_hover_success_rate']:.1%} in {results['stage1_actual_timesteps']:,} steps")
        print(f"         (Did not meet {results['stage1_threshold']:.1%} threshold)")
    print(f"Stage 2: {results['stage2_gates_passed']} gates passed")
    print("=" * 60)


if __name__ == "__main__":
    main()
