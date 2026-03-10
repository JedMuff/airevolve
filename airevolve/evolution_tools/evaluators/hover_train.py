# library imports
import os
import sys
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import warnings
from stable_baselines3 import PPO
from datetime import datetime
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from tqdm import tqdm

# Suppress the render_mode warning from stable_baselines3
warnings.filterwarnings("ignore", message="The `render_mode` attribute is not defined in your environment")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from airevolve.evolution_tools.evaluators.drone_hover_env import DroneHoverEnv
from airevolve.simulator.visualization.animation import view as animation_view
from airevolve.evolution_tools.inspection_tools.morphological_descriptors.hovering_info import get_sim
from airevolve.evolution_tools.inspection_tools.drone_visualizer import DroneVisualizer

import argparse


# ANIMATION FUNCTION
def animate_hover_policy(individual, model, env, deterministic=False, log=None, view_type="top",
                    motor_colors=['red', 'blue', 'green', 'orange', 'purple', 'brown'], **kwargs):
    """
    Animate a trained hover policy.

    Args:
        individual: Drone design array
        model: Trained PPO model
        env: Hover environment instance
        deterministic: Whether to use deterministic actions
        log: Optional logging function
        view_type: Camera view type ('top', 'iso', 'isometric')
        motor_colors: List of color names for motors
        **kwargs: Additional arguments passed to animation_view (e.g., record, record_steps, fps, etc.)
    """
    env.reset()

    # Convert individual to propellers configuration
    propellers, _ = env._convert_individual_to_propellers(individual)

    def get_drone_state():
        actions, _ = model.predict(env.states, deterministic=deterministic)

        states, rewards, dones, infos = env.step(actions)
        if log != None:
            log(states)

        # Return drone state in the format expected by the view function
        world_state = env.world_states[0]  # Get first environment
        num_motors = len(propellers)
        drone_state = {
            'x': world_state[0],
            'y': world_state[1],
            'z': world_state[2],
            'phi': world_state[6],
            'theta': world_state[7],
            'psi': world_state[8]
        }

        # Add motor thrust values (convert from [-1,1] to [0,1])
        for i in range(num_motors):
            if i < env.actions.shape[1]:
                # Actions are in [-1,1], convert to [0,1] for visualization
                drone_state[f'u{i+1}'] = (env.actions[0][i] + 1) / 2
            else:
                drone_state[f'u{i+1}'] = 0

        return drone_state

    # For hover visualization, we can add a marker at the target position
    animation_view(propellers, get_drone_state, view_type=view_type, motor_colors=motor_colors, **kwargs)


class ProgressBarCallback(BaseCallback):
    """Callback to display a progress bar during training."""
    def __init__(self, total_timesteps, num_envs):
        super().__init__()
        self.pbar = None
        self.total_timesteps = total_timesteps
        self.num_envs = num_envs

    def _on_training_start(self):
        self.pbar = tqdm(total=self.total_timesteps, desc="Training", unit="steps")

    def _on_step(self):
        # Update by num_envs because each step processes all environments
        self.pbar.update(self.num_envs)
        # Update postfix with current metrics if available
        if 'rollout/ep_rew_mean' in self.logger.name_to_value:
            self.pbar.set_postfix({
                'reward': f"{self.logger.name_to_value['rollout/ep_rew_mean']:.2f}",
                'fps': f"{self.logger.name_to_value.get('time/fps', 0):.0f}"
            })
        return True

    def _on_training_end(self):
        self.pbar.close()


class FullStatsCallback(BaseCallback):
    """Callback to log all training statistics."""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.tags = [
            'rollout/ep_len_mean', 'rollout/ep_rew_mean', 'time/fps',
            'train/approx_kl', 'train/clip_fraction', 'train/clip_range',
            'train/entropy_loss', 'train/explained_variance', 'train/learning_rate',
            'train/loss', 'train/policy_gradient_loss', 'train/std', 'train/value_loss'
        ]

    def _on_step(self) -> bool:
        for tag in self.tags:
            if tag in self.logger.name_to_value:
                self.logger.record(f"monitor/{tag}", self.logger.name_to_value[tag])
        return True

    def _on_rollout_end(self) -> None:
        # Force flush for debugging; can remove later
        self.logger.dump(self.num_timesteps)


def train(individual,
          target_pos=np.array([0.0, 0.0, -1.5], dtype=np.float32),
          total_timesteps=int(1E7),
          save_dir="./logs",
          num_envs=100,
          device="cpu",
          num=None,
          difficulty="easy",
          position_tolerance=0.5,
          velocity_tolerance=0.3):
    """
    Train a drone to hover at a target position.

    Args:
        individual: Drone design array
        target_pos: Target hover position [x, y, z]
        total_timesteps: Total training timesteps
        save_dir: Directory to save results
        num_envs: Number of parallel environments
        device: Device to run on (cpu/cuda)
        num: Optional number for multiple runs
        difficulty: Difficulty level ("easy", "medium", "hard")
        position_tolerance: Position tolerance for hover success
        velocity_tolerance: Velocity tolerance for hover success
    """

    # Configure difficulty
    if difficulty == "easy":
        add_disturbances = False
        disturbance_strength = 0.0
        x_bounds = [-5, 5]
        y_bounds = [-5, 5]
        z_bounds = [-5, 0]
    elif difficulty == "medium":
        add_disturbances = True
        disturbance_strength = 0.05
        x_bounds = [-4, 4]
        y_bounds = [-4, 4]
        z_bounds = [-4, 0]
    elif difficulty == "hard":
        add_disturbances = True
        disturbance_strength = 0.1
        x_bounds = [-3, 3]
        y_bounds = [-3, 3]
        z_bounds = [-3, 0]
        position_tolerance = 0.3
        velocity_tolerance = 0.2
    else:
        raise ValueError("Invalid difficulty level. Choose 'easy', 'medium', or 'hard'")

    # SETUP LOGGING
    save_dir = save_dir+"/"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    env = DroneHoverEnv(
        num_envs=num_envs,
        individual=individual,
        target_pos=target_pos,
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        z_bounds=z_bounds,
        position_tolerance=position_tolerance,
        velocity_tolerance=velocity_tolerance,
        num_state_history=0,
        num_action_history=0,
        history_step_size=1,
        render_mode=None,
        device=device,
        add_disturbances=add_disturbances,
        disturbance_strength=disturbance_strength
    )

    test_env = DroneHoverEnv(
        num_envs=1,
        individual=individual,
        target_pos=target_pos,
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        z_bounds=z_bounds,
        position_tolerance=position_tolerance,
        velocity_tolerance=velocity_tolerance,
        num_state_history=0,
        num_action_history=0,
        history_step_size=1,
        render_mode=None,
        device=device,
        add_disturbances=False,  # No disturbances for testing
        disturbance_strength=0.0
    )

    # Wrap the environment in a Monitor wrapper
    if num is not None:
        monitor_file = save_dir+f"m{num}"
    else:
        monitor_file = save_dir

    env = VecMonitor(env, filename=monitor_file)

    # MODEL DEFINITION
    policy_kwargs = dict(
        activation_fn=torch.nn.ReLU,
        net_arch=dict(pi=[64, 64, 64], vf=[64, 64, 64]),
        log_std_init=0.041  # Matched to gate training parameters for curriculum learning
    )
    model = PPO(
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
        ent_coef=0.0044,  # Matched to gate training parameters for curriculum learning
        vf_coef=0.55,
        max_grad_norm=3.19,
        device=device
    )

    # TRAINING
    callbacks = [ProgressBarCallback(total_timesteps, num_envs), FullStatsCallback()]
    model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False, log_interval=100, callback=callbacks)
    if num is None:
        model.save(save_dir + '/' + "policy")
    else:
        model.save(save_dir + '/' + f"policy{num}")

    # Plotting the training curve
    try:
        data = pd.read_csv(monitor_file+".monitor.csv", skiprows=1)  # Skip the first row (comments)
    except:
        data = pd.read_csv(monitor_file+"monitor.csv", skiprows=1)
    episode_rewards = data["r"]  # Rewards per episode
    time_steps = data["t"]  # Timesteps at each episode
    plt.figure(figsize=(10, 6))
    plt.plot(time_steps, episode_rewards, label="Episode Reward")
    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    plt.title("Hover Training: Reward per Episode")
    plt.legend()
    if num is None:
        plt.savefig(save_dir+"/training_curve.png")
    else:
        plt.savefig(save_dir+f"/training_curve{num}.png")
    plt.close()

    # TESTING - evaluate hover performance
    test_env.reset()
    hover_success_steps_list = []
    total_hover_time_list = []
    hover_success_rate_list = []
    episode_lengths = []
    out_of_bounds_count = 0

    # Run 10 test episodes
    for episode in range(10):
        test_env.reset()
        for step in range(1200): # 12 seconds at 100Hz
            actions, _ = model.predict(test_env.states, deterministic=True)
            states, rewards, dones, infos = test_env.step(actions)
            if dones[0]:
                hover_success_steps_list.append(infos[0].get("hover_success_steps", 0))
                total_hover_time_list.append(infos[0].get("total_hover_time", 0.0))
                hover_success_rate_list.append(infos[0].get("hover_success_rate", 0.0))
                episode_lengths.append(step + 1)
                if infos[0].get("out_of_bounds", False):
                    out_of_bounds_count += 1
                break

    # Calculate metrics
    avg_hover_success_steps = np.mean(hover_success_steps_list)
    avg_hover_time = np.mean(total_hover_time_list)
    avg_hover_success_rate = np.mean(hover_success_rate_list)
    avg_episode_length = np.mean(episode_lengths)

    print(f"Test Results (10 episodes):")
    print(f"  Average hover success steps: {avg_hover_success_steps:.1f}")
    print(f"  Average hover time: {avg_hover_time:.2f}s")
    print(f"  Average hover success rate: {avg_hover_success_rate:.2%}")
    print(f"  Average episode length: {avg_episode_length:.1f} steps")
    print(f"  Out of bounds episodes: {out_of_bounds_count}/10")

    # Save metrics
    metrics = {
        "avg_hover_success_steps": avg_hover_success_steps,
        "avg_hover_time": avg_hover_time,
        "avg_hover_success_rate": avg_hover_success_rate,
        "difficulty": difficulty,
        "total_timesteps": total_timesteps
    }

    if num is None:
        np.save(save_dir + "/hover_metrics.npy", metrics)
    else:
        np.save(save_dir + f"/hover_metrics{num}.npy", metrics)

    return avg_hover_success_rate


def evaluate_individual(individual, ind_save_dir, training_ts, num_envs, difficulty="easy", device="cpu", num=None) -> float:
    """
    Evaluate a drone design on the hovering task.

    Args:
        individual: Drone design array
        ind_save_dir: Directory to save individual results
        training_ts: Training timesteps
        num_envs: Number of parallel environments
        difficulty: Difficulty level
        device: Device to run on
        num: Optional number for multiple runs

    Returns:
        Hover success rate (0-1)
    """
    start_time = time.time()

    # Check if drone can hover statically
    sim = get_sim(individual)
    sim.compute_hover(verbose=False)
    if sim.static_success == False:
        spinning_success = sim.spinning_success
    else:
        spinning_success = False

    success = sim.static_success  # or spinning_success
    if not success:
        try:
            fig = plt.figure(figsize=plt.figaspect(0.5))
            ax = fig.add_subplot(111, projection='3d')
            visualizer = DroneVisualizer()
            visualizer.plot_3d(individual, ax=ax, title=f"Failed Individual (Gen {num})", fitness=0, generation=num)
            if num is not None:
                plt.savefig(ind_save_dir + f"/morphology{num}.png")
            else:
                plt.savefig(ind_save_dir + "/morphology.png")
            plt.close()
        except:
            print(f"Failed to plot:\n {individual}")
        return 0.0

    # Plot pre-training morphology
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(111, projection='3d')
    visualizer = DroneVisualizer()
    visualizer.plot_3d(individual, ax=ax, title=f"Pre-training (Gen {num})", fitness=np.nan, generation=num)
    if num is not None:
        plt.savefig(ind_save_dir + f"/morphology{num}.png")
    else:
        plt.savefig(ind_save_dir + "/morphology.png")
    plt.close()

    # Train hover policy
    hover_success_rate = train(
        individual,
        total_timesteps=int(float(training_ts)),
        save_dir=ind_save_dir,
        num_envs=int(num_envs),
        device=device,
        num=num,
        difficulty=difficulty
    )

    # Plot post-training morphology with fitness
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(111, projection='3d')
    visualizer = DroneVisualizer()
    visualizer.plot_3d(individual, ax=ax, title=f"Post-training (Gen {num})", fitness=hover_success_rate, generation=num)
    if num is not None:
        plt.savefig(ind_save_dir + f"/morphology_final{num}.png")
    else:
        plt.savefig(ind_save_dir + "/morphology_final.png")
    plt.close()

    end_time = time.time()
    print(f"{end_time-start_time:.1f} seconds to evaluate, hover_success_rate={hover_success_rate:.2%}")
    return hover_success_rate


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    parser.add_argument('--training_timesteps', default=1E7)
    parser.add_argument('--num_envs', default=100)
    parser.add_argument('--difficulty', default='easy', choices=['easy', 'medium', 'hard'])
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--num', default=None)
    args = parser.parse_args()

    # Load individual from directory
    individual = np.load(args.filename + "/individual.npy", allow_pickle=True)
    individual = individual.astype(np.float32)

    if args.num is None:
        num = None
    else:
        num = int(args.num)

    hover_success_rate = evaluate_individual(
        individual,
        args.filename,
        args.training_timesteps,
        args.num_envs,
        args.difficulty,
        args.device,
        num=num
    )

    print(f"Final hover success rate: {hover_success_rate:.2%}")
