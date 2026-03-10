"""
Curriculum learning evaluator for evolutionary algorithm.

Two-stage training process:
1. Hover training (basic stabilization) with threshold-based early stopping
2. Gate navigation training (with policy transfer from hover)

If hover training fails to meet threshold, returns fitness of 0 without proceeding to gate stage.
Designed for multi-parallelism: minimal console output, returns only integer fitness.
"""

import numpy as np
import os
import sys
import time
import torch
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
from collections import deque

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from airevolve.evolution_tools.evaluators import gate_train
from airevolve.evolution_tools.evaluators.drone_gate_env import DroneGateEnv
from airevolve.evolution_tools.evaluators.drone_hover_env import DroneHoverEnv
from airevolve.evolution_tools.inspection_tools.morphological_descriptors.hovering_info import get_sim
from airevolve.evolution_tools.inspection_tools.drone_visualizer import DroneVisualizer
from airevolve.evolution_tools.genome_handlers.repair_workflow import (
    repair_operation_process,
    is_nan_individual
)


class ThresholdStoppingCallback(BaseCallback):
    """
    Callback that monitors hover success rate during training and stops early when threshold is met.

    Collects hover_success_rate from training episodes and checks threshold periodically.
    """

    def __init__(self, success_threshold=0.90, window_size=100, check_freq=10000, verbose=0):
        """
        Args:
            success_threshold: Success rate threshold to stop training (0.0-1.0)
            window_size: Number of recent episodes to track
            check_freq: Check threshold every N steps
            verbose: Verbosity level (0=silent, 1=verbose)
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
                    print(f"  [{self.num_timesteps:,} steps] Hover success: {mean_success_rate:.2%}")

                if mean_success_rate >= self.success_threshold:
                    self.threshold_met = True
                    if self.verbose >= 1:
                        print(f"  Threshold met ({mean_success_rate:.2%} >= {self.success_threshold:.1%})")
                    return False  # Stop training

        return True  # Continue training


class BestModelCallback(BaseCallback):
    """
    Callback that saves the best model based on episode reward.
    Only starts saving after a certain fraction of total timesteps.
    """

    def __init__(self, save_dir, total_timesteps, start_fraction=0.75, verbose=0):
        """
        Args:
            save_dir: Directory to save best models
            total_timesteps: Total training timesteps
            start_fraction: Fraction of training to wait before saving (default: 0.75)
            verbose: Verbosity level (0=silent)
        """
        super().__init__(verbose)
        self.save_dir = save_dir
        self.total_timesteps = total_timesteps
        self.start_fraction = start_fraction
        self.start_timestep = int(total_timesteps * start_fraction)
        self.best_reward = -np.inf
        self.best_model_path = None
        self.saving_active = False

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

        # Collect episode rewards
        if len(self.locals.get("infos", [])) > 0:
            for info in self.locals["infos"]:
                if "episode" in info:
                    episode_reward = info["episode"]["r"]

                    # Check if this is a new best (only if saving is active)
                    if self.saving_active and episode_reward > self.best_reward:
                        self.best_reward = episode_reward

                        # Save the model
                        model_name = f"best_model_reward_{episode_reward:.2f}_step_{self.num_timesteps}"
                        model_path = os.path.join(self.save_dir, model_name)
                        self.model.save(model_path)
                        self.best_model_path = model_path + ".zip"

        return True  # Continue training


class GateProgressCallback(BaseCallback):
    """
    Callback that monitors gate passing performance during training.
    Tracks gates passed per episode and saves to CSV.
    """

    def __init__(self, window_size=100, check_freq=10000, save_dir=None, verbose=0):
        """
        Args:
            window_size: Number of recent episodes to track
            check_freq: Report progress every N steps
            save_dir: Directory to save gates_passed.csv (optional)
            verbose: Verbosity level (0=silent)
        """
        super().__init__(verbose)
        self.window_size = window_size
        self.check_freq = check_freq
        self.save_dir = save_dir
        self.gates_passed = deque(maxlen=window_size)
        self.last_check = 0

        # Initialize CSV file if save_dir is provided
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)
            self.csv_path = os.path.join(self.save_dir, "gates_passed.csv")
            with open(self.csv_path, 'w') as f:
                f.write("timestep,gates_passed\n")

    def _on_step(self) -> bool:
        """
        Called at every training step.
        Collects num_gates_passed from episode completions.

        Returns:
            True to continue training
        """
        # Collect gates passed from completed episodes
        if len(self.locals.get("infos", [])) > 0:
            for info in self.locals["infos"]:
                if "num_gates_passed" in info:
                    gates = info["num_gates_passed"]
                    if hasattr(gates, '__iter__'):
                        gates_value = gates[0]
                    else:
                        gates_value = gates

                    self.gates_passed.append(gates_value)

                    # Write to CSV file immediately if save_dir is set
                    if self.save_dir is not None:
                        with open(self.csv_path, 'a') as f:
                            f.write(f"{self.num_timesteps},{gates_value}\n")

        # Report progress periodically (only if verbose)
        if self.verbose >= 1 and self.num_timesteps - self.last_check >= self.check_freq:
            self.last_check = self.num_timesteps

            if len(self.gates_passed) >= 10:
                mean_gates = np.mean(self.gates_passed)
                max_gates = np.max(self.gates_passed)
                print(f"  [{self.num_timesteps:,} steps] Gates: avg={mean_gates:.1f}, max={max_gates}")

        return True  # Continue training


def stage1_hover_training(individual, save_dir, max_timesteps, num_envs, device,
                         success_threshold=0.90, window_size=100, check_freq=10000,
                         verbose=0):
    """
    Train hovering policy from scratch with early stopping based on performance threshold.

    Args:
        individual: Drone design array
        save_dir: Directory to save Stage 1 results
        max_timesteps: Maximum training timesteps for hover stage
        num_envs: Number of parallel environments
        device: Device ('cpu' or 'cuda:0')
        success_threshold: Hover success rate threshold to advance to stage 2
        window_size: Number of recent episodes to track
        check_freq: Check threshold every N steps
        verbose: Verbosity level (0=silent, 1=verbose)

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
        motor_limit=1.0,
        num_state_history=0,
        num_action_history=0,
        history_step_size=1,
        render_mode=None,
        device=device,
        add_disturbances=False,
        disturbance_strength=0.0
    )

    # Wrap training environment in monitor
    monitor_file = os.path.join(save_dir, "stage1_hover")
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

    # Setup callbacks (NO progress bar for evolution compatibility)
    threshold_callback = ThresholdStoppingCallback(
        success_threshold=success_threshold,
        window_size=window_size,
        check_freq=check_freq,
        verbose=verbose
    )

    callbacks = [threshold_callback]

    # Train with automatic early stopping
    model.learn(
        total_timesteps=max_timesteps,
        reset_num_timesteps=True,
        log_interval=100,
        callback=callbacks
    )

    # Get actual timesteps trained
    actual_timesteps = model.num_timesteps

    # Calculate final success rate from the rolling window
    if len(threshold_callback.success_rates) > 0:
        hover_success_rate = np.mean(threshold_callback.success_rates)
    else:
        hover_success_rate = 0.0

    # Save final model
    model_path = os.path.join(save_dir, "hover_policy.zip")
    model.save(os.path.join(save_dir, "hover_policy"))

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


def stage2_gate_training(individual, hover_model_path, save_dir,
                         timesteps, num_envs, gate_cfg, device,
                         window_size=100, check_freq=10000, verbose=0):
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
        window_size: Number of recent episodes to track
        check_freq: Report progress every N steps
        verbose: Verbosity level (0=silent, 1=verbose)

    Returns:
        dict: Results including final_gates_passed, avg_gates_passed, etc.
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
        motor_limit=1.0,
        num_state_history=0,
        num_action_history=0,
        history_step_size=1,
        render_mode=None,
        device=device
    )

    # Wrap in monitor
    monitor_file = os.path.join(save_dir, "stage2_gate")
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
    gate_model.policy.action_net.load_state_dict(hover_model.policy.action_net.state_dict())

    # Setup callbacks for gate training (NO progress bar)
    gate_progress_callback = GateProgressCallback(
        window_size=window_size,
        check_freq=check_freq,
        save_dir=save_dir,
        verbose=verbose
    )
    best_model_callback = BestModelCallback(
        save_dir=os.path.join(save_dir, "best_models"),
        total_timesteps=timesteps,
        start_fraction=0.75,
        verbose=verbose
    )

    callbacks = [gate_progress_callback, best_model_callback]

    # Continue training on gate task
    gate_model.learn(
        total_timesteps=timesteps,
        reset_num_timesteps=False,
        log_interval=100,
        callback=callbacks
    )

    # Save the gate policy
    gate_model.save(os.path.join(save_dir, "gate_policy"))

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
        motor_limit=1.0,
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

    # Return results
    results = {
        "final_gates_passed": final_gates_passed,
        "avg_gates_passed": avg_gates_passed,
        "max_gates_passed": max_gates_passed,
        "best_model_reward": best_model_callback.best_reward,
        "best_model_path": best_model_callback.best_model_path
    }

    return results


def evaluate_individual_curriculum(individual, ind_save_dir,
                                   hover_max_timesteps, hover_success_threshold,
                                   gate_timesteps, num_envs, gate_cfg, device,
                                   hover_window_size=100, hover_check_freq=10000,
                                   gate_window_size=100, gate_check_freq=10000,
                                   num=None):
    """
    Evaluate individual using 2-stage curriculum learning.

    CRITICAL: Returns fitness of 0 if hover stage fails to meet threshold.
    Only proceeds to gate training if hover success rate >= threshold.

    Designed for multi-parallelism: minimal console output, returns only integer fitness.

    Args:
        individual: Drone design array
        ind_save_dir: Directory to save results for this individual
        hover_max_timesteps: Maximum timesteps for hover training
        hover_success_threshold: Threshold to advance to gate stage (e.g., 0.90)
        gate_timesteps: Timesteps for gate training
        num_envs: Number of parallel environments
        gate_cfg: Gate configuration ('figure8', 'circle', 'backandforth', 'slalom')
        device: Device ('cpu' or 'cuda:0')
        hover_window_size: Window size for hover success tracking
        hover_check_freq: Check frequency for hover threshold
        gate_window_size: Window size for gate progress tracking
        gate_check_freq: Check frequency for gate progress
        num: Individual number (for file naming)

    Returns:
        int: Number of gates passed (0 if hover stage failed threshold)
    """
    start_time = time.time()

    # Apply repair pipeline to ensure individual is valid (fixes collisions, aligns thrust)
    individual, repair_status = repair_operation_process(
        individual,
        coordinate_system='spherical',
        verbose=False
    )
    if is_nan_individual(individual):
        print(f"  [ind {num}] Repair failed: {repair_status}", flush=True)
        return 0

    # Save repaired genome so the actual evaluated design is preserved
    os.makedirs(ind_save_dir, exist_ok=True)
    np.save(os.path.join(ind_save_dir, "repaired_genome.npy"), individual)

    # Check if individual can hover (static stability check)
    sim = get_sim(individual)
    sim.compute_hover(verbose=False)

    if sim.static_success == False:
        spinning_success = sim.spinning_success
    else:
        spinning_success = False

    success = sim.static_success  # or spinning_success

    if not success:
        print(f"  [ind {num}] Failed static hover check (static={sim.static_success})", flush=True)
        # Individual cannot hover - save morphology and return 0
        try:
            fig = plt.figure(figsize=plt.figaspect(0.5))
            ax = fig.add_subplot(111, projection='3d')
            visualizer = DroneVisualizer()
            visualizer.plot_3d(individual, ax=ax, title=f"Failed Hover Check (Gen {num})",
                             fitness=0, generation=num)
            if num is not None:
                plt.savefig(ind_save_dir + f"/morphology{num}.png")
            else:
                plt.savefig(ind_save_dir + "/morphology.png")
            plt.close()
        except:
            pass
        return 0

    # Save pre-training morphology
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(111, projection='3d')
    visualizer = DroneVisualizer()
    visualizer.plot_3d(individual, ax=ax, title=f"Pre-training (Gen {num})",
                      fitness=np.nan, generation=num)
    if num is not None:
        plt.savefig(ind_save_dir + f"/morphology{num}_pre.png")
    else:
        plt.savefig(ind_save_dir + "/morphology_pre.png")
    plt.close()

    # STAGE 1: HOVER TRAINING (SILENT MODE)
    try:
        hover_success_rate, hover_model_path, actual_hover_timesteps = stage1_hover_training(
            individual=individual,
            save_dir=ind_save_dir,
            max_timesteps=hover_max_timesteps,
            num_envs=num_envs,
            device=device,
            success_threshold=hover_success_threshold,
            window_size=hover_window_size,
            check_freq=hover_check_freq,
            verbose=0  # SILENT for evolution
        )
    except (ValueError, RuntimeError) as e:
        # NaN propagation in simulation can cause PyTorch distribution errors
        print(f"  [ind {num}] Hover training crashed: {e}", flush=True)
        return 0

    # CRITICAL: Check if hover threshold met
    if hover_success_rate < hover_success_threshold:
        print(f"  [ind {num}] Hover threshold not met: {hover_success_rate:.2%} < {hover_success_threshold:.2%} ({actual_hover_timesteps:,} steps)", flush=True)
        # Failed to meet hover threshold - save morphology and return 0
        fig = plt.figure(figsize=plt.figaspect(0.5))
        ax = fig.add_subplot(111, projection='3d')
        visualizer = DroneVisualizer()
        visualizer.plot_3d(individual, ax=ax,
                          title=f"Failed Hover Threshold (Gen {num}): {hover_success_rate:.2%} < {hover_success_threshold:.2%}",
                          fitness=0, generation=num)
        if num is not None:
            plt.savefig(ind_save_dir + f"/morphology{num}_failed_hover.png")
        else:
            plt.savefig(ind_save_dir + "/morphology_failed_hover.png")
        plt.close()

        return 0  # Return 0 gates passed

    # STAGE 2: GATE TRAINING (SILENT MODE)
    # Only reached if hover threshold was met
    stage2_results = stage2_gate_training(
        individual=individual,
        hover_model_path=hover_model_path,
        save_dir=ind_save_dir,
        timesteps=gate_timesteps,
        num_envs=num_envs,
        gate_cfg=gate_cfg,
        device=device,
        window_size=gate_window_size,
        check_freq=gate_check_freq,
        verbose=0  # SILENT for evolution
    )

    num_gates_passed = stage2_results['final_gates_passed']

    # Save post-training morphology
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(111, projection='3d')
    visualizer = DroneVisualizer()
    visualizer.plot_3d(individual, ax=ax, title=f"Post-training (Gen {num})",
                      fitness=num_gates_passed, generation=num)
    if num is not None:
        plt.savefig(ind_save_dir + f"/morphology{num}_post.png")
    else:
        plt.savefig(ind_save_dir + "/morphology_post.png")
    plt.close()

    end_time = time.time()

    # Return only integer fitness (for multi-parallelism compatibility)
    return num_gates_passed
