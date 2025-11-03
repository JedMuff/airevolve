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

# Suppress the render_mode warning from stable_baselines3
warnings.filterwarnings("ignore", message="The `render_mode` attribute is not defined in your environment")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from airevolve.evolution_tools.evaluators.drone_gate_env import DroneGateEnv
from airevolve.simulator.visualization.animation import view as animation_view
from airevolve.evolution_tools.inspection_tools.morphological_descriptors.hovering_info import get_sim
from airevolve.evolution_tools.inspection_tools.drone_visualizer import DroneVisualizer 

import argparse

class backandforth():
    # back and forth
    gate_pos = np.array([
        [  2.0,  0.0,  0.0],
        [  8.0,  0.0,  0.0],
        [  8.0,  0.0,  0.0],
        [  2.0,  0.0,  0.0],
    ], dtype=np.float32)
    gate_yaw = np.array([0,0,2,2], dtype=np.float32) * np.pi / 2
    x_bounds = np.array([-1, 11], dtype=np.float32)
    y_bounds = np.array([-1, 1], dtype=np.float32)
    z_bounds = np.array([-1, 1], dtype=np.float32)
    starting_pos = np.array([0.0, 0.0, 0.0])

class figure8():
    gate_pos = np.array([
        [ -1.5,  1.5,  0.0],
        [  0.0,  0.0,  0.0],
        [  1.5, -1.5,  0.0],
        [  3.0,  0.0,  0.0],
        [  1.5,  1.5,  0.0],
        [  0.0,  0.0,  0.0],
        [ -1.5, -1.5,  0.0],
        [ -3.0,  0.0,  0.0],
    ], dtype=np.float32)
    gate_yaw = np.array([0,-1,0,1,2,-1,2,1], dtype=np.float32) * np.pi / 2
    x_bounds = np.array([-4, 4], dtype=np.float32)
    y_bounds = np.array([-2.5, 2.5], dtype=np.float32)
    z_bounds = np.array([-1, 1], dtype=np.float32)
    starting_pos = np.array([-2.0, 1.5, 0.0])

class circle():
    gate_pos = np.array([
        [  0.0, -1.5,  0.0],
        [  1.5,  0.0,  0.0],
        [  0.0,  1.5,  0.0],
        [ -1.5,  0.0,  0.0]
    ], dtype=np.float32)
    gate_yaw = np.array([0,1,2,3], dtype=np.float32) * np.pi / 2
    x_bounds = np.array([-3, 3], dtype=np.float32)
    y_bounds = np.array([-3, 3], dtype=np.float32)
    z_bounds = np.array([-1, 1], dtype=np.float32)
    starting_pos = np.array([-1.5, -1.5, 0.0])

class slalom():

    gate_pos = np.array([[x, (i % 2) * (1 if i % 4 == 1 else -1), 0] for i, x in enumerate(range(0, 82, 2))], dtype=np.float32)
    ng = len(gate_pos)
    # gate_pos = np.array([[  i*3.0,  0.0,  0.0] for i in range(ng)], dtype=np.float32)
    gate_yaw = np.tile([1, 0, -1, 0], ng) * np.pi / 2
    x_bounds = np.array([-2, 82+1], dtype=np.float32)
    y_bounds = np.array([-3, 3], dtype=np.float32)
    z_bounds = np.array([-1, 1], dtype=np.float32)
    starting_pos = np.array([0, -1, 0])

class timedlr():
    """
    Task B: Timed left-right gates with variable positioning.
    Gates are dynamically generated during episode resets.
    Exact replication of Task B from AirframeOptimization2.
    """
    # Task B parameters from AirframeOptimization2/src/main.py
    gate_d = 0.25  # Distance between gates
    gate_r_min = 0.5  # Minimum radius from center
    gate_r_max = 0.7  # Maximum radius from center
    gate_z_min = -0.1  # Minimum z offset (symmetric around 0)
    gate_z_max = 0.1   # Maximum z offset
    gate_width = 0.5   # Width of each gate
    gate_prob_lr = 0.05  # Probability of left/right gate
    num_gates = 100  # Number of gates in the course
    
    # Generate initial gate configuration (will be regenerated during resets)
    gate_pos, gate_yaw = None, None  # Will be generated dynamically
    x_bounds = np.array([0, num_gates * gate_d + 2], dtype=np.float32)
    y_bounds = np.array([-gate_r_max - 1, gate_r_max + 1], dtype=np.float32)
    z_bounds = np.array([gate_z_min - 0.5, gate_z_max + 0.5], dtype=np.float32)
    starting_pos = np.array([0.0, 0.0, 0.0])
    
    @classmethod
    def generate_gates(cls, seed=None):
        """
        Generate random gate positions for the timed left-right task.
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            gate_pos: Array of gate positions (num_gates, 3)
            gate_yaw: Array of gate yaw angles (num_gates,)
        """
        if seed is not None:
            np.random.seed(seed)
        
        gate_pos = np.zeros((cls.num_gates, 3), dtype=np.float32)
        gate_yaw = np.zeros(cls.num_gates, dtype=np.float32)
        
        for i in range(cls.num_gates):
            # X position: linear progression
            gate_pos[i, 0] = i * cls.gate_d
            
            # Y and Z positions: random within bounds
            # Determine if gate goes left or right based on probability
            if np.random.rand() < cls.gate_prob_lr:
                # Left gate
                gate_pos[i, 1] = -np.random.uniform(cls.gate_r_min, cls.gate_r_max)
            elif np.random.rand() < cls.gate_prob_lr:
                # Right gate  
                gate_pos[i, 1] = np.random.uniform(cls.gate_r_min, cls.gate_r_max)
            else:
                # Center gate
                gate_pos[i, 1] = 0.0
            
            # Random z offset
            gate_pos[i, 2] = np.random.uniform(cls.gate_z_min, cls.gate_z_max)
            
            # Yaw: always facing forward (0 degrees)
            gate_yaw[i] = 0.0
        
        return gate_pos, gate_yaw

class lrcontinuous():
    """
    Task A: Continuous left-right gates with fixed offset pattern.
    Gates are dynamically generated during episode resets.
    Exact replication of Task A from AirframeOptimization2.
    """
    # Task A parameters from AirframeOptimization2/src/main.py
    gate_d = 0.5  # Distance between gates (2x more spacing than Task B)
    gate_r = 0.25  # Fixed lateral offset
    gate_width = 0.5  # Width of each gate
    num_gates = 100  # Number of gates in the course
    
    # Bounds
    gate_pos, gate_yaw = None, None  # Will be generated dynamically
    x_bounds = np.array([0, num_gates * gate_d + 2], dtype=np.float32)
    y_bounds = np.array([-gate_r - 1, gate_r + 1], dtype=np.float32)
    z_bounds = np.array([-1.0, 1.0], dtype=np.float32)
    starting_pos = np.array([0.0, 0.0, 0.0])
    
    @classmethod
    def generate_gates(cls, seed=None):
        """
        Generate gate positions for continuous left-right pattern.
        
        Args:
            seed: Random seed for reproducibility (affects the pattern variation)
            
        Returns:
            gate_pos: Array of gate positions (num_gates, 3)
            gate_yaw: Array of gate yaw angles (num_gates,)
        """
        if seed is not None:
            np.random.seed(seed)
        
        gate_pos = np.zeros((cls.num_gates, 3), dtype=np.float32)
        gate_yaw = np.zeros(cls.num_gates, dtype=np.float32)
        
        # Create continuous left-right pattern
        # Pattern alternates: left, center, right, center, repeat
        pattern = ['left', 'center', 'right', 'center']
        
        for i in range(cls.num_gates):
            # X position: linear progression
            gate_pos[i, 0] = i * cls.gate_d
            
            # Y position: follow pattern with fixed offset
            pattern_idx = i % len(pattern)
            if pattern[pattern_idx] == 'left':
                gate_pos[i, 1] = -cls.gate_r
            elif pattern[pattern_idx] == 'right':
                gate_pos[i, 1] = cls.gate_r
            else:  # center
                gate_pos[i, 1] = 0.0
            
            # Z position: stays at 0 (no vertical variation in Task A)
            gate_pos[i, 2] = 0.0
            
            # Yaw: always facing forward (0 degrees)
            gate_yaw[i] = 0.0
        
        return gate_pos, gate_yaw
    
# ANIMATION FUNCTION
def animate_policy(individual, model, env, deterministic=False, log_times=False, print_vel=False, log=None, view_type="top",
                    motor_colors=['red', 'blue', 'green', 'orange', 'purple', 'brown'], **kwargs):
    env.reset()
    
    # Convert individual to propellers configuration
    propellers = env._convert_individual_to_propellers(individual)
    
    def get_drone_state():
        actions, _ = model.predict(env.states, deterministic=deterministic)

        states, rewards, dones, infos = env.step(actions)
        if log != None:
            log(states)
        if print_vel:
            # compute mean velocity
            vels = env.world_states[:,3:6]
            mean_vel = np.linalg.norm(vels, axis=1).mean()
            print(mean_vel)
        if log_times:
            if rewards[0] == 10:
                print(env.step_counts[0]*env.dt)
        
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
        
        # Add motor thrust values
        for i in range(num_motors):
            if i < env.prev_actions.shape[1]:
                drone_state[f'u{i+1}'] = env.prev_actions[0][i]
            else:
                drone_state[f'u{i+1}'] = 0
        
        return drone_state
    
    animation_view(propellers, get_drone_state, gate_pos=env.gate_pos, gate_yaw=env.gate_yaw, view_type=view_type, motor_colors=motor_colors, **kwargs)

class FullStatsCallback(BaseCallback):
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
        # for tag in self.tags:
        #     if tag in self.logger.name_to_value:
        #         self.logger.record(f"monitor/{tag}", self.logger.name_to_value[tag])
        # Force flush for debugging; can remove later
        self.logger.dump(self.num_timesteps)

def train(individual, gate_cfg, total_timesteps=int(1E8), save_dir="./logs", num_envs=100, device="cuda:0", num=None, task_seed=None):

    if gate_cfg == "backandforth":
        gate_pos = backandforth.gate_pos
        gate_yaw = backandforth.gate_yaw
        start_pos = backandforth.starting_pos
        x_bounds = backandforth.x_bounds
        y_bounds = backandforth.y_bounds
        z_bounds = backandforth.z_bounds
    elif gate_cfg == "figure8":
        gate_pos = figure8.gate_pos
        gate_yaw = figure8.gate_yaw
        start_pos = figure8.starting_pos
        x_bounds = figure8.x_bounds
        y_bounds = figure8.y_bounds
        z_bounds = figure8.z_bounds
    elif gate_cfg == "circle":
        gate_pos = circle.gate_pos
        gate_yaw = circle.gate_yaw
        start_pos = circle.starting_pos
        x_bounds = circle.x_bounds
        y_bounds = circle.y_bounds
        z_bounds = circle.z_bounds
    elif gate_cfg == "slalom":
        gate_pos = slalom.gate_pos
        gate_yaw = slalom.gate_yaw
        start_pos = slalom.starting_pos
        x_bounds = slalom.x_bounds
        y_bounds = slalom.y_bounds
        z_bounds = slalom.z_bounds
    elif gate_cfg == "timedlr":
        # Generate initial gates for Task B (will be regenerated per episode)
        gate_pos, gate_yaw = timedlr.generate_gates(seed=task_seed)
        start_pos = timedlr.starting_pos
        x_bounds = timedlr.x_bounds
        y_bounds = timedlr.y_bounds
        z_bounds = timedlr.z_bounds
        # Use dynamic gate generation for timedlr
        gate_generator = timedlr.generate_gates
        dynamic_gates = True
    elif gate_cfg == "lrcontinuous":
        # Generate initial gates for Task A (will be regenerated per episode)
        gate_pos, gate_yaw = lrcontinuous.generate_gates(seed=task_seed)
        start_pos = lrcontinuous.starting_pos
        x_bounds = lrcontinuous.x_bounds
        y_bounds = lrcontinuous.y_bounds
        z_bounds = lrcontinuous.z_bounds
        # Use dynamic gate generation for lrcontinuous
        gate_generator = lrcontinuous.generate_gates
        dynamic_gates = True
    else:
        raise ValueError("Invalid gate configuration")

    # SETUP LOGGING
    save_dir = save_dir+"/"
    # models_dir = save_dir+'/models/'
    # log_dir = save_dir+'/logs/'
    # video_dir = save_dir+'/videos/'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # if not os.path.exists(models_dir):
    #     os.makedirs(models_dir)
    # if not os.path.exists(log_dir):
    #     os.makedirs(log_dir)    

    # Set up gate generator and dynamic gates flag (for timedlr and lrcontinuous)
    if gate_cfg not in ["timedlr", "lrcontinuous"]:
        gate_generator = None
        dynamic_gates = False
    
    # Use advanced reward function only for timedlr task (not lrcontinuous)
    use_advanced_reward = (gate_cfg == "timedlr")

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
        num_state_history=0,
        num_action_history=0,
        history_step_size=1,
        render_mode=None,
        device=device,
        gate_generator=gate_generator,
        dynamic_gates=dynamic_gates,
        use_advanced_reward=use_advanced_reward,
        seed=task_seed
    )
    test_env = DroneGateEnv(
        num_envs=1,
        individual=individual,
        gates_pos=gate_pos,
        gate_yaw=gate_yaw,
        start_pos=start_pos,
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        z_bounds=z_bounds,
        initialize_at_random_gates=False,
        gates_ahead=1,
        num_state_history=0,
        num_action_history=0,
        history_step_size=1,
        render_mode=None,
        device=device,
        gate_generator=gate_generator,
        dynamic_gates=dynamic_gates,
        use_advanced_reward=use_advanced_reward,
        seed=task_seed
    )

    # Wrap the environment in a Monitor wrapper
    # Make monitor1 folder
    if num is not None:
        monitor_file = save_dir+f"m{num}"
    else: 
        monitor_file = save_dir
    
    env = VecMonitor(env, filename=monitor_file)
    # custom_logger = configure(save_dir, ["stdout", "csv", "tensorboard"])

    # MODEL DEFINITION
    policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=dict(pi=[64, 64, 64], vf=[64, 64, 64]), log_std_init = 0)
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=0,
        tensorboard_log=save_dir,
        n_steps=1000,
        batch_size=1000,
        n_epochs=10,
        gamma=0.999,
        device=device
    )
    # model.set_logger(custom_logger)

    # TRAINING
    model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False, log_interval=100, callback=FullStatsCallback())
    if num is None:
        model.save(save_dir + '/' + "policy")
    else:
        model.save(save_dir + '/' + f"policy{num}")
    # model_path = save_dir + "/"+"best_model.zip"
    # PPO_model = PPO.load(model_path)

    # Plotting the training curve
    try:
        data = pd.read_csv(monitor_file+".monitor.csv", skiprows=1)  # Skip the first row (comments)
    except:
        data = pd.read_csv(monitor_file+"monitor.csv", skiprows=1)
    episode_rewards = data["r"]  # Rewards per episode
    episode_lengths = data["l"]  # Episode lengths in simulation steps
    cumulative_timesteps = episode_lengths.cumsum()  # Cumulative simulation timesteps
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_timesteps, episode_rewards, label="Episode Reward")
    # plt.fill_between(cumulative_timesteps[:,0], episode_rewards_mean - episode_rewards_std, episode_rewards_mean + episode_rewards_std, alpha=0.2)
    plt.xlabel("Simulation Timesteps")
    plt.ylabel("Reward")
    plt.title("Reward per Episode")
    plt.legend()
    if num is None:
        plt.savefig(save_dir+"/figure.pdf")
    else:
        plt.savefig(save_dir+f"/figure{num}.pdf")
    plt.close()
    # plt.show()
    # TESTING
    test_env.reset()
    # if num is None:
    #     animate_policy(individual, model, test_env, deterministic=False, log_times=False, print_vel=False, log=None, 
    #                 record_steps=1200, record_file=save_dir + f'v.mp4',
    #                 show_window=False)
    # else:
    #     animate_policy(individual, model, test_env, deterministic=False, log_times=False, print_vel=False, log=None, 
    #                 record_steps=1200, record_file=save_dir + f'v{num}.mp4',
    #                 show_window=False)

    test_env.reset()
    # do 1000 steps and check gate passes
    gates_passed_total = 0
    for i in range(1000):
        num = test_env.num_state_history+1
        state_len = int(len(test_env.states[0])/num)
        actions, _ = model.predict(test_env.states, deterministic=True)
        states, rewards, dones, infos = test_env.step(actions)
        
        # Track gate passages
        if infos[0]["gate_passed"]:
            gates_passed_total += 1
            print(f"Gate passed at step {i}! Total: {gates_passed_total}")
    
    final_gates_passed = infos[0]["num_gates_passed"][0]
    print(f"\nTest evaluation complete:")
    print(f"  Gates passed during 1000 steps: {gates_passed_total}")
    print(f"  Final num_gates_passed value: {final_gates_passed}")
    print(f"  Episode done: {dones[0]}")
    
    return final_gates_passed

def evaluate_individual(individual, ind_save_dir, training_ts, num_envs, gate_cfg, device="cuda:0", num=None, task_seed=None) -> list:
    start_time = time.time()
    sim = get_sim(individual)
    sim.compute_hover(verbose=False)
    if sim.static_success == False:
        spinning_success = sim.spinning_success
    else:  
        spinning_success = False

    success = sim.static_success# or spinning_success
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
        return 0
    
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(111, projection='3d')
    visualizer = DroneVisualizer()
    visualizer.plot_3d(individual, ax=ax, title=f"Pre-training (Gen {num})", fitness=np.nan, generation=num)
    if num is not None:
        plt.savefig(ind_save_dir + f"/morphology{num}.png")
    else:
        plt.savefig(ind_save_dir + "/morphology.png")
    plt.close()

    num_gates_passed = train(individual, gate_cfg, total_timesteps=int(float(training_ts)), save_dir=ind_save_dir, num_envs=int(num_envs), device=device, num=num, task_seed=task_seed)

    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(111, projection='3d')
    visualizer = DroneVisualizer()
    visualizer.plot_3d(individual, ax=ax, title=f"Post-training (Gen {num})", fitness=num_gates_passed, generation=num)
    if num is not None:
        plt.savefig(ind_save_dir + f"/morphology{num}.png")
    else:
        plt.savefig(ind_save_dir + "/morphology.png")
    plt.close()

    end_time = time.time()
    # print(f"{end_time-start_time} seconds to evaluate, fitness={num_gates_passed}")
    return num_gates_passed

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    parser.add_argument('--training_timesteps', default=1E8) 
    parser.add_argument('--num_envs', default=100)
    parser.add_argument('--gate_cfg', default='figure8')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--num', default=None)
    parser.add_argument('--task_seed', default=None, type=int, help='Seed for generating dynamic gates (timedlr task)')
    args = parser.parse_args()

    # Load Bf and Bm from directory
    individual = np.load(args.filename + "/individual.npy", allow_pickle=True)
    individual = individual.astype(np.float32)

    if args.num is None:
        num = None
    else:
        num = int(args.num)
    num_gates_passed = evaluate_individual(individual, args.filename, args.training_timesteps, args.num_envs, args.gate_cfg, args.device, num=num, task_seed=args.task_seed)

    print(num_gates_passed)
