#!/usr/bin/env python3
"""
Training script with optimized hyperparameters.
After running Optuna optimization, copy the best hyperparameters here and use this script.
"""

import numpy as np
import pickle
import os
import time
import argparse
import torch
from airevolve.evolution_tools.evaluators.gate_train import evaluate_individual
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor
from airevolve.evolution_tools.evaluators.drone_gate_env import DroneGateEnv
from airevolve.evolution_tools.evaluators.gate_train import backandforth, figure8, circle, slalom
from airevolve.evolution_tools.inspection_tools.morphological_descriptors.hovering_info import get_sim


# TODO: Replace these with your optimized hyperparameters from Optuna
OPTIMIZED_HYPERPARAMETERS = { # best for slalom task
    'learning_rate': 0.0000113,          # Replace with Optuna result
    'n_steps': 256,                # Replace with Optuna result
    'batch_size': 64,              # Replace with Optuna result
    'n_epochs': 27,                 # Replace with Optuna result
    'gamma': 0.997,                 # Replace with Optuna result
    'gae_lambda': 0.9835,             # Replace with Optuna result
    'clip_range': 0.107,              # Replace with Optuna result
    'ent_coef': 0.0175,               # Replace with Optuna result
    'vf_coef': 0.4322,                 # Replace with Optuna result
    'max_grad_norm': 3.8497,           # Replace with Optuna result
    'log_std_init': 0.3505,            # Replace with Optuna result
    'net_arch': 'large',           # Replace with Optuna result ('small', 'medium', 'large')
}


def get_gate_config(gate_cfg: str):
    """Get gate configuration for the specified task."""
    configs = {
        'backandforth': backandforth,
        'figure8': figure8,
        'circle': circle,
        'slalom': slalom
    }
    
    if gate_cfg not in configs:
        raise ValueError(f"Invalid gate configuration: {gate_cfg}")
    
    cfg = configs[gate_cfg]
    return {
        'gate_pos': cfg.gate_pos,
        'gate_yaw': cfg.gate_yaw,
        'start_pos': cfg.starting_pos,
        'x_bounds': cfg.x_bounds,
        'y_bounds': cfg.y_bounds,
        'z_bounds': cfg.z_bounds
    }


def train_with_optimized_hyperparameters(individual, gate_cfg, total_timesteps, save_dir, 
                                       num_envs, device, hyperparams):
    """Train using the optimized hyperparameters."""
    
    # Check if drone can hover
    sim = get_sim(individual)
    sim.compute_hover(verbose=False)
    if not sim.static_success:
        print("Warning: Drone cannot hover. Training may fail.")
        return 0
    
    # Get gate configuration
    cfg = get_gate_config(gate_cfg)
    
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    
    # Create environment
    env = DroneGateEnv(
        num_envs=num_envs,
        individual=individual,
        gates_pos=cfg['gate_pos'],
        gate_yaw=cfg['gate_yaw'],
        start_pos=cfg['start_pos'],
        x_bounds=cfg['x_bounds'],
        y_bounds=cfg['y_bounds'],
        z_bounds=cfg['z_bounds'],
        gates_ahead=1,
        num_state_history=0,
        num_action_history=0,
        history_step_size=1,
        render_mode=None,
        device=device
    )
    
    env = VecMonitor(env, filename=save_dir)
    
    # Set up network architecture
    if hyperparams['net_arch'] == 'small':
        net_arch = dict(pi=[32, 32], vf=[32, 32])
    elif hyperparams['net_arch'] == 'medium':
        net_arch = dict(pi=[64, 64], vf=[64, 64])
    else:  # large
        net_arch = dict(pi=[128, 128, 128], vf=[128, 128, 128])
    
    # Create model with optimized hyperparameters
    policy_kwargs = dict(
        activation_fn=torch.nn.ReLU,
        net_arch=net_arch,
        log_std_init=hyperparams['log_std_init']
    )
    
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=hyperparams['learning_rate'],
        n_steps=hyperparams['n_steps'],
        batch_size=hyperparams['batch_size'],
        n_epochs=hyperparams['n_epochs'],
        gamma=hyperparams['gamma'],
        gae_lambda=hyperparams['gae_lambda'],
        clip_range=hyperparams['clip_range'],
        ent_coef=hyperparams['ent_coef'],
        vf_coef=hyperparams['vf_coef'],
        max_grad_norm=hyperparams['max_grad_norm'],
        verbose=1,  # Show training progress
        tensorboard_log=save_dir,
        device=device
    )
    
    print(f"Training with optimized hyperparameters:")
    for key, value in hyperparams.items():
        print(f"  {key}: {value}")
    print()
    
    # Train the model
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    model.save(os.path.join(save_dir, "optimized_policy"))
    
    # Evaluate
    test_env = DroneGateEnv(
        num_envs=1,
        individual=individual,
        gates_pos=cfg['gate_pos'],
        gate_yaw=cfg['gate_yaw'],
        start_pos=cfg['start_pos'],
        x_bounds=cfg['x_bounds'],
        y_bounds=cfg['y_bounds'],
        z_bounds=cfg['z_bounds'],
        initialize_at_random_gates=False,
        gates_ahead=1,
        num_state_history=0,
        num_action_history=0,
        history_step_size=1,
        render_mode=None,
        device=device
    )
    
    test_env.reset()
    for _ in range(1000):
        actions, _ = model.predict(test_env.states, deterministic=True)
        states, rewards, dones, infos = test_env.step(actions)
    
    fitness = infos[0]["num_gates_passed"][0]
    
    # Clean up
    env.close()
    test_env.close()
    
    return fitness


def load_hyperparameters_from_optuna(optuna_results_dir: str):
    """Load the best hyperparameters from Optuna results."""
    hyperparams_file = os.path.join(optuna_results_dir, 'best_hyperparameters.pkl')
    
    if not os.path.exists(hyperparams_file):
        print(f"Warning: {hyperparams_file} not found. Using default hyperparameters.")
        return OPTIMIZED_HYPERPARAMETERS
    
    try:
        with open(hyperparams_file, 'rb') as f:
            optuna_results = pickle.load(f)
        
        # Convert Optuna results to our format
        best_params = optuna_results['best_params']
        
        # Map network architecture
        if 'net_arch' in best_params:
            net_arch_map = {'small': 'small', 'medium': 'medium', 'large': 'large'}
            best_params['net_arch'] = net_arch_map.get(best_params['net_arch'], 'medium')
        
        print(f"Loaded optimized hyperparameters from: {hyperparams_file}")
        print(f"Best trial achieved fitness: {optuna_results.get('best_fitness', 'unknown')}")
        
        return best_params
        
    except Exception as e:
        print(f"Error loading hyperparameters: {e}")
        print("Using default hyperparameters.")
        return OPTIMIZED_HYPERPARAMETERS


def main():
    parser = argparse.ArgumentParser(description='Train drone with optimized hyperparameters')
    parser.add_argument('--task', type=str, default='figure8',
                       choices=['circle', 'figure8', 'slalom', 'backandforth'],
                       help='Flight task to train on')
    parser.add_argument('--timesteps', type=float, default=1e7,
                       help='Number of training timesteps')
    parser.add_argument('--num-envs', type=int, default=50,
                       help='Number of parallel environments')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda:0', 'cuda:1'],
                       help='Device to use for training')
    parser.add_argument('--output-dir', type=str, default='optimized_results',
                       help='Directory to save results')
    parser.add_argument('--optuna-results', type=str, default=None,
                       help='Directory containing Optuna results (to auto-load best hyperparameters)')
    parser.add_argument('--production', action='store_true',
                       help='Use production settings (1e8 timesteps, 100 envs)')
    
    args = parser.parse_args()
    
    # Override with production settings if requested
    if args.production:
        print("Using production settings: 1e8 timesteps, 100 parallel envs")
        args.timesteps = 1e8
        args.num_envs = 100
        args.device = 'cpu'  # CPU recommended for MLP policies
    
    # Load hyperparameters
    if args.optuna_results:
        hyperparams = load_hyperparameters_from_optuna(args.optuna_results)
    else:
        hyperparams = OPTIMIZED_HYPERPARAMETERS
        print("Using manually specified hyperparameters (update OPTIMIZED_HYPERPARAMETERS in script)")
    
    # Example drone design (simple quadcopter)
    example_individual = np.array([
        [0.25,  np.pi/4,   0.0,  0.0,  0.0,  1.0],  # Front-right motor (CW)
        [0.25, -np.pi/4,   0.0,  0.0,  0.0,  0.0],  # Front-left motor (CCW)
        [0.25,  3*np.pi/4, 0.0,  0.0,  0.0,  0.0],  # Back-left motor (CCW)  
        [0.25, -3*np.pi/4, 0.0,  0.0,  0.0,  1.0],  # Back-right motor (CW)
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],  # Unused
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]   # Unused
    ], dtype=np.float32)
    
    print(f"\nOptimized Training Configuration:")
    print(f"  Task: {args.task}")
    print(f"  Timesteps: {int(args.timesteps):,}")
    print(f"  Parallel environments: {args.num_envs}")
    print(f"  Device: {args.device}")
    print(f"  Output directory: {args.output_dir}")
    print()
    
    # Save configuration and individual
    os.makedirs(args.output_dir, exist_ok=True)
    np.save(os.path.join(args.output_dir, "individual.npy"), example_individual)
    
    metadata = {
        'task': args.task,
        'timesteps': int(args.timesteps),
        'num_envs': args.num_envs,
        'device': args.device,
        'hyperparameters': hyperparams,
        'script_type': 'optimized_training',
        'start_time': time.time()
    }
    
    with open(os.path.join(args.output_dir, "metadata.pkl"), 'wb') as f:
        pickle.dump(metadata, f)
    
    # Train with optimized hyperparameters
    start_time = time.time()
    
    fitness = train_with_optimized_hyperparameters(
        individual=example_individual,
        gate_cfg=args.task,
        total_timesteps=int(args.timesteps),
        save_dir=args.output_dir,
        num_envs=args.num_envs,
        device=args.device,
        hyperparams=hyperparams
    )
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Save results
    results = {
        'fitness': fitness,
        'duration': duration,
        'success': True,
        'task': args.task,
        'hyperparameters': hyperparams,
        'end_time': end_time
    }
    
    with open(os.path.join(args.output_dir, "results.pkl"), 'wb') as f:
        pickle.dump(results, f)
    
    with open(os.path.join(args.output_dir, "fitness.txt"), 'w') as f:
        f.write(f"{fitness}\n")
    
    print(f"\nOptimized training completed!")
    print(f"Final fitness: {fitness}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()