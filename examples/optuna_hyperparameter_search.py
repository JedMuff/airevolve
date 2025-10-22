#!/usr/bin/env python3
"""
Optuna hyperparameter optimization for PPO drone training.
Searches for optimal PPO hyperparameters for a given drone design and task.
"""

import numpy as np
import optuna
import os
import pickle
import argparse
from typing import Dict, Any
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.callbacks import EvalCallback

from airevolve.evolution_tools.evaluators.drone_gate_env import DroneGateEnv
from airevolve.evolution_tools.evaluators.gate_train import backandforth, figure8, circle, slalom
from airevolve.evolution_tools.inspection_tools.morphological_descriptors.hovering_info import get_sim


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


def objective(trial: optuna.Trial, individual: np.ndarray, gate_cfg: str, 
              num_envs: int, timesteps: int, device: str) -> float:
    """
    Objective function for Optuna optimization.
    
    Args:
        trial: Optuna trial object
        individual: Drone design array
        gate_cfg: Gate configuration name
        num_envs: Number of parallel environments
        timesteps: Training timesteps per trial
        device: Device to use for training
        
    Returns:
        float: Negative number of gates passed (Optuna minimizes, so we negate for maximization)
    """
    
    # Check if drone can hover
    sim = get_sim(individual)
    sim.compute_hover(verbose=False)
    if not sim.static_success:
        print(f"Trial {trial.number}: Drone cannot hover, skipping...")
        return -0.0  # Return worst possible fitness
    
    # Sample hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    n_steps = trial.suggest_categorical('n_steps', [256, 512, 1000, 2048])
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256, 512, 1000])
    n_epochs = trial.suggest_int('n_epochs', 3, 30)
    gamma = trial.suggest_float('gamma', 0.9, 0.9999, log=True)
    gae_lambda = trial.suggest_float('gae_lambda', 0.8, 0.99)
    clip_range = trial.suggest_float('clip_range', 0.1, 0.4)
    ent_coef = trial.suggest_float('ent_coef', 1e-8, 0.1, log=True)
    vf_coef = trial.suggest_float('vf_coef', 0.1, 1.0)
    max_grad_norm = trial.suggest_float('max_grad_norm', 0.3, 5.0)
    
    # Network architecture
    net_arch_type = trial.suggest_categorical('net_arch', ['small', 'medium', 'large'])
    if net_arch_type == 'small':
        net_arch = dict(pi=[32, 32], vf=[32, 32])
    elif net_arch_type == 'medium':
        net_arch = dict(pi=[64, 64], vf=[64, 64])
    else:  # large
        net_arch = dict(pi=[128, 128, 128], vf=[128, 128, 128])
    
    log_std_init = trial.suggest_float('log_std_init', -1.0, 1.0)
    
    # Get gate configuration
    cfg = get_gate_config(gate_cfg)
    
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
    
    env = VecMonitor(env)
    
    # Create model with sampled hyperparameters
    policy_kwargs = dict(
        activation_fn=torch.nn.ReLU,
        net_arch=net_arch,
        log_std_init=log_std_init
    )
    
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        verbose=0,
        device=device
    )
    
    # Train the model
    try:
        model.learn(total_timesteps=timesteps, progress_bar=False)
        
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
        del model
        torch.cuda.empty_cache() if device != 'cpu' else None
        
        print(f"Trial {trial.number}: fitness = {fitness}")
        
        # Return negative fitness (Optuna minimizes)
        return -float(fitness)
        
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        env.close()
        return -0.0


def main():
    parser = argparse.ArgumentParser(description='Optuna hyperparameter search for PPO drone training')
    parser.add_argument('--task', type=str, default='figure8',
                       choices=['circle', 'figure8', 'slalom', 'backandforth'],
                       help='Flight task to optimize for')
    parser.add_argument('--n-trials', type=int, default=50,
                       help='Number of Optuna trials to run')
    parser.add_argument('--timesteps-per-trial', type=float, default=1e6,
                       help='Training timesteps per trial (smaller = faster search)')
    parser.add_argument('--num-envs', type=int, default=10,
                       help='Number of parallel environments per trial')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda:0', 'cuda:1'],
                       help='Device to use for training')
    parser.add_argument('--study-name', type=str, default='ppo_drone_optimization',
                       help='Name for the Optuna study')
    parser.add_argument('--storage', type=str, default=None,
                       help='Database URL for Optuna storage (e.g., sqlite:///optuna.db)')
    parser.add_argument('--n-jobs', type=int, default=1,
                       help='Number of parallel jobs (use >1 for parallel optimization)')
    parser.add_argument('--output-dir', type=str, default='optuna_results',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Example drone design (simple quadcopter)
    example_individual = np.array([
        [0.25,  np.pi/4,   0.0,  0.0,  0.0,  1.0],  # Front-right motor (CW)
        [0.25, -np.pi/4,   0.0,  0.0,  0.0,  0.0],  # Front-left motor (CCW)
        [0.25,  3*np.pi/4, 0.0,  0.0,  0.0,  0.0],  # Back-left motor (CCW)  
        [0.25, -3*np.pi/4, 0.0,  0.0,  0.0,  1.0],  # Back-right motor (CW)
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],  # Unused
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]   # Unused
    ], dtype=np.float32)
    
    print(f"\nOptuna Hyperparameter Search Configuration:")
    print(f"  Task: {args.task}")
    print(f"  Number of trials: {args.n_trials}")
    print(f"  Timesteps per trial: {int(args.timesteps_per_trial):,}")
    print(f"  Parallel environments: {args.num_envs}")
    print(f"  Device: {args.device}")
    print(f"  Study name: {args.study_name}")
    print(f"  Parallel jobs: {args.n_jobs}")
    print(f"  Output directory: {args.output_dir}\n")
    
    # Create Optuna study
    if args.storage:
        study = optuna.create_study(
            study_name=args.study_name,
            storage=args.storage,
            direction='minimize',  # Minimize negative fitness
            load_if_exists=True
        )
    else:
        study = optuna.create_study(
            study_name=args.study_name,
            direction='minimize'
        )
    
    # Run optimization
    study.optimize(
        lambda trial: objective(
            trial, 
            example_individual, 
            args.task, 
            args.num_envs, 
            int(args.timesteps_per_trial),
            args.device
        ),
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        show_progress_bar=True
    )
    
    # Print results
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)
    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best fitness (gates passed): {-study.best_value:.2f}")
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Save results
    results = {
        'best_trial': study.best_trial.number,
        'best_fitness': -study.best_value,
        'best_params': study.best_params,
        'task': args.task,
        'timesteps_per_trial': int(args.timesteps_per_trial),
        'num_envs': args.num_envs,
        'device': args.device,
        'n_trials': args.n_trials
    }
    
    with open(os.path.join(args.output_dir, 'best_hyperparameters.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    # Save as text file too
    with open(os.path.join(args.output_dir, 'best_hyperparameters.txt'), 'w') as f:
        f.write(f"Best Trial: {study.best_trial.number}\n")
        f.write(f"Best Fitness: {-study.best_value:.2f}\n\n")
        f.write("Best Hyperparameters:\n")
        for key, value in study.best_params.items():
            f.write(f"  {key}: {value}\n")
    
    print(f"\nResults saved to {args.output_dir}/")
    
    # Try to create visualization
    try:
        import matplotlib.pyplot as plt
        from optuna.visualization import plot_optimization_history, plot_param_importances
        
        # Optimization history
        fig = plot_optimization_history(study)
        fig.write_image(os.path.join(args.output_dir, 'optimization_history.png'))
        
        # Parameter importances
        fig = plot_param_importances(study)
        fig.write_image(os.path.join(args.output_dir, 'param_importances.png'))
        
        print(f"Visualization plots saved to {args.output_dir}/")
    except Exception as e:
        print(f"Could not create visualizations: {e}")
        print("Install plotly and kaleido for visualizations: pip install plotly kaleido")


if __name__ == "__main__":
    main()
