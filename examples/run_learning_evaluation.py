#!/usr/bin/env python3
"""
Simple evaluation script for a single drone design.
"""

import numpy as np
import pickle
import os
import time
import argparse
from airevolve.evolution_tools.evaluators.gate_train import evaluate_individual

def evaluate_drone(individual, task='circle', output_dir='evaluation_results', 
                  timesteps=int(1e8), num_envs=100, create_videos=False, device='cpu'):
    """
    Evaluate a single drone design on a specified task.
    
    Args:
        individual (np.ndarray): Drone design array (6x6 format)
        task (str): Task name ('circle', 'figure8', 'slalom', 'backandforth')
        output_dir (str): Directory to save results
        timesteps (int): Training timesteps
        num_envs (int): Number of parallel environments
        create_videos (bool): Whether to create visualization videos after training
        device (str): Device to use for training ('cpu' or 'cuda:0')
    
    Returns:
        float: Fitness score
    """
    
    # Validate task
    valid_tasks = ['backandforth', 'figure8', 'circle', 'slalom']
    if task not in valid_tasks:
        raise ValueError(f"Task must be one of {valid_tasks}, got '{task}'")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the individual
    np.save(os.path.join(output_dir, "individual.npy"), individual)
    
    # Save metadata
    metadata = {
        'task': task,
        'timesteps': timesteps,
        'num_envs': num_envs,
        'start_time': time.time(),
        'individual_shape': individual.shape
    }
    
    with open(os.path.join(output_dir, "metadata.pkl"), 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"Starting evaluation on task: {task}")
    print(f"Individual shape: {individual.shape}")
    print(f"Results will be saved to: {output_dir}")
    
    start_time = time.time()
    
    try:
        # Run the evaluation
        fitness = evaluate_individual(individual, output_dir, timesteps, num_envs, task, device=device)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Save results
        results = {
            'fitness': fitness,
            'duration': duration,
            'success': True,
            'task': task,
            'end_time': end_time
        }
        
        with open(os.path.join(output_dir, "results.pkl"), 'wb') as f:
            pickle.dump(results, f)
        
        # Also save fitness as text for easy reading
        with open(os.path.join(output_dir, "fitness.txt"), 'w') as f:
            f.write(f"{fitness}\n")
        
        print(f"Evaluation completed successfully!")
        print(f"Fitness: {fitness}")
        print(f"Duration: {duration:.2f} seconds")
        
        # Create videos if requested
        if create_videos:
            try:
                from make_video import process_individual
                print("Creating visualization videos...")
                process_individual(output_dir, gate_cfg=task)
                print("Videos created successfully!")
            except ImportError:
                print("Warning: make_video module not found. Videos not created.")
            except Exception as e:
                print(f"Warning: Failed to create videos: {e}")
        
        return fitness
        
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        
        # Save error information
        results = {
            'fitness': None,
            'duration': duration,
            'success': False,
            'error': str(e),
            'task': task,
            'end_time': end_time
        }
        
        with open(os.path.join(output_dir, "results.pkl"), 'wb') as f:
            pickle.dump(results, f)
        
        print(f"Evaluation failed after {duration:.2f} seconds")
        print(f"Error: {e}")
        raise


def main():
    """
    Example usage with one of the predefined designs.
    """
    parser = argparse.ArgumentParser(description='Evaluate a drone design on a flight task')
    parser.add_argument('--task', type=str, default='figure8',
                       choices=['circle', 'figure8', 'slalom', 'backandforth'],
                       help='Flight task to evaluate on')
    parser.add_argument('--timesteps', type=float, default=1e5,
                       help='Number of training timesteps (e.g., 1e5, 1e7, 1e8)')
    parser.add_argument('--num-envs', type=int, default=1,
                       help='Number of parallel environments')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda:0', 'cuda:1'],
                       help='Device to use for training (cpu recommended for MLP policies)')
    parser.add_argument('--output-dir', type=str, default='simple_evaluation',
                       help='Directory to save results')
    parser.add_argument('--no-videos', action='store_true',
                       help='Disable video creation after training')
    parser.add_argument('--production', action='store_true',
                       help='Use production settings (1e8 timesteps, 100 envs, cpu)')
    
    args = parser.parse_args()
    
    # Override with production settings if requested
    if args.production:
        print("Using production settings: 1e8 timesteps, 100 parallel envs, CPU device")
        args.timesteps = 1e8
        args.num_envs = 100
        args.device = 'cpu'
    
    # Example: Use a simple quadcopter design (4 motors in X configuration)
    example_individual = np.array([
        [0.25,  np.pi/4,   0.0,  0.0,  0.0,  1.0],  # Front-right motor (CW)
        [0.25, -np.pi/4,   0.0,  0.0,  0.0,  0.0],  # Front-left motor (CCW)
        [0.25,  3*np.pi/4, 0.0,  0.0,  0.0,  0.0],  # Back-left motor (CCW)  
        [0.25, -3*np.pi/4, 0.0,  0.0,  0.0,  1.0],  # Back-right motor (CW)
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],  # Unused
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]   # Unused
    ])
    
    print(f"\nConfiguration:")
    print(f"  Task: {args.task}")
    print(f"  Timesteps: {int(args.timesteps):,}")
    print(f"  Parallel environments: {args.num_envs}")
    print(f"  Device: {args.device}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Create videos: {not args.no_videos}\n")
    
    # Evaluate the drone
    fitness = evaluate_drone(
        individual=example_individual,
        task=args.task,
        output_dir=args.output_dir,
        timesteps=int(args.timesteps),
        num_envs=args.num_envs,
        create_videos=not args.no_videos,
        device=args.device
    )
    
    print(f"\nFinal fitness: {fitness}")


if __name__ == "__main__":
    main()