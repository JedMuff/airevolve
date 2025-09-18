#!/usr/bin/env python3
"""
Simple evaluation script for a single drone design.
"""

import numpy as np
import pickle
import os
import time
from airevolve.evolution_tools.evaluators.gate_train import evaluate_individual

def evaluate_drone(individual, task='circle', output_dir='evaluation_results', 
                  timesteps=int(1e8), num_envs=100, create_videos=False):
    """
    Evaluate a single drone design on a specified task.
    
    Args:
        individual (np.ndarray): Drone design array (6x6 format)
        task (str): Task name ('circle', 'figure8', 'slalom', 'backandforth')
        output_dir (str): Directory to save results
        timesteps (int): Training timesteps
        num_envs (int): Number of parallel environments
        create_videos (bool): Whether to create visualization videos after training
    
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
        fitness = evaluate_individual(individual, output_dir, timesteps, num_envs, task, device='cpu')
        
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
    
    # Example: Use a simple quadcopter design (4 motors in X configuration)
    example_individual = np.array([
        [0.25,  np.pi/4,   0.0,  0.0,  0.0,  1.0],  # Front-right motor (CW)
        [0.25, -np.pi/4,   0.0,  0.0,  0.0,  0.0],  # Front-left motor (CCW)
        [0.25,  3*np.pi/4, 0.0,  0.0,  0.0,  0.0],  # Back-left motor (CCW)  
        [0.25, -3*np.pi/4, 0.0,  0.0,  0.0,  1.0],  # Back-right motor (CW)
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],  # Unused
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]   # Unused
    ])
    
    # Evaluate the drone
    fitness = evaluate_drone(
        individual=example_individual,
        task='figure8',
        output_dir='simple_evaluation',
        timesteps=int(1e8),  # Much smaller for debugging
        num_envs=1,          # Single environment for debugging
        create_videos=True   # Enable video creation after training
    )
    
    print(f"Final fitness: {fitness}")


if __name__ == "__main__":
    main()