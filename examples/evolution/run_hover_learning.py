#!/usr/bin/env python3
"""
Simple hover learning script for a single drone design.

This script trains a drone to hover at a target position, which serves as
a foundational task for curriculum learning before moving to more complex
tasks like gate racing.
"""

import numpy as np
import pickle
import os
import time
from airevolve.evolution_tools.evaluators.hover_train import evaluate_individual


def evaluate_drone_hovering(individual,
                           output_dir='hover_evaluation_results',
                           timesteps=int(1e7),
                           num_envs=100,
                           difficulty='easy',
                           create_videos=False):
    """
    Evaluate a single drone design on the hovering task.

    Args:
        individual (np.ndarray): Drone design array (6x6 format)
        output_dir (str): Directory to save results
        timesteps (int): Training timesteps
        num_envs (int): Number of parallel environments
        difficulty (str): Difficulty level ('easy', 'medium', 'hard')
        create_videos (bool): Whether to create visualization videos after training

    Returns:
        float: Hover success rate (0-1)
    """

    # Validate difficulty
    valid_difficulties = ['easy', 'medium', 'hard']
    if difficulty not in valid_difficulties:
        raise ValueError(f"Difficulty must be one of {valid_difficulties}, got '{difficulty}'")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save the individual
    np.save(os.path.join(output_dir, "individual.npy"), individual)

    # Save metadata
    metadata = {
        'difficulty': difficulty,
        'timesteps': timesteps,
        'num_envs': num_envs,
        'start_time': time.time(),
        'individual_shape': individual.shape
    }

    with open(os.path.join(output_dir, "metadata.pkl"), 'wb') as f:
        pickle.dump(metadata, f)

    print(f"Starting hover learning evaluation")
    print(f"Difficulty: {difficulty}")
    print(f"Individual shape: {individual.shape}")
    print(f"Results will be saved to: {output_dir}")

    start_time = time.time()

    try:
        # Run the evaluation
        hover_success_rate = evaluate_individual(
            individual,
            output_dir,
            timesteps,
            num_envs,
            difficulty,
            device='cpu'
        )

        end_time = time.time()
        duration = end_time - start_time

        # Save results
        results = {
            'hover_success_rate': hover_success_rate,
            'duration': duration,
            'success': True,
            'difficulty': difficulty,
            'end_time': end_time
        }

        with open(os.path.join(output_dir, "results.pkl"), 'wb') as f:
            pickle.dump(results, f)

        # Also save hover success rate as text for easy reading
        with open(os.path.join(output_dir, "hover_success_rate.txt"), 'w') as f:
            f.write(f"{hover_success_rate}\n")

        print(f"Evaluation completed successfully!")
        print(f"Hover Success Rate: {hover_success_rate:.2%}")
        print(f"Duration: {duration:.2f} seconds")

        # Create videos if requested
        if create_videos:
            try:
                # Add examples directory to path if needed
                import sys
                examples_dir = os.path.dirname(os.path.abspath(__file__))
                if examples_dir not in sys.path:
                    sys.path.insert(0, examples_dir)

                from make_hover_video import process_individual
                print("\nCreating visualization videos...")
                videos = process_individual(output_dir, difficulty=difficulty)
                print("Videos created successfully!")
            except ImportError as ie:
                print(f"Warning: make_hover_video module not found. Videos not created.")
                print(f"Import error: {ie}")
                print("Make sure make_hover_video.py is in the examples directory.")
            except Exception as e:
                print(f"Warning: Failed to create videos: {e}")
                import traceback
                traceback.print_exc()

        return hover_success_rate

    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time

        # Save error information
        results = {
            'hover_success_rate': None,
            'duration': duration,
            'success': False,
            'error': str(e),
            'difficulty': difficulty,
            'end_time': end_time
        }

        with open(os.path.join(output_dir, "results.pkl"), 'wb') as f:
            pickle.dump(results, f)

        print(f"Evaluation failed after {duration:.2f} seconds")
        print(f"Error: {e}")
        raise


def main():
    """
    Example usage with a simple quadcopter design.
    """

    # Example: Use a simple quadcopter design (4 motors in X configuration)
    example_individual = np.array([
        [0.07,  np.pi/4,   0.0,  0.0,  0.0,  1.0],  # Front-right motor (CW)
        [0.07, -np.pi/4,   0.0,  0.0,  0.0,  0.0],  # Front-left motor (CCW)
        [0.07,  3*np.pi/4, 0.0,  0.0,  0.0,  0.0],  # Back-left motor (CCW)
        [0.07, -3*np.pi/4, 0.0,  0.0,  0.0,  1.0],  # Back-right motor (CW)
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],  # Unused
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]   # Unused
    ])

    # Evaluate the drone on hovering task
    hover_success_rate = evaluate_drone_hovering(
        individual=example_individual,
        output_dir='hover_evaluation',
        timesteps=int(1e6 * 2),      # 1 million timesteps for quick testing
        num_envs=100,            # 100 parallel environments (vectorized)
        difficulty='easy',       # Start with easy difficulty
        create_videos=True       # Enable video creation after training
    )

    print(f"\nFinal hover success rate: {hover_success_rate:.2%}")
    print("\nNext steps for curriculum learning:")
    print("1. Train on 'easy' difficulty until high success rate (>90%)")
    print("2. Progress to 'medium' difficulty with disturbances")
    print("3. Progress to 'hard' difficulty with strong disturbances")
    print("4. Use trained hover policy as initialization for gate racing tasks")


if __name__ == "__main__":
    main()
