"""Example: Standalone Learning Phase (From the Research Paper).

Demonstrates how an individual morphology is evaluated within the NSGA-II 
framework. The drone is trained using PPO on a purely task-oriented reward 
(Darwinian learning), and then evaluated post-training to extract both the 
gates passed (task performance) and total energy consumed (power efficiency).

This script simulates the evaluation of a single morphology, equivalent to 
one worker process during the evolutionary loop.

Example
-------
  # Train a standard quadcopter morphology and evaluate its energy:
  python examples/learning/run_nsga2_learning_phase.py \
      --training-timesteps 500000 --gate-cfg figure8 --device cpu
"""

import os
import argparse
import numpy as np

# Ensure root directory is in python path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from airevolve.evolution_tools.evaluators.gate_train import evaluate_individual
from airevolve.simulator.simulation.propeller_data import create_standard_propeller_config
from airevolve.evolution_tools.genome_handlers.spherical_angular_genome_handler import SphericalAngularDroneGenomeHandler


def main():
    parser = argparse.ArgumentParser(description="Standalone Learning Evaluation")
    parser.add_argument("--training-timesteps", type=int, default=500000, help="Timesteps to train PPO")
    parser.add_argument("--gate-cfg", type=str, default="figure8", choices=["figure8", "backandforth"], help="Track to fly")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device")
    parser.add_argument("--num-envs", type=int, default=4, help="Parallel environments for PPO")
    args = parser.parse_args()

    print("================================================================")
    print(" NSGA-II Learning Phase Demonstration")
    print("================================================================")
    print(f" Track       : {args.gate_cfg}")
    print(f" Timesteps   : {args.training_timesteps}")
    print(" Algorithm   : PPO (Darwinian, task-only reward)")
    print(" Evaluation  : Post-training flight (Strict LiPo Battery Model)")
    print("================================================================")

    # 1. Create a dummy representation of a standard quadcopter genome
    # format: magnitude, arm_rotation, arm_pitch, motor_rotation, motor_pitch, direction
    arms = []
    directions = [1, -1, 1, -1] # CW, CCW, CW, CCW
    angles = [np.pi/4, 3*np.pi/4, -3*np.pi/4, -np.pi/4]
    for i in range(4):
        arms.append([0.11, angles[i], 0.0, 0.0, 0.0, directions[i]])
    
    genome = np.array(arms)
    handler = SphericalAngularDroneGenomeHandler(genome)

    # Create dummy output directory for the artifacts (model, video, etc)
    save_dir = "results/standalone_learning_demo/"
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n[1] Initializing genome array ({len(genome)} arms).")
    print(f"[2] Beginning RL Training Phase... (saving to {save_dir})")
    
    # 2. Train the drone using the Darwinian task-only reward
    try:
        # evaluate_individual does the entire training + power-aware testing logic
        # It strictly mirrors the EA evaluation step.
        fitness = evaluate_individual(
            genome_handler=handler,
            total_timesteps=args.training_timesteps,
            save_dir=save_dir,
            gate_cfg=args.gate_cfg,
            device=args.device,
            n_envs=args.num_envs,
            n_eval_episodes=5,
            z_drag_multiplier=1.0 # Standard from paper
        )
        
        # 3. Print the Bi-Objective Results
        gates_passed, total_energy = fitness
        print("\n================================================================")
        print(" Evaluation Complete! Bi-Objective Fitness Extracted:")
        print("================================================================")
        print(f" Objective 1 (Maximize) : {gates_passed:.2f} Gates Passed")
        print(f" Objective 2 (Minimize) : {total_energy:.2f} Joules Consumed")
        print("================================================================")
        
    except Exception as e:
        print(f"\n[Error during evaluation]: {e}")


if __name__ == "__main__":
    main()
