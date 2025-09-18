#!/usr/bin/env python3
"""
Main evaluation script for drone designs across multiple tasks.
This script handles individual evaluations and is called by SLURM jobs.
"""

import numpy as np
import pickle
import os
import sys
import time
import argparse
# Note: Orie3DSym and Orie3D have been replaced with airevolve.evolution_tools.genome_handlers.spherical_angular_genome_handler
from airevolve.evolution_tools.strategies.mu_lambda import MuLambdaEvolution
from airevolve.evolution_tools.evaluators.gate_train import evaluate_individual
from airevolve.evolution_tools.inspection_tools.morphological_descriptors.hovering_info import get_sim

# Define all drone designs
def get_drone_designs():
    """Return dictionary of all drone designs"""
    
    # Rank 2 - Fitness: 41.000000 - Generation: 37
    asym_slalom_design_rank_02 = np.array([[ 0.4     ,  1.89446 ,  0.801666, -1.570796,  0.      ,  1.      ],
     [ 0.329549,  1.0915  ,  0.889086, -2.501271, -1.751686,  1.      ],
     [ 0.276753, -0.971952,  1.357406,  0.548501, -1.781706,  1.      ],
     [ 0.4     ,  1.449467,  0.1723  ,  1.678308, -1.832867,  0.      ],
     [ 0.354759,  1.570796, -0.565578, -1.881148, -2.170079,  1.      ],
     [ 0.374488, -0.383297,  0.584533, -0.151673, -0.078287,  1.      ]])

    # Rank 1 - Fitness: 26.000000 - Generation: 31
    asym_shuttlerun_design_rank_01 = np.array([[ 0.344964,  2.603963, -0.761325,  3.654638,  0.054631,  1.      ],
     [ 0.4     ,  4.310401, -1.114352,  3.575547,  0.064733,  0.      ],
     [ 0.318794,  0.942187, -0.086741,  1.527007,  2.375427,  0.      ],
     [ 0.126851,  0.636773,  1.465724,  0.155476,  1.291502,  0.      ],
     [ 0.349252,  3.006736,  0.947084,  3.902128,  2.89007 ,  1.      ],
     [ 0.299382,  0.286999,  0.449577,  0.962113,  0.355372,  0.      ]])

    # Rank 1 - Fitness: 40.000000 - Generation: 35
    asym_circle_design_rank_01 = np.array([[ 0.340822, -0.539542, -0.451414,  6.096941, -1.176661,  0.      ],
     [ 0.221664, -2.426965, -0.735865,  0.191393, -0.780863,  1.      ],
     [ 0.327957,  1.450648,  0.544781,  0.727472, -1.004778,  0.      ],
     [ 0.213118, -0.145561, -0.12256 , -2.38425 , -1.196305,  1.      ],
     [ 0.225842,  1.833185, -0.730086, -2.496145, -3.073945,  0.      ],
     [ 0.150381,  0.670137,  0.08913 ,  0.655402, -1.215598,  1.      ]])

    # Rank 1 - Fitness: 26.000000 - Generation: 23
    asym_figure8_design_rank_01 = np.array([[ 0.096679,  3.540168,  0.458683, -1.547709, -1.505117,  1.      ],
     [ 0.224846, -0.566428,  1.010784,  0.982794,  2.904744,  1.      ],
     [ 0.178852, -1.616751, -0.497266,  1.389464, -1.928233,  1.      ],
     [ 0.287111,  0.021663,  0.441443, -0.667844, -0.861259,  0.      ],
     [ 0.337033,  0.758134,  0.810865,  2.997894, -2.018742,  0.      ],
     [ 0.32918 , -2.306165,  0.825378,  2.553069, -0.013092,  0.      ]])

    # Spider Hexacopter
    Lf, Lm, Lr = 0.26, 0.24, 0.22
    spiderhex = np.array([
        [Lf, 11*np.pi/6, 0.0, 0.0, 0.0, 1.0],  # front-right 330°
        [Lf,    np.pi/6, 0.0, 0.0, 0.0, 0.0],  # front-left  30°
        [Lm, 11*np.pi/18, 0.0, 0.0, 0.0, 1.0], # mid-left   110°
        [Lm, 25*np.pi/18, 0.0, 0.0, 0.0, 0.0], # mid-right  250°
        [Lr,   155*np.pi/180, 0.0, 0.0, 0.0, 1.0], # rear-left  150°
        [Lr,   205*np.pi/180, 0.0, 0.0, 0.0, 0.0], # rear-right 210°
    ])
    
    crosshexcopter = np.array([[0.24, np.pi/3, 0.0, 0.0, 0.0, 1.0],
                            [0.24, 2*np.pi/3, 0.0, 0.0, 0.0, 0.0],
                            [0.24, np.pi, 0.0, 0.0, 0.0, 1.0],
                            [0.24, 4*np.pi/3, 0.0, 0.0, 0.0, 0.0],
                            [0.24, 5*np.pi/3, 0.0, 0.0, 0.0, 1.0],
                            [0.24, 0.0, 0.0, 0.0, 0.0, 0.0]])

    sym_circle_design_rank_01 = np.array([[ 0.202709, -2.484418,  0.774754, -2.965614, -2.661134,  1.      ],
                                    [ 0.4     ,  3.122542,  1.141333,  0.495574, -0.821297,  1.      ],
                                    [ 0.195239,  2.323065,  0.796576, -1.377989, -1.139204,  1.      ],
                                    [ 0.202709, -0.657175,  0.774754, -0.175978, -2.661134,  0.      ],
                                    [ 0.4     ,  0.019051,  1.141333,  2.646018, -0.821297,  0.      ],
                                    [ 0.195239,  0.818528,  0.796576, -1.763604, -1.139204,  0.      ]])

    sym_figure8_design_rank_01 = np.array([[ 0.359335, -2.387032,  0.752666, -2.01062 , -0.730203,  0.      ],
                            [ 0.4     ,  2.219551,  0.969661,  2.926972, -0.877654,  1.      ],
                            [ 0.365247,  0.002004,  0.274898,  1.570796, -3.141593,  1.      ],
                            [ 0.359335, -0.754561,  0.752666, -1.130973, -0.730203,  1.      ],
                            [ 0.4     ,  0.922042,  0.969661,  0.21462 , -0.877654,  0.      ],
                            [ 0.365247,  3.139589,  0.274898, -1.570796, -3.141593,  0.      ]])

    sym_shuttlerun_design_rank_01 = np.array([[ 0.336354, -2.98105 , -1.190237,  0.593774, -2.385773,  1.      ],
                            [ 0.292263,  0.45299 ,  0.376814, -0.005449, -0.55884 ,  1.      ],
                            [ 0.170953, -0.620938,  0.521514, -2.924926, -0.622975,  0.      ],
                            [ 0.336354, -0.160543, -1.190237,  2.547818, -2.385773,  0.      ],
                            [ 0.292263,  2.688602,  0.376814, -3.136144, -0.55884 ,  0.      ],
                            [ 0.170953, -2.520654,  0.521514, -0.216667, -0.622975,  1.      ]])

    sym_slalom_design_rank_01 = np.array([[ 0.350312, -2.113661, -0.690451,  0.024832, -1.442763,  1.      ],
                                    [ 0.35096 ,  0.215668,  0.025228, -0.51232 , -0.322239,  1.      ],
                                    [ 0.285844,  3.141593,  1.183405, -2.874088, -2.751085,  1.      ],
                                    [ 0.350312, -1.027932, -0.690451,  3.116761, -1.442763,  0.      ],
                                    [ 0.35096 ,  2.925924,  0.025228, -2.629273, -0.322239,  0.      ],
                                    [ 0.285844,  0.      ,  1.183405, -0.267505, -2.751085,  0.      ]])

    return {
        # 'asym_slalom_design_rank_02': asym_slalom_design_rank_02,
        # 'asym_shuttlerun_design_rank_01': asym_shuttlerun_design_rank_01,
        # 'asym_circle_design_rank_01': asym_circle_design_rank_01,
        # 'asym_figure8_design_rank_01': asym_figure8_design_rank_01,
        # 'spiderhex': spiderhex,
        # 'crosshexcopter' : crosshexcopter,
        'sym_circle_design_rank_01' : sym_circle_design_rank_01,
        'sym_figure8_design_rank_01' : sym_figure8_design_rank_01,
        'sym_shuttlerun_design_rank_01' : sym_shuttlerun_design_rank_01,
        'sym_slalom_design_rank_01' : sym_slalom_design_rank_01
    }

def get_task_configs():
    """Return task configurations"""
    return {
        'backandforth': 'backandforth',
        'figure8': 'figure8', 
        'circle': 'circle',
        'slalom': 'slalom'
    }

def main():
    parser = argparse.ArgumentParser(description='Evaluate drone design on specific task')
    parser.add_argument('--design', type=str, required=True, help='Design name')
    parser.add_argument('--task', type=str, required=True, help='Task name')
    parser.add_argument('--rep', type=int, required=True, help='Repetition number (0-9)')
    parser.add_argument('--base_dir', type=str, default='drone_evaluations', help='Base directory for results')
    
    args = parser.parse_args()
    
    # Timestamp
    # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # experiment_name = f"{args.base_dir}_{timestamp}"
    # print(experiment_name)

    # Configuration
    timesteps_per_evaluation = int(1e8)
    training_ts = timesteps_per_evaluation
    num_envs = 100
    
    # Get designs and tasks
    designs = get_drone_designs()
    tasks = get_task_configs()
    
    # Validate inputs
    if args.design not in designs:
        print(f"Error: Design '{args.design}' not found. Available: {list(designs.keys())}")
        sys.exit(1)
        
    if args.task not in tasks:
        print(f"Error: Task '{args.task}' not found. Available: {list(tasks.keys())}")
        sys.exit(1)
        
    if not 0 <= args.rep <= 9:
        print(f"Error: Repetition must be 0-9, got {args.rep}")
        sys.exit(1)
    
    # Create directory structure: base_dir/design_name/task_name/rep_X
    result_dir = os.path.join(args.base_dir, args.design, args.task, f"rep_{args.rep}")
    os.makedirs(result_dir, exist_ok=True)
    
    # Save the individual design
    individual = designs[args.design]
    np.save(os.path.join(result_dir, "individual.npy"), individual)
    
    # Save metadata
    metadata = {
        'design': args.design,
        'task': args.task,
        'repetition': args.rep,
        'timesteps': training_ts,
        'num_envs': num_envs,
        'start_time': time.time()
    }
    
    with open(os.path.join(result_dir, "metadata.pkl"), 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"Starting evaluation: {args.design} on {args.task}, rep {args.rep}")
    print(f"Results will be saved to: {result_dir}")
    
    start_time = time.time()
    
    try:
        # Run the evaluation
        gate_cfg = tasks[args.task]
        fitness = evaluate_individual(individual, result_dir, training_ts, num_envs, gate_cfg)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Save results
        results = {
            'fitness': fitness,
            'duration': duration,
            'success': True,
            'end_time': end_time
        }
        
        with open(os.path.join(result_dir, "results.pkl"), 'wb') as f:
            pickle.dump(results, f)
        
        # Also save fitness as text for easy reading
        with open(os.path.join(result_dir, "fitness.txt"), 'w') as f:
            f.write(f"{fitness}\n")
        
        print(f"Evaluation completed successfully!")
        print(f"Design: {args.design}")
        print(f"Task: {args.task}")
        print(f"Repetition: {args.rep}")
        print(f"Fitness: {fitness}")
        print(f"Duration: {duration:.2f} seconds")
        
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        
        # Save error information
        results = {
            'fitness': None,
            'duration': duration,
            'success': False,
            'error': str(e),
            'end_time': end_time
        }
        
        with open(os.path.join(result_dir, "results.pkl"), 'wb') as f:
            pickle.dump(results, f)
        
        print(f"Evaluation failed after {duration:.2f} seconds")
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()