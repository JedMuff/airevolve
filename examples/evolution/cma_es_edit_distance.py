#!/usr/bin/env python3
"""
CMA-ES Optimization 

This version focuses solely on CMA-ES optimization for drone evolution.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path 
import time
from typing import List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

import cma

from airevolve.evolution_tools.evaluators.edit_distance import evaluate_individual
from airevolve.evolution_tools.inspection_tools.drone_visualizer import DroneVisualizer

def create_target_drone(genome_type="Cartesian") -> np.ndarray:
    """
    Create a target drone configuration for optimization.
    
    Returns:
        Target drone as numpy array
    """
    if genome_type == "Cartesian":
        target = np.array([
            [0.2, 0.2, 0.2, 0.0, 0.0, 0.0, 0],      # Arm 1: front-right
            [0.2, -0.2, 0.2, 0.0, 0.0, 0.0, 1],     # Arm 2: front-left
            [-0.2, -0.2, -0.2, 0.0, 0.0, 0.0, 0],   # Arm 3: back-left
            [-0.2, 0.2, -0.2, 0.0, 0.0, 0.0, 1],    # Arm 4: back-right
        ])
    else:
        # Create a simple quadcopter configuration
        target = np.array([
            [0.2, -np.pi/4, 0.0, 0.0, 0.0, 0],      # Arm 1: front-right
            [0.2, -3*np.pi/4, 0.0, 0.0, 0.0, 1],    # Arm 2: front-left
            [0.2, -5*np.pi/4, 0.0, 0.0, 0.0, 0],    # Arm 3: back-left
            [0.2, -7*np.pi/4, 0.0, 0.0, 0.0, 1],    # Arm 4: back-right
        ])
        # Wrap the angles
        target[:, 1] = np.mod(target[:, 1] + np.pi, 2 * np.pi) - np.pi
    return target

class GenomeWrapper:
    def __init__(self, genome):
        self.body = genome

def vector_to_genome_handler(param_vector: np.ndarray, genome_type="Cartesian", narms=4) -> Any:
    if "Cartesian" in genome_type:
        genome = param_vector.reshape(narms, 7)  # 7 parameters per arm
    elif "Spherical" in genome_type:
        genome = param_vector.reshape(narms, 6)  # 6 parameters per arm

    return GenomeWrapper(genome)

def create_fitness_function(genome_type, target, param_limits):
    """Create fitness function for the given target and parameter limits."""
    def fitness_function(param_vector):
        """
        Evaluate fitness for a parameter vector.
        """
        handler = vector_to_genome_handler(
            param_vector, 
            genome_type=genome_type, 
            narms=target.shape[0]
        )
        
        return -evaluate_individual(handler, None, target, 
                                  param_limits[:,0], param_limits[:,1])
    
    return fitness_function

class CMAESWrapper:
    """CMA-ES wrapper for drone optimization"""
    
    def __init__(self, genome_type="Cartesian", narms: int = 4):
        if genome_type == "Cartesian":
            self.genome_limits = np.array([
                [-0.4, 0.4], [-np.pi, np.pi], [-np.pi, np.pi],
                [-np.pi, np.pi], [-np.pi, np.pi], [-np.pi, np.pi],
                [0, 1]
            ])
            self.lower_bounds = np.tile(self.genome_limits[:, 0], narms)
            self.upper_bounds = np.tile(self.genome_limits[:, 1], narms)

        elif genome_type == "Spherical":
            self.genome_limits = np.array([
                [0.09, 0.4], [-np.pi, np.pi], [-np.pi, np.pi],
                [-np.pi, np.pi], [-np.pi, np.pi], [0, 1]
            ])
            self.lower_bounds = np.tile(self.genome_limits[:, 0], narms)
            self.upper_bounds = np.tile(self.genome_limits[:, 1], narms)

        self.narms = narms
        self.genome_type = genome_type

        self.fitness_function = create_fitness_function(
            genome_type=genome_type,
            target=create_target_drone(genome_type),
            param_limits=self.genome_limits
        )
    
    def get_initial_parameters(self) -> np.ndarray:
        params = np.random.uniform(
            self.lower_bounds,
            self.upper_bounds,
            size=(len(self.lower_bounds),)
        )
        return params

def run_cma_es_optimization(wrapper: CMAESWrapper, target: np.ndarray, 
                           genome_name: str, max_generations: int = 50, target_fitness=0.01) -> dict:
    """
    Run CMA-ES optimization for a specific genome type.
    """
    print(f"\n{'='*60}")
    print(f"Starting CMA-ES optimization for {genome_name}")
    print(f"{'='*60}")
    
    # Get initial parameters and bounds
    initial_params = wrapper.get_initial_parameters()

    # Set up CMA-ES options
    opts = {
        'bounds': [wrapper.lower_bounds, wrapper.upper_bounds],
        'popsize':30,
        'maxiter': max_generations,
        'tolfun': 1e-8,
        'tolx': 1e-8,
        'ftarget': target_fitness, 
        'seed': 42,
        'verb_disp': 100,
    }
    
    # Initial standard deviation
    sigma0 = np.mean(wrapper.upper_bounds - wrapper.lower_bounds) / 4.0
    
    # Initialize CMA-ES
    es = cma.CMAEvolutionStrategy(initial_params, sigma0, opts)
    
    # Track optimization progress
    fitness_history = []
    best_fitness = float('inf')
    best_individual = None
    generation = 0
    
    start_time = time.time()
    
    print(f"Initial sigma: {sigma0:.6f}")
    print(f"Population size: {opts['popsize']}")
    print(f"Parameter vector size: {len(initial_params)}")
    
    # Optimization loop
    while not es.stop():
        generation += 1
        
        # Generate new population
        solutions = es.ask()
        
        # Evaluate fitness for each solution
        fitness_values = []
        for sol in solutions:
            fitness = wrapper.fitness_function(sol)
            fitness_values.append(fitness)
        
        # Update CMA-ES with fitness values
        es.tell(solutions, fitness_values)
        
        # Track best solution
        gen_best_idx = np.argmin(fitness_values)
        gen_best_fitness = fitness_values[gen_best_idx]
        
        if gen_best_fitness < best_fitness:
            best_fitness = gen_best_fitness
            best_individual = solutions[gen_best_idx].copy()
        
        fitness_history.append({
            'generation': generation,
            'best_fitness': gen_best_fitness,
            'mean_fitness': np.mean(fitness_values),
            'std_fitness': np.std(fitness_values),
            'overall_best': best_fitness
        })
        
        # Print progress
        if generation % 100 == 0:
            print(f"Gen {generation:3d}: Best={gen_best_fitness:.6f}, "
                  f"Mean={np.mean(fitness_values):.6f}, "
                  f"Overall Best={best_fitness:.6f}")
    
    elapsed_time = time.time() - start_time
    
    print(f"\nOptimization completed for {genome_name}")
    print(f"Total generations: {generation}")
    print(f"Total time: {elapsed_time:.2f} seconds")
    print(f"Best fitness: {best_fitness:.6f}")
    print(f"Termination reason: {es.stop()}")
    
    best_drone = vector_to_genome_handler(
        best_individual,
        genome_type=genome_name,
        narms=target.shape[0]
    ).body
    print(f"Best individual: \n{np.round(best_drone,2)}")
    print()
    print(f"Target drone: \n{np.round(target,2)}")
    
    return {
        'genome_name': genome_name,
        'optimizer_type': 'CMA-ES',
        'best_fitness': best_fitness,
        'best_individual': best_individual,
        'fitness_history': fitness_history,
        'total_generations': generation,
        'elapsed_time': elapsed_time,
        'termination_reason': str(es.stop())
    }

def visualize_results(results: List[dict], target: np.ndarray, save_plots: bool = True):
    """
    Visualize CMA-ES optimization results and final drone designs.
    """
    print(f"\n{'='*60}")
    print("Visualizing Results")
    print(f"{'='*60}")
    
    # Create output directory
    output_dir = Path("cmaes_optimization_results")
    output_dir.mkdir(exist_ok=True)
    
    # Plot fitness convergence
    plt.figure(figsize=(12, 8))
    
    colors = ['blue', 'green', 'red']
    
    for i, result in enumerate(results):
        history = result['fitness_history']
        generations = [h['generation'] for h in history]
        best_fitness = [h['overall_best'] for h in history]
        
        label = f"{result['genome_name']} - Final: {result['best_fitness']:.4f}"
        
        plt.semilogy(generations, best_fitness, 
                    label=label, 
                    linewidth=2,
                    color=colors[i % len(colors)])
    
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness (log scale)')
    plt.title('CMA-ES Optimization Results')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_plots:
        plt.savefig(output_dir / "fitness_convergence.png", dpi=300, bbox_inches='tight')
    
    # Visualize best drone designs
    visualizer = DroneVisualizer()
    
    # Create subplot for drone visualizations
    n_results = len(results)
    n_cols = 3
    n_rows = (n_results + n_cols) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows), subplot_kw={'projection': '3d'})
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    # Plot target drone
    visualizer.plot_3d(target, title="Target Drone", ax=axes[0])

    # Plot best designs from each optimizer
    for i, result in enumerate(results):
        if i >= len(axes) - 1:
            break
        
        narms = target.shape[0]
        drone_data = vector_to_genome_handler(result['best_individual'], 
                                              genome_type=result['genome_name'], 
                                              narms=narms).body

        title = f"{result['genome_name']}\nFitness: {result['best_fitness']:.4f}"
        if 'elapsed_time' in result:
            title += f"\nTime: {result['elapsed_time']:.1f}s"
        
        visualizer.plot_3d(drone_data, title=title, ax=axes[i+1])
    
    # Hide unused subplots
    for i in range(len(results) + 1, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    if save_plots:
        plt.savefig(output_dir / "best_designs.png", dpi=300, bbox_inches='tight')
    
    # Summary statistics
    print(f"\n{'='*80}")
    print("CMA-ES OPTIMIZATION SUMMARY")
    print(f"{'='*80}")
    
    # Print detailed results
    for result in results:
        print(f"\n{result['genome_name']}:")
        print(f"  Best Fitness: {result['best_fitness']:.6f}")
        print(f"  Generations: {result['total_generations']}")
        print(f"  Time: {result['elapsed_time']:.2f} seconds")
        print(f"  Time/Gen: {result['elapsed_time']/max(1, result['total_generations']):.3f} s")
        print(f"  Termination: {result['termination_reason']}")
    
    # Find best overall result
    best_result = min(results, key=lambda x: x['best_fitness'])
    print(f"\nBest Overall Result: {best_result['genome_name']} "
          f"(Fitness: {best_result['best_fitness']:.6f}, Time: {best_result['elapsed_time']:.1f}s)")

def main():
    """
    Main function for CMA-ES optimization.
    """
    print("CMA-ES Optimization for AirEvolve2")
    print("=" * 80)
    print("Running CMA-ES optimization on Cartesian and Spherical genome types")
    print()

    # Main CMA-ES optimization - only Cartesian and Spherical
    genome_types = ["Cartesian", "Spherical"]

    results = []
    
    for genome_type in genome_types:
        target = create_target_drone(genome_type)
        narms = target.shape[0]
        
        # Run CMA-ES optimization
        print(f"\n{'='*80}")
        print(f"PROCESSING {genome_type.upper()} GENOME")
        print(f"{'='*80}")
        
        cmaes_wrapper = CMAESWrapper(
            genome_type=genome_type, 
            narms=narms
        )
        cmaes_result = run_cma_es_optimization(
            wrapper=cmaes_wrapper,
            target=target,
            genome_name=genome_type,
            max_generations=10000
        )
        results.append(cmaes_result)
    
    # Visualize and analyze results
    if results:
        # Use the target from the last genome type for visualization
        visualize_results(results, target, save_plots=True)
    else:
        print("No successful optimizations to analyze.")

if __name__ == "__main__":
    main()