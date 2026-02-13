import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from airevolve.evolution_tools.evaluators.gate_train import evaluate_individual
from airevolve.evolution_tools.strategies.mu_lambda import evolve
from airevolve.evolution_tools.selectors.tournament import tournament_selection
from airevolve.evolution_tools.genome_handlers.spherical_angular_genome_handler import SphericalAngularDroneGenomeHandler
from airevolve.evolution_tools.genome_handlers.cartesian_euler_genome_handler import CartesianEulerDroneGenomeHandler
from airevolve.evolution_tools.inspection_tools.utils import evolution_dataframe_to_fitness_array
from airevolve.evolution_tools.inspection_tools.plot_fitness import plot_fitness
from airevolve.evolution_tools.genome_handlers.repair_workflow import (
    repair_operation_process,
    is_nan_individual
)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import functools
import argparse
import time
from datetime import datetime

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run evolution with optimization-based repair workflow'
    )

    parser.add_argument('--genome-handler', choices=['spherical', 'cartesian'],
                       default='spherical', help='Genome handler to use (default: spherical)')
    parser.add_argument('--population-size', type=int, default=12,
                       help='Population size (default: 12)')
    parser.add_argument('--generations', type=int, default=40,
                       help='Number of generations (default: 40)')
    parser.add_argument('--num-mutate', type=int, default=12,
                       help='Number of individuals to mutate per generation (default: 12)')
    parser.add_argument('--num-crossover', type=int, default=0,
                       help='Number of crossover operations per generation (default: 0)')
    parser.add_argument('--log-dir', default='./logs',
                       help='Directory for logs (default: ./logs)')
    parser.add_argument('--show-plot', action='store_true',
                       help='Show fitness plot at the end')
    parser.add_argument('--save-all-plots', action='store_true',
                       help='Save comprehensive visualization plots including diversity')
    parser.add_argument('--training-timesteps', type=float, default=1E6,
                       help='Training timesteps for gate evaluation (default: 1E6)')
    parser.add_argument('--num-envs', type=int, default=2,
                       help='Number of environments for gate training (default: 2)')
    parser.add_argument('--gate-cfg', choices=['backandforth', 'figure8', 'circle', 'slalom'],
                       default='figure8', help='Gate configuration (default: figure8)')
    parser.add_argument('--device', default='cuda:0',
                       help='Device for training (default: cuda:0)')
    parser.add_argument('--strategy-type', choices=['plus', 'comma'],
                       default='plus', help='Evolution strategy type (default: plus)')
    parser.add_argument('--repair-verbose', action='store_true',
                       help='Print detailed repair progress messages')

    return parser.parse_args()

def generate_initial_pop(genotype, pop_size, coordinate_system='spherical', verbose=False):
    """
    Generate initial population using the three-stage repair workflow.

    Args:
        genotype: Genome handler instance
        pop_size: Size of population to generate
        coordinate_system: 'spherical' or 'cartesian'
        verbose: Print detailed repair messages

    Returns:
        Array of repaired individuals
    """
    start_time = time.time()
    pop = []
    failed_count = 0
    attempts = 0
    max_attempts = pop_size * 100  # Prevent infinite loops

    print(f"Generating initial population of size {pop_size} with repair workflow...")
    print("Repair stages: 1) Optimization Repair → 2) Hover Check → 3) Hover Repair\n")
    sys.stdout.flush()

    while len(pop) < pop_size and attempts < max_attempts:
        attempts += 1

        if verbose:
            print(f"Attempt {attempts}: Generating and repairing individual...")
            sys.stdout.flush()

        # Generate random individual
        ind = genotype.random_population(1)[0]

        # Apply three-stage repair workflow
        repaired_ind, status_msg = repair_operation_process(
            ind,
            coordinate_system=coordinate_system,
            verbose=verbose
        )

        # Check if repair succeeded
        if not is_nan_individual(repaired_ind):
            pop.append(repaired_ind)
            if verbose or len(pop) % max(1, pop_size // 10) == 0:
                print(f"  [{len(pop)}/{pop_size}] Individual added. Status: {status_msg}")
        else:
            failed_count += 1
            if verbose:
                print(f"  [Failed {failed_count}] {status_msg}")

    end_time = time.time()

    if len(pop) < pop_size:
        print(f"\n⚠ Warning: Could only generate {len(pop)}/{pop_size} individuals after {attempts} attempts")
        print(f"  Failed repairs: {failed_count}")
    else:
        print(f"\n✓ Successfully generated {len(pop)} individuals")
        print(f"  Total attempts: {attempts}")
        print(f"  Failed repairs: {failed_count}")
        print(f"  Success rate: {len(pop)/attempts*100:.1f}%")

    print(f"  Time taken: {end_time - start_time:.1f}s\n")

    return np.array(pop) if len(pop) > 0 else None

def get_genome_handler_config(handler_type):
    """
    Get genome handler class and configuration based on type.

    Note: Symmetry is NOT supported in this script (always None).
    Note: Built-in repair is DISABLED (repair=False) - we use external repair workflow.
    """

    # Spherical coordinate parameters: [magnitude, arm_rotation, arm_pitch, motor_rotation, motor_pitch, direction]
    spherical_params = np.array([[0.055,0.105], [-np.pi, np.pi] ,[-2*np.pi/3, 2*np.pi/3], [-np.pi, np.pi], [-np.pi, np.pi], [0,1]])

    if handler_type == 'spherical':
        return {
            'handler_class': SphericalAngularDroneGenomeHandler,
            'handler_kwargs': {
                'min_max_narms': (3, 6),
                'append_arm_chance': 0.0,
                'parameter_limits': spherical_params,
                'bilateral_plane_for_symmetry': None,  # No symmetry
                'repair': False  # Disable built-in repair (use external workflow)
            },
            'param_limits': spherical_params,
            'coordinate_system': 'spherical'
        }

    elif handler_type == 'cartesian':
        return {
            'handler_class': CartesianEulerDroneGenomeHandler,
            'handler_kwargs': {
                'min_max_narms': (6, 6),
                'append_arm_chance': 0.0,
                'bilateral_plane_for_symmetry': None,  # No symmetry
                'repair': False  # Disable built-in repair (use external workflow)
            },
            'param_limits': spherical_params,
            'coordinate_system': 'cartesian'
        }

    else:
        raise ValueError(f"Unknown genome handler type: {handler_type}")


def create_fitness_function(args):
    """Create fitness function for gate training evaluation."""
    def fitness_function_wrapper(genome_array, log_dir):
        """
        Wrapper function that converts raw genome arrays to the format expected by evaluate_individual.
        """
        return evaluate_individual(genome_array, log_dir, args.training_timesteps, args.num_envs, 
                                 args.gate_cfg, args.device, num=None)
    
    return fitness_function_wrapper

def create_genome_handler_wrapper(handler_class, handler_kwargs):
    """Create a wrapper class that provides the correct constructor interface."""
    class GenomeHandlerWrapper(handler_class):
        def __init__(self, *_args, **_kwargs):
            # Ignore args/kwargs from evolution function and use our configured parameters
            super().__init__(**handler_kwargs)

    return GenomeHandlerWrapper

def convert_dataframe_to_population_data(all_individuals, population_size, num_generations):
    """
    Convert evolution DataFrame to population data format for diversity calculation.
    
    Args:
        all_individuals: DataFrame from evolve() with columns ['generation', 'genome', 'fitness', 'in_pop']
        population_size: Size of population per generation
        num_generations: Number of generations
        
    Returns:
        Tuple of (population_data, fitness_data) as lists of arrays
    """
    population_data = []
    fitness_data = []
    
    for gen in range(num_generations):
        # Get individuals from this generation that were in the population
        gen_data = all_individuals[
            (all_individuals['generation'] == gen) & 
            (all_individuals['in_pop'] == True)
        ].sort_values(by='fitness', ascending=False)
        
        # Extract genomes and fitnesses
        genomes = []
        fitnesses = []
        
        for _, row in gen_data.iterrows():
            genomes.append(row['genome'])
            fitnesses.append(row['fitness'])
        
        # Pad with NaN if necessary to maintain consistent shape
        while len(genomes) < population_size:
            genomes.append(np.full_like(genomes[0] if genomes else np.array([[np.nan]*6]), np.nan))
            fitnesses.append(np.nan)
        
        population_data.append(genomes[:population_size])
        fitness_data.append(np.array(fitnesses[:population_size]))
    
    return population_data, fitness_data

def main():
    """Main evolution function with optimization-based repair workflow."""
    args = parse_arguments()

    # Generate automatic experiment name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"optrepair_{args.gate_cfg}_{timestamp}"

    # Create full log directory path
    full_log_dir = os.path.join(args.log_dir, exp_name)
    os.makedirs(full_log_dir, exist_ok=True)

    print("=" * 80)
    print("Evolution with Optimization-Based Repair Workflow")
    print("=" * 80)
    sys.stdout.flush()
    print(f"Experiment name: {exp_name}")
    print(f"Log directory: {full_log_dir}")
    print(f"Genome handler: {args.genome_handler.upper()}")
    print(f"Population size: {args.population_size}")
    print(f"Generations: {args.generations}")
    print(f"Mutations per generation: {args.num_mutate}")
    print(f"Crossovers per generation: {args.num_crossover}")
    print(f"Strategy type: {args.strategy_type}")
    print(f"Gate configuration: {args.gate_cfg}")
    print(f"Training timesteps: {args.training_timesteps}")
    print(f"Number of environments: {args.num_envs}")
    print(f"Device: {args.device}")
    print(f"Repair workflow: ENABLED (3-stage: Optimization → Hover Check → Hover Repair)")
    print(f"Symmetry: DISABLED (not supported)")
    print("=" * 80)
    print()

    # Get genome handler configuration
    config = get_genome_handler_config(args.genome_handler)

    # Create fitness function
    fitness_function = create_fitness_function(args)

    # Create a wrapper class for the genome handler that uses our configured parameters
    WrappedHandler = create_genome_handler_wrapper(config['handler_class'], config['handler_kwargs'])

    # Generate initial population with repair workflow
    print("=" * 80)
    print("Phase 1: Initial Population Generation")
    print("=" * 80)
    initial_population = generate_initial_pop(
        WrappedHandler(),
        args.population_size,
        coordinate_system=config['coordinate_system'],
        verbose=args.repair_verbose
    )

    if initial_population is None or len(initial_population) == 0:
        print("\n✗ Failed to generate initial population. Exiting.")
        return

    # Run evolution
    print("=" * 80)
    print("Phase 2: Evolution")
    print("=" * 80)
    all_individuals = evolve(
        fitness_function=fitness_function,
        population_size=args.population_size,
        num_generations=args.generations,
        num_mutate=args.num_mutate,
        num_crossover=args.num_crossover,
        mutate_after_crossover=True,
        strategy_type=args.strategy_type,
        parent_selection=tournament_selection,
        genome_handler=WrappedHandler,
        log_dir=full_log_dir,
        initial_population=initial_population,
    )
    
    # Save complete evolution data as CSV
    evolution_csv_path = f"{full_log_dir}/evolution_data.csv"
    # Ensure ID column stays as string
    all_individuals_copy = all_individuals.copy()
    all_individuals_copy['id'] = all_individuals_copy['id'].astype(str)
    all_individuals_copy.to_csv(evolution_csv_path, index=False)
    print(f"Evolution data saved to: {evolution_csv_path}")
    
    # Get the best individual from the last generation
    last_gen = args.generations - 1
    best_individual = all_individuals.loc[all_individuals['generation'] == last_gen].sort_values(by='fitness', ascending=False).iloc[0]
    print(f"\nBest individual in generation {last_gen}: {best_individual['id']}, Fitness: {best_individual['fitness']} gates passed")
    
    print(f"Genome: \n{best_individual['genome']}")
    
    # Plot fitness evolution
    print("\nGenerating fitness evolution plot...")
    fitness_array = evolution_dataframe_to_fitness_array(all_individuals, population_size=args.population_size)

    _fig, ax = plt.subplots(figsize=(12, 8))
    plot_fitness(ax, fitness_array)
    ax.set_title(f"Gate Training Fitness Evolution ({args.genome_handler.upper()} Handler, {args.gate_cfg})")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Number of Gates Passed")
    plt.tight_layout()
    
    # Save plot
    fitness_plot_path = f"{full_log_dir}/fitness_evolution_{args.genome_handler}_{args.gate_cfg}.png"
    plt.savefig(fitness_plot_path, dpi=300, bbox_inches='tight')
    print(f"Fitness plot saved to: {fitness_plot_path}")
    
    if args.show_plot:
        plt.show()
    
    # Generate additional comprehensive plots if requested
    if args.save_all_plots:
        print("\nGenerating comprehensive visualization plots...")
        
        # Convert dataframe to population data format for diversity plotting
        population_data, fitness_data_array = convert_dataframe_to_population_data(
            all_individuals, args.population_size, args.generations
        )
        
        # Generate diversity plot
        print("Creating diversity plot...")
        _fig_diversity, ax_diversity = plt.subplots(figsize=(10, 6))
        from airevolve.evolution_tools.inspection_tools.plot_diversity import plot_diversity
        plot_diversity(ax_diversity, population_data, parameter_limits=(config['param_limits'][:,0], config['param_limits'][:,1]))
        ax_diversity.set_title(f"Population Diversity Over Generations ({args.genome_handler.upper()} Handler, {args.gate_cfg})")
        plt.tight_layout()
        
        # Save diversity plot
        diversity_plot_path = f"{full_log_dir}/diversity_evolution_{args.genome_handler}_{args.gate_cfg}.png"
        plt.savefig(diversity_plot_path, dpi=300, bbox_inches='tight')
        print(f"Diversity plot saved to: {diversity_plot_path}")
        
        if args.show_plot:
            plt.show()
        else:
            plt.close()
        
        # Generate comprehensive summary plot
        print("Creating evolution summary plot...")
        # Convert to numpy arrays as expected by the summary function
        population_array = np.array(population_data)
        fitness_array = np.array(fitness_data_array)
        
        from airevolve.evolution_tools.inspection_tools.evolution_plotters import create_evolution_summary_plot
        fig_summary = create_evolution_summary_plot(
            population_array, 
            fitness_array,
            title=f"Gate Training Evolution Summary ({args.genome_handler.upper()} Handler, {args.gate_cfg})",
            parameter_limits=(config['param_limits'][:,0], config['param_limits'][:,1])
        )
        
        # Save summary plot
        summary_plot_path = f"{full_log_dir}/evolution_summary_{args.genome_handler}_{args.gate_cfg}.png"
        fig_summary.savefig(summary_plot_path, dpi=300, bbox_inches='tight')
        print(f"Evolution summary plot saved to: {summary_plot_path}")
        
        if args.show_plot:
            plt.show()
        else:
            plt.close(fig_summary)
    
    print(f"\nEvolution completed! Best fitness: {best_individual['fitness']} gates passed")

if __name__ == "__main__":
    main()