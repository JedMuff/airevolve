import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from airevolve.evolution_tools.evaluators.curriculum_train import evaluate_individual_curriculum
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
from airevolve.evolution_tools.genome_handlers.operators.optimization_repair_operator import (
    OptimizationRepairConfig
)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import functools
import argparse
import time
from datetime import datetime
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run evolution with curriculum learning (hover → gate training)'
    )

    # Genome and evolution parameters
    parser.add_argument('--genome-handler', choices=['spherical', 'cartesian'],
                       default='spherical', help='Genome handler to use (default: spherical)')
    parser.add_argument('--population-size', type=int, default=12,
                       help='Population size (default: 30)')
    parser.add_argument('--generations', type=int, default=40,
                       help='Number of generations (default: 30)')
    parser.add_argument('--num-mutate', type=int, default=12,
                       help='Number of individuals to mutate per generation (default: 30)')
    parser.add_argument('--num-crossover', type=int, default=0,
                       help='Number of crossover operations per generation (default: 0)')
    parser.add_argument('--log-dir', default='./.data',
                       help='Directory for logs (default: ./.data)')
    parser.add_argument('--show-plot', action='store_true',
                       help='Show fitness plot at the end')
    parser.add_argument('--save-all-plots', action='store_true',
                       help='Save comprehensive visualization plots including diversity')
    parser.add_argument('--strategy-type', choices=['plus', 'comma'],
                       default='plus', help='Evolution strategy type (default: plus)')
    parser.add_argument('--repair-verbose', action='store_true',
                       help='Print detailed repair progress messages')

    # Hover training parameters (Stage 1)
    parser.add_argument('--hover-max-timesteps', type=float, default=2e6,
                       help='Maximum training timesteps for hover stage (default: 2e6)')
    parser.add_argument('--hover-success-threshold', type=float, default=0.90,
                       help='Hover success rate threshold to advance to gate stage (default: 0.90)')
    parser.add_argument('--hover-window-size', type=int, default=100,
                       help='Window size for hover success tracking (default: 100)')
    parser.add_argument('--hover-check-freq', type=int, default=10000,
                       help='Check hover threshold every N steps (default: 10000)')

    # Gate training parameters (Stage 2)
    parser.add_argument('--gate-timesteps', type=float, default=1e8/4,
                       help='Training timesteps for gate stage (default: 1e8/4)')
    parser.add_argument('--gate-window-size', type=int, default=100,
                       help='Window size for gate progress tracking (default: 100)')
    parser.add_argument('--gate-check-freq', type=int, default=10000,
                       help='Check gate progress every N steps (default: 10000)')

    # Environment parameters
    parser.add_argument('--num-envs', type=int, default=100,
                       help='Number of environments for training (default: 100)')
    parser.add_argument('--gate-cfg', choices=['backandforth', 'figure8', 'circle', 'slalom'],
                       default='figure8', help='Gate configuration (default: figure8)')
    parser.add_argument('--device', default='cuda:0',
                       help='Device for training (default: cuda:0)')
    parser.add_argument('--num-workers', type=int, default=1,
                       help='Number of parallel workers for evaluation (default: 1). '
                            'When >1, automatically uses device=cpu for parallel CPU-based PPO.')

    # Morphology parameters
    parser.add_argument('--min-narms', type=int, default=6,
                       help='Minimum number of arms (default: 6)')
    parser.add_argument('--max-narms', type=int, default=6,
                       help='Maximum number of arms (default: 6)')

    return parser.parse_args()

def _try_generate_individual(args):
    """
    Worker function to try generating a single hoverable individual.

    Returns:
        Tuple of (success_individual, status_dict) where status_dict tracks failures.
    """
    idx, base_seed, handler_kwargs, param_limits, coordinate_system = args

    from airevolve.evolution_tools.genome_handlers.repair_workflow import (
        stage1_optimization_repair,
        stage2_hover_check,
        stage3_hover_repair
    )
    from airevolve.evolution_tools.genome_handlers.operators.optimization_repair_operator import (
        OptimizationRepairConfig
    )

    # Create genome handler with unique seed
    # Combine base_seed (random per run) with idx (unique per individual)
    # This ensures different individuals across runs while maintaining parallelism
    seed = base_seed + idx
    handler = SphericalAngularDroneGenomeHandler(
        **handler_kwargs,
        rnd=np.random.default_rng(seed)
    )

    status = {
        'failed_hover': 0,
        'failed_stage1': 0,
        'failed_stage3': 0,
        'success': 0
    }

    # Generate random individual
    ind = handler.random_population(1)[0]

    # STEP 1: Check if it can hover (strict, no spinning)
    can_hover, _ = stage2_hover_check(
        ind,
        verbose=False,
        allow_spinning=False  # Strict hover check
    )

    if not can_hover:
        status['failed_hover'] = 1
        return None, status

    # STEP 2: Apply optimization repair to fix collisions
    # Fix motor orientation (pitch=3, yaw=4) — only arm positions (r, theta, phi) are moved
    repair_config = OptimizationRepairConfig(fixed_params=[3, 4])
    repaired, _ = stage1_optimization_repair(
        ind,
        coordinate_system=coordinate_system,
        config=repair_config,
        verbose=False
    )

    if repaired is None:
        status['failed_stage1'] = 1
        return None, status

    # STEP 3: Apply hover repair to align thrust vectors
    final_ind, _ = stage3_hover_repair(
        repaired,
        coordinate_system=coordinate_system,
        verbose=False
    )

    if final_ind is None:
        status['failed_stage3'] = 1
        return None, status

    # Success!
    status['success'] = 1
    return final_ind, status


def generate_initial_pop_parallel(genotype, pop_size, coordinate_system='spherical', verbose=False, num_workers=None):
    """
    Generate initial population using parallel sampling (similar to test_repair_pipeline_stats_parallel.py).

    Strategy:
    1. Sample many individuals in parallel
    2. Keep only those that pass full repair pipeline
    3. Expected success rate: ~0.2% (1/500)

    Args:
        genotype: Genome handler instance
        pop_size: Size of population to generate
        coordinate_system: 'spherical' or 'cartesian'
        verbose: Print detailed messages
        num_workers: Number of parallel workers (defaults to CPU count)

    Returns:
        Array of repaired individuals
    """
    if num_workers is None:
        num_workers = cpu_count()

    print(f"Generating initial population of size {pop_size} using {num_workers} parallel workers...")
    print("Strategy: Parallel sampling → Strict hover check → Fix collisions → Align thrust")
    print("Expected success rate: ~0.2% (need ~{:,} samples for {} individuals)\n".format(pop_size * 500, pop_size))
    sys.stdout.flush()

    start_time = time.time()

    # Generate random base seed for this run
    # This ensures different individuals are generated each time the script runs
    base_seed = np.random.randint(0, 2**31)
    print(f"Base seed for this run: {base_seed}\n")

    # Prepare arguments for parallel workers
    # We'll sample in batches until we have enough
    batch_size = pop_size * 1000  # Sample 100x population size per batch
    max_batches = 100

    successful_individuals = []
    total_stats = {
        'failed_hover': 0,
        'failed_stage1': 0,
        'failed_stage3': 0,
        'success': 0,
        'total_attempts': 0
    }

    for batch_idx in range(max_batches):
        if len(successful_individuals) >= pop_size:
            break

        print(f"\nBatch {batch_idx + 1}: Sampling {batch_size} individuals...")

        # Prepare args for this batch
        # Extract only the necessary config (avoid passing entire __dict__)
        handler_config = {
            'min_max_narms': (genotype.min_narms, genotype.max_narms),
            'append_arm_chance': genotype.append_arm_chance,
            'parameter_limits': genotype.parameter_limits,
            'bilateral_plane_for_symmetry': genotype.bilateral_plane_for_symmetry,
            'repair': genotype.repair_enabled,
        }

        args_list = [
            (
                batch_idx * batch_size + i,  # idx: unique per individual
                base_seed,                    # base_seed: random per run
                handler_config,
                genotype.parameter_limits,
                coordinate_system
            )
            for i in range(batch_size)
        ]

        # Process in parallel with progress bar
        with Pool(processes=num_workers) as pool:
            with tqdm(total=batch_size, desc=f"Batch {batch_idx + 1}", unit="ind") as pbar:
                for result, status in pool.imap_unordered(_try_generate_individual, args_list, chunksize=10):
                    total_stats['total_attempts'] += 1
                    total_stats['failed_hover'] += status['failed_hover']
                    total_stats['failed_stage1'] += status['failed_stage1']
                    total_stats['failed_stage3'] += status['failed_stage3']
                    total_stats['success'] += status['success']

                    if result is not None:
                        successful_individuals.append(result)

                    pbar.update(1)
                    pbar.set_postfix({
                        'found': len(successful_individuals),
                        'rate': f"{total_stats['success']/total_stats['total_attempts']*100:.2f}%"
                    })

                    # Stop early if we have enough
                    if len(successful_individuals) >= pop_size:
                        break

        print(f"  Found {len(successful_individuals)}/{pop_size} so far...")

        if len(successful_individuals) >= pop_size:
            print(f"✓ Target reached!")
            break

    end_time = time.time()

    # Report results
    print(f"\n{'='*80}")
    print("Initial Population Generation Results")
    print(f"{'='*80}")
    print(f"Successfully generated: {len(successful_individuals)}/{pop_size}")
    print(f"Total attempts: {total_stats['total_attempts']:,}")
    print(f"Success rate: {total_stats['success']/total_stats['total_attempts']*100:.3f}%")
    print(f"\nFailure breakdown:")
    print(f"  Failed hover check: {total_stats['failed_hover']:,} ({total_stats['failed_hover']/total_stats['total_attempts']*100:.1f}%)")
    print(f"  Failed Stage 1 (optimization): {total_stats['failed_stage1']:,} ({total_stats['failed_stage1']/total_stats['total_attempts']*100:.1f}%)")
    print(f"  Failed Stage 3 (hover repair): {total_stats['failed_stage3']:,} ({total_stats['failed_stage3']/total_stats['total_attempts']*100:.1f}%)")
    print(f"\nTime taken: {end_time - start_time:.1f}s")
    print(f"{'='*80}\n")

    if len(successful_individuals) < pop_size:
        print(f"⚠ Warning: Could only generate {len(successful_individuals)}/{pop_size} individuals")
        print(f"Consider increasing max_batches or using a different approach.\n")

    return np.array(successful_individuals[:pop_size]) if len(successful_individuals) > 0 else None

def get_genome_handler_config(handler_type, min_narms=6, max_narms=6):
    """
    Get genome handler class and configuration based on type.

    Note: Symmetry is NOT supported in this script (always None).
    Note: Built-in repair is DISABLED (repair=False) - we use external repair workflow.
    """

    # Shared parameter limits for all representations:
    # [magnitude, arm_yaw, arm_pitch, motor_pitch, motor_yaw, direction]
    spherical_params = np.array([
        [0.055, 0.17],           # magnitude
        [-np.pi, np.pi],         # arm yaw (azimuth)
        [-np.pi / 2, np.pi / 2], # arm pitch (elevation)
        [-np.pi, np.pi],         # motor pitch
        [-np.pi, np.pi],         # motor yaw
        [0, 1],                  # direction
    ])

    append_arm_chance = 0.0 if min_narms == max_narms else 0.5

    if handler_type == 'spherical':
        return {
            'handler_class': SphericalAngularDroneGenomeHandler,
            'handler_kwargs': {
                'min_max_narms': (min_narms, max_narms),
                'append_arm_chance': append_arm_chance,
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
                'min_max_narms': (min_narms, max_narms),
                'append_arm_chance': append_arm_chance,
                'bilateral_plane_for_symmetry': None,  # No symmetry
                'repair': False  # Disable built-in repair (use external workflow)
            },
            'param_limits': spherical_params,
            'coordinate_system': 'cartesian'
        }

    else:
        raise ValueError(f"Unknown genome handler type: {handler_type}")


def create_fitness_function(args):
    """Create fitness function for curriculum learning evaluation.

    Uses functools.partial for picklability (required for multiprocessing).
    """
    return functools.partial(
        evaluate_individual_curriculum,
        hover_max_timesteps=int(args.hover_max_timesteps),
        hover_success_threshold=args.hover_success_threshold,
        gate_timesteps=int(args.gate_timesteps),
        num_envs=args.num_envs,
        gate_cfg=args.gate_cfg,
        device=args.device,
        hover_window_size=args.hover_window_size,
        hover_check_freq=args.hover_check_freq,
        gate_window_size=args.gate_window_size,
        gate_check_freq=args.gate_check_freq,
        num=None
    )

def create_genome_handler_wrapper(handler_class, handler_kwargs):
    """Create a wrapper class that provides the correct constructor interface."""
    class GenomeHandlerWrapper(handler_class):
        def __init__(self, *_args, genome=None, **_kwargs):
            # Use our configured parameters, but pass through the genome if provided
            super().__init__(genome=genome, **handler_kwargs)

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
    """Main evolution function with curriculum learning."""
    args = parse_arguments()

    # Generate automatic experiment name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    narms_str = f"{args.min_narms}arms" if args.min_narms == args.max_narms else f"{args.min_narms}-{args.max_narms}arms"
    exp_name = f"curriculum_{args.gate_cfg}_{narms_str}_{timestamp}"

    # Create full log directory path
    full_log_dir = os.path.join(args.log_dir, exp_name)
    os.makedirs(full_log_dir, exist_ok=True)

    print("=" * 80)
    print("Evolution with Curriculum Learning (Hover → Gate Training)")
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
    print(f"Number of arms: {args.min_narms}-{args.max_narms}")
    print(f"Device: {args.device}")
    print(f"Parallel workers: {args.num_workers}")
    print()
    print("CURRICULUM LEARNING PARAMETERS:")
    print(f"  Stage 1 (Hover):")
    print(f"    - Max timesteps: {int(args.hover_max_timesteps):,}")
    print(f"    - Success threshold: {args.hover_success_threshold:.2%}")
    print(f"    - Window size: {args.hover_window_size}")
    print(f"    - Check frequency: {args.hover_check_freq:,}")
    print(f"  Stage 2 (Gate):")
    print(f"    - Timesteps: {int(args.gate_timesteps):,}")
    print(f"    - Window size: {args.gate_window_size}")
    print(f"    - Check frequency: {args.gate_check_freq:,}")
    print(f"    - Number of environments: {args.num_envs}")
    print()
    print(f"Repair workflow: ENABLED (3-stage: Optimization → Hover Check → Hover Repair)")
    print(f"Symmetry: DISABLED (not supported)")
    print(f"CRITICAL: Individuals failing hover threshold ({args.hover_success_threshold:.2%}) receive fitness of 0")
    print("=" * 80)
    print()

    # Switch to CPU when using parallel workers
    if args.num_workers > 1 and args.device != 'cpu':
        print(f"Parallel mode: switching device to 'cpu' for {args.num_workers} workers")
        args.device = 'cpu'

    # Get genome handler configuration
    config = get_genome_handler_config(args.genome_handler, args.min_narms, args.max_narms)

    # Create fitness function
    fitness_function = create_fitness_function(args)

    # Create a wrapper class for the genome handler that uses our configured parameters
    WrappedHandler = create_genome_handler_wrapper(config['handler_class'], config['handler_kwargs'])

    # Generate initial population with parallel repair workflow
    print("=" * 80)
    print("Phase 1: Initial Population Generation (Parallel)")
    print("=" * 80)
    initial_population = generate_initial_pop_parallel(
        WrappedHandler(),
        args.population_size,
        coordinate_system=config['coordinate_system'],
        verbose=args.repair_verbose,
        num_workers=32  # Use all CPUs
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
        num_workers=args.num_workers,
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
    ax.set_title(f"Curriculum Learning Fitness Evolution ({args.genome_handler.upper()} Handler, {args.gate_cfg})")
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
            title=f"Curriculum Learning Evolution Summary ({args.genome_handler.upper()} Handler, {args.gate_cfg})",
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
