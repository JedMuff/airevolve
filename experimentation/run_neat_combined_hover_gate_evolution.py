"""
NEAT Combined Hover + Gate Fitness Evolution

Same fitness pipeline as run_combined_hover_gate_evolution.py but uses NEAT
(evolve_neat) with speciation instead of (Mu+Lambda).

Fitness pipeline per individual:
  1. Decode CPPN -> phenotype (if indirect encoding)
  2. stage1_optimization_repair (fix collisions)
  3. Compute continuous hover_fitness on optimization-repaired phenotype [0,3]
  4. stage2_hover_check (can it hover?)
     - Fails -> return hover_fitness (gradient signal)
     - Passes -> full repair from original phenotype, then CMA-ES gate eval
  5. Return hover_fitness + gates_passed

Usage:
    python experimentation/run_neat_combined_hover_gate_evolution.py \\
        --genome-handler cppn --population-size 20 --generations 10 \\
        --max-evals 100 --gate-cfg circle
"""

import sys
import os

# Limit BLAS/OpenMP threads to 1 per process — must be set before numpy import.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from airevolve.evolution_tools.strategies.neat import evolve_neat
from airevolve.evolution_tools.selectors.tournament import tournament_selection
from airevolve.evolution_tools.inspection_tools.utils import evolution_dataframe_to_fitness_array
from airevolve.evolution_tools.inspection_tools.plot_fitness import plot_fitness
from airevolve.evolution_tools.genome_handlers.cppn.network import CPPNNetwork
from airevolve.evolution_tools.genome_handlers.hybrid_cppn_genome_handler import (
    HybridGenome,
    _N_CPPN_INPUTS as _N_HYBRID_CPPN_INPUTS,
    _N_CPPN_OUTPUTS as _N_HYBRID_CPPN_OUTPUTS,
)

from examples.run_evolution_with_lee_tuning import (
    get_genome_handler_config,
    create_genome_handler_wrapper,
)
from airevolve.evolution_tools.genome_handlers.spherical_angular_genome_handler import (
    SphericalNeatGenome,
)
from experimentation.run_combined_hover_gate_evolution import (
    _CombinedHoverGateFitness,
    create_empty_cppn,
    create_empty_hybrid_genome,
    _N_INPUTS,
    _N_OUTPUTS,
)


# ============================================================================
# NEAT FITNESS WRAPPER
# ============================================================================

class _NeatCombinedFitness(_CombinedHoverGateFitness):
    """Thin wrapper that unwraps SphericalNeatGenome before evaluation.

    In NEAT, the spherical handler produces SphericalNeatGenome objects
    (not raw ndarrays).  The base fitness class expects an ndarray for
    non-indirect encodings, so we extract .arms here.
    """

    def __call__(self, genome, ind_save_dir):
        if isinstance(genome, SphericalNeatGenome):
            # Strip NaN-padded rows to get the active arms
            arms = genome.arms
            active = ~np.isnan(arms).any(axis=1)
            return super().__call__(arms[active], ind_save_dir)
        return super().__call__(genome, ind_save_dir)


# ============================================================================
# ARGUMENT PARSING
# ============================================================================

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run NEAT evolution with combined hover + gate fitness'
    )

    # Genome and evolution parameters
    parser.add_argument('--genome-handler', choices=['spherical', 'cartesian', 'cppn', 'hybrid-cppn'],
                       default='cppn', help='Genome handler to use (default: cppn)')
    parser.add_argument('--population-size', type=int, default=16,
                       help='Population size (default: 16)')
    parser.add_argument('--generations', type=int, default=50,
                       help='Number of generations (default: 50)')
    parser.add_argument('--log-dir', default='./.data',
                       help='Directory for logs (default: ./.data)')
    parser.add_argument('--show-plot', action='store_true',
                       help='Show fitness plot at the end')

    # NEAT-specific parameters
    parser.add_argument('--crossover-rate', type=float, default=0.75,
                       help='Fraction of offspring from crossover (default: 0.75)')
    parser.add_argument('--compatibility-threshold', type=float, default=3.0,
                       help='Compatibility distance threshold (default: 3.0)')
    parser.add_argument('--target-species-count', type=int, default=5,
                       help='Target number of species (default: 5)')
    parser.add_argument('--stagnation-limit', type=int, default=15,
                       help='Generations without improvement before species removal (default: 15)')

    # Lee controller tuning parameters (2-Stage CMA-ES)
    parser.add_argument('--max-evals', type=int, default=500,
                       help='Maximum CMA-ES evaluations per individual (default: 500)')
    parser.add_argument('--cma-workers', type=int, default=1,
                       help='Number of parallel workers for CMA-ES (default: 1)')
    parser.add_argument('--sim-time', type=float, default=20.0,
                       help='Simulation time in seconds (default: 20.0)')
    parser.add_argument('--dt', type=float, default=0.005,
                       help='Time step in seconds (default: 0.005)')
    parser.add_argument('--timeout', type=float, default=30.0,
                       help='Timeout per evaluation in seconds (default: 30.0)')
    parser.add_argument('--gate-cfg', choices=['backandforth', 'figure8', 'circle', 'slalom'],
                       default='figure8', help='Gate configuration (default: figure8)')

    # Evolution workers
    parser.add_argument('--num-workers', type=int, default=32,
                       help='Number of parallel workers for evolution (default: 32)')

    # Morphology parameters
    parser.add_argument('--min-narms', type=int, default=6,
                       help='Minimum number of arms (default: 6)')
    parser.add_argument('--max-narms', type=int, default=6,
                       help='Maximum number of arms (default: 6)')

    # CPPN-specific parameters
    parser.add_argument('--num-segments', type=int, default=8,
                       help='Number of CPPN evaluation segments (default: 8, CPPN only)')
    parser.add_argument('--initial-hidden-nodes', type=int, default=0,
                       help='Initial hidden nodes in CPPN topology (default: 0, CPPN only)')

    return parser.parse_args()


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main evolution function with NEAT + combined hover+gate fitness."""
    args = parse_arguments()

    # Generate automatic experiment name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    narms_str = f"{args.min_narms}arms" if args.min_narms == args.max_narms else f"{args.min_narms}-{args.max_narms}arms"
    exp_name = f"neat_combined_hover_gate_{args.gate_cfg}_{narms_str}_{timestamp}"

    full_log_dir = os.path.join(args.log_dir, exp_name)
    os.makedirs(full_log_dir, exist_ok=True)

    print("=" * 80)
    print("NEAT Combined Hover + Gate Fitness Evolution")
    print("=" * 80)
    print(f"Experiment: {exp_name}")
    print(f"Log dir: {full_log_dir}")
    print(f"Handler: {args.genome_handler.upper()}, Arms: {args.min_narms}-{args.max_narms}")
    print(f"Pop: {args.population_size}, Gens: {args.generations}")
    print(f"Crossover rate: {args.crossover_rate}, Compat threshold: {args.compatibility_threshold}")
    print(f"Target species: {args.target_species_count}, Stagnation limit: {args.stagnation_limit}")
    print(f"Gate: {args.gate_cfg}, CMA-ES evals: {args.max_evals}, Workers: {args.num_workers}")
    print(f"Initial population: empty CPPNs (no connections)")
    print("Fitness = hover_fitness [0,3] + gates_passed [0,N]")
    print("=" * 80)
    print()

    # Get genome handler configuration
    config = get_genome_handler_config(
        args.genome_handler, args.min_narms, args.max_narms,
        num_segments=args.num_segments,
        initial_hidden_nodes=args.initial_hidden_nodes,
    )

    # Create combined fitness function
    is_indirect = args.genome_handler in ('cppn', 'hybrid-cppn')
    fitness_function = _NeatCombinedFitness(
        gate_cfg=args.gate_cfg,
        max_evals=args.max_evals,
        cma_workers=args.cma_workers,
        sim_time=args.sim_time,
        dt=args.dt,
        timeout=args.timeout,
        coordinate_system=config['coordinate_system'],
        is_indirect=is_indirect,
        handler_class=config['handler_class'] if is_indirect else None,
        handler_kwargs=config['handler_kwargs'] if is_indirect else None,
    )

    # Create genome handler wrapper
    WrappedHandler = create_genome_handler_wrapper(config['handler_class'], config['handler_kwargs'])

    # Generate initial population of empty CPPNs / hybrid genomes
    if args.genome_handler == 'hybrid-cppn':
        initial_population = [create_empty_hybrid_genome(narms=args.min_narms)
                              for _ in range(args.population_size)]
        print(f"Generated {len(initial_population)} empty hybrid genomes "
              f"({args.min_narms} arms, {_N_HYBRID_CPPN_INPUTS} CPPN inputs, "
              f"{_N_HYBRID_CPPN_OUTPUTS} CPPN outputs, 0 connections)")
    elif args.genome_handler == 'cppn':
        initial_population = [create_empty_cppn() for _ in range(args.population_size)]
        print(f"Generated {len(initial_population)} empty CPPNs "
              f"({_N_INPUTS} inputs, {_N_OUTPUTS} outputs, 0 connections)")
    else:
        # For direct encodings, generate random genomes via the handler
        handler = WrappedHandler()
        initial_population = [h.genome for h in handler.generate_random_population(args.population_size)]
        print(f"Generated {len(initial_population)} random individuals")

    # Run NEAT evolution
    print()
    all_individuals = evolve_neat(
        fitness_function=fitness_function,
        population_size=args.population_size,
        num_generations=args.generations,
        crossover_rate=args.crossover_rate,
        parent_selection=tournament_selection,
        genome_handler=WrappedHandler,
        compatibility_threshold=args.compatibility_threshold,
        target_species_count=args.target_species_count,
        stagnation_limit=args.stagnation_limit,
        species_elitism=1,
        min_species_size=2,
        adjust_threshold=True,
        interspecies_mating_rate=0.001,
        mutate_after_crossover=True,
        initial_population=initial_population,
        log_dir=full_log_dir,
        verbose=True,
        num_workers=args.num_workers,
    )

    # Save evolution data
    evolution_csv_path = f"{full_log_dir}/evolution_data.csv"
    all_individuals_copy = all_individuals.copy()
    all_individuals_copy['id'] = all_individuals_copy['id'].astype(str)
    all_individuals_copy.to_csv(evolution_csv_path, index=False)
    print(f"Evolution data saved to: {evolution_csv_path}")

    # Best individual
    last_gen = args.generations
    best = all_individuals.loc[all_individuals['generation'] == last_gen].sort_values(
        by='fitness', ascending=False
    ).iloc[0]
    print(f"\nBest in gen {last_gen}: {best['id']}, Fitness: {best['fitness']}")

    # Plot fitness
    fitness_array = evolution_dataframe_to_fitness_array(all_individuals, population_size=args.population_size)
    _fig, ax = plt.subplots(figsize=(12, 8))
    plot_fitness(ax, fitness_array)
    ax.set_title(f"NEAT Combined Hover+Gate Evolution ({args.genome_handler.upper()}, {args.gate_cfg})")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness (hover_fitness + gates_passed)")
    plt.tight_layout()

    fitness_plot_path = f"{full_log_dir}/fitness_evolution_{args.genome_handler}_{args.gate_cfg}.png"
    plt.savefig(fitness_plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {fitness_plot_path}")

    if args.show_plot:
        plt.show()

    print(f"\nDone! Best fitness: {best['fitness']}")


if __name__ == "__main__":
    main()
