"""
NEAT Edit Distance Evolution

Runs NEAT evolution on the edit-distance target-matching task for any of the
three genome handlers (spherical, cppn, hybrid-cppn), selected via CLI argument.
After evolution, generates fitness/diversity plots and genome visualizations of
the best individual.

Usage:
    # Quick smoke test
    python examples/run_neat_edit_distance.py --handler spherical --generations 5 --population-size 10

    # Full run
    python examples/run_neat_edit_distance.py --handler spherical --generations 50 --population-size 30
"""

import sys
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from airevolve.evolution_tools.evaluators.edit_distance import (
    compute_edit_distance,
    compute_population_edit_distances,
)
from airevolve.evolution_tools.strategies.neat import evolve_neat
from airevolve.evolution_tools.selectors.tournament import tournament_selection
from airevolve.evolution_tools.genome_handlers.spherical_angular_genome_handler import (
    SphericalAngularDroneGenomeHandler,
)
from airevolve.evolution_tools.genome_handlers.cppn_neat_genome_handler import (
    CPPNNeatDroneGenomeHandler,
)
from airevolve.evolution_tools.genome_handlers.hybrid_cppn_genome_handler import (
    HybridCPPNDroneGenomeHandler,
)
from airevolve.evolution_tools.inspection_tools.drone_visualizer import (
    DroneVisualizer,
    VisualizationConfig,
)

# Genome visualization helpers (imported lazily to avoid fcl dependency issues)
import types
if 'fcl' not in sys.modules:
    sys.modules['fcl'] = types.ModuleType('fcl')

from experimentation.plot_genome_visualization import (
    draw_cppn_graph,
    draw_phenotype_heatmap,
    SPHERICAL_COLUMNS,
    HYBRID_DIRECT_COLUMNS,
)
from airevolve.evolution_tools.genome_handlers.cppn.network import CPPNNetwork
from airevolve.evolution_tools.genome_handlers.hybrid_cppn_genome_handler import HybridGenome


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run NEAT evolution on edit-distance target matching"
    )

    parser.add_argument(
        "--handler", required=True,
        choices=["spherical", "cppn", "hybrid-cppn"],
        help="Genome handler type",
    )
    parser.add_argument("--population-size", type=int, default=30)
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--crossover-rate", type=float, default=0.75)
    parser.add_argument("--target-narms", type=int, default=6)
    parser.add_argument("--min-narms", type=int, default=6)
    parser.add_argument("--max-narms", type=int, default=6)
    parser.add_argument("--num-segments", type=int, default=8,
                        help="CPPN evaluation segments (CPPN/hybrid only)")
    parser.add_argument("--initial-hidden-nodes", type=int, default=0,
                        help="Initial hidden nodes in CPPN topology")
    parser.add_argument("--compatibility-threshold", type=float, default=3.0)
    parser.add_argument("--target-species-count", type=int, default=5)
    parser.add_argument("--stagnation-limit", type=int, default=15)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log-dir", default="./.data/neat_results")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--show-plot", action="store_true")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Target drone creation
# ---------------------------------------------------------------------------

def create_target_drone(narms: int) -> np.ndarray:
    """Return a standard drone in spherical (narms, 6) format."""
    radius = 0.08
    if narms == 4:
        azimuths = np.array([np.pi / 4, 3 * np.pi / 4,
                             -3 * np.pi / 4, -np.pi / 4])
    else:
        azimuths = np.linspace(0, 2 * np.pi, narms, endpoint=False)
        azimuths = (azimuths + np.pi) % (2 * np.pi) - np.pi

    target = np.zeros((narms, 6))
    for i in range(narms):
        target[i, 0] = radius
        target[i, 1] = azimuths[i]
        target[i, 2] = 0.0
        target[i, 3] = 0.0
        target[i, 4] = 0.0
        target[i, 5] = i % 2
    return target


# ---------------------------------------------------------------------------
# Spherical parameter limits
# ---------------------------------------------------------------------------

SPHERICAL_PARAMS = np.array([
    [0.055, 0.11 + 0.06],   # magnitude
    [-np.pi, np.pi],        # azimuth
    [-np.pi / 2, np.pi / 2],  # pitch
    [-np.pi, np.pi],        # motor_yaw
    [-np.pi / 2, np.pi / 2],  # motor_pitch
    [0, 1],                 # direction
])


# ---------------------------------------------------------------------------
# Fitness function (picklable class for multiprocessing)
# ---------------------------------------------------------------------------

class _EditDistanceFitness:
    """Picklable fitness callable for use with ``evolve_neat()``."""

    def __init__(self, target, param_limits, handler_type, handler_kwargs):
        self.target = target
        self.min_vals = param_limits[:, 0]
        self.max_vals = param_limits[:, 1]
        self.handler_type = handler_type
        self.handler_kwargs = handler_kwargs

    def __call__(self, genome, log_dir):
        phenotype = decode_phenotype(genome, self.handler_type, self.handler_kwargs)
        return -compute_edit_distance(
            self.target, phenotype, self.min_vals, self.max_vals
        )


# ---------------------------------------------------------------------------
# Genome handler configuration
# ---------------------------------------------------------------------------

def get_genome_handler_config(handler_type, min_narms, max_narms,
                              num_segments=8, initial_hidden_nodes=0):
    """Return handler class, kwargs, and metadata for *handler_type*."""
    append_arm_chance = 0.0 if min_narms == max_narms else 0.5

    if handler_type == "spherical":
        return {
            "handler_class": SphericalAngularDroneGenomeHandler,
            "handler_kwargs": {
                "min_max_narms": (min_narms, max_narms),
                "append_arm_chance": append_arm_chance,
                "parameter_limits": SPHERICAL_PARAMS,
                "bilateral_plane_for_symmetry": None,
                "repair": False,
            },
            "handler_type": "spherical",
            "param_limits": SPHERICAL_PARAMS,
        }
    elif handler_type == "cppn":
        return {
            "handler_class": CPPNNeatDroneGenomeHandler,
            "handler_kwargs": {
                "num_segments": num_segments,
                "min_max_narms": (min_narms, max_narms),
                "initial_hidden_nodes": initial_hidden_nodes,
                "parameter_limits": SPHERICAL_PARAMS,
                "repair": False,
            },
            "handler_type": "cppn",
            "param_limits": SPHERICAL_PARAMS,
        }
    elif handler_type == "hybrid-cppn":
        return {
            "handler_class": HybridCPPNDroneGenomeHandler,
            "handler_kwargs": {
                "min_max_narms": (min_narms, max_narms),
                "initial_hidden_nodes": initial_hidden_nodes,
                "parameter_limits": SPHERICAL_PARAMS,
                "repair": False,
            },
            "handler_type": "hybrid-cppn",
            "param_limits": SPHERICAL_PARAMS,
        }
    else:
        raise ValueError(f"Unknown handler type: {handler_type}")


def create_genome_handler_wrapper(handler_class, handler_kwargs):
    """Wrap *handler_class* so ``evolve_neat()`` can call ``handler(genome=X)``.

    The wrapper intentionally overrides the kwargs produced by
    ``_extract_handler_kwargs`` inside ``evolve_neat`` by closing over the
    original *handler_kwargs* supplied here.  This ensures the handler is
    always constructed with the exact configuration the caller specified.
    """
    class GenomeHandlerWrapper(handler_class):
        def __init__(self, *_args, genome=None, **_ignored_kwargs):
            super().__init__(genome=genome, **handler_kwargs)

    return GenomeHandlerWrapper


# ---------------------------------------------------------------------------
# Helper: decode genome → phenotype
# ---------------------------------------------------------------------------

def decode_phenotype(genome, handler_type, handler_kwargs):
    """Convert raw genome → (narms, 6) phenotype array."""
    if handler_type == "spherical":
        # genome may be a SphericalNeatGenome or a raw ndarray
        from airevolve.evolution_tools.genome_handlers.spherical_angular_genome_handler import SphericalNeatGenome
        if isinstance(genome, SphericalNeatGenome):
            return genome.arms
        return genome
    elif handler_type == "cppn":
        h = CPPNNeatDroneGenomeHandler(genome=genome, **handler_kwargs)
        return h.get_phenotype()
    elif handler_type == "hybrid-cppn":
        h = HybridCPPNDroneGenomeHandler(genome=genome, **handler_kwargs)
        return h.get_phenotype()
    else:
        raise ValueError(f"Unknown handler type: {handler_type}")


# ---------------------------------------------------------------------------
# Post-evolution data extraction
# ---------------------------------------------------------------------------

def extract_generation_metrics(result_df, num_generations, handler_type,
                               handler_kwargs, param_limits):
    """Extract per-generation fitness, diversity, and species metrics."""
    n_gens = num_generations + 1
    max_fitness = np.zeros(n_gens)
    mean_fitness = np.zeros(n_gens)
    std_fitness = np.zeros(n_gens)
    n_species = np.zeros(n_gens, dtype=int)
    diversity = np.zeros(n_gens)

    min_vals = param_limits[:, 0]
    max_vals = param_limits[:, 1]

    for gen in range(n_gens):
        gen_pop = result_df[
            (result_df['generation'] == gen) & (result_df['in_pop'] == True)
        ]
        if gen_pop.empty:
            continue

        fitnesses = gen_pop['fitness'].values
        max_fitness[gen] = np.max(fitnesses)
        mean_fitness[gen] = np.mean(fitnesses)
        std_fitness[gen] = np.std(fitnesses)

        if 'species_id' in gen_pop.columns:
            n_species[gen] = gen_pop['species_id'].nunique()

        # Diversity: mean pairwise edit distance among current population
        phenotypes = [
            decode_phenotype(g, handler_type, handler_kwargs)
            for g in gen_pop['genome']
        ]
        phenotype_array = np.array(phenotypes)
        diversity[gen] = np.mean(
            compute_population_edit_distances(phenotype_array, min_vals, max_vals)
        )

        if gen % 10 == 0:
            print(f"  Extracted metrics for generation {gen}/{num_generations}")

    return {
        'max_fitness': max_fitness,
        'mean_fitness': mean_fitness,
        'std_fitness': std_fitness,
        'n_species': n_species,
        'diversity': diversity,
    }


# ---------------------------------------------------------------------------
# Plotting — fitness & diversity
# ---------------------------------------------------------------------------

def plot_fitness_over_generations(metrics, handler_type, log_dir, show):
    """Figure 1: Fitness over generations."""
    generations = np.arange(len(metrics['max_fitness']))

    fig, ax = plt.subplots(figsize=(10, 6))

    # Mean fitness (solid) with std band
    ax.plot(generations, metrics['mean_fitness'], label='Mean Fitness',
            color='tab:blue')
    ax.fill_between(
        generations,
        metrics['mean_fitness'] - metrics['std_fitness'],
        metrics['mean_fitness'] + metrics['std_fitness'],
        alpha=0.2, color='tab:blue',
    )

    # Max fitness (dashed)
    ax.plot(generations, metrics['max_fitness'], '--', label='Max Fitness',
            color='tab:red')

    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness')
    ax.set_title(f'NEAT Edit Distance — {handler_type}')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()

    path = os.path.join(log_dir, f'neat_{handler_type}_fitness.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"Fitness plot saved to {path}")

    if show:
        plt.show()
    plt.close(fig)


def plot_diversity_and_species(metrics, handler_type, log_dir, show):
    """Figure 2: Diversity + species count."""
    generations = np.arange(len(metrics['diversity']))

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Left y-axis: diversity
    ax1.plot(generations, metrics['diversity'], label='Diversity',
             color='tab:blue')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Diversity (Mean Pairwise Edit Distance)', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True)

    # Right y-axis: species count
    ax2 = ax1.twinx()
    ax2.step(generations, metrics['n_species'], where='mid',
             label='Species Count', color='tab:orange', alpha=0.8)
    ax2.set_ylabel('Number of Species', color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    ax1.set_title(f'NEAT Diversity & Species — {handler_type}')
    plt.tight_layout()

    path = os.path.join(log_dir, f'neat_{handler_type}_diversity.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"Diversity plot saved to {path}")

    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plotting — genome visualization
# ---------------------------------------------------------------------------

def plot_best_individual(best_genome, best_fitness, target, handler_type,
                         handler_kwargs, log_dir, show):
    """Visualize the best individual alongside the target drone."""
    viz = DroneVisualizer(
        VisualizationConfig(elevation=30, azimuth=45, show_axis_ticks=False)
    )

    phenotype = decode_phenotype(best_genome, handler_type, handler_kwargs)

    if handler_type == "spherical":
        fig = plt.figure(figsize=(14, 6))
        # Target drone
        ax_target = fig.add_subplot(1, 3, 1, projection='3d')
        viz.plot_3d(target, ax=ax_target, title='Target')
        # Best individual drone
        ax_drone = fig.add_subplot(1, 3, 2, projection='3d')
        viz.plot_3d(phenotype, ax=ax_drone,
                    title=f'Best (fitness={best_fitness:.4f})')
        # Phenotype heatmap
        ax_heatmap = fig.add_subplot(1, 3, 3)
        draw_phenotype_heatmap(phenotype, ax_heatmap,
                               columns=SPHERICAL_COLUMNS,
                               title='Spherical Genome')

    elif handler_type == "cppn":
        fig = plt.figure(figsize=(14, 6))
        # Target drone
        ax_target = fig.add_subplot(1, 3, 1, projection='3d')
        viz.plot_3d(target, ax=ax_target, title='Target')
        # Best individual drone
        ax_drone = fig.add_subplot(1, 3, 2, projection='3d')
        viz.plot_3d(phenotype, ax=ax_drone,
                    title=f'Best (fitness={best_fitness:.4f})')
        # CPPN graph
        ax_cppn = fig.add_subplot(1, 3, 3)
        if isinstance(best_genome, CPPNNetwork):
            draw_cppn_graph(best_genome, ax_cppn)
        else:
            ax_cppn.text(0.5, 0.5, 'No CPPN genotype\navailable',
                         ha='center', va='center',
                         transform=ax_cppn.transAxes, fontsize=12)
            ax_cppn.axis('off')

    elif handler_type == "hybrid-cppn":
        fig = plt.figure(figsize=(20, 6))
        # Target drone
        ax_target = fig.add_subplot(1, 4, 1, projection='3d')
        viz.plot_3d(target, ax=ax_target, title='Target')
        # Best individual drone
        ax_drone = fig.add_subplot(1, 4, 2, projection='3d')
        viz.plot_3d(phenotype, ax=ax_drone,
                    title=f'Best (fitness={best_fitness:.4f})')
        # Direct parameters heatmap
        ax_direct = fig.add_subplot(1, 4, 3)
        # CPPN graph
        ax_cppn = fig.add_subplot(1, 4, 4)

        if isinstance(best_genome, HybridGenome):
            draw_phenotype_heatmap(best_genome.direct, ax_direct,
                                   columns=HYBRID_DIRECT_COLUMNS,
                                   title='Direct Parameters')
            draw_cppn_graph(best_genome.cppn, ax_cppn)
        else:
            draw_phenotype_heatmap(phenotype, ax_direct,
                                   columns=SPHERICAL_COLUMNS,
                                   title='Phenotype (no genotype)')
            ax_cppn.text(0.5, 0.5, 'No HybridGenome\navailable',
                         ha='center', va='center',
                         transform=ax_cppn.transAxes, fontsize=12)
            ax_cppn.axis('off')

    fig.suptitle(f'NEAT Best Individual — {handler_type}', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    path = os.path.join(log_dir, f'neat_{handler_type}_best_individual.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f"Best individual plot saved to {path}")

    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_arguments()

    if args.seed is not None:
        np.random.seed(args.seed)

    # Timestamped log directory
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"neat_{args.handler}_{args.target_narms}arms_{timestamp}"
    full_log_dir = os.path.join(args.log_dir, exp_name)
    os.makedirs(full_log_dir, exist_ok=True)

    print("=" * 80)
    print("NEAT Edit Distance Evolution")
    print("=" * 80)
    print(f"Handler:              {args.handler}")
    print(f"Population size:      {args.population_size}")
    print(f"Generations:          {args.generations}")
    print(f"Crossover rate:       {args.crossover_rate}")
    print(f"Target arms:          {args.target_narms}")
    print(f"Arm range:            {args.min_narms}-{args.max_narms}")
    print(f"Compat threshold:     {args.compatibility_threshold}")
    print(f"Target species count: {args.target_species_count}")
    print(f"Stagnation limit:     {args.stagnation_limit}")
    print(f"Log directory:        {full_log_dir}")
    print("=" * 80)
    print()

    # Create target drone
    target = create_target_drone(args.target_narms)
    print(f"Target drone ({args.target_narms} arms):")
    print(target)
    print()

    # Handler configuration
    config = get_genome_handler_config(
        args.handler, args.min_narms, args.max_narms,
        num_segments=args.num_segments,
        initial_hidden_nodes=args.initial_hidden_nodes,
    )

    fitness_fn = _EditDistanceFitness(
        target=target,
        param_limits=config["param_limits"],
        handler_type=config["handler_type"],
        handler_kwargs=config["handler_kwargs"],
    )

    WrappedHandler = create_genome_handler_wrapper(
        config["handler_class"], config["handler_kwargs"]
    )

    # Run NEAT evolution
    wall_start = time.time()

    result_df = evolve_neat(
        fitness_function=fitness_fn,
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
        log_dir=full_log_dir,
        verbose=True,
        num_workers=args.num_workers,
    )

    wall_time = time.time() - wall_start

    # Find best individual across all generations
    best_row = result_df.loc[result_df['fitness'].idxmax()]
    best_genome = best_row['genome']
    best_fitness = best_row['fitness']
    best_generation = best_row['generation']

    # Extract per-generation metrics
    print("\nExtracting per-generation metrics...")
    metrics = extract_generation_metrics(
        result_df, args.generations,
        config["handler_type"], config["handler_kwargs"],
        config["param_limits"],
    )

    # Generate plots
    print("\nGenerating plots...")
    plot_fitness_over_generations(metrics, args.handler, full_log_dir, args.show_plot)
    plot_diversity_and_species(metrics, args.handler, full_log_dir, args.show_plot)
    plot_best_individual(
        best_genome, best_fitness, target, config["handler_type"],
        config["handler_kwargs"], full_log_dir, args.show_plot,
    )

    # Final summary
    last_gen_pop = result_df[
        (result_df['generation'] == args.generations)
        & (result_df['in_pop'] == True)
    ]
    final_fitnesses = last_gen_pop['fitness'].values

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Best fitness:         {best_fitness:.6f} (generation {best_generation})")
    print(f"Final mean fitness:   {np.mean(final_fitnesses):.6f}")
    print(f"Final std fitness:    {np.std(final_fitnesses):.6f}")
    if 'species_id' in last_gen_pop.columns:
        print(f"Final species count:  {last_gen_pop['species_id'].nunique()}")
    print(f"Wall-clock time:      {wall_time:.2f}s")
    print(f"Results saved to:     {full_log_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
