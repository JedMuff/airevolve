"""
Edit Distance Evolution Comparison

Compares three genome representations -- spherical, cppn, and hybrid-cppn --
on an edit distance target matching task.  Edit distance is a fast, purely
geometric fitness function: no simulation, no repair pipeline, no controller
tuning.

Usage:
    # Quick smoke test
    python examples/evolution/run_evolution_edit_distance_comparison.py \
        --handlers spherical --generations 5 --population-size 5

    # Full comparison (all three representations)
    python examples/evolution/run_evolution_edit_distance_comparison.py \
        --generations 50 --population-size 20

    # CPPN only with more segments
    python examples/evolution/run_evolution_edit_distance_comparison.py \
        --handlers cppn --num-segments 12 --generations 100
"""

import sys
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from airevolve.evolution_tools.evaluators.edit_distance import compute_edit_distance
from airevolve.evolution_tools.strategies.mu_lambda import evolve
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
from airevolve.evolution_tools.inspection_tools.utils import (
    evolution_dataframe_to_fitness_array,
)
from airevolve.evolution_tools.inspection_tools.plot_fitness import plot_fitness
from airevolve.evolution_tools.inspection_tools.drone_visualizer import DroneVisualizer


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Compare genome representations on edit-distance target matching"
    )

    parser.add_argument(
        "--handlers", nargs="+",
        choices=["spherical", "cppn", "hybrid-cppn"],
        default=["spherical", "cppn", "hybrid-cppn"],
        help="Genome handlers to compare (default: all three)",
    )
    parser.add_argument("--population-size", type=int, default=20)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--num-mutate", type=int, default=20)
    parser.add_argument("--num-crossover", type=int, default=0,
                        help="Crossover count (default 0; CPPN/hybrid don't support crossover)")
    parser.add_argument("--target-narms", type=int, default=6)
    parser.add_argument("--min-narms", type=int, default=6)
    parser.add_argument("--max-narms", type=int, default=6)
    parser.add_argument("--num-segments", type=int, default=8,
                        help="CPPN evaluation segments (CPPN/hybrid only)")
    parser.add_argument("--initial-hidden-nodes", type=int, default=0,
                        help="Initial hidden nodes in CPPN topology")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--show-plot", action="store_true")
    parser.add_argument("--log-dir", default="./.data")
    parser.add_argument("--num-workers", type=int, default=1)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Target drone creation
# ---------------------------------------------------------------------------

def create_target_drone(narms: int) -> np.ndarray:
    """Return a standard drone in spherical (narms, 6) format.

    Columns: [magnitude, azimuth, pitch, motor_yaw, motor_pitch, direction]
    """
    radius = 0.08
    if narms == 4:
        azimuths = np.array([np.pi / 4, 3 * np.pi / 4,
                             -3 * np.pi / 4, -np.pi / 4])
    else:
        azimuths = np.linspace(0, 2 * np.pi, narms, endpoint=False)
        # Wrap to [-pi, pi] to stay within parameter limits
        azimuths = (azimuths + np.pi) % (2 * np.pi) - np.pi

    target = np.zeros((narms, 6))
    for i in range(narms):
        target[i, 0] = radius          # magnitude
        target[i, 1] = azimuths[i]     # azimuth
        target[i, 2] = 0.0             # pitch (flat)
        target[i, 3] = 0.0             # motor_yaw
        target[i, 4] = 0.0             # motor_pitch
        target[i, 5] = i % 2           # alternating spin direction
    return target


# ---------------------------------------------------------------------------
# Spherical parameter limits (shared across all handlers for fair comparison)
# ---------------------------------------------------------------------------

SPHERICAL_PARAMS = np.array([
    [0.055, 0.17],           # magnitude
    [-np.pi, np.pi],         # arm yaw (azimuth)
    [-np.pi / 2, np.pi / 2], # arm pitch (elevation)
    [-np.pi, np.pi],         # motor pitch
    [-np.pi, np.pi],         # motor yaw
    [0, 1],                  # direction
])


# ---------------------------------------------------------------------------
# Fitness function (picklable class for multiprocessing)
# ---------------------------------------------------------------------------

class _EditDistanceFitness:
    """Picklable fitness callable for use with ``evolve()``."""

    def __init__(self, target, param_limits, handler_type, handler_kwargs):
        self.target = target
        self.min_vals = param_limits[:, 0]
        self.max_vals = param_limits[:, 1]
        self.handler_type = handler_type
        self.handler_kwargs = handler_kwargs

    def __call__(self, genome, ind_save_dir):
        if self.handler_type == "spherical":
            phenotype = genome
        elif self.handler_type == "cppn":
            handler = CPPNNeatDroneGenomeHandler(
                genome=genome, **self.handler_kwargs
            )
            phenotype = handler.get_phenotype()
        elif self.handler_type == "hybrid-cppn":
            handler = HybridCPPNDroneGenomeHandler(
                genome=genome, **self.handler_kwargs
            )
            phenotype = handler.get_phenotype()
        else:
            raise ValueError(f"Unknown handler type: {self.handler_type}")

        # Negate because evolve() maximises fitness
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
    """Wrap *handler_class* so ``evolve()`` can call ``handler(genome=X)``."""

    class GenomeHandlerWrapper(handler_class):
        def __init__(self, *_args, genome=None, **_kwargs):
            super().__init__(genome=genome, **handler_kwargs)

    return GenomeHandlerWrapper


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

HANDLER_COLORS = {
    "spherical": "tab:blue",
    "cppn": "tab:orange",
    "hybrid-cppn": "tab:green",
}


def plot_results(results, target, log_dir, show):
    """Generate fitness-curve and best-individual plots."""

    # --- Plot 1: Fitness curves ---
    fig, ax = plt.subplots(figsize=(12, 7))
    for handler_type, (fitness_array, _best_genome) in results.items():
        plot_fitness(
            ax,
            fitness_array.copy(),  # plot_fitness mutates the array
            label=handler_type,
            color=HANDLER_COLORS.get(handler_type),
            pltmax=False,
        )

    # Recompute y-limits across all data
    all_vals = np.concatenate(
        [fa.copy().flatten() for fa, _ in results.values()]
    )
    all_vals = all_vals[np.isfinite(all_vals)]
    if len(all_vals) > 0:
        ymin = np.min(all_vals)
        ymax = np.max(all_vals) * 1.1 if np.max(all_vals) > 0 else np.max(all_vals) * 0.9
        ax.set_ylim(ymin, ymax)

    ax.set_title("Edit Distance Evolution Comparison")
    ax.set_ylabel("Fitness (negative edit distance)")
    plt.tight_layout()

    fitness_path = os.path.join(log_dir, "fitness_comparison.png")
    plt.savefig(fitness_path, dpi=300, bbox_inches="tight")
    print(f"Fitness plot saved to {fitness_path}")

    if show:
        plt.show()
    plt.close(fig)

    # --- Plot 2: Best individuals ---
    n_handlers = len(results)
    n_cols = 1 + n_handlers  # target + one per handler
    fig = plt.figure(figsize=(5 * n_cols, 5))

    visualizer = DroneVisualizer()

    # Target
    ax_target = fig.add_subplot(1, n_cols, 1, projection="3d")
    visualizer.plot_3d(target, ax=ax_target, title="Target")

    for idx, (handler_type, (_fa, best_genome)) in enumerate(results.items()):
        ax_ind = fig.add_subplot(1, n_cols, idx + 2, projection="3d")

        if handler_type == "spherical":
            phenotype = best_genome
        elif handler_type == "cppn":
            # Need to decode; reuse the config to build a handler
            handler = CPPNNeatDroneGenomeHandler(
                genome=best_genome,
                parameter_limits=SPHERICAL_PARAMS,
                repair=False,
            )
            phenotype = handler.get_phenotype()
        elif handler_type == "hybrid-cppn":
            handler = HybridCPPNDroneGenomeHandler(
                genome=best_genome,
                parameter_limits=SPHERICAL_PARAMS,
                repair=False,
            )
            phenotype = handler.get_phenotype()
        else:
            phenotype = best_genome

        visualizer.plot_3d(phenotype, ax=ax_ind, title=f"Best {handler_type}")

    plt.tight_layout()
    individuals_path = os.path.join(log_dir, "best_individuals.png")
    plt.savefig(individuals_path, dpi=300, bbox_inches="tight")
    print(f"Best-individuals plot saved to {individuals_path}")

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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    narms_str = f"{args.target_narms}arms"
    exp_name = f"edit_distance_comparison_{narms_str}_{timestamp}"
    full_log_dir = os.path.join(args.log_dir, exp_name)
    os.makedirs(full_log_dir, exist_ok=True)

    print("=" * 80)
    print("Edit Distance Evolution Comparison")
    print("=" * 80)
    print(f"Handlers:        {', '.join(args.handlers)}")
    print(f"Population size: {args.population_size}")
    print(f"Generations:     {args.generations}")
    print(f"Mutations/gen:   {args.num_mutate}")
    print(f"Crossovers/gen:  {args.num_crossover}")
    print(f"Target arms:     {args.target_narms}")
    print(f"Arm range:       {args.min_narms}-{args.max_narms}")
    print(f"Log directory:   {full_log_dir}")
    print("=" * 80)
    print()

    # Create target drone
    target = create_target_drone(args.target_narms)
    print(f"Target drone ({args.target_narms} arms):")
    print(target)
    print()

    # Run evolution for each handler type
    results = {}  # handler_type -> (fitness_array, best_genome)

    for handler_type in args.handlers:
        print(f"\n{'='*60}")
        print(f"Running evolution with {handler_type.upper()} handler")
        print(f"{'='*60}")

        config = get_genome_handler_config(
            handler_type, args.min_narms, args.max_narms,
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

        handler_log_dir = os.path.join(full_log_dir, handler_type)
        os.makedirs(handler_log_dir, exist_ok=True)

        all_individuals = evolve(
            fitness_function=fitness_fn,
            population_size=args.population_size,
            num_generations=args.generations,
            num_mutate=args.num_mutate,
            num_crossover=args.num_crossover,
            mutate_after_crossover=True,
            strategy_type="plus",
            parent_selection=tournament_selection,
            genome_handler=WrappedHandler,
            log_dir=handler_log_dir,
            initial_population=None,
            num_workers=args.num_workers,
        )

        # Save CSV
        csv_path = os.path.join(handler_log_dir, "evolution_data.csv")
        df_copy = all_individuals.copy()
        df_copy["id"] = df_copy["id"].astype(str)
        df_copy.to_csv(csv_path, index=False)

        # Extract fitness array
        fitness_array = evolution_dataframe_to_fitness_array(
            all_individuals, population_size=args.population_size
        )

        # Get best individual from last generation
        last_gen = args.generations
        best_row = (
            all_individuals
            .loc[all_individuals["generation"] == last_gen]
            .sort_values(by="fitness", ascending=False)
            .iloc[0]
        )
        best_genome = best_row["genome"]
        best_fitness = best_row["fitness"]

        print(f"\n{handler_type} best fitness (gen {last_gen}): {best_fitness:.6f}")

        results[handler_type] = (fitness_array, best_genome)

    # Plot results
    print(f"\n{'='*60}")
    print("Generating plots")
    print(f"{'='*60}")
    plot_results(results, target, full_log_dir, show=args.show_plot)

    print(f"\nAll results saved to {full_log_dir}")


if __name__ == "__main__":
    main()
