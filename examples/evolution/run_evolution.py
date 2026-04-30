"""Unified evolution runner.

Replaces three near-duplicate example scripts. Picks brain, genome encoding,
fitness mode, initial-population strategy, and repair behavior from CLI flags.

See EVOLUTION_UNIFICATION_PLAN.md for the full spec.

Quick examples:

    # Lee × spherical × gate
    python examples/evolution/run_evolution.py \\
        --brain lee --genome spherical --fitness gate \\
        --init-pop-mode random \\
        --population-size 16 --generations 50

    # Lee × cppn × gate with repair-every-individual + parallel hover-repair init
    python examples/evolution/run_evolution.py \\
        --brain lee --genome cppn --fitness gate \\
        --init-pop-mode hover_repair --per-individual-repair \\
        --population-size 16 --generations 50

    # Hover-gradient mode (non-hoverable drones get [0, 3] signal)
    python examples/evolution/run_evolution.py \\
        --brain lee --genome spherical --fitness gate --hover-gradient \\
        --init-pop-mode random

    # Pure-hover fitness — no brain, no gate cfg consulted
    python examples/evolution/run_evolution.py \\
        --genome spherical --fitness pure_hover \\
        --init-pop-mode random

    # RL × cppn
    python examples/evolution/run_evolution.py \\
        --brain rl --genome cppn --fitness gate \\
        --training-timesteps 1e6 --num-envs 2 --device cuda:0
"""

import sys
import os

# BLAS thread caps must run before numpy import.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from airevolve.evolution_tools.evaluators.unified_fitness import UnifiedFitness
from airevolve.evolution_tools.strategies.mu_lambda import evolve
from airevolve.evolution_tools.strategies.init_population import (
    generate_initial_pop_parallel,
)
from airevolve.evolution_tools.selectors.tournament import tournament_selection
from airevolve.evolution_tools.genome_handlers.spherical_angular_genome_handler import (
    SphericalAngularDroneGenomeHandler,
)
from airevolve.evolution_tools.genome_handlers.cartesian_euler_genome_handler import (
    CartesianEulerDroneGenomeHandler,
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


def parse_arguments():
    p = argparse.ArgumentParser(description="Unified evolution runner")

    # Core axes
    p.add_argument("--brain", choices=["rl", "lee"], default=None,
                   help="Gate evaluator. Required when --fitness gate; "
                        "ignored (with warning) otherwise.")
    p.add_argument("--genome", choices=["spherical", "cartesian", "cppn", "hybrid-cppn"],
                   default="spherical")
    p.add_argument("--fitness",
                   choices=["gate", "pure_hover", "edit_distance", "zero"],
                   default="gate")
    p.add_argument("--init-pop-mode", choices=["random", "hover_repair"],
                   default="random")
    p.add_argument("--per-individual-repair", action="store_true",
                   help="Apply 3-stage repair to every individual before fitness eval.")
    p.add_argument("--hover-gradient", action="store_true",
                   help="With --fitness gate: non-hoverable drones receive "
                        "continuous_hover_fitness ∈ [0, 3] and skip the brain.")

    # EA knobs
    p.add_argument("--population-size", type=int, default=16)
    p.add_argument("--generations", type=int, default=50)
    p.add_argument("--num-mutate", type=int, default=None,
                   help="Default = population_size.")
    p.add_argument("--num-crossover", type=int, default=0)
    p.add_argument("--strategy-type", choices=["plus", "comma"], default="plus")
    p.add_argument("--num-workers", type=int, default=None,
                   help="Workers for evaluate_population. "
                        "Default: 1 if --brain rl, else 32.")

    # Logging / plotting
    p.add_argument("--log-dir", default="__data__/evolution")
    p.add_argument("--show-plot", action="store_true")
    p.add_argument("--save-all-plots", action="store_true")

    # Morphology bounds
    p.add_argument("--min-narms", type=int, default=6)
    p.add_argument("--max-narms", type=int, default=6)

    # Gate cfg (only consulted with --fitness gate)
    p.add_argument("--gate-cfg",
                   choices=["backandforth", "figure8", "circle", "slalom"],
                   default="figure8")

    # CPPN knobs
    p.add_argument("--num-segments", type=int, default=8)
    p.add_argument("--initial-hidden-nodes", type=int, default=0)

    # Lee brain knobs
    p.add_argument("--max-evals", type=int, default=500)
    p.add_argument("--cma-workers", type=int, default=1)
    p.add_argument("--sim-time", type=float, default=20.0)
    p.add_argument("--dt", type=float, default=0.005)
    p.add_argument("--timeout", type=float, default=30.0)

    # RL brain knobs
    p.add_argument("--training-timesteps", type=float, default=1e6)
    p.add_argument("--num-envs", type=int, default=100)
    p.add_argument("--device", default="cuda:0")

    args = p.parse_args()

    # Defaults that depend on other args
    if args.num_mutate is None:
        args.num_mutate = args.population_size
    if args.num_workers is None:
        args.num_workers = 1 if args.brain == "rl" else 32

    # Validation / warnings per the brain × fitness compatibility matrix.
    if args.fitness == "gate":
        if args.brain is None:
            p.error("--brain is required when --fitness gate")
    else:
        if args.brain is not None:
            print(f"[warn] --brain={args.brain!r} ignored: --fitness={args.fitness}")
            args.brain = None
        if args.hover_gradient:
            print(f"[warn] --hover-gradient ignored: --fitness={args.fitness}")
            args.hover_gradient = False

    return args


def get_genome_handler_config(handler_type, min_narms, max_narms,
                              num_segments, initial_hidden_nodes,
                              init_topology="empty"):
    """Genome handler class + kwargs for the four supported encodings.

    Symmetry is unsupported. Built-in repair is always disabled — repair is
    owned by UnifiedFitness via --per-individual-repair.
    """
    # [magnitude, arm_yaw, arm_pitch, motor_pitch, motor_yaw, direction]
    shared_params = np.array([
        [0.055, 0.17],
        [-np.pi, np.pi],
        [-np.pi / 2, np.pi / 2],
        [-np.pi, np.pi],
        [-np.pi, np.pi],
        [0, 1],
    ])

    append_arm_chance = 0.0 if min_narms == max_narms else 0.5

    if handler_type == "spherical":
        return {
            "handler_class": SphericalAngularDroneGenomeHandler,
            "handler_kwargs": {
                "min_max_narms": (min_narms, max_narms),
                "append_arm_chance": append_arm_chance,
                "parameter_limits": shared_params,
                "bilateral_plane_for_symmetry": None,
                "repair": False,
            },
            "param_limits": shared_params,
            "coordinate_system": "spherical",
        }
    if handler_type == "cartesian":
        return {
            "handler_class": CartesianEulerDroneGenomeHandler,
            "handler_kwargs": {
                "min_max_narms": (min_narms, max_narms),
                "append_arm_chance": append_arm_chance,
                "bilateral_plane_for_symmetry": None,
                "repair": False,
            },
            "param_limits": shared_params,
            "coordinate_system": "cartesian",
        }
    if handler_type == "cppn":
        return {
            "handler_class": CPPNNeatDroneGenomeHandler,
            "handler_kwargs": {
                "num_segments": num_segments,
                "min_max_narms": (min_narms, max_narms),
                "initial_hidden_nodes": initial_hidden_nodes,
                "init_topology": init_topology,
                "parameter_limits": shared_params,
                "repair": False,
            },
            "param_limits": shared_params,
            "coordinate_system": "cppn",
        }
    if handler_type == "hybrid-cppn":
        return {
            "handler_class": HybridCPPNDroneGenomeHandler,
            "handler_kwargs": {
                "min_max_narms": (min_narms, max_narms),
                "initial_hidden_nodes": initial_hidden_nodes,
                "init_topology": init_topology,
                "parameter_limits": shared_params,
                "repair": False,
            },
            "param_limits": shared_params,
            "coordinate_system": "hybrid-cppn",
        }
    raise ValueError(f"Unknown genome handler type: {handler_type}")


def create_genome_handler_wrapper(handler_class, handler_kwargs):
    """Wrap a handler class so the EA's no-arg constructor produces
    a configured handler. Used as the `genome_handler` arg to evolve()."""
    class GenomeHandlerWrapper(handler_class):
        def __init__(self, *_args, genome=None, **_kwargs):
            if genome is None:
                super().__init__(**handler_kwargs)
            else:
                super().__init__(genome=genome, **handler_kwargs)
    return GenomeHandlerWrapper


def build_experiment_dir(args, log_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    brain = args.brain or "none"
    suffix = ""
    if args.per_individual_repair:
        suffix += "_repair"
    if args.hover_gradient:
        suffix += "_hg"
    gate_part = f"_{args.gate_cfg}" if args.fitness == "gate" else ""
    name = (f"{args.fitness}_{brain}_{args.genome}_{args.init_pop_mode}"
            f"{suffix}{gate_part}_{timestamp}")
    return os.path.join(log_dir, name)


def build_initial_population(args, config, WrappedHandler):
    if args.init_pop_mode == "random":
        if args.genome in ("spherical", "cartesian"):
            return WrappedHandler().random_population(args.population_size), None
        # CPPN / hybrid-cppn: random_population is unavailable; use the
        # GenomeHandler interface and extract the underlying genomes.
        handlers = WrappedHandler().generate_random_population(args.population_size)
        return [h.genome for h in handlers], None
    pop, stats = generate_initial_pop_parallel(
        WrappedHandler(),
        args.population_size,
        coordinate_system=config["coordinate_system"],
        num_workers=None,  # cpu_count default
        handler_type=args.genome,
        handler_kwargs=config["handler_kwargs"],
        handler_class=config["handler_class"],
    )
    return pop, stats


def build_fitness(args, config):
    is_indirect = args.genome in ("cppn", "hybrid-cppn")
    brain_kwargs = {}
    if args.brain == "lee":
        brain_kwargs = {
            "gate_cfg": args.gate_cfg,
            "max_evals": args.max_evals,
            "cma_workers": args.cma_workers,
            "sim_time": args.sim_time,
            "dt": args.dt,
            "timeout": args.timeout,
        }
    elif args.brain == "rl":
        # DroneGateEnv runs at dt=0.01 by default → episode length is
        # `sim_time / dt` env steps. Using --sim-time keeps the gate-task
        # episode duration consistent between Lee (--sim-time) and RL.
        rl_env_dt = 0.01
        max_steps = max(1, int(round(args.sim_time / rl_env_dt)))
        brain_kwargs = {
            "gate_cfg": args.gate_cfg,
            "training_ts": args.training_timesteps,
            "num_envs": args.num_envs,
            "device": args.device,
            "max_steps": max_steps,
        }
    return UnifiedFitness(
        brain=args.brain,
        fitness_mode=args.fitness,
        hover_gradient=args.hover_gradient,
        per_individual_repair=args.per_individual_repair,
        is_indirect=is_indirect,
        handler_class=config["handler_class"] if is_indirect else None,
        handler_kwargs=config["handler_kwargs"] if is_indirect else None,
        coordinate_system=config["coordinate_system"],
        brain_kwargs=brain_kwargs,
    )


def main():
    args = parse_arguments()

    full_log_dir = build_experiment_dir(args, args.log_dir)
    os.makedirs(full_log_dir, exist_ok=True)

    print("=" * 80)
    print("Unified Evolution Runner")
    print("=" * 80)
    sys.stdout.flush()
    print(f"Log directory:           {full_log_dir}")
    print(f"Brain:                   {args.brain}")
    print(f"Genome:                  {args.genome}")
    print(f"Fitness:                 {args.fitness}")
    print(f"Init-pop mode:           {args.init_pop_mode}")
    print(f"Per-individual repair:   {args.per_individual_repair}")
    print(f"Hover gradient:          {args.hover_gradient}")
    print(f"Population size:         {args.population_size}")
    print(f"Generations:             {args.generations}")
    print(f"Num mutate / crossover:  {args.num_mutate} / {args.num_crossover}")
    print(f"Strategy:                {args.strategy_type}")
    print(f"Num workers (eval):      {args.num_workers}")
    if args.fitness == "gate":
        print(f"Gate cfg:                {args.gate_cfg}")
    if args.brain == "lee":
        print(f"Lee max-evals:           {args.max_evals}")
        print(f"Lee cma-workers:         {args.cma_workers}")
        print(f"Sim time / dt / timeout: {args.sim_time} / {args.dt} / {args.timeout}")
    if args.brain == "rl":
        print(f"Training timesteps:      {args.training_timesteps}")
        print(f"Num envs / device:       {args.num_envs} / {args.device}")
        print(f"Sim time:                {args.sim_time}s "
              f"(env max_steps = {int(round(args.sim_time / 0.01))} @ dt=0.01)")
    if args.genome in ("cppn", "hybrid-cppn"):
        print(f"CPPN segments:           {args.num_segments}")
        print(f"CPPN initial hidden:     {args.initial_hidden_nodes}")
    print("=" * 80)

    config = get_genome_handler_config(
        args.genome, args.min_narms, args.max_narms,
        num_segments=args.num_segments,
        initial_hidden_nodes=args.initial_hidden_nodes,
    )

    fitness_function = build_fitness(args, config)
    WrappedHandler = create_genome_handler_wrapper(
        config["handler_class"], config["handler_kwargs"]
    )

    print("\n--- Phase 1: Initial Population ---")
    initial_population, init_stats = build_initial_population(args, config, WrappedHandler)
    if init_stats is not None:
        with open(os.path.join(full_log_dir, "init_pop_stats.json"), "w") as f:
            json.dump(init_stats, f, indent=2)
    if initial_population is None or len(initial_population) == 0:
        print("\n[error] Failed to generate initial population. Exiting.")
        return

    print("\n--- Phase 2: Evolution ---")
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

    evolution_csv_path = os.path.join(full_log_dir, "evolution_data.csv")
    df = all_individuals.copy()
    df["id"] = df["id"].astype(str)
    df.to_csv(evolution_csv_path, index=False)
    print(f"Evolution data saved to: {evolution_csv_path}")

    last_gen = args.generations - 1
    best = all_individuals.loc[all_individuals["generation"] == last_gen].sort_values(
        by="fitness", ascending=False,
    ).iloc[0]
    print(f"\nBest individual gen {last_gen}: id={best['id']}, fitness={best['fitness']}")

    # Fitness plot
    fitness_array = evolution_dataframe_to_fitness_array(
        all_individuals, population_size=args.population_size,
    )
    _fig, ax = plt.subplots(figsize=(12, 8))
    plot_fitness(ax, fitness_array)
    ax.set_title(f"Evolution: {args.fitness} / {args.brain or 'no-brain'} / {args.genome}")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    plt.tight_layout()
    plot_path = os.path.join(full_log_dir, f"fitness_evolution_{args.genome}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Fitness plot saved to: {plot_path}")
    if args.show_plot:
        plt.show()
    else:
        plt.close()

    if args.save_all_plots:
        try:
            from airevolve.evolution_tools.inspection_tools.plot_diversity import plot_diversity
            from airevolve.evolution_tools.inspection_tools.evolution_plotters import (
                create_evolution_summary_plot,
            )
        except ImportError as e:
            print(f"[warn] --save-all-plots unavailable: {e}")
            return

        # Build a (gen, pop_size, n_arms, 6) tensor for diversity plotting.
        # Indirect encodings store CPPN/HybridGenome objects in df['genome'];
        # diversity plots require numeric arrays, so skip those.
        if args.genome in ("cppn", "hybrid-cppn"):
            print("[info] --save-all-plots skipped for indirect encodings")
            return

        pop_data = []
        fit_data = []
        for gen in range(args.generations):
            sub = all_individuals[
                (all_individuals["generation"] == gen)
                & (all_individuals["in_pop"] == True)  # noqa: E712
            ].sort_values(by="fitness", ascending=False)
            genomes = list(sub["genome"])
            fits = list(sub["fitness"])
            while len(genomes) < args.population_size:
                template = genomes[0] if genomes else np.full((1, 6), np.nan)
                genomes.append(np.full_like(template, np.nan))
                fits.append(np.nan)
            pop_data.append(genomes[:args.population_size])
            fit_data.append(np.array(fits[:args.population_size]))

        fig_div, ax_div = plt.subplots(figsize=(10, 6))
        plot_diversity(
            ax_div, pop_data,
            parameter_limits=(config["param_limits"][:, 0], config["param_limits"][:, 1]),
        )
        ax_div.set_title(f"Diversity ({args.genome}, {args.fitness})")
        plt.tight_layout()
        div_path = os.path.join(full_log_dir, f"diversity_{args.genome}.png")
        plt.savefig(div_path, dpi=300, bbox_inches="tight")
        print(f"Diversity plot saved to: {div_path}")
        plt.close()

        fig_summary = create_evolution_summary_plot(
            np.array(pop_data), np.array(fit_data),
            title=f"Evolution Summary ({args.genome}, {args.fitness})",
            parameter_limits=(config["param_limits"][:, 0], config["param_limits"][:, 1]),
        )
        sum_path = os.path.join(full_log_dir, f"evolution_summary_{args.genome}.png")
        fig_summary.savefig(sum_path, dpi=300, bbox_inches="tight")
        print(f"Evolution summary saved to: {sum_path}")
        plt.close(fig_summary)

    print(f"\nEvolution completed. Best fitness: {best['fitness']}")


if __name__ == "__main__":
    main()
