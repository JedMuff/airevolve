"""NSGA-II bi-objective evolution runner.

Evolves drone morphologies for two competing objectives:
  Obj 1 — Task Performance : Maximize waypoints (gates) passed.
  Obj 2 — Power Efficiency : Minimize total energy consumed (Joules)
           over a 12-second evaluation flight, measured directly from
           the LiPoBatteryModel.

The NSGA-II strategy maintains a diverse Pareto front of trade-off solutions
so the designer can later pick the best morphology for their priority.

Quick example:

    python examples/evolution/run_nsga2_evolution.py \\
        --genome spherical --population-size 16 --generations 30 \\
        --training-timesteps 5e5 --num-envs 4 --device cuda:0 \\
        --gate-cfg figure8

The script produces:
  - evolution_data.csv       — all individuals, with rank and crowding_distance
  - pareto_front_last_gen.csv — Pareto-optimal individuals from the final generation
  - pareto_front.png         — scatter plot of waypoints vs energy
"""

import sys
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from airevolve.evolution_tools.evaluators.bi_objective_fitness import BiObjectiveFitness
from airevolve.evolution_tools.strategies.nsga2_strategy import evolve_nsga2
from airevolve.evolution_tools.strategies.init_population import (
    generate_initial_pop_parallel,
)
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


# ── Re-use the shared helpers from run_evolution.py ──────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from run_evolution import (
    get_genome_handler_config,
    create_genome_handler_wrapper,
)


def parse_arguments():
    p = argparse.ArgumentParser(description="NSGA-II bi-objective evolution runner")

    p.add_argument("--genome",
                   choices=["spherical", "cartesian", "cppn", "hybrid-cppn"],
                   default="spherical")
    p.add_argument("--init-pop-mode", choices=["random", "hover_repair"], default="random")
    p.add_argument("--per-individual-repair", action="store_true")
    p.add_argument("--hover-gradient",        action="store_true")

    # EA knobs
    p.add_argument("--population-size",  type=int,   default=16)
    p.add_argument("--generations",      type=int,   default=30)
    p.add_argument("--num-mutate",       type=int,   default=None)
    p.add_argument("--num-crossover",    type=int,   default=0)
    p.add_argument("--num-workers",      type=int,   default=1)

    # Logging
    p.add_argument("--log-dir", default="__data__/nsga2_evolution")

    # Morphology bounds
    p.add_argument("--min-narms", type=int, default=6)
    p.add_argument("--max-narms", type=int, default=6)

    # Gate config
    p.add_argument("--gate-cfg",
                   choices=["backandforth", "figure8", "circle", "slalom"],
                   default="figure8")

    # CPPN knobs
    p.add_argument("--num-segments",         type=int, default=8)
    p.add_argument("--initial-hidden-nodes", type=int, default=0)

    # RL brain
    p.add_argument("--training-timesteps", type=float, default=1e6)
    p.add_argument("--num-envs",           type=int,   default=4)
    p.add_argument("--device",             default="cuda:0")

    args = p.parse_args()

    if args.num_mutate is None:
        args.num_mutate = args.population_size

    return args


def build_fitness(args, config):
    rl_env_dt = 0.01
    max_steps = max(1, int(round(12.0 / rl_env_dt)))  # 12-second window
    return BiObjectiveFitness(
        brain="rl",
        hover_gradient=args.hover_gradient,
        per_individual_repair=args.per_individual_repair,
        is_indirect=args.genome in ("cppn", "hybrid-cppn"),
        handler_class=config["handler_class"],
        handler_kwargs=config["handler_kwargs"],
        coordinate_system=config["coordinate_system"],
        brain_kwargs={
            "gate_cfg":       args.gate_cfg,
            "training_ts":    args.training_timesteps,
            "num_envs":       args.num_envs,
            "device":         args.device,
            "max_steps":      max_steps,
        },
    )


def build_initial_population(args, config, WrappedHandler):
    if args.init_pop_mode == "random":
        if args.genome in ("spherical", "cartesian"):
            return WrappedHandler().random_population(args.population_size), None
        handlers = WrappedHandler().generate_random_population(args.population_size)
        return [h.genome for h in handlers], None
    pop, stats = generate_initial_pop_parallel(
        WrappedHandler(),
        args.population_size,
        coordinate_system=config["coordinate_system"],
        num_workers=None,
        handler_type=args.genome,
        handler_kwargs=config["handler_kwargs"],
        handler_class=config["handler_class"],
    )
    return pop, stats


def plot_pareto(all_individuals, log_dir, last_gen):
    """Scatter plot: waypoints vs energy, coloured by Pareto front."""
    last = all_individuals[all_individuals["generation"] == last_gen].copy()
    last = last.dropna(subset=["rank"])

    waypoints = np.array([f[0] for f in last["fitness"].values])
    energies  = np.array([f[1] for f in last["fitness"].values])
    ranks     = last["rank"].values.astype(int)

    # Plot up to front 5 with distinct colours; rest in grey
    max_coloured = 5
    cmap = plt.cm.get_cmap("tab10", max_coloured)

    fig, ax = plt.subplots(figsize=(9, 6))
    for r in range(max_coloured + 1):
        mask = ranks == r
        if not mask.any():
            continue
        colour = cmap(r) if r < max_coloured else "lightgrey"
        label  = f"Front {r}" if r < max_coloured else f"Front ≥{max_coloured}"
        ax.scatter(energies[mask], waypoints[mask],
                   c=[colour], label=label, edgecolors="k", linewidths=0.5, s=60)

    mask_rest = ranks >= max_coloured
    if mask_rest.any():
        ax.scatter(energies[mask_rest], waypoints[mask_rest],
                   c="lightgrey", label=f"Front ≥{max_coloured}",
                   edgecolors="k", linewidths=0.5, s=60)

    ax.set_xlabel("Total Energy Consumed (J)  [minimize]")
    ax.set_ylabel("Gates Passed  [maximize]")
    ax.set_title(f"NSGA-II Pareto Front — Generation {last_gen}")
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    path = os.path.join(log_dir, "pareto_front.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Pareto plot saved to: {path}")


def main():
    args = parse_arguments()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name  = (
        f"nsga2_{args.genome}_{args.gate_cfg}_{args.init_pop_mode}_{timestamp}"
    )
    full_log_dir = os.path.join(args.log_dir, run_name)
    os.makedirs(full_log_dir, exist_ok=True)

    print("=" * 80)
    print("NSGA-II Bi-Objective Evolution Runner")
    print("  Obj 1: Maximize waypoints (gates passed)")
    print("  Obj 2: Minimize total energy consumed (Joules, 12-second eval)")
    print("=" * 80)
    print(f"Log directory   : {full_log_dir}")
    print(f"Genome          : {args.genome}")
    print(f"Population size : {args.population_size}")
    print(f"Generations     : {args.generations}")
    print(f"Mutate/Crossover: {args.num_mutate} / {args.num_crossover}")
    print(f"Gate cfg        : {args.gate_cfg}")
    print(f"Training ts     : {args.training_timesteps}")
    print(f"Num envs/device : {args.num_envs} / {args.device}")
    print("=" * 80)

    config = get_genome_handler_config(
        args.genome, args.min_narms, args.max_narms,
        num_segments=args.num_segments,
        initial_hidden_nodes=args.initial_hidden_nodes,
    )

    WrappedHandler   = create_genome_handler_wrapper(config["handler_class"], config["handler_kwargs"])
    fitness_function = build_fitness(args, config)

    print("\n--- Phase 1: Initial Population ---")
    initial_pop, init_stats = build_initial_population(args, config, WrappedHandler)
    if init_stats is not None:
        with open(os.path.join(full_log_dir, "init_pop_stats.json"), "w") as f:
            json.dump(init_stats, f, indent=2)
    if initial_pop is None or len(initial_pop) == 0:
        print("[error] Failed to generate initial population.")
        return

    print("\n--- Phase 2: NSGA-II Evolution ---")
    all_individuals = evolve_nsga2(
        fitness_function=fitness_function,
        population_size=args.population_size,
        num_generations=args.generations,
        num_mutate=args.num_mutate,
        num_crossover=args.num_crossover,
        mutate_after_crossover=True,
        initial_population=initial_pop,
        log_dir=full_log_dir,
        genome_handler=WrappedHandler,
        verbose=True,
        num_workers=args.num_workers,
    )

    # ── Save results ──────────────────────────────────────────────────────────
    csv_path = os.path.join(full_log_dir, "evolution_data.csv")
    df = all_individuals.copy()
    # Expand fitness tuple into two columns for readability
    df["waypoints"]     = df["fitness"].apply(lambda f: f[0])
    df["total_energy_j"] = df["fitness"].apply(lambda f: f[1])
    df["id"] = df["id"].astype(str)
    df.to_csv(csv_path, index=False)
    print(f"\nEvolution data saved to: {csv_path}")

    # Pareto front from last generation
    last_gen    = args.generations
    last_df     = all_individuals[all_individuals["generation"] == last_gen]
    pareto_df   = last_df[last_df["rank"] == 0] if "rank" in last_df.columns else last_df
    pareto_path = os.path.join(full_log_dir, "pareto_front_last_gen.csv")
    pareto_out  = pareto_df.copy()
    pareto_out["waypoints"]      = pareto_out["fitness"].apply(lambda f: f[0])
    pareto_out["total_energy_j"] = pareto_out["fitness"].apply(lambda f: f[1])
    pareto_out.to_csv(pareto_path, index=False)
    print(f"Pareto front saved to:  {pareto_path}")
    print(f"Pareto front size:      {len(pareto_df)} individuals")

    # ── Visualisation ─────────────────────────────────────────────────────────
    plot_pareto(all_individuals, full_log_dir, last_gen)

    print("\nDone.")


if __name__ == "__main__":
    main()
