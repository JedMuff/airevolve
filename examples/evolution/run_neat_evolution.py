"""NEAT evolution runner (speciation-based).

Variant of ``run_evolution.py`` that uses ``evolve_neat`` instead of the
(μ+λ) strategy. Same brain/genome/fitness axes; adds NEAT-specific knobs
(compatibility threshold, species elitism, stagnation limit, etc.) and
drops mu+lambda-specific knobs (``--num-mutate`` / ``--num-crossover`` /
``--strategy-type``).

Quick examples:

    # Lee × cppn × gate  (NEAT is most natural with indirect encodings)
    python examples/evolution/run_neat_evolution.py \\
        --brain lee --genome cppn --fitness gate \\
        --init-pop-mode hover_repair --per-individual-repair \\
        --population-size 16 --generations 50

    # Lee × spherical × gate
    python examples/evolution/run_neat_evolution.py \\
        --brain lee --genome spherical --fitness gate \\
        --population-size 16 --generations 50

    # Pure-hover fitness — no brain, no gate cfg consulted
    python examples/evolution/run_neat_evolution.py \\
        --genome cppn --fitness pure_hover \\
        --population-size 16 --generations 30
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

from airevolve.evolution_tools.strategies.neat import evolve_neat
from airevolve.evolution_tools.selectors.tournament import tournament_selection
from airevolve.evolution_tools.inspection_tools.utils import (
    evolution_dataframe_to_fitness_array,
)
from airevolve.evolution_tools.inspection_tools.plot_fitness import plot_fitness

# Reuse helpers from the (μ+λ) example — same genome/brain/fitness wiring.
from run_evolution import (
    get_genome_handler_config,
    create_genome_handler_wrapper,
    build_initial_population,
    build_fitness,
)


def parse_arguments():
    p = argparse.ArgumentParser(description="NEAT evolution runner (speciation)")

    # Core axes (same as run_evolution.py)
    p.add_argument("--brain", choices=["rl", "lee"], default=None,
                   help="Gate evaluator. Required when --fitness gate; "
                        "ignored (with warning) otherwise.")
    p.add_argument("--genome", choices=["spherical", "cartesian", "cppn", "hybrid-cppn"],
                   default="cppn")
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
    p.add_argument("--crossover-rate", type=float, default=0.75,
                   help="Fraction of offspring produced by crossover (vs mutation-only).")
    p.add_argument("--num-workers", type=int, default=None,
                   help="Workers for evaluate_population. "
                        "Default: 1 if --brain rl, else 32.")

    # NEAT-specific knobs
    p.add_argument("--compatibility-threshold", type=float, default=3.0,
                   help="Initial compatibility distance threshold for speciation.")
    p.add_argument("--species-elitism", type=int, default=1,
                   help="Top-N individuals per species copied unchanged each generation.")
    p.add_argument("--stagnation-limit", type=int, default=15,
                   help="Generations without improvement before species removal.")
    p.add_argument("--min-species-size", type=int, default=2,
                   help="Minimum offspring allocated per surviving species.")
    p.add_argument("--target-species-count", type=int, default=5,
                   help="Target number of species (drives dynamic threshold).")
    p.add_argument("--no-adjust-threshold", action="store_true",
                   help="Disable dynamic adjustment of compatibility threshold.")
    p.add_argument("--interspecies-mating-rate", type=float, default=0.001,
                   help="Probability of picking a crossover partner from another species.")
    p.add_argument("--no-mutate-after-crossover", action="store_true",
                   help="Skip mutation step on offspring produced by crossover.")

    # Logging / plotting
    p.add_argument("--log-dir", default="__data__/evolution_neat")
    p.add_argument("--show-plot", action="store_true")

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

    if args.num_workers is None:
        args.num_workers = 1 if args.brain == "rl" else 32

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


def build_experiment_dir(args, log_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    brain = args.brain or "none"
    suffix = ""
    if args.per_individual_repair:
        suffix += "_repair"
    if args.hover_gradient:
        suffix += "_hg"
    gate_part = f"_{args.gate_cfg}" if args.fitness == "gate" else ""
    name = (f"neat_{args.fitness}_{brain}_{args.genome}_{args.init_pop_mode}"
            f"{suffix}{gate_part}_{timestamp}")
    return os.path.join(log_dir, name)


def plot_species_count(all_individuals, num_generations, save_path):
    """Plot species count and per-species size over generations."""
    fig, (ax_count, ax_sizes) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    df = all_individuals[all_individuals["in_pop"] == True].copy()  # noqa: E712
    if "species_id" not in df.columns:
        plt.close(fig)
        return

    gens = sorted(df["generation"].unique())
    counts = [df[df["generation"] == g]["species_id"].nunique() for g in gens]
    ax_count.plot(gens, counts, marker="o", lw=1.5)
    ax_count.set_ylabel("# species")
    ax_count.set_title("NEAT speciation over generations")
    ax_count.grid(alpha=0.3)

    # Per-species size as stacked area
    pivot = (
        df.groupby(["generation", "species_id"]).size().unstack(fill_value=0).sort_index()
    )
    ax_sizes.stackplot(pivot.index, pivot.T.values, labels=[f"S{c}" for c in pivot.columns])
    ax_sizes.set_xlabel("Generation")
    ax_sizes.set_ylabel("Members")
    ax_sizes.grid(alpha=0.3)
    if pivot.shape[1] <= 12:
        ax_sizes.legend(loc="upper right", fontsize=8, ncol=2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_arguments()

    full_log_dir = build_experiment_dir(args, args.log_dir)
    os.makedirs(full_log_dir, exist_ok=True)

    print("=" * 80)
    print("NEAT Evolution Runner (speciation)")
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
    print(f"Crossover rate:          {args.crossover_rate}")
    print(f"Num workers (eval):      {args.num_workers}")
    print(f"Compatibility threshold: {args.compatibility_threshold} "
          f"(adjust={'off' if args.no_adjust_threshold else 'on'}, "
          f"target={args.target_species_count})")
    print(f"Species elitism:         {args.species_elitism}")
    print(f"Stagnation limit:        {args.stagnation_limit}")
    print(f"Min species size:        {args.min_species_size}")
    print(f"Interspecies mating:     {args.interspecies_mating_rate}")
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

    print("\n--- Phase 2: NEAT Evolution ---")
    all_individuals = evolve_neat(
        fitness_function=fitness_function,
        population_size=args.population_size,
        num_generations=args.generations,
        crossover_rate=args.crossover_rate,
        parent_selection=tournament_selection,
        genome_handler=WrappedHandler,
        compatibility_threshold=args.compatibility_threshold,
        species_elitism=args.species_elitism,
        stagnation_limit=args.stagnation_limit,
        min_species_size=args.min_species_size,
        adjust_threshold=not args.no_adjust_threshold,
        target_species_count=args.target_species_count,
        interspecies_mating_rate=args.interspecies_mating_rate,
        mutate_after_crossover=not args.no_mutate_after_crossover,
        initial_population=initial_population,
        log_dir=full_log_dir,
        num_workers=args.num_workers,
    )

    evolution_csv_path = os.path.join(full_log_dir, "evolution_data.csv")
    df = all_individuals.copy()
    df["id"] = df["id"].astype(str)
    df.to_csv(evolution_csv_path, index=False)
    print(f"Evolution data saved to: {evolution_csv_path}")

    # evolve_neat treats generation 0 as the initial pop and runs gens 1..num_generations.
    last_gen = args.generations
    last_gen_df = all_individuals.loc[all_individuals["generation"] == last_gen]
    if last_gen_df.empty:
        last_gen = int(all_individuals["generation"].max())
        last_gen_df = all_individuals.loc[all_individuals["generation"] == last_gen]
    best = last_gen_df.sort_values(by="fitness", ascending=False).iloc[0]
    print(f"\nBest individual gen {last_gen}: id={best['id']}, "
          f"fitness={best['fitness']}, species={best.get('species_id', 'n/a')}")

    # Fitness plot — evolve_neat keeps a variable-size in_pop, so pad with NaN
    # to fit the (gen, pop_size) shape expected by plot_fitness.
    fitness_array = evolution_dataframe_to_fitness_array(
        all_individuals, population_size=args.population_size,
    )
    _fig, ax = plt.subplots(figsize=(12, 8))
    plot_fitness(ax, fitness_array)
    ax.set_title(
        f"NEAT: {args.fitness} / {args.brain or 'no-brain'} / {args.genome}"
    )
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    plt.tight_layout()
    plot_path = os.path.join(full_log_dir, f"fitness_evolution_{args.genome}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Fitness plot saved to:   {plot_path}")
    if args.show_plot:
        plt.show()
    else:
        plt.close()

    species_plot_path = os.path.join(full_log_dir, f"species_{args.genome}.png")
    plot_species_count(all_individuals, args.generations, species_plot_path)
    print(f"Species plot saved to:   {species_plot_path}")

    print(f"\nNEAT evolution completed. Best fitness: {best['fitness']}")


if __name__ == "__main__":
    main()
