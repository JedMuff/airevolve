"""Master runner for the NSGA-II energy-reward ablation study.

Launches a single bi-objective NSGA-II run with a specified RL reward
formulation and saves all results under results/exp_<type>_w_<weight>/.

Three experiment types
-----------------------
  baseline  — Standard PPO, no energy term in the reward.
  dense     — Per-step power penalty: r -= dense_weight × P_instant.
  sparse    — End-of-episode energy penalty: r -= sparse_weight × E_episode_J.

Experiment hyperparameters (paper defaults)
-------------------------------------------
  --population-size   24
  --generations       30
  --training-timesteps 50_000_000
  --num-envs          100

Example invocations
--------------------
  # Experiment 1 — Baseline
  python experimentation/run_energy_ablation.py --experiment-type baseline

  # Experiment 2 — Dense w=1e-5
  python experimentation/run_energy_ablation.py --experiment-type dense --penalty-weight 1e-5

  # Experiment 3 — Sparse w=0.004
  python experimentation/run_energy_ablation.py --experiment-type sparse --penalty-weight 0.004
"""

import sys
import os

# BLAS caps must precede numpy
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Repo root on sys.path ────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "examples" / "evolution"))  # for run_evolution helpers

from airevolve.evolution_tools.evaluators.bi_objective_fitness import BiObjectiveFitness
from airevolve.evolution_tools.strategies.nsga2_strategy import evolve_nsga2
from airevolve.evolution_tools.strategies.init_population import (
    generate_initial_pop_parallel,
    generate_viable_initial_population,
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
from airevolve.evolution_tools.inspection_tools.plot_pareto_front import plot_pareto_front
from run_evolution import get_genome_handler_config, create_genome_handler_wrapper


# ── Experiment type → (experiment_type int, penalty_weights key) ─────────────
_EXP_MAP = {
    "baseline": (0, None),
    "dense":    (1, "dense_weight"),
    "sparse":   (2, "sparse_weight"),
}

# Generations at which to save morphology snapshots
_SNAPSHOT_GENS = {1, 10, 20, 25}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="NSGA-II energy-reward ablation study runner"
    )

    # Experiment identity
    p.add_argument(
        "--experiment-type", choices=["baseline", "dense", "sparse"],
        required=True,
        help="Reward formulation: baseline / dense / sparse",
    )
    p.add_argument(
        "--penalty-weight", type=float, default=0.0,
        help="Penalty coefficient (ignored for baseline).",
    )

    # EA / RL hyperparameters
    p.add_argument("--population-size",    type=int,   default=24)
    p.add_argument("--generations",        type=int,   default=30)
    p.add_argument("--training-timesteps", type=float, default=50_000_000)
    p.add_argument("--num-envs",           type=int,   default=100)
    p.add_argument("--num-workers",        type=int,   default=1)
    p.add_argument("--device",             default="cuda:0")
    p.add_argument("--num-mutate",         type=int,   default=None)
    p.add_argument("--num-crossover",      type=int,   default=0)

    # Morphology
    p.add_argument("--genome",
                   choices=["spherical", "cartesian", "cppn", "hybrid-cppn"],
                   default="spherical")
    p.add_argument("--min-narms", type=int, default=6)
    p.add_argument("--max-narms", type=int, default=6)
    p.add_argument("--gate-cfg",
                   choices=["backandforth", "figure8", "circle", "slalom"],
                   default="figure8")
    p.add_argument("--num-segments",         type=int, default=8)
    p.add_argument("--initial-hidden-nodes", type=int, default=0)
    p.add_argument("--init-pop-mode",
                   choices=["random", "hover_repair"], default="random")
    p.add_argument("--per-individual-repair", action="store_true")
    p.add_argument("--hover-gradient",        action="store_true")

    # Output
    p.add_argument("--results-dir", default="results",
                   help="Root directory for experiment outputs.")
    p.add_argument("--run-id",      default=None,
                   help="Override the auto-generated run directory name.")

    args = p.parse_args()

    if args.num_mutate is None:
        args.num_mutate = args.population_size

    return args


# ─────────────────────────────────────────────────────────────────────────────
# Directory helpers
# ─────────────────────────────────────────────────────────────────────────────

def _results_dir(args) -> Path:
    w_str = f"{args.penalty_weight:.0e}" if args.penalty_weight != 0 else "0"
    name  = args.run_id or f"exp_{args.experiment_type}_w_{w_str}"
    path  = Path(args.results_dir) / name
    path.mkdir(parents=True, exist_ok=True)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Snapshot callback — saves morphologies at specified generations
# ─────────────────────────────────────────────────────────────────────────────

class SnapshotCallback:
    """Saves genome arrays and 3-D morphology plots at selected generations."""

    def __init__(self, results_dir: Path, snapshot_gens: set):
        self.results_dir   = results_dir
        self.snapshot_gens = snapshot_gens

    def __call__(self, generation: int, population) -> None:
        if generation not in self.snapshot_gens:
            return

        snap_dir = self.results_dir / f"snapshots" / f"gen_{generation:03d}"
        snap_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save raw genomes
        genomes = []
        for _, row in population.iterrows():
            g = row["genome"]
            g = g.arms if hasattr(g, "arms") else g
            genomes.append(np.asarray(g))

        # Stack only if uniform shape
        try:
            np.save(str(snap_dir / "genomes.npy"), np.array(genomes))
        except ValueError:
            # Variable-arm-count genomes: save as object array
            arr = np.empty(len(genomes), dtype=object)
            for i, g in enumerate(genomes):
                arr[i] = g
            np.save(str(snap_dir / "genomes.npy"), arr)

        # 2. Save Pareto-front info as CSV
        pop_df = population.copy()
        pop_df["waypoints"]      = pop_df["fitness"].apply(lambda f: f[0])
        pop_df["total_energy_j"] = pop_df["fitness"].apply(lambda f: f[1])
        pop_df[["id", "rank", "crowding_distance", "waypoints", "total_energy_j"]].to_csv(
            str(snap_dir / "pareto_info.csv"), index=False
        )

        # 3. 3-D morphology visualizations (front-0 only, up to 8 individuals)
        try:
            from airevolve.evolution_tools.inspection_tools.drone_visualizer import DroneVisualizer
            viz = DroneVisualizer()

            front0 = population[population["rank"] == 0].head(8)
            for _, row in front0.iterrows():
                g = row["genome"]
                g = g.arms if hasattr(g, "arms") else g
                genome_arr = np.asarray(g)
                ind_id = str(row["id"])
                fitness = row["fitness"]

                try:
                    fig = plt.figure(figsize=plt.figaspect(0.5))
                    ax  = fig.add_subplot(111, projection="3d")
                    viz.plot_3d(
                        genome_arr, ax=ax,
                        title=f"Gen {generation} | id={ind_id} | "
                              f"gates={fitness[0]} energy={fitness[1]:.0f}J",
                        fitness=fitness[0],
                        generation=generation,
                    )
                    fig.savefig(str(snap_dir / f"morphology_{ind_id}.png"),
                                dpi=150, bbox_inches="tight")
                    plt.close(fig)
                except Exception as e:
                    print(f"[snapshot] Could not render morphology {ind_id}: {e}")
        except Exception as e:
            print(f"[snapshot] Visualization skipped: {e}")

        print(f"[snapshot] Generation {generation} saved → {snap_dir}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Build fitness function
# ─────────────────────────────────────────────────────────────────────────────

def build_fitness(args, config, experiment_type_int, penalty_weights_dict):
    rl_env_dt = 0.01
    max_steps = max(1, int(round(12.0 / rl_env_dt)))  # 1200 steps = 12 s

    return BiObjectiveFitness(
        brain="rl",
        hover_gradient=args.hover_gradient,
        per_individual_repair=args.per_individual_repair,
        is_indirect=args.genome in ("cppn", "hybrid-cppn"),
        handler_class=config["handler_class"],
        handler_kwargs=config["handler_kwargs"],
        coordinate_system=config["coordinate_system"],
        brain_kwargs={
            "gate_cfg":         args.gate_cfg,
            "training_ts":      args.training_timesteps,
            "num_envs":         args.num_envs,
            "device":           args.device,
            "max_steps":        max_steps,
            "experiment_type":  experiment_type_int,
            "penalty_weights":  penalty_weights_dict,
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Initial population
# ─────────────────────────────────────────────────────────────────────────────

def build_initial_population(args, config, WrappedHandler):
    is_indirect = args.genome in ("cppn", "hybrid-cppn")

    if args.init_pop_mode == "random":
        # Rejection-sample until we have exactly population_size hover-viable
        # genomes.  Bare random_population() produces ~0.1–5% viable rate for
        # typical arm-count / bound settings, so this replaces the old one-shot
        # call that silently handed unviable genomes to the evaluator.
        return generate_viable_initial_population(
            WrappedHandler(),
            args.population_size,
            is_indirect=is_indirect,
        )

    # hover_repair mode: full 3-stage repair pipeline (slower but higher quality)
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


# ─────────────────────────────────────────────────────────────────────────────
# Post-run artefacts
# ─────────────────────────────────────────────────────────────────────────────

def save_artefacts(all_individuals, results_dir: Path, args):
    # ── evolution_data.csv ────────────────────────────────────────────────────
    df = all_individuals.copy()
    df["waypoints"]      = df["fitness"].apply(lambda f: f[0])
    df["total_energy_j"] = df["fitness"].apply(lambda f: f[1])
    df["id"] = df["id"].astype(str)
    csv_path = results_dir / "evolution_data.csv"
    df.to_csv(str(csv_path), index=False)
    print(f"Evolution data  → {csv_path}")

    # ── pareto_front_final.csv (last generation, rank 0) ─────────────────────
    last_gen  = df["generation"].max()
    last_pop  = df[df["generation"] == last_gen]
    pareto_df = last_pop[last_pop["rank"] == 0] if "rank" in last_pop.columns else last_pop
    pareto_path = results_dir / "pareto_front_final.csv"
    pareto_df.to_csv(str(pareto_path), index=False)
    print(f"Pareto front    → {pareto_path}  ({len(pareto_df)} individuals)")

    # ── Pareto scatter plot ───────────────────────────────────────────────────
    try:
        fig = plot_pareto_front(
            str(csv_path),
            title=f"{args.experiment_type}  w={args.penalty_weight}  "
                  f"(gen {last_gen})",
        )
        plot_path = results_dir / "pareto_front.png"
        fig.savefig(str(plot_path), dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Pareto plot     → {plot_path}")
    except Exception as e:
        print(f"[warn] Pareto plot failed: {e}")

    # ── gates / energy over generations ──────────────────────────────────────
    try:
        _plot_objectives_over_generations(df, results_dir, args)
    except Exception as e:
        print(f"[warn] Objectives-over-gens plot failed: {e}")


def _plot_objectives_over_generations(df, results_dir: Path, args):
    """Line plots of max gates and min energy per generation."""
    gens   = sorted(df["generation"].unique())
    max_g  = [df[df["generation"] == g]["waypoints"].max()      for g in gens]
    min_e  = [df[df["generation"] == g]["total_energy_j"].apply(
                  lambda x: x if np.isfinite(x) else np.nan).min() for g in gens]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(gens, max_g, marker="o", markersize=3, linewidth=1.5, color="steelblue")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Max Gates Passed")
    ax1.set_title("Task Performance over Generations")
    ax1.grid(True, linestyle="--", alpha=0.4)

    ax2.plot(gens, min_e, marker="o", markersize=3, linewidth=1.5, color="darkorange")
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Min Energy (J)")
    ax2.set_title("Best Power Efficiency over Generations")
    ax2.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle(f"{args.experiment_type}  w={args.penalty_weight}", fontsize=11)
    fig.tight_layout()
    fig.savefig(str(results_dir / "objectives_over_generations.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Resolve experiment parameters
    experiment_type_int, penalty_key = _EXP_MAP[args.experiment_type]
    penalty_weights_dict = (
        {} if penalty_key is None
        else {penalty_key: args.penalty_weight}
    )

    # Output directory
    results_dir = _results_dir(args)

    # Save config for reproducibility
    config_snapshot = vars(args).copy()
    config_snapshot["experiment_type_int"]   = experiment_type_int
    config_snapshot["penalty_weights_dict"]  = penalty_weights_dict
    with open(str(results_dir / "config.json"), "w") as f:
        json.dump(config_snapshot, f, indent=2)

    print("=" * 80)
    print("NSGA-II Energy Ablation Experiment")
    print(f"  Type          : {args.experiment_type}")
    print(f"  Penalty weight: {args.penalty_weight}")
    print(f"  Results dir   : {results_dir}")
    print(f"  Pop / Gens    : {args.population_size} / {args.generations}")
    print(f"  Training ts   : {args.training_timesteps:.0f}")
    print(f"  Num envs      : {args.num_envs}  device: {args.device}")
    print(f"  Gate cfg      : {args.gate_cfg}")
    print("=" * 80, flush=True)

    # Genome handler
    config = get_genome_handler_config(
        args.genome, args.min_narms, args.max_narms,
        num_segments=args.num_segments,
        initial_hidden_nodes=args.initial_hidden_nodes,
    )
    WrappedHandler   = create_genome_handler_wrapper(config["handler_class"], config["handler_kwargs"])
    fitness_function = build_fitness(args, config, experiment_type_int, penalty_weights_dict)

    # Initial population
    print("\n--- Phase 1: Initial Population ---", flush=True)
    initial_pop, init_stats = build_initial_population(args, config, WrappedHandler)
    if initial_pop is None or len(initial_pop) == 0:
        print("[error] Failed to generate initial population.")
        return
    if init_stats is not None:
        with open(str(results_dir / "init_pop_stats.json"), "w") as f:
            json.dump(init_stats, f, indent=2)

    # Snapshot callback
    snapshot_cb = SnapshotCallback(results_dir, _SNAPSHOT_GENS)

    # Evolution
    print("\n--- Phase 2: NSGA-II Evolution ---", flush=True)
    all_individuals = evolve_nsga2(
        fitness_function=fitness_function,
        population_size=args.population_size,
        num_generations=args.generations,
        num_mutate=args.num_mutate,
        num_crossover=args.num_crossover,
        mutate_after_crossover=True,
        initial_population=initial_pop,
        log_dir=str(results_dir / "rl_logs"),
        genome_handler=WrappedHandler,
        verbose=True,
        num_workers=args.num_workers,
        snapshot_callback=snapshot_cb,
    )

    # Save all artefacts
    print("\n--- Phase 3: Saving Artefacts ---", flush=True)
    save_artefacts(all_individuals, results_dir, args)

    print("\nExperiment complete.")


if __name__ == "__main__":
    main()
