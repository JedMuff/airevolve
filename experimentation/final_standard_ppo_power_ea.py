"""Experiment: Lamarckian Standard-PPO + Power-Aware NSGA-II evaluation.

Architecture
------------
  RL Training  : Standard DroneGateEnv (no power penalties).
                 PPO optimises purely for gate-passing and distance-to-gate.
                 strict_voltage_kill=False, sparse_weight=0.0, overdraw_weight=0.0.

  EA Evaluation: Bi-objective NSGA-II with a standalone LiPoBatteryModel
                 (strict_voltage_kill=True).  Fitness returned to the EA is
                 the tuple (gates_passed, total_energy_j) — maximise gates,
                 minimise energy.

  Lamarckian   : Offspring inherit the parent's trained policy (policy.zip)
                 and continue training from that checkpoint.
                 - Generation 0: Darwinian start, 10M timesteps, no parent.
                 - Generation 1+: 2M timesteps, load parent's policy.zip.
                 - Fallback: if parent policy missing, 3M timesteps from scratch.

Hardware target: AMD EPYC 9654 (Genoa), 192 cores, 320 GB RAM.
  - 24 EA parallel workers via multiprocessing Pool.
  - Each worker uses --torch-threads threads (default 1; set to 8 for 192c Genoa).
  - No GPU required; CPU-only training.

Directory layout (all paths relative to --results-dir)
-------------------------------------------------------
  results/exp_lamarckian_standard_ppo_power_ea/
    config.json                        # Experiment hyperparameters snapshot
    init_pop_stats.json                # Initial population generation statistics
    evolution_data.csv                 # Every individual, every generation
    pareto_front_final.csv             # Final Pareto-front (rank-0 survivors)
    pareto_front.png                   # Scatter: gates_passed vs. total_energy_j
    objectives_over_generations.png    # Line plots of max-gates & min-energy
    snapshots/
      gen_001/
        genomes.npy                    # Genome arrays of all survivors
        pareto_info.csv                # id, rank, crowding_distance, fitness
        morphology_<id>.png            # 3-D renders of rank-0 individuals
    rl_logs/
      generation_00/
        individual_0000/
          genome.npy                   # Raw genome array
          morphology_config.json       # Arm-by-arm JSON config
          morphology_pre.png           # 3-D render before training
          morphology_post.png          # 3-D render after training
          best_model.zip               # Best checkpoint (EvalCallback)
          final_model.zip              # Checkpoint at end of training
          policy.zip                   # Alias of final_model (legacy compat)
          figure.png                   # Training reward curve
          PPO_1/                       # TensorBoard event files
          eval_logs/                   # EvalCallback evaluation logs

Example
-------
  source drone-project/bin/activate
  python experimentation/final_standard_ppo_power_ea.py

  # Dry-run (no actual training; prints config and exits):
  python experimentation/final_standard_ppo_power_ea.py --dry-run

  # Override key hyperparameters:
  python experimentation/final_standard_ppo_power_ea.py \\
      --population-size 16 --generations 8 --training-timesteps 500000 \\
      --num-workers 8 --torch-threads 4 --results-dir ./test_results
"""

from __future__ import annotations

import sys
import os

# Default to 1 thread — overridden by --torch-threads via pool initializer.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import traceback
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any

import multiprocessing
import multiprocessing.pool
import pandas as pd

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "examples" / "evolution"))

from airevolve.evolution_tools.evaluators.bi_objective_fitness import BiObjectiveFitness
from airevolve.evolution_tools.strategies.nsga2_strategy import evolve_nsga2
from airevolve.evolution_tools.strategies.init_population import (
    generate_viable_initial_population,
    generate_initial_pop_parallel,
)
from airevolve.evolution_tools.inspection_tools.plot_pareto_front import plot_pareto_front
from run_evolution import get_genome_handler_config, create_genome_handler_wrapper
import airevolve.evolution_tools.strategies.evolution_components as _evo_components


import multiprocessing.context


# Non-daemon process built on the *spawn* context. spawn starts a fresh
# interpreter per worker, so workers do NOT inherit the parent's already-imported
# torch/numpy/matplotlib threads + held locks — which is what was deadlocking
# fork-based workers and freezing the whole generation. Non-daemon is still
# required so each worker can spawn its own PPO SubprocVecEnv children.
class NoDaemonSpawnProcess(multiprocessing.context.SpawnProcess):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass


class NoDaemonSpawnContext(multiprocessing.context.SpawnContext):
    Process = NoDaemonSpawnProcess


class NonDaemonPool(multiprocessing.pool.Pool):
    def __init__(self, *args, **kwargs):
        kwargs["context"] = NoDaemonSpawnContext()
        super().__init__(*args, **kwargs)


_EXPERIMENT_NAME = "exp_lamarckian_standard_ppo_power_ea"
_SNAPSHOT_GENS   = {1, 5, 10, 12, 16, 20, 24}

# ── Lamarckian training budgets ──────────────────────────────────────────────
_GEN0_TRAINING_TS   = 10_000_000   # Darwinian start: train from scratch
_LAMARCK_TRAINING_TS = 2_000_000   # Lifetime learning: continue from parent
_FALLBACK_TRAINING_TS = 3_000_000  # Fallback: parent policy missing


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Lamarckian Standard-PPO + Power-Aware NSGA-II experiment runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--training-timesteps", type=float, default=_GEN0_TRAINING_TS,
                   help="PPO training budget for generation 0 (Darwinian start).")
    p.add_argument("--generations",        type=int,   default=24)
    p.add_argument("--population-size",    type=int,   default=24)
    p.add_argument("--num-workers",        type=int,   default=24,
                   help="Parallel EA workers (one per physical CPU core).")
    p.add_argument("--num-mutate",         type=int,   default=24,
                   help="Offspring per generation (default = population_size).")
    p.add_argument("--num-crossover",      type=int,   default=0)
    p.add_argument("--num-envs",           type=int,   default=4,
                   help="SubprocVecEnv workers per individual PPO training run.")
    p.add_argument("--device",             default="cpu",
                   help="PyTorch device for PPO (cpu recommended for Genoa).")
    p.add_argument("--torch-threads",      type=int,   default=1,
                   help="Threads per worker (OMP/MKL/torch). "
                        "Set to 8 for 192-core Genoa (24 workers × 8 = 192).")

    p.add_argument("--genome",    choices=["spherical", "cartesian"], default="spherical")
    p.add_argument("--min-narms", type=int, default=6)
    p.add_argument("--max-narms", type=int, default=6)
    p.add_argument("--gate-cfg",
                   choices=["backandforth", "figure8", "circle", "slalom"],
                   default="figure8")
    p.add_argument("--init-pop-mode",
                   choices=["random", "hover_repair"], default="random")

    p.add_argument("--z-drag-multiplier", type=float, default=5.0,
                   help="Anisotropic Z-axis drag multiplier in DroneGateEnv.")

    p.add_argument("--results-dir", default="results",
                   help="Root output directory.")
    p.add_argument("--run-id",      default=None,
                   help="Override auto-generated run directory name.")
    p.add_argument("--dry-run",     action="store_true",
                   help="Print config and exit without running evolution.")

    args = p.parse_args()
    if args.num_mutate is None:
        args.num_mutate = args.population_size
    return args


def _results_dir(args: argparse.Namespace) -> Path:
    name = args.run_id or _EXPERIMENT_NAME
    path = Path(args.results_dir) / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_config(args: argparse.Namespace, results_dir: Path) -> None:
    snapshot: dict[str, Any] = vars(args).copy()
    snapshot["experiment_name"]    = _EXPERIMENT_NAME
    snapshot["timestamp"]          = datetime.now().isoformat()
    snapshot["ea_strict_kill"]     = True
    snapshot["ea_fitness"]         = "(gates_passed, total_energy_j)"
    snapshot["lamarckian"]         = True
    snapshot["gen0_training_ts"]   = _GEN0_TRAINING_TS
    snapshot["lamarck_training_ts"] = _LAMARCK_TRAINING_TS
    snapshot["fallback_training_ts"] = _FALLBACK_TRAINING_TS
    with open(results_dir / "config.json", "w") as fh:
        json.dump(snapshot, fh, indent=2)
    print(f"  Config saved → {results_dir / 'config.json'}", flush=True)


def _build_fitness(args: argparse.Namespace, config: dict,
                   training_ts_override: int | None = None,
                   load_policy: str | None = None) -> BiObjectiveFitness:
    """Build a BiObjectiveFitness instance.

    Parameters
    ----------
    training_ts_override : if provided, overrides the training_ts in brain_kwargs.
    load_policy          : if provided, sets the load_policy path in brain_kwargs.
    """
    max_steps = max(1, int(round(12.0 / 0.01)))
    ts = int(training_ts_override) if training_ts_override is not None else int(args.training_timesteps)
    brain_kwargs = {
        "gate_cfg":           args.gate_cfg,
        "training_ts":        ts,
        "num_envs":           args.num_envs,
        "device":             args.device,
        "max_steps":          max_steps,
        "z_drag_multiplier":  args.z_drag_multiplier,
        "verbose":            0,
        "progress_bar":       False,
    }
    if load_policy is not None:
        brain_kwargs["load_policy"] = load_policy
    return BiObjectiveFitness(
        brain="rl",
        hover_gradient=False,
        per_individual_repair=False,
        is_indirect=False,
        handler_class=config["handler_class"],
        handler_kwargs=config["handler_kwargs"],
        coordinate_system=config["coordinate_system"],
        brain_kwargs=brain_kwargs,
    )


def _build_initial_population(
    args: argparse.Namespace,
    config: dict,
    WrappedHandler,
) -> tuple[list, dict | None]:
    if args.init_pop_mode == "random":
        return generate_viable_initial_population(
            WrappedHandler(), args.population_size, is_indirect=False,
        )
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


class _SnapshotCallback:
    """Saves per-generation Pareto snapshots and morphology renders."""

    def __init__(self, results_dir: Path, snapshot_gens: set[int]) -> None:
        self._results_dir   = results_dir
        self._snapshot_gens = snapshot_gens

    def __call__(self, generation: int, population) -> None:
        snap_dir = self._results_dir / "snapshots" / f"gen_{generation:03d}"
        snap_dir.mkdir(parents=True, exist_ok=True)

        self._save_genomes(snap_dir, population)
        self._save_pareto_csv(snap_dir, population, generation)
        self._save_morphology_renders(snap_dir, population, generation)

        print(f"[snapshot] Generation {generation:3d} → {snap_dir}", flush=True)

    def _save_genomes(self, snap_dir: Path, population) -> None:
        genomes = []
        for _, row in population.iterrows():
            g = row["genome"]
            genomes.append(np.asarray(g.arms if hasattr(g, "arms") else g))
        try:
            np.save(str(snap_dir / "genomes.npy"), np.array(genomes))
        except ValueError:
            arr = np.empty(len(genomes), dtype=object)
            for i, g in enumerate(genomes):
                arr[i] = g
            np.save(str(snap_dir / "genomes.npy"), arr)

    def _save_pareto_csv(self, snap_dir: Path, population, generation: int) -> None:
        df = population.copy()
        df["gates_passed"]   = df["fitness"].apply(lambda f: f[0])
        df["total_energy_j"] = df["fitness"].apply(lambda f: f[1])
        df["generation"]     = generation
        cols = ["id", "generation", "parent_ids", "rank",
                "crowding_distance", "gates_passed", "total_energy_j"]
        available = [c for c in cols if c in df.columns]
        df[available].to_csv(str(snap_dir / "pareto_info.csv"), index=False)

    def _save_morphology_renders(self, snap_dir: Path, population, generation: int) -> None:
        try:
            from airevolve.evolution_tools.inspection_tools.drone_visualizer import DroneVisualizer
            viz = DroneVisualizer()
            front0 = population[population["rank"] == 0].head(8)
            for _, row in front0.iterrows():
                g          = row["genome"]
                genome_arr = np.asarray(g.arms if hasattr(g, "arms") else g)
                ind_id     = str(row["id"])
                fitness    = row["fitness"]
                try:
                    fig = plt.figure(figsize=plt.figaspect(0.5))
                    ax  = fig.add_subplot(111, projection="3d")
                    viz.plot_3d(
                        genome_arr, ax=ax,
                        title=(f"Gen {generation} | id={ind_id} | "
                               f"gates={fitness[0]} | energy={fitness[1]:.0f}J"),
                        fitness=fitness[0],
                        generation=generation,
                    )
                    fig.savefig(str(snap_dir / f"morphology_{ind_id}.png"),
                                dpi=150, bbox_inches="tight")
                    plt.close(fig)
                except Exception as render_err:
                    print(f"[snapshot] Render failed for id={ind_id}: {render_err}")
        except Exception as viz_err:
            print(f"[snapshot] Visualization skipped: {viz_err}")


class _SafeFitnessCallable:
    """Picklable wrapper around BiObjectiveFitness.

    multiprocessing.Pool.map() serialises the callable with pickle before
    sending it to worker processes.  Closures (local functions returned from
    a factory) cannot be pickled, but instances of a *module-level* class can.
    This wrapper provides identical fault-isolation behaviour.
    """

    _FAIL_ENERGY: float = 1e9

    def __init__(self, fitness_fn: BiObjectiveFitness) -> None:
        self._fn = fitness_fn

    def __call__(self, genome, ind_save_dir):
        try:
            return self._fn(genome, ind_save_dir)
        except Exception:
            tb = traceback.format_exc()
            fail_log = os.path.join(ind_save_dir or ".", "EVAL_FAILED.txt")
            try:
                os.makedirs(ind_save_dir or ".", exist_ok=True)
                with open(fail_log, "w") as fh:
                    fh.write(tb)
            except Exception:
                pass
            print(
                f"[ERROR] Evaluation failed — returning sentinel. "
                f"See {fail_log}\n{tb}",
                flush=True,
            )
            return (0, self._FAIL_ENERGY)


def _save_artefacts(all_individuals, results_dir: Path, args: argparse.Namespace) -> None:
    df = all_individuals.copy()
    df["gates_passed"]   = df["fitness"].apply(lambda f: f[0])
    df["total_energy_j"] = df["fitness"].apply(lambda f: f[1])
    df["id"]             = df["id"].astype(str)

    csv_path = results_dir / "evolution_data.csv"
    df.to_csv(str(csv_path), index=False)
    print(f"  Evolution data  → {csv_path}", flush=True)

    last_gen    = int(df["generation"].max())
    last_pop    = df[df["generation"] == last_gen]
    pareto_df   = (last_pop[last_pop["rank"] == 0]
                   if "rank" in last_pop.columns else last_pop)
    pareto_path = results_dir / "pareto_front_final.csv"
    pareto_df.to_csv(str(pareto_path), index=False)
    print(f"  Pareto front    → {pareto_path}  ({len(pareto_df)} individuals)",
          flush=True)

    try:
        fig = plot_pareto_front(
            str(csv_path),
            title=f"{_EXPERIMENT_NAME}  (gen {last_gen})",
        )
        plot_path = results_dir / "pareto_front.png"
        fig.savefig(str(plot_path), dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Pareto plot     → {plot_path}", flush=True)
    except Exception as e:
        print(f"  [warn] Pareto plot failed: {e}", flush=True)

    try:
        _plot_objectives_over_generations(df, results_dir, args)
    except Exception as e:
        print(f"  [warn] Objectives-over-gens plot failed: {e}", flush=True)


def _plot_objectives_over_generations(df, results_dir: Path,
                                       args: argparse.Namespace) -> None:
    gens  = sorted(df["generation"].unique())
    max_g = [df[df["generation"] == g]["gates_passed"].max() for g in gens]
    min_e = [
        df[df["generation"] == g]["total_energy_j"]
          .apply(lambda x: x if np.isfinite(x) else np.nan).min()
        for g in gens
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
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

    fig.suptitle(_EXPERIMENT_NAME, fontsize=11)
    fig.tight_layout()
    out_path = results_dir / "objectives_over_generations.png"
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Objectives plot → {out_path}", flush=True)


def _print_header(args: argparse.Namespace, results_dir: Path) -> None:
    sep = "=" * 80
    print(sep)
    print(" Experiment: Lamarckian Standard-PPO Training + Power-Aware NSGA-II Evaluation")
    print(sep)
    print(f"  Results directory   : {results_dir}")
    print(f"  Genome encoding     : {args.genome}")
    print(f"  Gate track          : {args.gate_cfg}")
    print(f"  Population size     : {args.population_size}")
    print(f"  Generations         : {args.generations}")
    print(f"  Gen 0 timesteps     : {_GEN0_TRAINING_TS:,}  (Darwinian start)")
    print(f"  Gen 1+ timesteps    : {_LAMARCK_TRAINING_TS:,}  (Lamarckian inheritance)")
    print(f"  Fallback timesteps  : {_FALLBACK_TRAINING_TS:,}  (parent policy missing)")
    print(f"  EA parallel workers : {args.num_workers}")
    print(f"  Torch threads/worker: {args.torch_threads}")
    print(f"  PPO num_envs/indiv  : {args.num_envs}")
    print(f"  PPO device          : {args.device}")
    print()
    print("  ── RL Training (non-power-aware) ─────────────────────────────────")
    print("    sparse_weight          = 0.0  (no energy penalty during training)")
    print("    overdraw_penalty_weight= 0.0  (no current-limit penalty)")
    print("    strict_voltage_kill    = False (voltage sags allowed)")
    print("    use_power_env          = False (standard DroneGateEnv)")
    print()
    print("  ── EA Evaluation (power-aware) ───────────────────────────────────")
    print("    LiPoBatteryModel(strict_voltage_kill=True)")
    print("    Fitness objective      = (gates_passed ↑, total_energy_j ↓)")
    print()
    print("  ── Lamarckian Inheritance ─────────────────────────────────────────")
    print("    Gen 0  → Darwinian: train from scratch (no parent policy)")
    print("    Gen 1+ → Inherit parent policy.zip, continue training (2M ts)")
    print("    Fallback → Missing policy.zip: train from scratch (3M ts)")
    print(sep, flush=True)


# ── Pool worker initializer ──────────────────────────────────────────────────
# Accepts thread count via functools.partial so the CLI --torch-threads value
# propagates into every spawned worker without hardcoding.

def _lamarckian_pool_worker_init(torch_threads: int) -> None:
    """Limit each spawned worker to `torch_threads` threads."""
    t = str(torch_threads)
    os.environ["OMP_NUM_THREADS"] = t
    os.environ["MKL_NUM_THREADS"] = t
    os.environ["OPENBLAS_NUM_THREADS"] = t
    import torch as _torch
    _torch.set_num_threads(torch_threads)


# ── Lamarckian per-individual worker function ────────────────────────────────

def _lamarckian_evaluate_worker(args_tuple):
    """Worker function for Lamarckian evaluation.

    Identical to _evo_components._evaluate_individual_worker but accepts
    a per-individual fitness callable instead of the shared one.
    """
    fitness_function, genome, ind_id, generation, parent_ids, log_dir_base = args_tuple
    return _evo_components.evaluate_individual(
        fitness_function, genome, ind_id, generation, parent_ids, log_dir_base,
    )


def _extract_parent_id(parent_ids_entry) -> str | None:
    """Robustly extract the first valid parent ID from parent_ids.

    parent_ids might be: a list, a tuple, a single string, or None.
    Returns the first non-None parent ID, or None if no valid parent.
    """
    if parent_ids_entry is None:
        return None
    if isinstance(parent_ids_entry, str):
        return parent_ids_entry
    if isinstance(parent_ids_entry, (list, tuple)):
        for pid in parent_ids_entry:
            if pid is not None:
                return str(pid)
    return None


def main() -> None:
    args = parse_args()

    # Apply CLI-driven thread configuration to the main process as well.
    t = str(args.torch_threads)
    os.environ["OMP_NUM_THREADS"] = t
    os.environ["MKL_NUM_THREADS"] = t
    os.environ["OPENBLAS_NUM_THREADS"] = t
    torch.set_num_threads(args.torch_threads)

    results_dir = _results_dir(args)
    _save_config(args, results_dir)
    _print_header(args, results_dir)

    if args.dry_run:
        print("\n[dry-run] Configuration validated. Exiting without running evolution.")
        return

    # ── Genome handler config with custom Lamarckian mutation rates ────────
    config = get_genome_handler_config(
        args.genome, args.min_narms, args.max_narms,
        num_segments=8, initial_hidden_nodes=0,
    )

    # Override mutation parameters: strictly hexacopters (no arm add/remove),
    # custom per-parameter mutation probabilities and scales.
    config["handler_kwargs"]["append_arm_chance"] = 0.0
    config["handler_kwargs"]["mutation_probs"] = [
        0.19,  # Arm length
        0.19,  # Arm polar rotation (anticlockwise azimuth)
        0.19,  # Arm azimuth rotation (pitch/elevation)
        0.19,  # Motor polar rotation
        0.19,  # Motor azimuth rotation
        0.05,  # Motor spin direction (binary flip)
    ]
    config["handler_kwargs"]["mutation_scales_percentage"] = np.array([
        0.1,   # Arm length: 10% of range
        0.2,   # Arm polar rotation: 20% of range
        0.2,   # Arm azimuth rotation: 20% of range
        0.2,   # Motor polar rotation: 20% of range
        0.2,   # Motor azimuth rotation: 20% of range
        0.0,   # Direction: binary flip (scale irrelevant)
    ])

    WrappedHandler = create_genome_handler_wrapper(
        config["handler_class"], config["handler_kwargs"]
    )

    # ── Generation 0 fitness: Darwinian start (10M ts, no parent) ─────────
    fitness_fn_gen0 = _build_fitness(args, config,
                                     training_ts_override=_GEN0_TRAINING_TS,
                                     load_policy=None)
    safe_fitness_gen0 = _SafeFitnessCallable(fitness_fn_gen0)

    # Cache config for building per-individual fitness in Lamarckian generations.
    _cached_config = config

    print("\n--- Phase 1: Initial Population ---", flush=True)
    initial_pop, init_stats = _build_initial_population(args, config, WrappedHandler)
    if initial_pop is None or len(initial_pop) == 0:
        print("[error] Failed to generate initial population. Exiting.")
        return
    if init_stats is not None:
        with open(results_dir / "init_pop_stats.json", "w") as fh:
            json.dump(init_stats, fh, indent=2)
    print(f"  Initial population: {len(initial_pop)} viable individuals.", flush=True)

    snapshot_cb = _SnapshotCallback(results_dir, _SNAPSHOT_GENS)

    # ── Log directory base for resolving parent policy paths ──────────────
    log_dir_base = str(results_dir / "rl_logs")

    def _patched_evaluate_population(
        fitness_function,
        population,
        ids,
        generation,
        all_parent_ids,
        log_dir_base,
        num_workers=1,
    ):
        """Lamarckian-aware evaluate_population.

        Generation 0: uses the shared fitness_function (Darwinian, 10M ts).
        Generation 1+: builds per-individual fitness callables that inject the
        parent's policy.zip path via load_policy, with a 2M ts budget.
        Falls back to 3M ts from scratch if parent policy.zip is missing.
        """
        if num_workers > 1:
            # Build per-individual fitness callables for Lamarckian generations
            if generation >= 1:
                per_individual_fitness = []
                for i in range(len(population)):
                    parent_id = _extract_parent_id(all_parent_ids[i])

                    parent_policy_path = None
                    lamarck_ts = _LAMARCK_TRAINING_TS

                    if parent_id is not None:
                        candidate_path = os.path.join(
                            log_dir_base,
                            f"generation_{generation - 1:02d}",
                            f"individual_{parent_id}",
                            "policy.zip",
                        )
                        try:
                            if os.path.isfile(candidate_path):
                                parent_policy_path = candidate_path
                                lamarck_ts = _LAMARCK_TRAINING_TS
                                print(
                                    f"[Lamarck] Gen {generation} | ind {ids[i]} "
                                    f"← parent {parent_id} policy: {candidate_path}",
                                    flush=True,
                                )
                            else:
                                # Fallback: parent policy missing
                                parent_policy_path = None
                                lamarck_ts = _FALLBACK_TRAINING_TS
                                print(
                                    f"[Lamarck] Gen {generation} | ind {ids[i]} "
                                    f"← parent {parent_id} policy NOT FOUND at "
                                    f"{candidate_path} — fallback to {lamarck_ts:,} ts",
                                    flush=True,
                                )
                        except (OSError, TypeError) as e:
                            # Robust fallback for any file system error
                            parent_policy_path = None
                            lamarck_ts = _FALLBACK_TRAINING_TS
                            print(
                                f"[Lamarck] Gen {generation} | ind {ids[i]} "
                                f"← parent {parent_id} path error: {e} "
                                f"— fallback to {lamarck_ts:,} ts",
                                flush=True,
                            )
                    else:
                        # No parent (should not happen in gen >= 1 for mutations,
                        # but handle gracefully)
                        parent_policy_path = None
                        lamarck_ts = _FALLBACK_TRAINING_TS
                        print(
                            f"[Lamarck] Gen {generation} | ind {ids[i]} "
                            f"← no parent ID — fallback to {lamarck_ts:,} ts",
                            flush=True,
                        )

                    # Build a per-individual BiObjectiveFitness with the specific
                    # parent's policy path and training budget baked in.
                    ind_fitness = _build_fitness(
                        args, _cached_config,
                        training_ts_override=lamarck_ts,
                        load_policy=parent_policy_path,
                    )
                    per_individual_fitness.append(_SafeFitnessCallable(ind_fitness))

                args_list = [
                    (
                        per_individual_fitness[i],
                        genome,
                        ids[i],
                        generation,
                        all_parent_ids[i],
                        log_dir_base,
                    )
                    for i, genome in enumerate(population)
                ]
            else:
                # Generation 0: shared Darwinian fitness
                args_list = [
                    (
                        fitness_function,
                        genome,
                        ids[i],
                        generation,
                        all_parent_ids[i],
                        log_dir_base,
                    )
                    for i, genome in enumerate(population)
                ]

            PER_INDIVIDUAL_TIMEOUT = 7200  # 2 hours for gen 0 (10M ts)
            pool_init = partial(_lamarckian_pool_worker_init, args.torch_threads)
            with NonDaemonPool(
                processes=min(num_workers, len(population)),
                initializer=pool_init,
            ) as pool:
                async_results = [
                    pool.apply_async(
                        _lamarckian_evaluate_worker, (a,)
                    )
                    for a in args_list
                ]
                evaluated = []
                for i, ar in enumerate(async_results):
                    try:
                        evaluated.append(ar.get(timeout=PER_INDIVIDUAL_TIMEOUT))
                    except multiprocessing.TimeoutError:
                        gen_dir = os.path.join(
                            log_dir_base, f"generation_{generation:02d}"
                        )
                        indiv_log_dir = os.path.join(
                            gen_dir, f"individual_{ids[i]}"
                        )
                        print(
                            f"[TIMEOUT] individual {ids[i]} exceeded "
                            f"{PER_INDIVIDUAL_TIMEOUT}s — recording sentinel.",
                            flush=True,
                        )
                        evaluated.append({
                            "id": ids[i],
                            "generation": generation,
                            "genome": population[i],
                            "log_dir": indiv_log_dir,
                            "parent_ids": all_parent_ids[i],
                            "in_pop": False,
                            "fitness": (0, 1e9),
                        })
                pool.terminate()
            return pd.DataFrame(evaluated)
        return _evo_components._orig_evaluate_population(
            fitness_function, population, ids, generation,
            all_parent_ids, log_dir_base, num_workers=1,
        )

    _evo_components._orig_evaluate_population = _evo_components.evaluate_population
    _evo_components.evaluate_population = _patched_evaluate_population

    import airevolve.evolution_tools.strategies.nsga2_strategy as _nsga2
    _nsga2.evaluate_population = _patched_evaluate_population

    print("\n--- Phase 2: NSGA-II Evolution (Lamarckian) ---", flush=True)
    all_individuals = evolve_nsga2(
        fitness_function=safe_fitness_gen0,
        population_size=args.population_size,
        num_generations=args.generations,
        num_mutate=args.num_mutate,
        num_crossover=args.num_crossover,
        mutate_after_crossover=True,
        initial_population=initial_pop,
        log_dir=log_dir_base,
        genome_handler=WrappedHandler,
        verbose=True,
        num_workers=args.num_workers,
        snapshot_callback=snapshot_cb,
    )

    print("\n--- Phase 3: Saving Artefacts ---", flush=True)
    _save_artefacts(all_individuals, results_dir, args)

    print("\nExperiment complete.", flush=True)


if __name__ == "__main__":
    main()
