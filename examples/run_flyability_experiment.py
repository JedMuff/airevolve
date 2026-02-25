"""
Flyability Experiment

Measures what fraction of randomly sampled drone morphologies are "flyable" at
various pipeline stages: hover check, repair, initial flight with default gains,
and after CMA-ES controller tuning.

This is a pure sampling + measurement experiment — no evolution involved.

Usage:
    # Quick test run
    python examples/run_flyability_experiment.py --n-drones 50 --max-evals 30 --cma-workers 2

    # Full experiment
    python examples/run_flyability_experiment.py --n-drones 1000 --max-evals 200 --cma-workers 4

    # Custom gate config and arm count
    python examples/run_flyability_experiment.py --n-drones 500 --gate-cfg circle --min-narms 4 --max-narms 8
"""

import sys
import os
import argparse
import json
import time
import numpy as np
from datetime import datetime
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from airevolve.evolution_tools.genome_handlers.spherical_angular_genome_handler import (
    SphericalAngularDroneGenomeHandler,
)
from airevolve.evolution_tools.genome_handlers.repair_workflow import (
    stage1_optimization_repair,
    stage2_hover_check,
    stage3_hover_repair,
)
from airevolve.evolution_tools.genome_handlers.operators.optimization_repair_operator import (
    OptimizationRepairConfig,
)
from airevolve.evolution_tools.evaluators.lee_tune_evaluator import (
    simulate_with_gains,
    _evaluate_solution_wrapper,
)
from airevolve.controllers.utils.gate_configs import GATE_CONFIGS

try:
    import cma
except ImportError:
    cma = None
    print("WARNING: 'cma' package not found. Install with: pip install cma")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Flyability experiment: sample random drones and measure flyability funnel"
    )
    parser.add_argument("--n-drones", type=int, default=100_000,
                        help="Total drones to sample (default: 10_000)")
    parser.add_argument("--max-evals", type=int, default=1000,
                        help="Max CMA-ES evaluations per drone (default: 1000)")
    parser.add_argument("--gates-threshold", type=int, default=1,
                        help="Gates-passed target for early stop (default: 1)")
    parser.add_argument("--gate-cfg", choices=["backandforth", "figure8", "circle", "slalom"],
                        default="figure8", help="Gate layout (default: figure8)")
    parser.add_argument("--sim-time", type=float, default=20.0,
                        help="Simulation time per eval in seconds (default: 20.0)")
    parser.add_argument("--dt", type=float, default=0.005,
                        help="Simulation timestep (default: 0.005)")
    parser.add_argument("--cma-workers", type=int, default=4,
                        help="Parallel workers for CMA-ES (default: 4)")
    parser.add_argument("--min-narms", type=int, default=4,
                        help="Min arms (default: 4)")
    parser.add_argument("--max-narms", type=int, default=4,
                        help="Max arms (default: 4)")
    parser.add_argument("--output-dir", default=".data/flyability",
                        help="Output directory root (default: .data/flyability)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (default: None)")
    parser.add_argument("--timeout", type=float, default=30.0,
                        help="CMA-ES per-eval timeout in seconds (default: 30.0)")
    parser.add_argument("--n-startup-points", type=int, default=1,
                        help="Number of startup control points (default: 1)")
    parser.add_argument("--gate-only-mode", action="store_true",
                        help="Use gate-only mode (pure racing loop without startup phase)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Phase 1: Sampling + hover check + repair (parallel worker)
# ---------------------------------------------------------------------------

def _sample_and_check_single(args):
    """
    Worker function: generate one random drone and run hover check + repair.

    Returns dict with results for each pipeline stage.
    """
    idx, base_seed, handler_kwargs, coordinate_system = args

    seed = base_seed + idx
    handler = SphericalAngularDroneGenomeHandler(
        **handler_kwargs,
        rnd=np.random.default_rng(seed),
    )

    genome = handler.random_population(1)[0]
    n_arms = int(np.sum(~np.isnan(genome[:, 0])))

    result = {
        "drone_id": f"drone_{idx:05d}",
        "genome": genome.tolist(),
        "n_arms": n_arms,
        "hover_check_passed": False,
        "repair_succeeded": False,
    }

    # Stage 2: hover check (strict, no spinning)
    can_hover, _ = stage2_hover_check(genome, verbose=False, allow_spinning=False)
    if not can_hover:
        return result, None

    result["hover_check_passed"] = True

    # Stage 1: optimization repair (fix collisions)
    repair_config = OptimizationRepairConfig(fixed_params=[3, 4])
    repaired, _ = stage1_optimization_repair(
        genome, coordinate_system=coordinate_system, config=repair_config, verbose=False
    )
    if repaired is None:
        return result, None

    # Stage 3: hover repair (align thrust vectors)
    final_ind, _ = stage3_hover_repair(
        repaired, coordinate_system=coordinate_system, verbose=False
    )
    if final_ind is None:
        return result, None

    result["repair_succeeded"] = True
    result["genome"] = final_ind.tolist()
    return result, final_ind


# ---------------------------------------------------------------------------
# Phase 2: CMA-ES tuning with early stop
# ---------------------------------------------------------------------------

def optimize_controller_with_early_stop(
    individual, gate_config, max_evaluations=200, num_workers=4,
    sim_time=20.0, dt=0.005, timeout_per_eval=30.0, gates_threshold=9,
    n_startup_points=1, gate_only_mode=False,
):
    """
    Stage 1 CMA-ES optimization (4 gains only) with early stopping.

    Returns dict with tuning results including eval count, timing, and best gains.
    """
    if cma is None:
        return {
            "gates_passed": 0, "n_evaluations": 0, "tuning_time_seconds": 0.0,
            "best_gains": None, "early_stopped": False, "distance_bonus": 0.0,
            "crashed": True,
        }

    initial_guess = [2.0, 1.5, 0.6, -0.3]
    bounds = [
        [0.01, 10.0],   # pos_P
        [0.01, 10.0],   # vel_P
        [0.01, 10.0],   # att_P
        [-5.0, -0.01],  # rate_P
    ]
    initial_std = 0.8

    options = {
        "bounds": [list(b) for b in zip(*bounds)],
        "maxfevals": max_evaluations,
        "verb_disp": 0,
        "verb_log": 0,
        "tolx": 1e-11,
        "tolfun": 0,
        "tolfunhist": 0,
        "tolflatfitness": max_evaluations,
        "tolstagnation": max_evaluations,
    }

    best_fitness = -float("inf")
    best_result = None
    total_evals = 0
    early_stopped = False
    start_time = time.time()

    try:
        es = cma.CMAEvolutionStrategy(initial_guess, initial_std, options)

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            while not es.stop():
                solutions = es.ask()

                if num_workers > 1:
                    eval_args = [
                        (sol, individual, gate_config, sim_time, dt, n_startup_points, gate_only_mode)
                        for sol in solutions
                    ]
                    future_to_sol = {
                        executor.submit(_evaluate_solution_wrapper, a): a[0]
                        for a in eval_args
                    }
                    results_dict = {}
                    for future in as_completed(future_to_sol):
                        sol = future_to_sol[future]
                        try:
                            score, result = future.result(timeout=timeout_per_eval)
                            results_dict[tuple(sol)] = score
                            if result is not None and result["fitness"] > best_fitness and not result["crashed"]:
                                best_fitness = result["fitness"]
                                best_result = result
                        except (TimeoutError, Exception):
                            results_dict[tuple(sol)] = 1000.0

                    fitness_values = [results_dict[tuple(sol)] for sol in solutions]
                else:
                    fitness_values = []
                    for sol in solutions:
                        score, result = _evaluate_solution_wrapper(
                            (sol, individual, gate_config, sim_time, dt, n_startup_points, gate_only_mode)
                        )
                        fitness_values.append(score)
                        if result is not None and result["fitness"] > best_fitness and not result["crashed"]:
                            best_fitness = result["fitness"]
                            best_result = result

                total_evals += len(solutions)
                es.tell(solutions, fitness_values)

                # Early stop check
                if best_result is not None and best_result["gates_passed"] >= gates_threshold:
                    early_stopped = True
                    break

    except Exception as e:
        print(f"  CMA-ES error: {e}")

    elapsed = time.time() - start_time

    if best_result is not None:
        return {
            "gates_passed": best_result["gates_passed"],
            "n_evaluations": total_evals,
            "tuning_time_seconds": round(elapsed, 2),
            "best_gains": best_result["gains"],
            "early_stopped": early_stopped,
            "distance_bonus": best_result.get("distance_bonus", 0.0),
            "crashed": best_result["crashed"],
        }
    else:
        return {
            "gates_passed": 0,
            "n_evaluations": total_evals,
            "tuning_time_seconds": round(elapsed, 2),
            "best_gains": None,
            "early_stopped": False,
            "distance_bonus": 0.0,
            "crashed": True,
        }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_experiment(args):
    experiment_start = time.time()

    # Resolve gate config
    gate_config = GATE_CONFIGS[args.gate_cfg]

    # Build output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    narms_str = (
        f"{args.min_narms}arms"
        if args.min_narms == args.max_narms
        else f"{args.min_narms}-{args.max_narms}arms"
    )
    exp_dir = os.path.join(
        args.output_dir, f"flyability_{args.gate_cfg}_{narms_str}_{timestamp}"
    )
    drones_dir = os.path.join(exp_dir, "drones")
    os.makedirs(drones_dir, exist_ok=True)

    print("=" * 80)
    print("Flyability Experiment")
    print("=" * 80)
    print(f"Output directory : {exp_dir}")
    print(f"N drones         : {args.n_drones}")
    print(f"Gate config      : {args.gate_cfg}")
    print(f"Arms             : {args.min_narms}-{args.max_narms}")
    print(f"Max CMA-ES evals : {args.max_evals}")
    print(f"Gates threshold  : {args.gates_threshold}")
    print(f"CMA workers      : {args.cma_workers}")
    print(f"Sim time         : {args.sim_time}s  dt={args.dt}s")
    print(f"N startup points : {args.n_startup_points}")
    print(f"Gate-only mode   : {args.gate_only_mode}")
    print(f"Seed             : {args.seed}")
    print("=" * 80)
    print()
    sys.stdout.flush()

    # Genome handler config
    spherical_params = np.array([
        [0.06, 0.17], [-np.pi, np.pi], [-np.pi / 2, np.pi / 2],
        [-np.pi, np.pi], [-np.pi / 2, np.pi / 2], [0, 1],
    ])
    append_arm_chance = 0.0 if args.min_narms == args.max_narms else 0.5
    handler_kwargs = {
        "min_max_narms": (args.min_narms, args.max_narms),
        "append_arm_chance": append_arm_chance,
        "parameter_limits": spherical_params,
        "bilateral_plane_for_symmetry": None,
        "repair": False,
    }
    coordinate_system = "spherical"

    base_seed = args.seed if args.seed is not None else np.random.randint(0, 2**31)

    # ------------------------------------------------------------------
    # Phase 1: Parallel sampling + hover check + repair
    # ------------------------------------------------------------------
    print("Phase 1: Sampling + Hover Check + Repair")
    print("-" * 40)
    sys.stdout.flush()

    phase1_start = time.time()
    num_workers = cpu_count()

    pool_args = [
        (i, base_seed, handler_kwargs, coordinate_system)
        for i in range(args.n_drones)
    ]

    all_drone_records = []
    repaired_drones = []  # (record, individual_array)
    n_hover = 0
    n_repair = 0

    with Pool(processes=num_workers) as pool:
        with tqdm(total=args.n_drones, desc="Phase 1", unit="drone") as pbar:
            for result, individual in pool.imap_unordered(_sample_and_check_single, pool_args, chunksize=10):
                all_drone_records.append(result)
                if result["hover_check_passed"]:
                    n_hover += 1
                if individual is not None:
                    repaired_drones.append((result, individual))
                    n_repair += 1

                pbar.update(1)
                pbar.set_postfix(hover=n_hover, repair=n_repair)

    phase1_time = time.time() - phase1_start
    n_hover_passed = sum(1 for r in all_drone_records if r["hover_check_passed"])
    n_repair_passed = len(repaired_drones)

    print(f"\nPhase 1 complete in {phase1_time:.1f}s")
    print(f"  Sampled:       {args.n_drones}")
    print(f"  Hover passed:  {n_hover_passed} ({n_hover_passed / args.n_drones * 100:.2f}%)")
    print(f"  Repair passed: {n_repair_passed} ({n_repair_passed / args.n_drones * 100:.2f}%)")
    print()
    sys.stdout.flush()

    # ------------------------------------------------------------------
    # Phase 2: Sequential initial flight check + CMA-ES tuning
    # ------------------------------------------------------------------
    print("Phase 2: Initial Flight + CMA-ES Tuning")
    print("-" * 40)
    sys.stdout.flush()

    default_gains = {"pos_P": 2.0, "vel_P": 1.5, "att_P": 0.6, "rate_P": -0.3}
    n_initial_fly = 0
    n_tuned_flyable = 0

    pbar2 = tqdm(repaired_drones, desc="Phase 2", unit="drone")
    for idx, (record, individual) in enumerate(pbar2):
        drone_id = record["drone_id"]
        pbar2.set_postfix(drone=drone_id, fly=n_initial_fly, tuned=n_tuned_flyable)

        # --- Initial flight with default gains ---
        init_result = simulate_with_gains(
            individual,
            default_gains["pos_P"], default_gains["vel_P"],
            default_gains["att_P"], default_gains["rate_P"],
            gate_config, sim_time=args.sim_time, dt=args.dt,
            n_startup_points=args.n_startup_points,
            gate_only_mode=args.gate_only_mode,
        )

        record["initial_flight"] = {
            "gates_passed": init_result["gates_passed"],
            "crashed": init_result["crashed"],
            "flight_time": round(init_result["flight_time"], 3),
        }

        if init_result["gates_passed"] >= args.gates_threshold:
            n_initial_fly += 1

        # --- CMA-ES tuning ---
        tuning = optimize_controller_with_early_stop(
            individual, gate_config,
            max_evaluations=args.max_evals,
            num_workers=args.cma_workers,
            sim_time=args.sim_time,
            dt=args.dt,
            timeout_per_eval=args.timeout,
            gates_threshold=args.gates_threshold,
            n_startup_points=args.n_startup_points,
            gate_only_mode=args.gate_only_mode,
        )

        record["tuning"] = tuning
        if tuning["gates_passed"] >= args.gates_threshold:
            n_tuned_flyable += 1

        pbar2.set_postfix(
            drone=drone_id, fly=n_initial_fly, tuned=n_tuned_flyable,
            gates=tuning["gates_passed"], evals=tuning["n_evaluations"],
        )

        # Save per-drone JSON incrementally
        drone_path = os.path.join(drones_dir, f"{drone_id}.json")
        with open(drone_path, "w") as f:
            json.dump(record, f, indent=2, default=_json_default)

    print(f"\nPhase 2 complete")
    print(f"  Initial fly (>= {args.gates_threshold} gates): {n_initial_fly}")
    print(f"  Tuned flyable:  {n_tuned_flyable}")
    print()
    sys.stdout.flush()

    # ------------------------------------------------------------------
    # Phase 3: Save aggregate summary
    # ------------------------------------------------------------------
    total_time = time.time() - experiment_start

    summary = {
        "experiment_config": {
            "n_drones": args.n_drones,
            "max_evals": args.max_evals,
            "gates_threshold": args.gates_threshold,
            "gate_cfg": args.gate_cfg,
            "sim_time": args.sim_time,
            "dt": args.dt,
            "cma_workers": args.cma_workers,
            "min_narms": args.min_narms,
            "max_narms": args.max_narms,
            "seed": args.seed,
            "timeout": args.timeout,
        },
        "timestamp": datetime.now().isoformat(),
        "n_sampled": args.n_drones,
        "n_hover_passed": n_hover_passed,
        "n_repair_passed": n_repair_passed,
        "n_initial_fly": n_initial_fly,
        "n_tuned_flyable": n_tuned_flyable,
        "pct_hover_passed": round(n_hover_passed / args.n_drones * 100, 2),
        "pct_repair_passed": round(n_repair_passed / args.n_drones * 100, 2),
        "pct_initial_fly": round(n_initial_fly / args.n_drones * 100, 2),
        "pct_tuned_flyable": round(n_tuned_flyable / args.n_drones * 100, 2),
        "total_experiment_time_seconds": round(total_time, 2),
    }

    summary_path = os.path.join(exp_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    all_drones_path = os.path.join(exp_dir, "all_drones.json")
    with open(all_drones_path, "w") as f:
        json.dump(all_drone_records, f, indent=2, default=_json_default)

    # Also save per-drone JSONs for drones that didn't reach Phase 2
    for record in all_drone_records:
        drone_path = os.path.join(drones_dir, f"{record['drone_id']}.json")
        if not os.path.exists(drone_path):
            with open(drone_path, "w") as f:
                json.dump(record, f, indent=2, default=_json_default)

    print("=" * 80)
    print("Experiment Summary")
    print("=" * 80)
    print(f"  Sampled:           {args.n_drones}")
    print(f"  Hover passed:      {n_hover_passed} ({summary['pct_hover_passed']:.2f}%)")
    print(f"  Repair passed:     {n_repair_passed} ({summary['pct_repair_passed']:.2f}%)")
    print(f"  Initial fly:       {n_initial_fly} ({summary['pct_initial_fly']:.2f}%)")
    print(f"  Tuned flyable:     {n_tuned_flyable} ({summary['pct_tuned_flyable']:.2f}%)")
    print(f"  Total time:        {total_time:.1f}s")
    print(f"\nResults saved to: {exp_dir}")
    print("=" * 80)


def _json_default(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


if __name__ == "__main__":
    args = parse_arguments()
    run_experiment(args)
