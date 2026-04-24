"""
Flyability Experiment

Measures what fraction of randomly sampled drone morphologies are "flyable" at
various pipeline stages: hover check, repair, initial flight with default gains,
and after CMA-ES controller tuning.

This is a pure sampling + measurement experiment — no evolution involved.

Usage:
    # Quick test run
    python examples/evolution/run_flyability_experiment.py --n-drones 50 --max-evals 30 --cma-workers 2

    # Full experiment
    python examples/evolution/run_flyability_experiment.py --n-drones 1000 --max-evals 200 --cma-workers 4

    # Custom gate config and arm count
    python examples/evolution/run_flyability_experiment.py --n-drones 500 --gate-cfg circle --min-narms 4 --max-narms 8
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
    _run_cma_stage,
    optimize_controller_with_early_stop,
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
    parser.add_argument("--gates-threshold", type=int, default=8,
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
    parser.add_argument("--skip-repair", action="store_true",
                        help="Skip repair stages; hover-checked genomes go straight to tuning")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Phase 1: Sampling + hover check + repair (parallel worker)
# ---------------------------------------------------------------------------

def _sample_and_check_single(args):
    """
    Worker function: generate one random drone and run hover check + optional repair.

    Returns dict with results for each pipeline stage.
    """
    idx, base_seed, handler_kwargs, coordinate_system, skip_repair = args

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
        "repair_skipped": skip_repair,
    }

    # Hover check (strict, no spinning)
    can_hover, _ = stage2_hover_check(genome, verbose=False, allow_spinning=False)
    if not can_hover:
        return result, None

    result["hover_check_passed"] = True

    if skip_repair:
        # Pass hover-checked genome straight through without repair
        result["repair_succeeded"] = True
        return result, genome

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


def _tune_single_drone(args):
    """Worker function for Phase 2: run initial flight + CMA-ES tuning for one drone.

    Designed to be called via multiprocessing.Pool (top-level, picklable).
    Each worker runs CMA-ES single-threaded (num_workers=1).
    """
    (record, individual_list, gate_config, default_gains, bspline_timing,
     max_evals, sim_time, dt, timeout, gates_threshold) = args

    individual = np.array(individual_list)

    # --- Initial flight with default gains ---
    init_result = simulate_with_gains(
        individual,
        default_gains["pos_P"], default_gains["vel_P"],
        default_gains["att_P"], default_gains["rate_P"],
        gate_config, sim_time=sim_time, dt=dt,
        bspline_timing=bspline_timing,
    )

    record["initial_flight"] = {
        "gates_passed": init_result["gates_passed"],
        "crashed": init_result["crashed"],
        "flight_time": round(init_result["flight_time"], 3),
    }

    initial_fly = init_result["gates_passed"] >= gates_threshold

    # --- CMA-ES tuning (single-threaded inside worker) ---
    tuning = optimize_controller_with_early_stop(
        individual, gate_config,
        max_evaluations=max_evals,
        num_workers=1,
        sim_time=sim_time,
        dt=dt,
        timeout_per_eval=timeout,
        gates_threshold=gates_threshold,
        bspline_timing=bspline_timing,
    )

    record["tuning"] = tuning
    tuned_flyable = tuning["gates_passed"] >= gates_threshold

    return record, initial_fly, tuned_flyable


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
    print(f"Skip repair      : {args.skip_repair}")
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
    # Standard quad reference (pipeline sanity check)
    # ------------------------------------------------------------------
    # create_2inch_quad() morphology from tune_lee_controller_gates.py
    # expressed as a genome in ENU spherical coordinates.
    std_quad_arm = 0.06 * np.sqrt(2)
    std_quad_genome = np.array([
        [std_quad_arm,  np.pi / 4,      0, 0, 0, 0],   # NED: (+0.06, +0.06, 0), CCW
        [std_quad_arm, -np.pi / 4,      0, 0, 0, 1],   # NED: (-0.06, +0.06, 0), CW
        [std_quad_arm, -3 * np.pi / 4,  0, 0, 0, 0],   # NED: (-0.06, -0.06, 0), CCW
        [std_quad_arm,  3 * np.pi / 4,  0, 0, 0, 1],   # NED: (+0.06, -0.06, 0), CW
    ])

    # Run through hover check + repair like any other drone
    can_hover, _ = stage2_hover_check(std_quad_genome, verbose=False, allow_spinning=False)
    std_quad_record = {
        "drone_id": "drone_std_quad",
        "genome": std_quad_genome.tolist(),
        "n_arms": 4,
        "hover_check_passed": can_hover,
        "repair_succeeded": False,
        "repair_skipped": args.skip_repair,
    }

    std_quad_individual = None
    if can_hover:
        if args.skip_repair:
            std_quad_record["repair_succeeded"] = True
            std_quad_individual = std_quad_genome
        else:
            repair_config = OptimizationRepairConfig(fixed_params=[3, 4])
            repaired, _ = stage1_optimization_repair(
                std_quad_genome, coordinate_system=coordinate_system,
                config=repair_config, verbose=False,
            )
            if repaired is not None:
                final_ind, _ = stage3_hover_repair(
                    repaired, coordinate_system=coordinate_system, verbose=False,
                )
                if final_ind is not None:
                    std_quad_record["repair_succeeded"] = True
                    std_quad_record["genome"] = final_ind.tolist()
                    std_quad_individual = final_ind

    print(f"Standard quad (create_2inch_quad): hover={can_hover}, "
          f"repair={std_quad_record['repair_succeeded']}")
    sys.stdout.flush()

    # ------------------------------------------------------------------
    # Phase 1: Parallel sampling + hover check + repair
    # ------------------------------------------------------------------
    print("Phase 1: Sampling + Hover Check + Repair")
    print("-" * 40)
    sys.stdout.flush()

    phase1_start = time.time()
    num_workers = cpu_count()

    pool_args = [
        (i, base_seed, handler_kwargs, coordinate_system, args.skip_repair)
        for i in range(args.n_drones)
    ]

    all_drone_records = [std_quad_record]
    repaired_drones = []  # (record, individual_array)
    if std_quad_individual is not None:
        repaired_drones.append((std_quad_record, std_quad_individual))
    n_hover = 1 if can_hover else 0
    n_repair = 1 if std_quad_individual is not None else 0

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
    # Phase 2: Parallel initial flight check + CMA-ES tuning (1 CPU per drone)
    # ------------------------------------------------------------------
    print("Phase 2: Initial Flight + CMA-ES Tuning")
    print("-" * 40)
    sys.stdout.flush()

    # Gains and B-spline timing matched to tune_lee_controller_gates.py defaults
    default_gains = {"pos_P": 14.3, "vel_P": 9.0, "att_P": 2.9, "rate_P": -0.02}
    bspline_timing = np.array([12.7, 4.6, 1.9])  # total_time, velocity_scale, startup_time
    n_initial_fly = 0
    n_tuned_flyable = 0

    # Build args for parallel workers (convert numpy arrays to lists for pickling)
    phase2_args = [
        (record, individual.tolist(), gate_config, default_gains, bspline_timing,
         args.max_evals, args.sim_time, args.dt, args.timeout, args.gates_threshold)
        for record, individual in repaired_drones
    ]

    phase2_workers = cpu_count()
    with Pool(processes=phase2_workers) as pool:
        with tqdm(total=len(repaired_drones), desc="Phase 2", unit="drone") as pbar2:
            for record, initial_fly, tuned_flyable in pool.imap_unordered(_tune_single_drone, phase2_args):
                if initial_fly:
                    n_initial_fly += 1
                if tuned_flyable:
                    n_tuned_flyable += 1

                pbar2.update(1)
                pbar2.set_postfix(
                    drone=record["drone_id"], fly=n_initial_fly, tuned=n_tuned_flyable,
                    gates=record["tuning"]["gates_passed"],
                    evals=record["tuning"]["n_evaluations"],
                )

                # Save per-drone JSON incrementally (in main process)
                drone_path = os.path.join(drones_dir, f"{record['drone_id']}.json")
                with open(drone_path, "w") as f:
                    json.dump(record, f, indent=2, default=_json_default)

                # Update the record in all_drone_records for the final summary
                for i, r in enumerate(all_drone_records):
                    if r["drone_id"] == record["drone_id"]:
                        all_drone_records[i] = record
                        break

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
            "skip_repair": args.skip_repair,
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
