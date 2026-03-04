"""
Evolution with Lee Controller Tuning (2-Stage CMA-ES)

This script runs evolutionary optimization where each morphology is evaluated by
tuning its Lee controller via a 2-stage CMA-ES pipeline.

Key Features:
- Initial population generated with parallel repair workflow
- Each individual's fitness = max gates passed during controller tuning
- Individuals that fail tuning (0 gates) receive fitness of 0
- Stage 1: Optimise controller gains + trajectory timing (7 params)
- Stage 2: Optimise gains + timing + gate offset control points (7 + n_gates*3 params)

Usage:
    # Basic evolution with Lee tuning
    python examples/run_evolution_with_lee_tuning.py \\
        --gate-cfg circle \\
        --population-size 20 \\
        --generations 10 \\
        --max-evals 100

    # Parallel CMA-ES workers for faster tuning
    python examples/run_evolution_with_lee_tuning.py \\
        --gate-cfg figure8 \\
        --population-size 30 \\
        --generations 20 \\
        --max-evals 200 \\
        --cma-workers 8

    # Quick test run
    python examples/run_evolution_with_lee_tuning.py \\
        --gate-cfg circle \\
        --population-size 10 \\
        --generations 5 \\
        --max-evals 50 \\
        --cma-workers 4
"""

import sys
import os

# Limit BLAS/OpenMP threads to 1 per process — must be set before numpy import.
# Parallelism is handled by multiprocessing Pool, not BLAS threads.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import json
import pickle
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import functools
from datetime import datetime
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from airevolve.evolution_tools.evaluators.lee_tune_evaluator import (
    evaluate_individual_with_tuning,
    optimize_controller_with_early_stop,
)
from airevolve.controllers.utils.gate_configs import GATE_CONFIGS
from airevolve.evolution_tools.strategies.mu_lambda import evolve
from airevolve.evolution_tools.selectors.tournament import tournament_selection
from airevolve.evolution_tools.genome_handlers.spherical_angular_genome_handler import SphericalAngularDroneGenomeHandler
from airevolve.evolution_tools.genome_handlers.cartesian_euler_genome_handler import CartesianEulerDroneGenomeHandler
from airevolve.evolution_tools.genome_handlers.cppn_neat_genome_handler import CPPNNeatDroneGenomeHandler
from airevolve.evolution_tools.genome_handlers.hybrid_cppn_genome_handler import HybridCPPNDroneGenomeHandler
from airevolve.evolution_tools.inspection_tools.utils import evolution_dataframe_to_fitness_array
from airevolve.evolution_tools.inspection_tools.plot_fitness import plot_fitness
from airevolve.evolution_tools.genome_handlers.repair_workflow import (
    stage1_optimization_repair,
    stage2_hover_check,
    stage3_hover_repair
)
from airevolve.evolution_tools.genome_handlers.operators.optimization_repair_operator import (
    OptimizationRepairConfig
)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run evolution with Lee controller tuning (2-Stage CMA-ES)'
    )

    # Genome and evolution parameters
    parser.add_argument('--genome-handler', choices=['spherical', 'cartesian', 'cppn', 'hybrid-cppn'],
                       default='spherical', help='Genome handler to use (default: spherical)')
    parser.add_argument('--population-size', type=int, default=16,
                       help='Population size (default: 20)')
    parser.add_argument('--generations', type=int, default=50,
                       help='Number of generations (default: 50)')
    parser.add_argument('--num-mutate', type=int, default=16,
                       help='Number of individuals to mutate per generation (default: 20)')
    parser.add_argument('--num-crossover', type=int, default=0,
                       help='Number of crossover operations per generation (default: 0)')
    parser.add_argument('--log-dir', default='./.data',
                       help='Directory for logs (default: ./.data)')
    parser.add_argument('--show-plot', action='store_true',
                       help='Show fitness plot at the end')
    parser.add_argument('--save-all-plots', action='store_true',
                       help='Save comprehensive visualization plots including diversity')
    parser.add_argument('--strategy-type', choices=['plus', 'comma'],
                       default='plus', help='Evolution strategy type (default: plus)')

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
    # Gate configuration
    parser.add_argument('--gate-cfg', choices=['backandforth', 'figure8', 'circle', 'slalom'],
                       default='figure8', help='Gate configuration (default: figure8)')

    # Evolution workers (for parallel evaluation of individuals)
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

    # Initial population CMA-ES tuning parameters
    parser.add_argument('--init-pop-max-evals', type=int, default=500,
                       help='CMA-ES budget per drone for initial pop tuning (default: 500)')
    parser.add_argument('--init-pop-gates-threshold', type=int, default=8,
                       help='Min gates to pass for acceptance into initial pop (default: 1)')
    parser.add_argument('--init-pop-tuning-workers', type=int, default=32,
                       help='Workers for Phase 2 tuning pool (default: 32)')
    parser.add_argument('--skip-init-tuning', action='store_true',
                       help='Skip CMA-ES tuning in initial pop (revert to original behavior)')

    return parser.parse_args()


def _try_generate_individual(args):
    """
    Worker function to try generating a single hoverable individual.

    Returns:
        Tuple of (success_individual, status_dict) where status_dict tracks failures.
    """
    idx, base_seed, handler_kwargs, param_limits, coordinate_system = args

    # Create genome handler with unique seed
    seed = base_seed + idx
    handler = SphericalAngularDroneGenomeHandler(
        **handler_kwargs,
        rnd=np.random.default_rng(seed)
    )

    status = {
        'failed_hover': 0,
        'failed_stage1': 0,
        'failed_stage3': 0,
        'success': 0
    }

    # Generate random individual
    ind = handler.random_population(1)[0]

    # STEP 1: Check if it can hover (strict, no spinning)
    can_hover, _ = stage2_hover_check(
        ind,
        verbose=False,
        allow_spinning=False  # Strict hover check
    )

    if not can_hover:
        status['failed_hover'] = 1
        return None, status

    # STEP 2: Apply optimization repair to fix collisions
    repair_config = OptimizationRepairConfig(fixed_params=[3, 4])
    repaired, _ = stage1_optimization_repair(
        ind,
        coordinate_system=coordinate_system,
        config=repair_config,
        verbose=False
    )

    if repaired is None:
        status['failed_stage1'] = 1
        return None, status

    # STEP 3: Apply hover repair to align thrust vectors
    final_ind, _ = stage3_hover_repair(
        repaired,
        coordinate_system=coordinate_system,
        verbose=False
    )

    if final_ind is None:
        status['failed_stage3'] = 1
        return None, status

    # Success!
    status['success'] = 1
    return final_ind, status


def _try_generate_cppn_individual(args):
    """
    Worker function to try generating a single hoverable CPPN individual.

    Generates a random CPPN, decodes to phenotype, and runs the full
    3-stage repair pipeline (hover check → optimization repair → hover repair).
    Returns the CPPN genome if the decoded phenotype survives all stages.

    Returns:
        Tuple of (cppn_genome_or_None, status_dict)
    """
    idx, base_seed, handler_kwargs, handler_class = args

    seed = base_seed + idx
    rng = np.random.default_rng(seed)
    handler = handler_class(**handler_kwargs, rng=rng)

    status = {
        'failed_hover': 0,
        'failed_stage1': 0,
        'failed_stage3': 0,
        'success': 0
    }

    # Decode CPPN to phenotype
    phenotype = handler.get_phenotype()

    # STEP 1: Check if the decoded phenotype can hover (strict, no spinning)
    can_hover, _ = stage2_hover_check(
        phenotype,
        verbose=False,
        allow_spinning=False
    )

    if not can_hover:
        status['failed_hover'] = 1
        return None, status

    # STEP 2: Apply optimization repair to fix collisions
    repair_config = OptimizationRepairConfig(fixed_params=[3, 4])
    repaired, _ = stage1_optimization_repair(
        phenotype,
        coordinate_system='spherical',
        config=repair_config,
        verbose=False
    )

    if repaired is None:
        status['failed_stage1'] = 1
        return None, status

    # STEP 3: Apply hover repair to align thrust vectors
    final_ind, _ = stage3_hover_repair(
        repaired,
        coordinate_system='spherical',
        verbose=False
    )

    if final_ind is None:
        status['failed_stage3'] = 1
        return None, status

    # Success - return the CPPN genome (network object)
    status['success'] = 1
    return handler.genome, status


def _tune_cppn_individual(args):
    """
    Worker function to CMA-ES tune a single CPPN individual.

    Decodes the CPPN to phenotype, runs the 3-stage repair pipeline
    (hover check → optimization repair → hover repair), then tunes the
    controller on the repaired phenotype.
    Returns the CPPN genome if gates_passed >= threshold.

    Returns:
        Tuple of (cppn_genome_or_None, tuning_result_dict)
    """
    (cppn_genome, handler_kwargs, handler_class, gate_config_name, max_evals,
     gates_threshold, sim_time, dt, timeout) = args

    handler = handler_class(genome=cppn_genome, **handler_kwargs)
    phenotype = handler.get_phenotype()

    # Apply the same 3-stage repair pipeline used in _RepairAndEvaluateFitness
    # so the phenotype CMA-ES tunes matches what generation 0 will evaluate.

    # STEP 1: Hover check
    can_hover, _ = stage2_hover_check(
        phenotype, verbose=False, allow_spinning=False
    )
    if not can_hover:
        return None, {"gates_passed": 0, "skipped": "failed_hover_check"}

    # STEP 2: Optimization repair (fix collisions)
    repair_config = OptimizationRepairConfig(fixed_params=[3, 4])
    repaired, _ = stage1_optimization_repair(
        phenotype, coordinate_system='spherical',
        config=repair_config, verbose=False
    )
    if repaired is None:
        return None, {"gates_passed": 0, "skipped": "failed_optimization_repair"}

    # STEP 3: Hover repair (align thrust vectors)
    final, _ = stage3_hover_repair(
        repaired, coordinate_system='spherical', verbose=False
    )
    if final is None:
        return None, {"gates_passed": 0, "skipped": "failed_hover_repair"}

    # Tune the repaired phenotype
    gate_config = GATE_CONFIGS[gate_config_name]
    tuning = optimize_controller_with_early_stop(
        final, gate_config,
        max_evaluations=max_evals,
        num_workers=1,
        sim_time=sim_time,
        dt=dt,
        timeout_per_eval=timeout,
        gates_threshold=gates_threshold,
    )

    if tuning["gates_passed"] >= gates_threshold:
        return cppn_genome, tuning
    else:
        return None, tuning


def _tune_single_individual(args):
    """
    Worker function to CMA-ES tune a single hover+repair-validated individual.

    Runs the 3-stage repair pipeline (hover check → optimization repair →
    hover repair) then tunes the controller on the repaired phenotype.
    Top-level function (picklable for multiprocessing). Each worker runs
    CMA-ES single-threaded (num_workers=1) — the outer Pool provides
    parallelism across drones.

    Returns:
        Tuple of (individual_or_None, tuning_result_dict)
        individual is returned only if gates_passed >= gates_threshold.
    """
    (individual_list, gate_config_name, max_evals, gates_threshold,
     sim_time, dt, timeout) = args

    individual = np.array(individual_list)

    # Apply the same 3-stage repair pipeline used in _RepairAndEvaluateFitness
    # so the phenotype CMA-ES tunes matches what generation 0 will evaluate.

    # STEP 1: Hover check
    can_hover, _ = stage2_hover_check(
        individual, verbose=False, allow_spinning=False
    )
    if not can_hover:
        return None, {"gates_passed": 0, "skipped": "failed_hover_check"}

    # STEP 2: Optimization repair (fix collisions)
    repair_config = OptimizationRepairConfig(fixed_params=[3, 4])
    repaired, _ = stage1_optimization_repair(
        individual, coordinate_system='spherical',
        config=repair_config, verbose=False
    )
    if repaired is None:
        return None, {"gates_passed": 0, "skipped": "failed_optimization_repair"}

    # STEP 3: Hover repair (align thrust vectors)
    final, _ = stage3_hover_repair(
        repaired, coordinate_system='spherical', verbose=False
    )
    if final is None:
        return None, {"gates_passed": 0, "skipped": "failed_hover_repair"}

    # Tune the repaired phenotype
    gate_config = GATE_CONFIGS[gate_config_name]
    tuning = optimize_controller_with_early_stop(
        final, gate_config,
        max_evaluations=max_evals,
        num_workers=1,
        sim_time=sim_time,
        dt=dt,
        timeout_per_eval=timeout,
        gates_threshold=gates_threshold,
    )

    if tuning["gates_passed"] >= gates_threshold:
        return individual, tuning
    else:
        return None, tuning


def generate_initial_pop_parallel(genotype, pop_size, coordinate_system='spherical',
                                  verbose=False, num_workers=None,
                                  gate_cfg='circle', init_pop_max_evals=200,
                                  init_pop_gates_threshold=1,
                                  init_pop_tuning_workers=None,
                                  sim_time=20.0, dt=0.005,
                                  timeout=30.0, skip_init_tuning=False,
                                  handler_type='spherical', handler_kwargs=None,
                                  handler_class=None):
    """
    Generate initial population using parallel sampling + optional CMA-ES tuning.

    Two-phase strategy:
      Phase 1 (fast): Sample many individuals, keep hover+repair survivors (~0.2%)
      Phase 2 (slow): CMA-ES tune each survivor, keep those that pass gates threshold

    For CPPN handler_type: Phase 1 generates random CPPNs and filters by hover
    check on decoded phenotype (no repair stages). Phase 2 tunes controllers on
    the decoded phenotypes.

    Args:
        genotype: Genome handler instance
        pop_size: Size of population to generate
        coordinate_system: 'spherical', 'cartesian', or 'cppn'
        verbose: Print detailed messages
        num_workers: Number of parallel workers for Phase 1 (defaults to CPU count)
        gate_cfg: Gate configuration name for CMA-ES tuning
        init_pop_max_evals: CMA-ES budget per drone for Phase 2
        init_pop_gates_threshold: Min gates to pass for acceptance
        init_pop_tuning_workers: Workers for Phase 2 tuning pool (defaults to CPU count)
        sim_time: Simulation time in seconds
        dt: Time step in seconds
        timeout: Timeout per evaluation in seconds
        skip_init_tuning: If True, skip Phase 2 (revert to original behavior)
        handler_type: 'spherical', 'cartesian', or 'cppn'
        handler_kwargs: Handler constructor kwargs (required for CPPN)

    Returns:
        Array/list of individuals (numpy arrays for direct encoding, CPPNNetwork
        objects for CPPN)
    """
    is_indirect = handler_type in ('cppn', 'hybrid-cppn')
    if num_workers is None:
        num_workers = cpu_count()
    if init_pop_tuning_workers is None:
        init_pop_tuning_workers = cpu_count()

    tuning_label = "DISABLED (--skip-init-tuning)" if skip_init_tuning else "ENABLED"
    print(f"Generating initial population of size {pop_size} using {num_workers} parallel workers...")
    if is_indirect:
        print("Strategy: Random CPPN generation -> Decode to phenotype -> Strict hover check -> Fix collisions -> Align thrust")
    else:
        print("Strategy: Parallel sampling -> Strict hover check -> Fix collisions -> Align thrust")
    if not skip_init_tuning:
        print(f"       -> CMA-ES tuning (budget={init_pop_max_evals}, threshold={init_pop_gates_threshold} gates)")
    print(f"CMA-ES init-pop tuning: {tuning_label}")
    print("Expected Phase 1 success rate: ~0.2% (need ~{:,} samples for {} individuals)\n".format(pop_size * 500, pop_size))
    sys.stdout.flush()

    start_time = time.time()

    # Generate random base seed for this run
    base_seed = np.random.randint(0, 2**31)
    print(f"Base seed for this run: {base_seed}\n")

    batch_size = pop_size * 1000
    max_iterations = 100

    accepted_individuals = []
    total_stats = {
        'failed_hover': 0,
        'failed_stage1': 0,
        'failed_stage3': 0,
        'phase1_success': 0,
        'phase2_attempted': 0,
        'phase2_accepted': 0,
        'total_attempts': 0
    }

    for batch_idx in range(max_iterations):
        if len(accepted_individuals) >= pop_size:
            break

        remaining = pop_size - len(accepted_individuals)

        print(f"\n--- Iteration {batch_idx + 1}: Sampling {batch_size} individuals (need {remaining} more) ---")

        # ==============================================================
        # PHASE 1: Fast parallel hover check + repair
        # ==============================================================
        if is_indirect:
            args_list = [
                (
                    batch_idx * batch_size + i,
                    base_seed,
                    handler_kwargs,
                    handler_class,
                )
                for i in range(batch_size)
            ]
            phase1_worker = _try_generate_cppn_individual
        else:
            handler_config = {
                'min_max_narms': (genotype.min_narms, genotype.max_narms),
                'append_arm_chance': genotype.append_arm_chance,
                'parameter_limits': genotype.parameter_limits,
                'bilateral_plane_for_symmetry': genotype.bilateral_plane_for_symmetry,
                'repair': genotype.repair_enabled,
            }
            args_list = [
                (
                    batch_idx * batch_size + i,
                    base_seed,
                    handler_config,
                    genotype.parameter_limits,
                    coordinate_system
                )
                for i in range(batch_size)
            ]
            phase1_worker = _try_generate_individual

        phase1_survivors = []

        with Pool(processes=num_workers) as pool:
            with tqdm(total=batch_size, desc=f"Phase 1 (batch {batch_idx + 1})", unit="ind") as pbar:
                for result, status in pool.imap_unordered(phase1_worker, args_list, chunksize=10):
                    total_stats['total_attempts'] += 1
                    total_stats['failed_hover'] += status['failed_hover']
                    total_stats['failed_stage1'] += status['failed_stage1']
                    total_stats['failed_stage3'] += status['failed_stage3']
                    total_stats['phase1_success'] += status['success']

                    if result is not None:
                        phase1_survivors.append(result)

                    pbar.update(1)
                    pbar.set_postfix({
                        'survivors': len(phase1_survivors),
                        'rate': f"{total_stats['phase1_success']/max(1,total_stats['total_attempts'])*100:.2f}%"
                    })

        print(f"  Phase 1 survivors: {len(phase1_survivors)}")

        if len(phase1_survivors) == 0:
            print("  No survivors in this batch, continuing...")
            continue

        # ==============================================================
        # PHASE 2: CMA-ES tuning (skip if --skip-init-tuning)
        # ==============================================================
        if skip_init_tuning:
            accepted_individuals.extend(phase1_survivors)
            print(f"  (tuning skipped) Accepted: {len(accepted_individuals)}/{pop_size}")
            if len(accepted_individuals) >= pop_size:
                print(f"  Target reached!")
                break
            continue

        print(f"  Phase 2: CMA-ES tuning {len(phase1_survivors)} survivors "
              f"(budget={init_pop_max_evals}, threshold={init_pop_gates_threshold} gates, "
              f"workers={init_pop_tuning_workers})...")

        if is_indirect:
            tuning_args = [
                (
                    ind,
                    handler_kwargs,
                    handler_class,
                    gate_cfg,
                    init_pop_max_evals,
                    init_pop_gates_threshold,
                    sim_time, dt,
                    timeout,
                )
                for ind in phase1_survivors
            ]
            phase2_worker = _tune_cppn_individual
        else:
            tuning_args = [
                (
                    ind.tolist(),
                    gate_cfg,
                    init_pop_max_evals,
                    init_pop_gates_threshold,
                    sim_time, dt,
                    timeout,
                )
                for ind in phase1_survivors
            ]
            phase2_worker = _tune_single_individual

        batch_accepted = 0
        batch_attempted = 0

        with Pool(processes=init_pop_tuning_workers) as pool:
            with tqdm(total=len(phase1_survivors), desc=f"Phase 2 (batch {batch_idx + 1})", unit="drone") as pbar2:
                for tuned_ind, tuning_result in pool.imap_unordered(phase2_worker, tuning_args):
                    batch_attempted += 1
                    total_stats['phase2_attempted'] += 1

                    if tuned_ind is not None:
                        accepted_individuals.append(tuned_ind)
                        batch_accepted += 1
                        total_stats['phase2_accepted'] += 1

                    gates = tuning_result.get('gates_passed', 0)
                    evals = tuning_result.get('n_evaluations', 0)
                    pbar2.update(1)
                    pbar2.set_postfix({
                        'accepted': f"{len(accepted_individuals)}/{pop_size}",
                        'gates': gates,
                        'evals': evals,
                    })

                    # Break early if we have enough
                    if len(accepted_individuals) >= pop_size:
                        break

        print(f"  Phase 2: {batch_accepted}/{batch_attempted} passed "
              f"(total accepted: {len(accepted_individuals)}/{pop_size})")

        # Update Phase 2 success rate estimate for adaptive batch sizing
        if len(accepted_individuals) >= pop_size:
            print(f"  Target reached!")
            break

    end_time = time.time()

    # Report results
    print(f"\n{'='*80}")
    print("Initial Population Generation Results")
    print(f"{'='*80}")
    print(f"Successfully generated: {len(accepted_individuals)}/{pop_size}")
    print(f"Total Phase 1 attempts: {total_stats['total_attempts']:,}")
    print(f"Phase 1 success rate: {total_stats['phase1_success']/max(1,total_stats['total_attempts'])*100:.3f}%")
    print(f"\nPhase 1 failure breakdown:")
    print(f"  Failed hover check: {total_stats['failed_hover']:,} ({total_stats['failed_hover']/max(1,total_stats['total_attempts'])*100:.1f}%)")
    print(f"  Failed optimization repair: {total_stats['failed_stage1']:,} ({total_stats['failed_stage1']/max(1,total_stats['total_attempts'])*100:.1f}%)")
    print(f"  Failed hover repair: {total_stats['failed_stage3']:,} ({total_stats['failed_stage3']/max(1,total_stats['total_attempts'])*100:.1f}%)")
    if not skip_init_tuning:
        print(f"\nPhase 2 (CMA-ES tuning):")
        print(f"  Attempted: {total_stats['phase2_attempted']}")
        print(f"  Accepted:  {total_stats['phase2_accepted']}")
        if total_stats['phase2_attempted'] > 0:
            print(f"  Success rate: {total_stats['phase2_accepted']/total_stats['phase2_attempted']*100:.1f}%")
    print(f"\nTime taken: {end_time - start_time:.1f}s")
    print(f"{'='*80}\n")

    if len(accepted_individuals) < pop_size:
        print(f"Warning: Could only generate {len(accepted_individuals)}/{pop_size} individuals")
        print(f"Consider increasing max_iterations or adjusting parameters.\n")

    stats = {
        'total_attempts': total_stats['total_attempts'],
        'failed_hover': total_stats['failed_hover'],
        'failed_stage1': total_stats['failed_stage1'],
        'failed_stage3': total_stats['failed_stage3'],
        'phase1_success': total_stats['phase1_success'],
        'phase2_attempted': total_stats['phase2_attempted'],
        'phase2_accepted': total_stats['phase2_accepted'],
        'wall_clock_seconds': end_time - start_time,
        'num_iterations': batch_idx + 1,
        'pop_size_requested': pop_size,
        'pop_size_generated': len(accepted_individuals),
        'handler_type': handler_type,
        'skip_init_tuning': skip_init_tuning,
    }

    if len(accepted_individuals) == 0:
        return None, stats
    trimmed = accepted_individuals[:pop_size]
    if is_indirect:
        return trimmed, stats  # List of CPPNNetwork/HybridGenome objects
    return np.array(trimmed), stats


def get_genome_handler_config(handler_type, min_narms=6, max_narms=6,
                              num_segments=8, initial_hidden_nodes=0):
    """
    Get genome handler class and configuration based on type.

    Note: Symmetry is NOT supported (always None).
    Note: Built-in repair is DISABLED (repair=False) - we use external repair workflow.
    """
    # Spherical parameter limits: [r, theta, phi, pitch, yaw, direction]
    spherical_params = np.array([
        [0.055, 0.105],
        [-np.pi, np.pi],
        [0, np.pi],
        [-np.pi, np.pi],
        [-np.pi, np.pi],
        [0, 1]
    ])

    append_arm_chance = 0.0 if min_narms == max_narms else 0.5

    if handler_type == 'spherical':
        return {
            'handler_class': SphericalAngularDroneGenomeHandler,
            'handler_kwargs': {
                'min_max_narms': (min_narms, max_narms),
                'append_arm_chance': append_arm_chance,
                'parameter_limits': spherical_params,
                'bilateral_plane_for_symmetry': None,
                'repair': False
            },
            'param_limits': spherical_params,
            'coordinate_system': 'spherical'
        }
    elif handler_type == 'cartesian':
        return {
            'handler_class': CartesianEulerDroneGenomeHandler,
            'handler_kwargs': {
                'min_max_narms': (min_narms, max_narms),
                'append_arm_chance': append_arm_chance,
                'bilateral_plane_for_symmetry': None,
                'repair': False
            },
            'param_limits': spherical_params,
            'coordinate_system': 'cartesian'
        }
    elif handler_type == 'cppn':
        return {
            'handler_class': CPPNNeatDroneGenomeHandler,
            'handler_kwargs': {
                'num_segments': num_segments,
                'min_max_narms': (min_narms, max_narms),
                'initial_hidden_nodes': initial_hidden_nodes,
                'repair': False,
            },
            'param_limits': None,
            'coordinate_system': 'cppn'
        }
    elif handler_type == 'hybrid-cppn':
        return {
            'handler_class': HybridCPPNDroneGenomeHandler,
            'handler_kwargs': {
                'min_max_narms': (min_narms, max_narms),
                'initial_hidden_nodes': initial_hidden_nodes,
                'repair': False,
            },
            'param_limits': None,
            'coordinate_system': 'hybrid-cppn'
        }
    else:
        raise ValueError(f"Unknown genome handler type: {handler_type}")


class _RepairAndEvaluateFitness:
    """Picklable fitness wrapper that runs the full repair pipeline before evaluation.

    Pipeline: genome → (CPPN decode) → hover check → optimization repair →
    hover repair → evaluate.  Returns fitness 0 if any repair stage fails.
    """

    def __init__(self, gate_cfg, max_evals, cma_workers, sim_time, dt, timeout,
                 coordinate_system, is_indirect=False, handler_class=None,
                 handler_kwargs=None):
        self.gate_cfg = gate_cfg
        self.max_evals = max_evals
        self.cma_workers = cma_workers
        self.sim_time = sim_time
        self.dt = dt
        self.timeout = timeout
        self.coordinate_system = coordinate_system
        self.is_indirect = is_indirect
        self.handler_class = handler_class
        self.handler_kwargs = handler_kwargs

    def __call__(self, genome, ind_save_dir):
        # Save genome immediately so the directory is never left empty
        if ind_save_dir is not None:
            os.makedirs(ind_save_dir, exist_ok=True)
            if self.is_indirect:
                with open(os.path.join(ind_save_dir, "genotype.pkl"), 'wb') as f:
                    pickle.dump(genome, f)
            else:
                np.save(os.path.join(ind_save_dir, "genome.npy"), genome)

        # Decode indirect encoding to phenotype if needed
        if self.is_indirect:
            handler = self.handler_class(genome=genome, **self.handler_kwargs)
            phenotype = handler.get_phenotype()
            repair_coord = 'spherical'  # Indirect phenotype is in spherical format
        else:
            phenotype = genome
            repair_coord = self.coordinate_system

        # Hover check
        can_hover, _ = stage2_hover_check(
            phenotype, verbose=False, allow_spinning=False
        )
        if not can_hover:
            return 0

        # Optimization repair (fix collisions)
        repair_config = OptimizationRepairConfig(fixed_params=[3, 4])
        repaired, _ = stage1_optimization_repair(
            phenotype, coordinate_system=repair_coord,
            config=repair_config, verbose=False
        )
        if repaired is None:
            return 0

        # Hover repair (align thrust vectors)
        final, _ = stage3_hover_repair(
            repaired, coordinate_system=repair_coord, verbose=False
        )
        if final is None:
            return 0

        # Evaluate the repaired phenotype
        fitness = evaluate_individual_with_tuning(
            final, ind_save_dir,
            gate_cfg=self.gate_cfg,
            max_evals=self.max_evals,
            num_workers=self.cma_workers,
            sim_time=self.sim_time,
            dt=self.dt,
            timeout=self.timeout,
            num=None,
        )

        return fitness


def create_fitness_function(args, config=None):
    """Create fitness function with repair pipeline for all genome types."""
    is_indirect = args.genome_handler in ('cppn', 'hybrid-cppn')
    return _RepairAndEvaluateFitness(
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


def create_genome_handler_wrapper(handler_class, handler_kwargs):
    """Create a wrapper class that provides the correct constructor interface."""
    class GenomeHandlerWrapper(handler_class):
        def __init__(self, *_args, genome=None, **_kwargs):
            super().__init__(genome=genome, **handler_kwargs)

    return GenomeHandlerWrapper


def main():
    """Main evolution function with Lee controller tuning."""
    args = parse_arguments()

    # Generate automatic experiment name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    narms_str = f"{args.min_narms}arms" if args.min_narms == args.max_narms else f"{args.min_narms}-{args.max_narms}arms"
    exp_name = f"lee_tuning_{args.gate_cfg}_{narms_str}_{timestamp}"

    # Create full log directory path
    full_log_dir = os.path.join(args.log_dir, exp_name)
    os.makedirs(full_log_dir, exist_ok=True)

    print("=" * 80)
    print("Evolution with Lee Controller Tuning (2-Stage CMA-ES)")
    print("=" * 80)
    sys.stdout.flush()
    print(f"Experiment name: {exp_name}")
    print(f"Log directory: {full_log_dir}")
    print(f"Genome handler: {args.genome_handler.upper()}")
    print(f"Population size: {args.population_size}")
    print(f"Generations: {args.generations}")
    print(f"Mutations per generation: {args.num_mutate}")
    print(f"Crossovers per generation: {args.num_crossover}")
    print(f"Strategy type: {args.strategy_type}")
    print(f"Gate configuration: {args.gate_cfg}")
    print(f"Number of arms: {args.min_narms}-{args.max_narms}")
    print()
    print("2-STAGE CMA-ES TUNING PARAMETERS:")
    print(f"  Stage 1: gains + timing (7 params, 40% budget)")
    print(f"  Stage 2: gains + timing + gate offsets (7 + n_gates*3 params, 60% budget)")
    print(f"  Max evaluations per individual: {args.max_evals}")
    print(f"  CMA-ES parallel workers: {args.cma_workers}")
    print(f"  Simulation time: {args.sim_time}s")
    print(f"  Time step: {args.dt}s")
    print(f"  Timeout per evaluation: {args.timeout}s")
    print()
    print(f"Evolution parallel workers: {args.num_workers}")
    if args.genome_handler == 'cppn':
        print(f"CPPN segments: {args.num_segments}")
        print(f"CPPN initial hidden nodes: {args.initial_hidden_nodes}")
    elif args.genome_handler == 'hybrid-cppn':
        print(f"Hybrid-CPPN initial hidden nodes: {args.initial_hidden_nodes}")
    print(f"Repair workflow: ENABLED (3-stage: Optimization -> Hover Check -> Hover Repair)")
    print(f"Symmetry: DISABLED (not supported)")
    print()
    print("INITIAL POPULATION CMA-ES TUNING:")
    if args.skip_init_tuning:
        print(f"  Status: DISABLED (--skip-init-tuning)")
    else:
        print(f"  Status: ENABLED")
        print(f"  Max evals per drone: {args.init_pop_max_evals}")
        print(f"  Gates threshold: {args.init_pop_gates_threshold}")
        print(f"  Tuning workers: {args.init_pop_tuning_workers or 'cpu_count'}")
    print()
    print(f"CRITICAL: Each individual's controller is tuned via 2-Stage CMA-ES")
    print(f"CRITICAL: Fitness = max gates passed during tuning (0 if failed)")
    print("=" * 80)
    print()

    # Get genome handler configuration
    config = get_genome_handler_config(
        args.genome_handler, args.min_narms, args.max_narms,
        num_segments=args.num_segments,
        initial_hidden_nodes=args.initial_hidden_nodes,
    )

    # Create fitness function
    fitness_function = create_fitness_function(args, config=config)

    # Create a wrapper class for the genome handler
    WrappedHandler = create_genome_handler_wrapper(config['handler_class'], config['handler_kwargs'])

    # Generate initial population with parallel repair workflow + optional CMA-ES tuning
    print("=" * 80)
    print("Phase 1: Initial Population Generation (Parallel)")
    print("=" * 80)
    initial_population, init_pop_stats = generate_initial_pop_parallel(
        WrappedHandler(),
        args.population_size,
        coordinate_system=config['coordinate_system'],
        verbose=False,
        num_workers=args.init_pop_tuning_workers,
        gate_cfg=args.gate_cfg,
        init_pop_max_evals=args.init_pop_max_evals,
        init_pop_gates_threshold=args.init_pop_gates_threshold,
        init_pop_tuning_workers=args.init_pop_tuning_workers,
        sim_time=args.sim_time,
        dt=args.dt,
        timeout=args.timeout,
        skip_init_tuning=args.skip_init_tuning,
        handler_type=args.genome_handler,
        handler_kwargs=config['handler_kwargs'],
        handler_class=config['handler_class'],
    )

    # Save initial population generation stats
    with open(os.path.join(full_log_dir, "init_pop_stats.json"), 'w') as f:
        json.dump(init_pop_stats, f, indent=2)

    if initial_population is None or len(initial_population) == 0:
        print("\n✗ Failed to generate initial population. Exiting.")
        return

    # Run evolution
    print("=" * 80)
    print("Phase 2: Evolution with Lee Controller Tuning")
    print("=" * 80)
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

    # Save complete evolution data as CSV
    evolution_csv_path = f"{full_log_dir}/evolution_data.csv"
    all_individuals_copy = all_individuals.copy()
    all_individuals_copy['id'] = all_individuals_copy['id'].astype(str)
    all_individuals_copy.to_csv(evolution_csv_path, index=False)
    print(f"Evolution data saved to: {evolution_csv_path}")

    # Get the best individual from the last generation
    last_gen = args.generations - 1
    best_individual = all_individuals.loc[all_individuals['generation'] == last_gen].sort_values(
        by='fitness', ascending=False
    ).iloc[0]
    print(f"\nBest individual in generation {last_gen}: {best_individual['id']}, "
          f"Fitness: {best_individual['fitness']} gates passed")

    print(f"Genome: \n{best_individual['genome']}")

    # Plot fitness evolution
    print("\nGenerating fitness evolution plot...")
    fitness_array = evolution_dataframe_to_fitness_array(all_individuals, population_size=args.population_size)

    _fig, ax = plt.subplots(figsize=(12, 8))
    plot_fitness(ax, fitness_array)
    ax.set_title(f"Lee Tuning Evolution ({args.genome_handler.upper()} Handler, {args.gate_cfg})")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Number of Gates Passed (via CMA-ES Tuning)")
    plt.tight_layout()

    # Save plot
    fitness_plot_path = f"{full_log_dir}/fitness_evolution_{args.genome_handler}_{args.gate_cfg}.png"
    plt.savefig(fitness_plot_path, dpi=300, bbox_inches='tight')
    print(f"Fitness plot saved to: {fitness_plot_path}")

    if args.show_plot:
        plt.show()

    print(f"\nEvolution completed! Best fitness: {best_individual['fitness']} gates passed")


if __name__ == "__main__":
    main()
