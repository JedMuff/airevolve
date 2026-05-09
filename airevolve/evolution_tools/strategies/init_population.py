"""Parallel initial-population generation with the 3-stage repair workflow.

Phase 1: Sample many random individuals, keep those that survive
hover-check → optimization repair → hover repair.

Lifted from the prior `examples/evolution/run_evolution_with_lee_tuning.py`
(now deleted; see git history for provenance). The Phase 2 CMA-ES tuning
step that existed in the original script is intentionally dropped — the
unified runner does not use it.
"""
import sys
import time
from multiprocessing import Pool, cpu_count

import numpy as np
from tqdm import tqdm

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


def generate_viable_initial_population(
    handler_instance,
    population_size: int,
    is_indirect: bool = False,
    max_attempts: int = 50_000,
):
    """Rejection-sample random genomes until ``population_size`` hover-viable ones are found.

    For each candidate, only the fast static-hover rank-test
    (``stage2_hover_check``) is applied — no expensive repair pipeline.
    Unviable genomes are silently discarded and a new one is drawn.

    Prints a progress line for every accepted individual so progress is
    visible during long runs.

    Parameters
    ----------
    handler_instance : GenomeHandler
        A fully configured genome handler (already constructed with the
        experiment's bounds and operator settings).
    population_size  : int
        Number of viable individuals to collect before returning.
    is_indirect      : bool
        True for CPPN / hybrid-cppn genomes.  Uses ``generate_random_population``
        + ``get_phenotype()`` to decode; False (default) uses ``random_population``
        which returns arm-matrix arrays directly.
    max_attempts     : int
        Safety cap. Raises ``RuntimeError`` if reached before the population
        is complete (prevents an infinite loop on degenerate configs).

    Returns
    -------
    genomes : list
        Viable genomes — numpy arrays for direct encodings, CPPN/graph objects
        for indirect encodings.
    stats : dict
        ``{"total_attempts": int, "acceptance_rate": float}``
    """
    n_width = len(str(population_size))
    viable   = []
    attempts = 0

    print(
        f"\n[init-pop] Rejection sampling — need {population_size} "
        f"hover-viable drones…",
        flush=True,
    )

    while len(viable) < population_size:
        if attempts >= max_attempts:
            raise RuntimeError(
                f"[init-pop] Safety cap reached: {max_attempts} attempts, "
                f"only {len(viable)}/{population_size} viable drones found.  "
                f"Try loosening arm-count or geometry bounds."
            )

        attempts += 1

        if is_indirect:
            [h]       = handler_instance.generate_random_population(1)
            genome    = h.genome
            phenotype = h.get_phenotype()
        else:
            genome    = handler_instance.random_population(1)[0]
            phenotype = np.asarray(genome)

        can_hover, _ = stage2_hover_check(phenotype, verbose=False, allow_spinning=False)
        if not can_hover:
            continue

        viable.append(genome)
        n_found = len(viable)
        print(
            f"[init-pop]   Found viable initial drone "
            f"{n_found:{n_width}d}/{population_size}  (attempt {attempts})",
            flush=True,
        )

    rate = population_size / attempts
    print(
        f"[init-pop] Done — {population_size} viable drones in {attempts} attempts "
        f"(acceptance rate {rate:.1%})\n",
        flush=True,
    )
    return viable, {"total_attempts": attempts, "acceptance_rate": rate}


def _try_generate_individual(args):
    """Worker: generate one direct-encoded hoverable+repaired individual."""
    idx, base_seed, handler_kwargs, _param_limits, coordinate_system = args

    seed = base_seed + idx
    handler = SphericalAngularDroneGenomeHandler(
        **handler_kwargs,
        rnd=np.random.default_rng(seed),
    )

    status = {"failed_hover": 0, "failed_stage1": 0, "failed_stage3": 0, "success": 0}

    ind = handler.random_population(1)[0]

    can_hover, _ = stage2_hover_check(ind, verbose=False, allow_spinning=False)
    if not can_hover:
        status["failed_hover"] = 1
        return None, status

    repair_config = OptimizationRepairConfig(fixed_params=[3, 4])
    repaired, _ = stage1_optimization_repair(
        ind, coordinate_system=coordinate_system,
        config=repair_config, verbose=False,
    )
    if repaired is None:
        status["failed_stage1"] = 1
        return None, status

    final_ind, _ = stage3_hover_repair(
        repaired, coordinate_system=coordinate_system, verbose=False,
    )
    if final_ind is None:
        status["failed_stage3"] = 1
        return None, status

    status["success"] = 1
    return final_ind, status


def _try_generate_cppn_individual(args):
    """Worker: generate one CPPN/hybrid-cppn whose decoded phenotype is hoverable+repaired."""
    idx, base_seed, handler_kwargs, handler_class = args

    seed = base_seed + idx
    rng = np.random.default_rng(seed)
    handler = handler_class(**handler_kwargs, rng=rng)

    status = {"failed_hover": 0, "failed_stage1": 0, "failed_stage3": 0, "success": 0}

    phenotype = handler.get_phenotype()

    can_hover, _ = stage2_hover_check(phenotype, verbose=False, allow_spinning=False)
    if not can_hover:
        status["failed_hover"] = 1
        return None, status

    repair_config = OptimizationRepairConfig(fixed_params=[3, 4])
    repaired, _ = stage1_optimization_repair(
        phenotype, coordinate_system="spherical",
        config=repair_config, verbose=False,
    )
    if repaired is None:
        status["failed_stage1"] = 1
        return None, status

    final_ind, _ = stage3_hover_repair(
        repaired, coordinate_system="spherical", verbose=False,
    )
    if final_ind is None:
        status["failed_stage3"] = 1
        return None, status

    status["success"] = 1
    return handler.genome, status


def generate_initial_pop_parallel(
    genotype,
    pop_size,
    coordinate_system="spherical",
    num_workers=None,
    handler_type="spherical",
    handler_kwargs=None,
    handler_class=None,
):
    """Generate `pop_size` hoverable+repaired individuals via parallel sampling.

    For direct encodings (spherical, cartesian) the worker samples from
    `genotype` (a SphericalAngularDroneGenomeHandler instance whose attributes
    seed the per-process handlers).

    For indirect encodings (cppn, hybrid-cppn) it instantiates a fresh handler
    of `handler_class` with `handler_kwargs` per sample, decodes via
    `get_phenotype()`, and retains the CPPN genome only when the decoded
    phenotype survives the 3-stage repair pipeline.

    Returns
    -------
    (population, stats)
        population: numpy array (direct) or list of CPPN/HybridGenome (indirect).
                    None if no individuals could be generated.
        stats: dict with attempt/failure counts and wall-clock time.
    """
    is_indirect = handler_type in ("cppn", "hybrid-cppn")
    if num_workers is None:
        num_workers = cpu_count()

    print(
        f"Generating initial population of size {pop_size} "
        f"using {num_workers} parallel workers..."
    )
    if is_indirect:
        print("Strategy: Random CPPN -> Decode -> Strict hover check -> Fix collisions -> Align thrust")
    else:
        print("Strategy: Parallel sampling -> Strict hover check -> Fix collisions -> Align thrust")
    print("Expected success rate: ~0.2% (need ~{:,} samples for {} individuals)\n".format(
        pop_size * 500, pop_size,
    ))
    sys.stdout.flush()

    start_time = time.time()
    base_seed = np.random.randint(0, 2**31)
    print(f"Base seed for this run: {base_seed}\n")

    batch_size = pop_size * 1000
    max_iterations = 100

    accepted = []
    total_stats = {
        "failed_hover": 0,
        "failed_stage1": 0,
        "failed_stage3": 0,
        "phase1_success": 0,
        "total_attempts": 0,
    }

    batch_idx = 0
    for batch_idx in range(max_iterations):
        if len(accepted) >= pop_size:
            break

        remaining = pop_size - len(accepted)
        print(
            f"\n--- Iteration {batch_idx + 1}: Sampling {batch_size} individuals "
            f"(need {remaining} more) ---"
        )

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
            worker = _try_generate_cppn_individual
        else:
            handler_config = {
                "min_max_narms": (genotype.min_narms, genotype.max_narms),
                "append_arm_chance": genotype.append_arm_chance,
                "parameter_limits": genotype.parameter_limits,
                "bilateral_plane_for_symmetry": genotype.bilateral_plane_for_symmetry,
                "repair": genotype.repair_enabled,
            }
            args_list = [
                (
                    batch_idx * batch_size + i,
                    base_seed,
                    handler_config,
                    genotype.parameter_limits,
                    coordinate_system,
                )
                for i in range(batch_size)
            ]
            worker = _try_generate_individual

        survivors = []
        with Pool(processes=num_workers) as pool:
            with tqdm(total=batch_size, desc=f"batch {batch_idx + 1}", unit="ind") as pbar:
                for result, status in pool.imap_unordered(worker, args_list, chunksize=10):
                    total_stats["total_attempts"] += 1
                    total_stats["failed_hover"] += status["failed_hover"]
                    total_stats["failed_stage1"] += status["failed_stage1"]
                    total_stats["failed_stage3"] += status["failed_stage3"]
                    total_stats["phase1_success"] += status["success"]

                    if result is not None:
                        survivors.append(result)
                        if len(accepted) + len(survivors) >= pop_size:
                            pbar.update(1)
                            break

                    pbar.update(1)
                    pbar.set_postfix({
                        "survivors": len(survivors),
                        "rate": f"{total_stats['phase1_success']/max(1,total_stats['total_attempts'])*100:.2f}%",
                    })

        accepted.extend(survivors)
        print(f"  Survivors this batch: {len(survivors)} (total: {len(accepted)}/{pop_size})")

        if len(accepted) >= pop_size:
            print("  Target reached!")
            break

    end_time = time.time()

    print(f"\n{'='*80}")
    print("Initial Population Generation Results")
    print(f"{'='*80}")
    print(f"Successfully generated: {len(accepted)}/{pop_size}")
    print(f"Total attempts: {total_stats['total_attempts']:,}")
    rate = total_stats["phase1_success"] / max(1, total_stats["total_attempts"]) * 100
    print(f"Success rate: {rate:.3f}%")
    print(f"Time taken: {end_time - start_time:.1f}s")
    print(f"{'='*80}\n")

    if len(accepted) < pop_size:
        print(
            f"Warning: Could only generate {len(accepted)}/{pop_size} individuals.\n"
        )

    stats = {
        "total_attempts": total_stats["total_attempts"],
        "failed_hover": total_stats["failed_hover"],
        "failed_stage1": total_stats["failed_stage1"],
        "failed_stage3": total_stats["failed_stage3"],
        "phase1_success": total_stats["phase1_success"],
        "wall_clock_seconds": end_time - start_time,
        "num_iterations": batch_idx + 1,
        "pop_size_requested": pop_size,
        "pop_size_generated": len(accepted),
        "handler_type": handler_type,
    }

    if len(accepted) == 0:
        return None, stats
    trimmed = accepted[:pop_size]
    if is_indirect:
        return trimmed, stats
    return np.array(trimmed), stats
