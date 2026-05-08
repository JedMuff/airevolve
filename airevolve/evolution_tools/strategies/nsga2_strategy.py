"""NSGA-II elitist multi-objective evolution strategy.

Implements the classic NSGA-II loop (Deb et al., 2002) on top of the existing
airevolve GenomeHandler / evaluate_population infrastructure:

  1. Evaluate initial population μ → fitness tuples (waypoints, energy_j).
  2. Assign Pareto rank and crowding distance to each individual.
  3. Each generation:
       a. Select parents via the crowded-comparison tournament operator.
       b. Create λ offspring through crossover + mutation.
       c. Evaluate offspring.
       d. Combine parents ∪ offspring → size 2μ.
       e. Fast non-dominated sort + crowding distance on the combined pool.
       f. Fill the new population front-by-front; split the last front by
          crowding distance (descending) to fill remaining slots exactly.
"""

import os
import time
from typing import Callable, List, Optional

import numpy as np
import pandas as pd

from airevolve.evolution_tools.genome_handlers.base import GenomeHandler
from airevolve.evolution_tools.strategies.evolution_components import (
    evaluate_population,
)
from airevolve.evolution_tools.selectors.nsga2_utils import (
    fast_non_dominated_sort,
    calculate_crowding_distance,
)
from airevolve.evolution_tools.selectors.tournament import nsga2_tournament_selection


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def evolve_nsga2(
    fitness_function: Callable,
    population_size: int,
    num_generations: int,
    num_mutate: int,
    num_crossover: int,
    mutate_after_crossover: bool,
    initial_population: Optional[List] = None,
    log_dir: str = "./logs",
    genome_handler=GenomeHandler,
    verbose: bool = True,
    num_workers: int = 1,
    snapshot_callback=None,
):
    """Run NSGA-II evolution.

    Parameters
    ----------
    fitness_function   : callable(genome, log_dir) → (waypoints: int, energy_j: float)
    population_size    : μ — number of survivors per generation.
    num_generations    : number of generations to run.
    num_mutate         : offspring created by mutation each generation.
    num_crossover      : offspring created by crossover each generation.
    mutate_after_crossover : whether to mutate crossover offspring.
    initial_population : optional pre-built list of genomes (length == population_size).
    log_dir            : base directory for logs.
    genome_handler     : GenomeHandler class (callable with no args → instance).
    verbose            : print per-generation statistics.
    num_workers        : parallel workers for evaluate_population.
    snapshot_callback  : callable(generation, population_df) or None.
        Invoked at the end of every generation with a copy of the current
        survivor population DataFrame.  Use for checkpointing, artifact
        logging, or Pareto-front snapshots at specific generations.

    Returns
    -------
    all_individuals : pd.DataFrame
        Every individual evaluated across all generations.  Columns include
        'id', 'generation', 'genome', 'log_dir', 'parent_ids', 'in_pop',
        'fitness' (tuple), 'rank', 'crowding_distance'.
    """
    dummy = genome_handler()
    evo_start = time.time()

    # ── Phase 0: initial population ──────────────────────────────────────────
    if initial_population is not None:
        gene_pool = list(initial_population)
    else:
        handlers = dummy.generate_random_population(population_size)
        gene_pool = [h.genome for h in handlers]

    ids        = [str(i).zfill(4) for i in range(population_size)]
    parent_ids = [[None, None]] * population_size

    population = evaluate_population(
        fitness_function, gene_pool, ids, 0, parent_ids,
        log_dir_base=log_dir, num_workers=num_workers,
    )
    population["in_pop"] = True
    population = _assign_rank_and_crowding(population)

    if verbose:
        _log(0, time.time() - evo_start, population)

    all_individuals = population.copy()

    # ── Main loop ─────────────────────────────────────────────────────────────
    for generation in range(1, num_generations + 1):
        gen_start = time.time()

        # ── a. Parent selection ───────────────────────────────────────────────
        n_parents = num_crossover * 2 + num_mutate
        parents_raw = nsga2_tournament_selection(population, tournament_size=2, k=n_parents)

        p1_ids     = [parents_raw["id"][i] for i in range(0, num_crossover * 2, 2)]
        p2_ids     = [parents_raw["id"][i] for i in range(1, num_crossover * 2, 2)]
        mutant_ids = [parents_raw["id"][i] for i in range(num_crossover * 2, n_parents)]

        parent_genomes    = [parents_raw["genome"][i].copy() for i in range(n_parents)]
        crossover_genomes = parent_genomes[: num_crossover * 2]
        mutant_genomes    = parent_genomes[num_crossover * 2 :]

        # ── b. Variation ──────────────────────────────────────────────────────
        new_gene_pool = []

        if num_crossover > 0:
            p1_handlers = [genome_handler(genome=g) for g in crossover_genomes[::2]]
            p2_handlers = [genome_handler(genome=g) for g in crossover_genomes[1::2]]
            x_handlers  = dummy.crossover_population(p1_handlers, p2_handlers)
            if mutate_after_crossover:
                dummy.mutate_population(x_handlers)
            new_gene_pool.extend(h.genome for h in x_handlers)

        if num_mutate > 0:
            mut_handlers = [genome_handler(genome=g) for g in mutant_genomes]
            dummy.mutate_population(mut_handlers)
            new_gene_pool.extend(h.genome for h in mut_handlers)

        # ── c. Evaluate offspring ─────────────────────────────────────────────
        n_prev = len(all_individuals)
        new_ids = [str(n_prev + i).zfill(4) for i in range(len(new_gene_pool))]
        cross_pids  = [[a, b] for a, b in zip(p1_ids, p2_ids)]
        mutant_pids = [[mid, None] for mid in mutant_ids]
        offspring_pids = cross_pids + mutant_pids

        offspring = evaluate_population(
            fitness_function, new_gene_pool, new_ids, generation, offspring_pids,
            log_dir_base=log_dir, num_workers=num_workers,
        )

        # ── d. Combine parents + offspring (size 2μ) ─────────────────────────
        combined = pd.concat([population, offspring], ignore_index=True)
        combined["in_pop"] = False

        # ── e–f. NSGA-II survival: sort + crowding + elitist fill ─────────────
        population = _nsga2_survival(combined, population_size)
        population["in_pop"]     = True
        population["generation"] = generation

        all_individuals = pd.concat([all_individuals, offspring], ignore_index=True)

        if snapshot_callback is not None:
            snapshot_callback(generation, population.copy())

        if verbose:
            _log(generation, time.time() - gen_start, population)

    print(f"NSGA-II finished in {time.time() - evo_start:.1f}s")
    return all_individuals


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _fitness_matrix(df: pd.DataFrame) -> np.ndarray:
    """Extract a (n, 2) float64 array from the 'fitness' column of tuples."""
    return np.array([list(f) for f in df["fitness"].values], dtype=float)


def _assign_rank_and_crowding(df: pd.DataFrame) -> pd.DataFrame:
    """Add / overwrite 'rank' and 'crowding_distance' columns in df."""
    df = df.copy().reset_index(drop=True)
    F = _fitness_matrix(df)
    fronts = fast_non_dominated_sort(F)

    ranks = np.empty(len(df), dtype=int)
    cds   = np.empty(len(df), dtype=float)

    for rank_idx, front in enumerate(fronts):
        front_arr = np.array(front, dtype=int)
        ranks[front_arr] = rank_idx
        distances = calculate_crowding_distance(F[front_arr])
        cds[front_arr] = distances

    df["rank"]              = ranks
    df["crowding_distance"] = cds
    return df


def _nsga2_survival(combined: pd.DataFrame, population_size: int) -> pd.DataFrame:
    """Elitist NSGA-II survival: fill new population front-by-front.

    For the splitting front, individuals are selected by crowding distance
    in descending order (prefer more diverse individuals).
    """
    combined = combined.reset_index(drop=True)
    F = _fitness_matrix(combined)
    fronts = fast_non_dominated_sort(F)

    # Pre-allocate rank/crowding columns
    ranks = np.full(len(combined), -1, dtype=int)
    cds   = np.zeros(len(combined), dtype=float)

    selected_idx = []
    for rank_idx, front in enumerate(fronts):
        front_arr = np.array(front, dtype=int)
        distances = calculate_crowding_distance(F[front_arr])

        ranks[front_arr] = rank_idx
        cds[front_arr]   = distances

        slots_left = population_size - len(selected_idx)
        if len(front_arr) <= slots_left:
            # Entire front fits
            selected_idx.extend(front_arr.tolist())
        else:
            # Partial front — sort by crowding distance descending
            order = np.argsort(distances)[::-1]
            selected_idx.extend(front_arr[order[:slots_left]].tolist())
            break

    combined["rank"]              = ranks
    combined["crowding_distance"] = cds

    new_pop = combined.iloc[selected_idx].copy().reset_index(drop=True)
    return new_pop


def _log(generation: int, elapsed: float, population: pd.DataFrame) -> None:
    waypoints = np.array([f[0] for f in population["fitness"].values])
    energies  = np.array([f[1] for f in population["fitness"].values])
    finite_e  = energies[np.isfinite(energies)]
    front0    = int((population["rank"] == 0).sum())

    min_e_str = f"{np.min(finite_e):.1f}J" if len(finite_e) else "N/A"
    print(
        f"G:{generation:3d} Time:{elapsed:.1f}s | "
        f"Front0={front0:2d} | "
        f"MaxGates={int(np.max(waypoints)):4d} | "
        f"MeanGates={np.mean(waypoints):.1f} | "
        f"MinEnergy={min_e_str}",
        flush=True,
    )
