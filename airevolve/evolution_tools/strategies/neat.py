"""NEAT evolution strategy with speciation-based diversity protection."""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from airevolve.evolution_tools.genome_handlers.base import GenomeHandler
from airevolve.evolution_tools.strategies.evolution_components import evaluate_population
from airevolve.evolution_tools.strategies.speciation import SpeciationState


def evolve_neat(
    fitness_function: Callable,
    population_size: int,
    num_generations: int,
    crossover_rate: float,
    parent_selection: Callable,
    genome_handler: type = GenomeHandler,
    # NEAT-specific parameters
    compatibility_threshold: float = 3.0,
    species_elitism: int = 1,
    stagnation_limit: int = 15,
    min_species_size: int = 2,
    adjust_threshold: bool = True,
    target_species_count: int = 5,
    interspecies_mating_rate: float = 0.001,
    mutate_after_crossover: bool = True,
    # Standard parameters
    initial_population: Optional[List] = None,
    log_dir: str = "./logs",
    verbose: bool = True,
    num_workers: int = 1,
) -> pd.DataFrame:
    """NEAT evolution with speciation.

    Parameters
    ----------
    fitness_function : Callable
        ``(genome, log_dir) -> float``.
    population_size : int
        Target population size.
    num_generations : int
        Number of generations to evolve.
    crossover_rate : float
        Fraction of offspring produced by crossover (vs mutation-only).
    parent_selection : Callable
        Selection function ``(species_df, k=...) -> selected_df``.
    genome_handler : type
        GenomeHandler subclass.
    compatibility_threshold : float
        Initial compatibility distance threshold for speciation.
    species_elitism : int
        Top-N individuals per species copied unchanged.
    stagnation_limit : int
        Generations without improvement before species removal.
    min_species_size : int
        Minimum offspring allocated per surviving species.
    adjust_threshold : bool
        Whether to dynamically adjust compatibility threshold.
    target_species_count : int
        Target number of species (for dynamic threshold).
    interspecies_mating_rate : float
        Probability of picking a crossover partner from another species.
    mutate_after_crossover : bool
        Whether to mutate offspring produced by crossover.
    initial_population : list, optional
        Pre-existing genomes to seed the population.
    log_dir : str
        Base directory for logging.
    verbose : bool
        Print per-generation stats.
    num_workers : int
        Number of parallel evaluation workers.

    Returns
    -------
    pd.DataFrame
        All individuals across all generations with columns: id, generation,
        genome, log_dir, parent_ids, in_pop, fitness, species_id.
    """
    dummy = genome_handler()
    # Extract handler kwargs for constructing handlers from raw genomes
    handler_kwargs = _extract_handler_kwargs(dummy)

    evo_start = time.time()

    # ------------------------------------------------------------------ #
    # Initialisation
    # ------------------------------------------------------------------ #
    if initial_population is not None:
        gene_pool = list(initial_population)
    else:
        pop_handlers = dummy.generate_random_population(population_size)
        gene_pool = [h.genome for h in pop_handlers]

    ids = [str(i).zfill(4) for i in range(population_size)]
    parent_ids_list = [[None, None] for _ in range(population_size)]

    population = evaluate_population(
        fitness_function, gene_pool, ids, 0, parent_ids_list,
        log_dir_base=log_dir, num_workers=num_workers,
    )
    population["in_pop"] = True

    # Setup speciation
    spec_state = SpeciationState(compatibility_threshold=compatibility_threshold)
    assignments = spec_state.speciate(population, genome_handler, handler_kwargs)
    population["species_id"] = population["id"].map(assignments)

    spec_state.update_best_fitness(population)

    all_individuals = population.copy()
    next_id = population_size

    if verbose:
        _print_gen_stats(0, population, spec_state, time.time() - evo_start)

    # ------------------------------------------------------------------ #
    # Generation loop
    # ------------------------------------------------------------------ #
    for generation in range(1, num_generations + 1):
        gen_start = time.time()

        # Reset innovation counter if available
        if hasattr(genome_handler, "_innovation_counter"):
            genome_handler._innovation_counter.reset_generation()

        # --- Stagnation check ---
        spec_state.remove_stagnant_species(stagnation_limit)

        # If all species were removed, re-speciate from scratch
        if not spec_state.species:
            spec_state = SpeciationState(
                compatibility_threshold=spec_state.compatibility_threshold,
            )
            assignments = spec_state.speciate(population, genome_handler, handler_kwargs)
            population["species_id"] = population["id"].map(assignments)
            spec_state.update_best_fitness(population)

        # --- Fitness sharing & offspring allocation ---
        species_offspring = _allocate_offspring(
            population, spec_state, population_size, min_species_size,
        )

        # --- Reproduction ---
        new_genomes: List = []
        new_parent_ids: List = []
        elite_genomes: List = []
        elite_parent_ids: List = []

        for sid, n_offspring in species_offspring.items():
            sp = spec_state.species.get(sid)
            if sp is None or not sp.member_ids:
                continue

            # Species sub-population
            sp_df = population[population["id"].isin(sp.member_ids)].copy()
            if sp_df.empty:
                continue

            sp_df = sp_df.sort_values("fitness", ascending=False).reset_index(drop=True)

            # Elitism: copy champion of each species unchanged
            n_elite = min(species_elitism, len(sp_df), n_offspring)
            for i in range(n_elite):
                elite_genomes.append(sp_df.iloc[i]["genome"])
                elite_parent_ids.append([sp_df.iloc[i]["id"], None])

            remaining = n_offspring - n_elite
            if remaining <= 0:
                continue

            # Produce remaining offspring
            for _ in range(remaining):
                if np.random.random() < crossover_rate and len(sp_df) >= 2:
                    # Crossover
                    parents = parent_selection(sp_df, k=2)
                    p1_genome = parents.iloc[0]["genome"]
                    p2_genome = parents.iloc[1]["genome"]
                    p1_id = parents.iloc[0]["id"]
                    p1_fitness = parents.iloc[0]["fitness"]
                    p2_id = parents.iloc[1]["id"]
                    p2_fitness = parents.iloc[1]["fitness"]

                    # Interspecies mating
                    if np.random.random() < interspecies_mating_rate:
                        other_sids = [
                            s for s in spec_state.species if s != sid
                        ]
                        if other_sids:
                            other_sid = np.random.choice(other_sids)
                            other_sp = spec_state.species[other_sid]
                            other_df = population[
                                population["id"].isin(other_sp.member_ids)
                            ]
                            if not other_df.empty:
                                mate = parent_selection(other_df, k=1)
                                p2_genome = mate.iloc[0]["genome"]
                                p2_id = mate.iloc[0]["id"]
                                p2_fitness = mate.iloc[0]["fitness"]

                    p1_handler = genome_handler(genome=p1_genome, **handler_kwargs)
                    p2_handler = genome_handler(genome=p2_genome, **handler_kwargs)

                    # Set fitness for crossover gene inheritance
                    p1_handler.fitness = p1_fitness
                    p2_handler.fitness = p2_fitness

                    child_handler = p1_handler.crossover(p2_handler)
                    if mutate_after_crossover:
                        child_handler.mutate()

                    new_genomes.append(child_handler.genome)
                    new_parent_ids.append([p1_id, p2_id])
                else:
                    # Mutation only
                    parent = parent_selection(sp_df, k=1)
                    p_genome = parent.iloc[0]["genome"]
                    p_id = parent.iloc[0]["id"]

                    handler = genome_handler(genome=p_genome, **handler_kwargs)
                    handler.mutate()

                    new_genomes.append(handler.genome)
                    new_parent_ids.append([p_id, None])

        # --- Evaluate new offspring ---
        all_new_genomes = elite_genomes + new_genomes
        all_new_parent_ids = elite_parent_ids + new_parent_ids

        if not all_new_genomes:
            # Fallback: generate random population if reproduction failed
            pop_handlers = dummy.generate_random_population(population_size)
            all_new_genomes = [h.genome for h in pop_handlers]
            all_new_parent_ids = [[None, None]] * population_size

        new_ids = [
            str(i).zfill(4) for i in range(next_id, next_id + len(all_new_genomes))
        ]
        next_id += len(all_new_genomes)

        offspring_df = evaluate_population(
            fitness_function, all_new_genomes, new_ids, generation,
            all_new_parent_ids, log_dir_base=log_dir, num_workers=num_workers,
        )
        offspring_df["in_pop"] = True
        offspring_df["generation"] = generation

        # --- Re-speciate on new population ---
        population = offspring_df.nlargest(population_size, "fitness").copy()
        assignments = spec_state.speciate(population, genome_handler, handler_kwargs)
        population["species_id"] = population["id"].map(assignments)

        # --- Update speciation state ---
        spec_state.update_representatives(population)
        spec_state.update_best_fitness(population)

        if adjust_threshold:
            spec_state.adjust_threshold(target_species_count)

        # --- Logging ---
        offspring_df["species_id"] = offspring_df["id"].map(assignments).fillna(-1).astype(int)
        all_individuals = pd.concat([all_individuals, offspring_df], ignore_index=True)

        if verbose:
            _print_gen_stats(generation, population, spec_state, time.time() - gen_start)

    if verbose:
        print(f"Time taken to evolve: {time.time() - evo_start:.2f}s")

    return all_individuals


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #


def _extract_handler_kwargs(handler: GenomeHandler) -> dict:
    """Extract constructor kwargs from a handler instance for re-instantiation.

    This inspects the handler for common configuration attributes shared
    across genome handler types.
    """
    kwargs: dict = {}

    # Arm limits
    if hasattr(handler, "min_narms") and hasattr(handler, "max_narms"):
        kwargs["min_max_narms"] = (handler.min_narms, handler.max_narms)
    if hasattr(handler, "parameter_limits"):
        kwargs["parameter_limits"] = handler.parameter_limits

    # CPPN handler specifics
    if hasattr(handler, "num_segments"):
        kwargs["num_segments"] = handler.num_segments
    if hasattr(handler, "initial_hidden_nodes"):
        kwargs["initial_hidden_nodes"] = handler.initial_hidden_nodes

    # Mutation probabilities (CPPN)
    for attr in [
        "prob_add_node", "prob_add_connection", "prob_remove_node",
        "prob_remove_connection", "prob_mutate_weights", "prob_mutate_activation",
        "prob_toggle_connection", "weight_perturb_std", "weight_replace_prob",
        "weight_range", "bias_perturb_std", "bias_replace_prob", "bias_range",
    ]:
        if hasattr(handler, attr):
            kwargs[attr] = getattr(handler, attr)

    # Hybrid handler specifics
    if hasattr(handler, "prob_mutate_direct"):
        kwargs["prob_mutate_direct"] = handler.prob_mutate_direct
    if hasattr(handler, "direct_mutation_scale_pct"):
        kwargs["direct_mutation_scale_pct"] = handler.direct_mutation_scale_pct

    # Repair settings
    if hasattr(handler, "repair_enabled"):
        kwargs["repair"] = handler.repair_enabled
    if hasattr(handler, "enable_collision_repair"):
        kwargs["enable_collision_repair"] = handler.enable_collision_repair
    if hasattr(handler, "propeller_radius"):
        kwargs["propeller_radius"] = handler.propeller_radius
    if hasattr(handler, "inner_boundary_radius"):
        kwargs["inner_boundary_radius"] = handler.inner_boundary_radius
    if hasattr(handler, "outer_boundary_radius"):
        kwargs["outer_boundary_radius"] = handler.outer_boundary_radius
    if hasattr(handler, "max_repair_iterations"):
        kwargs["max_repair_iterations"] = handler.max_repair_iterations
    if hasattr(handler, "repair_step_size"):
        kwargs["repair_step_size"] = handler.repair_step_size
    if hasattr(handler, "propeller_tolerance"):
        kwargs["propeller_tolerance"] = handler.propeller_tolerance

    # Spherical handler specifics
    if hasattr(handler, "append_arm_chance"):
        kwargs["append_arm_chance"] = handler.append_arm_chance
    if hasattr(handler, "bilateral_plane_for_symmetry"):
        kwargs["bilateral_plane_for_symmetry"] = handler.bilateral_plane_for_symmetry

    # RNG
    if hasattr(handler, "rng"):
        kwargs["rng"] = handler.rng
    elif hasattr(handler, "rnd"):
        kwargs["rnd"] = handler.rnd

    return kwargs


def _allocate_offspring(
    population: pd.DataFrame,
    spec_state: SpeciationState,
    population_size: int,
    min_species_size: int,
) -> Dict[int, int]:
    """Allocate offspring using largest-remainder method.

    Guarantees the allocation sums to exactly *population_size* by
    construction.  Each surviving species receives at least
    *min_species_size* offspring; the remainder is distributed
    proportionally to adjusted fitness using the largest-remainder
    (Hamilton) method.
    """
    species_adj_fitness: Dict[int, float] = {}

    for sid, sp in spec_state.species.items():
        if not sp.member_ids:
            continue
        sp_df = population[population["id"].isin(sp.member_ids)]
        if sp_df.empty:
            continue
        sp_size = len(sp_df)
        adj_sum = float(sp_df["fitness"].sum() / sp_size)
        species_adj_fitness[sid] = adj_sum

    if not species_adj_fitness:
        return {}

    # Drop lowest-fitness species until min_species_size floor is feasible
    sorted_sids = sorted(
        species_adj_fitness, key=species_adj_fitness.get, reverse=True
    )
    while len(sorted_sids) * min_species_size > population_size and len(sorted_sids) > 1:
        dropped = sorted_sids.pop()
        del species_adj_fitness[dropped]

    # Shift fitnesses so all values are positive (handles negative fitness)
    min_adj = min(species_adj_fitness.values())
    if min_adj <= 0:
        shift = abs(min_adj) + 1e-6
        species_adj_fitness = {
            sid: adj + shift for sid, adj in species_adj_fitness.items()
        }

    total_adj = sum(species_adj_fitness.values())

    # Give each species its floor, then distribute remainder proportionally
    n_species = len(species_adj_fitness)
    remainder = population_size - n_species * min_species_size

    allocation: Dict[int, int] = {}
    fractional_parts: Dict[int, float] = {}

    for sid, adj in species_adj_fitness.items():
        # Exact (real-valued) share of the remainder
        exact_share = (adj / total_adj) * remainder if total_adj > 0 else 0.0
        floor_share = int(exact_share)
        allocation[sid] = min_species_size + floor_share
        fractional_parts[sid] = exact_share - floor_share

    # Distribute leftover seats to species with largest fractional parts
    leftover = population_size - sum(allocation.values())
    if leftover > 0:
        ranked = sorted(fractional_parts, key=fractional_parts.get, reverse=True)
        for i in range(leftover):
            allocation[ranked[i % len(ranked)]] += 1

    return allocation


def _print_gen_stats(
    generation: int,
    population: pd.DataFrame,
    spec_state: SpeciationState,
    elapsed: float,
) -> None:
    """Print per-generation statistics."""
    fitnesses = population["fitness"].values
    n_species = len(spec_state.species)
    print(
        f"G:{generation} Time:{elapsed:.2f}s "
        f"MaxF={np.max(fitnesses):.4f} AvgF={np.mean(fitnesses):.4f} "
        f"Species={n_species}",
        flush=True,
    )
    for sid, sp in sorted(spec_state.species.items()):
        sp_df = population[population["id"].isin(sp.member_ids)]
        if sp_df.empty:
            continue
        sp_fit = sp_df["fitness"].values
        print(
            f"  S{sid}: size={len(sp_df)} "
            f"best={np.max(sp_fit):.4f} avg={np.mean(sp_fit):.4f} "
            f"stag={sp.generations_since_improvement}",
            flush=True,
        )
