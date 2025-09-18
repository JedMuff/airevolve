import numpy as np
import os, time
import pandas as pd
from typing import Callable, List, Tuple, Optional

from airevolve.evolution_tools.strategies.evolution_components import (
    evaluate_population,
    generate_population,
)
from airevolve.evolution_tools.genome_handlers.base import GenomeHandler

def evolve(
    fitness_function: Callable[[np.ndarray, str], float],
    population_size: int,
    num_generations: int,
    num_mutate: int,
    num_crossover: int,
    mutate_after_crossover: bool,
    strategy_type: str,  # 'plus' or 'comma'
    parent_selection: Callable[[List[dict]], List[int]],
    initial_population: Optional[List[np.ndarray]] = None,
    reevaluate_old: bool = False,
    log_dir: str = "./logs",
    genome_handler: Optional[object] = GenomeHandler,
    verbose: bool = True,
):
    assert strategy_type in ['plus', 'comma'], "Strategy type must be 'plus' or 'comma'"
    dummy_genome = genome_handler()

    evo_start = time.time()

    if initial_population is not None:
        gene_pool = initial_population.copy()
    else:
        # Generate population using GenomeHandler interface and extract genomes
        population_handlers = dummy_genome.generate_random_population(population_size)
        gene_pool = [handler.genome for handler in population_handlers]
    
    ids = [str(i).zfill(4) for i in range(population_size)] # 4 digit string
    parent_ids = [[None, None] for _ in range(population_size)] # 4 digit string
    population = evaluate_population(fitness_function, gene_pool, ids, 0, parent_ids, log_dir_base=log_dir) 
    fitnesses = population['fitness'].values
    population['in_pop'] = True
    print(f"G:{0} Time:{np.round(time.time() - evo_start,2)}, MaxF={np.max(fitnesses)}, AvrF={np.mean(fitnesses)}", flush=True)

    all_individuals = population.copy()
    for generation in range(1, num_generations + 1):
        gen_start = time.time()
        # Select parents

        parent_ids_raw = parent_selection(population, k=num_crossover*2+num_mutate)
        # get genome of parents with id from a pandas dataframe. Id and genome are two columns
        
        p1_ids = [parent_ids_raw['id'][i] for i in range(0, num_crossover*2, 2)]
        p2_ids = [parent_ids_raw['id'][i] for i in range(1, num_crossover*2, 2)]
        mutant_ids = [parent_ids_raw['id'][i] for i in range(num_crossover*2, num_crossover*2+num_mutate)]

        parent_genomes = [parent_ids_raw['genome'][i].copy() for i in range(num_crossover*2+num_mutate)]
        pop_to_cross = parent_genomes[:num_crossover*2]
        pop_to_mutate = parent_genomes[num_crossover*2:]
        
        # Crossover, Split parents into two groups, odd and even up to num_crossover
        parent1s = [pop_to_cross[i] for i in range(0, num_crossover*2, 2)]
        parent2s = [pop_to_cross[i] for i in range(1, num_crossover*2, 2)]

        # Create GenomeHandler objects from genomes for crossover
        parent1_handlers = [genome_handler(genome=genome) for genome in parent1s]
        parent2_handlers = [genome_handler(genome=genome) for genome in parent2s]
        
        # Perform crossover using GenomeHandler interface
        x_gene_pool_handlers = dummy_genome.crossover_population(parent1_handlers, parent2_handlers)
        
        if mutate_after_crossover:
            dummy_genome.mutate_population(x_gene_pool_handlers)
        
        # Extract genomes from handlers
        x_gene_pool = [handler.genome for handler in x_gene_pool_handlers]
        
        # Create GenomeHandler objects for mutation
        pop_to_mutate_handlers = [genome_handler(genome=genome) for genome in pop_to_mutate]
        
        # Mutate using GenomeHandler interface
        dummy_genome.mutate_population(pop_to_mutate_handlers)

        # Extract genomes from mutated handlers
        pop_to_mutate = [handler.genome for handler in pop_to_mutate_handlers]

        new_gene_pool = x_gene_pool + pop_to_mutate
        
        # Evaluate offspring
        new_ids = [str(i).zfill(4) for i in range(len(all_individuals), len(all_individuals)+num_mutate+num_crossover)]
        cross_parent_ids = [[i, j] for i, j in zip(p1_ids, p2_ids)]
        mutant_parent_ids = [[i, None] for i in mutant_ids]
        parent_ids = cross_parent_ids + mutant_parent_ids

        offspring = evaluate_population(fitness_function, new_gene_pool, new_ids, generation, parent_ids, log_dir_base=log_dir) 

        # Possibly reevaluate old individuals
        if reevaluate_old and strategy_type == 'plus': # only for plus strategy
            old_genomes = [population['genome'][i] for i in range(population_size)]
            old_ids = [population['id'][i] for i in range(population_size)]
            old_parent_ids = [population['parent_ids'][i] for i in range(population_size)]
            new_old_pop = evaluate_population(fitness_function, old_genomes, old_ids, generation, old_parent_ids, log_dir_base=log_dir)
        else:
            new_old_pop = population.copy()
        # Select next generation
        if strategy_type == 'plus':
            combined = pd.concat([new_old_pop, offspring], ignore_index=True)
        else:  # comma strategy
            combined = offspring
        combined['in_pop'] = False
        combined['generation'] = generation
        
        sorted_df = combined.sort_values(by="fitness", ascending=False)
        sorted_df.loc[:population_size, 'in_pop'] = True
        population = sorted_df.head(n=population_size)

        all_individuals = pd.concat([all_individuals, sorted_df], ignore_index=True)

        if verbose:
            fitnesses = population['fitness'].values
            print(f"G:{generation} Time:{np.round(time.time() - gen_start,2)}, MaxF={np.max(fitnesses)}, AvrF={np.mean(fitnesses)}", flush=True)

    print(f"Time taken to evolve: {time.time() - evo_start}")
    return all_individuals


def evolve_vectorized(
    fitness_function: Callable[[np.ndarray, str], float],
    population_size: int,
    num_generations: int,
    num_mutate: int,
    num_crossover: int,
    mutate_after_crossover: bool,
    strategy_type: str,  # 'plus' or 'comma'
    parent_selection: Callable[[List[dict]], List[int]],
    initial_population: Optional[np.ndarray] = None,
    reevaluate_old: bool = False,
    log_dir: str = "./logs",
    genome_handler: Optional[object] = GenomeHandler,
    verbose: bool = True,
):
    """
    Vectorized version of the mu+lambda and mu,lambda evolution strategies.
    
    Uses vectorized crossover and mutation operations for improved performance
    while maintaining the same evolutionary methodology as the standard evolve function.
    
    Args:
        fitness_function: Function to evaluate individual fitness (genome, log_dir) -> float
        population_size: Size of the population (μ)
        num_generations: Number of generations to evolve
        num_mutate: Number of individuals to create via mutation (part of λ)
        num_crossover: Number of individuals to create via crossover (part of λ)
        mutate_after_crossover: Whether to mutate offspring after crossover
        strategy_type: Either 'plus' (μ+λ) or 'comma' (μ,λ) strategy
        parent_selection: Function to select parents from population
        initial_population: Optional initial population as numpy array
        reevaluate_old: Whether to reevaluate parents each generation
        log_dir: Directory for logging evolution data
        genome_handler: Genome handler class with vectorized methods
        verbose: Whether to print generation statistics
        
    Returns:
        DataFrame containing all individuals from all generations
    """
    assert strategy_type in ['plus', 'comma'], "Strategy type must be 'plus' or 'comma'"
    
    # Check that genome handler has vectorized methods
    dummy_genome = genome_handler()
    required_methods = ['random_population', 'crossover_vectorized', 'mutation_vectorized']
    for method in required_methods:
        if not hasattr(dummy_genome, method):
            raise AttributeError(f"Genome handler must have {method} method for vectorized evolution")
    
    evo_start = time.time()

    # Initialize population
    if initial_population is not None:
        gene_pool_array = initial_population.copy()
    else:
        # Generate random population as numpy array
        gene_pool_array = dummy_genome.random_population(population_size)
    
    ids = [str(i).zfill(4) for i in range(population_size)]
    parent_ids = [[None, None] for _ in range(population_size)]
    population = evaluate_population(fitness_function, gene_pool_array, ids, 0, parent_ids, log_dir_base=log_dir)
    fitnesses = population['fitness'].values
    population['in_pop'] = True
    print(f"G:{0} Time:{np.round(time.time() - evo_start,2)}, MaxF={np.max(fitnesses)}, AvrF={np.mean(fitnesses)}", flush=True)

    all_individuals = population.copy()
    
    for generation in range(1, num_generations + 1):
        gen_start = time.time()
        
        # Select parents
        parent_ids_raw = parent_selection(population, k=num_crossover*2+num_mutate)
        
        p1_ids = [parent_ids_raw['id'][i] for i in range(0, num_crossover*2, 2)]
        p2_ids = [parent_ids_raw['id'][i] for i in range(1, num_crossover*2, 2)]
        mutant_ids = [parent_ids_raw['id'][i] for i in range(num_crossover*2, num_crossover*2+num_mutate)]

        # Get parent genomes and convert to numpy arrays for vectorized operations
        parent_genomes = [parent_ids_raw['genome'][i].copy() for i in range(num_crossover*2+num_mutate)]
        
        # Split into crossover and mutation populations
        crossover_genomes = parent_genomes[:num_crossover*2]
        mutation_genomes = parent_genomes[num_crossover*2:]
        
        new_gene_pool = []
        
        # Vectorized crossover
        if num_crossover > 0:
            # Prepare parent arrays for vectorized crossover
            parent1_genomes = [crossover_genomes[i] for i in range(0, num_crossover*2, 2)]
            parent2_genomes = [crossover_genomes[i] for i in range(1, num_crossover*2, 2)]
            
            # Convert to numpy arrays for vectorized operations
            parent1_array = np.array(parent1_genomes)
            parent2_array = np.array(parent2_genomes)
            
            # Perform vectorized crossover
            crossover_offspring = dummy_genome.crossover_vectorized(
                gene_pool_array, parent1_array, parent2_array
            )
            
            if mutate_after_crossover:
                crossover_offspring = dummy_genome.mutation_vectorized(crossover_offspring)
            
            # Convert back to list
            new_gene_pool.extend([genome for genome in crossover_offspring])
        
        # Vectorized mutation
        if num_mutate > 0:
            mutation_array = np.array(mutation_genomes)
            mutated_offspring = dummy_genome.mutation_vectorized(mutation_array)
            
            # Convert back to list
            new_gene_pool.extend([genome for genome in mutated_offspring])
        
        # Evaluate offspring
        new_ids = [str(i).zfill(4) for i in range(len(all_individuals), len(all_individuals)+num_mutate+num_crossover)]
        cross_parent_ids = [[i, j] for i, j in zip(p1_ids, p2_ids)]
        mutant_parent_ids = [[i, None] for i in mutant_ids]
        parent_ids = cross_parent_ids + mutant_parent_ids

        offspring = evaluate_population(fitness_function, new_gene_pool, new_ids, generation, parent_ids, log_dir_base=log_dir)

        # Possibly reevaluate old individuals
        if reevaluate_old and strategy_type == 'plus':
            old_genomes = [population['genome'][i] for i in range(population_size)]
            old_ids = [population['id'][i] for i in range(population_size)]
            old_parent_ids = [population['parent_ids'][i] for i in range(population_size)]
            new_old_pop = evaluate_population(fitness_function, old_genomes, old_ids, generation, old_parent_ids, log_dir_base=log_dir)
        else:
            new_old_pop = population.copy()
            
        # Select next generation
        if strategy_type == 'plus':
            combined = pd.concat([new_old_pop, offspring], ignore_index=True)
        else:  # comma strategy
            combined = offspring
        combined['in_pop'] = False
        combined['generation'] = generation
        
        sorted_df = combined.sort_values(by="fitness", ascending=False)
        sorted_df.loc[:population_size, 'in_pop'] = True
        population = sorted_df.head(n=population_size)

        # Update gene_pool_array with new population for next iteration
        gene_pool_array = np.array([population['genome'].iloc[i] for i in range(population_size)])

        all_individuals = pd.concat([all_individuals, sorted_df], ignore_index=True)

        if verbose:
            fitnesses = population['fitness'].values
            print(f"G:{generation} Time:{np.round(time.time() - gen_start,2)}, MaxF={np.max(fitnesses)}, AvrF={np.mean(fitnesses)}", flush=True)

    print(f"Time taken to evolve: {time.time() - evo_start}")
    return all_individuals
