
from __future__ import annotations
from typing import Callable, List, Tuple, Optional
from typing import Any

import numpy as np
import numpy.typing as npt
import os
import pandas as pd

from airevolve.evolution_tools.genome_handlers.base import GenomeHandler

def generate_population(pop_size: int, genome_handler : object = GenomeHandler) -> list[GenomeHandler]:
    return [genome_handler() for _ in range(pop_size)]

def evaluate_individual(fitness_function : Callable, 
                        genome: np.ndarray, 
                        id: str, 
                        generation: int, 
                        parent_ids: List[str], 
                        log_dir_base: str) -> dict:
    if log_dir_base is not None:
        gen_dir = os.path.join(log_dir_base, f"generation_{generation:02d}")
        indiv_log_dir = os.path.join(gen_dir, f"individual_{id}")
        os.makedirs(indiv_log_dir, exist_ok=True)
    fitness = fitness_function(genome, indiv_log_dir)
    return {
        'id': id,
        'generation': generation,
        'genome': genome,
        'log_dir': indiv_log_dir,
        'parent_ids': parent_ids,
        'in_pop': False,
        'fitness': fitness
    }

def evaluate_population(fitness_function : Callable, 
                        population: np.ndarray, 
                        ids : List[str],
                        generation: int, 
                        all_parent_ids: List[List[str]], 
                        log_dir_base: str) -> list[float]:
    
    evalulated_individuals = []

    for i, genome in enumerate(population):
        individual = evaluate_individual(
            fitness_function,
            genome,
            ids[i],
            generation,
            all_parent_ids[i],
            log_dir_base
        )
        evalulated_individuals.append(individual)
    
    pop = pd.DataFrame(evalulated_individuals)

    return pop