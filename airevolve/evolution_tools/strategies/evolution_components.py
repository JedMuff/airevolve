
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

def _evaluate_individual_worker(args):
    """Worker function for multiprocessing Pool."""
    fitness_function, genome, id, generation, parent_ids, log_dir_base = args
    return evaluate_individual(fitness_function, genome, id, generation, parent_ids, log_dir_base)


def _pool_worker_init():
    """Limit each spawned worker to 1 thread to avoid oversubscription."""
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    import torch
    torch.set_num_threads(1)


def evaluate_population(fitness_function : Callable,
                        population: np.ndarray,
                        ids : List[str],
                        generation: int,
                        all_parent_ids: List[List[str]],
                        log_dir_base: str,
                        num_workers: int = 1) -> list[float]:

    if num_workers > 1:
        import multiprocessing
        ctx = multiprocessing.get_context('spawn')
        args_list = [
            (fitness_function, genome, ids[i], generation, all_parent_ids[i], log_dir_base)
            for i, genome in enumerate(population)
        ]
        with ctx.Pool(processes=min(num_workers, len(population)), initializer=_pool_worker_init) as pool:
            evaluated_individuals = pool.map(_evaluate_individual_worker, args_list)
    else:
        evaluated_individuals = []
        for i, genome in enumerate(population):
            individual = evaluate_individual(
                fitness_function,
                genome,
                ids[i],
                generation,
                all_parent_ids[i],
                log_dir_base
            )
            evaluated_individuals.append(individual)

    pop = pd.DataFrame(evaluated_individuals)

    return pop