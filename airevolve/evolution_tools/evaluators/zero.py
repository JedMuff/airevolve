
from airevolve.evolution_tools.genome_handlers.base import GenomeHandler


def fitness_function(member: GenomeHandler, log_dir=None) -> float:
    return 0