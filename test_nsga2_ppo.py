import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "examples", "evolution")))

from run_evolution import get_genome_handler_config, create_genome_handler_wrapper
from airevolve.evolution_tools.evaluators.bi_objective_fitness import BiObjectiveFitness
from airevolve.evolution_tools.strategies.nsga2_strategy import evolve_nsga2
from airevolve.evolution_tools.strategies.init_population import generate_viable_initial_population

def main():
    config = get_genome_handler_config("spherical", 6, 6, 8, 0)
    WrappedHandler = create_genome_handler_wrapper(config["handler_class"], config["handler_kwargs"])
    
    initial_pop, _ = generate_viable_initial_population(
        WrappedHandler(),
        population_size=8,
        is_indirect=False
    )
    
    fitness_function = BiObjectiveFitness(
        brain="rl",
        hover_gradient=True,
        per_individual_repair=True,
        is_indirect=False,
        handler_class=config["handler_class"],
        handler_kwargs=config["handler_kwargs"],
        coordinate_system=config["coordinate_system"],
        brain_kwargs={
            "gate_cfg": "figure8",
            "training_ts": 1000000,
            "num_envs": 16,
            "device": "cpu",
            "max_steps": 120,
            "sparse_weight": 0.002,
            "use_power_env": False
        }
    )
    
    all_individuals = evolve_nsga2(
        fitness_function=fitness_function,
        population_size=8,
        num_generations=6,
        num_mutate=4,
        num_crossover=4,
        mutate_after_crossover=True,
        initial_population=initial_pop,
        log_dir="__data__/test_nsga2_ppo",
        genome_handler=WrappedHandler,
        verbose=True,
        num_workers=1
    )
    
    print(all_individuals.to_string())

if __name__ == "__main__":
    main()
