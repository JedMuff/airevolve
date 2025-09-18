import pandas as pd

def tournament_selection(population: pd.DataFrame, tournament_size=3, k=1) -> pd.DataFrame:
    """
    Perform tournament selection on a population DataFrame.

    Args:
    - population (pd.DataFrame): DataFrame containing the population with a 'fitness' column.
    - tournament_size (int): Number of individuals to participate in each tournament. Default is 3.
    - k (int): Number of individuals to select. Default is 1.

    Returns:
    - selected_population (pd.DataFrame): DataFrame of selected individuals.
    """
    # Ensure the population is not empty and k is not greater than the population size
    assert len(population) > 0, "Population must not be empty"
    # assert k <= len(population), "k must not be greater than the population size"

    selected_individuals = []

    for _ in range(k):
        # Randomly select individuals to participate in the tournament
        tournament_individuals = population.sample(n=tournament_size, replace=True).reset_index(drop=True)

        selected_individual = tournament_individuals.loc[tournament_individuals['fitness'].idxmax()]

        selected_individuals.append(selected_individual.to_dict())

    # Create a DataFrame from the selected individuals
    selected_population = pd.DataFrame(selected_individuals).reset_index(drop=True)

    return selected_population