import pandas as pd

def nsga2_tournament_selection(population: pd.DataFrame, tournament_size=2, k=1) -> pd.DataFrame:
    """Binary tournament selection using the NSGA-II crowded-comparison operator.

    An individual i wins over j if:
      1. i has a lower front rank (i.rank < j.rank), OR
      2. same rank AND higher crowding distance (more diverse).

    Requires 'rank' and 'crowding_distance' columns in population.

    Parameters
    ----------
    population     : DataFrame with 'rank' and 'crowding_distance' columns.
    tournament_size: Number of individuals per tournament (default 2).
    k              : Number of individuals to select.
    """
    assert len(population) > 0, "Population must not be empty"
    assert "rank" in population.columns and "crowding_distance" in population.columns, (
        "nsga2_tournament_selection requires 'rank' and 'crowding_distance' columns"
    )

    selected = []
    for _ in range(k):
        tournament = population.sample(n=tournament_size, replace=True).reset_index(drop=True)
        winner_idx = 0
        for j in range(1, tournament_size):
            a_rank = tournament.at[winner_idx, "rank"]
            b_rank = tournament.at[j,          "rank"]
            a_cd   = tournament.at[winner_idx, "crowding_distance"]
            b_cd   = tournament.at[j,          "crowding_distance"]

            if b_rank < a_rank:
                winner_idx = j
            elif b_rank == a_rank and b_cd > a_cd:
                winner_idx = j

        selected.append(tournament.iloc[winner_idx].to_dict())

    return pd.DataFrame(selected).reset_index(drop=True)


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