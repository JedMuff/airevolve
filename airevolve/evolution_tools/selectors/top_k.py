import pandas as pd

def top_n(population: pd.DataFrame, k: int) -> pd.DataFrame:
    # Sort the population by fitness in descending order
    sorted_population = population.sort_values(by='fitness', ascending=False)
    
    # Select the top n rows
    top_k_population = sorted_population.head(k)
    
    return top_k_population