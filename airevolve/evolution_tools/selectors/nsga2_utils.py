"""NSGA-II utility functions: fast non-dominated sort and crowding distance.

Adapted from the math in the pymoo reference files (examples/evolution/nsga-ii/).
No pymoo imports — pure numpy.

Objective directions for this project:
  Obj 0: waypoints  (MAXIMIZE) — internally negated to convert to all-minimize
  Obj 1: total_energy_j (MINIMIZE)

Both functions accept fitnesses in the ORIGINAL direction (waypoints, energy).
"""

import numpy as np


def fast_non_dominated_sort(population_fitnesses):
    """Fast non-dominated sort for the bi-objective (waypoints, energy) problem.

    Internally converts to all-minimize by negating waypoints so that the
    standard domination rule applies uniformly.

    Parameters
    ----------
    population_fitnesses : array-like, shape (n, 2)
        Each row is (waypoints, total_energy_j) in the original direction.

    Returns
    -------
    fronts : list of list of int
        fronts[0] is the Pareto-optimal front (best individuals).
        Each sub-list contains indices into population_fitnesses.
    """
    F = np.asarray(population_fitnesses, dtype=float)
    n = len(F)
    if n == 0:
        return []

    # Convert to all-minimize: negate waypoints (obj 0 is maximised)
    F_min = np.empty_like(F)
    F_min[:, 0] = -F[:, 0]
    F_min[:, 1] =  F[:, 1]

    # For each individual: list of individuals it dominates
    is_dominating = [[] for _ in range(n)]
    # Number of individuals that dominate this one
    n_dominated = np.zeros(n, dtype=int)

    for i in range(n):
        for j in range(i + 1, n):
            if _dominates(F_min[i], F_min[j]):
                is_dominating[i].append(j)
                n_dominated[j] += 1
            elif _dominates(F_min[j], F_min[i]):
                is_dominating[j].append(i)
                n_dominated[i] += 1

    current_front = [i for i in range(n) if n_dominated[i] == 0]
    fronts = [current_front[:]]
    n_ranked = len(current_front)

    while n_ranked < n:
        next_front = []
        for i in current_front:
            for j in is_dominating[i]:
                n_dominated[j] -= 1
                if n_dominated[j] == 0:
                    next_front.append(j)
                    n_ranked += 1
        fronts.append(next_front[:])
        current_front = next_front

    return fronts


def calculate_crowding_distance(front_fitnesses):
    """Standard NSGA-II crowding distance for one Pareto front.

    Boundary individuals (extreme in any objective) receive infinite distance.
    Interior individuals receive the sum of normalised gaps to their neighbours
    across all objectives.

    Direction does not affect the crowding distance metric — it is purely a
    geometric diversity measure in objective space.

    Parameters
    ----------
    front_fitnesses : array-like, shape (m, 2)
        Rows are (waypoints, total_energy_j).  May be in any objective direction.

    Returns
    -------
    distances : np.ndarray, shape (m,)
        Crowding distance per individual.  Boundary individuals have np.inf.
    """
    F = np.asarray(front_fitnesses, dtype=float)
    n_points, n_obj = F.shape

    if n_points <= 2:
        return np.full(n_points, np.inf)

    # Per-objective sort indices
    I = np.argsort(F, axis=0, kind='mergesort')          # shape (n_points, n_obj)
    F_sorted = F[I, np.arange(n_obj)]                     # sorted values

    # Differences to left and right neighbours in sorted order
    # dist[k, obj] = F_sorted[k, obj] - F_sorted[k-1, obj]  (with ±inf boundaries)
    inf_row = np.full((1, n_obj), np.inf)
    dist = (np.vstack([F_sorted, inf_row])
            - np.vstack([np.full((1, n_obj), -np.inf), F_sorted]))
    # dist shape: (n_points + 1, n_obj)
    #   row 0   : inf  (left boundary)
    #   rows 1..n-1: forward differences
    #   row n   : inf  (right boundary)

    # Objective range for normalisation; zero range → contribution = 0
    norm = np.max(F_sorted, axis=0) - np.min(F_sorted, axis=0)
    norm[norm == 0] = np.nan

    dist_to_last = dist[:-1] / norm   # shape (n_points, n_obj)
    dist_to_next = dist[1:]  / norm   # shape (n_points, n_obj)

    dist_to_last[np.isnan(dist_to_last)] = 0.0
    dist_to_next[np.isnan(dist_to_next)] = 0.0

    # Inverse permutation: J[i, obj] = sorted rank of original point i for obj
    J = np.argsort(I, axis=0)

    # Sum contributions across objectives, then average
    cd = np.sum(
        dist_to_last[J, np.arange(n_obj)] + dist_to_next[J, np.arange(n_obj)],
        axis=1,
    ) / n_obj

    return cd


# ──────────────────────────────────────────────────────────────────────────────
# Internal helper
# ──────────────────────────────────────────────────────────────────────────────

def _dominates(a, b):
    """True if a dominates b in all-minimize space."""
    return bool(np.all(a <= b) and np.any(a < b))
