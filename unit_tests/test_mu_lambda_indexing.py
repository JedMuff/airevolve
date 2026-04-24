"""
Tests for mu_lambda indexing fixes.

Verifies that:
1. Reevaluation retrieves the correct genomes (positional, not label-based)
2. in_pop marks exactly population_size individuals (the top ones by fitness)
3. Reevaluated individuals get new log_dirs under the current generation
"""

import sys
import os
import tempfile
import shutil
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from airevolve.evolution_tools.strategies.mu_lambda import evolve
from airevolve.evolution_tools.strategies.evolution_components import evaluate_population


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class DummyGenomeHandler:
    """Minimal genome handler for testing."""
    def __init__(self, genome=None):
        self.genome = genome if genome is not None else np.random.rand(6)
        self.min_narms = 4
        self.max_narms = 4

    def generate_random_population(self, n):
        return [DummyGenomeHandler(genome=np.random.rand(6)) for _ in range(n)]

    def crossover_population(self, p1s, p2s):
        return [DummyGenomeHandler(genome=(a.genome + b.genome) / 2) for a, b in zip(p1s, p2s)]

    def mutate_population(self, handlers):
        for h in handlers:
            h.genome = h.genome + np.random.randn(len(h.genome)) * 0.01


def dummy_fitness(genome, log_dir):
    """Deterministic fitness so we can predict ordering."""
    os.makedirs(log_dir, exist_ok=True)
    # Write a marker file so we can verify the directory was created
    with open(os.path.join(log_dir, "result.txt"), "w") as f:
        f.write(f"fitness={float(genome.sum()):.6f}\n")
    return float(genome.sum())


def dummy_tournament(population, k=1):
    """Simple deterministic selection: pick top-k by fitness (with replacement)."""
    rows = []
    sorted_pop = population.sort_values("fitness", ascending=False).reset_index(drop=True)
    for i in range(k):
        rows.append(sorted_pop.iloc[i % len(sorted_pop)].to_dict())
    return pd.DataFrame(rows).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Test 1: evaluate_population indexing after sort
# ---------------------------------------------------------------------------

def test_in_pop_marks_top_n():
    """
    After sort + reset_index, exactly population_size individuals should be
    marked in_pop=True, and they should be the highest-fitness ones.
    """
    pop_size = 4

    # Simulate what evolve() does internally after evaluation
    records = [
        {"id": f"{i:04d}", "generation": 1, "genome": np.array([i]),
         "log_dir": f"/tmp/gen_01/ind_{i:04d}", "parent_ids": [None, None],
         "in_pop": False, "fitness": float(i)}
        for i in range(8)  # 8 candidates, select top 4
    ]
    combined = pd.DataFrame(records)
    combined["in_pop"] = False
    combined["generation"] = 1

    sorted_df = combined.sort_values(by="fitness", ascending=False).reset_index(drop=True)
    sorted_df.loc[:pop_size - 1, "in_pop"] = True
    population = sorted_df.head(n=pop_size)

    # Exactly pop_size should be in_pop
    assert sorted_df["in_pop"].sum() == pop_size, (
        f"Expected {pop_size} in_pop=True, got {sorted_df['in_pop'].sum()}"
    )

    # The top pop_size by fitness should be the ones marked
    top_ids = set(sorted_df.head(pop_size)["id"])
    marked_ids = set(sorted_df[sorted_df["in_pop"]]["id"])
    assert top_ids == marked_ids, (
        f"in_pop should mark top-{pop_size} by fitness.\n"
        f"  Top IDs: {top_ids}\n  Marked IDs: {marked_ids}"
    )

    # population should have sequential index after reset
    assert list(population.index) == list(range(pop_size)), (
        f"population index should be [0..{pop_size-1}], got {list(population.index)}"
    )

    print("PASSED: test_in_pop_marks_top_n")


# ---------------------------------------------------------------------------
# Test 2: .iloc retrieves correct genomes from non-sequential index
# ---------------------------------------------------------------------------

def test_iloc_retrieves_correct_genomes():
    """
    Verify that .iloc[i] retrieves the i-th row by position, not by label,
    even when the DataFrame has a non-sequential index.
    """
    pop_size = 4

    # Create a DataFrame with a deliberately non-sequential index
    records = [
        {"id": f"{i:04d}", "genome": np.array([float(i) * 10]), "fitness": float(i)}
        for i in range(8)
    ]
    df = pd.DataFrame(records)
    # Sort descending by fitness -> index becomes [7, 6, 5, 4, 3, 2, 1, 0]
    df = df.sort_values("fitness", ascending=False)
    population = df.head(pop_size)  # index: [7, 6, 5, 4]

    # .iloc should give positional access (top fitness first)
    genomes_iloc = [population["genome"].iloc[i] for i in range(pop_size)]
    ids_iloc = [population["id"].iloc[i] for i in range(pop_size)]

    # Position 0 should be the highest fitness (id=0007, genome=[70.0])
    assert ids_iloc[0] == "0007", f"Expected id '0007' at position 0, got '{ids_iloc[0]}'"
    assert ids_iloc[1] == "0006", f"Expected id '0006' at position 1, got '{ids_iloc[1]}'"
    np.testing.assert_array_equal(genomes_iloc[0], np.array([70.0]))
    np.testing.assert_array_equal(genomes_iloc[1], np.array([60.0]))

    # Now show what the OLD buggy code would have done:
    # population['genome'][0] does label-based lookup for index label 0,
    # which is NOT in the population (top-4 indices are 7,6,5,4).
    try:
        _ = population["genome"][0]
        # If this didn't raise, we're on old pandas with fallback — check value
        old_val = population["genome"][0]
        # With fallback it would return positional [0] which is correct,
        # but this behaviour is deprecated and unreliable.
        print("  (pandas fell back to positional — deprecated behaviour)")
    except KeyError:
        print("  (pandas correctly raised KeyError for label-based [0] — confirms bug)")

    print("PASSED: test_iloc_retrieves_correct_genomes")


# ---------------------------------------------------------------------------
# Test 3: Full evolve() with reevaluate_old=True saves new results
# ---------------------------------------------------------------------------

def test_reevaluation_saves_new_results():
    """
    Run evolve() for 2 generations with reevaluate_old=True and verify that
    individuals surviving from gen 0 get new result directories in gen 1.
    """
    tmpdir = tempfile.mkdtemp(prefix="test_mu_lambda_")
    try:
        pop_size = 4
        num_mutate = 4
        num_gens = 2

        # Use a fixed initial population so results are reproducible
        np.random.seed(42)
        initial_pop = [np.random.rand(6) for _ in range(pop_size)]

        all_individuals = evolve(
            fitness_function=dummy_fitness,
            population_size=pop_size,
            num_generations=num_gens,
            num_mutate=num_mutate,
            num_crossover=0,
            mutate_after_crossover=False,
            strategy_type="plus",
            parent_selection=dummy_tournament,
            initial_population=initial_pop,
            reevaluate_old=True,
            log_dir=tmpdir,
            genome_handler=DummyGenomeHandler,
            verbose=True,
            num_workers=1,
        )

        # Collect all individual records per generation
        gen0 = all_individuals[all_individuals["generation"] == 0]
        gen1 = all_individuals[all_individuals["generation"] == 1]
        gen2 = all_individuals[all_individuals["generation"] == 2]

        print(f"\n  Gen 0: {len(gen0)} records")
        print(f"  Gen 1: {len(gen1)} records")
        print(f"  Gen 2: {len(gen2)} records")

        # In plus strategy with reevaluation, gen 1 should have:
        #   pop_size (reevaluated old) + num_mutate (offspring) = 8 records
        assert len(gen1) == pop_size + num_mutate, (
            f"Gen 1 should have {pop_size + num_mutate} records, got {len(gen1)}"
        )

        # Find IDs that appear in gen 0 (original) and also in gen 1 (reevaluated)
        gen0_ids = set(gen0["id"])
        gen1_ids = set(gen1["id"])
        reevaluated_ids = gen0_ids & gen1_ids

        print(f"  Gen 0 IDs: {sorted(gen0_ids)}")
        print(f"  Gen 1 IDs: {sorted(gen1_ids)}")
        print(f"  Reevaluated (appear in both): {sorted(reevaluated_ids)}")

        assert len(reevaluated_ids) > 0, (
            "Expected some gen-0 individuals to be reevaluated in gen 1"
        )

        # Verify reevaluated individuals have DIFFERENT log_dirs from gen 0
        for rid in reevaluated_ids:
            gen0_logdir = gen0[gen0["id"] == rid]["log_dir"].iloc[0]
            gen1_logdir = gen1[gen1["id"] == rid]["log_dir"].iloc[0]

            assert gen0_logdir != gen1_logdir, (
                f"Individual {rid}: gen0 and gen1 log_dirs should differ.\n"
                f"  gen0: {gen0_logdir}\n  gen1: {gen1_logdir}"
            )

            # Verify that the gen-1 directory actually exists with results
            assert os.path.isdir(gen1_logdir), (
                f"Reevaluation directory does not exist: {gen1_logdir}"
            )
            result_file = os.path.join(gen1_logdir, "result.txt")
            assert os.path.isfile(result_file), (
                f"Reevaluation result file missing: {result_file}"
            )

            # Verify the gen-1 directory is under generation_01/
            assert "generation_01" in gen1_logdir, (
                f"Reevaluated log_dir should be under generation_01: {gen1_logdir}"
            )

            print(f"  ID {rid}: gen0={gen0_logdir}")
            print(f"         gen1={gen1_logdir} (exists with result.txt)")

        # Verify in_pop counts are correct per generation
        for gen_num in range(num_gens + 1):
            gen_df = all_individuals[all_individuals["generation"] == gen_num]
            in_pop_count = gen_df["in_pop"].sum()
            assert in_pop_count == pop_size, (
                f"Gen {gen_num}: expected {pop_size} in_pop=True, got {in_pop_count}"
            )

        print("\nPASSED: test_reevaluation_saves_new_results")

    finally:
        shutil.rmtree(tmpdir)


# ---------------------------------------------------------------------------
# Test 4: Verify the OLD buggy .loc[:pop_size] was wrong
# ---------------------------------------------------------------------------

def test_old_loc_bug_demonstration():
    """
    Demonstrate that the old code `sorted_df.loc[:pop_size, 'in_pop'] = True`
    (without reset_index) would mark the WRONG individuals.
    """
    pop_size = 3

    records = [
        {"id": "A", "fitness": 1.0},
        {"id": "B", "fitness": 5.0},
        {"id": "C", "fitness": 3.0},
        {"id": "D", "fitness": 7.0},
        {"id": "E", "fitness": 2.0},
        {"id": "F", "fitness": 6.0},
    ]
    combined = pd.DataFrame(records)
    combined["in_pop"] = False

    # OLD code (buggy): sort without reset_index, then .loc[:pop_size]
    sorted_old = combined.sort_values("fitness", ascending=False)
    # sorted_old index: [3, 5, 1, 2, 4, 0] (original row positions)
    # .loc[:3] selects labels 0, 1, 2, 3 — four rows, not three!
    sorted_old_copy = sorted_old.copy()
    sorted_old_copy.loc[:pop_size, "in_pop"] = True
    old_marked = set(sorted_old_copy[sorted_old_copy["in_pop"]]["id"])
    old_count = sorted_old_copy["in_pop"].sum()

    # NEW code (fixed): reset_index, then .loc[:pop_size - 1]
    sorted_new = combined.sort_values("fitness", ascending=False).reset_index(drop=True)
    sorted_new["in_pop"] = False
    sorted_new.loc[:pop_size - 1, "in_pop"] = True
    new_marked = set(sorted_new[sorted_new["in_pop"]]["id"])
    new_count = sorted_new["in_pop"].sum()

    print(f"  OLD code marked {old_count} individuals: {sorted(old_marked)}")
    print(f"  NEW code marked {new_count} individuals: {sorted(new_marked)}")

    # The correct top-3 by fitness are D(7), F(6), B(5)
    expected = {"D", "F", "B"}

    assert new_marked == expected, (
        f"NEW code should mark {expected}, got {new_marked}"
    )
    assert new_count == pop_size, (
        f"NEW code should mark exactly {pop_size}, got {new_count}"
    )

    # Old code was wrong (marked 4 instead of 3, and/or wrong individuals)
    if old_marked != expected or old_count != pop_size:
        print(f"  Confirmed: OLD code was buggy (marked {old_count} != {pop_size}, "
              f"or wrong set)")
    else:
        print("  Note: OLD code happened to be correct for this specific case")

    print("PASSED: test_old_loc_bug_demonstration")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Testing mu_lambda indexing fixes")
    print("=" * 60)
    print()

    test_in_pop_marks_top_n()
    print()
    test_iloc_retrieves_correct_genomes()
    print()
    test_old_loc_bug_demonstration()
    print()
    test_reevaluation_saves_new_results()

    print()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
