"""Tests for NEAT strategy with speciation (Phase 2)."""

import numpy as np
import pandas as pd
import pytest

from airevolve.evolution_tools.strategies.speciation import Species, SpeciationState
from airevolve.evolution_tools.strategies.neat import evolve_neat, _allocate_offspring
from airevolve.evolution_tools.genome_handlers.spherical_angular_genome_handler import (
    SphericalAngularDroneGenomeHandler,
)
from airevolve.evolution_tools.selectors.tournament import tournament_selection


# ---------------------------------------------------------------------- #
# Speciation tests
# ---------------------------------------------------------------------- #


class TestSpeciation:
    def test_distinct_clusters_different_species(self):
        """Two distinct genome clusters should be assigned to different species."""
        handler = SphericalAngularDroneGenomeHandler(
            min_max_narms=(4, 4),
        )

        # Create two clusters with very different genomes
        rng = np.random.default_rng(42)
        genomes = []

        # Cluster 1: small magnitudes
        for _ in range(5):
            g = handler.copy()
            g.genome.arms[:, 0] = rng.uniform(0.055, 0.07, size=4)
            genomes.append(g.genome.copy())

        # Cluster 2: large magnitudes (very different)
        for _ in range(5):
            g = handler.copy()
            g.genome.arms[:, 0] = rng.uniform(0.15, 0.17, size=4)
            # Also make other params very different
            g.genome.arms[:, 1] = rng.uniform(2.0, 3.0, size=4)
            genomes.append(g.genome.copy())

        pop_df = pd.DataFrame({
            "id": [str(i).zfill(4) for i in range(10)],
            "genome": genomes,
            "fitness": [float(i) for i in range(10)],
        })

        state = SpeciationState(compatibility_threshold=0.05)
        handler_kwargs = {
            "min_max_narms": (4, 4),
        }
        assignments = state.speciate(pop_df, SphericalAngularDroneGenomeHandler, handler_kwargs)

        species_ids = set(assignments.values())
        assert len(species_ids) >= 2, f"Expected >=2 species, got {len(species_ids)}"

    def test_stagnation_removal(self):
        """Species that haven't improved should be removed after stagnation_limit."""
        state = SpeciationState()
        state.species[0] = Species(
            id=0, representative_genome=None, member_ids=["a"],
            best_fitness=1.0, generations_since_improvement=20,
        )
        state.species[1] = Species(
            id=1, representative_genome=None, member_ids=["b"],
            best_fitness=5.0, generations_since_improvement=0,
        )
        state.species[2] = Species(
            id=2, representative_genome=None, member_ids=["c"],
            best_fitness=3.0, generations_since_improvement=20,
        )

        state.remove_stagnant_species(stagnation_limit=15, protect_top_n=1)

        assert 1 in state.species, "Top species should be protected"
        assert 0 not in state.species, "Stagnant low species should be removed"

    def test_threshold_adjustment(self):
        """Threshold should increase when too many species, decrease when too few."""
        state = SpeciationState(compatibility_threshold=3.0)
        for i in range(10):
            state.species[i] = Species(
                id=i, representative_genome=None, member_ids=[str(i)],
                best_fitness=0.0,
            )

        state.adjust_threshold(target_species_count=5)
        assert state.compatibility_threshold > 3.0

        state2 = SpeciationState(compatibility_threshold=3.0)
        state2.species[0] = Species(
            id=0, representative_genome=None, member_ids=["0"],
            best_fitness=0.0,
        )
        state2.adjust_threshold(target_species_count=5)
        assert state2.compatibility_threshold < 3.0


class TestOffspringAllocation:
    def test_allocation_sums_to_pop_size(self):
        """Total allocated offspring should equal population size."""
        pop_df = pd.DataFrame({
            "id": ["0", "1", "2", "3", "4", "5"],
            "fitness": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        })

        state = SpeciationState()
        state.species[0] = Species(
            id=0, representative_genome=None,
            member_ids=["0", "1", "2"], best_fitness=3.0,
        )
        state.species[1] = Species(
            id=1, representative_genome=None,
            member_ids=["3", "4", "5"], best_fitness=6.0,
        )

        alloc = _allocate_offspring(pop_df, state, population_size=10, min_species_size=2)
        total = sum(alloc.values())
        assert total == 10, f"Total {total} != 10"

    def test_fitter_species_gets_more(self):
        """Species with higher adjusted fitness should get more offspring."""
        pop_df = pd.DataFrame({
            "id": ["0", "1", "2", "3"],
            "fitness": [1.0, 1.0, 10.0, 10.0],
        })

        state = SpeciationState()
        state.species[0] = Species(
            id=0, representative_genome=None,
            member_ids=["0", "1"], best_fitness=1.0,
        )
        state.species[1] = Species(
            id=1, representative_genome=None,
            member_ids=["2", "3"], best_fitness=10.0,
        )

        alloc = _allocate_offspring(pop_df, state, population_size=20, min_species_size=2)
        assert alloc[1] >= alloc[0], "Fitter species should get >= offspring"


# ---------------------------------------------------------------------- #
# Integration test
# ---------------------------------------------------------------------- #


class TestNEATIntegration:
    def test_evolve_neat_runs(self, tmp_path):
        """evolve_neat should run for a few generations and return proper DataFrame."""

        def trivial_fitness(genome, log_dir):
            """Fitness = negative sum of magnitudes (maximize small drones)."""
            from airevolve.evolution_tools.genome_handlers.spherical_angular_genome_handler import SphericalNeatGenome
            arms = genome.arms if isinstance(genome, SphericalNeatGenome) else genome
            valid = arms[~np.isnan(arms[:, 0])]
            return -float(np.sum(valid[:, 0]))

        result = evolve_neat(
            fitness_function=trivial_fitness,
            population_size=10,
            num_generations=3,
            crossover_rate=0.5,
            parent_selection=tournament_selection,
            genome_handler=SphericalAngularDroneGenomeHandler,
            compatibility_threshold=2.0,
            species_elitism=1,
            stagnation_limit=15,
            min_species_size=2,
            adjust_threshold=True,
            target_species_count=3,
            log_dir=str(tmp_path),
            verbose=True,
            num_workers=1,
        )

        assert isinstance(result, pd.DataFrame)
        assert "species_id" in result.columns
        assert "fitness" in result.columns
        assert "genome" in result.columns
        assert len(result) > 10  # Initial pop + offspring over 3 gens
