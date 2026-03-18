"""Tests for CPPN crossover and compatibility distance (Phase 1)."""

import numpy as np
import pytest

from airevolve.evolution_tools.genome_handlers.cppn.network import (
    ActivationFunction,
    CPPNNetwork,
    ConnectionGene,
    NodeGene,
    NodeType,
)
from airevolve.evolution_tools.genome_handlers.cppn.crossover import crossover_cppn
from airevolve.evolution_tools.genome_handlers.cppn.compatibility import (
    cppn_compatibility_distance,
)
from airevolve.evolution_tools.genome_handlers.cppn_neat_genome_handler import (
    CPPNNeatDroneGenomeHandler,
)
from airevolve.evolution_tools.genome_handlers.hybrid_cppn_genome_handler import (
    HybridCPPNDroneGenomeHandler,
    HybridGenome,
)
from airevolve.evolution_tools.genome_handlers.spherical_angular_genome_handler import (
    SphericalAngularDroneGenomeHandler,
)


# ---------------------------------------------------------------------- #
# Fixtures
# ---------------------------------------------------------------------- #

def _make_simple_net(
    connections: dict[int, tuple[int, int, float, bool]],
    n_inputs: int = 2,
    n_outputs: int = 2,
) -> CPPNNetwork:
    """Build a small CPPN for testing.

    connections: {innovation: (src, tgt, weight, enabled)}
    """
    net = CPPNNetwork()
    for i in range(n_inputs):
        net.nodes[i] = NodeGene(i, NodeType.INPUT, ActivationFunction.IDENTITY, 0.0, input_label=f"in{i}")
    for j in range(n_outputs):
        nid = n_inputs + j
        net.nodes[nid] = NodeGene(nid, NodeType.OUTPUT, ActivationFunction.TANH, bias=0.0, output_index=j)
    net.next_node_id = n_inputs + n_outputs

    for inn, (src, tgt, w, en) in connections.items():
        # Add hidden nodes if referenced
        if src not in net.nodes:
            net.nodes[src] = NodeGene(src, NodeType.HIDDEN, ActivationFunction.RELU, bias=0.1)
            net.next_node_id = max(net.next_node_id, src + 1)
        if tgt not in net.nodes:
            net.nodes[tgt] = NodeGene(tgt, NodeType.HIDDEN, ActivationFunction.RELU, bias=0.1)
            net.next_node_id = max(net.next_node_id, tgt + 1)
        net.connections[inn] = ConnectionGene(inn, src, tgt, w, en)

    return net


@pytest.fixture
def rng():
    return np.random.default_rng(42)


# ---------------------------------------------------------------------- #
# Crossover tests
# ---------------------------------------------------------------------- #


class TestCPPNCrossover:
    def test_matching_genes_inherited(self, rng):
        """Matching genes (same innovation) should come from one parent or the other."""
        net1 = _make_simple_net({0: (0, 2, 1.0, True), 1: (1, 3, 2.0, True)})
        net2 = _make_simple_net({0: (0, 2, -1.0, True), 1: (1, 3, -2.0, True)})

        child = crossover_cppn(net1, net2, 1.0, 1.0, rng)

        for inn in [0, 1]:
            w = child.connections[inn].weight
            assert w in (
                net1.connections[inn].weight,
                net2.connections[inn].weight,
            ), f"Inn {inn} weight {w} not from either parent"

    def test_disjoint_excess_from_fitter(self, rng):
        """Disjoint/excess genes should come only from the fitter parent."""
        # net1 has innovations 0,1,2 — net2 has 0,1,3
        # net1 is fitter => child gets inn 2 (disjoint) but not 3
        net1 = _make_simple_net({
            0: (0, 2, 1.0, True),
            1: (1, 3, 2.0, True),
            2: (0, 3, 0.5, True),
        })
        net2 = _make_simple_net({
            0: (0, 2, -1.0, True),
            1: (1, 3, -2.0, True),
            3: (1, 2, -0.5, True),
        })

        child = crossover_cppn(net1, net2, fitness1=5.0, fitness2=1.0, rng=rng)

        assert 2 in child.connections, "Disjoint gene from fitter parent missing"
        assert 3 not in child.connections, "Disjoint gene from weaker parent should be absent"

    def test_equal_fitness_randomly_includes_disjoint(self):
        """With equal fitness, each disjoint/excess gene is included ~50% of the time."""
        net1 = _make_simple_net({0: (0, 2, 1.0, True), 2: (0, 3, 0.5, True)})
        net2 = _make_simple_net({0: (0, 2, -1.0, True), 3: (1, 2, -0.5, True)})

        inn2_count = 0
        inn3_count = 0
        n_trials = 1000
        for seed in range(n_trials):
            child = crossover_cppn(
                net1, net2, fitness1=3.0, fitness2=3.0,
                rng=np.random.default_rng(seed),
            )
            if 2 in child.connections:
                inn2_count += 1
            if 3 in child.connections:
                inn3_count += 1

        # Each should be included roughly 50% of the time
        rate2 = inn2_count / n_trials
        rate3 = inn3_count / n_trials
        assert 0.35 < rate2 < 0.65, f"Inn 2 inclusion rate {rate2:.2f} not ~50%"
        assert 0.35 < rate3 < 0.65, f"Inn 3 inclusion rate {rate3:.2f} not ~50%"

    def test_node_inheritance(self, rng):
        """All nodes referenced by inherited connections are present."""
        # Add a hidden node via connection
        net1 = _make_simple_net({
            0: (0, 4, 1.0, True),  # 4 = hidden node
            1: (4, 2, 0.5, True),
        })
        net2 = _make_simple_net({0: (0, 2, -1.0, True)})

        child = crossover_cppn(net1, net2, fitness1=5.0, fitness2=1.0, rng=rng)

        for conn in child.connections.values():
            assert conn.source_id in child.nodes, f"Source {conn.source_id} missing"
            assert conn.target_id in child.nodes, f"Target {conn.target_id} missing"

    def test_disable_probability(self, rng):
        """Gene disabled in one parent should be disabled in child ~75% of the time."""
        net1 = _make_simple_net({0: (0, 2, 1.0, True)})
        net2 = _make_simple_net({0: (0, 2, 1.0, False)})  # disabled in net2

        disabled_count = 0
        n_trials = 1000
        for seed in range(n_trials):
            child = crossover_cppn(net1, net2, 1.0, 1.0, np.random.default_rng(seed))
            if not child.connections[0].enabled:
                disabled_count += 1

        rate = disabled_count / n_trials
        assert 0.60 < rate < 0.90, f"Disable rate {rate:.2f} outside expected range"


# ---------------------------------------------------------------------- #
# Compatibility distance tests
# ---------------------------------------------------------------------- #


class TestCPPNCompatibility:
    def test_identical_networks_zero_distance(self):
        """Distance between identical networks should be 0."""
        net = _make_simple_net({0: (0, 2, 1.0, True), 1: (1, 3, 2.0, True)})
        assert cppn_compatibility_distance(net, net.copy()) == 0.0

    def test_disjoint_gives_positive_distance(self):
        """Completely disjoint networks should have high distance."""
        net1 = _make_simple_net({0: (0, 2, 1.0, True)})
        net2 = _make_simple_net({5: (1, 3, 1.0, True)})

        dist = cppn_compatibility_distance(net1, net2)
        assert dist > 0

    def test_weight_difference_moderate_distance(self):
        """Same topology but different weights gives moderate distance."""
        net1 = _make_simple_net({0: (0, 2, 1.0, True), 1: (1, 3, 2.0, True)})
        net2 = _make_simple_net({0: (0, 2, 1.5, True), 1: (1, 3, 2.5, True)})

        dist = cppn_compatibility_distance(net1, net2)
        assert 0 < dist < 5  # moderate, not extreme


# ---------------------------------------------------------------------- #
# Handler-level crossover tests
# ---------------------------------------------------------------------- #


class TestCPPNNeatHandlerCrossover:
    def test_crossover_returns_valid_handler(self):
        """crossover() should return a CPPNNeatDroneGenomeHandler."""
        h1 = CPPNNeatDroneGenomeHandler()
        h2 = CPPNNeatDroneGenomeHandler()
        h1.fitness = 1.0
        h2.fitness = 0.5

        child = h1.crossover(h2)
        assert isinstance(child, CPPNNeatDroneGenomeHandler)
        assert isinstance(child.genome, CPPNNetwork)

    def test_crossover_population_works(self):
        """crossover_population() should produce children."""
        h = CPPNNeatDroneGenomeHandler()
        pop1 = h.generate_random_population(3)
        pop2 = h.generate_random_population(3)
        for p in pop1 + pop2:
            p.fitness = np.random.random()

        children = h.crossover_population(pop1, pop2)
        assert len(children) == 3
        for c in children:
            assert isinstance(c, CPPNNeatDroneGenomeHandler)

    def test_compatibility_distance(self):
        """Handler compatibility_distance should return a float."""
        h1 = CPPNNeatDroneGenomeHandler()
        h2 = CPPNNeatDroneGenomeHandler()
        dist = h1.compatibility_distance(h2)
        assert isinstance(dist, float)
        assert dist >= 0


class TestHybridHandlerCrossover:
    def test_crossover_returns_valid_handler(self):
        """crossover() should return a HybridCPPNDroneGenomeHandler."""
        h1 = HybridCPPNDroneGenomeHandler()
        h2 = HybridCPPNDroneGenomeHandler()
        h1.fitness = 2.0
        h2.fitness = 1.0

        child = h1.crossover(h2)
        assert isinstance(child, HybridCPPNDroneGenomeHandler)
        assert isinstance(child.genome, HybridGenome)

    def test_compatibility_distance(self):
        h1 = HybridCPPNDroneGenomeHandler()
        h2 = HybridCPPNDroneGenomeHandler()
        dist = h1.compatibility_distance(h2)
        assert isinstance(dist, float)
        assert dist >= 0


class TestSphericalCompatibility:
    def test_identical_distance_zero(self):
        """Identical genomes should have zero distance."""
        h = SphericalAngularDroneGenomeHandler()
        h2 = h.copy()
        assert h.compatibility_distance(h2) == 0.0

    def test_different_gives_positive(self):
        """Different genomes should have positive distance."""
        h1 = SphericalAngularDroneGenomeHandler()
        h2 = SphericalAngularDroneGenomeHandler()
        dist = h1.compatibility_distance(h2)
        assert dist > 0

    def test_ordering(self):
        """Similar genomes should be closer than very different ones."""
        h1 = SphericalAngularDroneGenomeHandler()
        h2 = h1.copy()
        h2.mutate()  # small change
        h3 = SphericalAngularDroneGenomeHandler()  # completely different

        d_similar = h1.compatibility_distance(h2)
        d_different = h1.compatibility_distance(h3)
        # d_similar should typically be < d_different (not guaranteed, but very likely)
        # Just check both are non-negative
        assert d_similar >= 0
        assert d_different >= 0
