#!/usr/bin/env python3
"""Unit tests for CPPN-NEAT drone genome handler and supporting modules."""

import unittest
import numpy as np
import numpy.testing as npt

from airevolve.evolution_tools.genome_handlers.cppn.network import (
    ActivationFunction,
    CPPNNetwork,
    ConnectionGene,
    NodeGene,
    NodeType,
    apply_activation,
)
from airevolve.evolution_tools.genome_handlers.cppn.innovation import InnovationCounter
from airevolve.evolution_tools.genome_handlers.cppn.evaluation import (
    evaluate_cppn,
    topological_sort,
)
from airevolve.evolution_tools.genome_handlers.cppn.segment_decoder import (
    decode_cppn_to_phenotype,
    _map_tanh_to_range,
)
from airevolve.evolution_tools.genome_handlers.cppn.mutations import (
    mutate_cppn,
    _add_node,
    _add_connection,
    _remove_node,
    _remove_connection,
    _mutate_weights,
    _mutate_activation,
    _toggle_connection,
    _would_create_cycle,
)
from airevolve.evolution_tools.genome_handlers.cppn_neat_genome_handler import (
    CPPNNeatDroneGenomeHandler,
)


# ======================================================================
# Helpers
# ======================================================================

def _make_minimal_network(rng=None, weight=1.0):
    """Build a minimal 2-input, 7-output CPPN with uniform weights."""
    if rng is None:
        rng = np.random.default_rng(0)
    net = CPPNNetwork()
    # Inputs
    for i in range(2):
        net.nodes[i] = NodeGene(
            node_id=i, node_type=NodeType.INPUT,
            activation=ActivationFunction.IDENTITY, input_label=f"in{i}",
        )
    # Outputs (tanh)
    for j in range(7):
        nid = 2 + j
        net.nodes[nid] = NodeGene(
            node_id=nid, node_type=NodeType.OUTPUT,
            activation=ActivationFunction.TANH, output_index=j,
        )
    net.next_node_id = 9
    inn = 0
    for i in range(2):
        for j in range(7):
            tgt = 2 + j
            net.connections[inn] = ConnectionGene(
                innovation_number=inn, source_id=i, target_id=tgt,
                weight=weight, enabled=True,
            )
            inn += 1
    return net


# ======================================================================
# Network data structure tests
# ======================================================================

class TestNetworkDataStructures(unittest.TestCase):

    def test_minimal_network_structure(self):
        net = _make_minimal_network()
        self.assertEqual(len(net.get_input_nodes()), 2)
        self.assertEqual(len(net.get_output_nodes()), 7)
        self.assertEqual(len(net.get_hidden_nodes()), 0)
        self.assertEqual(len(net.connections), 14)

    def test_copy_is_deep(self):
        net = _make_minimal_network()
        net2 = net.copy()
        # Mutate the copy
        net2.connections[0].weight = 999.0
        self.assertNotEqual(net.connections[0].weight, 999.0)

    def test_get_enabled_connections(self):
        net = _make_minimal_network()
        net.connections[0].enabled = False
        enabled = net.get_enabled_connections()
        self.assertEqual(len(enabled), 13)

    def test_activation_functions(self):
        x = np.array([-1.0, 0.0, 1.0])
        # identity
        npt.assert_array_equal(apply_activation(ActivationFunction.IDENTITY, x), x)
        # tanh
        npt.assert_array_almost_equal(
            apply_activation(ActivationFunction.TANH, x), np.tanh(x)
        )
        # sigmoid at 0 → 0.5
        self.assertAlmostEqual(
            apply_activation(ActivationFunction.SIGMOID, np.array([0.0]))[0], 0.5
        )
        # relu
        npt.assert_array_equal(
            apply_activation(ActivationFunction.RELU, x), [0.0, 0.0, 1.0]
        )
        # step
        npt.assert_array_equal(
            apply_activation(ActivationFunction.STEP, x), [0.0, 0.0, 1.0]
        )
        # gaussian at 0 → 1
        self.assertAlmostEqual(
            apply_activation(ActivationFunction.GAUSSIAN, np.array([0.0]))[0], 1.0
        )
        # abs
        npt.assert_array_equal(
            apply_activation(ActivationFunction.ABS, x), [1.0, 0.0, 1.0]
        )
        # sin / cos
        npt.assert_array_almost_equal(
            apply_activation(ActivationFunction.SIN, x), np.sin(x)
        )
        npt.assert_array_almost_equal(
            apply_activation(ActivationFunction.COS, x), np.cos(x)
        )


# ======================================================================
# Innovation counter tests
# ======================================================================

class TestInnovationCounter(unittest.TestCase):

    def test_same_structural_mutation_same_number(self):
        ic = InnovationCounter()
        n1 = ic.get_innovation(0, 5)
        n2 = ic.get_innovation(0, 5)
        self.assertEqual(n1, n2)

    def test_different_mutations_different_numbers(self):
        ic = InnovationCounter()
        n1 = ic.get_innovation(0, 5)
        n2 = ic.get_innovation(1, 5)
        self.assertNotEqual(n1, n2)

    def test_reset_generation_clears_cache(self):
        ic = InnovationCounter()
        n1 = ic.get_innovation(0, 5)
        ic.reset_generation()
        # After reset, same pair gets a NEW number (counter keeps going)
        n2 = ic.get_innovation(0, 5)
        self.assertNotEqual(n1, n2)


# ======================================================================
# Evaluation tests
# ======================================================================

class TestEvaluation(unittest.TestCase):

    def test_topological_sort_minimal(self):
        net = _make_minimal_network()
        order = topological_sort(net)
        # Inputs should come before outputs
        input_ids = {n.node_id for n in net.get_input_nodes()}
        output_ids = {n.node_id for n in net.get_output_nodes()}
        for iid in input_ids:
            for oid in output_ids:
                self.assertLess(order.index(iid), order.index(oid))

    def test_cycle_detection(self):
        net = CPPNNetwork()
        net.nodes[0] = NodeGene(0, NodeType.HIDDEN, ActivationFunction.IDENTITY)
        net.nodes[1] = NodeGene(1, NodeType.HIDDEN, ActivationFunction.IDENTITY)
        net.connections[0] = ConnectionGene(0, 0, 1, 1.0, True)
        net.connections[1] = ConnectionGene(1, 1, 0, 1.0, True)
        with self.assertRaises(ValueError):
            topological_sort(net)

    def test_evaluate_known_weights(self):
        """All weights = 1, all biases = 0, tanh outputs.
        Each output receives sum of both inputs via weight 1 from each input.
        With inputs [0.5, 1.0]: raw = 0.5 + 1.0 = 1.5, tanh(1.5) ≈ 0.9051
        """
        net = _make_minimal_network(weight=1.0)
        outputs = evaluate_cppn(net, np.array([0.5, 1.0]))
        expected = np.tanh(1.5)
        self.assertEqual(outputs.shape, (7,))
        for v in outputs:
            self.assertAlmostEqual(v, expected, places=5)

    def test_evaluate_batch(self):
        net = _make_minimal_network(weight=1.0)
        inputs = np.array([[0.0, 1.0], [1.0, 1.0]])
        outputs = evaluate_cppn(net, inputs)
        self.assertEqual(outputs.shape, (2, 7))
        # Row 0: tanh(0 + 1) = tanh(1)
        npt.assert_array_almost_equal(outputs[0], np.tanh(1.0))
        # Row 1: tanh(1 + 1) = tanh(2)
        npt.assert_array_almost_equal(outputs[1], np.tanh(2.0))

    def test_disabled_connections_ignored(self):
        net = _make_minimal_network(weight=1.0)
        # Disable all connections from input 0
        for conn in net.connections.values():
            if conn.source_id == 0:
                conn.enabled = False
        outputs = evaluate_cppn(net, np.array([100.0, 1.0]))
        # Only bias input (1.0) should contribute
        expected = np.tanh(1.0)
        for v in outputs:
            self.assertAlmostEqual(v, expected, places=5)


# ======================================================================
# Segment decoder tests
# ======================================================================

class TestSegmentDecoder(unittest.TestCase):

    def setUp(self):
        self.parameter_limits = np.array([
            [0.055, 0.17],
            [-np.pi, np.pi],
            [-np.pi / 2, np.pi / 2],
            [-np.pi, np.pi],
            [-np.pi / 2, np.pi / 2],
            [0, 1],
        ])

    def test_map_tanh_to_range(self):
        # -1 → low, 0 → midpoint, +1 → high
        self.assertAlmostEqual(_map_tanh_to_range(-1.0, 2.0, 4.0), 2.0)
        self.assertAlmostEqual(_map_tanh_to_range(0.0, 2.0, 4.0), 3.0)
        self.assertAlmostEqual(_map_tanh_to_range(1.0, 2.0, 4.0), 4.0)

    def test_output_shape(self):
        net = _make_minimal_network(weight=1.0)
        phenotype = decode_cppn_to_phenotype(
            net, num_segments=8, arm_limit=6,
            parameter_limits=self.parameter_limits,
        )
        self.assertEqual(phenotype.shape, (6, 6))

    def test_arm_limit_respected(self):
        """Even with many segments, we never exceed arm_limit arms."""
        net = _make_minimal_network(weight=5.0)  # large weight → tanh saturates ≈ 1
        phenotype = decode_cppn_to_phenotype(
            net, num_segments=20, arm_limit=4,
            parameter_limits=self.parameter_limits,
        )
        arm_count = int(np.sum(~np.isnan(phenotype[:, 0])))
        self.assertLessEqual(arm_count, 4)

    def test_force_arms_rule(self):
        """Rule 2: if remaining_segments <= remaining_slots, arms are forced."""
        # Weight = -5 → arm_present output saturates to tanh(-10) ≈ -1 (skip arm)
        # but Rule 2 should override
        net = _make_minimal_network(weight=-5.0)
        arm_limit = 4
        num_segments = 4
        phenotype = decode_cppn_to_phenotype(
            net, num_segments=num_segments, arm_limit=arm_limit,
            parameter_limits=self.parameter_limits,
        )
        arm_count = int(np.sum(~np.isnan(phenotype[:, 0])))
        # With 4 segments and 4 arm_limit, all arms must be placed (rule 2)
        self.assertEqual(arm_count, arm_limit)

    def test_direction_mapping(self):
        """Direction should be 0 or 1."""
        net = _make_minimal_network(weight=1.0)
        phenotype = decode_cppn_to_phenotype(
            net, num_segments=8, arm_limit=6,
            parameter_limits=self.parameter_limits,
        )
        valid_mask = ~np.isnan(phenotype[:, 0])
        directions = phenotype[valid_mask, 5]
        for d in directions:
            self.assertIn(d, [0.0, 1.0])

    def test_parameters_within_bounds(self):
        """All decoded parameters should be within limits."""
        rng = np.random.default_rng(42)
        net = _make_minimal_network()
        # Randomise weights
        for conn in net.connections.values():
            conn.weight = rng.uniform(-3, 3)
        phenotype = decode_cppn_to_phenotype(
            net, num_segments=12, arm_limit=8,
            parameter_limits=self.parameter_limits,
        )
        valid_mask = ~np.isnan(phenotype[:, 0])
        valid = phenotype[valid_mask]
        for i in range(6):
            if i == 5:  # direction
                self.assertTrue(np.all(np.isin(valid[:, i], [0.0, 1.0])))
            elif i == 1:  # arm_yaw — constrained to segment range
                # Can extend up to half_width beyond [-π, π] at boundary segments
                half_width = np.pi / 12  # π / num_segments
                self.assertTrue(np.all(valid[:, i] >= -np.pi - half_width - 1e-9))
                self.assertTrue(np.all(valid[:, i] <= np.pi + half_width + 1e-9))
            else:
                lo, hi = self.parameter_limits[i]
                self.assertTrue(
                    np.all(valid[:, i] >= lo - 1e-9),
                    f"Param {i} below lower bound: {valid[:, i]}",
                )
                self.assertTrue(
                    np.all(valid[:, i] <= hi + 1e-9),
                    f"Param {i} above upper bound: {valid[:, i]}",
                )

    def test_arm_yaw_within_segment(self):
        """arm_yaw should lie within the segment's angular range."""
        net = _make_minimal_network(weight=0.5)
        num_segments = 8
        arm_limit = 8
        phenotype = decode_cppn_to_phenotype(
            net, num_segments=num_segments, arm_limit=arm_limit,
            parameter_limits=self.parameter_limits,
        )
        segment_width = 2 * np.pi / num_segments
        valid_mask = ~np.isnan(phenotype[:, 0])
        arm_yaws = phenotype[valid_mask, 1]
        # Each yaw should lie within ±half_width of some segment center
        half_width = segment_width / 2.0
        for yaw in arm_yaws:
            # Find closest segment center
            found = False
            for seg_idx in range(num_segments):
                center = seg_idx * segment_width - np.pi
                if center - half_width - 1e-9 <= yaw <= center + half_width + 1e-9:
                    found = True
                    break
            self.assertTrue(found, f"arm_yaw {yaw} not in any segment")


# ======================================================================
# Mutation tests
# ======================================================================

class TestMutations(unittest.TestCase):

    def test_add_node_increases_counts(self):
        net = _make_minimal_network()
        ic = InnovationCounter()
        # Advance innovation counter past existing
        for _ in range(20):
            ic.get_innovation(99, ic.current)
        ic.reset_generation()
        rng = np.random.default_rng(0)

        old_node_count = len(net.nodes)
        old_conn_count = len(net.connections)
        _add_node(net, ic, rng)
        self.assertEqual(len(net.nodes), old_node_count + 1)
        # +2 new connections, but original connection disabled (still in dict)
        self.assertEqual(len(net.connections), old_conn_count + 2)
        self.assertEqual(len(net.get_hidden_nodes()), 1)

    def test_add_connection(self):
        net = _make_minimal_network()
        ic = InnovationCounter()
        for _ in range(20):
            ic.get_innovation(99, ic.current)
        ic.reset_generation()
        rng = np.random.default_rng(0)

        # First add a hidden node so there's a potential new connection
        _add_node(net, ic, rng)
        old_conn_count = len(net.connections)
        _add_connection(net, ic, rng)
        # Should have added at least one (if any valid pair exists)
        self.assertGreaterEqual(len(net.connections), old_conn_count)

    def test_remove_node(self):
        net = _make_minimal_network()
        ic = InnovationCounter()
        for _ in range(20):
            ic.get_innovation(99, ic.current)
        ic.reset_generation()
        rng = np.random.default_rng(0)

        _add_node(net, ic, rng)
        self.assertEqual(len(net.get_hidden_nodes()), 1)
        _remove_node(net, rng)
        self.assertEqual(len(net.get_hidden_nodes()), 0)

    def test_remove_connection_disables(self):
        net = _make_minimal_network()
        rng = np.random.default_rng(0)
        before_enabled = len(net.get_enabled_connections())
        _remove_connection(net, rng)
        after_enabled = len(net.get_enabled_connections())
        self.assertEqual(after_enabled, before_enabled - 1)

    def test_mutate_weights_changes_values(self):
        net = _make_minimal_network(weight=0.0)
        rng = np.random.default_rng(42)
        _mutate_weights(net, rng)
        # At least some weights should have changed from 0
        weights = [c.weight for c in net.connections.values()]
        self.assertTrue(any(w != 0.0 for w in weights))

    def test_mutate_activation_changes(self):
        net = _make_minimal_network()
        ic = InnovationCounter()
        for _ in range(20):
            ic.get_innovation(99, ic.current)
        ic.reset_generation()
        rng = np.random.default_rng(0)

        _add_node(net, ic, rng)
        hidden = net.get_hidden_nodes()
        old_act = hidden[0].activation
        # Try many times to ensure the activation changes (random choice)
        changed = False
        for _ in range(50):
            _mutate_activation(net, np.random.default_rng())
            if net.get_hidden_nodes()[0].activation != old_act:
                changed = True
                break
        self.assertTrue(changed)

    def test_toggle_connection(self):
        net = _make_minimal_network()
        rng = np.random.default_rng(0)
        before_enabled = len(net.get_enabled_connections())
        _toggle_connection(net, rng)
        after_enabled = len(net.get_enabled_connections())
        # Should have toggled one
        self.assertEqual(abs(after_enabled - before_enabled), 1)

    def test_no_cycles_after_add_connection(self):
        """add_connection should never create cycles."""
        net = _make_minimal_network()
        ic = InnovationCounter()
        for _ in range(20):
            ic.get_innovation(99, ic.current)
        ic.reset_generation()
        rng = np.random.default_rng(42)

        # Add some hidden nodes first
        for _ in range(5):
            _add_node(net, ic, rng)

        # Now try adding many connections
        for _ in range(50):
            _add_connection(net, ic, rng)

        # Verify no cycle: topological sort should succeed
        try:
            topological_sort(net)
        except ValueError:
            self.fail("Cycle detected after add_connection mutations")

    def test_would_create_cycle_simple(self):
        net = CPPNNetwork()
        net.nodes[0] = NodeGene(0, NodeType.HIDDEN, ActivationFunction.IDENTITY)
        net.nodes[1] = NodeGene(1, NodeType.HIDDEN, ActivationFunction.IDENTITY)
        net.connections[0] = ConnectionGene(0, 0, 1, 1.0, True)
        # Adding 1→0 would create cycle
        self.assertTrue(_would_create_cycle(net, 1, 0))
        # Adding 0→1 already exists but direction is fine (no NEW cycle)
        self.assertFalse(_would_create_cycle(net, 0, 1))

    def test_mutate_cppn_runs_without_error(self):
        """Smoke test: run many mutations and ensure no crash."""
        net = _make_minimal_network()
        ic = InnovationCounter()
        for _ in range(20):
            ic.get_innovation(99, ic.current)
        ic.reset_generation()
        rng = np.random.default_rng(42)

        for _ in range(200):
            mutate_cppn(net, ic, rng)


# ======================================================================
# GenomeHandler interface tests
# ======================================================================

class TestCPPNNeatDroneGenomeHandler(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.default_rng(42)
        self.parameter_limits = np.array([
            [0.055, 0.17],
            [-np.pi, np.pi],
            [-np.pi / 2, np.pi / 2],
            [-np.pi, np.pi],
            [-np.pi / 2, np.pi / 2],
            [0, 1],
        ])

    def _make_handler(self, **kwargs):
        defaults = dict(
            num_segments=12,
            min_max_narms=(3, 6),
            parameter_limits=self.parameter_limits,
            rng=self.rng,
        )
        defaults.update(kwargs)
        return CPPNNeatDroneGenomeHandler(**defaults)

    def test_default_construction(self):
        handler = CPPNNeatDroneGenomeHandler()
        self.assertIsInstance(handler.genome, CPPNNetwork)
        self.assertEqual(len(handler.genome.get_input_nodes()), 2)
        self.assertEqual(len(handler.genome.get_output_nodes()), 7)
        # 3 initial hidden nodes by default
        self.assertEqual(len(handler.genome.get_hidden_nodes()), 3)
        # 14 base connections + 2 per hidden node (split adds 2, original disabled stays in dict)
        self.assertEqual(len(handler.genome.connections), 14 + 3 * 2)

    def test_construction_no_hidden_nodes(self):
        handler = CPPNNeatDroneGenomeHandler(initial_hidden_nodes=0)
        self.assertEqual(len(handler.genome.get_hidden_nodes()), 0)
        self.assertEqual(len(handler.genome.connections), 14)

    '''def test_output_biases_randomised(self):
        handler = self._make_handler()
        output_nodes = handler.genome.get_output_nodes()
        biases = [n.bias for n in output_nodes]
        # arm_present (index 0) should have positive bias
        self.assertGreater(output_nodes[0].bias, 0.0)
        # At least some output biases should be non-zero
        self.assertTrue(any(b != 0.0 for b in biases))
    '''

    '''def test_hidden_nodes_have_spatial_activations(self):
        handler = self._make_handler()
        hidden = handler.genome.get_hidden_nodes()
        self.assertEqual(len(hidden), 3)
        spatial_acts = {
            ActivationFunction.SIN, ActivationFunction.COS,
            ActivationFunction.GAUSSIAN, ActivationFunction.SIGMOID,
            ActivationFunction.ABS,
        }
        for node in hidden:
            self.assertIn(node.activation, spatial_acts)
    '''
    def test_construction_with_genome(self):
        net = _make_minimal_network(weight=2.0)
        handler = self._make_handler(genome=net)
        # Should be a copy
        self.assertIsNot(handler.genome, net)
        self.assertEqual(handler.genome.connections[0].weight, 2.0)

    def test_get_phenotype_shape(self):
        handler = self._make_handler()
        phenotype = handler.get_phenotype()
        self.assertEqual(phenotype.shape, (6, 6))

    def test_get_phenotype_arm_count_in_bounds(self):
        handler = self._make_handler()
        phenotype = handler.get_phenotype()
        arm_count = int(np.sum(~np.isnan(phenotype[:, 0])))
        self.assertGreaterEqual(arm_count, 3)
        self.assertLessEqual(arm_count, 6)

    def test_generate_random_population(self):
        handler = self._make_handler()
        pop = handler.generate_random_population(10)
        self.assertEqual(len(pop), 10)
        for individual in pop:
            self.assertIsInstance(individual, CPPNNeatDroneGenomeHandler)
            self.assertIsInstance(individual.genome, CPPNNetwork)

    def test_mutate_changes_genome(self):
        handler = self._make_handler()
        # Force weight mutation (high probability)
        handler.prob_mutate_weights = 1.0
        handler.prob_add_node = 0.0
        handler.prob_add_connection = 0.0
        handler.prob_remove_node = 0.0
        handler.prob_remove_connection = 0.0
        handler.prob_mutate_activation = 0.0
        handler.prob_toggle_connection = 0.0

        old_weights = [
            c.weight for c in handler.genome.connections.values()
        ]
        handler.mutate()
        new_weights = [
            c.weight for c in handler.genome.connections.values()
        ]
        self.assertNotEqual(old_weights, new_weights)

    def test_copy_is_independent(self):
        handler = self._make_handler()
        handler2 = handler.copy()
        handler2.genome.connections[0].weight = 999.0
        self.assertNotEqual(handler.genome.connections[0].weight, 999.0)

    def test_is_valid(self):
        handler = self._make_handler()
        self.assertTrue(handler.is_valid())

    '''def test_crossover_raises(self):
        handler = self._make_handler()
        with self.assertRaises(NotImplementedError):
            handler.crossover(handler)
    '''
    def test_crossover_population_returns_copies(self):
        handler = self._make_handler()
        pop = handler.generate_random_population(4)
        pop2 = handler.generate_random_population(4)
        children = handler.crossover_population(pop, pop2)
        self.assertEqual(len(children), 4)
        # Children should be independent from pop1
        for child, parent in zip(children, pop):
            self.assertIsNot(child.genome, parent.genome)

    def test_repair_is_noop(self):
        handler = self._make_handler()
        # Should not raise
        handler.repair()

    def test_repair_integration(self):
        """When repair is enabled, get_phenotype applies repair."""
        handler = self._make_handler(repair=True, enable_collision_repair=False)
        phenotype = handler.get_phenotype()
        # Should still produce valid shape and arm counts
        self.assertEqual(phenotype.shape, (6, 6))
        arm_count = int(np.sum(~np.isnan(phenotype[:, 0])))
        self.assertGreaterEqual(arm_count, 1)

    def test_mutate_population(self):
        handler = self._make_handler()
        pop = handler.generate_random_population(5)
        # Should not raise
        handler.mutate_population(pop)

    def test_population_validity_after_mutations(self):
        """After many mutations, individuals should still decode to valid phenotypes."""
        handler = self._make_handler()
        pop = handler.generate_random_population(10)
        for _ in range(20):
            handler.mutate_population(pop)
        for individual in pop:
            phenotype = individual.get_phenotype()
            self.assertEqual(phenotype.shape, (6, 6))
            arm_count = int(np.sum(~np.isnan(phenotype[:, 0])))
            self.assertGreaterEqual(arm_count, 0)
            self.assertLessEqual(arm_count, 6)

    def test_innovation_counter_shared(self):
        """All handler instances share the same innovation counter."""
        h1 = self._make_handler()
        h2 = self._make_handler()
        self.assertIs(
            CPPNNeatDroneGenomeHandler._innovation_counter,
            CPPNNeatDroneGenomeHandler._innovation_counter,
        )
        self.assertIs(h1._innovation_counter, h2._innovation_counter)


# ======================================================================
# Integration / smoke tests
# ======================================================================

class TestIntegrationSmoke(unittest.TestCase):
    """End-to-end smoke test mimicking a generation of mu_lambda evolution."""

    def test_one_generation_smoke(self):
        handler = CPPNNeatDroneGenomeHandler(
            num_segments=12,
            min_max_narms=(3, 6),
            rng=np.random.default_rng(123),
        )

        # Generate population
        pop = handler.generate_random_population(8)
        self.assertEqual(len(pop), 8)

        # Check all have valid phenotypes
        for ind in pop:
            p = ind.get_phenotype()
            self.assertEqual(p.shape[1], 6)

        # Crossover (returns copies)
        children = handler.crossover_population(pop[:4], pop[4:])
        self.assertEqual(len(children), 4)

        # Mutate
        handler.mutate_population(children)

        # Access .genome attribute (like mu_lambda does)
        genomes = [h.genome for h in children]
        self.assertEqual(len(genomes), 4)

        # Reconstruct handlers from genomes (like mu_lambda does)
        reconstructed = [
            CPPNNeatDroneGenomeHandler(
                genome=g, num_segments=12, min_max_narms=(3, 6),
                rng=np.random.default_rng(123),
            )
            for g in genomes
        ]
        for r in reconstructed:
            self.assertIsInstance(r.genome, CPPNNetwork)
            p = r.get_phenotype()
            self.assertEqual(p.shape, (6, 6))


if __name__ == "__main__":
    unittest.main()
