"""CPPN-NEAT indirect-encoding genome handler for drone evolution."""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

from .base import GenomeHandler
from .cppn.network import (
    ActivationFunction,
    CPPNNetwork,
    ConnectionGene,
    NodeGene,
    NodeType,
)
from .cppn.innovation import InnovationCounter
from .cppn.evaluation import evaluate_cppn
from .cppn.segment_decoder import decode_cppn_to_phenotype
from .cppn.mutations import mutate_cppn
from .operators import SphericalRepairOperator, RepairConfig


# Number of CPPN input nodes (segment_normalized, bias)
_N_INPUTS = 2
# Number of CPPN output nodes
_N_OUTPUTS = 7
# Labels for the output nodes
_OUTPUT_LABELS = [
    "arm_present", "magnitude", "arm_yaw", "arm_pitch",
    "motor_yaw", "motor_pitch", "direction",
]


class CPPNNeatDroneGenomeHandler(GenomeHandler):
    """Genome handler that uses a CPPN-NEAT indirect encoding.

    The genome is a :class:`CPPNNetwork` (directed acyclic graph).  The CPPN
    takes a normalised segment index and a bias as input and produces 7 outputs
    used to decide arm placement and parameters.
    """

    # Shared across the entire population so that structural mutations within
    # the same generation receive matching innovation numbers.
    _innovation_counter: InnovationCounter = InnovationCounter()

    def __init__(
        self,
        genome: Optional[CPPNNetwork] = None,
        num_segments: int = 8,
        min_max_narms: Optional[Tuple[int, int]] = None,
        parameter_limits: Optional[npt.NDArray[Any]] = None,
        # Mutation probabilities
        prob_add_node: float = 0.03,
        prob_add_connection: float = 0.05,
        prob_remove_node: float = 0.01,
        prob_remove_connection: float = 0.02,
        prob_mutate_weights: float = 0.80,
        prob_mutate_activation: float = 0.05,
        prob_toggle_connection: float = 0.02,
        # Initial topology complexity
        initial_hidden_nodes: int = 0,
        # Weight / bias mutation parameters
        weight_perturb_std: float = 0.5,
        weight_replace_prob: float = 0.1,
        weight_range: float = 3.0,
        bias_perturb_std: float = 0.3,
        bias_replace_prob: float = 0.1,
        bias_range: float = 3.0,
        # Repair
        repair: bool = False,
        enable_collision_repair: bool = False,
        propeller_radius: float = 0.0508 / 2,
        inner_boundary_radius: float = 0.0055,
        outer_boundary_radius: float = 0.11,
        max_repair_iterations: int = 100,
        repair_step_size: float = 1.0,
        propeller_tolerance: float = 0.1,
        # RNG
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        # --- Do NOT call super().__init__() ---
        self.rng = rng if rng is not None else np.random.default_rng()

        self.num_segments = num_segments

        if min_max_narms is None:
            self.min_narms, self.max_narms = 6, 6
        else:
            self.min_narms, self.max_narms = min_max_narms

        if parameter_limits is None:
            self.parameter_limits = np.array([
                [0.055, 0.17],           # magnitude
                [-np.pi, np.pi],         # arm yaw (azimuth)
                [-np.pi / 2, np.pi / 2], # arm pitch
                [-np.pi, np.pi],         # motor yaw
                [-np.pi / 2, np.pi / 2], # motor pitch
                [0, 1],                  # direction
            ])
        else:
            self.parameter_limits = np.asarray(parameter_limits)

        # Initial topology
        self.initial_hidden_nodes = initial_hidden_nodes

        # Mutation hyperparameters
        self.prob_add_node = prob_add_node
        self.prob_add_connection = prob_add_connection
        self.prob_remove_node = prob_remove_node
        self.prob_remove_connection = prob_remove_connection
        self.prob_mutate_weights = prob_mutate_weights
        self.prob_mutate_activation = prob_mutate_activation
        self.prob_toggle_connection = prob_toggle_connection
        self.weight_perturb_std = weight_perturb_std
        self.weight_replace_prob = weight_replace_prob
        self.weight_range = weight_range
        self.bias_perturb_std = bias_perturb_std
        self.bias_replace_prob = bias_replace_prob
        self.bias_range = bias_range

        # Repair settings
        self.repair_enabled = repair
        self.enable_collision_repair = enable_collision_repair
        self.propeller_radius = propeller_radius
        self.inner_boundary_radius = inner_boundary_radius
        self.outer_boundary_radius = outer_boundary_radius
        self.max_repair_iterations = max_repair_iterations
        self.repair_step_size = repair_step_size
        self.propeller_tolerance = propeller_tolerance

        self._setup_repair_operator()

        # Genome
        if genome is None:
            self.genome: CPPNNetwork = self._generate_random_genome()
        else:
            self.genome = genome.copy()

    # ------------------------------------------------------------------
    # Repair operator setup
    # ------------------------------------------------------------------

    def _setup_repair_operator(self) -> None:
        repair_config = RepairConfig(
            apply_symmetry=False,
            enable_collision_repair=self.enable_collision_repair,
            propeller_radius=self.propeller_radius,
            inner_boundary_radius=self.inner_boundary_radius,
            outer_boundary_radius=self.outer_boundary_radius,
            max_repair_iterations=self.max_repair_iterations,
            repair_step_size=self.repair_step_size,
            propeller_tolerance=self.propeller_tolerance,
        )
        self.repair_operator = SphericalRepairOperator(
            config=repair_config,
            min_narms=self.min_narms,
            max_narms=self.max_narms,
            parameter_limits=self.parameter_limits,
            symmetry_operator=None,
            rng=self.rng,
        )

    # ------------------------------------------------------------------
    # Random genome generation
    # ------------------------------------------------------------------

    # Activation functions that introduce useful spatial variation
    _HIDDEN_ACTIVATIONS = [
        ActivationFunction.SIN,
        ActivationFunction.COS,
        ActivationFunction.GAUSSIAN,
        ActivationFunction.TANH,
        ActivationFunction.ABS,
    ]

    def _generate_random_genome(self) -> CPPNNetwork:
        """Create a CPPN with random output biases and initial hidden nodes.

        The base topology is fully-connected (2 inputs x 7 outputs).  On top
        of that:

        * Each output node receives a random bias so the network starts with
          varied base values.  The ``arm_present`` output (index 0) gets a
          positive bias to encourage arm placement.
        * ``initial_hidden_nodes`` hidden nodes are inserted by splitting
          random connections, using spatially-interesting activations (SIN,
          COS, GAUSSIAN, …) so there is segment-dependent variation from the
          start.
        """
        net = CPPNNetwork()

        # --- Input nodes ---
        for i in range(_N_INPUTS):
            label = "seg_normalized" if i == 0 else "bias"
            net.nodes[i] = NodeGene(
                node_id=i,
                node_type=NodeType.INPUT,
                activation=ActivationFunction.IDENTITY,
                bias=0.0,
                input_label=label,
            )

        # --- Output nodes (tanh) with random biases ---
        for j in range(_N_OUTPUTS):
            nid = _N_INPUTS + j
            if j == 0:
                # arm_present: positive bias so arms are placed by default
                bias = self.rng.uniform(-self.bias_range, self.bias_range) # uniform(0.5, 1.5)
            else:
                bias = self.rng.uniform(-self.bias_range, self.bias_range)
            net.nodes[nid] = NodeGene(
                node_id=nid,
                node_type=NodeType.OUTPUT,
                activation=ActivationFunction.SIN,
                bias=bias,
                output_index=j,
            )

        net.next_node_id = _N_INPUTS + _N_OUTPUTS

        # --- Fully-connected input→output edges ---
        for i in range(_N_INPUTS):
            for j in range(_N_OUTPUTS):
                tgt = _N_INPUTS + j
                inn = self._innovation_counter.get_innovation(i, tgt)
                net.connections[inn] = ConnectionGene(
                    innovation_number=inn,
                    source_id=i,
                    target_id=tgt,
                    weight=self.rng.uniform(-self.weight_range, self.weight_range),
                    enabled=True,
                )

        # --- Insert initial hidden nodes by splitting random connections ---
        n_hidden = self.rng.integers(0, self.initial_hidden_nodes + 1)
        for _ in range(n_hidden):
            enabled = net.get_enabled_connections()
            if not enabled:
                break
            conn = enabled[self.rng.integers(len(enabled))]
            conn.enabled = False

            new_id = net.next_node_id
            net.next_node_id += 1

            activation = self._HIDDEN_ACTIVATIONS[
                self.rng.integers(len(self._HIDDEN_ACTIVATIONS))
            ]
            net.nodes[new_id] = NodeGene(
                node_id=new_id,
                node_type=NodeType.HIDDEN,
                activation=activation,
                bias=self.rng.uniform(-self.bias_range, self.bias_range),
            )

            inn1 = self._innovation_counter.get_innovation(conn.source_id, new_id)
            inn2 = self._innovation_counter.get_innovation(new_id, conn.target_id)
            net.connections[inn1] = ConnectionGene(
                innovation_number=inn1,
                source_id=conn.source_id,
                target_id=new_id,
                weight=self.rng.uniform(-self.weight_range, self.weight_range),
                enabled=True,
            )
            net.connections[inn2] = ConnectionGene(
                innovation_number=inn2,
                source_id=new_id,
                target_id=conn.target_id,
                weight=conn.weight,
                enabled=True,
            )

        return net

    # ------------------------------------------------------------------
    # Phenotype decoding
    # ------------------------------------------------------------------

    def get_phenotype(self) -> npt.NDArray[Any]:
        """Decode the CPPN into a ``(max_narms, 6)`` phenotype array.

        Applies repair to the decoded phenotype if enabled.
        """
        phenotype = decode_cppn_to_phenotype(
            self.genome,
            num_segments=self.num_segments,
            arm_limit=self.max_narms,
            parameter_limits=self.parameter_limits,
        )
        if self.repair_enabled:
            phenotype = self.repair_operator.repair(phenotype)
        return phenotype

    # ------------------------------------------------------------------
    # GenomeHandler interface
    # ------------------------------------------------------------------

    def generate_random_population(
        self, population_size: int
    ) -> List[CPPNNeatDroneGenomeHandler]:
        population: List[CPPNNeatDroneGenomeHandler] = []
        for _ in range(population_size):
            handler = CPPNNeatDroneGenomeHandler(
                genome=None,
                num_segments=self.num_segments,
                min_max_narms=(self.min_narms, self.max_narms),
                parameter_limits=self.parameter_limits,
                initial_hidden_nodes=self.initial_hidden_nodes,
                prob_add_node=self.prob_add_node,
                prob_add_connection=self.prob_add_connection,
                prob_remove_node=self.prob_remove_node,
                prob_remove_connection=self.prob_remove_connection,
                prob_mutate_weights=self.prob_mutate_weights,
                prob_mutate_activation=self.prob_mutate_activation,
                prob_toggle_connection=self.prob_toggle_connection,
                weight_perturb_std=self.weight_perturb_std,
                weight_replace_prob=self.weight_replace_prob,
                weight_range=self.weight_range,
                bias_perturb_std=self.bias_perturb_std,
                bias_replace_prob=self.bias_replace_prob,
                bias_range=self.bias_range,
                repair=self.repair_enabled,
                enable_collision_repair=self.enable_collision_repair,
                propeller_radius=self.propeller_radius,
                inner_boundary_radius=self.inner_boundary_radius,
                outer_boundary_radius=self.outer_boundary_radius,
                max_repair_iterations=self.max_repair_iterations,
                repair_step_size=self.repair_step_size,
                propeller_tolerance=self.propeller_tolerance,
                rng=self.rng,
            )
            population.append(handler)
        return population

    def crossover(self, other: GenomeHandler) -> CPPNNeatDroneGenomeHandler:
        raise NotImplementedError("CPPN-NEAT crossover not yet implemented")

    def crossover_population(
        self,
        population1: List[GenomeHandler],
        population2: List[GenomeHandler],
    ) -> List[GenomeHandler]:
        """No crossover — return copies of population1."""
        return [p.copy() for p in population1]

    def mutate(self) -> None:
        """Mutate the CPPN genome in-place."""
        mutate_cppn(
            self.genome,
            self._innovation_counter,
            self.rng,
            prob_add_node=self.prob_add_node,
            prob_add_connection=self.prob_add_connection,
            prob_remove_node=self.prob_remove_node,
            prob_remove_connection=self.prob_remove_connection,
            prob_mutate_weights=self.prob_mutate_weights,
            prob_mutate_activation=self.prob_mutate_activation,
            prob_toggle_connection=self.prob_toggle_connection,
            weight_perturb_std=self.weight_perturb_std,
            weight_replace_prob=self.weight_replace_prob,
            weight_range=self.weight_range,
            bias_perturb_std=self.bias_perturb_std,
            bias_replace_prob=self.bias_replace_prob,
            bias_range=self.bias_range,
        )

    def copy(self) -> CPPNNeatDroneGenomeHandler:
        return CPPNNeatDroneGenomeHandler(
            genome=self.genome,
            num_segments=self.num_segments,
            min_max_narms=(self.min_narms, self.max_narms),
            parameter_limits=self.parameter_limits,
            initial_hidden_nodes=self.initial_hidden_nodes,
            prob_add_node=self.prob_add_node,
            prob_add_connection=self.prob_add_connection,
            prob_remove_node=self.prob_remove_node,
            prob_remove_connection=self.prob_remove_connection,
            prob_mutate_weights=self.prob_mutate_weights,
            prob_mutate_activation=self.prob_mutate_activation,
            prob_toggle_connection=self.prob_toggle_connection,
            weight_perturb_std=self.weight_perturb_std,
            weight_replace_prob=self.weight_replace_prob,
            weight_range=self.weight_range,
            bias_perturb_std=self.bias_perturb_std,
            bias_replace_prob=self.bias_replace_prob,
            bias_range=self.bias_range,
            repair=self.repair_enabled,
            enable_collision_repair=self.enable_collision_repair,
            propeller_radius=self.propeller_radius,
            inner_boundary_radius=self.inner_boundary_radius,
            outer_boundary_radius=self.outer_boundary_radius,
            max_repair_iterations=self.max_repair_iterations,
            repair_step_size=self.repair_step_size,
            propeller_tolerance=self.propeller_tolerance,
            rng=self.rng,
        )

    def is_valid(self) -> bool:
        """Decode the CPPN to phenotype and check validity."""
        phenotype = self.get_phenotype()
        arm_count = int(np.sum(~np.isnan(phenotype[:, 0])))
        return self.min_narms <= arm_count <= self.max_narms

    def repair(self) -> None:
        """No-op — repair is applied to the decoded phenotype in get_phenotype()."""
        pass
