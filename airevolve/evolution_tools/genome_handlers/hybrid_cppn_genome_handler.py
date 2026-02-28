"""Hybrid direct + CPPN genome handler for drone evolution.

Arm geometry (magnitude, yaw, pitch) is directly encoded while motor
parameters (motor_yaw, motor_pitch, spin_direction) are produced by a CPPN
conditioned on each arm's geometry.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
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
from .cppn.mutations import mutate_cppn
from .operators import SphericalRepairOperator, RepairConfig


# CPPN topology constants
_N_CPPN_INPUTS = 4   # norm_magnitude, norm_arm_yaw, norm_arm_pitch, bias
_N_CPPN_OUTPUTS = 3  # motor_yaw, motor_pitch, direction
_INPUT_LABELS = ["norm_magnitude", "norm_arm_yaw", "norm_arm_pitch", "bias"]
_OUTPUT_LABELS = ["motor_yaw", "motor_pitch", "direction"]


@dataclass
class HybridGenome:
    """Two-part genome: direct arm parameters + CPPN for motor parameters."""

    direct: npt.NDArray[Any]  # shape (narms, 3): magnitude, arm_yaw, arm_pitch
    cppn: CPPNNetwork

    def copy(self) -> HybridGenome:
        return HybridGenome(
            direct=self.direct.copy(),
            cppn=self.cppn.copy(),
        )


class HybridCPPNDroneGenomeHandler(GenomeHandler):
    """Hybrid genome handler: direct arm geometry + CPPN motor parameters.

    The genome consists of two parts:

    1. **Direct encoding** — ``(narms, 3)`` array of ``[magnitude, arm_yaw,
       arm_pitch]`` per arm.  All rows are always active (fixed arm count).
    2. **CPPN** — A ``CPPNNetwork`` with 4 inputs (normalised arm params +
       bias) and 3 tanh outputs mapped to motor parameter ranges.

    The phenotype is the standard ``(narms, 6)`` array where columns 0–2 come
    from the direct encoding and columns 3–5 from CPPN evaluation.
    """

    # Shared innovation counter across the population
    _innovation_counter: InnovationCounter = InnovationCounter()

    # Activation functions for hidden nodes
    _HIDDEN_ACTIVATIONS = [
        ActivationFunction.SIN,
        ActivationFunction.COS,
        ActivationFunction.GAUSSIAN,
        ActivationFunction.TANH,
        ActivationFunction.ABS,
    ]

    def __init__(
        self,
        genome: Optional[HybridGenome] = None,
        min_max_narms: Optional[Tuple[int, int]] = None,
        parameter_limits: Optional[npt.NDArray[Any]] = None,
        # Direct mutation parameters
        prob_mutate_direct: float = 0.5,
        direct_mutation_scale_pct: float = 0.05,
        # CPPN mutation probabilities
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

        # Direct mutation hyperparameters
        self.prob_mutate_direct = prob_mutate_direct
        self.direct_mutation_scale_pct = direct_mutation_scale_pct

        # Initial topology
        self.initial_hidden_nodes = initial_hidden_nodes

        # CPPN mutation hyperparameters
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
            self.genome: HybridGenome = self._generate_random_genome()
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

    def _generate_random_genome(self) -> HybridGenome:
        """Create a random hybrid genome.

        Direct part: uniform sampling for magnitude/yaw, arcsin sampling for
        pitch (uniform coverage on the sphere), binary direction.

        CPPN part: fully-connected 4→3 network with optional initial hidden
        nodes.
        """
        narms = self.rng.integers(self.min_narms, self.max_narms + 1)
        direct = np.empty((narms, 3))

        mag_lo, mag_hi = self.parameter_limits[0]
        yaw_lo, yaw_hi = self.parameter_limits[1]
        pitch_lo, pitch_hi = self.parameter_limits[2]

        direct[:, 0] = self.rng.uniform(mag_lo, mag_hi, size=narms)
        direct[:, 1] = self.rng.uniform(yaw_lo, yaw_hi, size=narms)
        # Arcsin sampling for uniform sphere coverage
        direct[:, 2] = np.arcsin(
            self.rng.uniform(
                np.sin(pitch_lo), np.sin(pitch_hi), size=narms
            )
        )

        cppn = self._create_initial_cppn()
        return HybridGenome(direct=direct, cppn=cppn)

    def _create_initial_cppn(self) -> CPPNNetwork:
        """Build a 4-input, 3-output CPPN with optional initial hidden nodes."""
        net = CPPNNetwork()

        # --- Input nodes ---
        for i in range(_N_CPPN_INPUTS):
            net.nodes[i] = NodeGene(
                node_id=i,
                node_type=NodeType.INPUT,
                activation=ActivationFunction.IDENTITY,
                bias=0.0,
                input_label=_INPUT_LABELS[i],
            )

        # --- Output nodes (tanh) with random biases ---
        for j in range(_N_CPPN_OUTPUTS):
            nid = _N_CPPN_INPUTS + j
            net.nodes[nid] = NodeGene(
                node_id=nid,
                node_type=NodeType.OUTPUT,
                activation=ActivationFunction.TANH,
                bias=self.rng.uniform(-self.bias_range, self.bias_range),
                output_index=j,
            )

        net.next_node_id = _N_CPPN_INPUTS + _N_CPPN_OUTPUTS

        # --- Fully-connected input→output edges ---
        for i in range(_N_CPPN_INPUTS):
            for j in range(_N_CPPN_OUTPUTS):
                tgt = _N_CPPN_INPUTS + j
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
        """Decode the hybrid genome into a ``(narms, 6)`` phenotype array.

        Columns 0–2 come from the direct encoding.  Columns 3–5 are produced
        by evaluating the CPPN with normalised arm parameters as input and
        mapping the tanh outputs to the motor parameter ranges.
        """
        narms = self.genome.direct.shape[0]
        phenotype = np.empty((narms, 6))

        # Columns 0–2: direct arm geometry
        phenotype[:, :3] = self.genome.direct

        # Normalise direct params to [-1, 1] for CPPN input
        limits = self.parameter_limits[:3]  # (3, 2) for magnitude, yaw, pitch
        lo = limits[:, 0]  # (3,)
        hi = limits[:, 1]  # (3,)
        ranges = hi - lo
        # Avoid division by zero for degenerate limits
        ranges = np.where(ranges == 0, 1.0, ranges)
        normalised = 2.0 * (self.genome.direct - lo) / ranges - 1.0  # (narms, 3)

        # Build CPPN input: [norm_mag, norm_yaw, norm_pitch, bias=1.0]
        cppn_input = np.column_stack([
            normalised,
            np.ones(narms),
        ])  # (narms, 4)

        # Evaluate CPPN (batch mode)
        cppn_output = evaluate_cppn(self.genome.cppn, cppn_input)  # (narms, 3)

        # Map tanh outputs [-1, 1] to motor parameter ranges
        motor_limits = self.parameter_limits[3:]  # (3, 2)
        motor_lo = motor_limits[:, 0]
        motor_hi = motor_limits[:, 1]
        # tanh output in [-1, 1] → [lo, hi]
        phenotype[:, 3:5] = (
            motor_lo[:2]
            + (cppn_output[:, :2] + 1.0) * 0.5 * (motor_hi[:2] - motor_lo[:2])
        )
        # Direction: threshold at 0
        phenotype[:, 5] = np.where(cppn_output[:, 2] >= 0.0, 1.0, 0.0)

        if self.repair_enabled:
            phenotype = self.repair_operator.repair(phenotype)

        return phenotype

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def mutate(self) -> None:
        """Mutate either the direct or CPPN part of the genome."""
        if self.rng.random() < self.prob_mutate_direct:
            self._mutate_direct()
        else:
            self._mutate_cppn()

    def _mutate_direct(self) -> None:
        """Gaussian perturbation of one parameter of one random arm."""
        narms = self.genome.direct.shape[0]
        arm_idx = self.rng.integers(narms)
        param_idx = self.rng.integers(3)  # 0=magnitude, 1=yaw, 2=pitch

        lo, hi = self.parameter_limits[param_idx]
        scale = (hi - lo) * self.direct_mutation_scale_pct
        perturbation = self.rng.normal(0, scale)

        new_val = self.genome.direct[arm_idx, param_idx] + perturbation

        if param_idx in (1, 2):
            # Angle wrapping to [lo, hi] using modular arithmetic
            span = hi - lo
            new_val = lo + (new_val - lo) % span
        else:
            new_val = np.clip(new_val, lo, hi)

        self.genome.direct[arm_idx, param_idx] = new_val

    def _mutate_cppn(self) -> None:
        """Delegate to NEAT-style CPPN mutation."""
        mutate_cppn(
            self.genome.cppn,
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

    # ------------------------------------------------------------------
    # GenomeHandler interface
    # ------------------------------------------------------------------

    def generate_random_population(
        self, population_size: int
    ) -> List[HybridCPPNDroneGenomeHandler]:
        population: List[HybridCPPNDroneGenomeHandler] = []
        for _ in range(population_size):
            handler = HybridCPPNDroneGenomeHandler(
                genome=None,
                min_max_narms=(self.min_narms, self.max_narms),
                parameter_limits=self.parameter_limits,
                prob_mutate_direct=self.prob_mutate_direct,
                direct_mutation_scale_pct=self.direct_mutation_scale_pct,
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

    def crossover(self, other: GenomeHandler) -> HybridCPPNDroneGenomeHandler:
        raise NotImplementedError("Hybrid CPPN crossover not yet implemented")

    def crossover_population(
        self,
        population1: List[GenomeHandler],
        population2: List[GenomeHandler],
    ) -> List[GenomeHandler]:
        """No crossover — return copies of population1."""
        return [p.copy() for p in population1]

    def copy(self) -> HybridCPPNDroneGenomeHandler:
        return HybridCPPNDroneGenomeHandler(
            genome=self.genome,
            min_max_narms=(self.min_narms, self.max_narms),
            parameter_limits=self.parameter_limits,
            prob_mutate_direct=self.prob_mutate_direct,
            direct_mutation_scale_pct=self.direct_mutation_scale_pct,
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
        """Check that the arm count falls within bounds."""
        narms = self.genome.direct.shape[0]
        return self.min_narms <= narms <= self.max_narms

    def repair(self) -> None:
        """No-op — repair is applied to the decoded phenotype in get_phenotype()."""
        pass
