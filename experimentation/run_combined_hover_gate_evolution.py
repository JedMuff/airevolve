"""
Combined Hover + Gate Fitness Evolution

Fitness pipeline per individual:
  1. Decode CPPN → phenotype (if indirect encoding)
  2. stage1_optimization_repair (fix collisions)
  3. Compute continuous hover_fitness on optimization-repaired phenotype [0,3]
  4. stage2_hover_check (can it hover?)
     - Fails → return hover_fitness (gradient signal)
     - Passes → full repair from original phenotype, then CMA-ES gate eval
  5. Return hover_fitness + gates_passed

Non-hoverable drones get gradient signal in [0, 3].
Hoverable drones get hover_fitness + gates_passed.
Initial population: empty CPPNs (input/output nodes only, no connections).

Usage:
    python experimentation/run_combined_hover_gate_evolution.py \\
        --population-size 20 \\
        --generations 10 \\
        --max-evals 100 \\
        --gate-cfg circle
"""

import sys
import os

# Limit BLAS/OpenMP threads to 1 per process — must be set before numpy import.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import json
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from numpy.linalg import norm, eig

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from airevolve.evolution_tools.evaluators.lee_tune_evaluator import (
    evaluate_individual_with_tuning,
)
from airevolve.evolution_tools.strategies.mu_lambda import evolve
from airevolve.evolution_tools.selectors.tournament import tournament_selection
from airevolve.evolution_tools.inspection_tools.utils import evolution_dataframe_to_fitness_array
from airevolve.evolution_tools.inspection_tools.plot_fitness import plot_fitness
from airevolve.evolution_tools.genome_handlers.repair_workflow import (
    stage1_optimization_repair,
    stage2_hover_check,
    stage3_hover_repair,
)
from airevolve.evolution_tools.genome_handlers.operators.optimization_repair_operator import (
    OptimizationRepairConfig,
)
from airevolve.evolution_tools.genome_handlers.cppn.network import (
    CPPNNetwork, ConnectionGene, NodeGene, NodeType, ActivationFunction,
)
from airevolve.evolution_tools.genome_handlers.cppn.innovation import InnovationCounter
from airevolve.evolution_tools.genome_handlers.hybrid_cppn_genome_handler import (
    HybridGenome,
    _N_CPPN_INPUTS as _N_HYBRID_CPPN_INPUTS,
    _N_CPPN_OUTPUTS as _N_HYBRID_CPPN_OUTPUTS,
    _INPUT_LABELS as _HYBRID_INPUT_LABELS,
    _OUTPUT_LABELS as _HYBRID_OUTPUT_LABELS,
)
from airevolve.evolution_tools.inspection_tools.drone_visualizer import DroneVisualizer, VisualizationConfig
from experimentation.plot_genome_visualization import (
    draw_cppn_graph, draw_phenotype_heatmap,
    SPHERICAL_COLUMNS, HYBRID_DIRECT_COLUMNS,
)

from examples.run_evolution_with_lee_tuning import (
    get_genome_handler_config,
    create_genome_handler_wrapper,
)


# ============================================================================
# EMPTY CPPN GENERATION
# ============================================================================

_N_INPUTS = 2   # seg_normalized, bias
_N_OUTPUTS = 7  # arm_present + 6 arm parameters


def create_empty_cppn():
    """Create a CPPN with input/output nodes but no connections."""
    net = CPPNNetwork()

    # Input nodes
    for i in range(_N_INPUTS):
        label = "seg_normalized" if i == 0 else "bias"
        net.nodes[i] = NodeGene(
            node_id=i,
            node_type=NodeType.INPUT,
            activation=ActivationFunction.IDENTITY,
            bias=0.0,
            input_label=label,
        )

    # Output nodes with zero bias (no connections → outputs are just bias)
    for j in range(_N_OUTPUTS):
        nid = _N_INPUTS + j
        net.nodes[nid] = NodeGene(
            node_id=nid,
            node_type=NodeType.OUTPUT,
            activation=ActivationFunction.SIN,
            bias=0.0,
            output_index=j,
        )

    net.next_node_id = _N_INPUTS + _N_OUTPUTS
    return net


# Activation functions for seeded hidden nodes
_SEED_ACTIVATIONS = [
    ActivationFunction.SIGMOID,
    ActivationFunction.TANH,
    ActivationFunction.GAUSSIAN,
]

# Shared innovation counter for seeded initial topologies so that identical
# structural choices across the initial population get the same innovation
# numbers (consistent with NEAT's homology tracking).
_seed_innovation_counter = InnovationCounter()


def _seed_cppn_topology(
    net: CPPNNetwork,
    n_inputs: int,
    n_outputs: int,
    rng: np.random.Generator,
) -> None:
    """Add 2–5 hidden nodes and ~10–20 random feed-forward connections to a CPPN.

    Modifies *net* in place.  Assumes input node IDs are 0..n_inputs-1 and
    output node IDs are n_inputs..n_inputs+n_outputs-1.
    """
    n_hidden = int(rng.integers(2, 6))  # 2–5 hidden nodes

    input_ids = list(range(n_inputs))
    output_ids = list(range(n_inputs, n_inputs + n_outputs))
    hidden_ids = []

    for _ in range(n_hidden):
        nid = net.next_node_id
        net.next_node_id += 1
        activation = rng.choice(_SEED_ACTIVATIONS)
        net.nodes[nid] = NodeGene(
            node_id=nid,
            node_type=NodeType.HIDDEN,
            activation=activation,
            bias=float(rng.uniform(-1.0, 1.0)),
        )
        hidden_ids.append(nid)

    # Build pool of valid feed-forward connections (no output→anything,
    # no anything→input, no hidden→hidden-with-lower-id to keep it acyclic).
    # Layers: input(0) → hidden(1) → output(2)
    possible = []
    for src in input_ids:
        for tgt in hidden_ids + output_ids:
            possible.append((src, tgt))
    for src in hidden_ids:
        for tgt in output_ids:
            possible.append((src, tgt))
    # Allow connections between hidden nodes (higher id only, keeps DAG)
    for i, src in enumerate(hidden_ids):
        for tgt in hidden_ids[i + 1:]:
            possible.append((src, tgt))

    # Sample 10–20 connections (clamped to available)
    n_target = int(rng.integers(10, 21))
    n_conns = min(n_target, len(possible))
    chosen_indices = rng.choice(len(possible), size=n_conns, replace=False)

    for idx in chosen_indices:
        src, tgt = possible[idx]
        inn = _seed_innovation_counter.get_innovation(src, tgt)
        net.connections[inn] = ConnectionGene(
            innovation_number=inn,
            source_id=src,
            target_id=tgt,
            weight=float(rng.uniform(-1.0, 1.0)),
            enabled=True,
        )


def create_seeded_cppn(rng: np.random.Generator | None = None) -> CPPNNetwork:
    """Create a CPPN with 2–5 hidden nodes and ~10–20 random connections."""
    if rng is None:
        rng = np.random.default_rng()
    net = create_empty_cppn()
    _seed_cppn_topology(net, _N_INPUTS, _N_OUTPUTS, rng)
    return net


def create_seeded_hybrid_genome(
    narms: int = 6,
    rng: np.random.Generator | None = None,
) -> HybridGenome:
    """Create a HybridGenome with random direct params and a seeded CPPN."""
    if rng is None:
        rng = np.random.default_rng()
    direct = np.empty((narms, 3))
    direct[:, 0] = rng.uniform(0.055, 0.17, size=narms)
    direct[:, 1] = rng.uniform(-np.pi, np.pi, size=narms)
    direct[:, 2] = np.arcsin(rng.uniform(-1.0, 1.0, size=narms))

    # Build seeded CPPN
    net = CPPNNetwork()
    for i in range(_N_HYBRID_CPPN_INPUTS):
        net.nodes[i] = NodeGene(
            node_id=i,
            node_type=NodeType.INPUT,
            activation=ActivationFunction.IDENTITY,
            bias=0.0,
            input_label=_HYBRID_INPUT_LABELS[i],
        )
    for j in range(_N_HYBRID_CPPN_OUTPUTS):
        nid = _N_HYBRID_CPPN_INPUTS + j
        net.nodes[nid] = NodeGene(
            node_id=nid,
            node_type=NodeType.OUTPUT,
            activation=ActivationFunction.TANH,
            bias=0.0,
            output_index=j,
        )
    net.next_node_id = _N_HYBRID_CPPN_INPUTS + _N_HYBRID_CPPN_OUTPUTS
    _seed_cppn_topology(net, _N_HYBRID_CPPN_INPUTS, _N_HYBRID_CPPN_OUTPUTS, rng)

    return HybridGenome(direct=direct, cppn=net)


def create_empty_hybrid_genome(narms=6):
    """Create a HybridGenome with random direct params and an empty CPPN."""
    # Random direct arm geometry: magnitude, yaw, pitch
    rng = np.random.default_rng()
    direct = np.empty((narms, 3))
    direct[:, 0] = rng.uniform(0.055, 0.17, size=narms)      # magnitude
    direct[:, 1] = rng.uniform(-np.pi, np.pi, size=narms)     # arm yaw
    direct[:, 2] = np.arcsin(rng.uniform(-1.0, 1.0, size=narms))  # arm pitch (arcsin sampling)

    # Empty CPPN with 4 inputs, 3 outputs, no connections
    net = CPPNNetwork()
    for i in range(_N_HYBRID_CPPN_INPUTS):
        net.nodes[i] = NodeGene(
            node_id=i,
            node_type=NodeType.INPUT,
            activation=ActivationFunction.IDENTITY,
            bias=0.0,
            input_label=_HYBRID_INPUT_LABELS[i],
        )
    for j in range(_N_HYBRID_CPPN_OUTPUTS):
        nid = _N_HYBRID_CPPN_INPUTS + j
        net.nodes[nid] = NodeGene(
            node_id=nid,
            node_type=NodeType.OUTPUT,
            activation=ActivationFunction.TANH,
            bias=0.0,
            output_index=j,
        )
    net.next_node_id = _N_HYBRID_CPPN_INPUTS + _N_HYBRID_CPPN_OUTPUTS

    return HybridGenome(direct=direct, cppn=net)


# ============================================================================
# CONTINUOUS HOVER FITNESS
# ============================================================================

G = 9.81
C_AUTHORITY = 300.0  # sqrt(lambda_min) half-saturation constant


def continuous_hover_fitness(phenotype):
    """Compute continuous hover fitness for a decoded phenotype.

    Returns:
        Fitness score in [0, 3]. Higher = closer to hoverable.
    """
    from airevolve.evolution_tools.inspection_tools.morphological_descriptors.hovering_info import get_sim

    sim = get_sim(phenotype)
    if sim is None:
        return 0.0

    Bf = sim.Bf  # (3, n_props)
    Bm = sim.Bm  # (3, n_props)

    # Term 1: Rank feasibility [0, 1]
    rank_f = np.linalg.matrix_rank(Bf)
    rank_m = np.linalg.matrix_rank(Bm)
    f_rank = (rank_f + rank_m) / 6.0

    # Term 2: Force capability [0, 1]
    n_props = Bf.shape[1]
    eta_max = np.ones(n_props)
    f_vec = Bf @ eta_max
    thrust_ratio = norm(f_vec) / G
    f_force = min(thrust_ratio, 2.0) / 2.0

    # Term 3: Torque balance [0, 1]
    gram_m = Bm @ Bm.T
    eigs = np.real(eig(gram_m)[0])
    eigs = np.maximum(eigs, 0.0)

    lambda_min = np.min(eigs)
    lambda_max = np.max(eigs)

    condition = lambda_min / (lambda_max + 1e-12)
    authority = np.sqrt(lambda_min)
    authority_sat = authority / (authority + C_AUTHORITY)
    f_torque = condition * authority_sat

    return float(f_rank + f_force + f_torque)


# ============================================================================
# COMBINED FITNESS CLASS
# ============================================================================

class _CombinedHoverGateFitness:
    """Picklable fitness wrapper: continuous hover fitness + CMA-ES gate evaluation.

    Pipeline:
      1. Decode genome to phenotype (if indirect)
      2. stage1_optimization_repair → opt_repaired
      3. hover_fit = continuous_hover_fitness(opt_repaired)
      4. stage2_hover_check(opt_repaired)
         - Fails → return hover_fit
         - Passes → full repair from original phenotype, then CMA-ES gate eval
      5. Return hover_fit + gates_passed
    """

    def __init__(self, gate_cfg, max_evals, cma_workers, sim_time, dt, timeout,
                 coordinate_system, is_indirect=False, handler_class=None,
                 handler_kwargs=None):
        self.gate_cfg = gate_cfg
        self.max_evals = max_evals
        self.cma_workers = cma_workers
        self.sim_time = sim_time
        self.dt = dt
        self.timeout = timeout
        self.coordinate_system = coordinate_system
        self.is_indirect = is_indirect
        self.handler_class = handler_class
        self.handler_kwargs = handler_kwargs

    def __call__(self, genome, ind_save_dir):
        # Save genome immediately
        if ind_save_dir is not None:
            os.makedirs(ind_save_dir, exist_ok=True)
            if self.is_indirect:
                with open(os.path.join(ind_save_dir, "genotype.pkl"), 'wb') as f:
                    pickle.dump(genome, f)
            else:
                np.save(os.path.join(ind_save_dir, "genome.npy"), genome)

        # Decode indirect encoding to phenotype if needed
        if self.is_indirect:
            handler = self.handler_class(genome=genome, **self.handler_kwargs)
            phenotype = handler.get_phenotype()
            repair_coord = 'spherical'
        else:
            phenotype = genome
            repair_coord = self.coordinate_system

        # Save phenotype and visualization for ALL individuals
        self._save_phenotype_and_viz(genome, phenotype, ind_save_dir)

        # Stage 1: Optimization repair on phenotype (fix collisions)
        repair_config = OptimizationRepairConfig(fixed_params=[3, 4])
        opt_repaired, _ = stage1_optimization_repair(
            phenotype, coordinate_system=repair_coord,
            config=repair_config, verbose=False
        )

        if opt_repaired is None:
            # Optimization repair failed — compute hover fitness on raw phenotype
            hover_fit = continuous_hover_fitness(phenotype)
            self._save_breakdown(ind_save_dir, hover_fit, 0, "failed_stage1_opt_repair")
            return hover_fit

        # Compute continuous hover fitness on optimization-repaired phenotype
        hover_fit = continuous_hover_fitness(opt_repaired)

        # Stage 2: Hover check on optimization-repaired phenotype
        can_hover, _ = stage2_hover_check(
            opt_repaired, verbose=False, allow_spinning=False
        )

        if not can_hover:
            self._save_breakdown(ind_save_dir, hover_fit, 0, "failed_hover_check")
            return hover_fit

        # Hover check passed — run full repair pipeline from ORIGINAL phenotype
        # Stage 2 (hover check) on original
        can_hover_orig, _ = stage2_hover_check(
            phenotype, verbose=False, allow_spinning=False
        )
        if not can_hover_orig:
            self._save_breakdown(ind_save_dir, hover_fit, 0, "failed_hover_check_original")
            return hover_fit

        # Stage 1 (optimization repair) on original
        repaired, _ = stage1_optimization_repair(
            phenotype, coordinate_system=repair_coord,
            config=repair_config, verbose=False
        )
        if repaired is None:
            self._save_breakdown(ind_save_dir, hover_fit, 0, "failed_opt_repair_original")
            return hover_fit

        # Stage 3 (hover repair) on repaired
        final, _ = stage3_hover_repair(
            repaired, coordinate_system=repair_coord, verbose=False
        )
        if final is None:
            self._save_breakdown(ind_save_dir, hover_fit, 0, "failed_hover_repair")
            return hover_fit

        # CMA-ES gate evaluation on fully repaired phenotype
        gates_passed = evaluate_individual_with_tuning(
            final, ind_save_dir,
            gate_cfg=self.gate_cfg,
            max_evals=self.max_evals,
            num_workers=self.cma_workers,
            sim_time=self.sim_time,
            dt=self.dt,
            timeout=self.timeout,
            num=None,
        )

        self._save_breakdown(ind_save_dir, hover_fit, gates_passed, "full_evaluation")
        return hover_fit + gates_passed

    def _save_phenotype_and_viz(self, genome, phenotype, ind_save_dir):
        """Save phenotype .npy and a visualization figure for this individual."""
        if ind_save_dir is None:
            return
        try:
            os.makedirs(ind_save_dir, exist_ok=True)
            np.save(os.path.join(ind_save_dir, "phenotype.npy"), phenotype)

            # Determine genome type and create appropriate figure layout
            if isinstance(genome, HybridGenome):
                fig = plt.figure(figsize=(18, 6))
                ax_drone = fig.add_subplot(1, 3, 1, projection='3d')
                ax_direct = fig.add_subplot(1, 3, 2)
                ax_cppn = fig.add_subplot(1, 3, 3)

                viz = DroneVisualizer(VisualizationConfig(
                    elevation=30, azimuth=45, show_axis_ticks=False))
                viz.plot_3d(phenotype, ax=ax_drone, title='Drone Blueprint')
                draw_phenotype_heatmap(genome.direct, ax_direct,
                                       columns=HYBRID_DIRECT_COLUMNS,
                                       title='Direct Parameters')
                draw_cppn_graph(genome.cppn, ax_cppn)

            elif isinstance(genome, CPPNNetwork):
                fig = plt.figure(figsize=(14, 6))
                ax_drone = fig.add_subplot(1, 2, 1, projection='3d')
                ax_cppn = fig.add_subplot(1, 2, 2)

                viz = DroneVisualizer(VisualizationConfig(
                    elevation=30, azimuth=45, show_axis_ticks=False))
                viz.plot_3d(phenotype, ax=ax_drone, title='Drone Blueprint')
                draw_cppn_graph(genome, ax_cppn)

            else:
                # Direct encoding
                fig = plt.figure(figsize=(14, 6))
                ax_drone = fig.add_subplot(1, 2, 1, projection='3d')
                ax_heatmap = fig.add_subplot(1, 2, 2)

                viz = DroneVisualizer(VisualizationConfig(
                    elevation=30, azimuth=45, show_axis_ticks=False))
                viz.plot_3d(phenotype, ax=ax_drone, title='Drone Blueprint')
                draw_phenotype_heatmap(phenotype, ax_heatmap,
                                       columns=SPHERICAL_COLUMNS,
                                       title='Phenotype')

            fig.tight_layout()
            fig.savefig(os.path.join(ind_save_dir, "visualization.png"),
                        dpi=150, bbox_inches='tight')
            plt.close(fig)
        except Exception:
            pass

    def _save_breakdown(self, ind_save_dir, hover_fit, gates_passed, status):
        """Save hover fitness breakdown to individual's save directory."""
        if ind_save_dir is None:
            return
        breakdown = {
            "hover_fitness": hover_fit,
            "gates_passed": gates_passed,
            "total_fitness": hover_fit + gates_passed,
            "status": status,
        }
        try:
            path = os.path.join(ind_save_dir, "hover_breakdown.json")
            with open(path, 'w') as f:
                json.dump(breakdown, f, indent=2)
        except Exception:
            pass


# ============================================================================
# ARGUMENT PARSING
# ============================================================================

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run evolution with combined hover + gate fitness'
    )

    # Genome and evolution parameters
    parser.add_argument('--genome-handler', choices=['spherical', 'cartesian', 'cppn', 'hybrid-cppn'],
                       default='cppn', help='Genome handler to use (default: cppn)')
    parser.add_argument('--population-size', type=int, default=16,
                       help='Population size (default: 16)')
    parser.add_argument('--generations', type=int, default=50,
                       help='Number of generations (default: 50)')
    parser.add_argument('--num-mutate', type=int, default=16,
                       help='Number of individuals to mutate per generation (default: 16)')
    parser.add_argument('--num-crossover', type=int, default=0,
                       help='Number of crossover operations per generation (default: 0)')
    parser.add_argument('--log-dir', default='./.data',
                       help='Directory for logs (default: ./.data)')
    parser.add_argument('--show-plot', action='store_true',
                       help='Show fitness plot at the end')
    parser.add_argument('--strategy-type', choices=['plus', 'comma'],
                       default='plus', help='Evolution strategy type (default: plus)')

    # Lee controller tuning parameters (2-Stage CMA-ES)
    parser.add_argument('--max-evals', type=int, default=500,
                       help='Maximum CMA-ES evaluations per individual (default: 500)')
    parser.add_argument('--cma-workers', type=int, default=1,
                       help='Number of parallel workers for CMA-ES (default: 1)')
    parser.add_argument('--sim-time', type=float, default=20.0,
                       help='Simulation time in seconds (default: 20.0)')
    parser.add_argument('--dt', type=float, default=0.005,
                       help='Time step in seconds (default: 0.005)')
    parser.add_argument('--timeout', type=float, default=30.0,
                       help='Timeout per evaluation in seconds (default: 30.0)')
    parser.add_argument('--gate-cfg', choices=['backandforth', 'figure8', 'circle', 'slalom'],
                       default='figure8', help='Gate configuration (default: figure8)')

    # Evolution workers
    parser.add_argument('--num-workers', type=int, default=32,
                       help='Number of parallel workers for evolution (default: 32)')

    # Morphology parameters
    parser.add_argument('--min-narms', type=int, default=6,
                       help='Minimum number of arms (default: 6)')
    parser.add_argument('--max-narms', type=int, default=6,
                       help='Maximum number of arms (default: 6)')

    # CPPN-specific parameters
    parser.add_argument('--num-segments', type=int, default=8,
                       help='Number of CPPN evaluation segments (default: 8, CPPN only)')
    parser.add_argument('--initial-hidden-nodes', type=int, default=0,
                       help='Initial hidden nodes in CPPN topology (default: 0, CPPN only)')
    parser.add_argument('--init-topology', choices=['empty', 'seeded'], default='empty',
                       help='Initial CPPN topology: empty (no connections) or seeded '
                            '(2-5 hidden nodes, ~10-20 connections with sigmoid/tanh/gaussian). '
                            'Default: empty. Only affects cppn and hybrid-cppn handlers.')

    return parser.parse_args()


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main evolution function with combined hover + gate fitness."""
    args = parse_arguments()

    # Generate automatic experiment name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    narms_str = f"{args.min_narms}arms" if args.min_narms == args.max_narms else f"{args.min_narms}-{args.max_narms}arms"
    exp_name = f"combined_hover_gate_{args.gate_cfg}_{narms_str}_{timestamp}"

    full_log_dir = os.path.join(args.log_dir, exp_name)
    os.makedirs(full_log_dir, exist_ok=True)

    print("=" * 80)
    print("Combined Hover + Gate Fitness Evolution")
    print("=" * 80)
    print(f"Experiment: {exp_name}")
    print(f"Log dir: {full_log_dir}")
    print(f"Handler: {args.genome_handler.upper()}, Arms: {args.min_narms}-{args.max_narms}")
    print(f"Pop: {args.population_size}, Gens: {args.generations}, "
          f"Mutate: {args.num_mutate}, Strategy: {args.strategy_type}")
    print(f"Gate: {args.gate_cfg}, CMA-ES evals: {args.max_evals}, Workers: {args.num_workers}")
    print(f"Init topology: {args.init_topology}")
    print("Fitness = hover_fitness [0,3] + gates_passed [0,N]")
    print("=" * 80)
    print()

    # Get genome handler configuration
    config = get_genome_handler_config(
        args.genome_handler, args.min_narms, args.max_narms,
        num_segments=args.num_segments,
        initial_hidden_nodes=args.initial_hidden_nodes,
        init_topology=args.init_topology,
    )

    # Create combined fitness function
    is_indirect = args.genome_handler in ('cppn', 'hybrid-cppn')
    fitness_function = _CombinedHoverGateFitness(
        gate_cfg=args.gate_cfg,
        max_evals=args.max_evals,
        cma_workers=args.cma_workers,
        sim_time=args.sim_time,
        dt=args.dt,
        timeout=args.timeout,
        coordinate_system=config['coordinate_system'],
        is_indirect=is_indirect,
        handler_class=config['handler_class'] if is_indirect else None,
        handler_kwargs=config['handler_kwargs'] if is_indirect else None,
    )

    # Create genome handler wrapper
    WrappedHandler = create_genome_handler_wrapper(config['handler_class'], config['handler_kwargs'])

    # Generate initial population of CPPNs / hybrid genomes
    use_seeded = args.init_topology == 'seeded'
    if args.genome_handler == 'hybrid-cppn':
        if use_seeded:
            initial_population = [create_seeded_hybrid_genome(narms=args.min_narms)
                                  for _ in range(args.population_size)]
            print(f"Generated {len(initial_population)} seeded hybrid genomes "
                  f"({args.min_narms} arms, 2-5 hidden nodes, ~10-20 connections)")
        else:
            initial_population = [create_empty_hybrid_genome(narms=args.min_narms)
                                  for _ in range(args.population_size)]
            print(f"Generated {len(initial_population)} empty hybrid genomes "
                  f"({args.min_narms} arms, {_N_HYBRID_CPPN_INPUTS} CPPN inputs, "
                  f"{_N_HYBRID_CPPN_OUTPUTS} CPPN outputs, 0 connections)")
    elif args.genome_handler == 'cppn':
        if use_seeded:
            initial_population = [create_seeded_cppn() for _ in range(args.population_size)]
            print(f"Generated {len(initial_population)} seeded CPPNs "
                  f"({_N_INPUTS} inputs, {_N_OUTPUTS} outputs, 2-5 hidden, ~10-20 conns)")
        else:
            initial_population = [create_empty_cppn() for _ in range(args.population_size)]
            print(f"Generated {len(initial_population)} empty CPPNs "
                  f"({_N_INPUTS} inputs, {_N_OUTPUTS} outputs, 0 connections)")
    else:
        # For direct encodings, just use random genomes
        handler = WrappedHandler()
        initial_population = handler.generate_random_population(args.population_size)
        initial_population = np.array([h.genome if hasattr(h, 'genome') else h
                                       for h in initial_population])
        print(f"Generated {len(initial_population)} random individuals")

    # Run evolution
    print()
    all_individuals = evolve(
        fitness_function=fitness_function,
        population_size=args.population_size,
        num_generations=args.generations,
        num_mutate=args.num_mutate,
        num_crossover=args.num_crossover,
        mutate_after_crossover=True,
        strategy_type=args.strategy_type,
        parent_selection=tournament_selection,
        genome_handler=WrappedHandler,
        log_dir=full_log_dir,
        initial_population=initial_population,
        num_workers=args.num_workers,
    )

    # Save evolution data
    evolution_csv_path = f"{full_log_dir}/evolution_data.csv"
    all_individuals_copy = all_individuals.copy()
    all_individuals_copy['id'] = all_individuals_copy['id'].astype(str)
    all_individuals_copy.to_csv(evolution_csv_path, index=False)
    print(f"Evolution data saved to: {evolution_csv_path}")

    # Best individual
    last_gen = args.generations - 1
    best = all_individuals.loc[all_individuals['generation'] == last_gen].sort_values(
        by='fitness', ascending=False
    ).iloc[0]
    print(f"\nBest in gen {last_gen}: {best['id']}, Fitness: {best['fitness']}")

    # Plot fitness
    fitness_array = evolution_dataframe_to_fitness_array(all_individuals, population_size=args.population_size)
    _fig, ax = plt.subplots(figsize=(12, 8))
    plot_fitness(ax, fitness_array)
    ax.set_title(f"Combined Hover+Gate Evolution ({args.genome_handler.upper()}, {args.gate_cfg})")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness (hover_fitness + gates_passed)")
    plt.tight_layout()

    fitness_plot_path = f"{full_log_dir}/fitness_evolution_{args.genome_handler}_{args.gate_cfg}.png"
    plt.savefig(fitness_plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {fitness_plot_path}")

    if args.show_plot:
        plt.show()

    print(f"\nDone! Best fitness: {best['fitness']}")


if __name__ == "__main__":
    main()
