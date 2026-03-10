"""CPPN Empty-Init Mutation Experiment

Measures how many incremental NEAT mutations are needed to build a viable
drone starting from an empty CPPN (input/output nodes only, no connections).

For each candidate:
  1. Create an empty CPPN (2 inputs, 7 outputs, zero connections, zero biases)
  2. Apply one NEAT mutation at a time (biased toward structural mutations)
  3. After each mutation, decode to phenotype and check arm count
  4. If valid arm count: run 3-stage repair pipeline (Phase 1)
  5. If Phase 1 passes: run CMA-ES tuning (Phase 2)
  6. If Phase 2 passes: accept drone, record stats, save genome
  7. If max mutations exceeded: reset to new empty CPPN

Repeats until --pop-size viable drones are collected.

Usage:
    # Quick test
    python experimentation/cppn_empty_init_experiment.py \
        --pop-size 1 --max-mutations 100

    # Full experiment
    python experimentation/cppn_empty_init_experiment.py \
        --pop-size 20 --max-mutations 500
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
import time
import numpy as np
from datetime import datetime
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from airevolve.evolution_tools.genome_handlers.cppn.network import (
    ActivationFunction,
    CPPNNetwork,
    NodeGene,
    NodeType,
)
from airevolve.evolution_tools.genome_handlers.cppn.mutations import mutate_cppn
from airevolve.evolution_tools.genome_handlers.cppn.segment_decoder import decode_cppn_to_phenotype
from airevolve.evolution_tools.genome_handlers.cppn.innovation import InnovationCounter
from airevolve.evolution_tools.genome_handlers.repair_workflow import (
    stage1_optimization_repair,
    stage2_hover_check,
    stage3_hover_repair,
)
from airevolve.evolution_tools.genome_handlers.operators.optimization_repair_operator import (
    OptimizationRepairConfig,
)
from airevolve.evolution_tools.evaluators.lee_tune_evaluator import (
    optimize_controller_with_early_stop,
)
from airevolve.controllers.utils.gate_configs import GATE_CONFIGS


# Number of CPPN input / output nodes
_N_INPUTS = 2
_N_OUTPUTS = 7

# Default spherical parameter limits (same as CPPNNeatDroneGenomeHandler)
_DEFAULT_PARAM_LIMITS = np.array([
    [0.055, 0.17],           # magnitude
    [-np.pi, np.pi],         # arm yaw (azimuth)
    [-np.pi / 2, np.pi / 2], # arm pitch
    [-np.pi / 2, np.pi / 2], # motor pitch
    [-np.pi, np.pi],         # motor yaw
    [0, 1],                  # direction
])


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='CPPN empty-init mutation experiment'
    )
    parser.add_argument('--pop-size', type=int, default=20,
                        help='Number of viable drones to collect (default: 20)')
    parser.add_argument('--max-mutations', type=int, default=500,
                        help='Max mutations per candidate before reset (default: 500)')
    parser.add_argument('--max-candidates', type=int, default=10000,
                        help='Total candidates to try before giving up (default: 10000)')
    parser.add_argument('--init-pop-max-evals', type=int, default=200,
                        help='CMA-ES budget for Phase 2 (default: 200)')
    parser.add_argument('--init-pop-gates-threshold', type=int, default=1,
                        help='Min gates for Phase 2 acceptance (default: 1)')
    parser.add_argument('--gate-cfg', choices=['backandforth', 'figure8', 'circle', 'slalom'],
                        default='circle', help='Gate configuration (default: circle)')
    parser.add_argument('--sim-time', type=float, default=20.0,
                        help='Simulation time in seconds (default: 20.0)')
    parser.add_argument('--dt', type=float, default=0.005,
                        help='Time step in seconds (default: 0.005)')
    parser.add_argument('--num-segments', type=int, default=8,
                        help='Number of CPPN evaluation segments (default: 8)')
    parser.add_argument('--min-narms', type=int, default=6,
                        help='Minimum number of arms (default: 6)')
    parser.add_argument('--max-narms', type=int, default=6,
                        help='Maximum number of arms (default: 6)')
    parser.add_argument('--output-dir', default='.data/cppn_empty_init_experiment',
                        help='Output directory (default: .data/cppn_empty_init_experiment)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (default: random)')

    # Mutation probability overrides
    parser.add_argument('--prob-add-connection', type=float, default=0.30)
    parser.add_argument('--prob-add-node', type=float, default=0.10)
    parser.add_argument('--prob-mutate-weights', type=float, default=0.45)
    parser.add_argument('--prob-remove-node', type=float, default=0.01)
    parser.add_argument('--prob-remove-connection', type=float, default=0.02)
    parser.add_argument('--prob-mutate-activation', type=float, default=0.05)
    parser.add_argument('--prob-toggle-connection', type=float, default=0.02)

    return parser.parse_args()


def _create_empty_cppn() -> CPPNNetwork:
    """Create an empty CPPN with input/output nodes only, no connections."""
    net = CPPNNetwork()

    # Input nodes
    net.nodes[0] = NodeGene(
        node_id=0, node_type=NodeType.INPUT,
        activation=ActivationFunction.IDENTITY,
        bias=0.0, input_label="seg_normalized",
    )
    net.nodes[1] = NodeGene(
        node_id=1, node_type=NodeType.INPUT,
        activation=ActivationFunction.IDENTITY,
        bias=0.0, input_label="bias",
    )

    # Output nodes (SIN activation, zero bias)
    output_labels = [
        "arm_present", "magnitude", "arm_yaw", "arm_pitch",
        "motor_yaw", "motor_pitch", "direction",
    ]
    for j in range(_N_OUTPUTS):
        nid = _N_INPUTS + j
        net.nodes[nid] = NodeGene(
            node_id=nid, node_type=NodeType.OUTPUT,
            activation=ActivationFunction.SIN,
            bias=0.0, output_index=j,
        )

    net.next_node_id = _N_INPUTS + _N_OUTPUTS  # = 9
    return net


def _get_network_complexity(network: CPPNNetwork) -> dict:
    """Return complexity stats for a CPPN."""
    enabled = network.get_enabled_connections()
    hidden = network.get_hidden_nodes()
    return {
        'num_nodes': len(network.nodes),
        'num_connections': len(enabled),
        'num_hidden_nodes': len(hidden),
        'total_connections': len(network.connections),
    }


def _run_phase1(phenotype: np.ndarray) -> bool:
    """Run the 3-stage repair pipeline. Returns True if all stages pass."""
    # Stage 1: Hover check (strict, no spinning)
    can_hover, _ = stage2_hover_check(
        phenotype, verbose=False, allow_spinning=False,
    )
    if not can_hover:
        return False

    # Stage 2: Optimization repair (fix collisions)
    repair_config = OptimizationRepairConfig(fixed_params=[3, 4])
    repaired, _ = stage1_optimization_repair(
        phenotype, coordinate_system='spherical',
        config=repair_config, verbose=False,
    )
    if repaired is None:
        return False

    # Stage 3: Hover repair (align thrust vectors)
    final, _ = stage3_hover_repair(
        repaired, coordinate_system='spherical', verbose=False,
    )
    return final is not None


def _run_phase2(phenotype: np.ndarray, gate_cfg: str, max_evals: int,
                gates_threshold: int, sim_time: float, dt: float) -> dict:
    """Run CMA-ES tuning on the repaired phenotype. Returns tuning result."""
    # Re-run repair to get the final repaired phenotype for tuning
    can_hover, _ = stage2_hover_check(
        phenotype, verbose=False, allow_spinning=False,
    )
    if not can_hover:
        return {"gates_passed": 0, "skipped": "failed_hover_recheck"}

    repair_config = OptimizationRepairConfig(fixed_params=[3, 4])
    repaired, _ = stage1_optimization_repair(
        phenotype, coordinate_system='spherical',
        config=repair_config, verbose=False,
    )
    if repaired is None:
        return {"gates_passed": 0, "skipped": "failed_repair_recheck"}

    final, _ = stage3_hover_repair(
        repaired, coordinate_system='spherical', verbose=False,
    )
    if final is None:
        return {"gates_passed": 0, "skipped": "failed_hover_repair_recheck"}

    gate_config = GATE_CONFIGS[gate_cfg]
    tuning = optimize_controller_with_early_stop(
        final, gate_config,
        max_evaluations=max_evals,
        num_workers=1,
        sim_time=sim_time,
        dt=dt,
        timeout_per_eval=30.0,
        gates_threshold=gates_threshold,
    )
    return tuning


def main():
    args = parse_arguments()

    # Seed
    if args.seed is None:
        args.seed = np.random.randint(0, 2**31)
    rng = np.random.default_rng(args.seed)

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, timestamp)
    genomes_dir = os.path.join(output_dir, "genomes")
    os.makedirs(genomes_dir, exist_ok=True)

    print("=" * 80)
    print("CPPN Empty-Init Mutation Experiment")
    print("=" * 80)
    print(f"Seed: {args.seed}")
    print(f"Target viable drones: {args.pop_size}")
    print(f"Max mutations per candidate: {args.max_mutations}")
    print(f"Max candidates: {args.max_candidates}")
    print(f"Gate config: {args.gate_cfg}")
    print(f"Phase 2 CMA-ES budget: {args.init_pop_max_evals}")
    print(f"Phase 2 gates threshold: {args.init_pop_gates_threshold}")
    print(f"Num segments: {args.num_segments}")
    print(f"Arms: {args.min_narms}-{args.max_narms}")
    print(f"Output: {output_dir}")
    print()
    print("Mutation probabilities:")
    print(f"  add_connection: {args.prob_add_connection}")
    print(f"  add_node:       {args.prob_add_node}")
    print(f"  mutate_weights: {args.prob_mutate_weights}")
    print(f"  remove_node:    {args.prob_remove_node}")
    print(f"  remove_conn:    {args.prob_remove_connection}")
    print(f"  mutate_act:     {args.prob_mutate_activation}")
    print(f"  toggle_conn:    {args.prob_toggle_connection}")
    print("=" * 80)
    print()
    sys.stdout.flush()

    innovation_counter = InnovationCounter()

    # Mutation kwargs
    mutation_kwargs = dict(
        prob_add_node=args.prob_add_node,
        prob_add_connection=args.prob_add_connection,
        prob_remove_node=args.prob_remove_node,
        prob_remove_connection=args.prob_remove_connection,
        prob_mutate_weights=args.prob_mutate_weights,
        prob_mutate_activation=args.prob_mutate_activation,
        prob_toggle_connection=args.prob_toggle_connection,
    )

    param_limits = _DEFAULT_PARAM_LIMITS

    viable_drones = []
    failed_candidates = []
    total_start = time.time()

    # Running counters for the outer progress bar
    total_mutations = 0
    total_arm_checks_passed = 0
    total_phase1_passed = 0
    total_phase2_attempted = 0
    total_phase2_passed = 0

    outer_bar = tqdm(
        total=args.pop_size, desc="Viable drones collected",
        unit="drone", position=0,
    )

    candidate_idx = 0
    while len(viable_drones) < args.pop_size and candidate_idx < args.max_candidates:
        candidate_idx += 1
        candidate_start = time.time()

        network = _create_empty_cppn()
        mutations_to_phase1 = None

        # Per-candidate counters
        cand_arm_ok = 0
        cand_p1_pass = 0
        cand_p2_attempt = 0

        desc = f"Candidate {candidate_idx}"
        mut_bar = tqdm(
            total=args.max_mutations, desc=desc,
            unit="mut", position=1, leave=False,
        )

        accepted = False
        for mutation_count in range(1, args.max_mutations + 1):
            mutate_cppn(network, innovation_counter, rng, **mutation_kwargs)
            total_mutations += 1

            # Decode to phenotype
            phenotype = decode_cppn_to_phenotype(
                network,
                num_segments=args.num_segments,
                arm_limit=args.max_narms,
                parameter_limits=param_limits,
            )

            # Check arm count
            arm_count = int(np.sum(~np.isnan(phenotype[:, 0])))
            complexity = _get_network_complexity(network)

            mut_bar.update(1)
            mut_bar.set_postfix({
                'arms': arm_count,
                'nodes': complexity['num_nodes'],
                'conns': complexity['num_connections'],
                'arm_ok': cand_arm_ok,
                'p1': cand_p1_pass,
                'p2': cand_p2_attempt,
            })

            if arm_count < args.min_narms or arm_count > args.max_narms:
                continue

            cand_arm_ok += 1
            total_arm_checks_passed += 1

            # Phase 1: 3-stage repair pipeline
            tqdm.write(
                f"  [Cand {candidate_idx} | Mut {mutation_count}] "
                f"Arms={arm_count} -> Phase 1 (hover+repair)...",
                end="",
            )
            phase1_pass = _run_phase1(phenotype)
            if not phase1_pass:
                tqdm.write(" FAIL")
                continue

            cand_p1_pass += 1
            total_phase1_passed += 1
            tqdm.write(" PASS")

            if mutations_to_phase1 is None:
                mutations_to_phase1 = mutation_count

            # Phase 2: CMA-ES tuning
            cand_p2_attempt += 1
            total_phase2_attempted += 1
            tqdm.write(
                f"  [Cand {candidate_idx} | Mut {mutation_count}] "
                f"Phase 2 (CMA-ES, budget={args.init_pop_max_evals})...",
                end="",
            )
            tuning = _run_phase2(
                phenotype, args.gate_cfg, args.init_pop_max_evals,
                args.init_pop_gates_threshold, args.sim_time, args.dt,
            )

            gates_passed = tuning.get("gates_passed", 0)
            if gates_passed >= args.init_pop_gates_threshold:
                total_phase2_passed += 1
                tqdm.write(f" PASS (gates={gates_passed})")

                # ACCEPT
                candidate_time = time.time() - candidate_start

                record = {
                    'candidate_idx': candidate_idx,
                    'mutations_to_viable': mutation_count,
                    'mutations_to_phase1': mutations_to_phase1,
                    'wall_clock_seconds': candidate_time,
                    'network_complexity': complexity,
                    'gates_passed': gates_passed,
                    'arm_count': arm_count,
                }
                viable_drones.append(record)

                # Save genome
                genome_path = os.path.join(genomes_dir, f"drone_{len(viable_drones):04d}.pkl")
                with open(genome_path, 'wb') as f:
                    pickle.dump(network, f)
                record['genome_path'] = genome_path

                # Save phenotype
                phenotype_path = os.path.join(genomes_dir, f"drone_{len(viable_drones):04d}_phenotype.npy")
                np.save(phenotype_path, phenotype)

                accepted = True
                mut_bar.close()

                outer_bar.update(1)
                outer_bar.set_postfix({
                    'cand': candidate_idx,
                    'failed': len(failed_candidates),
                    'tot_mut': total_mutations,
                    'p1_rate': f"{total_phase1_passed}/{total_arm_checks_passed}",
                    'p2_rate': f"{total_phase2_passed}/{total_phase2_attempted}",
                })

                tqdm.write(
                    f"  >>> VIABLE DRONE #{len(viable_drones)}/{args.pop_size} "
                    f"(candidate {candidate_idx}, {mutation_count} mutations, "
                    f"{candidate_time:.1f}s, {gates_passed} gates, "
                    f"{complexity['num_nodes']} nodes, "
                    f"{complexity['num_connections']} conns)"
                )
                break
            else:
                tqdm.write(f" FAIL (gates={gates_passed})")

        if not accepted:
            mut_bar.close()
            # Max mutations exceeded — record failure
            candidate_time = time.time() - candidate_start
            failed_candidates.append({
                'candidate_idx': candidate_idx,
                'mutations_applied': args.max_mutations,
                'mutations_to_phase1': mutations_to_phase1,
                'wall_clock_seconds': candidate_time,
                'network_complexity': _get_network_complexity(network),
            })
            outer_bar.set_postfix({
                'cand': candidate_idx,
                'failed': len(failed_candidates),
                'tot_mut': total_mutations,
                'p1_rate': f"{total_phase1_passed}/{total_arm_checks_passed}",
                'p2_rate': f"{total_phase2_passed}/{total_phase2_attempted}",
            })
            p1_note = f", first P1 at mut {mutations_to_phase1}" if mutations_to_phase1 else ""
            tqdm.write(
                f"  --- Candidate {candidate_idx} FAILED after "
                f"{args.max_mutations} mutations ({candidate_time:.1f}s, "
                f"arm_ok={cand_arm_ok}, p1={cand_p1_pass}, "
                f"p2_tried={cand_p2_attempt}{p1_note})"
            )

    outer_bar.close()
    total_time = time.time() - total_start

    # Aggregate stats
    n_viable = len(viable_drones)
    n_failed = len(failed_candidates)
    n_total = n_viable + n_failed

    print()
    print(f"Pipeline totals: {total_mutations} mutations, "
          f"{total_arm_checks_passed} arm-valid, "
          f"{total_phase1_passed} Phase 1 passes, "
          f"{total_phase2_attempted} Phase 2 attempts, "
          f"{total_phase2_passed} Phase 2 passes")

    if n_viable > 0:
        mutations_list = [d['mutations_to_viable'] for d in viable_drones]
        phase1_list = [d['mutations_to_phase1'] for d in viable_drones]
        time_list = [d['wall_clock_seconds'] for d in viable_drones]
        nodes_list = [d['network_complexity']['num_nodes'] for d in viable_drones]
        conns_list = [d['network_complexity']['num_connections'] for d in viable_drones]
        hidden_list = [d['network_complexity']['num_hidden_nodes'] for d in viable_drones]

        stats = {
            'mutations_to_viable': {
                'mean': float(np.mean(mutations_list)),
                'median': float(np.median(mutations_list)),
                'std': float(np.std(mutations_list)),
                'min': int(np.min(mutations_list)),
                'max': int(np.max(mutations_list)),
            },
            'mutations_to_phase1': {
                'mean': float(np.mean(phase1_list)),
                'median': float(np.median(phase1_list)),
                'std': float(np.std(phase1_list)),
                'min': int(np.min(phase1_list)),
                'max': int(np.max(phase1_list)),
            },
            'time_per_viable': {
                'mean': float(np.mean(time_list)),
                'median': float(np.median(time_list)),
                'std': float(np.std(time_list)),
            },
            'network_complexity': {
                'num_nodes': {'mean': float(np.mean(nodes_list)), 'std': float(np.std(nodes_list))},
                'num_connections': {'mean': float(np.mean(conns_list)), 'std': float(np.std(conns_list))},
                'num_hidden_nodes': {'mean': float(np.mean(hidden_list)), 'std': float(np.std(hidden_list))},
            },
        }
    else:
        stats = {}

    results = {
        'config': {
            'seed': args.seed,
            'pop_size': args.pop_size,
            'max_mutations': args.max_mutations,
            'max_candidates': args.max_candidates,
            'gate_cfg': args.gate_cfg,
            'init_pop_max_evals': args.init_pop_max_evals,
            'init_pop_gates_threshold': args.init_pop_gates_threshold,
            'num_segments': args.num_segments,
            'min_narms': args.min_narms,
            'max_narms': args.max_narms,
            'mutation_probs': mutation_kwargs,
        },
        'aggregate': {
            'total_candidates': n_total,
            'viable_count': n_viable,
            'failed_count': n_failed,
            'success_rate': n_viable / max(1, n_total),
            'total_mutations': total_mutations,
            'total_arm_checks_passed': total_arm_checks_passed,
            'total_phase1_passed': total_phase1_passed,
            'total_phase2_attempted': total_phase2_attempted,
            'total_phase2_passed': total_phase2_passed,
            'total_wall_clock_seconds': total_time,
            **stats,
        },
        'viable_drones': viable_drones,
        'failed_candidates': failed_candidates,
    }

    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Print summary
    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Total candidates attempted: {n_total}")
    print(f"Viable drones collected: {n_viable}/{args.pop_size}")
    print(f"Failed candidates: {n_failed}")
    print(f"Success rate: {n_viable / max(1, n_total) * 100:.1f}%")
    print(f"Total wall clock time: {total_time:.1f}s")

    if n_viable > 0:
        print()
        print("Mutations to viability:")
        s = stats['mutations_to_viable']
        print(f"  Mean: {s['mean']:.1f}, Median: {s['median']:.1f}, "
              f"Std: {s['std']:.1f}, Min: {s['min']}, Max: {s['max']}")

        print("Mutations to Phase 1:")
        s = stats['mutations_to_phase1']
        print(f"  Mean: {s['mean']:.1f}, Median: {s['median']:.1f}, "
              f"Std: {s['std']:.1f}, Min: {s['min']}, Max: {s['max']}")

        print("Time per viable drone:")
        s = stats['time_per_viable']
        print(f"  Mean: {s['mean']:.1f}s, Median: {s['median']:.1f}s, Std: {s['std']:.1f}s")

        print("Network complexity (viable drones):")
        c = stats['network_complexity']
        print(f"  Nodes: {c['num_nodes']['mean']:.1f} +/- {c['num_nodes']['std']:.1f}")
        print(f"  Connections: {c['num_connections']['mean']:.1f} +/- {c['num_connections']['std']:.1f}")
        print(f"  Hidden nodes: {c['num_hidden_nodes']['mean']:.1f} +/- {c['num_hidden_nodes']['std']:.1f}")

    print(f"\nResults saved to: {results_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
