"""Visualize a random CPPN-NEAT genome: network graph + drone blueprint.

Generates one random genome with default settings, prints stats, and displays:
  1. A CPPN network graph (nodes, connections, activations)
  2. A 4-view drone blueprint from the decoded phenotype
"""

import sys
import os

import numpy as np
import matplotlib.pyplot as plt

# Add the project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from airevolve.evolution_tools.genome_handlers.cppn_neat_genome_handler import CPPNNeatDroneGenomeHandler
from airevolve.evolution_tools.genome_handlers.cppn.network import NodeType
from airevolve.evolution_tools.inspection_tools.cppn_visualizer import CPPNVisualizer
from airevolve.evolution_tools.inspection_tools.drone_visualizer import DroneVisualizer
from airevolve.evolution_tools.genome_handlers.operators.optimization_repair_operator import (
    OptimizationBasedRepairOperator,
)


def main():
    # 1. Create a random genome handler
    handler = CPPNNeatDroneGenomeHandler()
    genome = handler.genome

    # 2. Print genome stats
    input_nodes = genome.get_input_nodes()
    hidden_nodes = genome.get_hidden_nodes()
    output_nodes = genome.get_output_nodes()
    enabled_conns = genome.get_enabled_connections()
    disabled_conns = [c for c in genome.connections.values() if not c.enabled]

    print("=== CPPN Genome Stats ===")
    print(f"  Input nodes:  {len(input_nodes)}")
    print(f"  Hidden nodes: {len(hidden_nodes)}")
    print(f"  Output nodes: {len(output_nodes)}")
    print(f"  Connections:  {len(enabled_conns)} enabled, {len(disabled_conns)} disabled")
    print()

    # 3. Decode phenotype
    phenotype = handler.get_phenotype()
    arm_count = int(np.sum(~np.isnan(phenotype[:, 0])))
    print(f"  Decoded arms: {arm_count}")
    print()

    # 4. Repair phenotype
    repair_op = OptimizationBasedRepairOperator(coordinate_system='spherical')
    repaired_phenotype = repair_op.repair(phenotype)
    repaired_arm_count = int(np.sum(~np.isnan(repaired_phenotype[:, 0])))
    print(f"  Repaired arms: {repaired_arm_count}")
    print()

    # 5. Save outputs to tmp/ directory next to this script
    tmp_dir = os.path.join(os.path.dirname(__file__), '..', 'tmp')
    os.makedirs(tmp_dir, exist_ok=True)

    # CPPN network graph
    cppn_viz = CPPNVisualizer()
    fig_cppn, ax_cppn = cppn_viz.plot_network(handler, title="CPPN Network Graph")
    cppn_path = os.path.join(tmp_dir, 'cppn_network.png')
    fig_cppn.savefig(cppn_path, dpi=150)
    print(f"  Saved CPPN graph:        {cppn_path}")

    drone_viz = DroneVisualizer()

    # Initial phenotype blueprint
    fig_bp, axes_bp = drone_viz.plot_blueprint(phenotype, title="Initial Phenotype")
    blueprint_path = os.path.join(tmp_dir, 'drone_blueprint_initial.png')
    fig_bp.savefig(blueprint_path, dpi=150)
    print(f"  Saved initial blueprint: {blueprint_path}")

    # Repaired phenotype blueprint
    fig_rep, axes_rep = drone_viz.plot_blueprint(repaired_phenotype, title="Repaired Phenotype")
    repaired_path = os.path.join(tmp_dir, 'drone_blueprint_repaired.png')
    fig_rep.savefig(repaired_path, dpi=150)
    print(f"  Saved repaired blueprint: {repaired_path}")

    plt.close('all')


if __name__ == "__main__":
    main()
