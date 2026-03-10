"""
Visualize CPPN genomes + drone blueprints for best individuals in a run directory.

Scans generation_*/individual_*/genotype.pkl + hover_breakdown.json directly,
ranks by fitness, and visualizes the top-k.

Usage:
    python experimentation/plot_cppn_visualization.py \
        --run-dir .data/combined_hover_gate_circle_6arms_20260310_113449 \
        --top-k 10
"""

import argparse
import glob
import json
import os
import pickle
import sys
import types

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
import numpy as np

# Mock fcl before importing genome handler modules
sys.modules['fcl'] = types.ModuleType('fcl')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from airevolve.evolution_tools.inspection_tools.drone_visualizer import DroneVisualizer, VisualizationConfig
from airevolve.evolution_tools.genome_handlers.cppn.network import (
    CPPNNetwork, NodeType, ActivationFunction,
)
from airevolve.evolution_tools.genome_handlers.cppn_neat_genome_handler import CPPNNeatDroneGenomeHandler


# Color map for activation functions
ACTIVATION_COLORS = {
    ActivationFunction.IDENTITY: '#808080',   # gray
    ActivationFunction.SIGMOID: '#FF6B6B',    # red
    ActivationFunction.TANH: '#4ECDC4',       # teal
    ActivationFunction.SIN: '#45B7D1',        # blue
    ActivationFunction.COS: '#96CEB4',        # green
    ActivationFunction.GAUSSIAN: '#FFEAA7',   # yellow
    ActivationFunction.ABS: '#DDA0DD',        # plum
    ActivationFunction.RELU: '#FF8C00',       # orange
    ActivationFunction.STEP: '#8B4513',       # brown
}


def load_genotype(pkl_path):
    """Load a genotype from a pickle file."""
    if not pkl_path or not os.path.exists(pkl_path):
        return None
    try:
        with open(pkl_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"  Warning: failed to load genotype from {pkl_path}: {e}")
        return None


def draw_cppn_graph(cppn, ax):
    """Draw a CPPN network graph using networkx + matplotlib.

    Layered layout: inputs bottom, outputs top, hidden in between.
    Nodes colored by activation, edges colored by weight sign.
    """
    G = nx.DiGraph()

    # Classify nodes
    input_nodes = cppn.get_input_nodes()
    output_nodes = cppn.get_output_nodes()
    hidden_nodes = cppn.get_hidden_nodes()

    all_nodes = list(cppn.nodes.values())
    for node in all_nodes:
        label = node.activation.value
        if node.input_label:
            label = node.input_label
        elif node.node_type == NodeType.OUTPUT and node.output_index is not None:
            output_labels = [
                'arm_present', 'magnitude', 'arm_yaw', 'arm_pitch',
                'motor_yaw', 'motor_pitch', 'direction',
            ]
            if node.output_index < len(output_labels):
                label = output_labels[node.output_index]
        G.add_node(node.node_id, label=label, activation=node.activation,
                   node_type=node.node_type)

    # Add edges
    enabled_connections = cppn.get_enabled_connections()
    for conn in enabled_connections:
        G.add_edge(conn.source_id, conn.target_id, weight=conn.weight)

    # Layered positioning
    pos = {}
    input_ids = [n.node_id for n in input_nodes]
    output_ids = [n.node_id for n in output_nodes]
    hidden_ids = [n.node_id for n in hidden_nodes]

    # Inputs at y=0
    for i, nid in enumerate(input_ids):
        pos[nid] = (i / max(len(input_ids) - 1, 1), 0.0)

    # Outputs at y=1
    for i, nid in enumerate(output_ids):
        pos[nid] = (i / max(len(output_ids) - 1, 1), 1.0)

    # Hidden nodes at intermediate y levels
    if hidden_ids:
        n_hidden = len(hidden_ids)
        n_layers = min(n_hidden, 3)
        for i, nid in enumerate(hidden_ids):
            layer = (i % n_layers) + 1
            y = layer / (n_layers + 1)
            col_in_layer = i // n_layers
            total_in_layer = (n_hidden + n_layers - 1) // n_layers
            x = col_in_layer / max(total_in_layer - 1, 1)
            pos[nid] = (x, y)

    # Draw edges
    for u, v, data in G.edges(data=True):
        w = data['weight']
        color = '#2196F3' if w >= 0 else '#F44336'  # blue positive, red negative
        width = min(abs(w) * 0.8, 4.0)
        ax.annotate(
            '', xy=pos[v], xytext=pos[u],
            arrowprops=dict(
                arrowstyle='->', color=color, lw=max(width, 0.5),
                connectionstyle='arc3,rad=0.1', alpha=0.7,
            ),
        )

    # Draw nodes
    for node in all_nodes:
        nid = node.node_id
        if nid not in pos:
            continue
        x, y = pos[nid]
        color = ACTIVATION_COLORS.get(node.activation, '#808080')
        circle = plt.Circle((x, y), 0.04, color=color, ec='black', lw=1.0, zorder=5)
        ax.add_patch(circle)

        label = G.nodes[nid].get('label', str(nid))
        ax.text(x, y - 0.07, label, ha='center', va='top', fontsize=6, zorder=6)

    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.15, 1.15)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('CPPN Network', fontsize=10)


def process_run_directory(run_dir, output_dir, top_k=10):
    """Scan a run directory for genotype.pkl + hover_breakdown.json,
    rank by fitness, and visualize the top-k as drone blueprint + CPPN graph."""

    os.makedirs(output_dir, exist_ok=True)

    # Collect all individuals with genotype.pkl
    individuals = []
    for pkl_path in glob.glob(os.path.join(run_dir, "generation_*/individual_*/genotype.pkl")):
        ind_dir = os.path.dirname(pkl_path)
        breakdown_path = os.path.join(ind_dir, "hover_breakdown.json")
        if not os.path.exists(breakdown_path):
            continue
        with open(breakdown_path) as f:
            breakdown = json.load(f)
        individuals.append({
            "ind_dir": ind_dir,
            "pkl_path": pkl_path,
            "fitness": breakdown["total_fitness"],
            "hover_fitness": breakdown["hover_fitness"],
            "gates_passed": breakdown["gates_passed"],
            "status": breakdown["status"],
        })

    if not individuals:
        print("No individuals found with genotype.pkl + hover_breakdown.json")
        return

    # Sort by fitness descending, take top-k
    individuals.sort(key=lambda x: x["fitness"], reverse=True)
    top = individuals[:top_k]

    print(f"Found {len(individuals)} individuals, visualizing top {len(top)}")

    for rank, ind in enumerate(top):
        genotype = load_genotype(ind["pkl_path"])
        if not isinstance(genotype, CPPNNetwork):
            print(f"  Rank {rank}: skipping (not a CPPNNetwork)")
            continue

        # Decode to phenotype
        handler = CPPNNeatDroneGenomeHandler(genome=genotype, num_segments=8, repair=False)
        phenotype = handler.get_phenotype()

        # Create figure: drone blueprint + CPPN graph
        fig = plt.figure(figsize=(14, 6))
        ax_drone = fig.add_subplot(1, 2, 1, projection='3d')
        ax_cppn = fig.add_subplot(1, 2, 2)

        # Drone view
        viz = DroneVisualizer(VisualizationConfig(elevation=30, azimuth=45, show_axis_ticks=False))
        viz.plot_3d(phenotype, ax=ax_drone, title=f'Fitness: {ind["fitness"]:.2f}')

        # CPPN graph
        draw_cppn_graph(genotype, ax_cppn)

        ind_name = os.path.basename(ind["ind_dir"])
        gen_name = os.path.basename(os.path.dirname(ind["ind_dir"]))
        fig.suptitle(
            f'Rank {rank} | {gen_name}/{ind_name} | '
            f'Fitness: {ind["fitness"]:.2f} (hover: {ind["hover_fitness"]:.2f}, gates: {ind["gates_passed"]})',
            fontsize=11,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.93])

        filename = f"rank{rank:02d}_{gen_name}_{ind_name}.png"
        fig.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {filename}")


def main():
    parser = argparse.ArgumentParser(description='Visualize top CPPN genomes from a run directory')
    parser.add_argument('--run-dir', default='.data/combined_hover_gate_circle_6arms_20260310_113449',
                        help='Path to run directory')
    parser.add_argument('--top-k', type=int, default=10,
                        help='Number of top individuals to visualize (default: 10)')
    args = parser.parse_args()

    output_dir = os.path.join(args.run_dir, "plots", "genome_visualizations")
    print(f"Run directory: {args.run_dir}")
    print(f"Output: {output_dir}")
    process_run_directory(args.run_dir, output_dir, top_k=args.top_k)
    print("Done!")


if __name__ == "__main__":
    main()
