"""
Visualize genome representations alongside drone blueprints for best individuals.

Loads best_individual_data.csv and creates combined figures:
- Left: 3D isometric drone view (from DroneVisualizer)
- Right: Genome representation (heatmap for spherical/hybrid, CPPN graph for CPPN/hybrid)
"""

import ast
import os
import pickle
import sys
import types

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
import numpy as np
import pandas as pd

# Mock fcl before importing genome handler modules
sys.modules['fcl'] = types.ModuleType('fcl')

from airevolve.evolution_tools.inspection_tools.drone_visualizer import DroneVisualizer, VisualizationConfig
from airevolve.evolution_tools.genome_handlers.cppn.network import (
    CPPNNetwork, NodeType, ActivationFunction,
)
from airevolve.evolution_tools.genome_handlers.hybrid_cppn_genome_handler import HybridGenome


# Column labels for the (narms, 6) phenotype array
SPHERICAL_COLUMNS = ['magnitude', 'arm_rot', 'arm_pitch', 'motor_rot', 'motor_pitch', 'direction']
HYBRID_DIRECT_COLUMNS = ['magnitude', 'arm_yaw', 'arm_pitch']

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
    if not pkl_path or not isinstance(pkl_path, str) or not os.path.exists(pkl_path):
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
            output_labels = ['motor_yaw', 'motor_pitch', 'direction']
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
        # Simple: spread hidden nodes evenly between input and output layers
        n_hidden = len(hidden_ids)
        n_layers = min(n_hidden, 3)  # max 3 hidden layers visually
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

        # Label
        label = G.nodes[nid].get('label', str(nid))
        ax.text(x, y - 0.07, label, ha='center', va='top', fontsize=6, zorder=6)

    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.15, 1.15)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('CPPN Network', fontsize=10)


def draw_phenotype_heatmap(phenotype, ax, columns=None, title='Phenotype'):
    """Draw a heatmap of the (narms, ncols) phenotype array."""
    if columns is None:
        columns = [f'col_{i}' for i in range(phenotype.shape[1])]

    im = ax.imshow(phenotype, aspect='auto', cmap='viridis')
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(columns, rotation=45, ha='right', fontsize=7)
    ax.set_yticks(range(phenotype.shape[0]))
    ax.set_yticklabels([f'arm {i}' for i in range(phenotype.shape[0])], fontsize=7)
    ax.set_title(title, fontsize=10)

    # Add value annotations
    for i in range(phenotype.shape[0]):
        for j in range(phenotype.shape[1]):
            val = phenotype[i, j]
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=5,
                    color='white' if val < (phenotype.max() + phenotype.min()) / 2 else 'black')

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def create_spherical_figure(phenotype, fitness, run_dir, rank, ax_drone, ax_genome):
    """Create visualization for a spherical genome individual."""
    # Drone view
    viz = DroneVisualizer(VisualizationConfig(elevation=30, azimuth=45, show_axis_ticks=False))
    viz.plot_3d(phenotype, ax=ax_drone, title=f'Fitness: {fitness:.4f}')

    # Heatmap of full phenotype
    draw_phenotype_heatmap(phenotype, ax_genome, columns=SPHERICAL_COLUMNS,
                           title='Spherical Genome')


def create_cppn_figure(phenotype, fitness, genotype, run_dir, rank, ax_drone, ax_genome):
    """Create visualization for a CPPN genome individual."""
    # Drone view
    viz = DroneVisualizer(VisualizationConfig(elevation=30, azimuth=45, show_axis_ticks=False))
    viz.plot_3d(phenotype, ax=ax_drone, title=f'Fitness: {fitness:.4f}')

    # CPPN graph
    if isinstance(genotype, CPPNNetwork):
        draw_cppn_graph(genotype, ax_genome)
    else:
        ax_genome.text(0.5, 0.5, 'No CPPN genotype\navailable', ha='center', va='center',
                       transform=ax_genome.transAxes, fontsize=12)
        ax_genome.axis('off')


def create_hybrid_figure(phenotype, fitness, genotype, run_dir, rank,
                         ax_drone, ax_direct, ax_cppn):
    """Create visualization for a hybrid genome individual."""
    # Drone view
    viz = DroneVisualizer(VisualizationConfig(elevation=30, azimuth=45, show_axis_ticks=False))
    viz.plot_3d(phenotype, ax=ax_drone, title=f'Fitness: {fitness:.4f}')

    if isinstance(genotype, HybridGenome):
        # Direct parameters heatmap
        draw_phenotype_heatmap(genotype.direct, ax_direct,
                               columns=HYBRID_DIRECT_COLUMNS,
                               title='Direct Parameters')
        # CPPN graph
        draw_cppn_graph(genotype.cppn, ax_cppn)
    else:
        # Fallback: show full phenotype heatmap
        draw_phenotype_heatmap(phenotype, ax_direct,
                               columns=SPHERICAL_COLUMNS,
                               title='Phenotype (no genotype)')
        ax_cppn.text(0.5, 0.5, 'No HybridGenome\navailable', ha='center', va='center',
                     transform=ax_cppn.transAxes, fontsize=12)
        ax_cppn.axis('off')


def process_experiment(experiment_name, experiment_dir, output_dir):
    """Process a single experiment's best individuals and generate visualizations."""
    csv_path = os.path.join(experiment_dir, "best_individual_data.csv")
    if not os.path.exists(csv_path):
        print(f"  No best_individual_data.csv found in {experiment_dir}")
        return

    df = pd.read_csv(csv_path)
    print(f"  Processing {len(df)} best individuals for {experiment_name}")

    os.makedirs(output_dir, exist_ok=True)

    for idx, row in df.iterrows():
        fitness = row['fitness']
        rank = row['rank']
        run_dir = row['run_directory']
        genome_type = row.get('genome_type', 'spherical')

        # Parse phenotype
        try:
            individual_data = ast.literal_eval(row['individual_data'])
            phenotype = np.array(individual_data)
        except Exception as e:
            print(f"  Warning: failed to parse individual_data for row {idx}: {e}")
            continue

        # Load genotype if available
        genotype_path = row.get('genotype_pkl_path', '')
        genotype = load_genotype(genotype_path) if genotype_path else None

        # Create figure based on genome type
        if genome_type == 'hybrid_cppn':
            fig = plt.figure(figsize=(18, 6))
            ax_drone = fig.add_subplot(1, 3, 1, projection='3d')
            ax_direct = fig.add_subplot(1, 3, 2)
            ax_cppn = fig.add_subplot(1, 3, 3)
            create_hybrid_figure(phenotype, fitness, genotype, run_dir, rank,
                                 ax_drone, ax_direct, ax_cppn)
        elif genome_type == 'cppn':
            fig = plt.figure(figsize=(14, 6))
            ax_drone = fig.add_subplot(1, 2, 1, projection='3d')
            ax_cppn = fig.add_subplot(1, 2, 2)
            create_cppn_figure(phenotype, fitness, genotype, run_dir, rank,
                               ax_drone, ax_cppn)
        else:
            # spherical
            fig = plt.figure(figsize=(14, 6))
            ax_drone = fig.add_subplot(1, 2, 1, projection='3d')
            ax_genome = fig.add_subplot(1, 2, 2)
            create_spherical_figure(phenotype, fitness, run_dir, rank,
                                    ax_drone, ax_genome)

        run_name = os.path.basename(run_dir)
        fig.suptitle(f'{experiment_name} | {run_name} | Rank {rank}', fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        filename = f"{experiment_name}_rank{rank:02d}_{run_name}.png"
        fig.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"    Saved {filename}")


def main():
    base_dir = "/media/jed/My Passport/airevolve030326"
    output_dir = os.path.join(base_dir, "plots", "genome_visualizations")

    experiments = {
        "spherical": os.path.join(base_dir, "spherical"),
        "cppn": os.path.join(base_dir, "cppn"),
        "hybrid_cppn": os.path.join(base_dir, "hybrid_cppn"),
    }

    for experiment_name, experiment_dir in experiments.items():
        if not os.path.exists(experiment_dir):
            print(f"Skipping {experiment_name}: directory not found")
            continue

        print(f"Processing {experiment_name}...")
        process_experiment(experiment_name, experiment_dir, output_dir)

    print("Done!")


if __name__ == "__main__":
    main()
