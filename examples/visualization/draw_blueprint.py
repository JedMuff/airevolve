"""
Comprehensive demo of the genome visualization system.

This script demonstrates the new DroneVisualizer class and compares it with
the original visualization functions. Shows various genome formats, styling
options, and use cases.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any

# Add the project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from airevolve.evolution_tools.inspection_tools.drone_visualizer import DroneVisualizer, VisualizationConfig
import airevolve.evolution_tools.inspection_tools.utils as u

def demo_blueprint_views():
    """Demo the blueprint visualization with multiple views."""
    print("\n=== Blueprint Views Demo ===")
    
if __name__ == "__main__":
    # Arm lengths (front slightly longer to keep props out of view) — tweak as needed
    Lf, Lr = 0.26, 0.24  # front, rear arms

    # Y6 with 3 arms at 0°, 120°, 240°, two coaxial motors per arm (counter-rotating)
    y6_true = np.array([
        [Lf, 0.0,         0.0, 0.0, 0.0, 1.0],   # front arm, motor A
        [Lf, 0.0,         0.0, 0.0, 0.0, 0.0],   # front arm, motor B (opposite spin)
        [Lr, 2*np.pi/3,   0.0, 0.0, 0.0, 0.0],   # left-rear arm, motor A
        [Lr, 2*np.pi/3,   0.0, 0.0, 0.0, 1.0],   # left-rear arm, motor B
        [Lr, 4*np.pi/3,   0.0, 0.0, 0.0, 1.0],   # right-rear arm, motor A
        [Lr, 4*np.pi/3,   0.0, 0.0, 0.0, 0.0],   # right-rear arm, motor B
    ])
    # Small inward offset for the second motor in each coaxial pair
    delta = 0.015  # meters

    y6_stacked = np.array([
        [0.24, 0.0,       np.pi/18, 0.0, 0.0, 1.0],  # front, motor pitched +5°
        [0.24, 0.0,       -np.pi/18, 0.0, np.pi, 0.0],  # front, motor pitched -5°
        [0.24, 2*np.pi/3, np.pi/18, 0.0, 0.0, 0.0],  # left-rear, motor pitched +5°
        [0.24, 2*np.pi/3, -np.pi/18, 0.0, np.pi, 1.0],  # left-rear, motor pitched -5°
        [0.24, 4*np.pi/3, np.pi/18, 0.0, 0.0, 1.0],  # right-rear, motor pitched +5°
        [0.24, 4*np.pi/3, -np.pi/18, 0.0, np.pi, 0.0],  # right-rear, motor pitched -5°
    ])
    # Create blueprint of tilted motors configuration
    visualizer = DroneVisualizer()
    fig, axes = visualizer.plot_blueprint(
        y6_stacked,
        title="Blueprint Views"
    )
    plt.show()