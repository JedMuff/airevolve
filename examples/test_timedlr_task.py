#!/usr/bin/env python3
"""
Test script for the timedlr (Task B) implementation.
This script demonstrates gate generation and basic task setup.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from airevolve.evolution_tools.evaluators.gate_train import timedlr

def test_gate_generation():
    """Test that gates are generated correctly."""
    print("Testing timedlr gate generation...")
    print(f"Gate configuration:")
    print(f"  - Number of gates: {timedlr.num_gates}")
    print(f"  - Gate distance: {timedlr.gate_d}m")
    print(f"  - Gate radius range: {timedlr.gate_r_min}m - {timedlr.gate_r_max}m")
    print(f"  - Gate z range: {timedlr.gate_z_min}m - {timedlr.gate_z_max}m")
    print(f"  - Gate width: {timedlr.gate_width}m")
    print(f"  - Left/Right probability: {timedlr.gate_prob_lr}")
    print()
    
    # Generate gates with a seed for reproducibility
    seed = 42
    gate_pos, gate_yaw = timedlr.generate_gates(seed=seed)
    
    print(f"Generated {len(gate_pos)} gates with seed {seed}")
    print(f"First 5 gate positions:")
    for i in range(min(5, len(gate_pos))):
        print(f"  Gate {i}: pos={gate_pos[i]}, yaw={gate_yaw[i]:.2f} rad")
    print()
    
    # Verify properties
    print("Verification:")
    x_positions = gate_pos[:, 0]
    y_positions = gate_pos[:, 1]
    z_positions = gate_pos[:, 2]
    
    print(f"  X range: [{x_positions.min():.2f}, {x_positions.max():.2f}]")
    print(f"  Y range: [{y_positions.min():.2f}, {y_positions.max():.2f}]")
    print(f"  Z range: [{z_positions.min():.2f}, {z_positions.max():.2f}]")
    print(f"  All yaw angles are 0: {np.allclose(gate_yaw, 0.0)}")
    print()
    
    # Check that gates are evenly spaced in X
    x_diffs = np.diff(x_positions)
    print(f"  X spacing: mean={x_diffs.mean():.3f}, std={x_diffs.std():.3f}")
    print(f"  Expected X spacing: {timedlr.gate_d}")
    print()
    
    # Test reproducibility
    gate_pos2, gate_yaw2 = timedlr.generate_gates(seed=seed)
    reproducible = np.allclose(gate_pos, gate_pos2) and np.allclose(gate_yaw, gate_yaw2)
    print(f"  Gates are reproducible with same seed: {reproducible}")
    print()
    
    # Test different seed produces different gates
    gate_pos3, gate_yaw3 = timedlr.generate_gates(seed=123)
    different = not np.allclose(gate_pos, gate_pos3)
    print(f"  Different seed produces different gates: {different}")
    print()
    
    return gate_pos, gate_yaw

def print_task_comparison():
    """Print comparison with other tasks."""
    print("=" * 60)
    print("Task Comparison")
    print("=" * 60)
    
    from airevolve.evolution_tools.evaluators.gate_train import (
        backandforth, figure8, circle, slalom
    )
    
    tasks = {
        'backandforth': backandforth,
        'figure8': figure8,
        'circle': circle,
        'slalom': slalom,
        'timedlr': timedlr
    }
    
    for name, task in tasks.items():
        if name == 'timedlr':
            gate_pos, gate_yaw = task.generate_gates(seed=42)
            num_gates = len(gate_pos)
        else:
            num_gates = len(task.gate_pos)
            gate_pos = task.gate_pos
        
        print(f"\n{name}:")
        print(f"  Gates: {num_gates}")
        print(f"  X bounds: {task.x_bounds}")
        print(f"  Y bounds: {task.y_bounds}")
        print(f"  Z bounds: {task.z_bounds}")
        print(f"  Starting position: {task.starting_pos}")
        if name == 'timedlr':
            print(f"  Dynamic generation: Yes")
        else:
            print(f"  Dynamic generation: No")

if __name__ == "__main__":
    print("=" * 60)
    print("TimedLR Task (Task B) Test")
    print("=" * 60)
    print()
    
    gate_pos, gate_yaw = test_gate_generation()
    print_task_comparison()
    
    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)
    print("\nTo run a training example with timedlr:")
    print("  python examples/run_learning_evaluation.py --task timedlr --task-seed 42 --timesteps 1e5")
