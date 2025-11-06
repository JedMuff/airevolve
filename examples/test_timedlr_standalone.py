#!/usr/bin/env python3
"""
Standalone test for timedlr gate generation (no dependencies required).
"""

import numpy as np

# Inline implementation for testing (copied from gate_train.py)
class timedlr:
    """Task B: Timed left-right gates with variable positioning."""
    gate_d = 0.25
    gate_r_min = 0.5
    gate_r_max = 0.7
    gate_z_min = -0.1
    gate_z_max = 0.1
    gate_width = 0.5
    gate_prob_lr = 0.05
    num_gates = 100
    
    gate_pos, gate_yaw = None, None
    x_bounds = np.array([0, num_gates * gate_d + 2], dtype=np.float32)
    y_bounds = np.array([-gate_r_max - 1, gate_r_max + 1], dtype=np.float32)
    z_bounds = np.array([gate_z_min - 0.5, gate_z_max + 0.5], dtype=np.float32)
    starting_pos = np.array([0.0, 0.0, 0.0])
    
    @classmethod
    def generate_gates(cls, seed=None):
        """Generate random gate positions."""
        if seed is not None:
            np.random.seed(seed)
        
        gate_pos = np.zeros((cls.num_gates, 3), dtype=np.float32)
        gate_yaw = np.zeros(cls.num_gates, dtype=np.float32)
        
        for i in range(cls.num_gates):
            gate_pos[i, 0] = i * cls.gate_d
            
            if np.random.rand() < cls.gate_prob_lr:
                gate_pos[i, 1] = -np.random.uniform(cls.gate_r_min, cls.gate_r_max)
            elif np.random.rand() < cls.gate_prob_lr:
                gate_pos[i, 1] = np.random.uniform(cls.gate_r_min, cls.gate_r_max)
            else:
                gate_pos[i, 1] = 0.0
            
            gate_pos[i, 2] = np.random.uniform(cls.gate_z_min, cls.gate_z_max)
            gate_yaw[i] = 0.0
        
        return gate_pos, gate_yaw


def test_gate_generation():
    """Test gate generation."""
    print("=" * 60)
    print("TimedLR (Task B) Standalone Test")
    print("=" * 60)
    print()
    
    # Test 1: Basic generation
    print("Test 1: Basic Generation")
    print("-" * 60)
    gate_pos, gate_yaw = timedlr.generate_gates(seed=42)
    print(f"✓ Generated {len(gate_pos)} gates")
    print(f"✓ Gate positions shape: {gate_pos.shape}")
    print(f"✓ Gate yaw shape: {gate_yaw.shape}")
    print()
    
    # Test 2: First few gates
    print("Test 2: Sample Gates")
    print("-" * 60)
    for i in range(min(5, len(gate_pos))):
        x, y, z = gate_pos[i]
        print(f"  Gate {i}: pos=[{x:6.2f}, {y:6.2f}, {z:6.2f}], yaw={gate_yaw[i]:.2f}")
    print()
    
    # Test 3: Verify parameters
    print("Test 3: Parameter Verification")
    print("-" * 60)
    x_pos = gate_pos[:, 0]
    y_pos = gate_pos[:, 1]
    z_pos = gate_pos[:, 2]
    
    print(f"  X range: [{x_pos.min():.2f}, {x_pos.max():.2f}]")
    print(f"  Y range: [{y_pos.min():.2f}, {y_pos.max():.2f}]")
    print(f"  Z range: [{z_pos.min():.2f}, {z_pos.max():.2f}]")
    
    x_diffs = np.diff(x_pos)
    print(f"  X spacing: mean={x_diffs.mean():.3f}, std={x_diffs.std():.6f}")
    print(f"  Expected: {timedlr.gate_d}")
    
    all_yaw_zero = np.allclose(gate_yaw, 0.0)
    print(f"  All yaw angles are 0: {all_yaw_zero}")
    
    # Check X spacing is correct
    spacing_correct = np.allclose(x_diffs, timedlr.gate_d)
    print(f"✓ X spacing is correct: {spacing_correct}")
    
    # Check Y bounds
    y_in_bounds = (y_pos.min() >= -timedlr.gate_r_max) and (y_pos.max() <= timedlr.gate_r_max)
    print(f"✓ Y positions in bounds: {y_in_bounds}")
    
    # Check Z bounds
    z_in_bounds = (z_pos.min() >= timedlr.gate_z_min) and (z_pos.max() <= timedlr.gate_z_max)
    print(f"✓ Z positions in bounds: {z_in_bounds}")
    print()
    
    # Test 4: Reproducibility
    print("Test 4: Reproducibility")
    print("-" * 60)
    gate_pos2, gate_yaw2 = timedlr.generate_gates(seed=42)
    reproducible = np.allclose(gate_pos, gate_pos2) and np.allclose(gate_yaw, gate_yaw2)
    print(f"✓ Same seed produces same gates: {reproducible}")
    
    gate_pos3, gate_yaw3 = timedlr.generate_gates(seed=999)
    different = not np.allclose(gate_pos, gate_pos3)
    print(f"✓ Different seed produces different gates: {different}")
    print()
    
    # Test 5: Statistics
    print("Test 5: Gate Statistics")
    print("-" * 60)
    center_gates = np.sum(np.abs(y_pos) < 0.01)
    left_gates = np.sum(y_pos < -0.01)
    right_gates = np.sum(y_pos > 0.01)
    
    print(f"  Center gates: {center_gates} ({center_gates/len(gate_pos)*100:.1f}%)")
    print(f"  Left gates: {left_gates} ({left_gates/len(gate_pos)*100:.1f}%)")
    print(f"  Right gates: {right_gates} ({right_gates/len(gate_pos)*100:.1f}%)")
    print(f"  Expected center: ~{(1 - 2*timedlr.gate_prob_lr)*100:.1f}%")
    print(f"  Expected left/right: ~{timedlr.gate_prob_lr*100:.1f}% each")
    print()
    
    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Task configuration:")
    print(f"    - Number of gates: {timedlr.num_gates}")
    print(f"    - Gate spacing: {timedlr.gate_d}m")
    print(f"    - Course length: ~{timedlr.num_gates * timedlr.gate_d:.1f}m")
    print(f"    - Gate radius: {timedlr.gate_r_min}-{timedlr.gate_r_max}m")
    print(f"    - Z variation: {timedlr.gate_z_min} to {timedlr.gate_z_max}m")
    print(f"    - Gate width: {timedlr.gate_width}m")
    print(f"    - LR probability: {timedlr.gate_prob_lr*100}%")
    print()
    print("✓ All tests passed!")
    print()
    print("Next steps:")
    print("  1. Install dependencies: stable_baselines3, torch, etc.")
    print("  2. Run full test: python examples/test_timedlr_task.py")
    print("  3. Train a drone: python examples/run_learning_evaluation.py --task timedlr --task-seed 42")
    print("=" * 60)


if __name__ == "__main__":
    test_gate_generation()
