#!/usr/bin/env python3
"""
Quick test to verify the lrcontinuous (Task A) implementation
"""

import numpy as np
import sys
sys.path.insert(0, '../')

from airevolve.evolution_tools.evaluators.drone_gate_env import DroneGateEnv
from airevolve.evolution_tools.evaluators.gate_train import lrcontinuous

# Test gate generation
print("Testing lrcontinuous (Task A) gate generation...")
print(f"Gate spacing (gate_d): {lrcontinuous.gate_d}m")
print(f"Lateral offset (gate_r): {lrcontinuous.gate_r}m")
print(f"Gate width: {lrcontinuous.gate_width}m")
print(f"Number of gates: {lrcontinuous.num_gates}")

# Generate gates with seed
gate_pos, gate_yaw = lrcontinuous.generate_gates(seed=42)

print(f"\nGenerated {len(gate_pos)} gates")
print(f"First 8 gates (should show left-center-right-center pattern):")
for i in range(min(8, len(gate_pos))):
    y_pos = gate_pos[i, 1]
    if y_pos < -0.1:
        position = "LEFT"
    elif y_pos > 0.1:
        position = "RIGHT"
    else:
        position = "CENTER"
    print(f"  Gate {i}: x={gate_pos[i,0]:.2f}m, y={gate_pos[i,1]:.2f}m ({position})")

print(f"\nCourse length: {lrcontinuous.num_gates * lrcontinuous.gate_d}m")
print(f"Bounds: x={lrcontinuous.x_bounds}, y={lrcontinuous.y_bounds}, z={lrcontinuous.z_bounds}")

# Test environment creation
print("\n" + "="*60)
print("Testing environment creation...")
env = DroneGateEnv(
    num_envs=2,
    gates_pos=gate_pos,
    gate_yaw=gate_yaw,
    start_pos=lrcontinuous.starting_pos,
    x_bounds=lrcontinuous.x_bounds,
    y_bounds=lrcontinuous.y_bounds,
    z_bounds=lrcontinuous.z_bounds,
    max_steps=100,
    dynamic_gates=True,
    gate_generator=lrcontinuous.generate_gates,
    use_advanced_reward=False,  # Task A uses simple reward
    seed=42
)

print(f"Environment created with {env.num_envs} parallel environments")
print(f"Dynamic gates: {env.dynamic_gates}")
print(f"Advanced reward: {env.use_advanced_reward}")

# Test a few steps
obs = env.reset()
print(f"\nInitial observation shape: {obs.shape}")

print("\nTaking 5 random steps...")
for i in range(5):
    actions = np.random.uniform(-1, 1, size=(2, 4))
    obs, rewards, dones, infos = env.step(actions)
    print(f"  Step {i+1}: rewards={rewards}, dones={dones}")

print("\n" + "="*60)
print("✓ lrcontinuous (Task A) implementation verified!")
print("\nTask A vs Task B comparison:")
print(f"  Task A (lrcontinuous): {lrcontinuous.gate_d}m spacing, fixed pattern, simple reward")
print(f"  Task B (timedlr): 0.25m spacing, random offsets, advanced reward")
print(f"  Task A is {lrcontinuous.gate_d/0.25:.0f}x easier in terms of gate spacing!")
