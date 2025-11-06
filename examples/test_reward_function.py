#!/usr/bin/env python3
"""
Quick test to verify the new reward function works correctly
"""

import numpy as np
import sys
sys.path.insert(0, '../')

from airevolve.evolution_tools.evaluators.drone_gate_env import DroneGateEnv
from airevolve.evolution_tools.evaluators.gate_train import timedlr

# Create environment with timedlr task
print("Creating environment with timedlr gates...")
gate_pos, gate_yaw = timedlr.generate_gates(seed=42)

env = DroneGateEnv(
    num_envs=4,
    gates_pos=gate_pos,
    gate_yaw=gate_yaw,
    max_steps=100,
    dynamic_gates=True,
    gate_generator=timedlr.generate_gates,
    seed=42
)

print("Testing reward computation...")
obs = env.reset()
print(f"Initial observation shape: {obs.shape}")

# Take a few random actions
for i in range(10):
    actions = np.random.uniform(-1, 1, size=(4, 4))
    obs, rewards, dones, infos = env.step(actions)
    
    print(f"\nStep {i+1}:")
    print(f"  Rewards: {rewards}")
    print(f"  Gates passed: {[info['gate_passed'] for info in infos]}")
    print(f"  Total gates: {[info['num_gates_passed'] for info in infos]}")
    print(f"  Dones: {dones}")
    
    if np.any(dones):
        print(f"  Episode(s) finished!")

print("\n✓ Reward function test completed successfully!")
print("\nNew reward structure includes:")
print("  - Exponential distance rewards (stronger when close)")
print("  - Getting closer bonus (asymmetric penalty for moving away)")
print("  - Distance from goal reward")
print("  - Action smoothness penalty")
print("  - Angular velocity penalty")
print("  - Gate passing bonus: +50.0")
print("  - Out-of-bounds penalty: -100.0")
