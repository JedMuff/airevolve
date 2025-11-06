"""Verify that dynamic gate generation is working by checking gate positions across resets."""
import numpy as np
import sys
sys.path.insert(0, '/home/keiichi-ito/Documents/sandbox/airevolve')

from airevolve.evolution_tools.evaluators import gate_train
from airevolve.evolution_tools.evaluators.drone_gate_env import DroneGateEnv

# Create environment with dynamic gates (using timedlr)
task_seed = 42
num_envs = 3

# Generate initial gates
gate_pos, gate_yaw = gate_train.timedlr.generate_gates(task_seed)

# Create a simple individual (just a basic drone configuration)
# We can use None and let the environment use a default drone
individual = None

env = DroneGateEnv(
    num_envs=num_envs,
    individual=individual,
    gates_pos=gate_pos,
    gate_yaw=gate_yaw,
    gate_generator=gate_train.timedlr.generate_gates,
    dynamic_gates=True,
    gates_ahead=1,
    x_bounds=gate_train.timedlr.x_bounds,
    y_bounds=gate_train.timedlr.y_bounds,
    z_bounds=gate_train.timedlr.z_bounds,
    start_pos=gate_train.timedlr.starting_pos,
    seed=task_seed,
)

# Initial gate positions
initial_gates = env.gate_pos_per_env.copy()
print("Initial gate positions for each environment:")
for i in range(num_envs):
    print(f"  Env {i}, Gate 0: {initial_gates[i, 0]}")
    print(f"  Env {i}, Gate 1: {initial_gates[i, 1]}")

# Reset all environments
obs = env.reset()

# Check gate positions after first reset
after_reset_1 = env.gate_pos_per_env.copy()
print("\nGate positions after first reset:")
for i in range(num_envs):
    print(f"  Env {i}, Gate 0: {after_reset_1[i, 0]}")
    print(f"  Env {i}, Gate 1: {after_reset_1[i, 1]}")

# Check if gates changed
changed = not np.allclose(initial_gates, after_reset_1)
print(f"\nGates changed after reset: {changed}")

# Step through to trigger another reset for one environment
# Find the shortest episode to trigger a reset
done = np.zeros(num_envs, dtype=bool)
step_count = 0
max_steps = 1000

while not np.any(done) and step_count < max_steps:
    action = np.zeros((num_envs, 4))  # Neutral action
    obs, reward, done, info = env.step(action)
    step_count += 1
    if np.any(done):
        break

if np.any(done):
    # Check gate positions after episode-triggered reset
    after_reset_2 = env.gate_pos_per_env.copy()
    print(f"\nGate positions after episode reset (step {step_count}):")
    for i in range(num_envs):
        print(f"  Env {i}, Gate 0: {after_reset_2[i, 0]}")
        print(f"  Env {i}, Gate 1: {after_reset_2[i, 1]}")
    
    # Check if gates changed for the reset environment(s)
    for i in range(num_envs):
        if done[i]:
            changed = not np.allclose(after_reset_1[i], after_reset_2[i])
            print(f"\nEnv {i} gates changed: {changed}")
            if changed:
                print("  ✓ Dynamic gate generation is working!")
            else:
                print("  ✗ Gates did not change (problem!)")
else:
    print("\nNo episode completed within max_steps")

print("\nVerification complete!")
