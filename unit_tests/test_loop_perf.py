import time
import numpy as np
import sys
from pathlib import Path
import os
import warnings

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from airevolve.evolution_tools.evaluators.drone_gate_env import DroneGateEnv

num_envs = 4000
env = DroneGateEnv(num_envs=num_envs)
env.reset()
actions = np.zeros((num_envs, env.num_motors))

# Warmup
for _ in range(10):
    env.step_async(actions)
    env.step_wait()

t0 = time.time()
for _ in range(100):
    env.step_async(actions)
    env.step_wait()
t1 = time.time()

print(f"Time for 100 steps: {t1-t0:.4f}s")
print(f"Time per step: {(t1-t0)/100:.6f}s")
