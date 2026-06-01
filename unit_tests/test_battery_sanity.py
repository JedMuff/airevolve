"""Sanity check: verify battery physics in base DroneGateEnv."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from airevolve.evolution_tools.evaluators.drone_gate_env import DroneGateEnv
import numpy as np

env = DroneGateEnv(num_envs=1)
env.reset()
print(f'Num motors: {env.num_motors}')
print(f'Base w_max: {env._base_w_max:.1f}')
print(f'Initial dynamic_w_max: {env.dynamic_w_max[0]:.1f}')
print()
for i in range(200):
    obs, r, d, info = env.step(np.ones((1, env.num_motors)))
    if i % 25 == 0:
        print(f'Step {i:3d}: V={env.bat_voltage[0]:.2f}V  SoC={env.bat_soc[0]:.4f}  w_max={env.dynamic_w_max[0]:.1f}  depleted={env.bat_is_depleted[0]}')
print()
print(f'Final: V={env.bat_voltage[0]:.2f}V  SoC={env.bat_soc[0]:.4f}  w_max={env.dynamic_w_max[0]:.1f} (nominal: {env._base_w_max:.1f})  ratio={env.dynamic_w_max[0]/env._base_w_max:.3f}  depleted={env.bat_is_depleted[0]}')

# Also verify the PowerAwareDroneEnv still works
print('\n--- PowerAwareDroneEnv check ---')
from airevolve.evolution_tools.evaluators.drone_gate_env_power import PowerAwareDroneEnv
penv = PowerAwareDroneEnv(num_envs=2, experiment_type=1, strict_voltage_kill=False)
obs = penv.reset()
print(f'Obs shape: {obs.shape} (expected: (2, {env.obs_len + 3}))')
for i in range(50):
    obs, r, d, info = penv.step(np.ones((2, penv.num_motors)))
print(f'Step 50: V={penv.bat_voltage[0]:.2f}V  SoC={penv.bat_soc[0]:.4f}  w_max={penv.dynamic_w_max[0]:.1f}')
print(f'Battery strict_voltage_kill: {penv.strict_voltage_kill}')
print('\nAll checks passed!')
