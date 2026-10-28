"""Generate flight C code for a policy trained with examples/learning/train_rl.py.

Same output and checks as optimal_quad_control_RL/generate_c_code.py (the
flight-verified export): neural_network.c/.h and nn_controller.c/.h, then
compiles them and compares against the Python policy open loop and in a
closed-loop simulation (skip with --no_verify). The drone, task and track
come from the run_config.json saved next to the checkpoints.

Usage:
    python examples/deployment/generate_c_code.py __data__/rl/models/hex3/hex3_r30
    python examples/deployment/generate_c_code.py __data__/rl/models/hex3/hex3_r30/50000000.zip --deterministic
"""
import argparse
import os

import numpy as np
from stable_baselines3 import PPO

from airevolve.deployment import latest_checkpoint, load_config, make_env
from airevolve.deployment.c_codegen import generate_c_code, verify_c_code

parser = argparse.ArgumentParser(description='Generate C code for a trained policy')
parser.add_argument('model', type=str, help='Checkpoint .zip, or a folder of checkpoints (uses the latest)')
parser.add_argument('--config', type=str, default=None, help='run_config.json (default: next to the checkpoint)')
parser.add_argument('--output', type=str, default=None, help='Output folder (default is c_code/<run name>)')
parser.add_argument('--deterministic', action='store_true', help='Default the controller to deterministic (no exploration noise)')
parser.add_argument('--no_verify', action='store_true', help='Skip compiling and testing the generated code')
args = parser.parse_args()

# LOAD POLICY
model_path = latest_checkpoint(args.model) if os.path.isdir(args.model) else args.model
run_dir = os.path.dirname(os.path.abspath(model_path))
run_name = os.path.basename(run_dir)
output_folder = args.output or os.path.join('c_code', run_name)
config = load_config(args.config or run_dir)
print('Loading', model_path)
model = PPO.load(model_path, device='cpu')

# ENVIRONMENT (gate data for the controller, and verification): nominal parameters
task = config['task']
overrides = dict(randomization=0.0)
if task == 'gates':
    overrides['initialize_at_random_gates'] = False
env = make_env(config, 1, **overrides)

paths = generate_c_code(model, env, task, output_folder, model_path=model_path,
                        deterministic=args.deterministic)
for path in paths:
    print(f'Generated {path}')

if not args.no_verify:
    result = verify_c_code(model, env, task, output_folder)
    episodes = result['episodes']
    print(f"\nnn_forward vs torch, max abs difference: {result['net_error']:.2e}")
    print(f"nn_control vs Python policy, max abs difference over {sum(e[0] for e in episodes)} steps: "
          f"{result['control_error']:.2e}")
    crashes = sum(e[2] for e in episodes)
    if task == 'gates':
        print(f'Closed loop with C controller: {crashes}/{len(episodes)} crashed, '
              f'{np.mean([e[1] for e in episodes]):.1f} gates per {env.max_steps*env.dt:.0f}s flight')
    else:
        print(f'Closed loop with C controller: {crashes}/{len(episodes)} crashed')
    if result['net_error'] > 1e-4 or result['control_error'] > 1e-3:
        print('WARNING: the C code does not match the Python policy')
    else:
        print('C code matches the Python policy')
