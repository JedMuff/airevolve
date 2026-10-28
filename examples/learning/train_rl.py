"""Train a PPO policy for a real drone (or any morphology) on gates or hover.

airevolve port of optimal_quad_control_RL/train_real.py, the flight-verified
training setup: same env, reward, domain randomization, network and PPO
hyperparameters. Checkpoints are saved every 10 rollouts together with a
run_config.json, which examples/deployment/generate_c_code.py uses to
export the policy as flight C code.

Usage:
    # The real hex (sysid mode), ±30% randomization, like hex3_gate_rand30:
    python examples/learning/train_rl.py hex3 hex3_gate_r30 --drone real_hex3 --randomization 30

    # Hover task:
    python examples/learning/train_rl.py hex3 hex3_hover_r30 --drone real_hex3 --task hover --randomization 30

    # Later identification of the hex (params_vuhex):
    python examples/learning/train_rl.py vuhex vuhex_gate_r30 --drone real_vuhex --randomization 30

    # A custom morphology with the real hex's motors (physics mode):
    python examples/learning/train_rl.py evolved my_drone --genome my_genome.npy --propsize hex3 --randomization 30

Outputs (under --save_dir, default __data__/rl):
    models/<session>/<name>/<timesteps>.zip and run_config.json
    logs/<session>/<name>_0/   (tensorboard)
"""
import argparse
import json
import os
import shutil
from time import time

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor

from airevolve.deployment import drone_config, run_config, save_config, make_env

parser = argparse.ArgumentParser(description='Training session configuration')
parser.add_argument('session_name', type=str, help='Name of the training session')
parser.add_argument('name', type=str, help='Name of the model')
parser.add_argument('--drone', type=str, default='real_hex3',
                    help='real_hex3 (sysid of the flight-verified hex), real_vuhex (params_vuhex), quad, hex, tri or octo (default real_hex3)')
parser.add_argument('--w_norm_max', type=float, default=None,
                    help='Motor-speed normalization (rad/s) shared with the C code (default: real_hex3 4700, real_vuhex 4000, else 4700)')
parser.add_argument('--genome', type=str, default=None, help='Spherical-angular genome (.npy) to train instead of --drone')
parser.add_argument('--propellers', type=str, default=None, help='Propeller list (.json) to train instead of --drone')
parser.add_argument('--propsize', type=str, default=None,
                    help='Prop-library entry for physics mode: 2-8, matched, or hex3 (real hex motors)')
parser.add_argument('--arm_length', type=float, default=0.11, help='Arm length (m) for the standard layouts')
parser.add_argument('--pi', type=int, nargs='+', default=[64, 64, 64], help='Architecture of the policy (default 64 64 64)')
parser.add_argument('--vf', type=int, nargs='+', default=[64, 64, 64], help='Architecture of the value function (default 64 64 64)')
parser.add_argument('--randomization', type=float, default=0, help='Domain randomization in percent (e.g. 30 for ±30%%, default 0)')
parser.add_argument('--task', type=str, default='gates', choices=['gates', 'hover'], help='Task to train (default is gates)')
parser.add_argument('--yaw_rate_penalty', type=float, default=0.05, help='Hover: reward penalty per rad/s of yaw rate, stops the policy spinning (default 0.05, 0 = reference reward)')
parser.add_argument('--timesteps', type=float, default=1e8, help='Training timesteps (default is 1e8)')
parser.add_argument('--r', type=float, default=1.5, help='Track radius (default is 1.5)')
parser.add_argument('--num_envs', type=int, default=100, help='Parallel environments (default 100)')
parser.add_argument('--transfer_learning', type=str, default=None, help='Checkpoint .zip to continue training from')
parser.add_argument('--seed', type=int, default=None, help='Seed for the env')
parser.add_argument('--device', type=str, default='auto', help='Torch device for PPO (default auto)')
parser.add_argument('--save_dir', type=str, default='__data__/rl', help='Output folder (default __data__/rl)')
args = parser.parse_args()


def parse_propsize(value):
    return int(value) if value is not None and value.isdigit() else value


# DRONE AND RUN CONFIG
if args.genome is not None:
    drone = drone_config(os.path.basename(args.genome), genome=np.load(args.genome),
                         propsize=parse_propsize(args.propsize), w_norm_max=args.w_norm_max)
elif args.propellers is not None:
    with open(args.propellers) as f:
        drone = drone_config(os.path.basename(args.propellers), propellers=json.load(f),
                             propsize=parse_propsize(args.propsize), w_norm_max=args.w_norm_max)
else:
    drone = drone_config(args.drone, propsize=parse_propsize(args.propsize), arm_length=args.arm_length,
                         w_norm_max=args.w_norm_max)
config = run_config(drone, task=args.task, track_radius=args.r, randomization=args.randomization / 100,
                    yaw_rate_penalty=args.yaw_rate_penalty)

# SETUP LOGGING
models_dir = os.path.join(args.save_dir, 'models', args.session_name)
log_dir = os.path.join(args.save_dir, 'logs', args.session_name)
run_dir = os.path.join(models_dir, args.name)
for folder in [log_dir + '/' + args.name + '_0', run_dir]:
    if os.path.exists(folder):
        print('Deleting', folder)
        shutil.rmtree(folder, ignore_errors=True)
save_config(config, run_dir)
os.makedirs(log_dir, exist_ok=True)

print("Training session configuration:")
print(f"Session name: {args.session_name}")
print(f"Model name: {args.name}")
print(f"Drone: {drone['name']} ({len(drone['propellers'])} motors, "
      f"{'sysid' if drone['sysid_params'] else 'physics'} mode, w_norm_max={drone['w_norm_max']})")
print(f"Policy architecture: {args.pi}")
print(f"Value function architecture: {args.vf}")
print(f"Randomization: ±{args.randomization}%")
print(f"Task: {args.task}" + (f" (yaw rate penalty {args.yaw_rate_penalty})" if args.task == 'hover' else ""))

# CREATE ENVIRONMENTS
env = make_env(config, args.num_envs, seed=args.seed)
env = VecMonitor(env)

# Tensorboard logging when it is installed
try:
    import tensorboard  # noqa: F401
except ImportError:
    print("tensorboard not installed: training without tensorboard logs")
    log_dir = None

# MODEL DEFINITION
policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=dict(pi=args.pi, vf=args.vf), log_std_init=0)
# Note: not forwarding `seed` to PPO. DroneGateEnv stores `self.seed`
# as an int, which shadows the VecEnv `.seed()` method that PPO would
# otherwise call during `set_random_seed`.
model = PPO(
    "MlpPolicy",
    env,
    policy_kwargs=policy_kwargs,
    verbose=0,
    tensorboard_log=log_dir,
    n_steps=1000,
    batch_size=5000,
    n_epochs=10,
    gamma=0.999,
    device=args.device,
)

# Load model for transfer learning
if args.transfer_learning is not None:
    model = PPO.load(args.transfer_learning, env=env, tensorboard_log=log_dir, device=args.device)

print(model.policy)
print("Logging to", log_dir)
print("Saving models to", run_dir)


# TRAINING
# training loop saves model every 10 policy rollouts
def train(model, log_name, n=int(args.timesteps)):
    start_time = time()
    # save every 10 policy rollouts
    TIMESTEPS = model.n_steps*args.num_envs*10
    reset = True
    while model.num_timesteps < n:
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=reset, tb_log_name=log_name)
        reset = False
        time_steps = model.num_timesteps
        model.save(os.path.join(run_dir, str(time_steps)))
        print(f'Time elapsed: {(time() - start_time):.2f}s')
        print(f'Model saved at {run_dir}/{time_steps}')


print("Training model", args.name)
train(model, args.name)
