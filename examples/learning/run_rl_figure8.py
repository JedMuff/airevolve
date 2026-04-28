"""Train a canonical 4-motor 2-inch X-quad on the figure-8 task with PPO.

This is the runtime end-to-end RL example: a single-script PPO trainer
exercising airevolve's reference-form `DroneSimulator` (post Session 5
migration) and `DroneGateEnv` (default figure-8 at z=-1.5 NED).

Reproduces the V1 5M-step single-seed result from
`experimentation/optimal_parity_quad.py` (median ~1500 gate_passes per
100 k window after 5M steps) without the parity-experiment knobs.

Usage:
    # Quick smoke (a few minutes on RTX A4000):
    python examples/learning/run_rl_figure8.py --total-steps 1e5 --seed 1 \\
        --num-envs 50 --device cpu

    # Full training run (~30 min on GPU):
    python examples/learning/run_rl_figure8.py --total-steps 5e6 --seed 1

Outputs a tensorboard log, a `window_metrics.csv` (gate_passes / 100k
steps), and a saved policy under `--save-dir` (default `./rl_logs/`).
"""
from __future__ import annotations

import argparse
import os
import time
from datetime import datetime

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor

from airevolve.evolution_tools.evaluators.drone_gate_env import DroneGateEnv
from airevolve.simulator.simulation.propeller_data import (
    create_standard_propeller_config,
)

from window_metrics_callback import WindowMetricsCallback


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--total-steps", type=float, default=5e6,
                   help="total PPO timesteps (default 5e6)")
    p.add_argument("--num-envs", type=int, default=100,
                   help="parallel environments (default 100)")
    p.add_argument("--seed", type=int, default=None,
                   help="seed for env + PPO")
    p.add_argument("--prop-size", type=int, default=2,
                   help="propeller size (default 2-inch)")
    p.add_argument("--arm-length", type=float, default=0.11,
                   help="arm length in metres (default 0.11)")
    p.add_argument("--save-dir", default="./rl_logs",
                   help="output dir for tensorboard logs and saved policy")
    p.add_argument("--device", default="cuda:0",
                   help="torch device (default cuda:0; pass cpu for portability)")
    p.add_argument("--tag", default="",
                   help="extra tag appended to the run name")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    total_steps = int(args.total_steps)

    run_name = f"figure8_prop{args.prop_size}"
    if args.seed is not None:
        run_name += f"_seed{args.seed}"
    if args.tag:
        run_name += f"_{args.tag}"
    run_name += f"_{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    save_dir = os.path.abspath(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    print(
        f"=== run_rl_figure8: PPO on DroneGateEnv "
        f"(prop={args.prop_size}, arm={args.arm_length}m, num_envs={args.num_envs}, "
        f"total_steps={total_steps:_}, seed={args.seed}, device={args.device}) ===",
        flush=True,
    )

    propellers = create_standard_propeller_config(
        "quad", arm_length=args.arm_length, prop_size=args.prop_size
    )

    # DroneGateEnv defaults to a figure-8 at z=-1.5 (NED) with ±5m horizontal
    # bounds and -7..0m vertical headroom — this matches the reference
    # figure-8 used by optimal_quad_control_RL.
    env = DroneGateEnv(
        num_envs=args.num_envs,
        propellers=propellers,
        gates_ahead=1,
        num_state_history=0,
        num_action_history=0,
        history_step_size=1,
        render_mode=None,
        device=args.device,
        dt=0.01,
        seed=args.seed,
    )
    obs = env.reset()
    print(
        f"obs shape: {obs.shape}, num_motors: {env.num_motors}, "
        f"mass: {env.drone_sim.mass:.3f} kg, motor_tau: {env.motor_tau}",
        flush=True,
    )

    monitor_path = os.path.join(save_dir, run_name)
    env = VecMonitor(env, filename=monitor_path)

    # PPO hyperparameters from `airevolve/evolution_tools/evaluators/gate_train.py`
    # (which mirror optimal_quad_control_RL/train.py:149-161).
    policy_kwargs = dict(
        activation_fn=torch.nn.ReLU,
        net_arch=dict(pi=[64, 64], vf=[64, 64]),
        log_std_init=0.0,
    )
    # Note: not forwarding `seed` to PPO. DroneGateEnv stores `self.seed`
    # as an int, which shadows the VecEnv `.seed()` method that PPO would
    # otherwise call during `set_random_seed`. The env-side seed (set via
    # the env constructor above) is what controls reset determinism.
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=0,
        tensorboard_log=save_dir,
        n_steps=1000,
        batch_size=5000,
        n_epochs=10,
        gamma=0.999,
        device=args.device,
    )

    csv_path = os.path.join(save_dir, run_name + "_window_metrics.csv")
    cb = WindowMetricsCallback(window=100_000, csv_path=csv_path)

    t0 = time.time()
    model.learn(total_timesteps=total_steps, callback=cb, tb_log_name=run_name)
    elapsed = time.time() - t0

    save_path = os.path.join(save_dir, run_name + "_final.zip")
    model.save(save_path)

    print(
        f"=== done in {elapsed:.1f}s ({total_steps / elapsed:.0f} steps/s). "
        f"saved {save_path} ===",
        flush=True,
    )
    print(f"window-metrics CSV: {csv_path}", flush=True)


if __name__ == "__main__":
    main()
