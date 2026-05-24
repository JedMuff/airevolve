"""Render top + iso videos of the max-gate rollouts from the multirollout
eval (analysis/final_policy_multirollout_n5.csv).

Reproduces a specific (morph, train_seed, env_seed) rollout by building
the env with `initialize_at_random_gates=True` and the recorded env seed,
then streams frames through the OpenCV `view()` renderer.

Usage:
    .venv/bin/python experimentation/make_best_rollout_videos.py
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

import numpy as np
import torch
from stable_baselines3 import PPO

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from airevolve.evolution_tools.evaluators.drone_gate_env import DroneGateEnv  # noqa: E402
from airevolve.simulator.visualization.animation import view as animation_view  # noqa: E402
from airevolve.simulator.simulation.propeller_data import (  # noqa: E402
    create_standard_propeller_config,
)


VIDEOS_DIR = "__data__/test_a_multiseed//videos"
TEST_A_DIR = "__data__/test_a_multiseed/"

GENOME_PATHS = {
    "ind0903": ("/Users/mikolajduchlinski/Desktop/results_folder_update/standard_ppo_power_ea/exp_standard_ppo_power_ea/rl_logs/generation_28/individual_0903/genome.npy"),
}


@dataclass
class Target:
    morph: str          # "quad" | "hex" | "ind1200"
    reward: str         # "pergate" | "finalgate"
    policy_zip: str
    train_seed: int
    env_seed: int
    expected_gates: int


TARGETS = [
    Target("ind0903", "finalgate",
           "/Users/mikolajduchlinski/Desktop/results_folder_update/standard_ppo_power_ea/exp_standard_ppo_power_ea/rl_logs/generation_28/individual_0903/policy.zip",
           5, 1001, 31),
]


def _load_genome_arms(path: str) -> np.ndarray:
    obj = np.load(path, allow_pickle=True)
    inner = obj.item() if obj.dtype == object else obj
    arms = inner.arms if hasattr(inner, "arms") else inner
    return np.asarray(arms, dtype=float)


def _build_env(morph: str, env_seed: int, device: str) -> DroneGateEnv:
    common = dict(
        num_envs=1, gates_ahead=1, num_state_history=0, num_action_history=0,
        history_step_size=1, render_mode=None, device=device, dt=0.01,
        initialize_at_random_gates=True, seed=env_seed,
    )
    if morph == "quad":
        return DroneGateEnv(propellers=create_standard_propeller_config("quad", 0.11, 2), **common)
    if morph == "hex":
        return DroneGateEnv(propellers=create_standard_propeller_config("hex", 0.11, 2), **common)
    if morph in GENOME_PATHS:
        arms = _load_genome_arms(os.path.join(REPO_ROOT, GENOME_PATHS[morph]))
        return DroneGateEnv(individual=arms, **common)
    raise ValueError(f"unknown morph: {morph}")


def render_target(t: Target, view_type: str, device: str, steps: int) -> str:
    env = _build_env(t.morph, t.env_seed, device)
    propellers = env.drone_sim.config.propellers
    policy_path = t.policy_zip
    model = PPO.load(policy_path, env=env, device=device)

    out_dir = os.path.join(REPO_ROOT, VIDEOS_DIR)
    os.makedirs(out_dir, exist_ok=True)
    out_name = f"best_{t.morph}_{t.reward}_seed{t.train_seed}_env{t.env_seed}_{t.expected_gates}gates_{view_type}.mp4"
    out_path = os.path.join(out_dir, out_name)
    print(f"rendering {t.morph} {t.reward} seed{t.train_seed} env{t.env_seed} "
          f"({t.expected_gates} gates expected, {view_type} view) → {out_path}",
          flush=True)

    env.reset()
    num_motors = len(propellers)

    def get_drone_state():
        actions, _ = model.predict(env.states, deterministic=True)
        env.step(actions)
        ws = env.world_states[0]
        state = {
            "x": ws[0], "y": ws[1], "z": ws[2],
            "phi": ws[6], "theta": ws[7], "psi": ws[8],
        }
        for i in range(num_motors):
            state[f"u{i+1}"] = (
                env.prev_actions[0][i] if i < env.prev_actions.shape[1] else 0
            )
        return state

    animation_view(
        propellers, get_drone_state,
        gate_pos=env.gate_pos, gate_yaw=env.gate_yaw,
        view_type=view_type,
        record_steps=steps, record_file=out_path,
        show_window=False, follow=True,
        draw_forces=True, draw_path=True,
        auto_play=True, record=True,
        motor_colors=["red", "blue", "green", "orange", "purple", "brown"],
    )
    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--device",
                   default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--views", nargs="+", default=["top", "iso"],
                   choices=["top", "iso"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    for t in TARGETS:
        for view in args.views:
            render_target(t, view, args.device, args.steps)
    print("done", flush=True)


if __name__ == "__main__":
    main()
