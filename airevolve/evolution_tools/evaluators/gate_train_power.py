"""Bi-objective RL gate-racing evaluator.

Extends gate_train.py to measure both:
  - num_gates_passed : int   — gates passed during the deterministic eval flight.
  - total_energy_j   : float — Joules consumed by a dedicated LiPoBatteryModel
                               over the same fixed 12-second (max_steps steps) window.

The battery is stepped independently from the physics environment so that:
  1. Energy accumulates over the full fixed evaluation window regardless of
     mid-flight episode resets (e.g. out-of-bounds terminations).
  2. The training phase supports three reward formulations via experiment_type:
       0 — Baseline: standard DroneGateEnv, no energy term in the reward.
       1 — Dense   : PowerAwareDroneEnv with per-step power penalty.
       2 — Sparse  : PowerAwareDroneEnv with end-of-episode energy penalty.
     The evaluation phase is identical for all three (fair comparison).

Usage (as a module):
    num_gates, energy_j = train_power(individual, "figure8", total_timesteps=1e6,
                                      save_dir="./logs", num_envs=4, device="cuda:0")

Usage (subprocess):
    python gate_train_power.py <dir> --training_timesteps 1e6 --num_envs 4 \
        --gate_cfg figure8 --device cuda:0
    stdout: "<num_gates> <total_energy_j>"
"""

import os
import sys
import time
import argparse
import warnings

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.callbacks import BaseCallback

warnings.filterwarnings("ignore", message="The `render_mode` attribute is not defined in your environment")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from airevolve.evolution_tools.evaluators.drone_gate_env import DroneGateEnv
from airevolve.evolution_tools.evaluators.drone_gate_env_power import PowerAwareDroneEnv
from airevolve.evolution_tools.evaluators.gate_train import (
    backandforth, figure8, circle, slalom, FullStatsCallback,
)
from airevolve.evolution_tools.inspection_tools.morphological_descriptors.hovering_info import (
    get_sim,
)
from airevolve.evolution_tools.inspection_tools.drone_visualizer import DroneVisualizer
from airevolve.simulator.simulation.battery_model import LiPoBatteryModel

# Energy returned for morphologies that cannot hover — large enough to guarantee
# domination by any individual that passes ≥ 1 gate.
_FAIL_ENERGY: float = 1e9


def train_power(
    individual,
    gate_cfg: str,
    total_timesteps: int,
    save_dir: str,
    num_envs: int,
    device: str = "cuda:0",
    num=None,
    max_steps: int = 1200,
    experiment_type: int = 0,
    penalty_weights: dict = None,
):
    """Train a PPO policy for gate racing and return bi-objective fitness.

    Parameters
    ----------
    individual       : numpy array — the drone morphology (arm matrix).
    gate_cfg         : gate track name ('backandforth', 'figure8', 'circle', 'slalom').
    total_timesteps  : PPO training budget.
    save_dir         : directory for model checkpoints and plots.
    num_envs         : number of parallel training environments.
    device           : torch device string.
    num              : optional integer suffix for multi-task evaluation filenames.
    max_steps        : evaluation length in env steps (default 1200 → 12 s at dt=0.01).
    experiment_type  : reward formulation used during training:
                         0 — Baseline (standard RL, no energy term).
                         1 — Dense penalty (per-step power penalty).
                         2 — Sparse penalty (end-of-episode energy penalty).
    penalty_weights  : dict forwarded to PowerAwareDroneEnv.  Keys:
                         experiment_type=1 → {"dense_weight": float}
                         experiment_type=2 → {"sparse_weight": float}
                       Ignored when experiment_type=0.

    Returns
    -------
    (num_gates_passed, total_energy_j) : (int, float)
    """
    cfg_map = {
        "backandforth": backandforth,
        "figure8":      figure8,
        "circle":       circle,
        "slalom":       slalom,
    }
    if gate_cfg not in cfg_map:
        raise ValueError(f"Unknown gate_cfg: {gate_cfg!r}")
    cfg = cfg_map[gate_cfg]

    gate_pos   = cfg.gate_pos
    gate_yaw   = cfg.gate_yaw
    start_pos  = cfg.starting_pos
    x_bounds   = cfg.x_bounds
    y_bounds   = cfg.y_bounds
    z_bounds   = cfg.z_bounds

    save_dir = save_dir + "/"
    os.makedirs(save_dir, exist_ok=True)

    # ── Training environment ──────────────────────────────────────────────────
    # Baseline (exp 0): standard DroneGateEnv — no power overhead.
    # Dense/Sparse (exp 1/2): PowerAwareDroneEnv with the requested penalty.
    _env_kwargs = dict(
        num_envs=num_envs,
        individual=individual,
        gates_pos=gate_pos,
        gate_yaw=gate_yaw,
        start_pos=start_pos,
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        z_bounds=z_bounds,
        gates_ahead=1,
        num_state_history=0,
        num_action_history=0,
        history_step_size=1,
        render_mode=None,
        device=device,
        max_steps=max_steps,
    )
    if experiment_type in (1, 2):
        env = PowerAwareDroneEnv(
            experiment_type=experiment_type,
            penalty_weights=penalty_weights or {},
            randomize_soc=True,
            **_env_kwargs,
        )
    else:
        env = DroneGateEnv(**_env_kwargs)

    monitor_file = save_dir + (f"m{num}" if num is not None else "")
    env = VecMonitor(env, filename=monitor_file)

    policy_kwargs = dict(
        activation_fn=torch.nn.ReLU,
        net_arch=dict(pi=[64, 64], vf=[64, 64]),
        log_std_init=0.0,
    )
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
        device=device,
    )
    model.learn(
        total_timesteps=int(total_timesteps),
        reset_num_timesteps=False,
        log_interval=100,
        callback=FullStatsCallback(),
    )
    policy_path = save_dir + (f"policy{num}" if num is not None else "policy")
    model.save(policy_path)

    # ── Training curve plot ───────────────────────────────────────────────────
    try:
        try:
            data = pd.read_csv(monitor_file + ".monitor.csv", skiprows=1)
        except Exception:
            data = pd.read_csv(monitor_file + "monitor.csv", skiprows=1)
        plt.figure(figsize=(10, 6))
        plt.plot(data["t"], data["r"], label="Episode Reward")
        plt.xlabel("Timesteps")
        plt.ylabel("Reward")
        plt.title("Reward per Episode")
        plt.legend()
        fig_path = save_dir + (f"figure{num}.png" if num is not None else "figure.png")
        plt.savefig(fig_path)
        plt.close()
    except Exception:
        pass

    # ── Deterministic evaluation: fixed max_steps window ─────────────────────
    # We use a single-env DroneGateEnv paired with an independent LiPoBatteryModel.
    # The battery accumulates energy across the full max_steps window regardless
    # of any episode resets triggered by out-of-bounds terminations.
    test_env = DroneGateEnv(
        num_envs=1,
        individual=individual,
        gates_pos=gate_pos,
        gate_yaw=gate_yaw,
        start_pos=start_pos,
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        z_bounds=z_bounds,
        initialize_at_random_gates=False,
        gates_ahead=1,
        num_state_history=0,
        num_action_history=0,
        history_step_size=1,
        render_mode=None,
        device=device,
        max_steps=max_steps,
    )

    test_env.reset()
    battery = LiPoBatteryModel()
    battery.reset()

    dt = float(test_env.dt)

    for _ in range(max_steps):
        # Capture state BEFORE the physics step (consistent with PowerAwareDroneEnv)
        pre_state = test_env.world_states[0].copy()
        actions, _ = model.predict(test_env.states, deterministic=True)
        _states, _rewards, _dones, infos = test_env.step(actions)
        battery.step(dt, pre_state)

    num_gates_passed = int(infos[0]["num_gates_passed"][0])
    total_energy_j   = float(battery.get_total_energy_consumed())

    return num_gates_passed, total_energy_j


def evaluate_individual(
    individual,
    ind_save_dir: str,
    training_ts,
    num_envs,
    gate_cfg: str,
    device: str = "cuda:0",
    num=None,
    max_steps: int = 1200,
    experiment_type: int = 0,
    penalty_weights: dict = None,
) -> tuple:
    """Hover-check, train, and evaluate one morphology.

    Returns (0, FAIL_ENERGY) immediately for non-hoverable morphologies so that
    the caller receives a consistently-typed tuple in all cases.

    Returns
    -------
    (num_gates_passed, total_energy_j) : (int, float)
    """
    start_time = time.time()

    sim = get_sim(individual)
    sim.compute_hover(verbose=False)
    hoverable = sim.static_success

    visualizer = DroneVisualizer()

    if not hoverable:
        try:
            fig = plt.figure(figsize=plt.figaspect(0.5))
            ax  = fig.add_subplot(111, projection="3d")
            visualizer.plot_3d(
                individual, ax=ax,
                title=f"Failed (Gen {num})", fitness=0, generation=num,
            )
            plt.savefig(
                ind_save_dir + (f"/morphology{num}.png" if num is not None else "/morphology.png")
            )
            plt.close()
        except Exception:
            print(f"Failed to plot:\n {individual}")
        return (0, _FAIL_ENERGY)

    # Pre-training morphology plot
    try:
        fig = plt.figure(figsize=plt.figaspect(0.5))
        ax  = fig.add_subplot(111, projection="3d")
        visualizer.plot_3d(
            individual, ax=ax,
            title=f"Pre-training (Gen {num})", fitness=np.nan, generation=num,
        )
        plt.savefig(
            ind_save_dir + (f"/morphology{num}.png" if num is not None else "/morphology.png")
        )
        plt.close()
    except Exception:
        pass

    num_gates_passed, total_energy_j = train_power(
        individual,
        gate_cfg,
        total_timesteps=int(float(training_ts)),
        save_dir=ind_save_dir,
        num_envs=int(num_envs),
        device=device,
        num=num,
        max_steps=max_steps,
        experiment_type=experiment_type,
        penalty_weights=penalty_weights,
    )

    # Post-training morphology plot
    try:
        fig = plt.figure(figsize=plt.figaspect(0.5))
        ax  = fig.add_subplot(111, projection="3d")
        visualizer.plot_3d(
            individual, ax=ax,
            title=f"Post-training (Gen {num})",
            fitness=num_gates_passed,
            generation=num,
        )
        plt.savefig(
            ind_save_dir + (f"/morphology{num}.png" if num is not None else "/morphology.png")
        )
        plt.close()
    except Exception:
        pass

    elapsed = time.time() - start_time
    print(
        f"Evaluated in {elapsed:.1f}s — gates={num_gates_passed}, energy={total_energy_j:.1f}J",
        flush=True,
    )
    return (num_gates_passed, total_energy_j)


# ── Subprocess entry-point ────────────────────────────────────────────────────

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    parser.add_argument("--training_timesteps", default=1e8)
    parser.add_argument("--num_envs",           default=100)
    parser.add_argument("--gate_cfg",           default="figure8")
    parser.add_argument("--device",             default="cuda:0")
    parser.add_argument("--num",                default=None)
    parser.add_argument("--max_steps",        default=1200,  type=int)
    parser.add_argument("--experiment_type",  default=0,     type=int,
                        help="0=baseline, 1=dense, 2=sparse")
    parser.add_argument("--dense_weight",     default=0.0,   type=float)
    parser.add_argument("--sparse_weight",    default=0.0,   type=float)
    args = parser.parse_args()

    individual = np.load(args.filename + "/individual.npy", allow_pickle=True).astype(np.float32)
    num        = int(args.num) if args.num is not None else None

    penalty_weights: dict = {}
    if args.experiment_type == 1:
        penalty_weights = {"dense_weight": args.dense_weight}
    elif args.experiment_type == 2:
        penalty_weights = {"sparse_weight": args.sparse_weight}

    gates, energy = evaluate_individual(
        individual,
        args.filename,
        args.training_timesteps,
        args.num_envs,
        args.gate_cfg,
        args.device,
        num=num,
        max_steps=args.max_steps,
        experiment_type=args.experiment_type,
        penalty_weights=penalty_weights,
    )

    # Subprocess contract: print "gates energy_j" on stdout
    print(f"{gates} {energy}")
