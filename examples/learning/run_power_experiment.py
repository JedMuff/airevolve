"""
run_power_experiment.py — PPO training orchestrator for the three
power-aware reward shaping experiments.

Mirrors examples/learning/run_rl_figure8.py in structure (same PPO
hyperparameters, same VecMonitor + WindowMetricsCallback pattern) while
swapping the base environment for PowerAwareDroneEnv and adding the
PowerStatsCallback for battery telemetry.

Usage
-----
# Experiment 1 — dense penalty, weight sweep
python examples/learning/run_power_experiment.py \\
    --experiment 1 --dense-weight 0.001 --num-envs 64 \\
    --total-steps 1000000 --device cuda:0 \\
    --save-dir /local/data/mdu219/drone-experiment-1/dense_w0.001

# Experiment 2 — sparse penalty
python examples/learning/run_power_experiment.py \\
    --experiment 2 --sparse-weight 0.05 --num-envs 64 \\
    --total-steps 1000000 --device cuda:0 \\
    --save-dir /local/data/mdu219/drone-experiment-2/sparse_w0.05

# Experiment 3 — hybrid
python examples/learning/run_power_experiment.py \\
    --experiment 3 --dense-weight 0.0001 --sparse-bonus 100 --num-envs 64 \\
    --total-steps 1000000 --device cuda:0 \\
    --save-dir /local/data/mdu219/drone-experiment-3/d0.0001_b100

# Baseline (no penalty, any experiment type)
python examples/learning/run_power_experiment.py \\
    --experiment 1 --dense-weight 0.0 --save-dir .../baseline
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor

# ── Repo-root bootstrap ────────────────────────────────────────────────────────
_ROOT = next(
    (p for p in [Path(__file__).resolve(), *Path(__file__).resolve().parents]
     if (p / "setup.py").exists()),
    Path(__file__).resolve().parents[2],
)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from airevolve.evolution_tools.evaluators.drone_gate_env_power import PowerAwareDroneEnv
from airevolve.simulator.simulation.propeller_data import create_standard_propeller_config
from examples.learning.power_stats_callback import PowerStatsCallback


# ═══════════════════════════════════════════════════════════════════════════════
# Hardware / morphology defaults
# ═══════════════════════════════════════════════════════════════════════════════
# Sub-250 g hexacopter: Flywoo GN405, EMAX ECO 1404 3700KV ×6,
# Gemfan 2609 (2.6" ≈ 3" library), Tattu R-Line 750mAh 14.8V 4S 95C
ARM_LENGTH = 0.09   # metres
PROP_SIZE  = 3      # inches (nearest library size to 2.6" Gemfan 2609)

# PPO hyperparameters — match gate_train.py / run_rl_figure8.py exactly
PPO_N_STEPS    = 1000
PPO_BATCH_SIZE = 5000
PPO_N_EPOCHS   = 10
PPO_GAMMA      = 0.999

# Gate track — figure-8 at z=-1.5 NED (1.5 m AGL), ±5 m horizontal
from airevolve.evolution_tools.evaluators.drone_gate_env import (
    gate_pos as FIGURE8_GATE_POS,
    gate_yaw as FIGURE8_GATE_YAW,
    start_pos as FIGURE8_START_POS,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Argument parser
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train PPO on PowerAwareDroneEnv (figure-8 + ECM battery).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Experiment identity
    p.add_argument("--experiment", type=int, choices=[4, 5, 6], default=1,
                   help="4=Dense, 5=Sparse, 6=Hybrid")

    # Penalty weights (provide only the relevant one for your experiment)
    p.add_argument("--dense-weight",  type=float, default=0.0,
                   help="Exp 4 & 6: per-step power penalty weight")
    p.add_argument("--sparse-weight", type=float, default=0.0,
                   help="Exp 5:      end-of-episode energy penalty weight")
    p.add_argument("--sparse-bonus",  type=float, default=0.0,
                   help="Exp 6:      end-of-episode SoC survival bonus")

    # Training scale
    p.add_argument("--total-steps", type=float, default=1e6,
                   help="total PPO environment steps")
    p.add_argument("--num-envs",    type=int,   default=64,
                   help="parallel environments (recommend 64 for L4 GPU)")
    p.add_argument("--max-steps",   type=int,   default=1200,
                   help="max steps per episode (12 s at dt=0.01)")

    # Infrastructure
    p.add_argument("--device",   default="cuda:0",
                   help="torch device (cuda:0 for L4, cpu for local testing)")
    p.add_argument("--save-dir", default=None,
                   help="output directory for logs, model, and CSVs; "
                        "defaults to __data__/power_exp<N>/<timestamp>")
    p.add_argument("--seed",     type=int, default=None,
                   help="random seed for env + numpy")

    # Domain randomisation
    p.add_argument("--no-randomize-soc", action="store_true",
                   help="disable starting-SoC randomisation (fixed full charge)")

    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()

    total_steps    = int(args.total_steps)
    randomize_soc  = not args.no_randomize_soc
    penalty_weights = {
        "dense_weight":  args.dense_weight,
        "sparse_weight": args.sparse_weight,
        "sparse_bonus":  args.sparse_bonus,
    }

    # ── Resolve save directory ─────────────────────────────────────────────────
    if args.save_dir is None:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.save_dir = str(
            Path(_ROOT) / "__data__" /
            f"power_exp{args.experiment}" /
            ts
        )
    save_dir = os.path.abspath(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    # ── Seed ──────────────────────────────────────────────────────────────────
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    # ── Print run configuration ────────────────────────────────────────────────
    exp_names = {4: "Dense", 5: "Sparse", 6: "Hybrid"}
    print("=" * 72)
    print(f"  PowerAwareDroneEnv — Experiment {args.experiment} ({exp_names[args.experiment]})")
    print(f"  penalty_weights   : {penalty_weights}")
    print(f"  total_steps       : {total_steps:_}")
    print(f"  num_envs          : {args.num_envs}")
    print(f"  randomize_soc     : {randomize_soc}")
    print(f"  device            : {args.device}")
    print(f"  save_dir          : {save_dir}")
    print("=" * 72, flush=True)

    # ── 1. Hexacopter morphology (no EA — direct propeller config) ─────────────
    propellers = create_standard_propeller_config(
        "hex", arm_length=ARM_LENGTH, prop_size=PROP_SIZE
    )
    print(f"\nMorphology: {len(propellers)}-motor hexacopter | "
          f"arm={ARM_LENGTH*100:.0f} cm | prop={PROP_SIZE}\"")

    # ── 2. Training environment ────────────────────────────────────────────────
    env = PowerAwareDroneEnv(
        # Power experiment settings
        experiment_type = args.experiment,
        penalty_weights = penalty_weights,
        randomize_soc   = randomize_soc,
        # DroneGateEnv settings
        num_envs        = args.num_envs,
        propellers      = propellers,
        gates_pos       = FIGURE8_GATE_POS,
        gate_yaw        = FIGURE8_GATE_YAW,
        start_pos       = FIGURE8_START_POS,
        x_bounds        = [-5, 5],
        y_bounds        = [-5, 5],
        z_bounds        = [-7, 0],
        gates_ahead     = 1,
        num_state_history   = 0,
        num_action_history  = 0,
        history_step_size   = 1,
        render_mode     = None,
        device          = args.device,
        dt              = 0.01,
        max_steps       = args.max_steps,
        seed            = args.seed,
        initialize_at_random_gates = True,
    )

    print(f"Observation space: {env.observation_space.shape}  "
          f"(base {env.obs_len} + 3 battery dims)")

    # ── 3. VecMonitor wrapper (CSV + episode logging for SB3) ─────────────────
    monitor_path = os.path.join(save_dir, "monitor")
    env_mon = VecMonitor(env, filename=monitor_path)

    # ── 4. PPO model — same hyperparameters as gate_train.py / run_rl_figure8.py ──
    policy_kwargs = dict(
        activation_fn = torch.nn.ReLU,
        net_arch      = dict(pi=[64, 64], vf=[64, 64]),
        log_std_init  = 0.0,
    )
    model = PPO(
        "MlpPolicy",
        env_mon,
        policy_kwargs    = policy_kwargs,
        verbose          = 0,
        tensorboard_log  = save_dir,
        n_steps          = PPO_N_STEPS,
        batch_size       = PPO_BATCH_SIZE,
        n_epochs         = PPO_N_EPOCHS,
        gamma            = PPO_GAMMA,
        device           = args.device,
    )
    print(f"\nPPO: {sum(p.numel() for p in model.policy.parameters()):,} parameters | "
          f"n_steps={PPO_N_STEPS} | batch={PPO_BATCH_SIZE} | "
          f"rollout_size={PPO_N_STEPS * args.num_envs:_}")

    # ── 5. Callbacks ──────────────────────────────────────────────────────────
    power_cb = PowerStatsCallback(
        save_dir        = save_dir,
        experiment_type = args.experiment,
        penalty_weights = penalty_weights,
        window          = 100_000,
        traj_save_interval = 200_000,
    )

    # ── 6. Training ───────────────────────────────────────────────────────────
    print(f"\nTraining for {total_steps:_} steps …  (estimated ~"
          f"{total_steps / (args.num_envs * 100):.0f} PPO updates)\n", flush=True)

    t0 = time.time()
    model.learn(
        total_timesteps     = total_steps,
        callback            = power_cb,
        tb_log_name         = f"exp{args.experiment}_"
                              f"dw{args.dense_weight}_"
                              f"sw{args.sparse_weight}_"
                              f"sb{args.sparse_bonus}",
        reset_num_timesteps = True,
        log_interval        = 100,
    )
    elapsed = time.time() - t0

    # ── 7. Save model ─────────────────────────────────────────────────────────
    model_path = os.path.join(save_dir, "policy.zip")
    model.save(model_path)

    print(f"\n{'='*72}")
    print(f"  Training complete in {elapsed:.1f} s  "
          f"({total_steps / elapsed:.0f} steps/s)")
    print(f"  Model saved  : {model_path}")
    print(f"  TensorBoard  : tensorboard --logdir {save_dir}")
    print(f"  Metrics CSV  : {os.path.join(save_dir, 'power_metrics.csv')}")
    print(f"{'='*72}")


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
