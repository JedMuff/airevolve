"""
Apples-to-apples diagnostic: train a canonical 4-motor quad in *airevolve's*
DroneGateEnv using *optimal_quad_control_RL*'s PPO hyperparameters and gate
configuration. If this learns to fly the figure-8, airevolve's pipeline is
fundamentally workable and the gap with the reference is in the airevolve
defaults (hyperparameters / reward / motor model / morphology). If it doesn't
learn, the airevolve env or simulator itself is the root cause.

Usage:
    python optimal_parity_quad.py [--prop-size 2|4|5|matched] [--motor-tau 0.04]
                                  [--total-steps 5e6] [--final-gate-only]
Outputs a tensorboard log + saved checkpoint under ./parity_logs/.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.callbacks import BaseCallback


from airevolve.evolution_tools.evaluators.drone_gate_env import DroneGateEnv
from airevolve.simulator.simulation.propeller_data import (
    create_standard_propeller_config,
)


class FinalGateOnlyRewardEnv(DroneGateEnv):
    """Subtract the per-gate +10 reward and only award it when the *final*
    gate of the lap is passed (i.e. when target_gates wraps back to 0)."""

    def step_wait(self):
        prev_target = self.target_gates.copy()
        states, rewards, dones, infos = super().step_wait()
        for i, info in enumerate(infos):
            if info.get("gate_passed", False):
                rewards[i] -= 10.0
                # final-gate detection: the *last* gate is the one whose passing
                # would advance target_gates from num_gates-1 to 0 (mod num_gates).
                if prev_target[i] == self.num_gates - 1:
                    rewards[i] += 10.0
        return states, rewards, dones, infos


# ---------------------------------------------------------------------------
# Reference (optimal_quad_control_RL) gate config: figure-8 at z=-1.5 (NED),
# with 5 m horizontal bounds and 7 m of vertical headroom.
# ---------------------------------------------------------------------------
R = 1.5
GATE_POS = np.array(
    [
        [R, -R, -1.5],
        [0, 0, -1.5],
        [-R, R, -1.5],
        [0, 2 * R, -1.5],
        [R, R, -1.5],
        [0, 0, -1.5],
        [-R, -R, -1.5],
        [0, -2 * R, -1.5],
    ],
    dtype=np.float32,
)
GATE_YAW = (np.array([1, 2, 1, 0, -1, -2, -1, 0]) * np.pi / 2).astype(np.float32)
START_POS = (GATE_POS[0] + np.array([0, -1.0, 0])).astype(np.float32)
X_BOUNDS = np.array([-5.0, 5.0], dtype=np.float32)
Y_BOUNDS = np.array([-5.0, 5.0], dtype=np.float32)
Z_BOUNDS = np.array([-7.0, 0.0], dtype=np.float32)


NUM_ENVS = 100
LOG_DIR = os.path.join(os.path.dirname(__file__), "parity_logs")


class GatePassRateCallback(BaseCallback):
    def __init__(self, log_every: int = 100_000, verbose: int = 0):
        super().__init__(verbose)
        self.log_every = log_every
        self.last_log = 0
        self.recent_gates = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if info.get("gate_passed", False):
                self.recent_gates += 1
        if self.num_timesteps - self.last_log >= self.log_every:
            print(
                f"[{self.num_timesteps:>10}] gate_passes_in_last_{self.log_every}_steps={self.recent_gates}",
                flush=True,
            )
            self.logger.record("custom/gate_passes_per_window", self.recent_gates)
            self.recent_gates = 0
            self.last_log = self.num_timesteps
        return True


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prop-size", default="2",
                   help="propeller size: 2,4,5,6,7,8 or 'matched' (1.2kg drone)")
    p.add_argument("--motor-tau", type=float, default=None,
                   help="override env motor_tau (default uses env's 0.01)")
    p.add_argument("--total-steps", type=float, default=5e6)
    p.add_argument("--final-gate-only", action="store_true",
                   help="only award +10 on final gate of lap")
    p.add_argument("--airevolve-hyper", action="store_true",
                   help="use airevolve gate_train.py PPO hyperparameters instead of reference")
    p.add_argument("--airevolve-figure8", action="store_true",
                   help="use airevolve's native figure8 (z=0 gates, z_bounds=[-1,1])")
    p.add_argument("--widen-z", action="store_true",
                   help="(with --airevolve-figure8) widen z_bounds to [-3, 3]")
    p.add_argument("--airevolve-track", default=None,
                   help="airevolve track name: backandforth, figure8, circle, slalom")
    p.add_argument("--reference-dynamics", action="store_true",
                   help="swap airevolve's morphology-derived dynamics for "
                        "optimal_quad_control_RL's sysid'd dynamics. "
                        "Used by the session 4 parity experiment in "
                        "RL_TRAINING_FIXES.md to isolate the dynamics gap.")
    p.add_argument("--reference-params", default="5inch",
                   choices=["5inch", "3inch", "2inch_derived"],
                   help="sysid parameter set for --reference-dynamics. "
                        "5inch and 3inch are real sysid sets from "
                        "optimal_quad_control_RL; 2inch_derived maps "
                        "airevolve's prop2 hardware specs into the "
                        "reference's 22-parameter form (some parameters "
                        "borrowed from 3inch — see derive_params_2inch_from_airevolve "
                        "in reference_drone_sim.py).")
    p.add_argument("--tag", default="", help="extra tag in run name")
    p.add_argument("--seed", type=int, default=None, help="seed for env + PPO")
    p.add_argument("--out-dir", default=None,
                   help="absolute output directory (defaults to ./parity_logs/)")
    return p.parse_args()


def main():
    args = parse_args()
    total_steps = int(args.total_steps)
    prop_size = args.prop_size
    if prop_size != "matched":
        prop_size = int(prop_size)
    tag = (
        f"prop{args.prop_size}"
        + (f"_tau{args.motor_tau:.3f}" if args.motor_tau is not None else "")
        + ("_finalonly" if args.final_gate_only else "")
        + (f"_refdyn-{args.reference_params}" if args.reference_dynamics else "")
        + (f"_{args.tag}" if args.tag else "")
    )
    if args.seed is not None:
        tag = f"{tag}_seed{args.seed}"
    name = f"parity_{tag}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    log_dir = args.out_dir or LOG_DIR
    os.makedirs(log_dir, exist_ok=True)

    print(
        f"=== optimal_parity_quad: PPO on airevolve "
        f"{'ReferenceDynamicsGateEnv' if args.reference_dynamics else 'DroneGateEnv'} "
        f"(prop={prop_size}, motor_tau={'env-default' if args.motor_tau is None else args.motor_tau}, "
        f"final_gate_only={args.final_gate_only}, "
        f"reference_dynamics={args.reference_dynamics}), {total_steps:_} steps ===",
        flush=True,
    )

    quad_props = create_standard_propeller_config("quad", arm_length=0.11, prop_size=prop_size)

    if args.airevolve_figure8 or args.airevolve_track:
        track_name = args.airevolve_track or "figure8"
        import airevolve.evolution_tools.evaluators.gate_train as gt
        track = getattr(gt, track_name)
        gate_pos = track.gate_pos
        gate_yaw = track.gate_yaw
        start_pos = track.starting_pos.astype(np.float32)
        x_bounds, y_bounds, z_bounds = track.x_bounds, track.y_bounds, track.z_bounds
        if args.widen_z:
            z_bounds = np.array([-3.0, 3.0], dtype=np.float32)
        print(f"using airevolve-native {track_name}: z_bounds={z_bounds.tolist()}", flush=True)
    else:
        gate_pos, gate_yaw, start_pos = GATE_POS, GATE_YAW, START_POS
        x_bounds, y_bounds, z_bounds = X_BOUNDS, Y_BOUNDS, Z_BOUNDS

    if args.reference_dynamics:
        # Lazy import — keeps optimal_parity_quad.py importable even if the
        # reference-dynamics module has a sympy dep issue.
        from reference_dynamics_env import ReferenceDynamicsGateEnv
        env_cls = ReferenceDynamicsGateEnv
        if args.final_gate_only:
            raise SystemExit(
                "--reference-dynamics and --final-gate-only are mutually "
                "exclusive: ReferenceDynamicsGateEnv inherits its reward "
                "shape from DroneGateEnv (final-only by default after "
                "session 3)."
            )
        extra_kwargs = {"params_variant": args.reference_params}
    else:
        env_cls = FinalGateOnlyRewardEnv if args.final_gate_only else DroneGateEnv
        extra_kwargs = {}
    env = env_cls(
        num_envs=NUM_ENVS,
        propellers=quad_props,
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
        device="cuda:0",
        dt=0.01,
        seed=args.seed,
        **extra_kwargs,
    )
    if args.motor_tau is not None:
        env.motor_tau = args.motor_tau
    obs = env.reset()
    print(
        f"obs shape: {obs.shape}, num_motors: {env.num_motors}, mass: {env.drone_sim.mass:.3f} kg, "
        f"motor_tau: {env.motor_tau}",
        flush=True,
    )

    monitor_path = os.path.join(log_dir, "monitor" if args.out_dir else name)
    env = VecMonitor(env, filename=monitor_path)

    if args.airevolve_hyper:
        # airevolve gate_train.py:244-265 (Bayesian-tuned, smaller batches, more epochs)
        policy_kwargs = dict(
            activation_fn=torch.nn.ReLU,
            net_arch=dict(pi=[64, 64, 64], vf=[64, 64, 64]),
            log_std_init=0.041,
        )
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose=0,
            tensorboard_log=log_dir,
            learning_rate=1.0e-4,
            n_steps=256,
            batch_size=128,
            n_epochs=22,
            gamma=0.9965,
            gae_lambda=0.894,
            clip_range=0.156,
            ent_coef=0.0044,
            vf_coef=0.55,
            max_grad_norm=3.19,
            device="cuda:0",
        )
    else:
        # Reference PPO hyperparameters from optimal_quad_control_RL/train.py:149-161
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
            tensorboard_log=log_dir,
            n_steps=1000,
            batch_size=5000,
            n_epochs=10,
            gamma=0.999,
            device="cuda:0",
        )

    from window_metrics_callback import WindowMetricsCallback
    cb = WindowMetricsCallback(
        window=100_000,
        csv_path=os.path.join(
            log_dir, "window_metrics.csv" if args.out_dir else name + "_window_metrics.csv"
        ),
    )
    t0 = time.time()
    model.learn(total_timesteps=total_steps, callback=cb, tb_log_name=name)
    dt = time.time() - t0
    save_path = os.path.join(log_dir, "final.zip" if args.out_dir else name + "_final.zip")
    model.save(save_path)
    print(f"=== done in {dt:.1f}s ({total_steps/dt:.0f} steps/s). saved {save_path} ===", flush=True)


if __name__ == "__main__":
    main()
