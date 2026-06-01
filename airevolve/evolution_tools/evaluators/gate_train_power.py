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
os.environ["OMP_NUM_THREADS"] = "1"
import sys
import time
import argparse
import warnings
from functools import partial

import numpy as np
import pandas as pd
import torch
torch.set_num_threads(1)
import gymnasium as gym
import matplotlib.pyplot as plt

import json

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

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

# ── Single-environment gym wrapper (one per SubprocVecEnv worker) ────────────

class _SingleDroneEnv(gym.Env):
    """gym.Env wrapping DroneGateEnv(num_envs=1) for use inside SubprocVecEnv.

    SubprocVecEnv requires callables that return gym.Env instances.
    DroneGateEnv is already a VecEnv (SB3 abstract), so this thin wrapper
    squeezes the batch dimension out of observations / rewards / dones and
    exposes the standard 4-tuple step() interface that SB3 expects.

    Must be defined at module level (not as a closure) so that the 'spawn'
    start method can pickle it across the process boundary without CUDA
    fork-safety issues.
    """

    def __init__(
        self,
        individual: np.ndarray,
        gate_pos: np.ndarray,
        gate_yaw: np.ndarray,
        start_pos: np.ndarray,
        x_bounds: np.ndarray,
        y_bounds: np.ndarray,
        z_bounds: np.ndarray,
        device: str,
        max_steps: int,
        sparse_weight: float,
        use_power_env: bool,
        overdraw_penalty_weight: float = 0.0,
        random_start: bool = True,
    ) -> None:
        super().__init__()

        _kwargs = dict(
            num_envs=1,
            individual=individual,
            gates_pos=gate_pos,
            gate_yaw=gate_yaw,
            start_pos=start_pos,
            x_bounds=x_bounds,
            y_bounds=y_bounds,
            z_bounds=z_bounds,
            gates_ahead=2,
            initialize_at_random_gates=random_start,
            num_state_history=0,
            num_action_history=0,
            history_step_size=1,
            render_mode=None,
            device=device,
            max_steps=max_steps,
        )
        if use_power_env:
            self._env = PowerAwareDroneEnv(
                experiment_type=2,
                penalty_weights={"sparse_weight": sparse_weight},
                randomize_soc=True,
                strict_voltage_kill=False,
                overdraw_penalty_weight=overdraw_penalty_weight,
                **_kwargs,
            )
        else:
            self._env = DroneGateEnv(**_kwargs)

        # Expose spaces directly — VecEnv spaces are already 1-D (no batch dim)
        self.observation_space = self._env.observation_space
        self.action_space      = self._env.action_space

    def reset(self, **kwargs):
        obs = self._env.reset()          # shape (1, obs_len) from VecEnv
        return obs[0], {}                # shape (obs_len,) + empty info dict

    def step(self, action):
        # VecEnv expects (num_envs, action_dim); we have (action_dim,)
        obs, rewards, dones, infos = self._env.step(action[np.newaxis])
        return obs[0], float(rewards[0]), bool(dones[0]), False, infos[0]

    def seed(self, seed=None):
        return self._env.seed(seed)

    def close(self):
        self._env.close()


def _env_init(
    individual,
    gate_pos,
    gate_yaw,
    start_pos,
    x_bounds,
    y_bounds,
    z_bounds,
    device,
    max_steps,
    sparse_weight,
    use_power_env,
    overdraw_penalty_weight=0.0,
    random_start=True,
):
    """Top-level (non-closure) factory — picklable for 'spawn' start method."""
    return _SingleDroneEnv(
        individual=individual,
        gate_pos=gate_pos,
        gate_yaw=gate_yaw,
        start_pos=start_pos,
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        z_bounds=z_bounds,
        device=device,
        max_steps=max_steps,
        sparse_weight=sparse_weight,
        use_power_env=use_power_env,
        overdraw_penalty_weight=overdraw_penalty_weight,
        random_start=random_start,
    )


# ── Energy sentinel ───────────────────────────────────────────────────────────

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
    sparse_weight: float = 0.002,
    use_power_env: bool = True,
    overdraw_penalty_weight: float = 0.0,
    verbose: int = 1,
    progress_bar: bool = True,
    random_start: bool = True,
    load_policy=None,
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

    # ── Training environment — SubprocVecEnv for true CPU parallelism ─────────
    # Each of the num_envs worker processes runs a single _SingleDroneEnv.
    # 'spawn' is required: it creates a fresh Python interpreter per worker,
    # initialising CUDA independently and avoiding fork-safety deadlocks.
    _factory = partial(
        _env_init,
        individual=individual,
        gate_pos=gate_pos,
        gate_yaw=gate_yaw,
        start_pos=start_pos,
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        z_bounds=z_bounds,
        device=device,
        max_steps=max_steps,
        sparse_weight=sparse_weight,
        use_power_env=use_power_env,
        overdraw_penalty_weight=overdraw_penalty_weight,
        random_start=random_start,
    )
    env = SubprocVecEnv(
        [_factory] * num_envs,
        start_method='spawn',
    )

    monitor_file = save_dir + (f"m{num}" if num is not None else "")
    env = VecMonitor(env, filename=monitor_file)

    if load_policy is not None:
        print(f"[diag] Loading existing policy from {load_policy}", flush=True)
        model = PPO.load(load_policy, env=env, device=device)
    else:
        policy_kwargs = dict(
            activation_fn=torch.nn.ReLU,
            net_arch=dict(pi=[64, 64], vf=[64, 64]),
            log_std_init=0.0,
        )
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            tensorboard_log=save_dir,
            n_steps=1000,
            batch_size=5000,
            n_epochs=10,
            gamma=0.999,
            device=device,
        )
    # ── Train then unconditionally release worker processes ──────────────────
    # SubprocVecEnv creates one OS pipe pair per worker.  Without an explicit
    # env.close(), those pipes accumulate across evaluations in the same OS
    # process and eventually exhaust the per-process file-descriptor limit
    # (default 1024 on Linux), causing "OSError: [Errno 24] Too many open files".
    # The finally block guarantees closure even when model.learn() raises.
    eval_env = DummyVecEnv([partial(
        _env_init,
        individual=individual,
        gate_pos=gate_pos,
        gate_yaw=gate_yaw,
        start_pos=start_pos,
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        z_bounds=z_bounds,
        device=device,
        max_steps=max_steps,
        sparse_weight=0.0,
        use_power_env=use_power_env,
        overdraw_penalty_weight=0.0,
        random_start=False,
    )])
    # Wrap in VecMonitor so EvalCallback can read episode stats and SB3 stops
    # warning about an unmonitored eval environment.
    eval_env = VecMonitor(eval_env)
    best_model_path = os.path.join(save_dir, "best_model")
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=os.path.join(save_dir, "eval_logs"),
        eval_freq=max(1, int(total_timesteps) // 20),
        n_eval_episodes=3,
        deterministic=True,
        render=False,
        verbose=0,
    )

    try:
        model.learn(
            total_timesteps=int(total_timesteps),
            reset_num_timesteps=False,
            log_interval=100,
            callback=[FullStatsCallback(), eval_callback],
            progress_bar=progress_bar,
        )
        final_model_path = os.path.join(save_dir, "final_model")
        model.save(final_model_path)

        policy_path = save_dir + (f"policy{num}" if num is not None else "policy")
        model.save(policy_path)

        # ── Training curve plot ───────────────────────────────────────────────
        try:
            try:
                data = pd.read_csv(monitor_file + ".monitor.csv", skiprows=1)
            except Exception:
                data = pd.read_csv(monitor_file + "monitor.csv", skiprows=1)
            # Smooth with a rolling average sized to 1% of the episode count so
            # the trend stays legible regardless of how many millions of
            # timesteps were run.
            window_size = max(1, len(data) // 100)
            smoothed_rewards = data["r"].rolling(window=window_size, min_periods=1).mean()
            smoothed_std = data["r"].rolling(window=window_size, min_periods=1).std()
            plt.figure(figsize=(10, 6))
            plt.plot(data["t"], smoothed_rewards, color="red", label=f"Episode Reward (rolling avg, window={window_size})")
            plt.fill_between(data["t"], smoothed_rewards - smoothed_std, smoothed_rewards + smoothed_std, color="red", alpha=0.2)
            plt.xlabel("Timesteps")
            plt.ylabel("Reward")
            plt.title("Reward per Episode")
            plt.legend()
            fig_path = save_dir + (f"figure{num}.png" if num is not None else "figure.png")
            plt.savefig(fig_path)
            plt.close()
        except Exception:
            pass
    finally:
        env.close()
        eval_env.close()

    # ── Deterministic evaluation: fixed max_steps window ─────────────────────
    # The test env MUST use the same class as the training env so that the
    # observation space (and therefore the number of features seen by the
    # policy) is identical.
    #
    # experiment_type 0  →  DroneGateEnv          (22-dim obs)
    # experiment_type 1/2 → PowerAwareDroneEnv    (22 + 3 battery = 25-dim obs)
    #
    # For power experiments we set penalty_weights={} and randomize_soc=False:
    # no reward shaping during eval, battery always starts at full charge for a
    # fair and deterministic 12-second measurement window.
    #
    # Energy is tracked by the env's built-in vectorized battery model.  Since
    # DroneGateEnv resets bat_energy_j on episode boundary, we accumulate
    # total energy across resets by snapshotting before each step.
    #
    # IMPORTANT: we use the observation *returned* by reset() / step() rather
    # than test_env.states.  For PowerAwareDroneEnv, self.states holds only the
    # base 22-dim obs; the 3 battery dims are concatenated onto the *return
    # value* of step_wait() / reset_().  Using test_env.states would feed a
    # 22-dim vector to a policy trained on 25-dim vectors.
    _test_kwargs = dict(
        num_envs=1,
        individual=individual,
        gates_pos=gate_pos,
        gate_yaw=gate_yaw,
        start_pos=start_pos,
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        z_bounds=z_bounds,
        initialize_at_random_gates=False,
        gates_ahead=2,
        num_state_history=0,
        num_action_history=0,
        history_step_size=1,
        render_mode=None,
        device=device,
        max_steps=max_steps,
    )
    if use_power_env:
        test_env = PowerAwareDroneEnv(
            experiment_type=2,
            penalty_weights={},       # no reward shaping during evaluation
            randomize_soc=False,      # full charge → deterministic 12-second window
            strict_voltage_kill=True, # NSGA-II eval: strict physical voltage cutoff
            **_test_kwargs,
        )
    else:
        test_env = DroneGateEnv(**_test_kwargs)
        # Enable strict voltage kill for NSGA-II eval: voltage sag below 12.8 V
        # triggers battery depletion (current → 0, dynamic_w_max collapses,
        # drone loses thrust and eventually goes OOB).
        test_env.strict_voltage_kill = True

    # Initialise return values before the try block so that a mid-eval crash
    # still yields a well-typed, dominated tuple rather than an UnboundLocalError.
    num_gates_passed = 0
    total_energy_j   = _FAIL_ENERGY

    try:
        obs = test_env.reset()

        # Track the maximum gates passed across the whole eval window: if the
        # drone crashes mid-flight the env resets and num_gates_passed drops back
        # to 0, so reading infos only after the loop would lose the count. We also
        # snapshot distance_to_gate / target_gate_idx at the moment the max was
        # reached (and keep the closest approach toward the next gate) so the
        # continuous fitness fraction reflects real progress.
        #
        # Energy is read from the env's built-in battery model (bat_energy_j).
        # Since reset_() zeros bat_energy_j, we accumulate across resets by
        # snapshotting the value before each step.
        max_gates_passed = 0
        best_distance_to_gate = float(np.linalg.norm(test_env.gate_pos[0] - test_env.start_pos))
        best_target_idx = 0
        cumulative_energy_j = 0.0
        prev_energy_snapshot = 0.0
        for _ in range(max_steps):
            actions, _ = model.predict(obs, deterministic=True)
            obs, _rewards, _dones, infos = test_env.step(actions)

            # Accumulate energy across resets: when the env resets a done episode,
            # bat_energy_j drops back to ~0.  Detect this and bank the pre-reset
            # total.  At most one dt's worth of energy is lost per reset (the
            # battery step that triggered the reset is already zeroed out).
            current_energy = float(test_env.bat_energy_j[0])
            if current_energy < prev_energy_snapshot:
                cumulative_energy_j += prev_energy_snapshot
            prev_energy_snapshot = current_energy

            gates_passed = int(infos[0]["num_gates_passed"][0])
            current_d2g = float(infos[0]["distance_to_gate"])
            if gates_passed > max_gates_passed:
                max_gates_passed = gates_passed
                best_distance_to_gate = current_d2g
                best_target_idx = int(infos[0]["target_gate_idx"])
            elif gates_passed == max_gates_passed and current_d2g < best_distance_to_gate:
                best_distance_to_gate = current_d2g
                best_target_idx = int(infos[0]["target_gate_idx"])

        # Add the final segment's energy (from the last reset to end of loop)
        cumulative_energy_j += prev_energy_snapshot

        num_gates_passed = max_gates_passed
        total_energy_j   = cumulative_energy_j

        distance_to_gate = best_distance_to_gate
        target_idx = best_target_idx

        if num_gates_passed == 0:
            gate_dist = float(np.linalg.norm(test_env.gate_pos[0] - test_env.start_pos))
        else:
            prev_idx = (target_idx - 1) % test_env.num_gates
            gate_dist = float(np.linalg.norm(test_env.gate_pos[target_idx] - test_env.gate_pos[prev_idx]))
            
        fraction = max(0.0, 1.0 - (distance_to_gate / gate_dist))
        continuous_fitness = float(num_gates_passed) + fraction

    finally:
        test_env.close()

    return continuous_fitness, total_energy_j


def evaluate_individual(
    individual,
    ind_save_dir: str,
    training_ts,
    num_envs,
    gate_cfg: str,
    device: str = "cuda:0",
    num=None,
    max_steps: int = 1200,
    sparse_weight: float = 0.0,
    use_power_env: bool = False,
    overdraw_penalty_weight: float = 0.0,
    verbose: int = 1,
    progress_bar: bool = True,
    random_start: bool = True,
    load_policy=None,
) -> tuple:
    """Hover-check, train, and evaluate one morphology.

    Returns (0, FAIL_ENERGY) immediately for non-hoverable morphologies so that
    the caller receives a consistently-typed tuple in all cases.

    Returns
    -------
    (num_gates_passed, total_energy_j) : (int, float)
    """
    start_time = time.time()
    os.makedirs(ind_save_dir, exist_ok=True)

    try:
        np.save(os.path.join(ind_save_dir, "genome.npy"), individual)
        morph_cfg = {
            "num_motors": int((~np.isnan(individual).any(axis=1)).sum()),
            "arms": [
                {k: float(v) for k, v in zip(
                    ["magnitude", "arm_yaw", "arm_pitch", "mot_pitch", "mot_yaw", "direction"],
                    row
                )}
                for row in individual[~np.isnan(individual).any(axis=1)]
            ],
        }
        with open(os.path.join(ind_save_dir, "morphology_config.json"), "w") as fh:
            json.dump(morph_cfg, fh, indent=2)
    except Exception as e:
        print(f"[warn] Could not save genome artifacts: {e}")

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
        plt.savefig(os.path.join(ind_save_dir, "morphology_pre.png"))
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
        sparse_weight=sparse_weight,
        use_power_env=use_power_env,
        overdraw_penalty_weight=overdraw_penalty_weight,
        verbose=verbose,
        progress_bar=progress_bar,
        random_start=random_start,
        load_policy=load_policy,
    )

    # Post-training morphology plot
    try:
        fig = plt.figure(figsize=plt.figaspect(0.5))
        ax  = fig.add_subplot(111, projection="3d")
        visualizer.plot_3d(
            individual, ax=ax,
            title=f"Post-training (Gen {num}) | gates={num_gates_passed} | energy={total_energy_j:.0f}J",
            fitness=num_gates_passed,
            generation=num,
        )
        plt.savefig(os.path.join(ind_save_dir, "morphology_post.png"))
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
    parser.add_argument("--sparse_weight",    default=0.002, type=float)
    parser.add_argument("--overdraw_weight",  default=0.0,   type=float)
    parser.add_argument("--no_power_env",     action="store_true")
    parser.add_argument("--no_random_start",  action="store_true")
    parser.add_argument("--load_policy",      default=None,  type=str)
    args = parser.parse_args()

    try:
        individual = np.load(args.filename + "/individual.npy", allow_pickle=True)
    except FileNotFoundError:
        individual = np.load(args.filename + "/genome.npy", allow_pickle=True)
    individual = individual.astype(np.float32)
    num        = int(args.num) if args.num is not None else None

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    ind_name = os.path.basename(os.path.normpath(args.filename))
    out_dir = os.path.join(project_root, "results_training", ind_name)
    os.makedirs(out_dir, exist_ok=True)

    gates, energy = evaluate_individual(
        individual,
        out_dir,
        args.training_timesteps,
        args.num_envs,
        args.gate_cfg,
        args.device,
        num=num,
        max_steps=args.max_steps,
        sparse_weight=args.sparse_weight,
        use_power_env=not args.no_power_env,
        overdraw_penalty_weight=args.overdraw_weight,
        random_start=not args.no_random_start,
        load_policy=args.load_policy,
    )

    # Subprocess contract: print "gates energy_j" on stdout
    print(f"{gates} {energy}")
