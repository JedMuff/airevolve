"""
power_stats_callback.py — PowerStatsCallback

TensorBoard + CSV telemetry callback for the three power-aware PPO experiments.

Logged every `window` env-steps
---------------------------------
TensorBoard scalars (prefix  battery/):
  mean_final_soc_pct                         — mean SoC at episode end [%]
  mean_final_voltage                         — mean terminal voltage [V]
  mean_remaining_cap_mah                     — mean remaining capacity [mAh]
  mean_ep_power_w                            — mean instantaneous power [W]
  mean_ep_energy_j                           — mean total energy per episode [J]
  battery_died_rate                          — fraction of episodes where battery died
  power_penalty_applied                      — boolean (1 if exp 2 or 3 with weight > 0)

TensorBoard scalars (prefix  train_ext/):
  gate_passes_per_window                     — gate passes in last window
  mean_ep_reward                             — mean episode reward in window
  mean_ep_length                             — mean episode length in window

CSV file (one row per window):
  timestep, gate_passes, mean_ep_reward, mean_ep_length, n_episodes,
  mean_final_soc_pct, mean_final_voltage, mean_ep_energy_j,
  battery_died_rate

Trajectory (env 0 only):
  Saved as  <save_dir>/traj/traj_<timestep>.npy  every `traj_save_interval` steps.
  Shape: (T, 3) float32 NED positions — load with np.load() for post-hoc 3D plotting.

Usage
-----
    cb = PowerStatsCallback(
        save_dir   = "/local/data/mdu219/drone-experiment-1/w0.001",
        experiment_type = 1,
        penalty_weights = {"dense_weight": 0.001},
        window     = 100_000,
    )
    model.learn(..., callback=cb)
"""
from __future__ import annotations

import csv
import os
from collections import defaultdict

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from airevolve.simulator.simulation.battery_model import LiPoBatteryModel

class PowerStatsCallback(BaseCallback):
    """
    Logs battery telemetry, mode distributions, and standard PPO metrics to
    TensorBoard and CSV at a fixed window cadence.

    Parameters
    ----------
    save_dir : str
        Root directory for this run's CSV and trajectory files.
    experiment_type : int
        1, 2, or 3 — used for the 'power_penalty_applied' flag.
    penalty_weights : dict | None
        Forwarded for the penalty flag; not used for computation here.
    window : int
        Number of env-steps between log writes (default 100_000).
    traj_save_interval : int
        Save a trajectory snapshot every this many env-steps (default 200_000).
        Set to 0 to disable trajectory saving.
    verbose : int
        SB3 verbosity (0 = silent).
    """

    def __init__(
        self,
        save_dir: str,
        experiment_type: int = 1,
        penalty_weights: dict | None = None,
        window: int = 100_000,
        traj_save_interval: int = 200_000,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)

        self.save_dir           = save_dir
        self.experiment_type    = experiment_type
        self.window             = window
        self.traj_save_interval = traj_save_interval

        pw = penalty_weights or {}
        # Penalty is "applied" if the relevant weight is non-zero
        self._penalty_applied = int(
            (experiment_type in (1, 3) and pw.get("dense_weight",  0.0) > 0)
            or
            (experiment_type in (2, 3) and pw.get("sparse_weight", 0.0) > 0)
            or
            (experiment_type == 3       and pw.get("sparse_bonus",  0.0) > 0)
        )

        # ── Windowed accumulators ──────────────────────────────────────────────
        self._gate_passes   = 0
        self._ep_rewards:   list[float] = []
        self._ep_lengths:   list[int]   = []
        self._ep_final_soc: list[float] = []
        self._ep_final_v:   list[float] = []
        self._ep_energy_j:  list[float] = []
        self._ep_died:      list[bool]  = []

        # Instantaneous power accumulator (for mean power estimation)
        self._power_sum   = 0.0
        self._power_count = 0

        # ── Trajectory buffer (env 0 only) ────────────────────────────────────
        self._traj_buf: list[np.ndarray] = []   # list of (3,) NED position arrays

        # ── Interval tracking ─────────────────────────────────────────────────
        self._last_log  = 0
        self._last_traj = 0

        # ── CSV setup ─────────────────────────────────────────────────────────
        os.makedirs(save_dir, exist_ok=True)
        self._csv_path = os.path.join(save_dir, "power_metrics.csv")
        with open(self._csv_path, "w", newline="") as f:
            csv.writer(f).writerow([
                "timestep", "gate_passes",
                "mean_ep_reward", "mean_ep_length", "n_episodes",
                "mean_final_soc_pct", "mean_final_voltage",
                "mean_ep_energy_j", "battery_died_rate",
            ])

        self._traj_dir = os.path.join(save_dir, "traj")
        if traj_save_interval > 0:
            os.makedirs(self._traj_dir, exist_ok=True)

    # ──────────────────────────────────────────────────────────────────────────

    def _get_power_env(self):
        """
        Unwrap VecMonitor → PowerAwareDroneEnv.
        Handles both wrapped (VecMonitor) and unwrapped (bare env) cases.
        """
        env = self.training_env
        return getattr(env, "venv", env)   # VecMonitor stores inner env in .venv

    # ──────────────────────────────────────────────────────────────────────────

    def _on_step(self) -> bool:
        """Called once per PPO step (= one call to env.step_wait())."""
        power_env = self._get_power_env()

        # ── Instantaneous power (env 0 only) for mean power estimate ─────────
        p = power_env.bat_power[0]
        self._power_sum   += p
        self._power_count += 1

        # ── Trajectory buffer (env 0 position) ───────────────────────────────
        if self.traj_save_interval > 0:
            self._traj_buf.append(power_env.world_states[0, :3].copy())

        # ── Episode-end telemetry ─────────────────────────────────────────────
        for i, info in enumerate(self.locals.get("infos", [])):
            # Gate passes (from parent gate logic)
            if info.get("gate_passed", False):
                self._gate_passes += 1

            # Episode-end data (added by VecMonitor)
            ep = info.get("episode")
            if ep is not None:
                self._ep_rewards.append(float(ep["r"]))
                self._ep_lengths.append(int(ep["l"]))

            # Battery telemetry (added by PowerAwareDroneEnv.step_wait())
            if "battery_final_soc" in info:
                self._ep_final_soc.append(float(info["battery_final_soc"]))
                self._ep_final_v.append(float(info["battery_final_voltage"]))
                self._ep_energy_j.append(float(info["battery_energy_j"]))
                self._ep_died.append(bool(info["battery_died"]))

        # ── Periodic logging ──────────────────────────────────────────────────
        if self.num_timesteps - self._last_log >= self.window:
            self._write_window()

        if (self.traj_save_interval > 0 and
                self.num_timesteps - self._last_traj >= self.traj_save_interval):
            self._save_trajectory()

        return True

    # ──────────────────────────────────────────────────────────────────────────

    def _write_window(self) -> None:
        """Flush window accumulators to TensorBoard + CSV."""
        t = self.num_timesteps

        # ── Episode-level battery metrics ─────────────────────────────────────
        mean_soc   = float(np.mean(self._ep_final_soc)) * 100.0 if self._ep_final_soc else float("nan")
        mean_v     = float(np.mean(self._ep_final_v))             if self._ep_final_v   else float("nan")
        mean_cap   = mean_soc / 100.0 * LiPoBatteryModel.CAPACITY_MAH  # mAh
        mean_ej    = float(np.mean(self._ep_energy_j))            if self._ep_energy_j  else float("nan")
        died_rate  = float(np.mean(self._ep_died))                if self._ep_died      else float("nan")

        self.logger.record("battery/mean_final_soc_pct",    mean_soc)
        self.logger.record("battery/mean_final_voltage",    mean_v)
        self.logger.record("battery/mean_remaining_cap_mah", mean_cap)
        self.logger.record("battery/mean_ep_energy_j",      mean_ej)
        self.logger.record("battery/battery_died_rate",     died_rate)
        self.logger.record("battery/power_penalty_applied", self._penalty_applied)

        # Mean instantaneous power (proxy — env 0 only, but representative)
        mean_p = self._power_sum / max(self._power_count, 1)
        self.logger.record("battery/mean_ep_power_w", mean_p)

        # ── Standard training metrics ─────────────────────────────────────────
        mean_r = float(np.mean(self._ep_rewards)) if self._ep_rewards else float("nan")
        mean_l = float(np.mean(self._ep_lengths)) if self._ep_lengths else float("nan")
        n_eps  = len(self._ep_rewards)

        self.logger.record("train_ext/gate_passes_per_window", self._gate_passes)
        self.logger.record("train_ext/mean_ep_reward",         mean_r)
        self.logger.record("train_ext/mean_ep_length",         mean_l)
        self.logger.dump(t)

        # ── CSV ───────────────────────────────────────────────────────────────
        with open(self._csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                t, self._gate_passes,
                mean_r, mean_l, n_eps,
                mean_soc, mean_v,
                mean_ej, died_rate,
            ])

        # ── Reset window accumulators ─────────────────────────────────────────
        self._gate_passes  = 0
        self._ep_rewards.clear()
        self._ep_lengths.clear()
        self._ep_final_soc.clear()
        self._ep_final_v.clear()
        self._ep_energy_j.clear()
        self._ep_died.clear()
        self._power_sum    = 0.0
        self._power_count  = 0
        self._last_log     = t

    def _save_trajectory(self) -> None:
        """Save env-0 position trajectory buffer as a .npy file."""
        if not self._traj_buf:
            return
        traj = np.stack(self._traj_buf, axis=0).astype(np.float32)  # (T, 3)
        fname = os.path.join(self._traj_dir, f"traj_{self.num_timesteps:09d}.npy")
        np.save(fname, traj)
        self._traj_buf.clear()
        self._last_traj = self.num_timesteps

    def _on_training_end(self) -> None:
        """Flush any remaining data when training finishes."""
        if self._ep_rewards or self._gate_passes:
            self._write_window()
        if self._traj_buf:
            self._save_trajectory()
