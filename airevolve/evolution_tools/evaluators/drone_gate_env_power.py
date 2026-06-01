"""
drone_gate_env_power.py — PowerAwareDroneEnv

DroneGateEnv extended with power-aware RL reward shaping, battery
observation augmentation, domain randomisation, and configurable
battery-depletion episode termination.

Class hierarchy
---------------
VecEnv (SB3 abstract)
└── DroneGateEnv          (base — physics, gates, base reward,
│                          battery model, current limiter, voltage sag)
    └── PowerAwareDroneEnv  (this file — RL rewards, obs, domain rand.)

Battery physics (current limiter, battery stepping, voltage-sag
dynamic_w_max) live in DroneGateEnv.  This subclass adds only the
RL-specific layer on top:

Observation extension (always the LAST 3 dims of the obs vector)
  obs[−3] = SoC                ∈ [0, 1]
  obs[−2] = V_norm             ∈ [0, 1]   (V − 12.8) / (16.8 − 12.8)
  obs[−1] = P_norm             ∈ [0, 1]   P / P_max  (P_max ≈ 1197.0 W)

Domain randomisation
  • reset_random()           → force SoC ∈ [0.3, 1.0] for this call only
  • randomize_soc=True       → randomise every episode reset automatically

Reward shaping  (experiment_type / penalty_weights)
  1  Dense  : −dense_weight  × P_instant          every step
  2  Sparse : −sparse_weight × E_episode_J         at episode end
  3  Hybrid : −dense_weight  × P_instant           every step
              +sparse_bonus  × SoC_final_%          at episode end
                                                   (only if battery survived)

Overdraw penalty
  • reward -= overdraw_penalty_weight × delta_I   when the base-class
    current limiter clamps actions.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from gymnasium import spaces

# ── Repo-root bootstrap (works from any CWD on macOS or Linux) ────────────────
_ROOT = next(
    (p for p in [Path(__file__).resolve(), *Path(__file__).resolve().parents]
     if (p / "setup.py").exists()),
    Path(__file__).resolve().parents[3],
)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from airevolve.evolution_tools.evaluators.drone_gate_env import DroneGateEnv
from airevolve.simulator.simulation.battery_model import LiPoBatteryModel


class PowerAwareDroneEnv(DroneGateEnv):
    """
    DroneGateEnv + power-aware RL reward shaping + battery observations +
    domain randomisation + battery-depletion episode termination.

    Battery physics (current limiter, battery step, voltage sag) are
    handled by the parent DroneGateEnv.  This subclass:
      • Reconfigures the parent's batteries (strict_voltage_kill setting)
      • Extends the observation space by 3 battery dims
      • Applies Dense / Sparse / Hybrid power rewards
      • Applies an overdraw penalty when the parent's current limiter fires
      • Optionally randomises starting SoC (domain randomisation)
      • Terminates episodes when the battery depletes (configurable)

    Parameters
    ----------
    experiment_type : int
        1 = Dense, 2 = Sparse, 3 = Hybrid  (see module docstring)
    penalty_weights : dict | None
        Exp 1 : {"dense_weight":  float}
        Exp 2 : {"sparse_weight": float}
        Exp 3 : {"dense_weight":  float, "sparse_bonus": float}
        Missing keys default to 0.0 (baseline = no penalty).
    randomize_soc : bool
        If True, every episode reset draws starting SoC ~ Uniform[0.3, 1.0].
        Enables curriculum / domain randomisation without code changes.
    strict_voltage_kill : bool
        Forwarded to LiPoBatteryModel.  True (default) = NSGA-II eval mode;
        False = PPO training mode (voltage sags do NOT kill episodes).
    **kwargs
        Forwarded verbatim to DroneGateEnv.__init__().
    """

    # ── Battery observation normalisation constants ────────────────────────────
    _V_MIN: float = LiPoBatteryModel.VOLTAGE_DEPLETED          # 12.8 V
    _V_MAX: float = LiPoBatteryModel.VOLTAGE_FULL              # 16.8 V
    # Maximum power: full current at full voltage ≈ 71.25 A × 16.8 V = 1197.0 W
    _P_MAX: float = (
        LiPoBatteryModel.BATTERY_MAX_CURRENT
        * LiPoBatteryModel.VOLTAGE_FULL
    )

    # Betaflight-style current limiter weight for the overdraw penalty
    OVERDRAW_PENALTY_WEIGHT: float = 0.01  # reward / Ampere

    def __init__(
        self,
        experiment_type: int = 1,
        penalty_weights: dict | None = None,
        randomize_soc: bool = True,
        strict_voltage_kill: bool = True,
        overdraw_penalty_weight: float | None = None,
        **kwargs,
    ) -> None:
        if experiment_type not in (1, 2, 3):
            raise ValueError(f"experiment_type must be 1, 2, or 3 — got {experiment_type!r}")

        # Parent sets up drone physics, world_states, obs/action spaces,
        # AND batteries (strict_voltage_kill=False by default in the base).
        super().__init__(**kwargs)

        self.experiment_type   = experiment_type
        self._randomize_soc    = randomize_soc
        self._default_rand_soc = randomize_soc  # preserve for reset_random()
        self._strict_voltage_kill = strict_voltage_kill

        self._overdraw_penalty_weight = (
            float(overdraw_penalty_weight)
            if overdraw_penalty_weight is not None
            else self.OVERDRAW_PENALTY_WEIGHT
        )

        pw = penalty_weights or {}
        self._dense_weight  = float(pw.get("dense_weight",  0.0))
        self._sparse_weight = float(pw.get("sparse_weight", 0.0))
        self._sparse_bonus  = float(pw.get("sparse_bonus",  0.0))

        # ── Reconfigure parent batteries with subclass settings ───────────────
        self.strict_voltage_kill = strict_voltage_kill

        # ── Terminal-state buffers (written by reset_(), read by step_wait()) ──
        # reset_() is called BEFORE physics reset, so these capture the genuine
        # end-of-episode values — not the fresh post-reset values.
        self._ep_terminal_energy_j = np.zeros(self.num_envs, dtype=np.float64)
        self._ep_terminal_soc      = np.ones(self.num_envs,  dtype=np.float64)
        self._ep_terminal_voltage  = np.full(
            self.num_envs, LiPoBatteryModel.VOLTAGE_FULL, dtype=np.float64
        )
        self._ep_terminal_power    = np.zeros(self.num_envs, dtype=np.float64)

        # ── Per-env battery observation buffer ────────────────────────────────
        self._batt_obs = np.zeros((self.num_envs, 3), dtype=np.float32)
        self._update_batt_obs()

        # ── Extend observation space by 3 battery dims ────────────────────────
        n_ext = self.obs_len + 3
        self.observation_space = spaces.Box(
            low=np.full(n_ext, -np.inf, dtype=np.float64),
            high=np.full(n_ext,  np.inf, dtype=np.float64),
            dtype=np.float64,
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # Internal helpers
    # ═══════════════════════════════════════════════════════════════════════════

    def _update_batt_obs(self) -> None:
        """Recompute the (num_envs, 3) battery observation from current battery states."""
        self._batt_obs[:, 0] = self.bat_soc
        self._batt_obs[:, 1] = (self.bat_voltage - self._V_MIN) / (self._V_MAX - self._V_MIN)
        self._batt_obs[:, 2] = self.bat_power / self._P_MAX

    def _apply_soc_randomisation(self, dones: np.ndarray) -> None:
        """
        If randomize_soc is active, draw a random starting SoC ∈ [0.5, 1.0]
        for battery i (domain randomisation).  Called AFTER the parent's
        reset_ has already reset the battery to full charge.
        """
        if self._randomize_soc:
            idx = np.where(dones)[0]
            if len(idx) > 0:
                soc = np.random.uniform(0.5, 1.0, size=len(idx))
                self.bat_soc[idx] = soc
                self.bat_capacity_ah[idx] = soc * LiPoBatteryModel.CAPACITY_AH
                
                # Recompute ideal voltage for the new SoC
                v_ideal = 13.0 + 3.8 * np.sqrt(soc)
                self.bat_voltage[idx] = v_ideal

    # ═══════════════════════════════════════════════════════════════════════════
    # Public API — battery domain randomisation
    # ═══════════════════════════════════════════════════════════════════════════

    def reset_random(self) -> np.ndarray:
        """
        Reset all environments with randomised starting SoC ∈ [0.3, 1.0].

        Temporarily enables SoC randomisation regardless of the randomize_soc
        constructor flag, making it safe to call from a training loop that
        normally uses full-charge resets for eval but random resets for training:

            train_obs = env.reset_random()   # stochastic SoC for PPO rollouts
            eval_obs  = env.reset()          # full charge for deterministic eval
        """
        prev = self._randomize_soc
        self._randomize_soc = True
        obs = self.reset()          # → reset_() for all envs, batteries randomised
        self._randomize_soc = prev
        return obs

    # ═══════════════════════════════════════════════════════════════════════════
    # Overridden DroneGateEnv methods
    # ═══════════════════════════════════════════════════════════════════════════

    def reset_(self, dones: np.ndarray) -> np.ndarray:
        """
        Extended reset hook called by the parent for every environment that
        finishes an episode.

        Phases:
        1. PRE-RESET  — capture terminal battery state into _ep_terminal_*
                        buffers so step_wait() can apply the sparse reward.
        2. PHYSICS RESET — delegate to DroneGateEnv.reset_() which resets
                          kinematics AND batteries (full charge).
        3. POST-RESET — apply SoC randomisation on top of the parent's
                        battery reset, refresh battery obs.
        """
        # Phase 1: capture terminal state BEFORE any battery reset
        self._ep_terminal_energy_j[dones] = self.bat_energy_j[dones]
        self._ep_terminal_soc[dones] = self.bat_soc[dones]
        self._ep_terminal_voltage[dones] = self.bat_voltage[dones]
        self._ep_terminal_power[dones] = self.bat_power[dones]

        # Phase 2: parent resets kinematic state, world_states, step_counts,
        #          AND batteries (to full charge).
        base_obs = super().reset_(dones)    # → self.states updated here

        # Phase 3: apply SoC randomisation on top of the parent's battery reset
        self._apply_soc_randomisation(dones)

        self._update_batt_obs()
        return np.concatenate([base_obs, self._batt_obs], axis=1)

    def reset(self) -> np.ndarray:
        """
        Full environment reset — delegates to reset_() for all envs.
        Returns the extended observation (std obs ++ battery obs).
        """
        obs = super().reset()
        # reset_() already updated _batt_obs; obs was extended there
        return obs

    def step_wait(self) -> tuple:
        """
        Extended step:

        1. Capture pre-depletion state (before parent's battery step).
        2. Compute overdraw penalties from current actions (before parent clamps).
        3. Run parent step_wait() — current limiter + battery step + voltage sag
           + physics + base rewards + base resets (which calls our reset_() override).
        4. Detect battery-depletion terminations (not caught by parent).
        5. Apply power-aware reward shaping (dense / sparse / hybrid + overdraw).
        6. Augment info dicts with battery telemetry.
        7. Extend observations with battery state.

        Returns
        -------
        obs_ext   : (num_envs, obs_len+3) float32
        rewards   : (num_envs,)
        dones     : (num_envs,) bool
        infos     : list of dicts — each done env includes battery telemetry
        """
        # ── 1. Capture pre-depletion state ────────────────────────────────────
        pre_depleted = self.bat_is_depleted.copy()

        # ── 2. Compute overdraw penalties from current actions ────────────────
        if self.action_filter_alpha < 1.0:
            check_actions = (
                self.action_filter_alpha * self.actions
                + (1.0 - self.action_filter_alpha) * self.filtered_actions
            )
        else:
            check_actions = self.actions

        requested_rpms = (check_actions + 1.0) / 2.0
        rpm_ratio = np.clip(requested_rpms / self._max_rpm, 0.0, 1.0)
        motor_currents = (
            LiPoBatteryModel.MOTOR_IDLE_CURRENT
            + (LiPoBatteryModel.MOTOR_MAX_CURRENT - LiPoBatteryModel.MOTOR_IDLE_CURRENT) * rpm_ratio ** 2
        )
        i_theoretical = LiPoBatteryModel.FC_BASELINE_CURRENT + np.sum(motor_currents, axis=1)
        
        excess = i_theoretical - LiPoBatteryModel.BATTERY_MAX_CURRENT
        overdraw_penalties = np.where(excess > 0, self._overdraw_penalty_weight * excess, 0.0)

        # ── 3. Parent step (current limiter + battery + physics + resets) ─────
        _obs, rewards, base_dones, infos = super().step_wait()

        # ── 4. Battery-depletion terminations ─────────────────────────────────
        post_depleted = self.bat_is_depleted
        just_depleted = post_depleted & ~pre_depleted & ~base_dones
        batt_dones = just_depleted
        if np.any(batt_dones):
            self.reset_(batt_dones)

        all_dones = base_dones | batt_dones

        # ── 5. Power-aware reward shaping ─────────────────────────────────────
        # Overdraw penalty (applied every step where the limiter fired)
        rewards -= overdraw_penalties

        if self.experiment_type in (1, 3) and self._dense_weight > 0.0:
            # Dense penalty: every step, for every env (including terminal step).
            rewards -= self._dense_weight * self.bat_power

        if self.experiment_type in (2, 3) and np.any(all_dones):
            done_idx = np.where(all_dones)[0]
            if self.experiment_type == 2:
                # Sparse penalty: deduct total episode energy from final reward.
                rewards[done_idx] -= self._sparse_weight * self._ep_terminal_energy_j[done_idx]
            elif self.experiment_type == 3:
                # Hybrid bonus: grant +sparse_bonus × SoC_final_%  if the battery survived
                survived = ~just_depleted[done_idx]
                rewards[done_idx[survived]] += self._sparse_bonus * (self._ep_terminal_soc[done_idx[survived]] * 100.0)

        # ── 6. Augment info dicts with battery telemetry ──────────────────────
        for i in np.where(all_dones)[0]:
            infos[i]["battery_final_soc"]      = float(self._ep_terminal_soc[i])
            infos[i]["battery_final_voltage"]  = float(self._ep_terminal_voltage[i])
            infos[i]["battery_energy_j"]       = float(self._ep_terminal_energy_j[i])
            infos[i]["battery_died"]           = bool(just_depleted[i])
            infos[i]["overdraw_penalty"]       = float(overdraw_penalties[i])

            if "terminal_observation" in infos[i]:
                term_soc = self._ep_terminal_soc[i]
                term_v_norm = (self._ep_terminal_voltage[i] - self._V_MIN) / (self._V_MAX - self._V_MIN)
                term_p_norm = self._ep_terminal_power[i] / self._P_MAX
                
                # Create the 3-element battery array
                term_batt_obs = np.array([term_soc, term_v_norm, term_p_norm], dtype=np.float32)
                
                # Append it to the base (22,) observation to make it (25,)
                infos[i]["terminal_observation"] = np.concatenate([
                    infos[i]["terminal_observation"], 
                    term_batt_obs
                ])

        # ── 7. Extend observations ────────────────────────────────────────────
        self._update_batt_obs()
        obs_ext = np.concatenate([self.states, self._batt_obs], axis=1)

        return obs_ext, rewards, all_dones, infos
