"""
drone_gate_env_power.py — PowerAwareDroneEnv

DroneGateEnv extended with motor-level ECM LiPo battery dynamics and three
power-aware PPO reward-shaping strategies, plus a Betaflight-style current
limiter that caps RPMs before they reach the physics engine.

Class hierarchy
---------------
VecEnv (SB3 abstract)
└── DroneGateEnv          (existing env — physics, gates, base reward)
    └── PowerAwareDroneEnv  (this file — battery, power rewards, domain rand.)

Additions over the base env
----------------------------
Battery (per env)
  • One LiPoBatteryModel instance per parallel environment.
  • Stepped every tick using per-motor RPMs and max_rpm via the quadratic ECM.
  • Episode terminates when battery.is_depleted  (SoC < 15% OR V < 12.8 V
    when strict_voltage_kill=True; SoC < 15% only when False).

Current Limiter (Betaflight-style)
  • Pre-physics: if I_theoretical > BATTERY_MAX_CURRENT (71.25 A),
    RPMs are scaled by sqrt(BATTERY_MAX_CURRENT / I_theoretical).
  • Overdraw penalty: reward -= OVERDRAW_PENALTY_WEIGHT * delta_I.

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
    DroneGateEnv + motor-level ECM battery + Betaflight current limiter +
    power-aware reward shaping.

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

        # Parent sets up drone physics, world_states, obs/action spaces, etc.
        super().__init__(**kwargs)

        self.experiment_type   = experiment_type
        self._randomize_soc    = randomize_soc
        self._default_rand_soc = randomize_soc  # preserve for reset_random()
        self._strict_voltage_kill = strict_voltage_kill

        # Actions are normalised to [-1, 1] (the reference-form motor convention:
        # w_i ∈ [-1, 1] maps to W_i ∈ [0, W_MAX_N]).  We use max_rpm = 1.0 so
        # that the quadratic RPM curve treats each action element directly as a
        # fraction of maximum throttle.  Negative values are clamped to 0 inside
        # compute_current_from_rpms() via np.clip, which correctly represents
        # below-idle commands drawing only the per-motor idle current.
        self._max_rpm = 1.0
        self._overdraw_penalty_weight = (
            float(overdraw_penalty_weight)
            if overdraw_penalty_weight is not None
            else self.OVERDRAW_PENALTY_WEIGHT
        )

        pw = penalty_weights or {}
        self._dense_weight  = float(pw.get("dense_weight",  0.0))
        self._sparse_weight = float(pw.get("sparse_weight", 0.0))
        self._sparse_bonus  = float(pw.get("sparse_bonus",  0.0))

        # ── Battery array — one instance per parallel environment ──────────────
        self._batteries: list[LiPoBatteryModel] = [
            LiPoBatteryModel(strict_voltage_kill=strict_voltage_kill)
            for _ in range(self.num_envs)
        ]

        # ── Terminal-state buffers (written by reset_(), read by step_wait()) ──
        # reset_() is called BEFORE physics reset, so these capture the genuine
        # end-of-episode values — not the fresh post-reset values.
        self._ep_terminal_energy_j = np.zeros(self.num_envs, dtype=np.float64)
        self._ep_terminal_soc      = np.ones(self.num_envs,  dtype=np.float64)
        self._ep_terminal_voltage  = np.full(
            self.num_envs, LiPoBatteryModel.VOLTAGE_FULL, dtype=np.float64
        )

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
        for i, bat in enumerate(self._batteries):
            self._batt_obs[i, 0] = bat.soc
            self._batt_obs[i, 1] = (bat.voltage - self._V_MIN) / (self._V_MAX - self._V_MIN)
            self._batt_obs[i, 2] = bat._last_power / self._P_MAX

    def _reset_single_battery(self, i: int) -> None:
        """
        Reset battery i.  If randomize_soc is active, draw a random starting
        SoC ∈ [0.3, 1.0] after resetting to force the policy to generalise
        across different charge levels (domain randomisation).
        """
        bat = self._batteries[i]
        bat.reset()
        if self._randomize_soc:
            soc = float(np.random.uniform(0.3, 1.0))
            bat._soc                   = soc
            bat._capacity_remaining_ah = soc * bat.CAPACITY_AH
            bat._last_voltage          = bat._compute_ecm_voltage(soc, 0.0)

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

        The method runs in two phases:
        1. PRE-RESET  — capture terminal battery state into _ep_terminal_*
                        buffers so step_wait() can apply the sparse reward.
        2. PHYSICS RESET — delegate to DroneGateEnv.reset_() (resets kinematics).
        3. POST-RESET — reset batteries (optionally with random SoC), refresh
                        battery obs, and return the extended observation.

        NOTE: The return value is used when reset() is called externally.
              When called internally by step_wait(), the return value is
              discarded — self.states is what step_wait() ultimately returns.
        """
        # Phase 1: capture terminal state BEFORE any battery reset
        for i in np.where(dones)[0]:
            bat = self._batteries[i]
            self._ep_terminal_energy_j[i] = bat.get_total_energy_consumed()
            self._ep_terminal_soc[i]      = bat.soc
            self._ep_terminal_voltage[i]  = bat.voltage

        # Phase 2: parent resets kinematic state, world_states, step_counts
        base_obs = super().reset_(dones)    # → self.states updated here

        # Phase 3: reset batteries for done envs
        for i in np.where(dones)[0]:
            self._reset_single_battery(int(i))

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

        0. Current Limiter — intercept requested RPMs; if I_theoretical exceeds
           BATTERY_MAX_CURRENT, scale RPMs by sqrt(scale) and apply overdraw penalty.
        1. Step batteries with the (possibly clamped) motor RPMs.
        2. Run parent step_wait() — physics, base rewards, base dones.
           Internally, parent calls reset_(base_dones) → our override fires,
           saving terminal state and resetting batteries for physics-done envs.
        3. Terminate any envs whose battery just depleted (not caught by parent).
        4. Apply power-aware reward shaping (dense / sparse / hybrid).
        5. Augment info dicts with battery telemetry for done episodes.
        6. Append battery observations to the returned obs.

        Returns
        -------
        obs_ext   : (num_envs, obs_len+3) float32
        rewards   : (num_envs,)
        dones     : (num_envs,) bool
        infos     : list of dicts — each done env includes battery telemetry
        """
        # ── 0. Betaflight-style current limiter ───────────────────────────────
        # actions holds the raw RPM commands from the policy (shape: num_envs × n_motors).
        # We must intercept BEFORE the physics step so the dynamics engine never
        # sees a physically impossible command.
        overdraw_penalties = np.zeros(self.num_envs, dtype=np.float64)
        safe_actions = self.actions.copy()  # self.actions set by step_async()

        bat_ref = self._batteries[0]  # constants are the same for all batteries
        for i in range(self.num_envs):
            requested_rpms = safe_actions[i]
            i_theoretical = bat_ref.compute_current_from_rpms(
                requested_rpms, self._max_rpm
            )
            if i_theoretical > LiPoBatteryModel.BATTERY_MAX_CURRENT:
                scale = LiPoBatteryModel.BATTERY_MAX_CURRENT / i_theoretical
                safe_actions[i] = requested_rpms * np.sqrt(scale)
                delta_i = i_theoretical - LiPoBatteryModel.BATTERY_MAX_CURRENT
                overdraw_penalties[i] = self._overdraw_penalty_weight * delta_i

        # Write clamped actions back so the parent physics step uses them.
        self.actions = safe_actions

        # ── 1. Battery step (clamped motor RPMs) ─────────────────────────────
        # Using the clamped RPMs ensures battery current is consistent with
        # the forces actually applied to the drone.
        pre_depleted = np.array([b.is_depleted for b in self._batteries])

        for i in range(self.num_envs):
            if not pre_depleted[i]:
                self._batteries[i].step(
                    float(self.dt),
                    safe_actions[i],
                    self._max_rpm,
                )

        # Detect envs that crossed the depletion threshold THIS step
        post_depleted  = np.array([b.is_depleted for b in self._batteries])
        just_depleted  = post_depleted & ~pre_depleted

        # ── 2. Parent physics step ────────────────────────────────────────────
        # Inside super().step_wait():
        #   • dynamics computed → world_states updated
        #   • base_dones determined (max_steps | OOB | diverged)
        #   • self.reset_(base_dones) called → our override:
        #       – saves _ep_terminal_* for base_dones envs  ✓
        #       – resets batteries (optionally random SoC)   ✓
        #   • self.states updated to post-reset gate-relative obs
        _obs, rewards, base_dones, infos = super().step_wait()
        # Note: _obs is self.states (may be stale for batt_dones envs below).
        # We rebuild obs from self.states at the end.

        # ── 3. Battery-depletion terminations ─────────────────────────────────
        # Envs that just depleted but were NOT already terminated by the parent.
        batt_dones = just_depleted & ~base_dones
        if np.any(batt_dones):
            # reset_() saves terminal state, resets batteries, updates self.states
            self.reset_(batt_dones)

        all_dones = base_dones | batt_dones

        # ── 4. Power-aware reward shaping ─────────────────────────────────────

        # Overdraw penalty (applied every step where the limiter fired)
        rewards -= overdraw_penalties

        if self.experiment_type in (1, 3) and self._dense_weight > 0.0:
            # Dense penalty: every step, for every env (including terminal step).
            # _last_power is valid because we stepped batteries above (step 1).
            for i in range(self.num_envs):
                rewards[i] -= self._dense_weight * self._batteries[i]._last_power

        if self.experiment_type in (2, 3) and np.any(all_dones):
            for i in np.where(all_dones)[0]:
                ep_energy  = self._ep_terminal_energy_j[i]   # set by reset_()
                ep_soc     = self._ep_terminal_soc[i]
                batt_died  = bool(just_depleted[i])

                if self.experiment_type == 2:
                    # Sparse penalty: deduct total episode energy from final reward.
                    # Weight is in units of reward / Joule — sweep to find the
                    # Pareto frontier between gate-passes and energy efficiency.
                    rewards[i] -= self._sparse_weight * ep_energy

                elif self.experiment_type == 3 and not batt_died:
                    # Hybrid bonus: grant +sparse_bonus × SoC_final_%  if the
                    # battery survived the episode (agent conserved energy).
                    # A battery that died earns NO bonus — the agent must learn
                    # to reach the goal AND land with charge remaining.
                    rewards[i] += self._sparse_bonus * (ep_soc * 100.0)

        # ── 5. Augment info dicts with battery telemetry ──────────────────────
        for i in np.where(all_dones)[0]:
            infos[i]["battery_final_soc"]     = float(self._ep_terminal_soc[i])
            infos[i]["battery_final_voltage"]  = float(self._ep_terminal_voltage[i])
            infos[i]["battery_energy_j"]       = float(self._ep_terminal_energy_j[i])
            infos[i]["battery_died"]           = bool(just_depleted[i])
            infos[i]["overdraw_penalty"]       = float(overdraw_penalties[i])

            if "terminal_observation" in infos[i]:
                # Calculate the normalized battery values as they were right before reset
                term_soc = self._ep_terminal_soc[i]
                term_v_norm = (self._ep_terminal_voltage[i] - self._V_MIN) / (self._V_MAX - self._V_MIN)
                term_p_norm = self._batteries[i]._last_power / self._P_MAX
                
                # Create the 3-element battery array
                term_batt_obs = np.array([term_soc, term_v_norm, term_p_norm], dtype=np.float32)
                
                # Append it to the base (22,) observation to make it (25,)
                infos[i]["terminal_observation"] = np.concatenate([
                    infos[i]["terminal_observation"], 
                    term_batt_obs
                ])

        # ── 6. Extend observations ────────────────────────────────────────────
        # Use self.states (authoritative post-all-resets gate-relative obs).
        # For done envs this is the fresh reset state; for active envs it is
        # the current flight state — consistent with standard SB3 VecEnv contract.
        self._update_batt_obs()
        obs_ext = np.concatenate([self.states, self._batt_obs], axis=1)

        return obs_ext, rewards, all_dones, infos
