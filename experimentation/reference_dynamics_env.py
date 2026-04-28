"""DroneGateEnv subclass that uses optimal_quad_control_RL's sysid'd 5-inch
dynamics in place of airevolve's morphology-derived simulator. All other
environment behavior — gate logic, reward shape, OOB checks, observation
construction, reset behavior — is inherited from DroneGateEnv.

Used by the session 4 parity experiment in RL_TRAINING_FIXES.md to isolate
whether the airevolve <-> reference training-stability gap is in the
dynamics model. See plan in
/home/jed/.claude/plans/can-you-read-home-jed-workspaces-airevol-floating-sparkle.md.
"""
from __future__ import annotations

import numpy as np

from airevolve.evolution_tools.evaluators.drone_gate_env import DroneGateEnv
from airevolve.simulator.simulation.propeller_data import (
    create_standard_propeller_config,
)

from reference_drone_sim import ReferenceDroneSim, get_variant


class ReferenceDynamicsGateEnv(DroneGateEnv):
    """Drone gate env with reference (optimal_quad_control_RL) dynamics.

    The parent's `step_wait` does its own action -> motor_command -> sqrt
    target -> first-order lag -> dynamics_func(base_state, motor_rpms) flow.
    The reference dynamics packages all of that *into* its dynamics_func,
    which takes the full 16-dim state and the action directly. So we
    re-implement step_wait to call dynamics_func with (state, action) and
    Euler-integrate the full 16-dim state. The reward / OOB / gate logic
    below is byte-for-byte the parent's, so any deviation has to be in the
    dynamics or the motor-state initialization (reference uses w in [-1, 1],
    parent uses [0, 1]).

    Args:
        params_variant: which sysid'd parameter set to use. One of
            "5inch" (default), "3inch", "2inch_derived". See
            reference_drone_sim.get_variant.
    """

    def __init__(self, num_envs: int, params_variant: str = "5inch", **kwargs):
        # Force a 4-motor canonical-quad shape so num_motors == 4. The
        # reference dynamics is hard-coded to 4 motors.
        if kwargs.get("propellers") is None and kwargs.get("individual") is None:
            kwargs["propellers"] = create_standard_propeller_config(
                "quad", arm_length=0.11, prop_size=2
            )
        super().__init__(num_envs=num_envs, **kwargs)
        if self.num_motors != 4:
            raise ValueError(
                f"ReferenceDynamicsGateEnv requires 4 motors, got {self.num_motors}"
            )
        # Replace the morphology-derived simulator with the reference port.
        params, mass, inertia = get_variant(params_variant)
        self.params_variant = params_variant
        self.drone_sim = ReferenceDroneSim(
            dt=float(self.dt), params=params, mass=mass, inertia=inertia
        )
        # Re-randomize the existing world_states' motor-speed slice into
        # [-1, 1] (the parent's __init__ doesn't call reset_, but be defensive).
        self.world_states[:, 12:16] = np.random.uniform(
            -1.0, 1.0, size=(num_envs, 4)
        ).astype(np.float32)

    def reset_(self, dones):
        # Run parent reset (which initializes motor speeds in [0, 1]),
        # then overwrite the motor-speed slice into [-1, 1].
        states = super().reset_(dones)
        num_reset = int(dones.sum())
        if num_reset > 0:
            self.world_states[dones, 12:16] = np.random.uniform(
                -1.0, 1.0, size=(num_reset, 4)
            ).astype(np.float32)
            self.update_states()
        return self.states

    def step_wait(self):
        # Reference dynamics: pass action directly into dynamics_func along
        # with the full 16-dim state. Action mapping (sqrt-poly to Wc),
        # first-order motor lag, and motor reaction torque on Mz are all
        # baked into the symbolic equations.
        full_state = self.world_states  # (num_envs, 16)
        full_state_dot = self.drone_sim.dynamics_func(
            full_state.T, self.actions.T
        ).T  # (num_envs, 16)
        new_states = full_state + self.dt * full_state_dot

        # Numerical divergence guard (parent does the same)
        diverged = np.any(
            ~np.isfinite(new_states) | (np.abs(new_states) > 1e6), axis=1
        )
        if np.any(diverged):
            new_states[diverged] = self.world_states[diverged]

        self.step_counts += 1

        # ---------- the rest is verbatim from DroneGateEnv.step_wait ----------
        pos_old = self.world_states[:, 0:3]
        pos_new = new_states[:, 0:3]
        pos_gate = self.gate_pos[self.target_gates % self.num_gates]
        yaw_gate = self.gate_yaw[self.target_gates % self.num_gates]

        d2g_old = np.linalg.norm(pos_old - pos_gate, axis=1)
        d2g_new = np.linalg.norm(pos_new - pos_gate, axis=1)
        rat_penalty = 0.001 * np.linalg.norm(new_states[:, 9:12], axis=1)
        prog_rewards = d2g_old - d2g_new
        rewards = prog_rewards - rat_penalty

        normal = np.array([np.cos(yaw_gate), np.sin(yaw_gate)]).T
        pos_old_projected = (pos_old[:, 0] - pos_gate[:, 0]) * normal[:, 0] + (
            pos_old[:, 1] - pos_gate[:, 1]
        ) * normal[:, 1]
        pos_new_projected = (pos_new[:, 0] - pos_gate[:, 0]) * normal[:, 0] + (
            pos_new[:, 1] - pos_gate[:, 1]
        ) * normal[:, 1]
        passed_gate_plane = (pos_old_projected < 0) & (pos_new_projected > 0)
        gate_size = 1.5
        gate_passed = passed_gate_plane & np.all(
            np.abs(pos_new - pos_gate) < gate_size / 2, axis=1
        )

        final_gate_passed = gate_passed & (self.target_gates == self.num_gates - 1)
        rewards[final_gate_passed] += 10.0

        x_bounds_broken = np.logical_or(
            new_states[:, 0] < self.x_bounds[0], new_states[:, 0] > self.x_bounds[1]
        )
        y_bounds_broken = np.logical_or(
            new_states[:, 1] < self.y_bounds[0], new_states[:, 1] > self.y_bounds[1]
        )
        z_bounds_broken = np.logical_or(
            new_states[:, 2] < self.z_bounds[0], new_states[:, 2] > self.z_bounds[1]
        )
        out_of_bounds = x_bounds_broken | y_bounds_broken | z_bounds_broken

        rewards[out_of_bounds] = -10
        rewards[diverged] = -10

        max_steps_reached = self.step_counts >= self.max_steps

        self.target_gates[gate_passed] += 1
        self.target_gates[gate_passed] %= self.num_gates
        self.num_gates_passed[gate_passed] += 1

        dones = max_steps_reached | out_of_bounds | diverged
        self.dones = dones

        gates_passed_before_reset = self.num_gates_passed.copy()

        if self.pause:
            dones = dones & ~dones
            self.dones = dones
        elif self.pause_if_collision:
            update = ~dones
            self.world_states[update] = new_states[update]
            self.update_states()
        else:
            self.world_states = new_states
            self.reset_(dones)

        infos = [{} for _ in range(self.num_envs)]
        for i in range(self.num_envs):
            if dones[i]:
                infos[i]["terminal_observation"] = self.states[i]
            if max_steps_reached[i]:
                infos[i]["TimeLimit.truncated"] = True
            infos[i]["out_of_bounds"] = out_of_bounds[i]
            infos[i]["gate_passed"] = gate_passed[i]
            infos[i]["num_gates_passed"] = gates_passed_before_reset

        return self.states, rewards, dones, infos
