"""Vectorized gate-racing and hover envs for PPO.

Mirrors optimal_quad_control_RL/real_quad_env.py (`Quadcopter3DGates`,
`Quadcopter3DHover`), whose policies are flight-verified on the real hex,
generalized to any morphology: observations, reward, termination, reset
distribution (including the order of the random draws) and per-env domain
randomization match the reference, so a seeded run with the real hex's
sysid reproduces it step for step (unit_tests/test_real_hex_parity.py).

`legacy_termination=True` restores the earlier airevolve behaviour (box
bounds only, no ground/gate-frame crashes or rate limit, +10 lap bonus,
deterministic start) for reproducing past evolution runs.
"""
import torch
import numpy as np

# Device is managed per-instance, not globally.
# Do NOT call torch.set_default_device() here — it is a global side effect
# that interferes with explicit device='cpu' passed to PPO/environments.

# Efficient vectorized version of the environment
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

from airevolve.simulator.simulation.drone_simulator import DroneSimulator
from airevolve.simulator.simulation.propeller_data import genome_to_propellers
from airevolve.simulator.simulation.randomization import make_randomization
from airevolve.evolution_tools.genome_handlers.mounting_points import (
    generate_disc_mounting_points, assign_nearest_mounting_point
)

# DEFINE RACE TRACK
r = 1.5
gate_pos = np.array([
    [ r,  -r, -1.5],
    [ 0,   0, -1.5],
    [-r,   r, -1.5],
    [ 0, 2*r, -1.5],
    [ r,   r, -1.5],
    [ 0,   0, -1.5],
    [-r,  -r, -1.5],
    [ 0,-2*r, -1.5]
])
gate_yaw = np.array([1,2,1,0,-1,-2,-1,0])*np.pi/2
start_pos = gate_pos[0] + np.array([0,-1.,0])

# HOVER TASK
hover_pos = np.array([0., 0., -1.5])


def figure8_track(radius=1.5):
    """Gate positions, yaws and start position of the figure-8 track."""
    gates = np.array([
        [ radius,  -radius, -1.5],
        [ 0,        0,      -1.5],
        [-radius,   radius, -1.5],
        [ 0,      2*radius, -1.5],
        [ radius,   radius, -1.5],
        [ 0,        0,      -1.5],
        [-radius,  -radius, -1.5],
        [ 0,     -2*radius, -1.5]
    ])
    return gates, gate_yaw.copy(), gates[0] + np.array([0, -1., 0])


class DroneGateEnv(VecEnv):

    metadata = {'render_modes': ['rgb_array', 'human']}
    render_mode = 'rgb_array'

    def __init__(self,
                 num_envs,
                 propellers=None,
                 individual=None,
                 gates_pos=gate_pos,
                 gate_yaw=gate_yaw,
                 start_pos=start_pos,
                 x_bounds=[-5,5],
                 y_bounds=[-5,5],
                 z_bounds=[-7,0],
                 gates_ahead=1,
                 pause_if_collision=False,
                 motor_limit=1.0,
                 initialize_at_random_gates=True,
                 num_state_history=0,
                 num_action_history=0,
                 history_step_size=1,
                 seed=None,
                 render_mode=None,
                 device=None,
                 dt=0.01,
                 action_filter_alpha=1.0,
                 max_steps=1200,
                 drone_sim=None,
                 sysid_params=None,
                 randomization=0.0,
                 start_pos_noise=0.,
                 legacy_termination=False,
                 ):
        """
        Args (beyond the reference env's):
            propellers / individual / drone_sim / sysid_params: the drone.
                Pass a prebuilt `drone_sim`, or `propellers` (optionally with
                `sysid_params` for an identified drone), or a genome as
                `individual`. Defaults to the standard 2-inch quad.
            x_bounds, y_bounds, z_bounds: flight volume (NED). Leaving it
                ends the episode with -10; z above z_bounds[1] is a ground
                crash. The defaults are the reference env's (|x|,|y| ≤ 5,
                ground at z=0, ceiling at z=-7).
            randomization: domain randomization, either a fraction (0.3 =
                ±30% on every coefficient, 0 = nominal) or a callable
                `draw(n) -> (n, P)` parameter vectors.
            legacy_termination: earlier airevolve termination/reward (box
                bounds x/y/z_bounds, no ground or gate-frame crashes, +10 on
                completing a lap, deterministic start).
        """

        # Set device
        if device is not None:
            self.device = device
            torch.set_default_device(device)

        if render_mode is not None:
            self.render_mode = render_mode

        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.z_bounds = z_bounds
        self.legacy_termination = legacy_termination

        # set seed
        self.seed = seed
        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        # Initialize drone simulator
        if drone_sim is not None:
            self.drone_sim = drone_sim
        elif propellers is not None:
            self.drone_sim = DroneSimulator(propellers=propellers, dt=dt, sysid_params=sysid_params)
        elif individual is not None:
            # Convert legacy individual array to propeller configuration
            propellers, mountpoints = self._convert_individual_to_propellers(individual)
            self.drone_sim = DroneSimulator(propellers=propellers, mountpoints=mountpoints, dt=dt)
        else:
            # Default quadrotor configuration
            self.drone_sim = DroneSimulator.create_standard_drone("quad", dt=dt)

        # Get allocation matrices from the configured simulator
        self.Bf, self.Bm = self.drone_sim.config.get_allocation_matrices()

        num_motors = self.drone_sim.num_motors
        self.num_motors = num_motors

        # Motor time constant (nominal) for reporting.
        self.motor_tau = self.drone_sim.params["tau"]

        # Define the race track
        self.start_pos = start_pos.astype(np.float32)
        self.gate_pos = gates_pos.astype(np.float32)
        self.gate_yaw = gate_yaw.astype(np.float32)
        self.num_gates = gates_pos.shape[0]
        self.gates_ahead = gates_ahead

        # Pause if collision
        self.pause_if_collision = pause_if_collision

        # Uniform noise (m) on the start position when not initializing at random gates
        self.start_pos_noise = start_pos_noise

        # Domain randomization: one parameter vector per env, redrawn on reset
        if callable(randomization):
            self.randomization = randomization
        else:
            self.randomization = make_randomization(self.drone_sim, randomization)
        self.params = self.randomization(num_envs)

        # Motor limit
        self.motor_limit = motor_limit

        # Initialize at random gates
        self.initialize_at_random_gates = initialize_at_random_gates

        # state, action history
        self.num_state_history = num_state_history
        self.num_action_history = num_action_history
        self.history_step_size = history_step_size

        # Calculate relative gates
        # pos,yaw of gate i in reference frame of gate i-1 (assumes a looped track)
        self.gate_pos_rel = np.zeros((self.num_gates,3), dtype=np.float32)
        self.gate_yaw_rel = np.zeros(self.num_gates, dtype=np.float32)
        for i in range(0,self.num_gates):
            self.gate_pos_rel[i] = self.gate_pos[i] - self.gate_pos[i-1]
            # Rotation matrix
            R = np.array([
                [np.cos(self.gate_yaw[i-1]), np.sin(self.gate_yaw[i-1])],
                [-np.sin(self.gate_yaw[i-1]), np.cos(self.gate_yaw[i-1])]
            ])
            self.gate_pos_rel[i,0:2] = R@self.gate_pos_rel[i,0:2]
            self.gate_yaw_rel[i] = self.gate_yaw[i] - self.gate_yaw[i-1]
            # wrap yaw
            self.gate_yaw_rel[i] %= 2*np.pi
            if self.gate_yaw_rel[i] > np.pi:
                self.gate_yaw_rel[i] -= 2*np.pi
            elif self.gate_yaw_rel[i] < -np.pi:
                self.gate_yaw_rel[i] += 2*np.pi

        # Define the target gate for each environment
        self.target_gates = np.zeros(num_envs, dtype=int)

        # Initialize number of gates passed
        self.num_gates_passed = np.zeros(num_envs, dtype=int)

        # action space: one command per motor
        # U = (u+1)/2 --> u = 2U-1
        u_lim = 2*self.motor_limit-1
        action_space = spaces.Box(low=-1, high=u_lim, shape=(num_motors,), dtype=np.float64)

        # observation space: pos[G], vel[G], att[eulerB->G], rates[B], rpms, future_gates[G], future_gate_dirs[G]
        # [G] = reference frame aligned with target gate
        # [B] = body frame
        self.state_len = 12+num_motors+4*self.gates_ahead+num_motors*self.num_action_history
        self.obs_len = self.state_len*(1+self.num_state_history)
        observation_space = spaces.Box(
            low  = np.array([-np.inf]*self.obs_len),
            high = np.array([ np.inf]*self.obs_len), dtype=np.float64
        )

        # Initialize the VecEnv
        VecEnv.__init__(self, num_envs, observation_space, action_space)

        # world state: pos[W], vel[W], att[eulerB->W], rates[B], rpms
        self.world_states = np.zeros((num_envs,12+num_motors), dtype=np.float32)
        # observation state
        self.states = np.zeros((num_envs,self.obs_len), dtype=np.float32)
        # state history tracking
        num_hist = 40
        self.state_hist = np.zeros((num_envs,num_hist,self.state_len), dtype=np.float32)
        # action history tracking
        self.action_hist = np.zeros((num_envs,num_hist,num_motors), dtype=np.float32)

        # Define any other environment-specific parameters
        self.max_steps = int(max_steps)  # Maximum number of steps in an episode
        self.dt = np.float32(dt)   # Time step duration

        # Action low-pass filter (one-pole IIR) modeling a flight controller's
        # RC-smoothing / setpoint-shaping stage. alpha=1.0 = pass-through (no
        # filter, default for backward compat). alpha<1.0 smooths actions:
        #   filtered_a = alpha · raw_a + (1 - alpha) · prev_filtered_a
        # At dt=0.01 (100 Hz), alpha=0.3 corresponds to roughly a 6 Hz cutoff,
        # alpha=0.5 → ~16 Hz, alpha=1.0 → no smoothing (raw policy output).
        # Real Betaflight RC smoothing typically targets 20-30 Hz cutoff.
        self.action_filter_alpha = float(action_filter_alpha)
        self.filtered_actions = np.zeros((num_envs, num_motors), dtype=np.float32)

        self.step_counts = np.zeros(num_envs, dtype=int)
        self.actions = np.zeros((num_envs,num_motors), dtype=np.float32)
        self.prev_actions = np.zeros((num_envs,num_motors), dtype=np.float32)
        self.dones = np.zeros(num_envs, dtype=bool)
        self.final_gate_passed = np.zeros(num_envs, dtype=bool)

        self.update_states = self.update_states_gate

        self.pause = False

    def _convert_individual_to_propellers(self, individual):
        """
        Convert legacy individual array to propeller configuration.

        Uses the same NED coordinate conventions as get_sim() in hovering_info.py:
        - Position: spherical → ENU cartesian → NED via (x,y,z) → (y,x,-z)
        - Thrust: orientation_to_unit_vector(0, pitch, yaw) with internal ENU→NED transform

        Returns:
            (propellers, mounting_points): Propeller configs and disc mounting points,
            matching the conventions used by get_sim() in hovering_info.py.
        """
        propellers = genome_to_propellers(individual)
        propeller_positions = [prop["loc"] for prop in propellers]

        # Compute mounting points matching get_sim() in hovering_info.py
        disc_mounting_points = generate_disc_mounting_points(num_points=8, diameter=0.060)
        mounting_points = assign_nearest_mounting_point(propeller_positions, disc_mounting_points)

        return propellers, mounting_points

    def reset_seed(self):
        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

    def update_states_gate(self):
        # Transform pos and vel in gate frame
        gate_pos = self.gate_pos[self.target_gates%self.num_gates]
        gate_yaw = self.gate_yaw[self.target_gates%self.num_gates]

        # Rotation matrix from world frame to gate frame
        R = np.array([
            [np.cos(gate_yaw), np.sin(gate_yaw)],
            [-np.sin(gate_yaw), np.cos(gate_yaw)]
        ]).transpose((2,1,0))

        # new state array to prevent the weird bug related to indexing ([:] syntax)
        new_states = np.zeros((self.num_envs,self.state_len), dtype=np.float32)

        # Update positions
        pos_W = self.world_states[:,0:3]
        pos_G = (pos_W[:,np.newaxis,0:2] - gate_pos[:,np.newaxis,0:2]) @ R
        new_states[:,0:2] = pos_G[:,0,:]
        new_states[:,2] = pos_W[:,2] - gate_pos[:,2]

        # Update velocities
        vel_W = self.world_states[:,3:6]
        vel_G = (vel_W[:,np.newaxis,0:2]) @ R
        new_states[:,3:5] = vel_G[:,0,:]
        new_states[:,5] = vel_W[:,2]

        # Update attitude
        new_states[:,6:8] = self.world_states[:,6:8]
        yaw = self.world_states[:,8] - gate_yaw
        yaw %= 2*np.pi
        yaw[yaw > np.pi] -= 2*np.pi
        yaw[yaw < -np.pi] += 2*np.pi
        new_states[:,8] = yaw

        # Update rates
        new_states[:,9:12] = self.world_states[:,9:12]

        # Update rpms
        n = self.num_motors
        new_states[:,12:12+n] = self.world_states[:,12:12+n]

        # Update future gates relative to current gate
        for i in range(self.gates_ahead):
            indices = (self.target_gates+i+1)
            # loop when out of bounds
            indices = indices % self.num_gates
            valid = indices < self.num_gates
            new_states[valid,12+n+4*i:12+n+4*i+3] = self.gate_pos_rel[indices[valid]]
            new_states[valid,12+n+4*i+3] = self.gate_yaw_rel[indices[valid]]

        # update action history
        self.action_hist = np.roll(self.action_hist, 1, axis=1)
        self.action_hist[:,0] = self.actions

        for i in range(self.num_action_history):
            start = 12+n+4*self.gates_ahead+n*i
            new_states[:,start:start+n] = self.action_hist[:,(i+1)*self.history_step_size-1]

        # update state history
        self.state_hist = np.roll(self.state_hist, 1, axis=1)
        self.state_hist[:,0] = new_states

        # stack history up to self.num_state_history
        self.states = self.state_hist[:,0:(self.num_state_history+1)*self.history_step_size:self.history_step_size].reshape((self.num_envs,-1))

    def reset_(self, dones):
        # Random draws happen in the same order as the reference env.
        num_reset = dones.sum()
        # Track number of gates passed
        self.num_gates_passed[dones] = np.zeros(num_reset)

        if self.initialize_at_random_gates:
            # set target gates to random gates
            self.target_gates[dones] = np.random.randint(0,self.num_gates, size=num_reset)
            # set position to 1m in front of the target gate
            # gate_pos + [cos(gate_yaw), sin(gate_yaw), 0]
            pos = self.gate_pos[self.target_gates[dones]%self.num_gates]
            yaw = self.gate_yaw[self.target_gates[dones]%self.num_gates]

            pos = pos - np.array([np.cos(yaw), np.sin(yaw), np.zeros_like(yaw)]).T
            x0, y0, z0 = pos.T
        elif self.legacy_termination:
            self._reset_legacy_fixed_start(dones, num_reset)
            return self.states
        else:
            # set target gates to 0
            self.target_gates[dones] = np.zeros(num_reset, dtype=int)
            # use start_pos
            x0 = np.random.uniform(-self.start_pos_noise,self.start_pos_noise, size=(num_reset,)) + self.start_pos[0]
            y0 = np.random.uniform(-self.start_pos_noise,self.start_pos_noise, size=(num_reset,)) + self.start_pos[1]
            z0 = np.random.uniform(-self.start_pos_noise,self.start_pos_noise, size=(num_reset,)) + self.start_pos[2]

        vx0 = np.random.uniform(-0.5,0.5, size=(num_reset,))
        vy0 = np.random.uniform(-0.5,0.5, size=(num_reset,))
        vz0 = np.random.uniform(-0.5,0.5, size=(num_reset,))

        phi0   = np.random.uniform(-np.pi/9,np.pi/9, size=(num_reset,))
        theta0 = np.random.uniform(-np.pi/9,np.pi/9, size=(num_reset,))
        psi0   = np.random.uniform(-np.pi,np.pi, size=(num_reset,))

        p0 = np.random.uniform(-0.1,0.1, size=(num_reset,))
        q0 = np.random.uniform(-0.1,0.1, size=(num_reset,))
        r0 = np.random.uniform(-0.1,0.1, size=(num_reset,))

        # Normalized motor speeds in [-1, 1], one draw per motor.
        w0 = [np.random.uniform(-1,1, size=(num_reset,)) for _ in range(self.num_motors)]

        self.world_states[dones] = np.stack([x0, y0, z0, vx0, vy0, vz0, phi0, theta0, psi0, p0, q0, r0, *w0], axis=1)
        self.step_counts[dones] = np.zeros(num_reset)
        # Clear the action-filter memory for envs that just reset, so the
        # next action isn't smoothed against a stale pre-reset action.
        self.filtered_actions[dones] = 0.0

        # update params (domain randomization)
        self.params[dones] = self.randomization(num_reset)

        # update states
        self.update_states()
        return self.states

    def _reset_legacy_fixed_start(self, dones, num_reset):
        """Earlier airevolve fixed start: at start_pos, at rest, level."""
        self.target_gates[dones] = np.zeros(num_reset, dtype=int)
        state = np.zeros((num_reset, 12+self.num_motors))
        state[:, 0:3] = self.start_pos
        self.world_states[dones] = state
        self.step_counts[dones] = np.zeros(num_reset)
        self.filtered_actions[dones] = 0.0
        self.params[dones] = self.randomization(num_reset)
        self.update_states()

    def reset(self):
        return self.reset_(np.ones(self.num_envs, dtype=bool))

    def step_async(self, actions):
        self.prev_actions = self.actions
        self.actions = actions

    def _integrate(self):
        """Euler step of the per-env dynamics; returns the new world states."""
        # Apply the action low-pass filter (FC setpoint shaping) before the
        # action enters the dynamics. alpha=1.0 is a no-op.
        if self.action_filter_alpha < 1.0:
            self.filtered_actions = (
                self.action_filter_alpha * self.actions
                + (1.0 - self.action_filter_alpha) * self.filtered_actions
            ).astype(np.float32)
            action_for_dynamics = self.filtered_actions
        else:
            action_for_dynamics = self.actions

        state_dot = np.asarray(self.drone_sim.dynamics_func_p(
            self.world_states.T, np.asarray(action_for_dynamics).T, self.params.T
        )).T
        new_states = self.world_states + self.dt*state_dot

        # Numerical divergence (NaN/Inf, e.g. at the Euler singularity):
        # keep the last finite state and end the episode.
        self.diverged = np.any(~np.isfinite(new_states), axis=1)
        if np.any(self.diverged):
            new_states[self.diverged] = self.world_states[self.diverged]
        return new_states

    def _crashed(self, new_states, rewards):
        """Ground / out-of-bounds / divergence: -10 reward, returns masks."""
        n = len(new_states)
        if self.legacy_termination:
            ground_collision = np.zeros(n, dtype=bool)
            out_of_bounds = np.zeros(n, dtype=bool)
            for axis, (lo, hi) in enumerate([self.x_bounds, self.y_bounds, self.z_bounds]):
                out_of_bounds |= (new_states[:,axis] < lo) | (new_states[:,axis] > hi)
        else:
            # Ground collision penalty (z > 0)
            ground_collision = new_states[:,2] > self.z_bounds[1]
            # Check out of bounds
            out_of_bounds = (new_states[:,0] < self.x_bounds[0]) | (new_states[:,0] > self.x_bounds[1])  # edges of the grid
            out_of_bounds |= (new_states[:,1] < self.y_bounds[0]) | (new_states[:,1] > self.y_bounds[1])
            out_of_bounds |= new_states[:,2] < self.z_bounds[0]                    # max height (z-axis point down)
            out_of_bounds |= np.any(np.abs(new_states[:,9:12]) > 1000, axis=1)     # prevent numerical issues
        out_of_bounds |= self.diverged
        rewards[ground_collision] = -10
        rewards[out_of_bounds] = -10
        return ground_collision, out_of_bounds

    def step_wait(self):
        new_states = self._integrate()

        self.step_counts += 1

        pos_old = self.world_states[:,0:3]
        pos_new = new_states[:,0:3]
        pos_gate = self.gate_pos[self.target_gates%self.num_gates]
        yaw_gate = self.gate_yaw[self.target_gates%self.num_gates]

        # Rewards
        d2g_old = np.linalg.norm(pos_old - pos_gate, axis=1)
        d2g_new = np.linalg.norm(pos_new - pos_gate, axis=1)
        rat_penalty = 0.001*np.linalg.norm(new_states[:,9:12], axis=1)

        prog_rewards = d2g_old - d2g_new
        rewards = prog_rewards - rat_penalty

        # Gate passing/collision
        normal = np.array([np.cos(yaw_gate), np.sin(yaw_gate)]).T
        # dot product of normal and position vector over axis 1
        pos_old_projected = (pos_old[:,0]-pos_gate[:,0])*normal[:,0] + (pos_old[:,1]-pos_gate[:,1])*normal[:,1]
        pos_new_projected = (pos_new[:,0]-pos_gate[:,0])*normal[:,0] + (pos_new[:,1]-pos_gate[:,1])*normal[:,1]
        passed_gate_plane = (pos_old_projected < 0) & (pos_new_projected > 0)
        gate_size = 1.5
        gate_passed = passed_gate_plane & np.all(np.abs(pos_new - pos_gate)<gate_size/2, axis=1)
        gate_collision = passed_gate_plane & np.any(np.abs(pos_new - pos_gate)>gate_size/2, axis=1)

        if self.legacy_termination:
            # +10 on the final gate of the lap; gate frames don't end the episode.
            final_gate_passed = gate_passed & (self.target_gates == self.num_gates - 1)
            rewards[final_gate_passed] += 10.0
            gate_collision = np.zeros_like(gate_collision)

        ground_collision, out_of_bounds = self._crashed(new_states, rewards)

        # Check number of steps
        max_steps_reached = self.step_counts >= self.max_steps

        # Update target gate
        self.target_gates[gate_passed] += 1
        self.target_gates[gate_passed] %= self.num_gates

        # Track number of gates passed
        self.num_gates_passed[gate_passed] += 1

        # Check if the episode is done
        dones = max_steps_reached | ground_collision | out_of_bounds | gate_collision
        self.dones = dones

        # Save gates passed before reset (for info dict)
        gates_passed_before_reset = self.num_gates_passed.copy()

        # Pause if collision
        if self.pause:
            dones = dones & ~dones
            self.dones = dones
        elif self.pause_if_collision:
            update = ~dones
            # Update world states
            self.world_states[update] = new_states[update]
            self.update_states()
        else:
            # Update world states
            self.world_states = new_states
            # reset env if done (and update states)
            self.reset_(dones)

        # Write info dicts
        infos = [{} for _ in range(self.num_envs)]
        for i in range(self.num_envs):
            if dones[i]:
                infos[i]["terminal_observation"] = self.states[i]
            if max_steps_reached[i]:
                infos[i]["TimeLimit.truncated"] = True
            # extra info for debugging
            infos[i]["ground_collision"] = ground_collision[i]
            infos[i]["out_of_bounds"] = out_of_bounds[i]
            infos[i]["gate_collision"] = gate_collision[i]
            infos[i]["gate_passed"] = gate_passed[i]
            infos[i]["num_gates_passed"] = gates_passed_before_reset

        return self.states, rewards, dones, infos

    def close(self):
        pass

    def seed(self, seed=None):
        pass

    def get_attr(self, attr_name, indices=None):
        raise AttributeError()

    def set_attr(self, attr_name, value, indices=None):
        pass

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        pass

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False]*self.num_envs

    def render(self, mode='human'):
        # Define base state variable names
        state_keys = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'phi', 'theta', 'psi', 'p', 'q', 'r']

        # Dynamically create RPM keys based on the number of motors
        rpm_keys = [f'w{i+1}' for i in range(self.num_motors)]

        # Combine all keys
        state_keys += rpm_keys

        # Convert `self.world_states.T` into a dictionary
        state_dict = dict(zip(state_keys, self.world_states.T))

        # Rescale actions to [0,1] for rendering
        action_keys = [f'u{i+1}' for i in range(self.num_motors)]
        action_dict = dict(zip(action_keys, (np.array(self.actions.T) + 1) / 2))

        return {**state_dict, **action_dict}


class DroneHoverEnv(DroneGateEnv):
    """Hover at a fixed target point (mirrors `Quadcopter3DHover`).

    Reuses the gate env with a single 'gate' at the target with zero yaw and
    no gates ahead, so the observed position is the offset from the target.

    `yaw_rate_penalty` adds -c·|r| to the reward. The reference reward only
    penalizes body rates by 0.001·|ω|, which leaves hovering while spinning
    about z almost free; c ≈ 0.05 makes the policy stop rotating (it still
    holds whatever heading it has). 0 reproduces `Quadcopter3DHover`.
    """

    def __init__(self, num_envs, target_pos=hover_pos, spawn_radius=1.0, yaw_rate_penalty=0.0, **kwargs):
        self.target_pos = np.array(target_pos, dtype=np.float32)
        self.yaw_rate_penalty = yaw_rate_penalty
        super().__init__(
            num_envs,
            gates_pos=self.target_pos[np.newaxis],
            gate_yaw=np.zeros(1),
            start_pos=self.target_pos,
            gates_ahead=0,
            initialize_at_random_gates=False,
            start_pos_noise=spawn_radius,
            **kwargs
        )

    def step_wait(self):
        new_states = self._integrate()

        self.step_counts += 1

        # Rewards: positive every step so that staying airborne always beats crashing
        pos_error = np.linalg.norm(new_states[:,0:3] - self.target_pos, axis=1)
        vel_penalty = 0.01*np.linalg.norm(new_states[:,3:6], axis=1)
        rat_penalty = 0.001*np.linalg.norm(new_states[:,9:12], axis=1)
        rewards = np.exp(-2*pos_error) - vel_penalty - rat_penalty
        if self.yaw_rate_penalty:
            rewards -= self.yaw_rate_penalty*np.abs(new_states[:,11])

        ground_collision, out_of_bounds = self._crashed(new_states, rewards)

        # Check number of steps
        max_steps_reached = self.step_counts >= self.max_steps

        dones = max_steps_reached | ground_collision | out_of_bounds
        self.dones = dones

        # Update world states and reset env if done (and update states)
        self.world_states = new_states
        self.reset_(dones)

        # Write info dicts
        infos = [{} for _ in range(self.num_envs)]
        for i in range(self.num_envs):
            if dones[i]:
                infos[i]["terminal_observation"] = self.states[i]
            if max_steps_reached[i]:
                infos[i]["TimeLimit.truncated"] = True
            infos[i]["ground_collision"] = ground_collision[i]
            infos[i]["out_of_bounds"] = out_of_bounds[i]
            infos[i]["pos_error"] = pos_error[i]
        return self.states, rewards, dones, infos
