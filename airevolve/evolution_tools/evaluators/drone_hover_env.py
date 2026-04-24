import torch
import numpy as np

# Efficient vectorized version of the environment
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

# Import new simulation API
from airevolve.simulator.simulation.drone_simulator import DroneSimulator
from airevolve.simulator.simulation.drone_configuration import DroneConfiguration
from airevolve.evolution_tools.genome_handlers.mounting_points import (
    generate_disc_mounting_points, assign_nearest_mounting_point
)


class DroneHoverEnv(VecEnv):
    """
    Vectorized environment for drone hovering task.

    The goal is to maintain a stable hover at a target position with minimal velocity
    and attitude deviation. This serves as a foundational task for curriculum learning.
    """

    metadata = {'render_modes': ['rgb_array', 'human']}
    render_mode = 'rgb_array'

    def __init__(self,
                 num_envs,
                 propellers=None,
                 individual=None,
                 target_pos=np.array([0.0, 0.0, -1.5], dtype=np.float32),
                 x_bounds=[-3, 3],
                 y_bounds=[-3, 3],
                 z_bounds=[-3, 0],
                 motor_limit=1.0,
                 position_tolerance=0.5,
                 velocity_tolerance=0.3,
                 attitude_tolerance=np.pi/6,
                 num_state_history=0,
                 num_action_history=0,
                 history_step_size=1,
                 seed=None,
                 render_mode=None,
                 device=None,
                 dt=0.01,
                 add_disturbances=False,
                 disturbance_strength=0.1
                 ):
        """
        Initialize the hover environment.

        Args:
            num_envs: Number of parallel environments
            propellers: Propeller configuration (new API)
            individual: Legacy individual array format
            target_pos: Target hover position [x, y, z]
            x_bounds, y_bounds, z_bounds: Spatial boundaries
            motor_limit: Maximum motor command
            position_tolerance: Tolerance for position error (for success metric)
            velocity_tolerance: Tolerance for velocity (for success metric)
            attitude_tolerance: Tolerance for roll/pitch angles
            num_state_history: Number of previous states to include in observation
            num_action_history: Number of previous actions to include in observation
            history_step_size: Step size for history sampling
            seed: Random seed
            render_mode: Rendering mode
            device: Torch device
            dt: Simulation timestep
            add_disturbances: Whether to add random disturbances
            disturbance_strength: Strength of random disturbances
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

        # set seed
        self.seed = seed
        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        # Initialize drone simulator
        if propellers is not None:
            # Use new API with propeller configuration
            self.drone_sim = DroneSimulator(propellers=propellers, dt=dt)
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

        # Hover target
        self.target_pos = target_pos.astype(np.float32)

        # Tolerances for success metrics
        self.position_tolerance = position_tolerance
        self.velocity_tolerance = velocity_tolerance
        self.attitude_tolerance = attitude_tolerance

        # Motor limit
        self.motor_limit = motor_limit

        # Disturbances
        self.add_disturbances = add_disturbances
        self.disturbance_strength = disturbance_strength

        # state, action history
        self.num_state_history = num_state_history
        self.num_action_history = num_action_history
        self.history_step_size = history_step_size

        # action space: [cmd1, cmd2, ..., cmdN] where N is number of motors
        # U = (u+1)/2 --> u = 2U-1
        u_lim = 2*self.motor_limit-1
        action_space = spaces.Box(low=-1, high=u_lim, shape=(num_motors,), dtype=np.float64)

        # observation space:
        # - position relative to target [3]
        # - velocity [3]
        # - attitude (roll, pitch, yaw) [3]
        # - angular rates [3]
        # - dummy gate info for curriculum learning compatibility [4]
        # - previous actions [num_motors * num_action_history]
        # Total: 16 + num_motors*num_action_history (compatible with gate env)
        self.state_len = 16 + num_motors*self.num_action_history
        self.obs_len = self.state_len*(1+self.num_state_history)
        observation_space = spaces.Box(
            low  = np.array([-np.inf]*self.obs_len),
            high = np.array([ np.inf]*self.obs_len), dtype=np.float64
        )

        # Initialize the VecEnv
        VecEnv.__init__(self, num_envs, observation_space, action_space)

        # world state: pos[W], vel[W], att[eulerB->W], rates[B]
        self.world_states = np.zeros((num_envs,12), dtype=np.float32)
        # observation state
        self.states = np.zeros((num_envs,self.obs_len), dtype=np.float32)
        # state history tracking
        num_hist = 40
        self.state_hist = np.zeros((num_envs,num_hist,self.state_len), dtype=np.float32)
        # action history tracking
        self.action_hist = np.zeros((num_envs,num_hist,num_motors), dtype=np.float32)

        # Define any other environment-specific parameters
        self.max_steps = 1200      # Maximum number of steps in an episode
        self.dt = np.float32(dt)   # Time step duration

        self.step_counts = np.zeros(num_envs, dtype=int)
        self.actions = np.zeros((num_envs,num_motors), dtype=np.float32)
        self.prev_actions = np.zeros((num_envs,num_motors), dtype=np.float32)
        self.dones = np.zeros(num_envs, dtype=bool)

        # Track hover success metrics
        self.hover_success_steps = np.zeros(num_envs, dtype=int)
        self.total_hover_time = np.zeros(num_envs, dtype=np.float32)

        self.num_motors = num_motors

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
        # Remove NaN rows
        valid_rows = ~np.isnan(individual).any(axis=1)
        individual_clean = individual[valid_rows]

        propellers = []
        propeller_positions = []
        for row in individual_clean:
            magnitude, arm_yaw, arm_pitch, mot_pitch, mot_yaw, direction = row

            # Position: spherical to ENU cartesian
            enu_x = magnitude * np.cos(arm_pitch) * np.cos(arm_yaw)
            enu_y = magnitude * np.cos(arm_pitch) * np.sin(arm_yaw)
            enu_z = magnitude * np.sin(arm_pitch)

            # ENU to NED: (x, y, z) → (y, x, -z)
            x, y, z = enu_y, enu_x, -enu_z

            # Thrust direction in NED frame
            # Matches orientation_to_unit_vector(0, mot_pitch, mot_yaw) from hovering_info.py:
            #   R = ENU_to_NED @ euler_R(0, pitch, yaw); thrust = R @ [0, 0, -1]
            sp, cp = np.sin(mot_pitch), np.cos(mot_pitch)
            sy, cy = np.sin(mot_yaw), np.cos(mot_yaw)
            thrust_x = -sy * sp
            thrust_y = -cy * sp
            thrust_z = cp

            # Rotation direction
            rotation = "cw" if direction > 0.5 else "ccw"

            propellers.append({
                "loc": [x, y, z],
                "dir": [thrust_x, thrust_y, thrust_z, rotation],
                "propsize": 2  # Default prop size
            })
            propeller_positions.append([x, y, z])

        # Compute mounting points matching get_sim() in hovering_info.py
        disc_mounting_points = generate_disc_mounting_points(num_points=8, diameter=0.060)
        mounting_points = assign_nearest_mounting_point(propeller_positions, disc_mounting_points)

        return propellers, mounting_points

    def reset_seed(self):
        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

    def update_states(self):
        """Update observation states based on world states."""
        # new state array to prevent the weird bug related to indexing ([:] syntax)
        new_states = np.zeros((self.num_envs,self.state_len), dtype=np.float32)

        # Position relative to target
        pos_W = self.world_states[:,0:3]
        new_states[:,0:3] = pos_W - self.target_pos

        # Velocity
        new_states[:,3:6] = self.world_states[:,3:6]

        # Attitude (roll, pitch, yaw)
        new_states[:,6:9] = self.world_states[:,6:9]

        # Angular rates
        new_states[:,9:12] = self.world_states[:,9:12]

        # Dummy gate info [12:16] (zeros for curriculum learning compatibility)
        # This matches the gate environment's "next gate" information
        new_states[:,12:16] = 0.0

        # update action history
        self.action_hist = np.roll(self.action_hist, 1, axis=1)
        self.action_hist[:,0] = self.actions

        for i in range(self.num_action_history):
            start_idx = 16 + self.num_motors*i
            end_idx = 16 + self.num_motors*(i+1)
            new_states[:,start_idx:end_idx] = self.action_hist[:,(i+1)*self.history_step_size-1]

        # update state history
        self.state_hist = np.roll(self.state_hist, 1, axis=1)
        self.state_hist[:,0] = new_states

        # stack history up to self.num_state_history
        self.states = self.state_hist[:,0:(self.num_state_history+1)*self.history_step_size:self.history_step_size].reshape((self.num_envs,-1))

    def reset_(self, dones):
        num_reset = dones.sum()

        # Initialize positions near target with some randomness
        x0 = np.full(num_reset, self.target_pos[0]) #+ np.random.uniform(-0.5, 0.5, size=(num_reset,))
        y0 = np.full(num_reset, self.target_pos[1]) #+ np.random.uniform(-0.5, 0.5, size=(num_reset,))
        z0 = np.full(num_reset, self.target_pos[2]) #+ np.random.uniform(-0.3, 0.3, size=(num_reset,))

        # Small initial velocities
        vx0 = np.zeros((num_reset,))
        vy0 = np.zeros((num_reset,))
        vz0 = np.zeros((num_reset,))

        # Small initial attitudes (mostly level)
        phi0   = np.zeros((num_reset,))
        theta0 = np.zeros((num_reset,))
        psi0   = np.zeros((num_reset,))  # Yaw set to zero

        # Small initial rates
        p0 = np.zeros((num_reset,))
        q0 = np.zeros((num_reset,))
        r0 = np.zeros((num_reset,))

        self.world_states[dones] = np.stack([x0, y0, z0, vx0, vy0, vz0, phi0, theta0, psi0, p0, q0, r0], axis=1)
        self.step_counts[dones] = np.zeros(num_reset)

        # Reset hover success tracking
        self.hover_success_steps[dones] = 0
        self.total_hover_time[dones] = 0.0

        # update states
        self.update_states()
        return self.states

    def reset(self):
        return self.reset_(np.ones(self.num_envs, dtype=bool))

    def step_async(self, actions):
        self.prev_actions = self.actions
        self.actions = actions

    def step_wait(self):
        # Convert actions from [-1,1] to [0,1] range for motor commands
        motor_commands = np.clip((self.actions + 1) / 2, 0, 1)

        # Add random disturbances if enabled
        if self.add_disturbances:
            disturbance = np.random.randn(self.num_envs, 12) * self.disturbance_strength
            disturbance[:, 0:3] *= 0.1  # Smaller position disturbances
            disturbance[:, 3:6] *= 1.0  # Velocity disturbances
            disturbance[:, 6:9] *= 0.05  # Small attitude disturbances
            disturbance[:, 9:12] *= 0.5  # Rate disturbances
        else:
            disturbance = 0

        # Use the new simulator's dynamics function
        new_states = self.world_states + self.dt * self.drone_sim.dynamics_func(self.world_states.T, motor_commands.T).T + disturbance * self.dt

        # Detect numerical divergence (NaN, Inf, or excessively large finite values)
        diverged = np.any(~np.isfinite(new_states) | (np.abs(new_states) > 1e6), axis=1)
        if np.any(diverged):
            new_states[diverged] = self.world_states[diverged]

        self.step_counts += 1

        pos_current = self.world_states[:,0:3]
        pos_new = new_states[:,0:3]

        # Calculate distance to target
        distance_to_target = np.linalg.norm(pos_new - self.target_pos, axis=1)

        # Calculate velocity magnitude
        velocity_magnitude = np.linalg.norm(new_states[:,3:6], axis=1)

        # Calculate attitude error (roll and pitch only - yaw doesn't matter for hovering)
        attitude_error = np.sqrt(new_states[:,6]**2 + new_states[:,7]**2)

        # POSITIVE REWARD FORMULATION
        # Reward for being close to target (1.0 when at target, 0.0 when 1m away)
        position_reward = np.maximum(0.0, 1.0 - distance_to_target)

        # Bonus for being in hover zone (close to target, low velocity)
        hover_success = (distance_to_target < self.position_tolerance) & (velocity_magnitude < self.velocity_tolerance) & (attitude_error < self.attitude_tolerance)
        hover_bonus = 1.0 * hover_success.astype(np.float32)

        # Track hover success
        self.hover_success_steps[hover_success] += 1
        self.total_hover_time[hover_success] += self.dt

        # Total reward (always positive, encourages staying alive and close to target)
        rewards = position_reward + hover_bonus

        # Check if drone is too far from target (this IS the out of bounds check)
        out_of_bounds = distance_to_target > 1.0

        # Check number of steps
        max_steps_reached = self.step_counts >= self.max_steps

        # Check if the episode is done
        dones = max_steps_reached | out_of_bounds | diverged
        self.dones = dones

        # Update world states
        self.world_states = new_states

        # Write info dicts BEFORE resetting (so we capture the final metrics)
        infos = [{} for _ in range(self.num_envs)]
        for i in range(self.num_envs):
            if dones[i]:
                infos[i]["terminal_observation"] = self.states[i]
                infos[i]["hover_success_steps"] = self.hover_success_steps[i]
                infos[i]["total_hover_time"] = self.total_hover_time[i]
                infos[i]["hover_success_rate"] = self.hover_success_steps[i] / max(1, self.step_counts[i])
            if max_steps_reached[i]:
                infos[i]["TimeLimit.truncated"] = True
            # extra info for debugging
            infos[i]["out_of_bounds"] = out_of_bounds[i]
            infos[i]["distance_to_target"] = distance_to_target[i]
            infos[i]["in_hover_zone"] = hover_success[i]

        # reset env if done (and update states) - AFTER capturing metrics
        self.reset_(dones)

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
        # Outputs a dict containing all information for rendering
        state_dict = dict(zip(['x','y','z','vx','vy','vz','phi','theta','psi','p','q','r'], self.world_states.T))
        # Rescale actions to [0,1] for rendering
        action_dict = dict(zip([f'u{i}' for i in range(1,self.num_motors+1)], (np.array(self.actions.T)+1)/2))
        return {**state_dict, **action_dict}
