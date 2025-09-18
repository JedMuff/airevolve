import torch
import stable_baselines3
import sys
import numpy as np

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

# Efficient vectorized version of the environment
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

# Import new simulation API
from airevolve.simulator.simulation.drone_simulator import DroneSimulator
from airevolve.simulator.simulation.drone_configuration import DroneConfiguration

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
                 z_bounds=[-5,5],
                 gates_ahead=2,
                 pause_if_collision=False,
                 motor_limit=1.0,
                 initialize_at_random_gates=False,
                 num_state_history=0,
                 num_action_history=0,
                 history_step_size=1,
                 seed=None,
                 render_mode=None,
                 device=None,
                 dt=0.01
                 ):
        
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
            propellers = self._convert_individual_to_propellers(individual)
            self.drone_sim = DroneSimulator(propellers=propellers, dt=dt)
        else:
            # Default quadrotor configuration
            self.drone_sim = DroneSimulator.create_standard_drone("quad", dt=dt)
        
        # Get allocation matrices from the configured simulator
        self.Bf, self.Bm = self.drone_sim.config.get_allocation_matrices()

        num_motors = self.drone_sim.num_motors

        # Define the race track
        self.start_pos = start_pos.astype(np.float32)
        self.gate_pos = gates_pos.astype(np.float32)
        self.gate_yaw = gate_yaw.astype(np.float32)
        self.num_gates = gates_pos.shape[0]
        self.gates_ahead = gates_ahead
        
        # Pause if collision
        self.pause_if_collision = pause_if_collision

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

        # action space: [cmd1, cmd2, cmd3, cmd4]
        # U = (u+1)/2 --> u = 2U-1
        u_lim = 2*self.motor_limit-1
        action_space = spaces.Box(low=-1, high=u_lim, shape=(num_motors,), dtype=np.float64)

        # observation space: pos[G], vel[G], att[eulerB->G], rates[B], rpms, future_gates[G], future_gate_dirs[G]
        # [G] = reference frame aligned with target gate
        # [B] = body frame
        self.state_len = 12+4*self.gates_ahead+4*self.num_action_history
        self.obs_len = self.state_len*(1+self.num_state_history)
        observation_space = spaces.Box(
            low  = np.array([-np.inf]*self.obs_len),
            high = np.array([ np.inf]*self.obs_len), dtype=np.float64
        )

        # Initialize the VecEnv
        VecEnv.__init__(self, num_envs, observation_space, action_space)

        # world state: pos[W], vel[W], att[eulerB->W], rates[B], rpms
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
        self.final_gate_passed = np.zeros(num_envs, dtype=bool)

        self.update_states = self.update_states_gate
        
        self.pause = False
        
        self.num_motors = num_motors
    
    def _convert_individual_to_propellers(self, individual):
        """Convert legacy individual array to propeller configuration."""
        # Remove NaN rows
        valid_rows = ~np.isnan(individual).any(axis=1)
        individual_clean = individual[valid_rows]
        
        propellers = []
        for row in individual_clean:
            magnitude, arm_yaw, arm_pitch, mot_yaw, mot_pitch, direction = row
            
            # Convert spherical to Cartesian coordinates
            x = magnitude * np.cos(arm_pitch) * np.cos(arm_yaw)
            y = magnitude * np.cos(arm_pitch) * np.sin(arm_yaw)
            z = magnitude * np.sin(arm_pitch)
            
            # Motor orientation from angles (simplified - assume mostly downward thrust)
            thrust_x = np.sin(mot_pitch) * np.cos(mot_yaw)
            thrust_y = np.sin(mot_pitch) * np.sin(mot_yaw)
            thrust_z = -np.cos(mot_pitch)  # Mostly downward
            
            # Rotation direction
            rotation = "cw" if direction > 0.5 else "ccw"
            
            propellers.append({
                "loc": [x, y, z],
                "dir": [thrust_x, thrust_y, thrust_z, rotation],
                "propsize": 5  # Default prop size
            })
        
        return propellers
    
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

        # Update future gates relative to current gate
        for i in range(self.gates_ahead):
            indices = (self.target_gates+i+1)
            # loop when out of bounds
            indices = indices % self.num_gates
            valid = indices < self.num_gates
            new_states[valid,12+4*i:12+4*i+3] = self.gate_pos_rel[indices[valid]]
            new_states[valid,12+4*i+3] = self.gate_yaw_rel[indices[valid]]

        # update action history
        self.action_hist = np.roll(self.action_hist, 1, axis=1)
        self.action_hist[:,0] = self.actions
        
        for i in range(self.num_action_history):
            new_states[:,12+4*self.gates_ahead+4*i:12+4*self.gates_ahead+4*i+4] = self.action_hist[:,(i+1)*self.history_step_size-1]
        
        # update state history
        self.state_hist = np.roll(self.state_hist, 1, axis=1)
        self.state_hist[:,0] = new_states

        # stack history up to self.num_state_history
        self.states = self.state_hist[:,0:(self.num_state_history+1)*self.history_step_size:self.history_step_size].reshape((self.num_envs,-1))

    def reset_(self, dones):
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

            vx0 = np.random.uniform(-0.5,0.5, size=(num_reset,))
            vy0 = np.random.uniform(-0.5,0.5, size=(num_reset,))
            vz0 = np.random.uniform(-0.5,0.5, size=(num_reset,))
            
            phi0   = np.random.uniform(-np.pi/9,np.pi/9, size=(num_reset,))
            theta0 = np.random.uniform(-np.pi/9,np.pi/9, size=(num_reset,))
            psi0   = np.random.uniform(-np.pi,np.pi, size=(num_reset,))
            
            p0 = np.random.uniform(-0.1,0.1, size=(num_reset,))
            q0 = np.random.uniform(-0.1,0.1, size=(num_reset,))
            r0 = np.random.uniform(-0.1,0.1, size=(num_reset,))

        else: # always start at the first gate, fixed orientation
            # set target gates to 0
            self.target_gates[dones] = np.zeros(num_reset, dtype=int)
            # use start_pos
            x0 = 0*np.random.uniform(-0.5,0.5, size=(num_reset,)) + self.start_pos[0]
            y0 = 0*np.random.uniform(-0.5,0.5, size=(num_reset,)) + self.start_pos[1]
            z0 = 0*np.random.uniform(-0.5,0.5, size=(num_reset,)) + self.start_pos[2]

            vx0 = np.zeros((num_reset,))
            vy0 = np.zeros((num_reset,))
            vz0 = np.zeros((num_reset,))
            
            phi0   = np.zeros((num_reset,))
            theta0 = np.zeros((num_reset,))
            psi0   = np.zeros((num_reset,))
            
            p0 = np.zeros((num_reset,))
            q0 = np.zeros((num_reset,))
            r0 = np.zeros((num_reset,))

        self.world_states[dones] = np.stack([x0, y0, z0, vx0, vy0, vz0, phi0, theta0, psi0, p0, q0, r0], axis=1)
        self.step_counts[dones] = np.zeros(num_reset)
        
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
        
        # Use the new simulator's dynamics function
        new_states = self.world_states + self.dt * self.drone_sim.dynamics_func(self.world_states.T, motor_commands.T).T

        self.step_counts += 1

        pos_old = self.world_states[:,0:3]
        pos_new = new_states[:,0:3]
        pos_gate = self.gate_pos[self.target_gates%self.num_gates]
        yaw_gate = self.gate_yaw[self.target_gates%self.num_gates]

        # Rewards
        d2g_old = np.linalg.norm(pos_old - pos_gate, axis=1)
        d2g_new = np.linalg.norm(pos_new - pos_gate, axis=1)
        rat_penalty = 0.001*np.linalg.norm(new_states[:,9:12], axis=1)
        action_penalty_delta = 0.001*np.linalg.norm((self.actions-self.prev_actions), axis=1)

        prog_rewards = d2g_old - d2g_new
        rewards = prog_rewards - rat_penalty

        # Gate passing/collision
        normal = np.array([np.cos(yaw_gate), np.sin(yaw_gate)]).T
        # dot product of normal and position vector over axis 1
        pos_old_projected = (pos_old[:,0]-pos_gate[:,0])*normal[:,0] + (pos_old[:,1]-pos_gate[:,1])*normal[:,1]
        pos_new_projected = (pos_new[:,0]-pos_gate[:,0])*normal[:,0] + (pos_new[:,1]-pos_gate[:,1])*normal[:,1]
        passed_gate_plane = (pos_old_projected < 0) & (pos_new_projected > 0)
        gate_size = 2.0
        gate_passed = passed_gate_plane & np.all(np.abs(pos_new - pos_gate)<gate_size/2, axis=1)
        
        # Check out of bounds
        x_bounds_broken = np.logical_or(new_states[:,0] < self.x_bounds[0], new_states[:,0] > self.x_bounds[1])
        y_bounds_broken = np.logical_or(new_states[:,1] < self.y_bounds[0], new_states[:,1] > self.y_bounds[1])
        z_bounds_broken = np.logical_or(new_states[:,2] < self.z_bounds[0], new_states[:,2] > self.z_bounds[1])
        out_of_bounds = x_bounds_broken | y_bounds_broken | z_bounds_broken

        rewards[out_of_bounds] = -10
        
        # Check number of steps
        max_steps_reached = self.step_counts >= self.max_steps

        # Update target gate
        self.target_gates[gate_passed] += 1
        self.target_gates[gate_passed] %= self.num_gates

        # Track number of gates passed
        self.num_gates_passed[gate_passed] += 1
        
        # Check if the episode is done
        dones = max_steps_reached | out_of_bounds
        self.dones = dones
        
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
        infos = [{}] * self.num_envs
        for i in range(self.num_envs):
            if dones[i]:
                infos[i]["terminal_observation"] = self.states[i]
            if max_steps_reached[i]:
                infos[i]["TimeLimit.truncated"] = True
            # extra info for debugging
            infos[i]["out_of_bounds"] = out_of_bounds[i]
            infos[i]["gate_passed"] = gate_passed[i]
            infos[i]["num_gates_passed"] = self.num_gates_passed
            
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