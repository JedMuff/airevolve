import torch
import stable_baselines3
import sys
import numpy as np

# Device is managed per-instance, not globally.
# Do NOT call torch.set_default_device() here — it is a global side effect
# that interferes with explicit device='cpu' passed to PPO/environments.

# Efficient vectorized version of the environment
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

# Import new simulation API
from airevolve.simulator.simulation.drone_simulator import DroneSimulator
from airevolve.simulator.simulation.drone_configuration import DroneConfiguration
from airevolve.simulator.simulation.battery_model import LiPoBatteryModel
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
                 k_quad_drag=0.05,
                 z_drag_multiplier=25.0,
                 phys_max_rate_rp=25.0,
                 phys_max_rate_yaw=10.0,
                 ):
        
        # Set device
        if device is not None:
            self.device = device
        
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
        
        # ── Action centering: compute U_hover ──────────────────────────────
        # We need the physical W_hover (rad/s) that makes total thrust = mg.
        # Use the raw k_f constant and mass directly (same formula as
        # derive_reference_params) to avoid the dynamics-frame normalisation
        # trap (k_fz_signed is k_f/m and operates on W_MAX_N=3000 scale).
        k_f = self.drone_sim.config.propellers[0]["constants"][0]
        mass = float(self.drone_sim.mass)
        F_hover_per_motor = mass * 9.81 / num_motors
        W_hover_phys = np.sqrt(F_hover_per_motor / k_f)   # rad/s, physical
        
        w_min = self.drone_sim.params["w_min"]             # 305.4 for prop3
        w_max = self.drone_sim.params["w_max"]             # 4399  for prop3
        k_poly = self.drone_sim.params["k"]                # 0.84  for prop3
        
        # Clamp: if the drone can't even hover at full throttle, cap at 1.0
        W_hover_clamped = min(W_hover_phys, w_max)
        
        # The sqrt-poly motor model: Wc = (w_max-w_min)*sqrt(k*U²+(1-k)*U)+w_min
        # Invert for U that yields W_hover:
        #   sqrt(k*U²+(1-k)*U) = (W_hover - w_min) / (w_max - w_min)
        #   k*U² + (1-k)*U = target_val
        target_val = ((W_hover_clamped - w_min) / (w_max - w_min)) ** 2
        
        if abs(k_poly) < 1e-12:
            U_hover = target_val
        else:
            a = k_poly
            b = 1.0 - k_poly
            c = -target_val
            discriminant = b**2 - 4*a*c
            U_hover = (-b + np.sqrt(max(0.0, discriminant))) / (2*a)
            
        self.U_hover = float(np.clip(U_hover, 0.01, 0.99))
        self.u_hover = 2.0 * self.U_hover - 1.0

        # Motor time constant for first-order dynamics (matches reference 5-inch sysid).
        self.motor_tau = 0.04

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
        self.state_len = 12+num_motors+4*self.gates_ahead+4*self.num_action_history
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
        self.actions = np.full((num_envs,num_motors), self.u_hover, dtype=np.float32)
        self.prev_actions = np.full((num_envs,num_motors), self.u_hover, dtype=np.float32)
        self.raw_actions = np.zeros((num_envs,num_motors), dtype=np.float32)
        self.prev_raw_actions = np.zeros((num_envs,num_motors), dtype=np.float32)
        self.dones = np.zeros(num_envs, dtype=bool)
        self.final_gate_passed = np.zeros(num_envs, dtype=bool)

        self.update_states = self.update_states_gate
        
        self.pause = False
        
        self.num_motors = num_motors

        # ── Battery physics (realistic power ceiling for all training) ─────
        # One LiPoBatteryModel per parallel env.  strict_voltage_kill=False so
        # battery depletion never *terminates* the episode — it only removes
        # thrust (BMS cutoff → I=0 → dynamic_w_max→0 → drone falls → OOB).
        self.strict_voltage_kill = False
        self.bat_soc = np.ones(num_envs, dtype=np.float64)
        self.bat_capacity_ah = np.full(num_envs, LiPoBatteryModel.CAPACITY_AH, dtype=np.float64)
        self.bat_voltage = np.full(num_envs, LiPoBatteryModel.VOLTAGE_FULL, dtype=np.float64)
        self.bat_current = np.zeros(num_envs, dtype=np.float64)
        self.bat_power = np.zeros(num_envs, dtype=np.float64)
        self.bat_energy_j = np.zeros(num_envs, dtype=np.float64)
        self.bat_is_depleted = np.zeros(num_envs, dtype=bool)
        # Normalised RPM convention for the battery ECM (actions [-1,1] → [0,1])
        self._max_rpm: float = 1.0
        # Nominal w_max used to scale voltage→w_max
        self._base_w_max: float = float(self.drone_sim.params["w_max"])
        # Per-env dynamic RPM limit (starts at nominal; updated each step)
        self.dynamic_w_max: np.ndarray = np.full(num_envs, self._base_w_max, dtype=np.float64)

        # Gyroscopic coupling coefficients from the drone's inertia tensor.
        # The lambdified dynamics use d_p = Mx (torque/Ixx) without the ω×(Iω) term.
        # For the full Euler rigid-body equation I·dω/dt = τ − ω×(Iω) with
        # diagonal inertia, the missing per-step correction is:
        #   Δp = (Iyy - Izz)/Ixx · q · r · dt
        #   Δq = (Izz - Ixx)/Iyy · p · r · dt
        #   Δr = (Ixx - Iyy)/Izz · p · q · dt
        Ixx_f = float(self.drone_sim.inertia[0, 0])
        Iyy_f = float(self.drone_sim.inertia[1, 1])
        Izz_f = float(self.drone_sim.inertia[2, 2])
        self._gyro_coeff_p = (Iyy_f - Izz_f) / Ixx_f
        self._gyro_coeff_q = (Izz_f - Ixx_f) / Iyy_f
        self._gyro_coeff_r = (Ixx_f - Iyy_f) / Izz_f
        # Quadratic aerodynamic drag coefficient (m⁻¹); tunable at construction.
        # Adds F_drag_quad = -k_quad * v_body * |v_body| on all 3 body axes.
        # Z-axis drag is further scaled by z_drag_multiplier to simulate the
        # parachute-like penalty of exposing the drone's flat top/bottom face
        # to the airflow (e.g. during inverted or knife-edge flight).
        self._k_quad_drag = float(k_quad_drag)
        self._z_drag_multiplier = float(z_drag_multiplier)

        # ── Morphology-aware per-axis angular rate caps ────────────────────
        # Tuning knobs (all exposed as __init__ kwargs):
        #   k_quad_drag       — quadratic translational drag coeff (m⁻¹)
        #   phys_max_rate_rp  — hard ceiling for roll/pitch (rad/s)
        #   phys_max_rate_yaw — hard ceiling for yaw (rad/s); yaw torque is small
        # Plus the fixed constant below:
        #   _tau_settle = 0.04 s — motor settling time constant; matches the
        #                          supervisor's previous paper (don't change for
        #                          scientific consistency across baselines).
        #
        # Formula: max_rate = min(alpha_max · tau_settle, phys_ceiling)
        #   alpha_max = sum(|k_axis_signed[i]|) · w_max²
        # k_p/q/r_signed are already (torque / I_axis), i.e. angular accel per W².
        # Relative scaling is preserved: a heavier / shorter-armed drone has
        # smaller k values → lower cap.  IMPORTANT: alpha_max for roll/pitch on
        # small prop3 drones is ~3000-4000 rad/s², so even with tau=0.04 s the
        # raw cap (~130 rad/s) is unrealistic; the phys ceiling clamps it.
        # To target ~24 gates/12s, tune phys_max_rate_rp downward (try 26 rad/s
        # ≈ 1500 deg/s for realistic racing) and/or increase k_quad_drag.
        _tau_settle = 0.04
        self._phys_max_rp  = float(phys_max_rate_rp)
        self._phys_max_yaw = float(phys_max_rate_yaw)
        _w_max = self.drone_sim.params["w_max"]
        _k_p_s = self.drone_sim.params["k_p_signed"]
        _k_q_s = self.drone_sim.params["k_q_signed"]
        _k_r_s = self.drone_sim.params["k_r_signed"]
        self.max_rate_roll  = min(sum(abs(k) for k in _k_p_s) * _w_max**2 * _tau_settle, self._phys_max_rp)
        self.max_rate_pitch = min(sum(abs(k) for k in _k_q_s) * _w_max**2 * _tau_settle, self._phys_max_rp)
        self.max_rate_yaw   = min(sum(abs(k) for k in _k_r_s) * _w_max**2 * _tau_settle, self._phys_max_yaw)
        print(
            f"[DroneGateEnv] tuning: k_quad_drag={self._k_quad_drag:.4f}  "
            f"z_drag_multiplier={self._z_drag_multiplier:.2f}  "
            f"phys_max_rp={self._phys_max_rp:.1f} rad/s  "
            f"phys_max_yaw={self._phys_max_yaw:.1f} rad/s  "
            f"tau_settle={_tau_settle:.3f}s",
            flush=True,
        )
        print(
            f"[DroneGateEnv] morphology caps: "
            f"roll={np.degrees(self.max_rate_roll):.0f} "
            f"pitch={np.degrees(self.max_rate_pitch):.0f} "
            f"yaw={np.degrees(self.max_rate_yaw):.0f} deg/s",
            flush=True,
        )

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
                "propsize": 3  # Default prop size
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
        new_states[:,12:12+self.num_motors] = self.world_states[:,12:12+self.num_motors]

        # Update future gates relative to current gate
        for i in range(self.gates_ahead):
            indices = (self.target_gates+i+1)
            # loop when out of bounds
            indices = indices % self.num_gates
            valid = indices < self.num_gates
            new_states[valid,12+self.num_motors+4*i:12+self.num_motors+4*i+3] = self.gate_pos_rel[indices[valid]]
            new_states[valid,12+self.num_motors+4*i+3] = self.gate_yaw_rel[indices[valid]]

        # update action history
        self.action_hist = np.roll(self.action_hist, 1, axis=1)
        self.action_hist[:,0] = self.actions

        for i in range(self.num_action_history):
            new_states[:,12+self.num_motors+4*self.gates_ahead+4*i:12+self.num_motors+4*self.gates_ahead+4*i+4] = self.action_hist[:,(i+1)*self.history_step_size-1]
        
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

            # Motor speeds w_i are in [-1, 1] (reference-form normalization;
            # see DroneSimulator state convention). Was [0, 1] under the old
            # dynamics — see Phase 2.3 in RUNTIME_DYNAMICS_MIGRATION.md.
            w0 = np.random.uniform(-1, 1, size=(num_reset,self.num_motors))

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

            w0 = np.zeros((num_reset,self.num_motors))

        w0 = np.hsplit(w0, self.num_motors)

        state_vars = [x0, y0, z0, vx0, vy0, vz0, phi0, theta0, psi0, p0, q0, r0]
        state_vars = [var.reshape(num_reset, 1) for var in state_vars]

        self.world_states[dones] = np.concatenate(state_vars + list(w0), axis=1)
        self.step_counts[dones] = np.zeros(num_reset)
        # Clear the action-filter memory for envs that just reset, so the
        # next action isn't smoothed against a stale pre-reset action.
        self.filtered_actions[dones] = 0.0

        # Reset batteries for done envs (full charge, no SoC randomisation).
        if np.any(dones):
            self.bat_soc[dones] = 1.0
            self.bat_capacity_ah[dones] = LiPoBatteryModel.CAPACITY_AH
            self.bat_voltage[dones] = LiPoBatteryModel.VOLTAGE_FULL
            self.bat_current[dones] = 0.0
            self.bat_power[dones] = 0.0
            self.bat_energy_j[dones] = 0.0
            self.bat_is_depleted[dones] = False
        # Restore nominal dynamic_w_max for freshly-reset envs.
        self.dynamic_w_max[dones] = self._base_w_max
        
        # update states
        self.update_states()
        return self.states
    
    def reset(self):
        return self.reset_(np.ones(self.num_envs, dtype=bool))

    def step_async(self, actions):
        self.prev_actions = self.actions.copy()
        if hasattr(self, 'raw_actions'):
            self.prev_raw_actions = self.raw_actions.copy()
        else:
            self.prev_raw_actions = np.zeros_like(actions, dtype=np.float32)
            
        # Actions from RL are in [-1, 1]. Center them using a smooth Power Curve (Throttle Expo).
        # We want RL=0 (which maps to x=0.5) to produce U_hover.
        # U = x^gamma  =>  0.5^gamma = U_hover  =>  gamma = ln(U_hover) / ln(0.5)
        # This prevents the severe asymmetric heave coupling caused by piecewise linear mapping!
        
        clipped_actions = np.clip(actions, -1.0, 1.0)
        x = (clipped_actions + 1.0) / 2.0
        
        safe_U_hover = np.clip(self.U_hover, 1e-4, 0.9999)
        gamma = np.log(safe_U_hover) / np.log(0.5)
        
        U = np.power(x, gamma)
        centered_actions = 2.0 * U - 1.0
        
        self.actions = centered_actions
        self.raw_actions = actions.copy()
    
    def step_wait(self):
        # Reference-form dynamics: dynamics_func takes the full 12+N state
        # and the action directly. Motor model (sqrt-poly mapping U → Wc,
        # then first-order lag) is baked into the symbolic equations.
        # Action is in [-1, 1]; motor state w_i is in [-1, 1].
        #
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

        # ── Betaflight-style current limiter & Battery Step ───────────────
        safe_actions = action_for_dynamics.copy()
        requested_rpms = (safe_actions + 1.0) / 2.0

        # 1. Compute theoretical current (vectorized)
        rpm_ratio = np.clip(requested_rpms / self._max_rpm, 0.0, 1.0)
        motor_currents = (
            LiPoBatteryModel.MOTOR_IDLE_CURRENT
            + (LiPoBatteryModel.MOTOR_MAX_CURRENT - LiPoBatteryModel.MOTOR_IDLE_CURRENT) * rpm_ratio ** 2
        )
        i_theoretical = LiPoBatteryModel.FC_BASELINE_CURRENT + np.sum(motor_currents, axis=1)

        # 2. Voltage-aware current limit: prevent terminal voltage from dropping
        #    below the depletion threshold.
        #      V_terminal = V_ideal - I * R_int >= V_DEPLETION_THRESHOLD
        #      => I <= (V_ideal - V_DEPLETION_THRESHOLD) / R_int
        v_ideal = 13.0 + 3.8 * np.sqrt(np.clip(self.bat_soc, 0.0, 1.0))
        i_max_voltage = (
            (v_ideal - LiPoBatteryModel.VOLTAGE_DEPLETION_THRESHOLD)
            / LiPoBatteryModel.INTERNAL_RESISTANCE
        )
        effective_i_max = np.minimum(
            float(LiPoBatteryModel.BATTERY_MAX_CURRENT),
            np.maximum(i_max_voltage, 0.0),
        )

        # 3. Current limiter: scale all motor throttles uniformly so that total
        #    current does not exceed effective_i_max.  Accounts for the constant
        #    idle-current component so the scaling is exact:
        #      I_total = I_const + (I_MAX_M - I_IDLE_M) * Σ throttle_i²
        #    Scaling each throttle_i by α gives:
        #      I_new = I_const + α² * (I_total - I_const)  = effective_i_max
        #      => α = sqrt((effective_i_max - I_const) / (I_total - I_const))
        _i_const = (
            LiPoBatteryModel.FC_BASELINE_CURRENT
            + self.num_motors * LiPoBatteryModel.MOTOR_IDLE_CURRENT
        )
        exceeds = i_theoretical > effective_i_max
        if np.any(exceeds):
            i_var = i_theoretical[exceeds] - _i_const
            i_var_allowed = np.maximum(effective_i_max[exceeds] - _i_const, 0.0)
            alpha = np.sqrt(
                np.where(i_var > 0, np.clip(i_var_allowed / i_var, 0.0, 1.0), 0.0)
            )
            safe_actions[exceeds] = (requested_rpms[exceeds] * alpha[:, None] * 2.0) - 1.0

            # Recalculate physical values for the battery step
            requested_rpms[exceeds] = (safe_actions[exceeds] + 1.0) / 2.0
            rpm_ratio[exceeds] = np.clip(requested_rpms[exceeds] / self._max_rpm, 0.0, 1.0)
            motor_currents[exceeds] = (
                LiPoBatteryModel.MOTOR_IDLE_CURRENT
                + (LiPoBatteryModel.MOTOR_MAX_CURRENT - LiPoBatteryModel.MOTOR_IDLE_CURRENT) * rpm_ratio[exceeds] ** 2
            )
            i_theoretical[exceeds] = LiPoBatteryModel.FC_BASELINE_CURRENT + np.sum(motor_currents[exceeds], axis=1)

        action_for_dynamics = safe_actions

        # 3. Vectorized Battery Step
        active = ~self.bat_is_depleted
        if np.any(active):
            current = i_theoretical[active]
            
            # Capacity drain
            delta_ah = current * (float(self.dt) / 3600.0)
            self.bat_capacity_ah[active] = np.maximum(0.0, self.bat_capacity_ah[active] - delta_ah)
            self.bat_soc[active] = np.clip(self.bat_capacity_ah[active] / LiPoBatteryModel.CAPACITY_AH, 0.0, 1.0)
            
            # ECM voltage: V_ideal = 13.0 + 3.8 * √SoC
            v_ideal = 13.0 + 3.8 * np.sqrt(self.bat_soc[active])
            v_terminal = v_ideal - (current * LiPoBatteryModel.INTERNAL_RESISTANCE)
            self.bat_voltage[active] = v_terminal
            
            # Power & Energy
            power_w = v_terminal * current
            self.bat_power[active] = power_w
            self.bat_current[active] = current
            self.bat_energy_j[active] += power_w * float(self.dt)
            
            # Depletion check
            soc_depleted = self.bat_soc[active] < LiPoBatteryModel.SOC_DEPLETION_THRESHOLD
            if self.strict_voltage_kill:
                v_depleted = v_terminal < LiPoBatteryModel.VOLTAGE_DEPLETION_THRESHOLD
            else:
                v_depleted = np.zeros_like(soc_depleted, dtype=bool)
                
            depleted_now = soc_depleted | v_depleted
            
            # Update global depleted mask
            new_depleted = np.zeros_like(self.bat_is_depleted)
            new_depleted[active] = depleted_now
            self.bat_is_depleted |= new_depleted
            
            # If depleted now, zero out current/power
            just_depleted = active & new_depleted
            self.bat_current[just_depleted] = 0.0
            self.bat_power[just_depleted] = 0.0

        # 4. Voltage sag -> dynamic w_max
        self.dynamic_w_max = self.bat_voltage * (self._base_w_max / 14.8)

        full_state = self.world_states
        w_max_array = self.dynamic_w_max
            
        full_state_dot = self.drone_sim.dynamics_func(
            full_state.T, action_for_dynamics.T, w_max_array
        ).T  # (num_envs, 12+N)
        new_states = (full_state + self.dt * full_state_dot).astype(np.float32)
        dt_f = float(self.dt)

        # ── Quadratic aerodynamic drag ────────────────────────────────────
        # The lambdified dynamics already apply linear drag: -k_x * v_b * sum_W.
        # This block adds the quadratic term: -k_quad * v_b * |v_b| on all three
        # body-frame axes.  Equivalent to a pre-integration force in Euler scheme.
        phi_n   = new_states[:, 6].astype(np.float64)
        theta_n = new_states[:, 7].astype(np.float64)
        psi_n   = new_states[:, 8].astype(np.float64)
        vw_x    = new_states[:, 3].astype(np.float64)
        vw_y    = new_states[:, 4].astype(np.float64)
        vw_z    = new_states[:, 5].astype(np.float64)
        cphi = np.cos(phi_n);  sphi = np.sin(phi_n)
        cth  = np.cos(theta_n); sth  = np.sin(theta_n)
        cpsi = np.cos(psi_n);  spsi = np.sin(psi_n)
        # World → body  (R^T @ v_world)
        vbx_q = cpsi*cth*vw_x + spsi*cth*vw_y - sth*vw_z
        vby_q = (cpsi*sphi*sth - spsi*cphi)*vw_x + (spsi*sphi*sth + cpsi*cphi)*vw_y + sphi*cth*vw_z
        vbz_q = (cpsi*cphi*sth + spsi*sphi)*vw_x + (spsi*cphi*sth - cpsi*sphi)*vw_y + cphi*cth*vw_z
        kq = self._k_quad_drag
        Fq_bx = -kq * vbx_q * np.abs(vbx_q)
        Fq_by = -kq * vby_q * np.abs(vby_q)
        # Anisotropic Z drag: the flat top/bottom of the drone acts like a
        # parachute when exposed to airflow (inverted / knife-edge flight).
        Fq_bz = -(kq * self._z_drag_multiplier) * vbz_q * np.abs(vbz_q)
        # Body → world  (R @ F_body)
        dv_wx = cpsi*cth*Fq_bx + (cpsi*sphi*sth - spsi*cphi)*Fq_by + (cpsi*cphi*sth + spsi*sphi)*Fq_bz
        dv_wy = spsi*cth*Fq_bx + (spsi*sphi*sth + cpsi*cphi)*Fq_by + (spsi*cphi*sth - cpsi*sphi)*Fq_bz
        dv_wz = -sth*Fq_bx + sphi*cth*Fq_by + cphi*cth*Fq_bz
        new_states[:, 3] += (dv_wx * dt_f).astype(np.float32)
        new_states[:, 4] += (dv_wy * dt_f).astype(np.float32)
        new_states[:, 5] += (dv_wz * dt_f).astype(np.float32)

        # Detect numerical divergence (NaN, Inf, or excessively large finite values)
        diverged = np.any(~np.isfinite(new_states) | (np.abs(new_states) > 1e6), axis=1)
        if np.any(diverged):
            new_states[diverged] = self.world_states[diverged]

        # ── Gyroscopic coupling correction ────────────────────────────────
        # Completes the Euler rigid-body equation I·dω/dt = τ − ω×(Iω).
        # Applied before the rate cap so asymmetric drones must expend torque
        # fighting precession before the cap has any effect.
        p0 = self.world_states[:, 9]
        q0 = self.world_states[:, 10]
        r0 = self.world_states[:, 11]
        new_states[:, 9]  += (self._gyro_coeff_p * q0 * r0 * dt_f).astype(np.float32)
        new_states[:, 10] += (self._gyro_coeff_q * p0 * r0 * dt_f).astype(np.float32)
        new_states[:, 11] += (self._gyro_coeff_r * p0 * q0 * dt_f).astype(np.float32)

        # ── Morphology-aware angular rate cap ────────────────────────────
        # Per-axis limits derived from drone's own k_signed parameters in __init__:
        #   max_rate = sum(|k_axis_signed|) * w_max² * tau_settle
        # Heavier / shorter-armed drones get a lower cap automatically.
        new_states[:, 9]  = np.clip(new_states[:, 9],  -self.max_rate_roll,  self.max_rate_roll)
        new_states[:, 10] = np.clip(new_states[:, 10], -self.max_rate_pitch, self.max_rate_pitch)
        new_states[:, 11] = np.clip(new_states[:, 11], -self.max_rate_yaw,   self.max_rate_yaw)

        self.step_counts += 1

        pos_old = self.world_states[:,0:3]
        pos_new = new_states[:,0:3]
        pos_gate = self.gate_pos[self.target_gates%self.num_gates]
        yaw_gate = self.gate_yaw[self.target_gates%self.num_gates]

        # Rewards
        d2g_old = np.linalg.norm(pos_old - pos_gate, axis=1)
        d2g_new = np.linalg.norm(pos_new - pos_gate, axis=1)
        rat_penalty = 0.003*np.linalg.norm(new_states[:,9:12], axis=1)
        action_penalty_delta = 0.003*np.linalg.norm((self.raw_actions-self.prev_raw_actions), axis=1)

        prog_rewards = d2g_old - d2g_new
        rewards = prog_rewards - rat_penalty - action_penalty_delta

        # Gate passing/collision
        normal = np.array([np.cos(yaw_gate), np.sin(yaw_gate)]).T
        # dot product of normal and position vector over axis 1
        pos_old_projected = (pos_old[:,0]-pos_gate[:,0])*normal[:,0] + (pos_old[:,1]-pos_gate[:,1])*normal[:,1]
        pos_new_projected = (pos_new[:,0]-pos_gate[:,0])*normal[:,0] + (pos_new[:,1]-pos_gate[:,1])*normal[:,1]
        passed_gate_plane = (pos_old_projected < 0) & (pos_new_projected > 0)
        gate_size = 1.5
        gate_passed = passed_gate_plane & np.all(np.abs(pos_new - pos_gate)<gate_size/2, axis=1)

        # Reward for passing ANY gate in the correct sequence.
        # This gives a dense spike to offset the sudden drop in `prog_rewards` when the target switches.
        rewards[gate_passed] += 5.0
        
        # Additional +10 only on the final gate of the lap.
        final_gate_passed = gate_passed & (self.target_gates == self.num_gates - 1)
        rewards[final_gate_passed] += 10.0

        # Check out of bounds
        x_bounds_broken = np.logical_or(new_states[:,0] < self.x_bounds[0], new_states[:,0] > self.x_bounds[1])
        y_bounds_broken = np.logical_or(new_states[:,1] < self.y_bounds[0], new_states[:,1] > self.y_bounds[1])
        z_bounds_broken = np.logical_or(new_states[:,2] < self.z_bounds[0], new_states[:,2] > self.z_bounds[1])
        out_of_bounds = x_bounds_broken | y_bounds_broken | z_bounds_broken

        rewards[out_of_bounds] = -10
        rewards[diverged] = -10

        # Check number of steps
        max_steps_reached = self.step_counts >= self.max_steps

        # Update target gate
        self.target_gates[gate_passed] += 1
        self.target_gates[gate_passed] %= self.num_gates

        # Track number of gates passed
        self.num_gates_passed[gate_passed] += 1

        # Check if the episode is done
        dones = max_steps_reached | out_of_bounds | diverged
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
            infos[i]["out_of_bounds"] = out_of_bounds[i]
            infos[i]["gate_passed"] = gate_passed[i]
            infos[i]["num_gates_passed"] = gates_passed_before_reset
            infos[i]["distance_to_gate"] = float(d2g_new[i])
            infos[i]["target_gate_idx"] = int(self.target_gates[i] % self.num_gates)
            
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