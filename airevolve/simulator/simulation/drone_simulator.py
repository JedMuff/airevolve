"""
Drone Simulator with Propeller-Based Configuration

This module provides the core drone simulation framework using propeller configurations
for automatic computation of physical properties and allocation matrices.
"""

import numpy as np
from sympy import *
from .drone_configuration import DroneConfiguration
from .propeller_data import create_standard_propeller_config

class DroneSimulator:
    """
    Core drone simulation framework.
    Uses propeller configurations for automatic computation of mass, inertia, and allocation matrices.
    """
    
    def __init__(self, propellers=None, dt=0.01, gravity=9.81):
        """
        Initialize drone simulator from propeller configuration.
        
        Args:
            propellers (list): List of propeller dictionaries, each containing:
                - "loc": [x, y, z] position in body frame (meters)
                - "dir": [x, y, z, rotation] thrust direction and spin direction  
                - "propsize": propeller size in inches (4-8)
            dt (float): Integration time step
            gravity (float): Gravitational acceleration
            
        Example:
            # Standard quadrotor
            propellers = [
                {"loc": [0.11, 0.11, 0], "dir": [0, 0, -1, "ccw"], "propsize": 5},
                {"loc": [-0.11, 0.11, 0], "dir": [0, 0, -1, "cw"], "propsize": 5},
                {"loc": [-0.11, -0.11, 0], "dir": [0, 0, -1, "ccw"], "propsize": 5},
                {"loc": [0.11, -0.11, 0], "dir": [0, 0, -1, "cw"], "propsize": 5}
            ]
            drone = DroneSimulator(propellers=propellers)
        """
        
        # Use default quadrotor if no propellers specified
        if propellers is None:
            propellers = create_standard_propeller_config("quad", arm_length=0.11, prop_size=5)
        
        # Create drone configuration and compute physical properties
        self.config = DroneConfiguration(propellers)
        
        # Extract computed properties
        self.Bf, self.Bm = self.config.get_allocation_matrices()
        self.num_motors = self.config.num_motors
        self.mass = self.config.mass
        self.inertia = self.config.inertia_matrix
        self.center_of_gravity = self.config.cg
        
        # Simulation parameters
        self.dt = dt
        self.g = gravity
        
        # Initialize symbolic dynamics
        self._setup_dynamics()
        
        # State: [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]
        self.state = np.zeros(12, dtype=np.float64)
        self.motor_commands = np.zeros(self.num_motors, dtype=np.float64)
        
        # History for plotting/analysis
        self.time_history = []
        self.state_history = []
        self.control_history = []
    
    @classmethod
    def create_standard_drone(cls, drone_type="quad", arm_length=0.11, prop_size=5, **kwargs):
        """
        Create standard drone configuration.
        
        Args:
            drone_type (str): Type of drone ('quad', 'hex', 'tri', 'octo')
            arm_length (float): Length of drone arms in meters
            prop_size (int): Propeller size in inches (4-8)
            **kwargs: Additional arguments for DroneSimulator
            
        Returns:
            DroneSimulator: Configured drone simulator
        """
        propellers = create_standard_propeller_config(drone_type, arm_length, prop_size)
        return cls(propellers=propellers, **kwargs)
    
    def _setup_dynamics(self):
        """Setup symbolic equations of motion using SymPy."""
        
        # State variables
        state_vars = symbols('x y z v_x v_y v_z phi theta psi p q r')
        x, y, z, vx, vy, vz, phi, theta, psi, p, q, r = state_vars
        
        # Control inputs (motor commands in range [0,1])
        control_symbols = [symbols(f'U_{i}') for i in range(1, self.num_motors + 1)]

        # Motor commands (already in [0,1] range)
        U = Matrix(control_symbols)
        
        # Rotation matrices
        Rx = Matrix([[1, 0, 0], 
                    [0, cos(phi), -sin(phi)], 
                    [0, sin(phi), cos(phi)]])
        Ry = Matrix([[cos(theta), 0, sin(theta)], 
                    [0, 1, 0], 
                    [-sin(theta), 0, cos(theta)]])
        Rz = Matrix([[cos(psi), -sin(psi), 0], 
                    [sin(psi), cos(psi), 0], 
                    [0, 0, 1]])
        R = Rz * Ry * Rx
        
        # Convert allocation matrices to SymPy
        Bf_sym = Matrix(self.Bf)
        Bm_sym = Matrix(self.Bm)
        
        # Forces and moments in body frame
        F_body = Bf_sym @ U  # [Fx, Fy, Fz] in body frame
        M_body = Bm_sym @ U  # [Mx, My, Mz] in body frame
        
        # Translational dynamics (Newton's laws)
        d_x = vx
        d_y = vy  
        d_z = vz
        
        # Transform body forces to world frame and add gravity
        F_world = R @ F_body + Matrix([0, 0, self.g * self.mass])
        d_vx = F_world[0] / self.mass
        d_vy = F_world[1] / self.mass
        d_vz = F_world[2] / self.mass
        
        # Rotational dynamics (Euler's equations)
        d_phi = p + q * sin(phi) * tan(theta) + r * cos(phi) * tan(theta)
        d_theta = q * cos(phi) - r * sin(phi)
        d_psi = q * sin(phi) / cos(theta) + r * cos(phi) / cos(theta)
        
        # Rotational dynamics using inertia matrix
        # M = I * omega_dot + omega x (I * omega)
        # Simplified version: omega_dot = I^(-1) * M_body
        I_inv = Matrix(np.linalg.inv(self.inertia))
        omega = Matrix([p, q, r])
        I_omega = Matrix(self.inertia) @ omega
        
        # Gyroscopic term: omega x (I * omega)
        gyroscopic = Matrix([
            q * I_omega[2] - r * I_omega[1],
            r * I_omega[0] - p * I_omega[2], 
            p * I_omega[1] - q * I_omega[0]
        ])
        
        omega_dot = I_inv @ (M_body - gyroscopic)
        d_p = omega_dot[0]
        d_q = omega_dot[1]
        d_r = omega_dot[2]
        
        # Complete state derivative
        state_dot = [d_x, d_y, d_z, d_vx, d_vy, d_vz, d_phi, d_theta, d_psi, d_p, d_q, d_r]
        
        # Create numerical function
        self.dynamics_func = lambdify((Array(state_vars), Array(control_symbols)), 
                                    Array(state_dot), 'numpy')

    def get_configuration_info(self):
        """
        Get comprehensive drone configuration information.
        
        Returns:
            dict: Complete configuration including physical properties and capabilities
        """
        info = self.config.get_physical_properties()
        info.update(self.config.get_motor_configuration_info())
        info['propeller_configuration'] = self.config.propellers
        return info
    
    def get_propeller_info(self):
        """
        Get propeller configuration details.
        
        Returns:
            list: List of propeller specifications
        """
        return [prop.copy() for prop in self.config.propellers]
    
    def set_state(self, position=None, velocity=None, attitude=None, angular_velocity=None):
        """
        Set drone state.
        
        Args:
            position: [x, y, z] in world frame
            velocity: [vx, vy, vz] in world frame  
            attitude: [phi, theta, psi] (roll, pitch, yaw) in radians
            angular_velocity: [p, q, r] in body frame
        """
        if position is not None:
            self.state[0:3] = position
        if velocity is not None:
            self.state[3:6] = velocity
        if attitude is not None:
            self.state[6:9] = attitude
        if angular_velocity is not None:
            self.state[9:12] = angular_velocity
    
    def get_state(self):
        """Get current state as dictionary."""
        return {
            'position': self.state[0:3].copy(),
            'velocity': self.state[3:6].copy(), 
            'attitude': self.state[6:9].copy(),
            'angular_velocity': self.state[9:12].copy(),
            'time': len(self.time_history) * self.dt
        }
    
    def set_motor_commands(self, commands):
        """
        Set motor commands.
        
        Args:
            commands: Array of motor commands in range [0, 1]
        """
        self.motor_commands = np.clip(commands, 0, 1)
    
    def step(self, motor_commands=None):
        """
        Advance simulation by one time step using RK4 integration.
        
        Args:
            motor_commands: Optional motor commands for this step
        """
        if motor_commands is not None:
            self.set_motor_commands(motor_commands)
        
        # RK4 integration for better numerical stability
        k1 = self.dt * self.dynamics_func(self.state, self.motor_commands)
        k2 = self.dt * self.dynamics_func(self.state + 0.5 * k1, self.motor_commands)
        k3 = self.dt * self.dynamics_func(self.state + 0.5 * k2, self.motor_commands)
        k4 = self.dt * self.dynamics_func(self.state + k3, self.motor_commands)
        
        # Update state using RK4 formula
        self.state = self.state + (k1 + 2*k2 + 2*k3 + k4) / 6.0
        
        # Store history
        self.time_history.append(len(self.time_history) * self.dt)
        self.state_history.append(self.state.copy())
        self.control_history.append(self.motor_commands.copy())
        
        return self.get_state()
    
    def simulate(self, time_span, control_function=None):
        """
        Run simulation for specified time span.
        
        Args:
            time_span: Total simulation time
            control_function: Function that takes (time, state) and returns motor commands
        """
        num_steps = int(time_span / self.dt)
        
        for i in range(num_steps):
            current_time = i * self.dt
            current_state = self.get_state()
            
            if control_function is not None:
                commands = control_function(current_time, current_state)
                self.step(commands)
            else:
                self.step()
    
    def reset(self):
        """Reset simulation to initial conditions."""
        self.state = np.zeros(12)
        self.motor_commands = np.zeros(self.num_motors)
        self.time_history = []
        self.state_history = []
        self.control_history = []


# Factory functions for easy drone creation
def create_quadrotor(arm_length=0.11, prop_size=5, **kwargs):
    """Create standard quadrotor configuration."""
    return DroneSimulator.create_standard_drone("quad", arm_length, prop_size, **kwargs)

def create_hexarotor(arm_length=0.10, prop_size=4, **kwargs):
    """Create standard hexarotor configuration.""" 
    return DroneSimulator.create_standard_drone("hex", arm_length, prop_size, **kwargs)

def create_tricopter(arm_length=0.12, prop_size=6, **kwargs):
    """Create standard tricopter configuration."""
    return DroneSimulator.create_standard_drone("tri", arm_length, prop_size, **kwargs)

def create_octorotor(arm_length=0.09, prop_size=4, **kwargs):
    """Create standard octorotor configuration."""
    return DroneSimulator.create_standard_drone("octo", arm_length, prop_size, **kwargs)