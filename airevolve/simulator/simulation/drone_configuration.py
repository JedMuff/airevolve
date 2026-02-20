"""
Drone Configuration Class

This module provides automatic computation of drone physical properties and allocation
matrices from propeller configurations. Integrates algorithms from drone-hover package
for realistic mass, center of gravity, inertia, and control allocation calculation.
"""

import numpy as np
from numpy.linalg import norm, inv
from .propeller_data import (
    get_propeller_specs, validate_propeller_config,
    GRAVITY, CONTROLLER_MASS, BATTERY_MASS, BEAM_DENSITY
)

class DroneConfiguration:
    """
    Automatically compute drone physical properties from propeller configuration.
    
    This class takes a list of propeller specifications and computes:
    - Total mass including controller, motors, and structure
    - Center of gravity location
    - Inertia matrix (Ix, Iy, Iz, Ixy, Ixz, Iyz)
    - Force allocation matrix (Bf) mapping motor commands to body forces
    - Moment allocation matrix (Bm) mapping motor commands to body moments
    """
    
    def __init__(self, propellers, mountpoints=None):
        """
        Initialize drone configuration from propeller specifications.

        Args:
            propellers (list): List of propeller dictionaries, each containing:
                - "loc": [x, y, z] position in body frame (meters)
                - "dir": [x, y, z, rotation] thrust direction and spin direction
                - "propsize": propeller size in inches (4-8)
            mountpoints (list, optional): List of np.array mounting points for each propeller.
                If None, defaults to origin [0,0,0] for all propellers (backward compatible).

        Example:
            propellers = [
                {"loc": [0.11, 0.11, 0], "dir": [0, 0, -1, "ccw"], "propsize": 5},
                {"loc": [-0.11, 0.11, 0], "dir": [0, 0, -1, "cw"], "propsize": 5},
                # ... more propellers
            ]

            # Optional: specify mounting points
            mountpoints = [
                np.array([0.03, 0.03, 0]),   # Mounting point for propeller 1
                np.array([-0.03, 0.03, 0]),  # Mounting point for propeller 2
                # ...
            ]
        """
        # Validate input format
        validate_propeller_config(propellers)

        self.propellers = propellers
        self.num_motors = len(propellers)

        # Set up mounting points
        self.mountpoints = mountpoints
        if self.mountpoints is None:
            self.mountpoints = [np.array([0.0, 0.0, 0.0]) for _ in range(self.num_motors)]

        if len(self.mountpoints) != self.num_motors:
            raise ValueError(f"Number of mounting points ({len(self.mountpoints)}) "
                           f"must equal number of propellers ({self.num_motors})")

        # Add propeller specifications to each propeller
        self._add_propeller_specs()

        # Compute physical properties
        self._compute_mass_and_cg()
        self._compute_inertia()
        self._compute_allocation_matrices()
    
    def _add_propeller_specs(self):
        """Add force/moment constants and specifications to each propeller."""
        for prop in self.propellers:
            specs = get_propeller_specs(prop["propsize"])
            prop["constants"] = specs["constants"]  # [k_f, k_m]
            prop["wmax"] = specs["wmax"]
            prop["mass"] = specs["mass"]
    
    def _compute_mass_and_cg(self):
        """Compute total mass and center of gravity location."""
        # Start with controller and battery mass
        self.mass = CONTROLLER_MASS + BATTERY_MASS

        # Add propeller and beam masses
        for prop, mountpoint in zip(self.propellers, self.mountpoints):
            prop_mass = prop["mass"]
            prop_loc = np.array(prop["loc"])
            beam_length = norm(prop_loc - mountpoint)
            beam_mass = BEAM_DENSITY * beam_length

            self.mass += prop_mass + beam_mass

        # Compute center of gravity
        self.cg = np.zeros(3)
        for prop, mountpoint in zip(self.propellers, self.mountpoints):
            prop_mass = prop["mass"]
            prop_loc = np.array(prop["loc"])
            beam_length = norm(prop_loc - mountpoint)
            beam_mass = BEAM_DENSITY * beam_length

            # Propeller contributes at its location
            self.cg += (prop_mass / self.mass) * prop_loc

            # Beam contributes at its midpoint
            beam_midpoint = (prop_loc + mountpoint) * 0.5
            self.cg += (beam_mass / self.mass) * beam_midpoint
    
    def _compute_inertia(self):
        """Compute inertia matrix components using parallel axis theorem."""
        # Battery inertia (approximated as rectangular block)
        # Typical battery dimensions: 60mm x 16mm x 18mm
        battery_Ix = (1/12) * BATTERY_MASS * (0.016**2 + 0.018**2)
        battery_Iy = (1/12) * BATTERY_MASS * (0.060**2 + 0.018**2)
        battery_Iz = (1/12) * BATTERY_MASS * (0.060**2 + 0.016**2)

        # Controller inertia (approximated as rectangular block)
        # Speedybee F405 dimensions: ~33mm x 8mm x 8mm
        controller_Ix = (1/12) * CONTROLLER_MASS * (0.008**2 + 0.033**2)
        controller_Iy = (1/12) * CONTROLLER_MASS * (0.008**2 + 0.033**2)
        controller_Iz = (1/12) * CONTROLLER_MASS * (0.033**2 + 0.033**2)

        # Translate to center of gravity using parallel axis theorem
        self.Ix = battery_Ix + BATTERY_MASS * (self.cg[1]**2 + self.cg[2]**2)
        self.Ix += controller_Ix + CONTROLLER_MASS * (self.cg[1]**2 + self.cg[2]**2)

        self.Iy = battery_Iy + BATTERY_MASS * (self.cg[0]**2 + self.cg[2]**2)
        self.Iy += controller_Iy + CONTROLLER_MASS * (self.cg[0]**2 + self.cg[2]**2)

        self.Iz = battery_Iz + BATTERY_MASS * (self.cg[0]**2 + self.cg[1]**2)
        self.Iz += controller_Iz + CONTROLLER_MASS * (self.cg[0]**2 + self.cg[1]**2)

        # Initialize products of inertia
        self.Ixy = -BATTERY_MASS * self.cg[0] * self.cg[1] - CONTROLLER_MASS * self.cg[0] * self.cg[1]
        self.Ixz = -BATTERY_MASS * self.cg[0] * self.cg[2] - CONTROLLER_MASS * self.cg[0] * self.cg[2]
        self.Iyz = -BATTERY_MASS * self.cg[1] * self.cg[2] - CONTROLLER_MASS * self.cg[1] * self.cg[2]
        
        # Add contributions from propellers and beams (using mounting points)
        for prop, mountpoint in zip(self.propellers, self.mountpoints):
            prop_mass = prop["mass"]
            pos = np.array(prop["loc"])
            arm_length = norm(pos - mountpoint)
            arm_midpoint = (pos + mountpoint) / 2
            beam_mass = BEAM_DENSITY * arm_length

            # Propeller position relative to CG
            r_prop = pos - self.cg

            # Propeller contributions (treated as point mass)
            self.Ix += prop_mass * (r_prop[1]**2 + r_prop[2]**2)
            self.Iy += prop_mass * (r_prop[0]**2 + r_prop[2]**2)
            self.Iz += prop_mass * (r_prop[0]**2 + r_prop[1]**2)
            self.Ixy -= prop_mass * r_prop[0] * r_prop[1]
            self.Ixz -= prop_mass * r_prop[0] * r_prop[2]
            self.Iyz -= prop_mass * r_prop[1] * r_prop[2]

            # Beam contributions (rod from mounting point to propeller)
            r_beam = arm_midpoint - self.cg

            # Use cross product method for inertia (drone-hover approach)
            # For moments of inertia (Ix, Iy, Iz)
            self.Ix += (1/12) * norm(np.cross(np.array([1, 0, 0]), pos))**2 * beam_mass
            self.Ix += beam_mass * norm(np.cross(np.array([1, 0, 0]), r_beam))**2

            self.Iy += (1/12) * norm(np.cross(np.array([0, 1, 0]), pos))**2 * beam_mass
            self.Iy += beam_mass * norm(np.cross(np.array([0, 1, 0]), r_beam))**2

            self.Iz += (1/12) * norm(np.cross(np.array([0, 0, 1]), pos))**2 * beam_mass
            self.Iz += beam_mass * norm(np.cross(np.array([0, 0, 1]), r_beam))**2

            # Products of inertia
            self.Ixy -= (1/12) * pos[0] * pos[1] * beam_mass
            self.Ixy -= beam_mass * (arm_midpoint[0] - self.cg[0]) * (arm_midpoint[1] - self.cg[1])

            self.Ixz -= (1/12) * pos[0] * pos[2] * beam_mass
            self.Ixz -= beam_mass * (arm_midpoint[0] - self.cg[0]) * (arm_midpoint[2] - self.cg[2])

            self.Iyz -= (1/12) * pos[1] * pos[2] * beam_mass
            self.Iyz -= beam_mass * (arm_midpoint[1] - self.cg[1]) * (arm_midpoint[2] - self.cg[2])
        
        # Create inertia matrix with numerical stability improvements
        self.inertia_matrix = np.array([
            [self.Ix, self.Ixy, self.Ixz],
            [self.Ixy, self.Iy, self.Iyz],
            [self.Ixz, self.Iyz, self.Iz]
        ])
        
        # Clean up extremely small values that cause numerical instability
        # Values smaller than 1e-12 are likely numerical noise
        tolerance = 1e-12
        self.inertia_matrix[np.abs(self.inertia_matrix) < tolerance] = 0.0
        
        # Ensure matrix is symmetric (fix any tiny asymmetries from numerical errors)
        self.inertia_matrix = 0.5 * (self.inertia_matrix + self.inertia_matrix.T)
        
        # Ensure positive definite by checking eigenvalues
        eigenvals = np.linalg.eigvals(self.inertia_matrix)
        if np.any(eigenvals <= 0):
            print(f"Warning: Non-positive eigenvalues detected: {eigenvals}")
            # Add small positive value to diagonal to ensure positive definiteness
            min_eigenval = max(1e-6, -np.min(eigenvals) + 1e-6)
            self.inertia_matrix += min_eigenval * np.eye(3)

        # Add minimum inertia values to prevent numerical instability in dynamics
        min_inertia = 0.01  # kg*m^2 - reasonable minimum for small drones
        self.inertia_matrix[0,0] = max(self.inertia_matrix[0,0], min_inertia)
        self.inertia_matrix[1,1] = max(self.inertia_matrix[1,1], min_inertia)
        self.inertia_matrix[2,2] = max(self.inertia_matrix[2,2], min_inertia)

    def _compute_allocation_matrices(self):
        """Compute force and moment allocation matrices."""
        self.Bf = np.zeros((3, self.num_motors))
        self.Bm = np.zeros((3, self.num_motors))
        
        for idx, prop in enumerate(self.propellers):
            k_f, k_m = prop["constants"]
            w_max = prop["wmax"]
            prop_loc = np.array(prop["loc"])
            prop_r = prop_loc - self.cg  # Position relative to CG
            
            # Thrust direction (normalized)
            prop_dir = np.array(prop["dir"][:3])
            prop_dir = prop_dir / norm(prop_dir)
            
            # Rotation direction (-1 for CCW, +1 for CW when viewed from above)
            prop_rot = -1 if prop["dir"][-1] == "ccw" else 1
            
            # Force allocation (thrust in prop_dir direction)
            self.Bf[:, idx] = k_f * w_max**2 * prop_dir
            
            # Moment allocation (cross product + propeller torque)
            moment_from_thrust = np.cross(prop_r, k_f * w_max**2 * prop_dir)
            moment_from_torque = k_m * w_max**2 * prop_rot * prop_dir
            self.Bm[:, idx] = moment_from_thrust + moment_from_torque
        
        # Store unscaled matrices for reference
        self.Bf_unscaled = self.Bf.copy()
        self.Bm_unscaled = self.Bm.copy()
        
        # Store original matrices without pre-scaling
        # The dynamics equations will handle the mass/inertia scaling to avoid numerical issues
        
        # Combined allocation matrix for control allocation
        self.B_combined = np.vstack([self.Bf, self.Bm])  # (6 x num_motors)
        
        # Clean up tiny numerical values in allocation matrices to prevent NaN propagation
        tolerance = 1e-15
        self.Bf[np.abs(self.Bf) < tolerance] = 0.0
        self.Bm[np.abs(self.Bm) < tolerance] = 0.0
        self.B_combined[np.abs(self.B_combined) < tolerance] = 0.0
        
        # Compute pseudo-inverse with enhanced numerical stability
        try:
            self.B_pinv = np.linalg.pinv(self.B_combined, rcond=1e-10)
        except np.linalg.LinAlgError:
            print("Warning: Using fallback pseudo-inverse computation")
            # Fallback: use SVD decomposition with explicit tolerance
            U, s, Vt = np.linalg.svd(self.B_combined, full_matrices=False)
            s_inv = np.where(s > 1e-10, 1/s, 0)
            self.B_pinv = Vt.T @ np.diag(s_inv) @ U.T
        
        # Clean up tiny values in pseudo-inverse to prevent NaN propagation
        self.B_pinv[np.abs(self.B_pinv) < tolerance] = 0.0
    
    def get_physical_properties(self):
        """
        Get physical properties dictionary.
        
        Returns:
            dict: Physical properties including mass, CG, inertia components
        """
        return {
            'mass': self.mass,
            'center_of_gravity': self.cg.tolist(),
            'Ix': self.Ix,
            'Iy': self.Iy, 
            'Iz': self.Iz,
            'Ixy': self.Ixy,
            'Ixz': self.Ixz,
            'Iyz': self.Iyz,
            'inertia_matrix': self.inertia_matrix.tolist(),
            'num_motors': self.num_motors
        }
    
    def get_allocation_matrices(self):
        """
        Get force and moment allocation matrices.
        
        Returns:
            tuple: (Bf, Bm) force and moment allocation matrices
        """
        return self.Bf.copy(), self.Bm.copy()
    
    def get_control_allocation(self):
        """
        Get combined allocation matrix and its pseudo-inverse.
        
        Returns:
            tuple: (B_combined, B_pinv) for control allocation
        """
        return self.B_combined.copy(), self.B_pinv.copy()
    
    def is_over_actuated(self):
        """
        Check if drone is over-actuated (more motors than DOF).
        
        Returns:
            bool: True if over-actuated (num_motors > 6)
        """
        return self.num_motors > 6
    
    def get_motor_configuration_info(self):
        """
        Get comprehensive motor configuration information.
        
        Returns:
            dict: Configuration analysis including ranks, condition numbers, etc.
        """
        rank_f = np.linalg.matrix_rank(self.Bf)
        rank_m = np.linalg.matrix_rank(self.Bm)
        rank_combined = np.linalg.matrix_rank(self.B_combined)
        
        # Condition numbers for numerical analysis
        cond_f = np.linalg.cond(self.Bf @ self.Bf.T)
        cond_m = np.linalg.cond(self.Bm @ self.Bm.T)
        cond_combined = np.linalg.cond(self.B_combined @ self.B_combined.T)
        
        return {
            'num_motors': self.num_motors,
            'is_over_actuated': self.is_over_actuated(),
            'force_rank': rank_f,
            'moment_rank': rank_m,
            'combined_rank': rank_combined,
            'force_condition_number': cond_f,
            'moment_condition_number': cond_m,
            'combined_condition_number': cond_combined,
            'mass': self.mass,
            'center_of_gravity': self.cg.tolist()
        }