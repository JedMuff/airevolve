"""
Drone Visualization Components

This module provides drone-specific graphics functions including mesh creation,
coordinate transformations, and drone configuration utilities.

Functions:
    Coordinate Conversion:
        - convert_to_cartesian: Convert magnitude/angles to cartesian coordinates
        - ENU_to_NED: Convert between coordinate systems
        - euler_to_rotation_matrix: Create rotation matrices from euler angles
    
    Mesh Creation:
        - create_grid: Create ground grid mesh
        - create_path: Create path/line mesh from points
        - create_circle: Create circular mesh (for propellers)
        - group: Combine multiple meshes
        
    Drone Construction:
        - create_drone: Build complete drone mesh from individual configuration
        - set_thrust: Update force vectors for thrust visualization
        - create_individual_from_config: Generate drone configuration arrays
"""

import numpy as np
from .graphics_3d import Mesh, Force, rotation_matrix

def convert_to_cartesian(magnitude, yaw, pitch, in_degrees=False):
    """
    Convert a vector from magnitude, yaw, and pitch to Cartesian coordinates (x, y, z).
    
    Parameters:
    magnitude (float): The magnitude of the vector.
    yaw (float): The yaw angle in degrees or radians.
    pitch (float): The pitch angle in degrees or radians.
    in_degrees (bool): Whether the yaw and pitch angles are given in degrees. Default is True.
    
    Returns:
    tuple: Cartesian coordinates (x, y, z).
    """
    if in_degrees:
        yaw = np.radians(yaw)
        pitch = np.radians(pitch)
    
    # Calculate Cartesian coordinates
    x = magnitude * np.cos(pitch) * np.cos(yaw)
    y = magnitude * np.cos(pitch) * np.sin(yaw)
    z = magnitude * np.sin(pitch)
    
    return (x, y, z)

def ENU_to_NED(x, y, z):
    """
    Convert Earth-Centered, Earth-Fixed (ENU) coordinates to North-East-Down (NED) coordinates.
    
    Parameters:
    x (float): The x-coordinate in ENU.
    y (float): The y-coordinate in ENU.
    z (float): The z-coordinate in ENU.
    
    Returns:
    tuple: The x, y, and z coordinates in NED.
    """
    return (y, x, -z)

def euler_to_rotation_matrix(roll, pitch, yaw):
    """
    Converts Euler angles (roll, pitch, yaw) to a rotation matrix.
    The rotation matrix follows the z-y-x convention (yaw-pitch-roll).
    
    Args:
        roll, pitch, yaw: Euler angles in radians
        
    Returns:
        3x3 rotation matrix
    """
    # Compute individual rotation matrices
    R_z = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0,            0,           1]
    ])
    
    R_y = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0,             1, 0            ],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    R_x = np.array([
        [1, 0,            0           ],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll),  np.cos(roll)]
    ])
    
    # Combined rotation matrix
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R

def create_grid(rows, cols, length):
    """
    Create a grid mesh for ground plane visualization.
    
    Args:
        rows: Number of rows
        cols: Number of columns  
        length: Grid cell size
        
    Returns:
        Mesh object representing the grid
    """
    rows, cols = rows+1, cols+1     # extra vertex in each direction
    vertices = np.zeros([rows * cols, 3])
    edges = []
    for i in range(rows):
        for j in range(cols):
            vertices[i * cols + j] = [
                i * length - (rows - 1) * length / 2,
                j * length - (cols - 1) * length / 2,
                0.
            ]
            if i != 0:
                edges.append((cols * (i - 1) + j, cols * i + j))
            if j != 0:
                edges.append((cols * i + j - 1, cols * i + j))
    return Mesh(vertices, np.array(edges))

def create_path(vertices, loop=False):
    """
    Create a path mesh connecting vertices in sequence.
    
    Args:
        vertices: Array of 3D points to connect
        loop: Whether to connect last point back to first
        
    Returns:
        Mesh object representing the path
    """
    edges = [(i, i+1) for i in range(len(vertices)-1)]
    if loop:
        edges.append((0, len(vertices)-1))
    return Mesh(np.array(vertices), np.array(edges))

def create_circle(r, px, py, pz, num=20, angle_x=0, angle_y=0, angle_z=0):
    """
    Create a circular mesh (used for propellers).
    
    Args:
        r: Radius of circle
        px, py, pz: Center position
        num: Number of vertices around circle
        angle_x, angle_y, angle_z: Rotation angles
        
    Returns:
        Mesh object representing the circle
    """
    vertices = np.array([[
        r * np.cos(i * 2 * np.pi / num),
        r * np.sin(i * 2 * np.pi / num),
        0
    ] for i in range(num)])
    
    R = euler_to_rotation_matrix(0, angle_y, angle_z)
    transform_from_ENU_to_NED = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])

    R = transform_from_ENU_to_NED @ R

    vertices = (R @ vertices.T).T
    vertices += np.array([px, py, pz])

    return create_path(vertices, loop=True)

def group(mesh_list):
    """
    Combine multiple meshes into a single mesh.
    
    Args:
        mesh_list: List of Mesh objects to combine
        
    Returns:
        Combined Mesh object
    """
    vertices = np.concatenate([
        mesh.vertices for mesh in mesh_list
    ])
    index_shifts = np.cumsum(
        [0] + [len(mesh.vertices) for mesh in mesh_list][:-1]
    )
    edges = np.concatenate([
        mesh.edges + shift for (mesh, shift) in zip(mesh_list, index_shifts)
    ])
    return Mesh(vertices, edges)

def create_drone(propellers, box_size=[0.2,0.2,0.2], prop_radius=0.08, scale=0.5, motor_colors=None):
    """
    Create a complete drone mesh from propeller configuration.
    
    Args:
        propellers: List of propeller dictionaries, each containing:
            - "loc": [x, y, z] position in body frame (meters)
            - "dir": [x, y, z, rotation] thrust direction and spin direction
            - "propsize": propeller size in inches (optional, not used for visualization)
        box_size: [width, depth, height] of central body
        prop_radius: Radius of propeller circles
        scale: Overall scale factor
        
    Returns:
        tuple: (drone_mesh, force_objects)
            drone_mesh: Complete Mesh object for the drone
            force_objects: List of Force objects for thrust visualization
    """

    box_size = np.array(box_size) * scale
    prop_radius *= scale

    # Create central body box
    bot_box = create_path(np.array([
        [ box_size[0]/2,  box_size[1]/2, box_size[2]/2],
        [-box_size[0]/2,  box_size[1]/2, box_size[2]/2],
        [-box_size[0]/2, -box_size[1]/2, box_size[2]/2],
        [ box_size[0]/2, -box_size[1]/2, box_size[2]/2],
    ]), loop=True)

    top_box = create_path(np.array([
        [ box_size[0]/2,  box_size[1]/2, -box_size[2]/2],
        [-box_size[0]/2,  box_size[1]/2, -box_size[2]/2],
        [-box_size[0]/2, -box_size[1]/2, -box_size[2]/2],
        [ box_size[0]/2, -box_size[1]/2, -box_size[2]/2],
    ]), loop=True)

    # Create vertical edges connecting top and bottom
    box_side_line1 = create_path(np.array([
        [ box_size[0]/2,  box_size[1]/2, box_size[2]/2],
        [ box_size[0]/2,  box_size[1]/2,-box_size[2]/2],
    ]))

    box_side_line2 = create_path(np.array([
        [-box_size[0]/2,  box_size[1]/2, box_size[2]/2],
        [-box_size[0]/2,  box_size[1]/2,-box_size[2]/2],
    ]))

    box_side_line3 = create_path(np.array([
        [-box_size[0]/2, -box_size[1]/2, box_size[2]/2],
        [-box_size[0]/2, -box_size[1]/2,-box_size[2]/2],
    ]))

    box_side_line4 = create_path(np.array([
        [ box_size[0]/2, -box_size[1]/2, box_size[2]/2],
        [ box_size[0]/2, -box_size[1]/2,-box_size[2]/2],
    ]))

    # Start with central body components
    drawings = [bot_box, top_box, box_side_line1, box_side_line2, box_side_line3, box_side_line4]
    centres = []
    
    # Create arms and propellers based on propeller configuration
    for prop in propellers:
        # Get propeller location and direction directly
        loc = np.array(prop["loc"]) * scale
        x, y, z = loc[0], loc[1], loc[2]
        
        # Get motor direction for propeller orientation
        motor_dir = np.array(prop["dir"][:3])
        
        # motor_dir represents thrust direction in NED frame
        # For standard downward thrust [0,0,-1], propeller disk should be horizontal
        # The create_circle function creates a circle in XY plane, then rotates it
        
        # Calculate angles for propeller disk orientation (perpendicular to thrust)
        if np.allclose(motor_dir, [0, 0, -1]):  # Standard downward thrust
            motor_yaw = 0
            motor_pitch = 0  # Horizontal disk
        elif np.allclose(motor_dir, [0, 0, 1]):  # Upward thrust
            motor_yaw = 0  
            motor_pitch = np.pi  # Inverted horizontal disk
        else:
            # General case: calculate angles for arbitrary thrust direction
            motor_yaw = np.arctan2(motor_dir[1], motor_dir[0])
            # Pitch is angle between thrust vector and horizontal plane
            motor_pitch = np.arcsin(np.clip(-motor_dir[2], -1, 1))
        
        # Create propeller circle with motor orientation
        circle = create_circle(prop_radius, x, y, z, num=20, angle_y=motor_pitch, angle_z=motor_yaw)
        # Create arm line from center to propeller
        arm_line = create_path(np.array([[0, 0, 0], [x, y, z]]))

        drawings.append(circle)
        drawings.append(arm_line)
        centres.append([x, y, z])
    
    # Combine all mesh components
    drone = group(drawings)
    
    # Add propeller centers as additional vertices for force attachment
    drone.vertices = np.concatenate([
        drone.vertices,
        np.array(centres)  # centers of the circles
    ])

    # Create force objects at each propeller location
    forces = []
    for v in drone.vertices[-len(propellers):]:
        forces.append(Force(v))

    return drone, forces

def set_thrust(drone, forces, T):
    """
    Update force vectors to represent thrust magnitudes.
    
    Args:
        drone: Drone mesh object (for orientation)
        forces: List of Force objects
        T: Array of thrust magnitudes for each motor
    """
    # Handle mismatch between number of forces and thrust commands
    num_forces = len(forces)
    num_thrusts = len(T) if hasattr(T, '__len__') else 1
    
    # Use the minimum to avoid index errors
    num_motors = min(num_forces, num_thrusts)
    
    for i in range(num_motors):
        if i < len(forces) and i < len(T):
            forces[i].F = - T[i] * rotation_matrix(drone.theta)[:, 2]
    
    # If we have more forces than thrust commands, set remaining forces to zero
    for i in range(num_motors, num_forces):
        forces[i].F = np.zeros(3)

