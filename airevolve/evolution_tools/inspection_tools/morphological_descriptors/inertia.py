import copy
import numpy as np

from airevolve.evolution_tools.inspection_tools import utils as u
from airevolve.simulator.simulation.drone_configuration import DroneConfiguration


def _euler_to_rotation_matrix(roll, pitch, yaw):
    """Converts Euler angles (roll, pitch, yaw) to a rotation matrix."""
    R_z = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1],
        ]
    )
    R_y = np.array(
        [
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)],
        ]
    )
    R_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)],
        ]
    )
    return R_z @ R_y @ R_x


def _orientation_to_unit_vector(roll, pitch, yaw):
    """Converts roll, pitch, and yaw to a unit vector where the negative z-axis is pointing up."""
    default_rotation = np.array([0, 0, -1])
    R = _euler_to_rotation_matrix(roll, pitch, yaw)
    transform_from_ENU_to_NED = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
    R = transform_from_ENU_to_NED @ R
    transformed_vector = R @ default_rotation
    return transformed_vector / np.linalg.norm(transformed_vector)


def inertia(individual, motor_template=None):
    """Compute inertia components using DroneConfiguration.

    Returns: Ix, Iy, Iz, Ixy, Ixz, Iyz
    """
    if motor_template is None:
        motor_template = {"propsize": 5}

    individual = individual[~np.isnan(individual).any(axis=1)]

    props = []
    mypypd = individual[:, :6]  # [mag, arm_yaw, arm_pitch, mot_yaw, mot_pitch, dir]
    for mag, arm_yaw, arm_pitch, mot_yaw, mot_pitch, dir in mypypd:
        global_x, global_y, global_z = u.convert_to_cartesian(mag, arm_yaw, arm_pitch)
        x, y, z = u.ENU_to_NED(global_x, global_y, global_z)

        tmp = copy.deepcopy(motor_template)
        tmp.update({"loc": [float(x), float(y), float(z)]})
        d = "ccw" if dir == 0 else "cw"

        unit_vector = _orientation_to_unit_vector(0, mot_pitch, mot_yaw)
        tmp.update({"dir": [float(unit_vector[0]), float(unit_vector[1]), float(unit_vector[2]), d]})

        props.append(tmp)

    config = DroneConfiguration(props)
    return [config.Ix, config.Iy, config.Iz, config.Ixy, config.Ixz, config.Iyz]