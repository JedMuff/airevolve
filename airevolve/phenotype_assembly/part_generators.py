"""
CAD Part Generation Functions

This module provides functions to generate individual drone parts using CadQuery.
It extracts and refactors the part generation logic from the original scripts.
"""

import cadquery as cq
import math
from typing import Dict, Any, Optional
from . import config


def create_arm_assembly(
    arm_tilt: float,
    arm_azimuth: float,
    motor_tilt: float,
    motor_azimuth: float,
    arm_length: float,
    # Sphere parameters
    sphere_radius: float = 12,
    # Clamp parameters
    clamp_jaw_length: float = 0,
    clamp_jaw_height: float = 2.2,
    clamp_base_width: float = 18,
    clamp_jaw_thickness: float = 6,
    clamp_inset: float = 10.5,
    # Arm plate parameters
    arm_plate_diameter: Optional[float] = None,
    arm_plate_thickness: Optional[float] = None,
    # Arm screw holes
    arm_screw_hole_diameter: Optional[float] = None,
    arm_screw_hole_inset: float = 7.5/2,
    arm_screw_hole_angles: list = None,
    # Arm cylinder parameters
    pocket_thickness: float = 1,
    cylinder_outer_radius: Optional[float] = None,
    # Motor disc parameters
    disc_diameter: float = 23,
    disc_thickness: float = 3,
    sphere_offset: float = 5,
    cylinder_extension: float = 13.428,
    # Motor screw holes
    motor_screw_diameter: Optional[float] = None,
    motor_screw_pattern_radius: Optional[float] = None,
    motor_screw_count: int = 4,
    motor_screw_start_angle: float = 45,
    motor_screw_depth: float = 25,
    center_hole_diameter: float = 0,
    # Flattening disc
    flattening_disc_diameter: Optional[float] = None,
    flattening_disc_thickness: float = 10,
    flattening_disc_is_cutter: bool = True
) -> Dict[str, cq.Workplane]:
    """
    Create a single arm assembly with sphere mount, cylinder arm, and motor disc.

    This function is extracted from full_drone_assembly.py's create_arm_assembly function.

    Args:
        arm_tilt: Tilt angle for the arm attachment (rotation around Y axis) in degrees
        arm_azimuth: Azimuth angle for the arm attachment (rotation around Z axis) in degrees
        motor_tilt: Tilt angle for the motor disc (rotation around Y axis) in degrees
        motor_azimuth: Azimuth angle for the motor disc (rotation around Z axis) in degrees
        arm_length: Length of the arm cylinder in mm
        [Additional parameters use config defaults if not provided]

    Returns:
        Dictionary with keys: 'sphere', 'upper_jaw', 'lower_jaw', 'motor_arm'
    """
    # Set defaults from config
    if arm_plate_diameter is None:
        arm_plate_diameter = config.plate_diameter
    if arm_plate_thickness is None:
        arm_plate_thickness = config.plate_thickness + config.plate_thickness_tolerance
    if arm_screw_hole_diameter is None:
        arm_screw_hole_diameter = config.screw_size + config.screw_size_tolerance
    if arm_screw_hole_angles is None:
        arm_screw_hole_angles = [180-11.25, 180+11.25]
    if cylinder_outer_radius is None:
        cylinder_outer_radius = config.arm_cylinder_inner_radius + pocket_thickness
    if motor_screw_diameter is None:
        motor_screw_diameter = config.screw_size + config.screw_size_tolerance
    if motor_screw_pattern_radius is None:
        motor_screw_pattern_radius = config.intermediary_outer_screw_pattern_radius
    if flattening_disc_diameter is None:
        flattening_disc_diameter = disc_diameter

    arm_plate_radius = arm_plate_diameter / 2

    # ═══════════════════════════════════════════════════════════════════
    # PART 1: ARM ATTACHMENT MOUNT (Sphere with Clamp)
    # ═══════════════════════════════════════════════════════════════════

    # 1. Create sphere
    sphere_with_hole = cq.Workplane("XY").sphere(sphere_radius)

    # 2. Create clamp jaws
    jaw_x_center = sphere_radius + clamp_jaw_length/2 - clamp_inset

    # Upper jaw
    upper_jaw_z = clamp_jaw_height/2
    upper_jaw = (
        cq.Workplane("XY")
        .rect(clamp_jaw_length, clamp_base_width)
        .extrude(clamp_jaw_thickness)
        .translate((jaw_x_center, 0, upper_jaw_z))
    )

    # Lower jaw
    lower_jaw_z = -clamp_jaw_thickness - clamp_jaw_height/2
    lower_jaw = (
        cq.Workplane("XY")
        .rect(clamp_jaw_length, clamp_base_width)
        .extrude(clamp_jaw_thickness)
        .translate((jaw_x_center, 0, lower_jaw_z))
    )

    # 3. Calculate screw hole positions
    plate_x_offset = arm_plate_radius + sphere_radius - clamp_inset
    screw_hole_radius = arm_plate_radius - arm_screw_hole_inset

    screw_hole_positions = []
    for angle in arm_screw_hole_angles:
        angle_rad = math.radians(angle)
        x_pos = plate_x_offset + screw_hole_radius * math.cos(angle_rad)
        y_pos = 0 + screw_hole_radius * math.sin(angle_rad)
        screw_hole_positions.append((x_pos, y_pos))

    # 4. Create and apply screw holes to jaws and sphere
    for x_pos, y_pos in screw_hole_positions:
        screw_hole = (
            cq.Workplane("XY")
            .workplane(offset=-sphere_radius*2)
            .center(x_pos, y_pos)
            .circle(arm_screw_hole_diameter / 2)
            .extrude(sphere_radius*4)
        )

        upper_jaw = upper_jaw.cut(screw_hole)
        lower_jaw = lower_jaw.cut(screw_hole)
        sphere_with_hole = sphere_with_hole.cut(screw_hole)

    # 5. Create mounting plate
    arm_plate = (
        cq.Workplane("XY")
        .circle(arm_plate_diameter / 2)
        .extrude(arm_plate_thickness)
        .translate((plate_x_offset, 0, -arm_plate_thickness/2))
    )

    # Cut screw holes through the plate
    for x_pos, y_pos in screw_hole_positions:
        screw_hole = (
            cq.Workplane("XY")
            .workplane(offset=-sphere_radius*2)
            .center(x_pos, y_pos)
            .circle(arm_screw_hole_diameter / 2)
            .extrude(sphere_radius*4)
        )

        arm_plate = arm_plate.cut(screw_hole)

    # Use the plate to cut the slot in the sphere
    sphere_with_hole = sphere_with_hole.cut(arm_plate)

    # 6. Create jaw cutter
    jaw_cutter_thickness = clamp_jaw_height * 4
    jaw_cutter_inset = -1
    jaw_cutter_x_offset = plate_x_offset + clamp_jaw_length - jaw_cutter_inset

    jaw_cutter = (
        cq.Workplane("XY")
        .circle(arm_plate_diameter / 2)
        .extrude(jaw_cutter_thickness)
        .translate((jaw_cutter_x_offset, 0, -jaw_cutter_thickness/2))
    )

    upper_jaw = upper_jaw.cut(jaw_cutter)
    lower_jaw = lower_jaw.cut(jaw_cutter)

    # ═══════════════════════════════════════════════════════════════════
    # PART 2: MOTOR ARM ATTACHMENT (Cylinder with Motor Disc)
    # ═══════════════════════════════════════════════════════════════════

    # 1. Create solid cylinder
    total_cylinder_height = arm_length + cylinder_extension

    arm_socket = (
        cq.Workplane("XY")
        .circle(cylinder_outer_radius)
        .extrude(total_cylinder_height)
    )

    # 2. Create motor mount disc at origin
    motor_disc = (
        cq.Workplane("XY")
        .circle(disc_diameter / 2)
        .extrude(disc_thickness)
    )

    if center_hole_diameter > 0:
        center_hole = (
            cq.Workplane("XY")
            .circle(center_hole_diameter / 2)
            .extrude(disc_thickness)
        )
        motor_disc = motor_disc.cut(center_hole)

    # 3. Rotate disc (yaw and pitch) then translate to sphere surface
    epsilon = 1e-3
    disc_pitch = motor_tilt
    disc_yaw = motor_azimuth

    if abs(disc_pitch - 90) < epsilon:
        disc_pitch = disc_pitch - epsilon
    elif abs(disc_pitch + 90) < epsilon:
        disc_pitch = disc_pitch + epsilon

    pitch_rad = math.radians(disc_pitch)
    yaw_rad = math.radians(disc_yaw)

    normal_x = math.sin(pitch_rad) * math.cos(yaw_rad)
    normal_y = math.sin(pitch_rad) * math.sin(yaw_rad)
    normal_z = math.cos(pitch_rad)

    sphere_surface_x = 0
    sphere_surface_y = 0
    sphere_surface_z = arm_length + sphere_offset

    motor_disc_solid = motor_disc.val()

    if disc_pitch != 0:
        motor_disc_solid = motor_disc_solid.rotate(
            (0, 0, 0),
            (0, 1, 0),
            disc_pitch
        )

    if disc_yaw != 0:
        motor_disc_solid = motor_disc_solid.rotate(
            (0, 0, 0),
            (0, 0, 1),
            disc_yaw
        )

    motor_disc_solid = motor_disc_solid.translate((sphere_surface_x, sphere_surface_y, sphere_surface_z))
    motor_disc = cq.Workplane("XY").add(motor_disc_solid)

    # 4. Combine cylinder and disc
    motor_arm_attachment = arm_socket.union(motor_disc)

    # 5. Create flattening disc
    flattening_offset_x = disc_thickness * normal_x
    flattening_offset_y = disc_thickness * normal_y
    flattening_offset_z = disc_thickness * normal_z

    flattening_disc = (
        cq.Workplane("XY")
        .circle(flattening_disc_diameter / 2)
        .extrude(flattening_disc_thickness)
    )

    flattening_disc_solid = flattening_disc.val()

    if disc_pitch != 0:
        flattening_disc_solid = flattening_disc_solid.rotate(
            (0, 0, 0),
            (0, 1, 0),
            disc_pitch
        )

    if disc_yaw != 0:
        flattening_disc_solid = flattening_disc_solid.rotate(
            (0, 0, 0),
            (0, 0, 1),
            disc_yaw
        )

    flattening_disc_solid = flattening_disc_solid.translate((
        sphere_surface_x + flattening_offset_x,
        sphere_surface_y + flattening_offset_y,
        sphere_surface_z + flattening_offset_z
    ))
    flattening_disc = cq.Workplane("XY").add(flattening_disc_solid)

    if flattening_disc_is_cutter:
        motor_arm_attachment = motor_arm_attachment.cut(flattening_disc)
    else:
        motor_arm_attachment = motor_arm_attachment.union(flattening_disc)

    # 6. Add motor screw holes
    if motor_screw_depth > 0:
        for i in range(motor_screw_count):
            angle = motor_screw_start_angle + (360 / motor_screw_count) * i
            angle_rad = math.radians(angle)

            x_pos_local = motor_screw_pattern_radius * math.cos(angle_rad)
            y_pos_local = motor_screw_pattern_radius * math.sin(angle_rad)

            screw_hole_cylinder = (
                cq.Workplane("XY")
                .center(x_pos_local, y_pos_local)
                .circle(motor_screw_diameter / 2)
                .extrude(motor_screw_depth)
            )

            screw_hole_solid = screw_hole_cylinder.val()

            if disc_pitch != 0:
                screw_hole_solid = screw_hole_solid.rotate(
                    (0, 0, 0),
                    (0, 1, 0),
                    disc_pitch
                )

            if disc_yaw != 0:
                screw_hole_solid = screw_hole_solid.rotate(
                    (0, 0, 0),
                    (0, 0, 1),
                    disc_yaw
                )

            screw_hole_solid = screw_hole_solid.translate((
                sphere_surface_x,
                sphere_surface_y,
                sphere_surface_z
            ))

            screw_hole_wp = cq.Workplane("XY").add(screw_hole_solid)
            motor_arm_attachment = motor_arm_attachment.cut(screw_hole_wp)

    # 7. Align motor arm attachment with arm attachment mount
    motor_arm_solid = motor_arm_attachment.val()

    if arm_tilt != 0:
        motor_arm_solid = motor_arm_solid.rotate(
            (0, 0, 0),
            (0, 1, 0),
            arm_tilt
        )

    if arm_azimuth != 0:
        motor_arm_solid = motor_arm_solid.rotate(
            (0, 0, 0),
            (0, 0, 1),
            arm_azimuth
        )

    motor_arm_attachment = cq.Workplane("XY").add(motor_arm_solid)

    # Return all components as a dict
    return {
        'sphere': sphere_with_hole,
        'upper_jaw': upper_jaw,
        'lower_jaw': lower_jaw,
        'motor_arm': motor_arm_attachment
    }


def create_landing_leg(
    # Cylinder (M3 mounting boss)
    cylinder_diameter: float = 2.8,
    cylinder_length: float = 6,
    # Cuboid 1 (first leg section)
    cuboid1_length: float = 50,
    cuboid1_width: float = 8,
    cuboid1_height: float = 2.8,
    cuboid1_wall_thickness: float = 2,
    # Cuboid 2 (second leg section)
    cuboid2_length: float = 50,
    cuboid2_width: float = 8,
    cuboid2_height: float = 2.8,
    cuboid2_wall_thickness: float = 2,
    # Cylinder end cap for smoothing
    add_half_cylinder_end: bool = True,
    half_cylinder_radius: Optional[float] = None,
    # Angle between the two cuboids (in degrees)
    angle_between_cuboids: float = -30,
    # Position of cuboid2 relative to cuboid1
    cuboid2_offset_z: float = 0,
    cuboid2_offset_x: float = -7,
    cuboid2_offset_y: float = 0,
    # Interlocking joint cutouts
    cuboid1_cutter_length: Optional[float] = None,
    cuboid1_cutter_width: Optional[float] = None,
    cuboid1_cutter_height: Optional[float] = None,
    cuboid1_cutter_offset_x: Optional[float] = None,
    cuboid1_cutter_offset_y: float = 0,
    cuboid2_cutter_length: Optional[float] = None,
    cuboid2_cutter_width: Optional[float] = None,
    cuboid2_cutter_height: Optional[float] = None,
    cuboid2_cutter_offset_x: Optional[float] = None,
    cuboid2_cutter_offset_y: float = 0
) -> cq.Workplane:
    """
    Create a landing leg using a 3-shape design (cylinder + 2 cuboids).

    Args:
        [All parameters with defaults - see landing_leg.py for details]

    Returns:
        CadQuery Workplane containing the landing leg geometry
    """
    # Set defaults
    if half_cylinder_radius is None:
        half_cylinder_radius = cuboid2_width / 2
    if cuboid1_cutter_length is None:
        cuboid1_cutter_length = cuboid1_length
    if cuboid1_cutter_width is None:
        cuboid1_cutter_width = cuboid1_width
    if cuboid1_cutter_height is None:
        cuboid1_cutter_height = cuboid1_height
    if cuboid1_cutter_offset_x is None:
        cuboid1_cutter_offset_x = -cuboid1_width
    if cuboid2_cutter_length is None:
        cuboid2_cutter_length = cuboid2_length
    if cuboid2_cutter_width is None:
        cuboid2_cutter_width = cuboid2_width
    if cuboid2_cutter_height is None:
        cuboid2_cutter_height = cuboid2_height
    if cuboid2_cutter_offset_x is None:
        cuboid2_cutter_offset_x = -cuboid2_width

    # Shape 1: CYLINDER (M3 mounting boss)
    cylinder = (
        cq.Workplane("XY")
        .circle(cylinder_diameter / 2)
        .extrude(cylinder_length)
        .translate((0, 0, 0))
    )

    # Shape 2: CUBOID 1 (first leg section, extends downward from cylinder)
    cuboid1 = (
        cq.Workplane("XY")
        .rect(cuboid1_width, cuboid1_height)
        .extrude(cuboid1_length)
        .translate((0, 0, -cuboid1_length))
    )

    # Make cuboid1 hollow if wall thickness is specified
    if cuboid1_wall_thickness > 0:
        cuboid1_inner = (
            cq.Workplane("XY")
            .rect(cuboid1_width - 2*cuboid1_wall_thickness, cuboid1_height)
            .extrude(cuboid1_length - 2*cuboid1_wall_thickness)
            .translate((0, 0, -cuboid1_length + cuboid1_wall_thickness))
        )
        cuboid1 = cuboid1.cut(cuboid1_inner)

    # Shape 3: CUBOID 2 (second leg section, at angle to cuboid1)
    cuboid2 = (
        cq.Workplane("XY")
        .rect(cuboid2_length, cuboid2_height)
        .extrude(cuboid2_width)
        .translate((cuboid2_length/2, 0, cuboid2_width/2))
    )

    # Make cuboid2 hollow if wall thickness is specified
    if cuboid2_wall_thickness > 0:
        cuboid2_inner = (
            cq.Workplane("XY")
            .rect(cuboid2_length - 2*cuboid2_wall_thickness, cuboid2_height)
            .extrude(cuboid2_width - 2*cuboid2_wall_thickness)
            .translate((cuboid2_length/2, 0, cuboid2_width/2 + cuboid2_wall_thickness))
        )
        cuboid2 = cuboid2.cut(cuboid2_inner)

    # Add cylinder end cap for smoothing
    if add_half_cylinder_end:
        end_cylinder = (
            cq.Workplane("XZ")
            .center(cuboid2_length, cuboid2_width)
            .circle(half_cylinder_radius)
            .extrude(cuboid2_height)
            .translate((0, cuboid2_height/2, 0))
        )
        cuboid2 = cuboid2.union(end_cylinder)

    # Rotate and position cuboid2
    cuboid2 = (
        cuboid2.val()
        .rotate((0, 0, 0), (0, 1, 0), -angle_between_cuboids)
        .translate((cuboid2_offset_x, cuboid2_offset_y, -cuboid1_length + cuboid2_offset_z))
    )
    cuboid2 = cq.Workplane("XY").add(cuboid2)

    # Create interlocking cutouts
    cuboid1_cutter = (
        cq.Workplane("XY")
        .rect(cuboid1_cutter_width, cuboid1_cutter_height)
        .extrude(cuboid1_cutter_length)
        .translate((cuboid1_cutter_offset_x, cuboid1_cutter_offset_y, -cuboid1_cutter_length))
    )

    cuboid2_cutter = (
        cq.Workplane("XY")
        .rect(cuboid2_cutter_length, cuboid2_cutter_height)
        .extrude(cuboid2_cutter_width)
        .translate((cuboid2_cutter_length/2, cuboid2_cutter_offset_y, cuboid2_cutter_width/2 + cuboid2_cutter_offset_x))
    )

    cuboid2_cutter = (
        cuboid2_cutter.val()
        .rotate((0, 0, 0), (0, 1, 0), -angle_between_cuboids)
        .translate((cuboid2_offset_x, cuboid2_offset_y, -cuboid1_length + cuboid2_offset_z))
    )
    cuboid2_cutter = cq.Workplane("XY").add(cuboid2_cutter)

    # Apply the cutouts
    cuboid1 = cuboid1.cut(cuboid2_cutter)
    cuboid2 = cuboid2.cut(cuboid1_cutter)

    # Combine all three shapes
    landing_leg = cylinder.union(cuboid1).union(cuboid2)

    return landing_leg
