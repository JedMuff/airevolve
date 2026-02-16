import cadquery as cq
import math
import sys
from . import config

# ═══════════════════════════════════════════════════════════════════
# PARAMETERS - Motor-to-Arm Attachment
# ═══════════════════════════════════════════════════════════════════

# Arm index (0-3 for arm1-arm4, can be overridden via command line)
arm_index = 0
if len(sys.argv) > 1:
    try:
        arm_index = int(sys.argv[1])
        if arm_index < 0 or arm_index >= len(config.individual_parameter_vector):
            print(f"Warning: arm_index {arm_index} out of range, using 0")
            arm_index = 0
    except ValueError:
        print(f"Warning: invalid arm_index, using 0")
        arm_index = 0

# Get angles from config for this arm [arm_tilt, arm_azimuth, motor_tilt, motor_azimuth]
_, _, motor_tilt, motor_azimuth = config.individual_parameter_vector[arm_index]

# Hollow Cylinder (arm socket)
pocket_thickness = 1  # mm - thickness of the mounting pocket wall
cylinder_outer_radius = config.arm_cylinder_inner_radius + pocket_thickness
cylinder_inner_radius = config.arm_cylinder_inner_radius
cylinder_height = 20
cylinder_bottom_thickness = 20

# Cylinder mounting screw hole (horizontal through cylinder)
cylinder_screw_diameter = config.screw_size + config.screw_size_tolerance
cylinder_screw_inset = 5

# Motor Mount Disc
disc_diameter = 23
disc_thickness = 3
disc_yaw = motor_azimuth
disc_pitch = motor_tilt

# Sphere offset for disc positioning
sphere_offset = 5

cylinder_extension = 13.428

# Motor Screw Holes
motor_screw_diameter = config.screw_size + config.screw_size_tolerance
motor_screw_pattern_radius = config.intermediary_outer_screw_pattern_radius
motor_screw_count = 4
motor_screw_start_angle = 45
motor_screw_depth = 25

# Center hole (optional, for motor shaft or wiring)
center_hole_diameter = 0

# Flattening Disc (second disc on top of motor disc)
flattening_disc_diameter = disc_diameter
flattening_disc_thickness = 10
flattening_disc_is_cutter = True

# ═══════════════════════════════════════════════════════════════════
# 1. CREATE HOLLOW CYLINDER (ARM SOCKET) WITH TOP CAP
# ═══════════════════════════════════════════════════════════════════

# Calculate total cylinder height including extension
total_cylinder_height = cylinder_height + cylinder_extension

# Create outer cylinder extending upward along +Z axis from z=0
outer_cylinder = (
    cq.Workplane("XY")
    .circle(cylinder_outer_radius)
    .extrude(total_cylinder_height)
)

# Create inner cylinder to hollow it out, stopping before the top to leave a cap
inner_cylinder = (
    cq.Workplane("XY")
    .circle(cylinder_inner_radius)
    .extrude(total_cylinder_height - cylinder_bottom_thickness)
)

# Create hollow cylinder by cutting inner from outer (leaves solid top cap)
arm_socket = outer_cylinder.cut(inner_cylinder)

# Add horizontal screw hole through cylinder if specified
if cylinder_screw_diameter > 0:
    # Create horizontal screw hole (along Y axis, through cylinder)
    screw_hole = (
        cq.Workplane("XZ")
        .workplane(offset=0)
        .center(0, cylinder_screw_inset)
        .circle(cylinder_screw_diameter / 2)
        .extrude(cylinder_outer_radius * 3, both=True)
    )
    arm_socket = arm_socket.cut(screw_hole)

# ═══════════════════════════════════════════════════════════════════
# 2. CREATE MOTOR MOUNT DISC (at origin, then rotate, then translate)
# ═══════════════════════════════════════════════════════════════════

# Create disc at origin (z = 0)
motor_disc = (
    cq.Workplane("XY")
    .circle(disc_diameter / 2)
    .extrude(disc_thickness)
)

# Note: Motor screw holes will be added after rotation and positioning
# This allows them to extend through both the disc and arm socket

# Add center hole if specified
if center_hole_diameter > 0:
    center_hole = (
        cq.Workplane("XY")
        .circle(center_hole_diameter / 2)
        .extrude(disc_thickness)
    )
    motor_disc = motor_disc.cut(center_hole)

# ═══════════════════════════════════════════════════════════════════
# 3. ROTATE DISC (YAW AND PITCH) then translate to sphere surface
# ═══════════════════════════════════════════════════════════════════



# Add small epsilon to avoid numerical issues at exact ±90 degrees
# This prevents potential geometry/rendering issues and improves stability
epsilon = 1e-3
if abs(disc_pitch - 90) < epsilon:
    print("Adjusting pitch from exact +90 degrees to avoid numerical issues.")
    disc_pitch = disc_pitch - epsilon
elif abs(disc_pitch + 90) < epsilon:
    print("Adjusting pitch from exact -90 degrees to avoid numerical issues.")
    disc_pitch = disc_pitch + epsilon

# Calculate position based on pitch and yaw angles
pitch_rad = math.radians(disc_pitch)
yaw_rad = math.radians(disc_yaw)
print(f"Calculating disc position for pitch={disc_pitch}°, yaw={disc_yaw}°")

# Normal vector (unit vector)
normal_x = math.sin(pitch_rad) * math.cos(yaw_rad)
normal_y = math.sin(pitch_rad) * math.sin(yaw_rad)
normal_z = math.cos(pitch_rad)

# Position at cylinder top with sphere offset
sphere_surface_x = 0
sphere_surface_y = 0
sphere_surface_z = cylinder_height + sphere_offset

# Apply rotations to the disc around origin, then translate to sphere surface
motor_disc_solid = motor_disc.val()

# First apply pitch (tilt around Y axis at origin)
if disc_pitch != 0:
    motor_disc_solid = motor_disc_solid.rotate(
        (0, 0, 0),
        (0, 1, 0),
        disc_pitch
    )

# Then apply yaw (rotation around Z axis at origin)
if disc_yaw != 0:
    motor_disc_solid = motor_disc_solid.rotate(
        (0, 0, 0),
        (0, 0, 1),
        disc_yaw
    )

# Translate to sphere surface
motor_disc_solid = motor_disc_solid.translate((sphere_surface_x, sphere_surface_y, sphere_surface_z))
motor_disc = cq.Workplane("XY").add(motor_disc_solid)

# ═══════════════════════════════════════════════════════════════════
# 4. COMBINE CYLINDER AND DISC
# ═══════════════════════════════════════════════════════════════════

# Union the arm socket and motor disc
motor_arm_attachment = arm_socket.union(motor_disc)

# ═══════════════════════════════════════════════════════════════════
# 5. CREATE FLATTENING DISC (positioned exactly on top of motor disc)
# ═══════════════════════════════════════════════════════════════════

# Calculate offset to position flattening disc exactly on top of motor disc
# Move along the normal vector by the motor disc thickness
flattening_offset_x = disc_thickness * normal_x
flattening_offset_y = disc_thickness * normal_y
flattening_offset_z = disc_thickness * normal_z

# Create flattening disc at origin, rotate it same as motor disc
flattening_disc = (
    cq.Workplane("XY")
    .circle(flattening_disc_diameter / 2)
    .extrude(flattening_disc_thickness)
)

flattening_disc_solid = flattening_disc.val()

# Apply the SAME rotations as motor disc (pitch and yaw)
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

# Translate to sit exactly on top of the motor disc on sphere surface
flattening_disc_solid = flattening_disc_solid.translate((
    sphere_surface_x + flattening_offset_x,
    sphere_surface_y + flattening_offset_y,
    sphere_surface_z + flattening_offset_z
))
flattening_disc = cq.Workplane("XY").add(flattening_disc_solid)

# Apply flattening disc as either cutter or physical object
if flattening_disc_is_cutter:
    # Cut mode: subtract the flattening disc to create flat top
    motor_arm_attachment = motor_arm_attachment.cut(flattening_disc)
else:
    # Physical mode: add as real object for debugging
    motor_arm_attachment = motor_arm_attachment.union(flattening_disc)

# ═══════════════════════════════════════════════════════════════════
# 5a. ADD MOTOR SCREW HOLES (after rotation, cuts through disc and arm)
# ═══════════════════════════════════════════════════════════════════

if motor_screw_depth > 0:
    # Create motor screw holes that extend through disc into arm socket
    for i in range(motor_screw_count):
        # Calculate angle for this screw hole
        angle = motor_screw_start_angle + (360 / motor_screw_count) * i
        angle_rad = math.radians(angle)

        # Position on the disc surface (before rotation)
        x_pos_local = motor_screw_pattern_radius * math.cos(angle_rad)
        y_pos_local = motor_screw_pattern_radius * math.sin(angle_rad)

        # Create a cylinder at origin oriented along Z axis
        screw_hole_cylinder = (
            cq.Workplane("XY")
            .center(x_pos_local, y_pos_local)
            .circle(motor_screw_diameter / 2)
            .extrude(motor_screw_depth)
        )

        # Apply the SAME rotations as motor disc
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

        # Translate to disc position on sphere surface
        screw_hole_solid = screw_hole_solid.translate((
            sphere_surface_x,
            sphere_surface_y,
            sphere_surface_z
        ))

        # Convert back to workplane and cut from assembly
        screw_hole_wp = cq.Workplane("XY").add(screw_hole_solid)
        motor_arm_attachment = motor_arm_attachment.cut(screw_hole_wp)

# ═══════════════════════════════════════════════════════════════════
# EXPORT
# ═══════════════════════════════════════════════════════════════════

print(f"Generating motor arm attachment for arm {arm_index + 1}")
print(f"  Motor tilt: {motor_tilt}°")
print(f"  Motor azimuth: {motor_azimuth}°")

# Export as STEP file
motor_arm_attachment.val().exportStep(f"motor_arm_attachment_arm{arm_index + 1}.step")

# Export as STL file
cq.exporters.export(motor_arm_attachment, f"daedalus_forge/stl/motor_arm_attachment_arm{arm_index + 1}.stl")

print(f"Motor-to-arm attachment for arm {arm_index + 1} created successfully!")
print(f"\nArm Socket (hollow cylinder):")
print(f"  Outer radius: {cylinder_outer_radius}mm")
print(f"  Inner radius: {cylinder_inner_radius}mm")
print(f"  Wall thickness: {cylinder_outer_radius - cylinder_inner_radius}mm")
print(f"  Base height: {cylinder_height}mm")
print(f"  Extension: {cylinder_extension}mm")
print(f"  Total height: {total_cylinder_height}mm")
print(f"  Top cap thickness: {cylinder_bottom_thickness}mm")
if cylinder_screw_diameter > 0:
    print(f"  Horizontal screw hole: {cylinder_screw_diameter}mm diameter, {cylinder_screw_inset}mm from bottom")
else:
    print(f"  Horizontal screw hole: Disabled")
print(f"\nMotor Mount Disc:")
print(f"  Diameter: {disc_diameter}mm")
print(f"  Thickness: {disc_thickness}mm")
print(f"  Yaw angle: {disc_yaw}°")
print(f"  Pitch angle: {disc_pitch}°")
print(f"  Position: ({sphere_surface_x:.2f}, {sphere_surface_y:.2f}, {sphere_surface_z:.2f})mm")
print(f"\nMotor Screw Holes:")
print(f"  Count: {motor_screw_count}")
print(f"  Diameter: {motor_screw_diameter}mm")
print(f"  Pattern radius: {motor_screw_pattern_radius}mm")
print(f"  Start angle: {motor_screw_start_angle}°")
if motor_screw_depth > 0:
    print(f"  Depth: {motor_screw_depth}mm (cuts through disc and into arm tube)")
else:
    print(f"  Depth: Disabled (set motor_screw_depth > 0 to enable)")
if center_hole_diameter > 0:
    print(f"\nCenter Hole:")
    print(f"  Diameter: {center_hole_diameter}mm")

print(f"\nFlattening Disc:")
print(f"  Diameter: {flattening_disc_diameter}mm")
print(f"  Thickness: {flattening_disc_thickness}mm")
print(f"  Mode: {'CUTTER (subtracts)' if flattening_disc_is_cutter else 'PHYSICAL (adds for debugging)'}")
print(f"  Rotated same as motor disc (pitch={disc_pitch}°, yaw={disc_yaw}°)")
flattening_pos_x = sphere_surface_x + flattening_offset_x
flattening_pos_y = sphere_surface_y + flattening_offset_y
flattening_pos_z = sphere_surface_z + flattening_offset_z
print(f"  Position: ({flattening_pos_x:.2f}, {flattening_pos_y:.2f}, {flattening_pos_z:.2f})mm")
print(f"  Offset from motor disc: ({flattening_offset_x:.2f}, {flattening_offset_y:.2f}, {flattening_offset_z:.2f})mm")
