import cadquery as cq
import math
import sys
from . import config

# ═══════════════════════════════════════════════════════════════════
# PARAMETERS
# ═══════════════════════════════════════════════════════════════════

# Arm length (controls cylinder height)
arm_length = 60  # mm - length of the cylinder connecting sphere to motor disc

# Angle parameters (in degrees)
arm_tilt = -30      # Tilt angle for the arm attachment (rotation around Y axis)
arm_azimuth = 30     # Azimuth angle for the arm attachment (rotation around Z axis)
motor_tilt = -90    # Tilt angle for the motor disc (rotation around Y axis)
motor_azimuth = 00   # Azimuth angle for the motor disc (rotation around Z axis)

# ═══════════════════════════════════════════════════════════════════
# ARM ATTACHMENT MOUNT PARAMETERS (Sphere with Clamp)
# ═══════════════════════════════════════════════════════════════════

# Sphere
sphere_radius = 17

# Cylinder alignment angles (used to orient the motor arm attachment)
hole_tilt = arm_tilt
hole_azimuth = arm_azimuth

# Clamp (grips sphere from outside)
clamp_jaw_length = 13
clamp_jaw_height = 2.2
clamp_base_width = 22
clamp_jaw_thickness = 4
clamp_inset = 7

# Plate (circular mounting plate)
plate_diameter = config.plate_diameter
plate_thickness = config.plate_thickness + config.plate_thickness_tolerance
plate_radius = plate_diameter / 2

# Jaw screw holes (corner mounting holes)
screw_hole_diameter = config.screw_size + config.screw_size_tolerance
screw_hole_inset = 8.0
screw_hole_angles = [170, 190]
jaw_cutter_inset = -1

# ═══════════════════════════════════════════════════════════════════
# MOTOR ARM ATTACHMENT PARAMETERS (Tube with Motor Disc)
# ═══════════════════════════════════════════════════════════════════

# Solid Cylinder (arm connection)
pocket_thickness = 1  # mm - used for radius calculation
cylinder_outer_radius = config.arm_cylinder_inner_radius + pocket_thickness
cylinder_height = arm_length  # Use arm_length parameter

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
# PART 1: ARM ATTACHMENT MOUNT (Sphere with Clamp)
# ═══════════════════════════════════════════════════════════════════

# 1. SPHERE (NO HOLE NEEDED)
# ═══════════════════════════════════════════════════════════════════

# Create base sphere (no hole needed since we're combining with solid cylinder)
sphere_with_hole = cq.Workplane("XY").sphere(sphere_radius)

# ═══════════════════════════════════════════════════════════════════
# 2. CLAMP JAWS
# ═══════════════════════════════════════════════════════════════════

# Calculate jaw position (centered at X axis, offset inward by clamp_inset)
jaw_x_center = sphere_radius + clamp_jaw_length/2 - clamp_inset

# Upper jaw (above XY plane)
upper_jaw_z = clamp_jaw_height/2
upper_jaw = (
    cq.Workplane("XY")
    .rect(clamp_jaw_length, clamp_base_width)
    .extrude(clamp_jaw_thickness)
    .translate((jaw_x_center, 0, upper_jaw_z))
)

# Lower jaw (below XY plane)
lower_jaw_z = -clamp_jaw_thickness - clamp_jaw_height/2
lower_jaw = (
    cq.Workplane("XY")
    .rect(clamp_jaw_length, clamp_base_width)
    .extrude(clamp_jaw_thickness)
    .translate((jaw_x_center, 0, lower_jaw_z))
)

# ═══════════════════════════════════════════════════════════════════
# 3. JAW SCREW HOLES
# ═══════════════════════════════════════════════════════════════════

# Calculate plate position for use in angled screw holes
plate_x_offset = plate_radius + sphere_radius - clamp_inset

# Calculate screw hole positions radially around disc circumference
screw_hole_radius = plate_radius - screw_hole_inset  # Distance from disc center

# Calculate (x, y) positions for each angle
screw_hole_positions = []
for angle in screw_hole_angles:
    angle_rad = math.radians(angle)
    x_pos = plate_x_offset + screw_hole_radius * math.cos(angle_rad)
    y_pos = 0 + screw_hole_radius * math.sin(angle_rad)
    screw_hole_positions.append((x_pos, y_pos))

# Create and apply screw holes to upper jaw and sphere
# Holes are vertical (along Z axis)
for x_pos, y_pos in screw_hole_positions:
    # Create vertical hole
    screw_hole = (
        cq.Workplane("XY")
        .workplane(offset=-sphere_radius*2)
        .center(x_pos, y_pos)
        .circle(screw_hole_diameter / 2)
        .extrude(sphere_radius*4)
    )

    upper_jaw = upper_jaw.cut(screw_hole)
    sphere_with_hole = sphere_with_hole.cut(screw_hole)

# Create and apply screw holes to lower jaw and sphere
for x_pos, y_pos in screw_hole_positions:
    # Create vertical hole
    screw_hole = (
        cq.Workplane("XY")
        .workplane(offset=-sphere_radius*2)
        .center(x_pos, y_pos)
        .circle(screw_hole_diameter / 2)
        .extrude(sphere_radius*4)
    )

    lower_jaw = lower_jaw.cut(screw_hole)
    sphere_with_hole = sphere_with_hole.cut(screw_hole)

# ═══════════════════════════════════════════════════════════════════
# 4. MOUNTING PLATE (created early to use for cutting slot)
# ═══════════════════════════════════════════════════════════════════

# Create circular plate horizontal in XY plane
# (plate_x_offset already calculated in section 3 for screw hole angles)
plate = (
    cq.Workplane("XY")
    .circle(plate_diameter / 2)
    .extrude(plate_thickness)
    .translate((plate_x_offset, 0, -plate_thickness/2))
)

# Cut screw holes through the plate (vertical holes)
for x_pos, y_pos in screw_hole_positions:
    # Create vertical hole
    screw_hole = (
        cq.Workplane("XY")
        .workplane(offset=-sphere_radius*2)
        .center(x_pos, y_pos)
        .circle(screw_hole_diameter / 2)
        .extrude(sphere_radius*4)
    )

    plate = plate.cut(screw_hole)

# Use the plate to cut the slot in the sphere
sphere_with_hole = sphere_with_hole.cut(plate)

# Create a thicker disc to cut the jaws for a spherical look
jaw_cutter_thickness = clamp_jaw_height * 4  # Make it thick enough to cut through jaws
jaw_cutter_x_offset = plate_x_offset + clamp_jaw_length - jaw_cutter_inset

jaw_cutter = (
    cq.Workplane("XY")
    .circle(plate_diameter / 2)
    .extrude(jaw_cutter_thickness)
    .translate((jaw_cutter_x_offset, 0, -jaw_cutter_thickness/2))
)

# Cut the jaws with the thicker disc
upper_jaw = upper_jaw.cut(jaw_cutter)
lower_jaw = lower_jaw.cut(jaw_cutter)


# ═══════════════════════════════════════════════════════════════════
# PART 2: MOTOR ARM ATTACHMENT (Tube with Motor Disc)
# ═══════════════════════════════════════════════════════════════════

# 1. CREATE SOLID CYLINDER (ARM CONNECTION)
# ═══════════════════════════════════════════════════════════════════

# Calculate total cylinder height including extension
total_cylinder_height = cylinder_height + cylinder_extension

# Create solid cylinder extending upward along +Z axis from z=0
arm_socket = (
    cq.Workplane("XY")
    .circle(cylinder_outer_radius)
    .extrude(total_cylinder_height)
)

# ═══════════════════════════════════════════════════════════════════
# 2. CREATE MOTOR MOUNT DISC (at origin, then rotate, then translate)
# ═══════════════════════════════════════════════════════════════════

# Create disc at origin (z = 0)
motor_disc = (
    cq.Workplane("XY")
    .circle(disc_diameter / 2)
    .extrude(disc_thickness)
)

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
# 6. ADD MOTOR SCREW HOLES (after rotation, cuts through disc and arm)
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
# 7. ALIGN MOTOR ARM ATTACHMENT WITH ARM ATTACHMENT MOUNT
# ═══════════════════════════════════════════════════════════════════

# Apply the same rotations as the hole in the arm attachment mount
# This aligns the tube to fit into the hole
motor_arm_solid = motor_arm_attachment.val()

# First apply tilt (rotation around Y axis)
if hole_tilt != 0:
    motor_arm_solid = motor_arm_solid.rotate(
        (0, 0, 0),
        (0, 1, 0),
        hole_tilt
    )

# Then apply azimuth (rotation around Z axis)
if hole_azimuth != 0:
    motor_arm_solid = motor_arm_solid.rotate(
        (0, 0, 0),
        (0, 0, 1),
        hole_azimuth
    )

motor_arm_attachment = cq.Workplane("XY").add(motor_arm_solid)

# ═══════════════════════════════════════════════════════════════════
# EXPORT
# ═══════════════════════════════════════════════════════════════════

print(f"Generating combined arm-motor mount")
print(f"  Arm length: {arm_length}mm")
print(f"  Arm tilt: {arm_tilt}°")
print(f"  Arm azimuth: {arm_azimuth}°")
print(f"  Motor tilt: {motor_tilt}°")
print(f"  Motor azimuth: {motor_azimuth}°")

# Create assembly with separate colored parts for visualization
assembly = (
    cq.Assembly()
    .add(sphere_with_hole, name="sphere", color=cq.Color("lightblue"))
    .add(upper_jaw, name="upper_jaw", color=cq.Color("red"))
    .add(lower_jaw, name="lower_jaw", color=cq.Color("green"))
    .add(motor_arm_attachment, name="motor_arm_attachment", color=cq.Color("orange"))
)

assembly.save(f"combined_arm_motor_mount.step")

# Export as single STL file
combined = cq.Compound.makeCompound([
    sphere_with_hole.val(),
    upper_jaw.val(),
    lower_jaw.val(),
    motor_arm_attachment.val()
])

cq.exporters.export(combined, f"daedalus_forge/stl/combined_arm_motor_mount.stl")

print(f"\nCombined arm-motor mount created successfully!")
print(f"\n=== ARM ATTACHMENT MOUNT (Sphere with Clamp) ===")
print(f"Sphere radius: {sphere_radius}mm")
print(f"Clamp jaw length: {clamp_jaw_length}mm")
print(f"Plate diameter: {plate_diameter}mm")
print(f"Alignment angles: tilt={hole_tilt}°, azimuth={hole_azimuth}°")

print(f"\n=== MOTOR ARM ATTACHMENT (Solid Cylinder with Motor Disc) ===")
print(f"Arm Socket (solid cylinder):")
print(f"  Radius: {cylinder_outer_radius}mm")
print(f"  Base height: {cylinder_height}mm")
print(f"  Extension: {cylinder_extension}mm")
print(f"  Total height: {total_cylinder_height}mm")

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
print(f"  Depth: {motor_screw_depth}mm")
