import cadquery as cq
import math
import sys
from . import config

# ═══════════════════════════════════════════════════════════════════
# PARAMETERS
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
arm_tilt, arm_azimuth, _, _ = config.individual_parameter_vector[arm_index]

# Sphere
sphere_radius = 17

# Main hole (drilled from top of sphere)
hole_diameter = (config.arm_cylinder_inner_radius + config.arm_cylinder_inner_radius_tolerance) * 2
hole_depth = 22
hole_tilt = arm_tilt
hole_azimuth = arm_azimuth

# Center screw hole (perpendicular to main hole, through sphere center)
center_screw_diameter = config.screw_size + config.screw_size_tolerance
center_screw_rotation = 0

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
# 1. SPHERE WITH MAIN HOLE
# ═══════════════════════════════════════════════════════════════════

# Create base sphere
sphere = cq.Workplane("XY").sphere(sphere_radius)

# Create main hole cylinder (initially vertical along -Z axis)
hole_cylinder = (
    cq.Workplane("XY")
    .workplane(offset=sphere_radius - hole_depth)
    .circle(hole_diameter / 2)
    .extrude(hole_depth)
)

# Apply tilt and azimuth rotations to hole
if hole_tilt != 0 or hole_azimuth != 0:
    hole_cylinder = (
        hole_cylinder.val()
        .rotate((0, 0, 0), (0, 1, 0), hole_tilt)
        .rotate((0, 0, 0), (0, 0, 1), hole_azimuth)
    )
    hole_cylinder = cq.Workplane("XY").add(hole_cylinder)

# Cut main hole from sphere
sphere_with_hole = sphere.cut(hole_cylinder)

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
# 5. CENTER SCREW HOLE
# ═══════════════════════════════════════════════════════════════════

# Create screw hole perpendicular to main hole, through sphere center
# Initially oriented along X axis (perpendicular to initial -Z hole direction)
center_screw_hole = (
    cq.Workplane("YZ")
    .center(0, 0)
    .circle(center_screw_diameter / 2)
    .extrude(sphere_radius * 2, both=True)
)

# Apply same rotations as main hole to maintain perpendicular relationship
if center_screw_rotation != 0 or hole_tilt != 0 or hole_azimuth != 0:
    center_screw_hole = (
        center_screw_hole.val()
        .rotate((0, 0, 0), (0, 0, 1), center_screw_rotation)
        .rotate((0, 0, 0), (0, 1, 0), hole_tilt)
        .rotate((0, 0, 0), (0, 0, 1), hole_azimuth)
    )
    center_screw_hole = cq.Workplane("XY").add(center_screw_hole)

sphere_with_hole = sphere_with_hole.cut(center_screw_hole)

# ═══════════════════════════════════════════════════════════════════
# EXPORT
# ═══════════════════════════════════════════════════════════════════

print(f"Generating arm attachment mount for arm {arm_index + 1}")
print(f"  Arm tilt: {arm_tilt}°")
print(f"  Arm azimuth: {arm_azimuth}°")

# Create assembly with separate colored parts for visualization
assembly = (
    cq.Assembly()
    .add(sphere_with_hole, name="sphere", color=cq.Color("lightblue"))
    .add(upper_jaw, name="upper_jaw", color=cq.Color("red"))
    .add(lower_jaw, name="lower_jaw", color=cq.Color("green"))
)

assembly.save(f"arm_attachement_mount_arm{arm_index + 1}.step")

# Export as single STL file
combined = cq.Compound.makeCompound([
    sphere_with_hole.val(),
    upper_jaw.val(),
    lower_jaw.val()
])

cq.exporters.export(combined, f"daedalus_forge/stl/arm_attachement_mount_arm{arm_index + 1}.stl")

print(f"Arm attachment mount for arm {arm_index + 1} created successfully!")
