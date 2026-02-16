import cadquery as cq
import math
from . import config

# ═══════════════════════════════════════════════════════════════════
# PARAMETERS
# ═══════════════════════════════════════════════════════════════════

# Import full drone parameter vector from config
# Format: [arm_tilt, arm_azimuth, motor_tilt, motor_azimuth, arm_attachment_angle, arm_length]
arm_parameters = config.full_drone_parameter_vector

# ═══════════════════════════════════════════════════════════════════
# CORE PLATE PARAMETERS
# ═══════════════════════════════════════════════════════════════════

plate_diameter = config.plate_diameter
plate_thickness = config.plate_thickness
plate_radius = plate_diameter / 2

screw_hole_diameter = config.screw_size + config.screw_size_tolerance
screw_pattern_radius = config.screw_pattern_radius
number_holes = config.number_holes
screw_square_size = config.screw_square_size
central_hole_diameter = config.central_hole_diameter
center_hub_radius = config.center_hub_radius
outer_ring_width = config.outer_ring_width
strut_width = config.strut_width
number_struts = config.number_struts

outer_ring_inner_radius = plate_radius - outer_ring_width
strut_length = outer_ring_inner_radius - center_hub_radius + 2
outer_ring_middle_radius = (plate_radius + outer_ring_inner_radius) / 2

# ═══════════════════════════════════════════════════════════════════
# ARM ASSEMBLY PARAMETERS
# ═══════════════════════════════════════════════════════════════════

# Sphere parameters
sphere_radius = 12

# Clamp parameters
clamp_jaw_length = 0#8
clamp_jaw_height = 2.2
clamp_base_width = 18
clamp_jaw_thickness = 6
clamp_inset = 10.5

# Arm plate parameters
arm_plate_diameter = config.plate_diameter
arm_plate_thickness = config.plate_thickness + config.plate_thickness_tolerance
arm_plate_radius = arm_plate_diameter / 2

# Arm screw holes
arm_screw_hole_diameter = config.screw_size + config.screw_size_tolerance
arm_screw_hole_inset = 7.5/2
arm_screw_hole_angles = [180-11.25, 180+11.25]

# Arm cylinder parameters
pocket_thickness = 1
cylinder_outer_radius = config.arm_cylinder_inner_radius + pocket_thickness

# Motor disc parameters
disc_diameter = 23
disc_thickness = 3
sphere_offset = 5
cylinder_extension = 13.428

# Motor screw holes
motor_screw_diameter = config.screw_size + config.screw_size_tolerance
motor_screw_pattern_radius = config.intermediary_outer_screw_pattern_radius
motor_screw_count = 4
motor_screw_start_angle = 45
motor_screw_depth = 25
center_hole_diameter = 0

# Flattening disc
flattening_disc_diameter = disc_diameter
flattening_disc_thickness = 10
flattening_disc_is_cutter = True

# ═══════════════════════════════════════════════════════════════════
# FUNCTION: CREATE CORE PLATE
# ═══════════════════════════════════════════════════════════════════

def create_core_plate():
    """Create the central drone core plate with hub-and-spoke structure."""

    # 1. Create center hub disc
    center_hub = (
        cq.Workplane("XY")
        .circle(center_hub_radius)
        .extrude(plate_thickness)
    )

    # 2. Create outer ring
    outer_ring = (
        cq.Workplane("XY")
        .circle(plate_radius)
        .circle(outer_ring_inner_radius)
        .extrude(plate_thickness)
    )

    # 3. Start with center hub and outer ring
    plate = center_hub.union(outer_ring)

    # 4. Add radial struts connecting hub to outer ring
    for i in range(number_struts):
        angle = (360 / number_struts) * i + 45/2
        angle_rad = math.radians(angle)

        gap_midpoint_radius = (center_hub_radius + outer_ring_inner_radius) / 2
        strut_x = gap_midpoint_radius * math.cos(angle_rad)
        strut_y = gap_midpoint_radius * math.sin(angle_rad)

        strut = (
            cq.Workplane("XY")
            .rect(strut_length, strut_width)
            .extrude(plate_thickness)
        )

        strut = (
            strut.val()
            .rotate((0, 0, 0), (0, 0, 1), angle)
            .translate((strut_x, strut_y, 0))
        )
        strut = cq.Workplane("XY").add(strut)

        plate = plate.union(strut)

    # 5. Outer perimeter drill holes
    for i in range(number_holes):
        angle = (360 / number_holes) * i
        angle_rad = math.radians(angle)

        x_pos = outer_ring_middle_radius * math.cos(angle_rad)
        y_pos = outer_ring_middle_radius * math.sin(angle_rad)

        screw_hole = (
            cq.Workplane("XY")
            .workplane(offset=-plate_thickness)
            .center(x_pos, y_pos)
            .circle(screw_hole_diameter / 2)
            .extrude(plate_thickness * 3)
        )

        plate = plate.cut(screw_hole)

    # 6. Center square mount holes
    half_square = screw_square_size / 2
    square_hole_positions = [
        (half_square, half_square),
        (-half_square, half_square),
        (-half_square, -half_square),
        (half_square, -half_square)
    ]

    # 7. Central square hole
    central_hole = (
        cq.Workplane("XY")
        .workplane(offset=-plate_thickness)
        .rect(central_hole_diameter, central_hole_diameter)
        .extrude(plate_thickness * 3)
    )

    plate = plate.cut(central_hole)

    # 8. Add filled circles around the square mount screw holes
    circle_around_screw_diameter = screw_hole_diameter * 3
    for x_pos, y_pos in square_hole_positions:
        filled_circle = (
            cq.Workplane("XY")
            .center(x_pos, y_pos)
            .circle(circle_around_screw_diameter / 2)
            .extrude(plate_thickness)
        )

        plate = plate.union(filled_circle)

    # 9. Cut the screw holes through everything
    for x_pos, y_pos in square_hole_positions:
        screw_hole = (
            cq.Workplane("XY")
            .workplane(offset=-plate_thickness)
            .center(x_pos, y_pos)
            .circle(screw_hole_diameter / 2)
            .extrude(plate_thickness * 3)
        )

        plate = plate.cut(screw_hole)

    return plate

# ═══════════════════════════════════════════════════════════════════
# FUNCTION: CREATE ARM ASSEMBLY
# ═══════════════════════════════════════════════════════════════════

def create_arm_assembly(arm_tilt, arm_azimuth, motor_tilt, motor_azimuth, arm_length):
    """
    Create a single arm assembly with sphere mount, cylinder arm, and motor disc.

    Parameters:
    - arm_tilt: tilt angle for the arm attachment (rotation around Y axis)
    - arm_azimuth: azimuth angle for the arm attachment (rotation around Z axis)
    - motor_tilt: tilt angle for the motor disc (rotation around Y axis)
    - motor_azimuth: azimuth angle for the motor disc (rotation around Z axis)
    - arm_length: length of the arm cylinder in mm
    """

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
    jaw_cutter_x_offset = plate_x_offset + clamp_jaw_length - (-1)  # jaw_cutter_inset = -1

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

    # 1. Create solid cylinder - arm connecting sphere to motor disc
    # Key insight: The plate is in the +X direction. The arm cylinder goes in the OPPOSITE
    # direction (negative X), starting from the sphere surface closest to the center plate.
    # After assembly, a 180° rotation will flip it to point outward from the drone center.
    total_cylinder_height = arm_length + cylinder_extension

    # Create cylinder on YZ plane, extruding in -X direction (toward center, away from plate)
    # This way after the 180° rotation at assembly time, it will point radially outward
    arm_socket = (
        cq.Workplane("YZ")  # YZ plane at x=0
        .workplane(offset=sphere_radius)  # Start at sphere surface (x = sphere_radius, closest to plate)
        .circle(cylinder_outer_radius)
        .extrude(-total_cylinder_height)  # Extrude in -X direction (away from plate)
    )

    # 2. Create motor mount disc perpendicular to the arm cylinder
    # Since the cylinder extends in -X direction, the disc should be in the YZ plane
    # and extrude in the -X direction (same direction as the arm)
    motor_disc = (
        cq.Workplane("YZ")  # Disc in YZ plane
        .circle(disc_diameter / 2)
        .extrude(-disc_thickness)  # Extrude in -X direction
    )

    if center_hole_diameter > 0:
        center_hole = (
            cq.Workplane("YZ")
            .circle(center_hole_diameter / 2)
            .extrude(-disc_thickness)
        )
        motor_disc = motor_disc.cut(center_hole)

    # 3. Rotate disc (yaw and pitch) then translate to end of cylinder
    epsilon = 1e-3
    disc_pitch = motor_tilt
    disc_yaw = motor_azimuth

    if abs(disc_pitch - 90) < epsilon:
        disc_pitch = disc_pitch - epsilon
    elif abs(disc_pitch + 90) < epsilon:
        disc_pitch = disc_pitch + epsilon

    pitch_rad = math.radians(disc_pitch)
    yaw_rad = math.radians(disc_yaw)

    # Normal vector for the motor disc orientation
    normal_x = math.sin(pitch_rad) * math.cos(yaw_rad)
    normal_y = math.sin(pitch_rad) * math.sin(yaw_rad)
    normal_z = math.cos(pitch_rad)

    # Motor disc position: at the end of the cylinder (in -X direction from sphere)
    # Cylinder starts at x = sphere_radius and extends to x = sphere_radius - total_cylinder_height
    disc_end_x = sphere_radius - total_cylinder_height
    disc_end_y = 0
    disc_end_z = 0

    motor_disc_solid = motor_disc.val()

    # Rotate to achieve the desired motor orientation
    # Since the disc is now in YZ plane (perpendicular to X), we need different rotation logic
    # First rotate 90° around Y to make it perpendicular to Z (like the original XY disc)
    motor_disc_solid = motor_disc_solid.rotate((0, 0, 0), (0, 1, 0), 90)

    # Now apply the motor orientation rotations
    if disc_pitch != 0:
        motor_disc_solid = motor_disc_solid.rotate((0, 0, 0), (0, 1, 0), disc_pitch)

    if disc_yaw != 0:
        motor_disc_solid = motor_disc_solid.rotate((0, 0, 0), (0, 0, 1), disc_yaw)

    # Translate to the end of the cylinder
    motor_disc_solid = motor_disc_solid.translate((disc_end_x, disc_end_y, disc_end_z))
    motor_disc = cq.Workplane("XY").add(motor_disc_solid)

    # 4. Combine cylinder and disc
    motor_arm_attachment = arm_socket.union(motor_disc)

    # 5. Create flattening disc (positioned on top of motor disc)
    # The flattening disc extends further in the -X direction from the motor disc
    flattening_offset_x = -disc_thickness  # Negative because extending in -X direction
    flattening_offset_y = 0
    flattening_offset_z = 0

    flattening_disc = (
        cq.Workplane("YZ")
        .circle(flattening_disc_diameter / 2)
        .extrude(-flattening_disc_thickness)  # Extrude in -X direction
    )

    flattening_disc_solid = flattening_disc.val()

    # Apply same rotation as motor disc
    flattening_disc_solid = flattening_disc_solid.rotate((0, 0, 0), (0, 1, 0), 90)

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

    # Position at the end of the motor disc
    flattening_disc_solid = flattening_disc_solid.translate((
        disc_end_x + flattening_offset_x,
        disc_end_y + flattening_offset_y,
        disc_end_z + flattening_offset_z
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

            # Screw hole positions in the disc's YZ plane
            y_pos_local = motor_screw_pattern_radius * math.cos(angle_rad)
            z_pos_local = motor_screw_pattern_radius * math.sin(angle_rad)

            # Create screw holes extruding in -X direction (into the arm)
            screw_hole_cylinder = (
                cq.Workplane("YZ")
                .center(y_pos_local, z_pos_local)
                .circle(motor_screw_diameter / 2)
                .extrude(-motor_screw_depth)  # Extrude in -X direction
            )

            screw_hole_solid = screw_hole_cylinder.val()

            # Apply same rotations as motor disc
            screw_hole_solid = screw_hole_solid.rotate((0, 0, 0), (0, 1, 0), 90)

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

            # Translate to the disc position
            screw_hole_solid = screw_hole_solid.translate((
                disc_end_x,
                disc_end_y,
                disc_end_z
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

# ═══════════════════════════════════════════════════════════════════
# MAIN ASSEMBLY
# ═══════════════════════════════════════════════════════════════════

print("Generating full drone assembly")
print(f"  Number of arms: {len(arm_parameters)}")

# Create core plate
print("\nCreating core plate...")
core_plate = create_core_plate()

# Create assembly
assembly = cq.Assembly()
assembly.add(core_plate, name="core_plate", color=cq.Color("gray"))

# List to hold all parts for STL export
all_parts = [core_plate.val()]

# Create and position each arm
for i, params in enumerate(arm_parameters):
    arm_tilt, arm_azimuth, motor_tilt, motor_azimuth, arm_attachment_angle, arm_length = params

    print(f"\nCreating arm {i+1}:")
    print(f"  Arm tilt: {arm_tilt}°")
    print(f"  Arm azimuth: {arm_azimuth}°")
    print(f"  Motor tilt: {motor_tilt}°")
    print(f"  Motor azimuth: {motor_azimuth}°")
    print(f"  Attachment angle: {arm_attachment_angle}°")
    print(f"  Arm length: {arm_length}mm")

    # Create arm assembly
    arm_parts = create_arm_assembly(arm_tilt, arm_azimuth, motor_tilt, motor_azimuth, arm_length)

    # Calculate position so the clamp jaws align with the outer ring screw holes
    # The arm assembly is created with the mounting plate at plate_x_offset from origin (in +X direction)
    # After rotating 180°, the plate will be at -plate_x_offset from sphere center (pointing inward)
    # This must match the calculation in create_arm_assembly() at line 239
    plate_x_offset = arm_plate_radius + sphere_radius - clamp_inset

    # Thinking step by step after all transformations:
    # 1. Rotate 180°: plate is now at (-plate_x_offset, 0) relative to sphere at (0, 0)
    # 2. Rotate by attachment_angle: both rotate together
    # 3. Translate sphere to (radial_distance, 0) in the direction of attachment_angle
    # After step 3, plate is at: radial_distance - plate_x_offset (in radial direction)
    # We want plate at outer_ring_middle_radius, so: radial_distance - plate_x_offset = outer_ring_middle_radius
    # Therefore: radial_distance = outer_ring_middle_radius + plate_x_offset
    radial_distance = outer_ring_middle_radius + plate_x_offset 

    attachment_angle_rad = math.radians(arm_attachment_angle)
    attachment_x = radial_distance * math.cos(attachment_angle_rad)
    attachment_y = radial_distance * math.sin(attachment_angle_rad)
    attachment_z = plate_thickness/2  # Position at the top of the core plate

    # Rotate and translate each part of the arm
    for part_name, part in arm_parts.items():
        # First rotate around Z axis to position at the correct angle around the disc
        part_solid = part.val()

        # Rotate 180° to point outward from center (arms are created pointing in +X direction,
        # we need them to point away from the disc center)
        part_solid = part_solid.rotate(
            (0, 0, 0),
            (0, 0, 1),
            180
        )

        # Rotate around Z axis by the attachment angle
        part_solid = part_solid.rotate(
            (0, 0, 0),
            (0, 0, 1),
            arm_attachment_angle
        )

        # Translate to the edge of the disc
        part_solid = part_solid.translate((attachment_x, attachment_y, attachment_z))

        # Add to assembly with unique name and color
        part_wp = cq.Workplane("XY").add(part_solid)

        colors = {
            'sphere': cq.Color("lightblue"),
            'upper_jaw': cq.Color("red"),
            'lower_jaw': cq.Color("green"),
            'motor_arm': cq.Color("orange")
        }

        assembly.add(part_wp, name=f"arm{i+1}_{part_name}", color=colors.get(part_name, cq.Color("white")))
        all_parts.append(part_solid)

# ═══════════════════════════════════════════════════════════════════
# EXPORT
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("EXPORTING ASSEMBLY")
print("="*70)

# Save assembly as STEP file (preserves colors and separate parts)
assembly.save("full_drone_assembly.step")
print("Saved assembly as: full_drone_assembly.step")

# Export as single STL file
combined = cq.Compound.makeCompound(all_parts)
cq.exporters.export(combined, "daedalus_forge/stl/full_drone_assembly.stl")
print("Saved STL as: daedalus_forge/stl/full_drone_assembly.stl")

print("\n" + "="*70)
print("FULL DRONE ASSEMBLY CREATED SUCCESSFULLY!")
print("="*70)
print(f"\nCore Plate:")
print(f"  Diameter: {plate_diameter}mm")
print(f"  Thickness: {plate_thickness}mm")
print(f"\nArms: {len(arm_parameters)}")
for i, params in enumerate(arm_parameters):
    print(f"  Arm {i+1}: tilt={params[0]}°, azimuth={params[1]}°, " +
          f"motor_tilt={params[2]}°, motor_azimuth={params[3]}°, " +
          f"attachment={params[4]}°, length={params[5]}mm")
