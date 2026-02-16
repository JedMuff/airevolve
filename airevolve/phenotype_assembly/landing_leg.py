import cadquery as cq

# ═══════════════════════════════════════════════════════════════════
# PARAMETERS - Simple 3 shape landing leg design
# ═══════════════════════════════════════════════════════════════════

# Cylinder (M3 mounting boss)
cylinder_diameter = 2.8  # mm
cylinder_length = 6  # mm

# Cuboid 1 (first leg section)
cuboid1_length = 50  # mm
cuboid1_width = 8  # mm
cuboid1_height = 2.8  # mm
cuboid1_wall_thickness = 2  # mm - wall thickness (0 = solid, >0 = hollow)

# Cuboid 2 (second leg section)
cuboid2_length = 50  # mm
cuboid2_width = 8  # mm
cuboid2_height = 2.8  # mm
cuboid2_wall_thickness = 2  # mm - wall thickness (0 = solid, >0 = hollow)

# Cylinder end cap for smoothing
add_half_cylinder_end = True  # Set to False to disable
half_cylinder_radius = cuboid2_width / 2  # mm - radius of cylinder (typically half the width)

# Angle between the two cuboids (in degrees)
# 0 degrees = both cuboids aligned vertically
# 90 degrees = cuboid2 perpendicular to cuboid1
angle_between_cuboids = -30

# Position of cuboid2 relative to cuboid1
cuboid2_offset_z = 0  # mm - offset along cuboid1 length (0 = at end of cuboid1, negative moves up)
cuboid2_offset_x = -7  # mm - offset perpendicular to cuboid1 (lateral positioning)
cuboid2_offset_y = 0  # mm - offset perpendicular to cuboid1 (lateral positioning)

# Interlocking joint cutouts
# Cuboid 1 cutter dimensions (parallel to cuboid1, cuts into cuboid 2)
cuboid1_cutter_length = cuboid1_length  # mm - typically same as cuboid1_length
cuboid1_cutter_width = cuboid1_width  # mm - typically same as cuboid1_width
cuboid1_cutter_height = cuboid1_height  # mm - typically same as cuboid1_height
cuboid1_cutter_offset_x = -cuboid1_width  # mm - offset perpendicular to cuboid1 (negative = opposite side)
cuboid1_cutter_offset_y = 0  # mm - lateral offset

# Cuboid 2 cutter dimensions (parallel to cuboid2, cuts into cuboid 1)
cuboid2_cutter_length = cuboid2_length  # mm - typically same as cuboid2_length
cuboid2_cutter_width = cuboid2_width  # mm - typically same as cuboid2_width
cuboid2_cutter_height = cuboid2_height  # mm - typically same as cuboid2_height
cuboid2_cutter_offset_x = -cuboid2_width  # mm - offset perpendicular to cuboid2 (negative = opposite side)
cuboid2_cutter_offset_y = 0  # mm - lateral offset

# ═══════════════════════════════════════════════════════════════════
# CREATE LANDING LEG - 3 SHAPES ONLY
# ═══════════════════════════════════════════════════════════════════

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

# Make cuboid1 hollow if wall thickness is specified (only XZ perimeter walls remain)
if cuboid1_wall_thickness > 0:
    cuboid1_inner = (
        cq.Workplane("XY")
        .rect(cuboid1_width - 2*cuboid1_wall_thickness, cuboid1_height)  # Full height in Y
        .extrude(cuboid1_length - 2*cuboid1_wall_thickness)  # Leave walls at ends in Z
        .translate((0, 0, -cuboid1_length + cuboid1_wall_thickness))
    )
    cuboid1 = cuboid1.cut(cuboid1_inner)

# Shape 3: CUBOID 2 (second leg section, at angle to cuboid1)
# Start with cuboid2 extending along X axis
cuboid2 = (
    cq.Workplane("XY")
    .rect(cuboid2_length, cuboid2_height)
    .extrude(cuboid2_width)
    .translate((cuboid2_length/2, 0, cuboid2_width/2))
)

# Make cuboid2 hollow if wall thickness is specified (only XZ perimeter walls remain)
if cuboid2_wall_thickness > 0:
    cuboid2_inner = (
        cq.Workplane("XY")
        .rect(cuboid2_length - 2*cuboid2_wall_thickness, cuboid2_height)  # Full height in Y
        .extrude(cuboid2_width - 2*cuboid2_wall_thickness)  # Leave walls at ends in Z
        .translate((cuboid2_length/2, 0, cuboid2_width/2 + cuboid2_wall_thickness))
    )
    cuboid2 = cuboid2.cut(cuboid2_inner)

# Add cylinder end cap for smoothing
if add_half_cylinder_end:
    # Create full cylinder at the end of cuboid2, oriented along Y axis (height)
    # Position at the end face so half overlaps with cuboid2 and half extends beyond
    end_cylinder = (
        cq.Workplane("XZ")
        .center(cuboid2_length, cuboid2_width)
        .circle(half_cylinder_radius)
        .extrude(cuboid2_height)
        .translate((0, cuboid2_height/2, 0))
    )

    # Union with cuboid2
    cuboid2 = cuboid2.union(end_cylinder)

# Rotate cuboid2 by the angle parameter around the Y axis
# Then position it at the end of cuboid1 with optional offsets
cuboid2 = (
    cuboid2.val()
    .rotate((0, 0, 0), (0, 1, 0), -angle_between_cuboids)
    .translate((cuboid2_offset_x, cuboid2_offset_y, -cuboid1_length + cuboid2_offset_z))
)
cuboid2 = cq.Workplane("XY").add(cuboid2)

# ═══════════════════════════════════════════════════════════════════
# CREATE INTERLOCKING CUTOUTS
# ═══════════════════════════════════════════════════════════════════

# Cutter 1: Shaped like cuboid1, positioned adjacent to cuboid1, will cut cuboid2
cuboid1_cutter = (
    cq.Workplane("XY")
    .rect(cuboid1_cutter_width, cuboid1_cutter_height)
    .extrude(cuboid1_cutter_length)
    .translate((cuboid1_cutter_offset_x, cuboid1_cutter_offset_y, -cuboid1_cutter_length))
)

# Cutter 2: Shaped like cuboid2, positioned adjacent to cuboid2, will cut cuboid1
# Start with cutter extending along X axis (same as cuboid2)
cuboid2_cutter = (
    cq.Workplane("XY")
    .rect(cuboid2_cutter_length, cuboid2_cutter_height)
    .extrude(cuboid2_cutter_width)
    .translate((cuboid2_cutter_length/2, cuboid2_cutter_offset_y, cuboid2_cutter_width/2 + cuboid2_cutter_offset_x))
)

# Rotate and position cuboid2_cutter same as cuboid2
cuboid2_cutter = (
    cuboid2_cutter.val()
    .rotate((0, 0, 0), (0, 1, 0), -angle_between_cuboids)
    .translate((cuboid2_offset_x, cuboid2_offset_y, -cuboid1_length + cuboid2_offset_z))
)
cuboid2_cutter = cq.Workplane("XY").add(cuboid2_cutter)

# Apply the cutouts
cuboid1 = cuboid1.cut(cuboid2_cutter)
cuboid2 = cuboid2.cut(cuboid1_cutter)

# COMBINE ALL THREE SHAPES
landing_leg = cylinder.union(cuboid1).union(cuboid2)

# ═══════════════════════════════════════════════════════════════════
# EXPORT
# ═══════════════════════════════════════════════════════════════════

# Export as STEP file
landing_leg.val().exportStep("landing_leg.step")

# Export as STL file
cq.exporters.export(landing_leg, "daedalus_forge/stl/landing_leg.stl")

print(f"Landing leg created successfully!")
print(f"\nCylinder: {cylinder_diameter}mm diameter × {cylinder_length}mm length")
print(f"Cuboid 1: {cuboid1_length}mm × {cuboid1_width}mm × {cuboid1_height}mm (wall: {cuboid1_wall_thickness}mm)")
print(f"Cuboid 2: {cuboid2_length}mm × {cuboid2_width}mm × {cuboid2_height}mm (wall: {cuboid2_wall_thickness}mm)")
if add_half_cylinder_end:
    print(f"  Half-cylinder end cap: radius={half_cylinder_radius}mm")
print(f"Angle between cuboids: {angle_between_cuboids}°")
print(f"Cuboid 2 offset: x={cuboid2_offset_x}mm, y={cuboid2_offset_y}mm, z={cuboid2_offset_z}mm")
print(f"\nInterlocking joint cutouts applied (adjacent/parallel to main cuboids):")
print(f"  Cuboid 1 cutter: {cuboid1_cutter_length}mm × {cuboid1_cutter_width}mm × {cuboid1_cutter_height}mm")
print(f"    Offset: x={cuboid1_cutter_offset_x}mm, y={cuboid1_cutter_offset_y}mm")
print(f"  Cuboid 2 cutter: {cuboid2_cutter_length}mm × {cuboid2_cutter_width}mm × {cuboid2_cutter_height}mm")
print(f"    Offset: x={cuboid2_cutter_offset_x}mm, y={cuboid2_cutter_offset_y}mm")
