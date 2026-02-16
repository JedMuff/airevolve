


screw_size = 2
screw_size_tolerance = 0.3

individual_parameter_vector = [ # arm tilt, arm azimuth, motor tilt, motor azimuth
    [-90, 0, -90, 0],  # arm1
    [-90, 0, -90, 0],  # arm2
    [-90, 0, -90, 0],  # arm3
    [-90, 0, -90, 0],  # arm4
]

# Full drone assembly parameter vector
# Format: [arm_tilt, arm_azimuth, motor_tilt, motor_azimuth, arm_attachment_angle, arm_length]
# arm_attachment_angle: angle (0-360°) around the disc edge where the arm attaches
# arm_length: length of the arm cylinder in mm
full_drone_parameter_vector = [
    [-90, 0, 90, 0, 0, 60],     # arm1 - at 0°
    [-90, 0, 90, 0, 90, 60],    # arm2 - at 90°
    [-90, 0, 90, 0, 180, 60],   # arm3 - at 180°
    [-90, 0, 90, 0, 270, 60],   # arm4 - at 270°
]

# Plate (circular mounting plate)
plate_diameter = 60
plate_thickness = 2.0
plate_thickness_tolerance = 0.2

# Arm cylinder dimensions
arm_cylinder_inner_radius = 8/2
arm_cylinder_inner_radius_tolerance = 0.3

# Intermediary screw pattern
intermediary_outer_screw_pattern_radius = 18/2

# Core plate parameters
screw_pattern_radius = 60/2 - 7.5  # 7.5mm inset from edge
number_holes = 32
screw_square_size = 25.5
central_hole_diameter = 25
center_hub_radius = 20
outer_ring_width = 7.5
strut_width = 3
number_struts = 8
