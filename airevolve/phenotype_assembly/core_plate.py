"""
Core Plate Generation Module

Generates the central drone core plate with hub-and-spoke structure.
"""

import cadquery as cq
import math
from typing import Optional, Dict, Any
from . import config


def create_core_plate(
    plate_diameter: Optional[float] = None,
    plate_thickness: Optional[float] = None,
    screw_size: Optional[float] = None,
    screw_size_tolerance: Optional[float] = None,
    screw_pattern_radius: Optional[float] = None,
    number_holes: Optional[int] = None,
    screw_square_size: Optional[float] = None,
    central_hole_diameter: Optional[float] = None,
    center_hub_radius: Optional[float] = None,
    outer_ring_width: Optional[float] = None,
    strut_width: Optional[float] = None,
    number_struts: Optional[int] = None
) -> cq.Workplane:
    """
    Create the central drone core plate with hub-and-spoke structure.

    Args:
        plate_diameter: Diameter of the plate in mm (default from config)
        plate_thickness: Thickness of the plate in mm (default from config)
        screw_size: Diameter of screws in mm (default from config)
        screw_size_tolerance: Tolerance for screw holes in mm (default from config)
        screw_pattern_radius: Radius of outer screw pattern (default from config)
        number_holes: Number of holes in outer ring (default from config)
        screw_square_size: Size of center square mount pattern (default from config)
        central_hole_diameter: Size of central square hole (default from config)
        center_hub_radius: Radius of center hub (default from config)
        outer_ring_width: Width of outer ring (default from config)
        strut_width: Width of radial struts (default from config)
        number_struts: Number of radial struts (default from config)

    Returns:
        CadQuery Workplane containing the core plate geometry
    """
    # Use config defaults if not provided
    plate_diameter = plate_diameter or config.plate_diameter
    plate_thickness = plate_thickness or config.plate_thickness
    screw_size = screw_size or config.screw_size
    screw_size_tolerance = screw_size_tolerance or config.screw_size_tolerance
    screw_pattern_radius = screw_pattern_radius or config.screw_pattern_radius
    number_holes = number_holes or config.number_holes
    screw_square_size = screw_square_size or config.screw_square_size
    central_hole_diameter = central_hole_diameter or config.central_hole_diameter
    center_hub_radius = center_hub_radius or config.center_hub_radius
    outer_ring_width = outer_ring_width or config.outer_ring_width
    strut_width = strut_width or config.strut_width
    number_struts = number_struts or config.number_struts

    # Calculate derived parameters
    plate_radius = plate_diameter / 2
    screw_hole_diameter = screw_size + screw_size_tolerance
    outer_ring_inner_radius = plate_radius - outer_ring_width
    strut_length = outer_ring_inner_radius - center_hub_radius + 2  # Add 1mm overlap on each side
    outer_ring_middle_radius = (plate_radius + outer_ring_inner_radius) / 2

    # ═══════════════════════════════════════════════════════════════════
    # 1. CREATE CENTER HUB DISC
    # ═══════════════════════════════════════════════════════════════════

    # Create solid center hub disc
    center_hub = (
        cq.Workplane("XY")
        .circle(center_hub_radius)
        .extrude(plate_thickness)
    )

    # ═══════════════════════════════════════════════════════════════════
    # 2. CREATE OUTER RING
    # ═══════════════════════════════════════════════════════════════════

    # Create outer annular ring
    outer_ring = (
        cq.Workplane("XY")
        .circle(plate_radius)
        .circle(outer_ring_inner_radius)
        .extrude(plate_thickness)
    )

    # ═══════════════════════════════════════════════════════════════════
    # 3. CREATE RADIAL STRUTS
    # ═══════════════════════════════════════════════════════════════════

    # Start with center hub and outer ring
    plate = center_hub.union(outer_ring)

    # Add radial struts connecting hub to outer ring
    for i in range(number_struts):
        angle = (360 / number_struts) * i + 45/2  # Offset by 45 degrees
        angle_rad = math.radians(angle)

        # Calculate strut center position (midpoint between hub edge and ring inner edge)
        gap_midpoint_radius = (center_hub_radius + outer_ring_inner_radius) / 2
        strut_x = gap_midpoint_radius * math.cos(angle_rad)
        strut_y = gap_midpoint_radius * math.sin(angle_rad)

        # Create rectangular strut (aligned with X-axis initially)
        strut = (
            cq.Workplane("XY")
            .rect(strut_length, strut_width)
            .extrude(plate_thickness)
        )

        # Rotate strut to align radially, THEN translate to position
        strut = (
            strut.val()
            .rotate((0, 0, 0), (0, 0, 1), angle)
            .translate((strut_x, strut_y, 0))
        )
        strut = cq.Workplane("XY").add(strut)

        # Union strut with plate
        plate = plate.union(strut)

    # ═══════════════════════════════════════════════════════════════════
    # 4. OUTER PERIMETER DRILL HOLES
    # ═══════════════════════════════════════════════════════════════════

    # Create holes evenly distributed around the perimeter in the middle of the outer ring
    for i in range(number_holes):
        angle = (360 / number_holes) * i
        angle_rad = math.radians(angle)

        # Calculate position at the middle of the outer ring
        x_pos = outer_ring_middle_radius * math.cos(angle_rad)
        y_pos = outer_ring_middle_radius * math.sin(angle_rad)

        # Create and cut hole
        screw_hole = (
            cq.Workplane("XY")
            .workplane(offset=-plate_thickness)
            .center(x_pos, y_pos)
            .circle(screw_hole_diameter / 2)
            .extrude(plate_thickness * 3)
        )

        plate = plate.cut(screw_hole)

    # ═══════════════════════════════════════════════════════════════════
    # 5. CENTER SQUARE MOUNT HOLES
    # ═══════════════════════════════════════════════════════════════════

    # Create 4 holes at the corners of a square pattern
    half_square = screw_square_size / 2
    square_hole_positions = [
        (half_square, half_square),
        (-half_square, half_square),
        (-half_square, -half_square),
        (half_square, -half_square)
    ]

    # ═══════════════════════════════════════════════════════════════════
    # 6. CENTRAL SQUARE HOLE WITH FILLED CIRCLES AROUND SCREW HOLES
    # ═══════════════════════════════════════════════════════════════════

    # Create central square hole through the plate
    central_hole = (
        cq.Workplane("XY")
        .workplane(offset=-plate_thickness)
        .rect(central_hole_diameter, central_hole_diameter)
        .extrude(plate_thickness * 3)
    )

    plate = plate.cut(central_hole)

    # Add filled circles around the square mount screw holes that overlap the square hole
    circle_around_screw_diameter = screw_hole_diameter * 3  # 3x the screw hole diameter
    for x_pos, y_pos in square_hole_positions:
        # Create filled circle around the screw hole
        filled_circle = (
            cq.Workplane("XY")
            .center(x_pos, y_pos)
            .circle(circle_around_screw_diameter / 2)
            .extrude(plate_thickness)
        )

        plate = plate.union(filled_circle)

    # Now cut the screw holes through everything (including the filled circles)
    for x_pos, y_pos in square_hole_positions:
        # Create and cut hole
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
# SCRIPT EXECUTION (for backward compatibility)
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Create the core plate with default parameters
    plate = create_core_plate()

    # Print info
    print("Generating core plate with hub-and-spoke structure")
    print(f"  Plate diameter: {config.plate_diameter}mm")
    print(f"  Plate thickness: {config.plate_thickness}mm")
    print(f"  Center hub radius: {config.center_hub_radius}mm")
    print(f"  Outer ring width: {config.outer_ring_width}mm")
    print(f"  Number of struts: {config.number_struts}")
    print(f"  Strut width: {config.strut_width}mm")
    print(f"  Outer holes: {config.number_holes} at radius {config.screw_pattern_radius}mm")
    print(f"  Square mount: {config.screw_square_size}mm x {config.screw_square_size}mm")
    print(f"  Central square hole: {config.central_hole_diameter}mm x {config.central_hole_diameter}mm")

    # Export as STL file
    import os
    os.makedirs("stls", exist_ok=True)
    cq.exporters.export(plate, "stls/core_plate.stl")

    print("Core plate created successfully!")
    print("Saved to: stls/core_plate.stl")
