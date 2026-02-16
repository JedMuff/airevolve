"""
STL Generator - Main API for Drone Phenotype Assembly

This module provides the main API for generating STL files from evolved drone genomes.
"""

import cadquery as cq
import math
import os
from pathlib import Path
from typing import Optional, Dict, List, Any
from dataclasses import dataclass

from .genome_adapter import genome_to_cad_parameters, cad_parameters_to_assembly_vector
from .core_plate import create_core_plate
from .part_generators import create_arm_assembly, create_landing_leg
from . import config


@dataclass
class STLGenerationResult:
    """Result of STL generation."""
    output_dir: Path
    core_plate_file: Optional[Path] = None
    arm_files: List[Path] = None
    assembly_file: Optional[Path] = None
    landing_leg_file: Optional[Path] = None
    step_files: List[Path] = None

    def __post_init__(self):
        if self.arm_files is None:
            self.arm_files = []
        if self.step_files is None:
            self.step_files = []


def generate_stl_files(
    genome_handler,
    output_dir: str = "./drone_stls",
    include_assembly: bool = True,
    include_landing_leg: bool = False,
    include_individual_parts: bool = True,
    include_step_files: bool = True,
    distribute_arms_evenly: bool = True,
    magnitude_to_length_scale: float = 100.0,
    part_config: Optional[Dict[str, Any]] = None
) -> STLGenerationResult:
    """
    Generate STL files for a drone from a genome handler.

    Args:
        genome_handler: SphericalAngularDroneGenomeHandler instance with evolved genome
        output_dir: Directory to save STL files (default: "./drone_stls")
        include_assembly: If True, generate full assembled drone STL (default: True)
        include_landing_leg: If True, generate landing leg STL (default: False)
        include_individual_parts: If True, generate separate STL for each arm (default: True)
        include_step_files: If True, save STEP files for CAD editing (default: True)
        distribute_arms_evenly: If True, distribute arms evenly around disc (default: True)
        magnitude_to_length_scale: Scale factor to convert genome magnitude to mm (default: 100.0)
        part_config: Optional dictionary to override default part parameters

    Returns:
        STLGenerationResult with paths to all generated files

    Example:
        >>> from airevolve.evolution_tools.genome_handlers import SphericalAngularDroneGenomeHandler
        >>> from airevolve.phenotype_assembly import generate_stl_files
        >>>
        >>> # Create evolved individual
        >>> handler = SphericalAngularDroneGenomeHandler(...)
        >>> # ... evolve ...
        >>>
        >>> # Generate STL files
        >>> result = generate_stl_files(handler, output_dir="./my_drone")
        >>> print(f"Assembly saved to: {result.assembly_file}")
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Use default config if not provided
    if part_config is None:
        part_config = {}

    # Initialize result
    result = STLGenerationResult(output_dir=output_path)

    # Convert genome to CAD parameters
    cad_params = genome_to_cad_parameters(
        genome_handler,
        magnitude_to_length_scale=magnitude_to_length_scale,
        distribute_arms_evenly=distribute_arms_evenly
    )

    if cad_params.num_arms == 0:
        print("Warning: No valid arms in genome, skipping STL generation")
        return result

    # Get assembly vector for full drone assembly
    assembly_vector = cad_parameters_to_assembly_vector(cad_params)

    print(f"Generating STL files for drone with {cad_params.num_arms} arms")
    print(f"Output directory: {output_path}")

    # Generate core plate
    print("\nGenerating core plate...")
    core_plate = create_core_plate()

    if include_individual_parts:
        core_plate_file = output_path / "core_plate.stl"
        cq.exporters.export(core_plate, str(core_plate_file))
        result.core_plate_file = core_plate_file
        print(f"  Saved: {core_plate_file.name}")

    if include_step_files:
        step_file = output_path / "core_plate.step"
        core_plate.val().exportStep(str(step_file))
        result.step_files.append(step_file)

    # Generate arms
    all_parts = [core_plate.val()]
    arm_parts_list = []

    for i, params in enumerate(assembly_vector):
        arm_tilt, arm_azimuth, motor_tilt, motor_azimuth, arm_attachment_angle, arm_length = params

        print(f"\nGenerating arm {i+1}...")
        print(f"  Attachment angle: {arm_attachment_angle:.1f}°, Length: {arm_length:.1f}mm")

        # Create arm assembly
        arm_parts = create_arm_assembly(
            arm_tilt=arm_tilt,
            arm_azimuth=arm_azimuth,
            motor_tilt=motor_tilt,
            motor_azimuth=motor_azimuth,
            arm_length=arm_length,
            **part_config
        )

        arm_parts_list.append(arm_parts)

        # Position the arm assembly parts relative to the core plate
        # (same logic as in full_drone_assembly.py)
        sphere_radius = part_config.get('sphere_radius', 12)
        clamp_inset = part_config.get('clamp_inset', 10.5)
        arm_screw_hole_inset = part_config.get('arm_screw_hole_inset', 7.5/2)
        plate_thickness = config.plate_thickness

        plate_radius = config.plate_diameter / 2
        outer_ring_width = config.outer_ring_width
        outer_ring_inner_radius = plate_radius - outer_ring_width
        outer_ring_middle_radius = (plate_radius + outer_ring_inner_radius) / 2

        plate_x_offset = sphere_radius - clamp_inset + arm_screw_hole_inset
        radial_distance = outer_ring_middle_radius + plate_x_offset

        attachment_angle_rad = math.radians(arm_attachment_angle)
        attachment_x = radial_distance * math.cos(attachment_angle_rad)
        attachment_y = radial_distance * math.sin(attachment_angle_rad)
        attachment_z = plate_thickness/2

        # Transform and collect arm parts
        arm_solids = []
        for part_name, part in arm_parts.items():
            part_solid = part.val()

            # Rotate 180° to point outward
            part_solid = part_solid.rotate((0, 0, 0), (0, 0, 1), 180)

            # Rotate by attachment angle
            part_solid = part_solid.rotate((0, 0, 0), (0, 0, 1), arm_attachment_angle)

            # Translate to position
            part_solid = part_solid.translate((attachment_x, attachment_y, attachment_z))

            arm_solids.append(part_solid)
            all_parts.append(part_solid)

        # Save individual arm if requested
        if include_individual_parts:
            arm_combined = cq.Compound.makeCompound(arm_solids)
            arm_file = output_path / f"arm_{i+1}.stl"
            cq.exporters.export(arm_combined, str(arm_file))
            result.arm_files.append(arm_file)
            print(f"  Saved: {arm_file.name}")

    # Generate full assembly
    if include_assembly:
        print("\nGenerating full assembly...")
        combined = cq.Compound.makeCompound(all_parts)
        assembly_file = output_path / "full_drone_assembly.stl"
        cq.exporters.export(combined, str(assembly_file))
        result.assembly_file = assembly_file
        print(f"  Saved: {assembly_file.name}")

        if include_step_files:
            # Create assembly with separate parts for STEP export
            assembly = cq.Assembly()
            assembly.add(core_plate, name="core_plate", color=cq.Color("gray"))

            for i, arm_parts in enumerate(arm_parts_list):
                params = assembly_vector[i]
                arm_attachment_angle = params[4]

                # Re-apply transformations for assembly
                for part_name, part in arm_parts.items():
                    part_solid = part.val()
                    part_solid = part_solid.rotate((0, 0, 0), (0, 0, 1), 180)
                    part_solid = part_solid.rotate((0, 0, 0), (0, 0, 1), arm_attachment_angle)

                    # Calculate position (same as above)
                    sphere_radius = part_config.get('sphere_radius', 12)
                    clamp_inset = part_config.get('clamp_inset', 10.5)
                    arm_screw_hole_inset = part_config.get('arm_screw_hole_inset', 7.5/2)
                    plate_x_offset = sphere_radius - clamp_inset + arm_screw_hole_inset
                    radial_distance = outer_ring_middle_radius + plate_x_offset
                    attachment_angle_rad = math.radians(arm_attachment_angle)
                    attachment_x = radial_distance * math.cos(attachment_angle_rad)
                    attachment_y = radial_distance * math.sin(attachment_angle_rad)
                    attachment_z = plate_thickness/2

                    part_solid = part_solid.translate((attachment_x, attachment_y, attachment_z))
                    part_wp = cq.Workplane("XY").add(part_solid)

                    colors = {
                        'sphere': cq.Color("lightblue"),
                        'upper_jaw': cq.Color("red"),
                        'lower_jaw': cq.Color("green"),
                        'motor_arm': cq.Color("orange")
                    }

                    assembly.add(part_wp, name=f"arm{i+1}_{part_name}", color=colors.get(part_name, cq.Color("white")))

            step_file = output_path / "full_drone_assembly.step"
            assembly.save(str(step_file))
            result.step_files.append(step_file)
            print(f"  Saved STEP: {step_file.name}")

    # Generate landing leg if requested
    if include_landing_leg:
        print("\nGenerating landing leg...")
        landing_leg = create_landing_leg()
        landing_leg_file = output_path / "landing_leg.stl"
        cq.exporters.export(landing_leg, str(landing_leg_file))
        result.landing_leg_file = landing_leg_file
        print(f"  Saved: {landing_leg_file.name}")

        if include_step_files:
            step_file = output_path / "landing_leg.step"
            landing_leg.val().exportStep(str(step_file))
            result.step_files.append(step_file)

    print("\n" + "=" * 70)
    print("STL generation completed successfully!")
    print("=" * 70)
    print(f"\nGenerated files in: {output_path}")
    if result.core_plate_file:
        print(f"  Core plate: {result.core_plate_file.name}")
    if result.arm_files:
        print(f"  Arms: {len(result.arm_files)} files")
    if result.assembly_file:
        print(f"  Assembly: {result.assembly_file.name}")
    if result.landing_leg_file:
        print(f"  Landing leg: {result.landing_leg_file.name}")
    if result.step_files:
        print(f"  STEP files: {len(result.step_files)} files")

    return result


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def quick_visualize_genome(genome_handler, output_file: str = "quick_drone.stl"):
    """
    Quickly generate a single STL file for visualization.

    Args:
        genome_handler: SphericalAngularDroneGenomeHandler instance
        output_file: Output filename (default: "quick_drone.stl")

    Returns:
        Path to generated STL file
    """
    result = generate_stl_files(
        genome_handler,
        output_dir=os.path.dirname(output_file) or ".",
        include_assembly=True,
        include_individual_parts=False,
        include_step_files=False,
        include_landing_leg=False
    )

    # Rename assembly file to requested name
    if result.assembly_file:
        output_path = Path(output_file)
        result.assembly_file.rename(output_path)
        return output_path

    return None
