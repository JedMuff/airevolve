"""
Assembly vs Visualizer Comparison

Renders the DroneVisualizer blueprint (simulation space) alongside a
parameter table showing what genome_to_cad_parameters() produces for the
same genome, so you can verify the assembly matches the sim view before
opening the STL.

Usage
-----
    python examples/test_assembly_vs_visualizer.py

    # Skip STL generation (matplotlib only, no cadquery needed):
    python examples/test_assembly_vs_visualizer.py --no-stl

Output
------
    - STL + STEP files saved to ./assembly_comparison/  (unless --no-stl)
    - Matplotlib figure shown after STL generation and saved as comparison.png

What to check
-------------
  attachment_angle  == arm_rotation in the blueprint top-down view (azimuth)
  arm_elevation     == arm_pitch (elevation from XY plane) in the side/3D view
  arm_length        == magnitude×100 − attach_radius×cos(pitch)
                       (tube length so motor tip matches simulation position)
  motor_tilt        == motor_pitch (elevation) shown by the orientation arrow
"""

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from airevolve.evolution_tools.genome_handlers.spherical_angular_genome_handler import (
    SphericalAngularDroneGenomeHandler,
)
from airevolve.evolution_tools.inspection_tools.drone_visualizer import (
    DroneVisualizer,
    VisualizationConfig,
)
from airevolve.phenotype_assembly import genome_to_cad_parameters, AssemblyConfig

# ─────────────────────────────────────────────────────────────────────────────
# Option A: fixed quadcopter genome (uncomment to use instead of random)
#
# SphericalAngular format per row:
#   [magnitude, arm_rotation(θ), arm_pitch(φ), motor_rotation, motor_pitch, direction]
#
# Standard flat quadcopter:
#   arm_pitch = 0    → arms lie in the horizontal plane (elevation = 0°)
#   motor_pitch = 0  → motors face straight up
#   arm_rotation at 0°, 60°, 180°, 270°
# ─────────────────────────────────────────────────────────────────────────────
ARM_LENGTH_MAGNITUDE = 0.055   # × 100 mm/unit = 60 mm motor distance
QUADCOPTER_GENOME = np.array([
    #  mag                    θ (arm azimuth)    φ (arm pitch)  motor_rot  motor_pitch  dir
    [ARM_LENGTH_MAGNITUDE,  0.0,                 -np.pi/4,             np.pi/4,       np.pi/4,         1],  # arm 1 →   0°
    [ARM_LENGTH_MAGNITUDE,  np.pi / 3,          0,    0.0,       0,           0],  # arm 2 →  60° (drooping)
    [ARM_LENGTH_MAGNITUDE,  np.pi,               0,             0.0,       0.0,         1],  # arm 3 → 180°
    [ARM_LENGTH_MAGNITUDE,  3 * np.pi / 2,       0,             0.0,       0.0,         0],  # arm 4 → 270°
])


def build_random_handler(min_arms: int = 3, max_arms: int = 6) -> SphericalAngularDroneGenomeHandler:
    """Generate a random drone genome."""
    spherical_params = np.array([[0.055,0.11+0.06], [-np.pi, np.pi], [-np.pi/2, np.pi/2], [-np.pi, np.pi], [-np.pi/2, np.pi/2], [0,1]])

    handler = SphericalAngularDroneGenomeHandler(
        min_max_narms=(min_arms, max_arms),
        bilateral_plane_for_symmetry=None,
        repair=False,
        enable_collision_repair=False,
        parameter_limits=spherical_params,
    )
    population = handler.random_population(1)
    handler.genome = population[0]
    return handler


def build_handler(genome: np.ndarray) -> SphericalAngularDroneGenomeHandler:
    """Build a handler from a fixed genome array."""
    handler = SphericalAngularDroneGenomeHandler(
        min_max_narms=(4, 4),
        bilateral_plane_for_symmetry=None,
        repair=False,
    )
    handler.genome = genome
    return handler


def make_param_table(cad_params) -> tuple:
    """Return (col_labels, row_labels, cell_data) for a matplotlib table."""
    col_labels = ["mount_angle (°)", "arm_azimuth (°)", "arm_elevation (°)",
                  "arm_length (mm)", "motor_tilt local (°)", "direction"]
    rows = []
    row_labels = []
    for i, arm in enumerate(cad_params.arms):
        rows.append([
            f"{arm.mount_angle:.1f}",
            f"{arm.attachment_angle:.1f}",
            f"{arm.arm_elevation:.1f}",
            f"{arm.arm_length:.1f}",
            f"{arm.motor_tilt:.1f}",
            "CW" if arm.direction else "CCW",
        ])
        row_labels.append(f"Arm {i + 1}")
    return col_labels, row_labels, rows


def main(generate_stl: bool = True, output_dir: str = "./assembly_comparison"):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Build handler (random genome) ─────────────────────────────────────────
    # To use a fixed genome instead, comment the line below and uncomment the
    # build_handler(QUADCOPTER_GENOME) call (and the definitions above).
    handler = build_random_handler(min_arms=4, max_arms=6)
    # handler = build_handler(QUADCOPTER_GENOME)

    num_arms = len(handler.get_valid_arms())
    print(f"Generated random drone with {num_arms} arms")

    cfg = AssemblyConfig()
    cad_params = genome_to_cad_parameters(
        handler,
        magnitude_to_length_scale=100.0,
        assembly_config=cfg,
        snap_mounts=True,
        num_mount_positions=8,
    )

    # ── Figure layout ─────────────────────────────────────────────────────────
    # Row 0: DroneVisualizer blueprint (4 views)
    # Row 1: parameter comparison table
    fig = plt.figure(figsize=(18, 11))
    fig.suptitle(f"Assembly vs Visualizer — {num_arms}-arm random drone", fontsize=14, fontweight="bold")

    gs = gridspec.GridSpec(
        2, 4,
        figure=fig,
        height_ratios=[2.5, 1],
        hspace=0.35,
        wspace=0.3,
    )

    # ── DroneVisualizer blueprint ─────────────────────────────────────────────
    viz_config = VisualizationConfig(
        include_motor_orientation=True,
        include_motor_orientation_2d=1,  # show orientation arrows in 2D
        motor_color="blue",
        orientation_color="red",
        circle_radius=0.0508/2,             # ~2-inch prop radius in metres
    )
    visualizer = DroneVisualizer(config=viz_config)

    # Top-down (2D)
    ax_top = fig.add_subplot(gs[0, 0])
    visualizer.plot_2d(handler, ax=ax_top, title="Top-down (sim space)")

    # 3D isometric
    ax_3d = fig.add_subplot(gs[0, 1], projection="3d")
    visualizer.plot_3d(handler, ax=ax_3d, title="3D isometric", elevation=30, azimuth=45)

    # Front view
    ax_front = fig.add_subplot(gs[0, 2], projection="3d")
    visualizer.plot_3d(handler, ax=ax_front, title="Front view", elevation=0, azimuth=0)

    # Side view
    ax_side = fig.add_subplot(gs[0, 3], projection="3d")
    visualizer.plot_3d(handler, ax=ax_side, title="Side view", elevation=0, azimuth=90)

    # ── Parameter table ───────────────────────────────────────────────────────
    ax_table = fig.add_subplot(gs[1, :])
    ax_table.axis("off")

    col_labels, _, cell_data = make_param_table(cad_params)

    # Add a header row showing the genome values alongside CAD values
    genome_rows = []
    attach_r = cfg.arm_attach_radial_distance
    mount_z = cfg.plate_thickness / 2.0
    sphere_offset = cfg.sphere_offset
    step_deg = 360.0 / 8  # 8 mount positions
    for i, row in enumerate(handler.get_valid_arms()):
        mag, theta, phi, m_rot, m_pitch, direction = row
        motor_dist = mag * 1000.0
        # Snapped mount angle
        snapped_mount = round(float(np.degrees(theta)) / step_deg) * step_deg % 360.0
        # Motor tip
        mx = motor_dist * np.cos(float(phi)) * np.cos(float(theta))
        my = motor_dist * np.cos(float(phi)) * np.sin(float(theta))
        mz = motor_dist * np.sin(float(phi))
        # Vector from snapped mount to motor tip
        snap_rad = np.radians(snapped_mount)
        vx = mx - attach_r * np.cos(snap_rad)
        vy = my - attach_r * np.sin(snap_rad)
        vz = mz - mount_z
        arm_azimuth_deg = float(np.degrees(np.arctan2(vy, vx))) % 360.0
        arm_elev = float(np.degrees(np.arctan2(vz, np.sqrt(vx**2 + vy**2))))
        tube_len = max(np.sqrt(vx**2 + vy**2 + vz**2) - sphere_offset, 1.0)
        motor_tilt_local = (arm_elev - 90.0) + float(np.degrees(m_pitch))
        genome_rows.append([
            f"{snapped_mount:.1f}°",
            f"{arm_azimuth_deg:.1f}°",
            f"{arm_elev:.1f}°",
            f"{tube_len:.1f} mm",
            f"{motor_tilt_local:.1f}°",
            "CW" if int(direction) else "CCW",
        ])

    # Interleave rows: genome row (grey) then CAD row (white) per arm
    table_cells = []
    table_row_labels = []
    row_colors = []
    for i in range(len(cad_params.arms)):
        table_cells.append(genome_rows[i])
        table_row_labels.append(f"  Arm {i + 1}  genome →")
        row_colors.append(["#e8e8e8"] * len(col_labels))

        table_cells.append(cell_data[i])
        table_row_labels.append(f"  Arm {i + 1}  CAD    →")
        row_colors.append(["#ffffff"] * len(col_labels))

    tbl = ax_table.table(
        cellText=table_cells,
        rowLabels=table_row_labels,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        cellColours=row_colors,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.4)

    # Style header
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#3a7ebf")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    ax_table.set_title(
        "Genome values  vs  CAD assembly parameters  "
        "(mount snapped to nearest 45°;  arm_azimuth / elevation / length "
        "recomputed from snapped-mount → motor-tip vector)",
        fontsize=9, pad=8,
    )

    fig_path = out / "comparison.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"Saved comparison figure: {fig_path}")

    # ── STL generation ────────────────────────────────────────────────────────
    if generate_stl:
        print("\nGenerating STL files…")
        try:
            from airevolve.phenotype_assembly import generate_stl_files

            result = generate_stl_files(
                genome_handler=handler,
                output_dir=str(out),
                include_assembly=True,
                include_individual_parts=True,
                include_motor_mounts=True,
                include_arm_mounts=True,
                include_step_files=True,
                distribute_arms_evenly=False,
                magnitude_to_length_scale=1000.0,
                assembly_config=cfg,
                snap_mounts=True,
                num_mount_positions=8,
            )
            print(f"\nGenerated files summary:")
            print(f"  Assembly: {result.assembly_file.name if result.assembly_file else 'None'}")
            print(f"  Core plate: {result.core_plate_file.name if result.core_plate_file else 'None'}")
            print(f"  Arms (integrated): {len(result.arm_files)} files")
            print(f"  Motor mounts (separate): {len(result.motor_mount_files)} files")
            print(f"  Arm mounts (separate): {len(result.arm_mount_files)} files")
            print(f"  STEP files: {len(result.step_files)} files")

            print(f"\nOpen the assembly to compare against the blueprint above:")
            print(f"  STL : {result.assembly_file}")
            if result.step_files:
                step = next((f for f in result.step_files if "assembly" in f.name), None)
                if step:
                    print(f"  STEP: {step}")
        except ImportError:
            print("cadquery not available — skipping STL generation.")
        except Exception as exc:
            print(f"STL generation failed: {exc}")
    else:
        print("\nSkipped STL generation (--no-stl).")

    print("\nWhat to verify:")
    print("  attach_angle  matches the arm azimuth in the top-down blueprint")
    print("  arm_elevation matches arm_pitch (elevation) in the front / side views")
    print("  arm_length    = magnitude×100 − attach_radius×cos(pitch)")
    print("  motor_tilt    matches motor_pitch (elevation), shown by orientation arrow")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare assembly parameters with DroneVisualizer")
    parser.add_argument("--no-stl", action="store_true", help="Skip STL/STEP generation")
    parser.add_argument("--output-dir", default="./assembly_comparison", help="Output directory")
    args = parser.parse_args()

    main(generate_stl=not args.no_stl, output_dir=args.output_dir)
