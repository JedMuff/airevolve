"""
Render multi-angle pyvista turnarounds for the top-10 hybrid figure-8 morphologies.

Reads top10_hybrid_figure8_morphologies.csv, and for each of the 10 ranked
individuals renders 8 camera angles (front/side/rear/top + 4 isometric) as
separate square PNGs, plus the original matplotlib blueprint (4-view
wireframe) for comparison, each saved into its own directory under one
parent output directory.
"""

import argparse
import ast
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import airevolve.evolution_tools.inspection_tools.utils as u
from airevolve.evolution_tools.inspection_tools.drone_visualizer import DroneVisualizer

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DEFAULT_CSV = os.path.join(REPO_ROOT, "top10_hybrid_figure8_morphologies.csv")
DEFAULT_OUTPUT_DIR = os.path.join(REPO_ROOT, "__data__", "visualizations", "top10_hybrid_figure8_pyvista")

VIEWS = [
    (0, 90, "front"),
    (0, 0, "side"),
    (0, 180, "rear"),
    (90, 0, "top"),
    (30, 45, "iso_45"),
    (30, 135, "iso_135"),
    (30, 225, "iso_225"),
    (30, 315, "iso_315"),
]

CIRCLE_RADIUS = 0.0254  # 2-inch propeller radius in meters
ARM_RADIUS = 0.004
CORE_SIZE = 0.03  # square core footprint, meters
CORE_THICKNESS = 0.01
MOTOR_RADIUS = 0.01
MOTOR_HEIGHT = 0.012
SPIN_ARC_DEG = 100  # how much of the propeller rim the spin arrow sweeps
CORE_COLOR = "blue"
ARM_COLOR = "gray"
MOTOR_COLOR = "gray"
CW_COLOR = "green"
CCW_COLOR = "red"


def load_individuals(csv_path):
    individuals = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            genome = np.array(ast.literal_eval(row["individual_data"]), dtype=float)
            name = f"rank{int(row['overall_rank']):02d}_{row['algorithm']}"
            individuals.append((name, genome))
    return individuals


def build_spin_arrow(pos, u_axis, v_axis, direction):
    """Tube + cone tracing an arc around the propeller rim, in the spin direction."""
    sign = 1.0 if direction == 1 else -1.0
    thetas = sign * np.radians(np.linspace(0.0, SPIN_ARC_DEG, 24))

    pts = pos + CIRCLE_RADIUS * (
        np.outer(np.cos(thetas), u_axis) + np.outer(np.sin(thetas), v_axis)
    )
    arc = pv.Spline(pts, len(pts)).tube(radius=ARM_RADIUS * 0.6)

    tangent = sign * (-np.sin(thetas[-1]) * u_axis + np.cos(thetas[-1]) * v_axis)
    tangent = tangent / np.linalg.norm(tangent)
    cone_height = ARM_RADIUS * 6
    tip = pv.Cone(
        center=pts[-1] + tangent * cone_height * 0.5,
        direction=tuple(tangent),
        height=cone_height,
        radius=ARM_RADIUS * 2.2,
        resolution=16,
    )
    return arc, tip


def build_drone_scene(plotter, genome_array):
    data = u.auto_extract_genome_data(genome_array)

    core = pv.Cube(
        center=(0, 0, 0),
        x_length=CORE_SIZE,
        y_length=CORE_SIZE,
        z_length=CORE_THICKNESS,
    )
    plotter.add_mesh(core, color=CORE_COLOR)

    forward_line = pv.Line(
        (0, 0, CORE_THICKNESS / 2), (CORE_SIZE / 2, 0, CORE_THICKNESS / 2)
    ).tube(radius=ARM_RADIUS * 0.5)
    plotter.add_mesh(forward_line, color="orange")

    for pos, orient, direction in zip(data["positions"], data["orientations"], data["directions"]):
        pos = np.asarray(pos, dtype=float)
        R = u.create_rotation_matrix_euler(*orient)
        u_axis, v_axis, normal = R[:, 0], R[:, 1], R[:, 2]

        arm = pv.Line((0, 0, 0), tuple(pos)).tube(radius=ARM_RADIUS)
        plotter.add_mesh(arm, color=ARM_COLOR)

        motor = pv.Cylinder(
            center=tuple(pos + normal * MOTOR_HEIGHT / 2),
            direction=tuple(normal),
            radius=MOTOR_RADIUS,
            height=MOTOR_HEIGHT,
            resolution=24,
        )
        plotter.add_mesh(motor, color=MOTOR_COLOR)

        prop_pos = pos + normal * MOTOR_HEIGHT

        spin_color = CW_COLOR if direction == 1 else CCW_COLOR

        disc = pv.Disc(center=tuple(prop_pos), inner=0, outer=CIRCLE_RADIUS, normal=tuple(normal), c_res=60)
        plotter.add_mesh(disc, color=spin_color, opacity=0.85)

        arc, tip = build_spin_arrow(prop_pos, u_axis, v_axis, direction)
        plotter.add_mesh(arc, color=spin_color)
        plotter.add_mesh(tip, color=spin_color)

        thrust_arrow = pv.Arrow(start=tuple(prop_pos), direction=tuple(normal), scale=CIRCLE_RADIUS * 2)
        plotter.add_mesh(thrust_arrow, color="black")


def set_view(plotter, elev_deg, azim_deg, center, distance):
    elev = np.radians(elev_deg)
    azim = np.radians(azim_deg)

    # Genome coordinates are NED (+Z is down), so "up" on screen is -Z.
    # Flipping the sign of the z-offset/up-vector here (rather than negating
    # z on the mesh geometry itself) keeps the drone physically right-side-up
    # without mirroring it -- a geometry-level flip would invert the
    # handedness of the propeller spin-direction arrows.
    offset = distance * np.array([
        np.cos(elev) * np.cos(azim),
        np.cos(elev) * np.sin(azim),
        -np.sin(elev),
    ])
    # Assigning the individual plotter.camera.position/.focal_point/.up
    # attributes does not propagate to the renderer; the whole tuple must
    # be assigned to plotter.camera_position at once.
    plotter.camera_position = [tuple(center + offset), tuple(center), (0.0, 0.0, -1.0)]
    plotter.reset_camera_clipping_range()


def render_individual(genome_array, out_dir, window_size):
    os.makedirs(out_dir, exist_ok=True)

    plotter = pv.Plotter(off_screen=True, window_size=(window_size, window_size))
    plotter.set_background("white", top="lightsteelblue")
    build_drone_scene(plotter, genome_array)

    bounds = np.array(plotter.bounds).reshape(3, 2)
    center = bounds.mean(axis=1)
    diagonal = np.linalg.norm(bounds[:, 1] - bounds[:, 0])
    distance = max(diagonal * 1.8, CIRCLE_RADIUS * 8)

    plotter.add_axes()
    plotter.show_bounds(
        bounds=tuple(plotter.bounds),
        grid="back",
        location="outer",
        color="black",
        # Scale with window_size so labels stay legible instead of shrinking
        # relative to the image as resolution increases (tuned at 10pt/800px).
        font_size=max(10, round(window_size / 80)),
    )
    plotter.enable_anti_aliasing()
    # SSAO/shadows fail to compile their shaders under this environment's
    # software (OSMesa) OpenGL renderer and produce no visible effect, just
    # a flood of shader errors -- skipped.

    for elev_deg, azim_deg, label in VIEWS:
        set_view(plotter, elev_deg, azim_deg, center, distance)
        # Reusing one off_screen Plotter across screenshots requires an
        # explicit render() before each screenshot, otherwise VTK reuses
        # the first frame's cached image regardless of camera changes.
        plotter.render()
        plotter.screenshot(os.path.join(out_dir, f"{label}.png"))

    plotter.close()


def render_blueprint(genome_array, out_dir, dpi=300):
    visualizer = DroneVisualizer()
    fig, _ = visualizer.plot_blueprint(genome_array)
    fig.savefig(os.path.join(out_dir, "blueprint.png"), dpi=dpi)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--window-size", type=int, default=4000)
    parser.add_argument("--blueprint-dpi", type=int, default=600)
    args = parser.parse_args()

    individuals = load_individuals(args.csv)

    for name, genome in individuals:
        out_dir = os.path.join(args.output_dir, name)
        render_individual(genome, out_dir, args.window_size)
        render_blueprint(genome, out_dir, args.blueprint_dpi)
        print(f"[{name}] saved {len(VIEWS)} pyvista views + blueprint -> {out_dir}")

    print(f"\nDone: {len(individuals)} individuals x ({len(VIEWS)} pyvista views + 1 blueprint) "
          f"= {len(individuals) * (len(VIEWS) + 1)} images in {args.output_dir}")


if __name__ == "__main__":
    main()
