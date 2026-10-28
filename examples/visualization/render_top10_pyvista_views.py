"""
Render multi-angle pyvista turnarounds for the top-10 hybrid figure-8 morphologies.

Reads top10_hybrid_figure8_morphologies.csv, and for each of the 10 ranked
individuals renders 8 camera angles (front/side/rear/top + 4 isometric) as
separate square PNGs, plus the original matplotlib blueprint (4-view
wireframe) for comparison, each saved into its own directory under one
parent output directory.

Also writes hover_summary.png per individual: four of the views next to six
horizontal bar charts (Fx, Fy, Fz, Mx, My, Mz in the body frame) showing how
much each motor contributes at a steady level hover, with a Σ bar underneath
showing that the contributions (plus gravity) sum to zero.
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
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from scipy.optimize import least_squares

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import airevolve.evolution_tools.inspection_tools.utils as u
from airevolve.evolution_tools.inspection_tools.drone_visualizer import DroneVisualizer
from airevolve.simulator.simulation import DroneSimulator, genome_to_propellers

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

# Hover summary figure
SUMMARY_VIEWS = ["front", "side", "top", "iso_45"]
GRAVITY_COLOR = "gray"
MOMENT_ARM_SCALE = 0.1  # m; weights N·m against N when solving for hover
FORCE_TOL = 1e-4  # N; |Σ| below this counts as balanced
MOMENT_TOL = 1e-6  # N·m
# (column index into contributions, title, unit scale, unit label)
DOF_PANELS = [
    (0, "Fx (forward)", 1.0, "N"),
    (1, "Fy (right)", 1.0, "N"),
    (2, "Fz (down)", 1.0, "N"),
    (3, "Mx (roll)", 1e3, "mN·m"),
    (4, "My (pitch)", 1e3, "mN·m"),
    (5, "Mz (yaw)", 1e3, "mN·m"),
]


def load_individuals(csv_path):
    individuals = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            genome = np.array(ast.literal_eval(row["individual_data"]), dtype=float)
            name = f"rank{int(row['overall_rank']):02d}_{row['algorithm']}"
            individuals.append((name, genome, float(row["fitness"])))
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


def build_drone_scene(plotter, genome_array, label_font_size=None):
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

    label_points = []
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

        # Motor number label just outboard of the propeller, clear of the thrust arrow.
        label_points.append(prop_pos + pos / np.linalg.norm(pos) * CIRCLE_RADIUS * 1.4)

    if label_font_size:
        # Numbering follows genome row order, matching M1..Mn in hover_summary.png.
        plotter.add_point_labels(
            np.array(label_points),
            [f"M{i + 1}" for i in range(len(label_points))],
            font_size=label_font_size,
            bold=True,
            text_color="black",
            shape_color="white",
            shape_opacity=0.8,
            show_points=False,
            always_visible=True,
        )


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
    build_drone_scene(plotter, genome_array, label_font_size=max(12, round(window_size / 20)))

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


def level_hover_contributions(genome_array):
    """Per-motor body-frame force/moment contributions at a steady level hover.

    Uses the simulator's own (tilt-aware) coefficients converted back to
    physical units, at steady motor speed (dW = 0), and solves for the motor
    speeds that best cancel gravity with zero net moment while level.

    Returns:
        contributions: (n, 6) array of [Fx, Fy, Fz, Mx, My, Mz] per motor (N, N·m)
        gravity: (6,) gravity wrench in the body frame (N, N·m)
        throttle: (n,) motor speed as a fraction of w_max
        at_idle: (n,) True where a motor sits at its minimum speed
        spins: list of "cw"/"ccw"
    """
    propellers = genome_to_propellers(genome_array)
    sim = DroneSimulator(propellers=propellers)
    p = sim.params
    inertia_diag = np.diag(sim.inertia)
    f_coef = np.array(p["f_coef"]) * sim.mass
    m_coef = np.array(p["m_coef"]) * inertia_diag

    def contributions(W):
        W = W[:, None]
        return np.hstack([f_coef * W**2, m_coef * W**2])

    gravity = np.array([0.0, 0.0, sim.mass * sim.g, 0.0, 0.0, 0.0])

    def residual(W):
        total = contributions(W).sum(axis=0) + gravity
        return np.hstack([total[:3], total[3:] / MOMENT_ARM_SCALE])

    W0 = np.full(sim.num_motors, np.sqrt(gravity[2] / sim.num_motors / -f_coef[0, 2]))
    W0 = np.clip(W0, p["w_min"], p["w_max"])
    sol = least_squares(
        residual, W0, bounds=(p["w_min"], p["w_max"]),
        xtol=1e-15, ftol=1e-15, gtol=1e-15,
    )
    W = sol.x
    return (
        contributions(W),
        gravity,
        W / p["w_max"],
        np.isclose(W, p["w_min"], rtol=0, atol=1e-6 * p["w_max"]),
        [prop["dir"][3] for prop in propellers],
    )


def plot_dof_panel(ax, values, gravity, total, labels, colors, title, unit, tol):
    """Horizontal bars per motor (+ gravity), then a separated Σ bar."""
    show_gravity = gravity != 0.0
    rows = list(values) + ([gravity] if show_gravity else [])
    row_labels = list(labels) + (["gravity"] if show_gravity else [])
    row_colors = list(colors) + ([GRAVITY_COLOR] if show_gravity else [])
    y = np.arange(len(rows))
    sum_y = len(rows) + 0.6
    balanced = abs(total) < tol

    ax.barh(y, rows, color=row_colors, edgecolor="black", linewidth=0.5)
    ax.barh(sum_y, total, color="black" if balanced else "red", edgecolor="black", linewidth=0.5)
    ax.axhline(len(rows) - 0.2, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)

    limit = max(np.max(np.abs(rows)), abs(total), 1e-12) * 1.25
    ax.set_xlim(-limit, limit)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, symmetric=True))
    ax.set_yticks(list(y) + [sum_y])
    ax.set_yticklabels(row_labels + ["Σ"])
    ax.invert_yaxis()
    ax.set_title(f"{title} [{unit}]")
    ax.grid(axis="x", alpha=0.3)

    sum_text = "Σ = 0" if balanced else f"Σ = {total:+.3g} {unit}"
    # Place the label on the opposite side from the Σ bar so it stays legible.
    text_on_left = total > 0 and not balanced
    ax.text(
        0.02 if text_on_left else 0.98, sum_y, sum_text, transform=ax.get_yaxis_transform(),
        ha="left" if text_on_left else "right", va="center", fontsize=10,
        color="black" if balanced else "red", fontweight="bold",
    )


def render_hover_summary(genome_array, out_dir, name, fitness):
    contributions, gravity, throttle, at_idle, spins = level_hover_contributions(genome_array)
    totals = contributions.sum(axis=0) + gravity
    tols = [FORCE_TOL] * 3 + [MOMENT_TOL] * 3
    balanced = all(abs(t) < tol for t, tol in zip(totals, tols))

    labels = [
        f"M{i + 1} · {'idle' if idle else f'{thr:.0%}'}"
        for i, (thr, idle) in enumerate(zip(throttle, at_idle))
    ]
    colors = [CW_COLOR if spin == "cw" else CCW_COLOR for spin in spins]

    fig = plt.figure(figsize=(25, 10))
    gs = GridSpec(2, 5, figure=fig, wspace=0.35, hspace=0.3)

    for idx, view in enumerate(SUMMARY_VIEWS):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        ax.imshow(plt.imread(os.path.join(out_dir, f"{view}.png")))
        ax.set_title(view, fontsize=11)
        ax.axis("off")

    for k, (col, title, scale, unit) in enumerate(DOF_PANELS):
        ax = fig.add_subplot(gs[k // 3, 2 + k % 3])
        plot_dof_panel(
            ax,
            contributions[:, col] * scale,
            gravity[col] * scale,
            totals[col] * scale,
            labels,
            colors,
            title,
            unit,
            tols[col] * scale,
        )

    rank, algorithm = name.split("_", 1)
    spin_pattern = "".join("+" if spin == "cw" else "−" for spin in spins)
    title = (
        f"#{int(rank[4:])} {algorithm}  ·  fitness {fitness:.2f}  ·  spins {spin_pattern} (+ cw, − ccw)"
        f"  ·  level hover, body frame (NED)"
    )
    if not balanced:
        title += "  ·  cannot balance level hover (red Σ)"
    fig.suptitle(title, fontsize=15)
    fig.legend(
        handles=[
            Patch(color=CW_COLOR, label="cw motor"),
            Patch(color=CCW_COLOR, label="ccw motor"),
            Patch(color=GRAVITY_COLOR, label="gravity"),
        ],
        loc="lower center", ncol=3, frameon=False, fontsize=11,
    )

    path = os.path.join(out_dir, "hover_summary.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path, balanced


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--window-size", type=int, default=4000)
    parser.add_argument("--blueprint-dpi", type=int, default=600)
    parser.add_argument("--summary-only", action="store_true",
                        help="only (re)build hover_summary.png from existing view renders")
    args = parser.parse_args()

    individuals = load_individuals(args.csv)

    for name, genome, fitness in individuals:
        out_dir = os.path.join(args.output_dir, name)
        if not args.summary_only:
            render_individual(genome, out_dir, args.window_size)
            render_blueprint(genome, out_dir, args.blueprint_dpi)
            print(f"[{name}] saved {len(VIEWS)} pyvista views + blueprint -> {out_dir}")
        path, balanced = render_hover_summary(genome, out_dir, name, fitness)
        print(f"[{name}] hover summary -> {path}" + ("" if balanced else "  (cannot balance level hover)"))

    if args.summary_only:
        return

    print(f"\nDone: {len(individuals)} individuals x ({len(VIEWS)} pyvista views + 1 blueprint) "
          f"= {len(individuals) * (len(VIEWS) + 1)} images in {args.output_dir}")


if __name__ == "__main__":
    main()
