"""
Interactive check of the genome → Fusion 360 arm parameters.

Edit a genome with sliders and see the drone drawn from the CAD parameters
(slot, arm_yaw_offset, arm_pitch, arm_length, arm_roll, motor_pitch) produced
by ``airevolve.fabrication.fusion_params``. Everything you edit (the genome)
is in the teal box; everything computed for Fusion (the CAD parameters) is in
purple — the table, and the labels on the selected arm in the 3D view. The simulator's own motor positions
and thrust directions (``genome_to_propellers``, converted back from NED) are
overlaid as black ghosts — if the maths is right, the CAD-drawn motors sit
exactly on them and the error readout stays at ~0.

Drawing frame = CAD frame: plate in XY, Z up, front of the drone = +Y
(green arrow). The discrete attachment slots are drawn on the plate rim
(hollow = free, filled = used, labelled ``#slot angle°``); each arm starts at
its slot. Mounting-hole pairs are drawn as purple dot pairs: one across the
arm root (its angle is ``arm_roll``, 0 = vertical) and one on the motor mount,
always across the arm: the motor pitches about the axis through these holes
by ``motor_pitch`` (0 = perpendicular to the arm). Each motor is drawn as a black shaft with its prop disc on
the side the motor faces. Discs are coloured by the motor's spin as seen
from its shaft end (blue = ccw, red = cw) and carry an arc arrow showing it.
The straight coloured arrow is the thrust. Motors whose thrust points back
along the arm are mounted flipped (pusher, spin reversed) so the prop is
clear of the arm; their thrust arrow then points out of the motor's back.

The left panel shows the same genome drawn by the project's usual
``DroneVisualizer.plot_3d`` for comparison (red line = thrust, magenta arc =
spin). Rotating either 3D panel rotates both.

Usage:
    python examples/visualization/fusion_params_viewer.py
    python examples/visualization/fusion_params_viewer.py --genome individual.npy --csv arms.csv
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, RadioButtons, Slider
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from airevolve.fabrication.fusion_params import (
    DEFAULT_N_SLOTS,
    DEFAULT_PLATE_RADIUS_MM,
    DEFAULT_SLOT_OFFSET_DEG,
    arm_geometry_in_g,
    format_table,
    genome_to_fusion_params,
    hole_pair_directions,
    slot_angles,
    to_csv,
)
from airevolve.simulator.simulation.propeller_data import genome_to_propellers

# The usual visualizer pulls in the genome-handler stack (which needs fcl);
# without it the viewer still runs, just without the comparison panel.
try:
    from airevolve.evolution_tools.inspection_tools.drone_visualizer import DroneVisualizer
    _DRONE_VISUALIZER_ERROR = None
except ImportError as exc:
    DroneVisualizer = None
    _DRONE_VISUALIZER_ERROR = exc

# Flat X-quad, motors thrusting up, alternating spin.
DEFAULT_GENOME = np.array([
    [0.1, np.pi / 4 + k * np.pi / 2, 0.0, np.pi, 0.0, k % 2] for k in range(4)
])

# Default drawn prop radius: what the evolution repair and DroneVisualizer
# use. A real 2" (diameter) prop is 12.7 mm — pass --prop-radius 12.7.
DEFAULT_PROP_RADIUS_MM = 25.4
ARROW_MM = 35.0
PROP_OFFSET_MM = 10.0   # motor centre → prop plane, along the shaft
SPIN_ARC_DEG = 120.0
SPIN_COLOURS = {"ccw": "tab:blue", "cw": "tab:red"}
HOLE_SPACING_MM = 8.0   # drawn spacing of each mounting-hole pair
ROOT_HOLES_MM = 6.0     # slot pivot → arm-root hole pair, along the arm
GENOME_COLOUR = "darkcyan"     # what the sliders edit
CAD_COLOUR = "tab:purple"      # what is computed for Fusion
HOLE_COLOUR = CAD_COLOUR

# (label, genome column, min, max) — angles shown in degrees.
SLIDERS = [
    ("magnitude [m]", 0, 0.03, 0.25),
    ("arm_yaw [°]", 1, -180.0, 180.0),
    ("arm_pitch [°]", 2, -90.0, 90.0),
    ("motor_pitch [°]", 3, -180.0, 180.0),
    ("motor_yaw [°]", 4, -180.0, 180.0),
]


def ned_to_g(v):
    v = np.asarray(v, dtype=float)
    return np.array([v[1], v[0], -v[2]])


def disc_basis(normal):
    """Unit e1, e2 spanning the disc with e1 × e2 = normal, so increasing
    angle runs counter-clockwise about the normal."""
    normal = normal / np.linalg.norm(normal)
    helper = np.array([1.0, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(normal, helper)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(normal, e1)


def disc_points(centre, normal, radius, n=40):
    e1, e2 = disc_basis(normal)
    t = np.linspace(0, 2 * np.pi, n)
    return centre + radius * (np.outer(np.cos(t), e1) + np.outer(np.sin(t), e2))


def draw_spin_arc(ax, centre, facing, radius, spin, colour, linewidth):
    """Arc arrow on the prop disc: ccw = counter-clockwise about the thrust."""
    e1, e2 = disc_basis(facing)
    sign = 1.0 if spin == "ccw" else -1.0
    t = sign * np.radians(np.linspace(0.0, SPIN_ARC_DEG, 30))
    arc = centre + radius * (np.outer(np.cos(t), e1) + np.outer(np.sin(t), e2))
    ax.plot(*arc.T, color=colour, linewidth=linewidth)
    tangent = sign * (-np.sin(t[-1]) * e1 + np.cos(t[-1]) * e2)
    ax.quiver(*arc[-1], *(0.35 * radius * tangent), color=colour,
              linewidth=linewidth, arrow_length_ratio=0.6)


class Viewer:
    def __init__(self, genome, plate_radius_mm, csv_path=None,
                 prop_radius_mm=DEFAULT_PROP_RADIUS_MM, n_slots=DEFAULT_N_SLOTS,
                 slot_offset_deg=DEFAULT_SLOT_OFFSET_DEG):
        self.prop_radius = prop_radius_mm
        self.n_slots = n_slots
        self.slot_offset = slot_offset_deg
        self.genome = genome.copy()
        self.csv_path = csv_path
        self.selected = 0
        self._syncing = False

        self.fig = plt.figure(figsize=(20, 9))
        # matplotlib holds widget callbacks weakly: without this reference the
        # Viewer (and its sliders) is garbage-collected after the first redraw
        # and the sliders stop responding.
        self.fig._fusion_viewer = self
        self.normal_ax = self.fig.add_axes([0.0, 0.05, 0.33, 0.9], projection="3d")
        self.ax = self.fig.add_axes([0.32, 0.05, 0.38, 0.92], projection="3d")
        self.text_ax = self.fig.add_axes([0.715, 0.48, 0.275, 0.48])
        self.text_ax.axis("off")
        # Output (CAD) and input (genome) sections, boxed in their colours.
        self._section(0.71, 0.475, 0.98, CAD_COLOUR,
                      "CAD PARAMETERS — computed, for Fusion (read only)")
        self._section(0.71, 0.155, 0.465, GENOME_COLOUR,
                      "GENOME — edit with the sliders (selected arm)")
        self.normal_visualizer = DroneVisualizer() if DroneVisualizer else None
        self.fig.canvas.mpl_connect("motion_notify_event", self._sync_views)

        x0, w = 0.78, 0.18
        self.arm_slider = Slider(self.fig.add_axes([x0, 0.40, w, 0.025]), "arm #",
                                 0, len(self.genome) - 1, valinit=0, valstep=1,
                                 color=GENOME_COLOUR)
        self.param_sliders = []
        for i, (label, col, lo, hi) in enumerate(SLIDERS):
            s = Slider(self.fig.add_axes([x0, 0.35 - 0.045 * i, w, 0.025]), label, lo, hi,
                       valinit=self._genome_value(0, col), color=GENOME_COLOUR)
            s.on_changed(lambda val, col=col: self._on_param(col, val))
            self.param_sliders.append(s)
        self.radius_slider = Slider(self.fig.add_axes([x0, 0.10, w, 0.025]), "plate R [mm]",
                                    5.0, 80.0, valinit=plate_radius_mm)
        self.spin_radio = RadioButtons(self.fig.add_axes([0.90, 0.015, 0.06, 0.075]), ("ccw", "cw"),
                                       active=int(self.genome[0, 5] > 0.5))
        self.csv_button = Button(self.fig.add_axes([x0, 0.03, 0.1, 0.045]), "print / save CSV")

        self.arm_slider.on_changed(self._on_select)
        self.radius_slider.on_changed(lambda _: self.redraw())
        self.spin_radio.on_clicked(self._on_spin)
        self.csv_button.on_clicked(self._on_csv)
        self.redraw()

    def _section(self, x0, y0, y1, colour, title):
        self.fig.patches.append(Rectangle(
            (x0, y0), 0.28, y1 - y0, transform=self.fig.transFigure, facecolor=colour,
            alpha=0.07, edgecolor=colour, linewidth=1.5, zorder=-1))
        self.fig.text(x0 + 0.005, y1 - 0.005, title, color=colour, fontsize=10,
                      fontweight="bold", va="top")

    # ── slider callbacks ────────────────────────────────────────────────────
    def _genome_value(self, arm, col):
        v = self.genome[arm, col]
        return v if col == 0 else np.degrees(v)

    def _on_select(self, val):
        self.selected = int(val)
        self._syncing = True
        for s, (_, col, lo, hi) in zip(self.param_sliders, SLIDERS):
            v = self._genome_value(self.selected, col)
            if col != 0:
                v = (v + 180.0) % 360.0 - 180.0
            s.set_val(np.clip(v, lo, hi))
        self.spin_radio.set_active(int(self.genome[self.selected, 5] > 0.5))
        self._syncing = False
        self.redraw()

    def _on_param(self, col, val):
        if self._syncing:
            return
        self.genome[self.selected, col] = val if col == 0 else np.radians(val)
        self.redraw()

    def _on_spin(self, label):
        if self._syncing:
            return
        self.genome[self.selected, 5] = 1.0 if label == "cw" else 0.0
        self.redraw()

    def _fusion_params(self):
        return genome_to_fusion_params(self.genome, self.radius_slider.val,
                                       self.n_slots, self.slot_offset)

    def _on_csv(self, _event):
        arms = self._fusion_params()
        print(format_table(arms))
        if self.csv_path:
            to_csv(arms, self.csv_path)
            print(f"saved {self.csv_path}")

    def _sync_views(self, event):
        # While dragging one 3D panel, give the other the same camera angle.
        if event.button is None or event.inaxes not in (self.ax, self.normal_ax):
            return
        other = self.normal_ax if event.inaxes is self.ax else self.ax
        other.view_init(event.inaxes.elev, event.inaxes.azim)
        self.fig.canvas.draw_idle()

    # ── drawing ─────────────────────────────────────────────────────────────
    def redraw(self):
        R = self.radius_slider.val
        arms = self._fusion_params()
        sim_props = genome_to_propellers(self.genome)

        ax = self.ax
        elev, azim = ax.elev, ax.azim
        ax.cla()
        ax.view_init(elev, azim)

        plate = disc_points(np.zeros(3), np.array([0, 0, 1.0]), R, n=60)
        ax.add_collection3d(Poly3DCollection([plate], facecolor="lightgrey",
                                             edgecolor="grey", alpha=0.6))
        ax.quiver(0, 0, 0, 0, R * 1.2, 0, color="tab:green", linewidth=2.5)
        ax.text(0, R * 1.3, 0, "front (+Y)", color="tab:green")
        self._draw_slots(ax, R, {arm.slot for arm in arms})

        pos_err, ang_err = 0.0, 0.0
        # Limits depend only on the largest arm magnitude (and plate size), so
        # rotating arms or motors never rescales the view.
        extent = max(R, 1000.0 * self.genome[:, 0].max()) + self.prop_radius
        for i, (arm, prop) in enumerate(zip(arms, sim_props)):
            pivot, motor, shaft, thrust = arm_geometry_in_g(arm, R)
            colour = SPIN_COLOURS[arm.spin]
            lw = 4 if i == self.selected else 2
            ax.plot(*np.c_[pivot, motor], color="dimgrey" if not arm.warnings else "orange",
                    linewidth=lw)
            self._draw_hole_pairs(ax, arm, pivot, motor)
            if i == self.selected:
                self._label_cad_values(ax, arm, pivot, motor)
            # Motor body/shaft, with the prop on the shaft side.
            prop_centre = motor + PROP_OFFSET_MM * shaft
            ax.plot(*np.c_[motor, prop_centre], color="k", linewidth=lw + 2)
            disc = disc_points(prop_centre, shaft, self.prop_radius)
            ax.add_collection3d(Poly3DCollection([disc], facecolor=colour, edgecolor=colour,
                                                 alpha=0.35 if i == self.selected else 0.18))
            draw_spin_arc(ax, prop_centre, shaft, self.prop_radius, arm.spin, colour, lw)
            ax.quiver(*motor, *(ARROW_MM * thrust), color=colour, linewidth=lw)
            label = f"{i} (flipped)" if arm.motor_flipped else str(i)
            ax.text(*(prop_centre + 6 * shaft), label, fontsize=9)

            sim_loc = 1000.0 * ned_to_g(prop["loc"])
            sim_dir = ned_to_g(prop["dir"][:3])
            ax.scatter(*sim_loc, color="k", marker="x", s=50)
            ax.quiver(*sim_loc, *(ARROW_MM * sim_dir), color="k", linestyle="--",
                      linewidth=0.8, arrow_length_ratio=0.15)

            pos_err = max(pos_err, np.linalg.norm(motor - sim_loc))
            ang_err = max(ang_err, np.degrees(np.arctan2(np.linalg.norm(np.cross(thrust, sim_dir)),
                                                         thrust @ sim_dir)))

        ax.set_xlim(-extent, extent)
        ax.set_ylim(-extent, extent)
        ax.set_zlim(-extent, extent)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlabel("X right [mm]")
        ax.set_ylabel("Y front [mm]")
        ax.set_zlabel("Z up [mm]")
        ax.set_title("Drawn from the CAD parameters (purple labels = selected arm)\n"
                     "vs simulator from the genome (black ×, dashed)")

        self._draw_normal(extent / 1000.0, elev, azim)

        self.text_ax.cla()
        self.text_ax.axis("off")
        self.text_ax.text(
            0, 0.93, format_table(arms)
            + f"\n\nmax |CAD − sim| position: {pos_err:.2e} mm"
            + f"\nmax |CAD − sim| thrust angle: {ang_err:.2e}°",
            family="monospace", fontsize=8, va="top")
        self._draw_selected_comparison(arms[self.selected])
        self.fig.canvas.draw_idle()

    def _draw_selected_comparison(self, arm):
        """Selected arm: the genome values (input) beside the CAD values (output)."""
        g = self.genome[self.selected]
        genome_lines = [
            f"magnitude    {1000 * g[0]:7.1f} mm",
            f"arm_yaw      {np.degrees(g[1]):7.1f}°",
            f"arm_pitch    {np.degrees(g[2]):7.1f}°",
            f"motor_pitch  {np.degrees(g[3]):7.1f}°",
            f"motor_yaw    {np.degrees(g[4]):7.1f}°",
            f"spin         {'cw' if g[5] > 0.5 else 'ccw':>7}",
        ]
        cad_lines = [
            f"slot         {arm.slot:>4} ({arm.slot_angle:g}°)",
            f"yaw offset   {arm.arm_yaw_offset:7.1f}°",
            f"arm pitch    {arm.arm_pitch:7.1f}°",
            f"arm length   {arm.arm_length:7.1f} mm",
            f"arm roll     {arm.arm_roll:7.1f}°",
            f"motor pitch  {arm.motor_pitch:7.1f}°",
            f"spin (shaft) {arm.spin:>7}{'  flipped' if arm.motor_flipped else ''}",
        ]
        y = 0.40
        self.text_ax.text(0, y, f"selected arm {self.selected}:", fontsize=9,
                          fontweight="bold", va="top")
        self.text_ax.text(0, y - 0.06, "GENOME (input)\n" + "\n".join(genome_lines),
                          color=GENOME_COLOUR, family="monospace", fontsize=8, va="top")
        self.text_ax.text(0.5, y - 0.06, "CAD (output)\n" + "\n".join(cad_lines),
                          color=CAD_COLOUR, family="monospace", fontsize=8, va="top")

    def _draw_slots(self, ax, R, used):
        for k, angle in enumerate(slot_angles(self.n_slots, self.slot_offset)):
            a = np.radians(angle)
            p = R * np.array([np.cos(a), np.sin(a), 0.0])
            if k in used:
                ax.scatter(*p, color="k", s=25, depthshade=False)
                ax.text(*(1.45 * p), f"#{k} {angle:g}°", fontsize=8, ha="center")
            else:
                ax.scatter(*p, facecolors="none", edgecolors="grey", s=20, depthshade=False)

    def _draw_hole_pairs(self, ax, arm, pivot, motor):
        root_dir, mount_dir = hole_pair_directions(arm)
        u = (motor - pivot) / max(np.linalg.norm(motor - pivot), 1e-9)
        half = HOLE_SPACING_MM / 2
        for centre, direction in ((pivot + ROOT_HOLES_MM * u, root_dir), (motor, mount_dir)):
            pair = np.array([centre - half * direction, centre + half * direction])
            ax.plot(*pair.T, color=HOLE_COLOUR, linewidth=1)
            ax.scatter(*pair.T, color=HOLE_COLOUR, s=14, depthshade=False)

    def _label_cad_values(self, ax, arm, pivot, motor):
        """CAD values of the selected arm, next to the part they set."""
        root_dir, mount_dir = hole_pair_directions(arm)
        length = np.linalg.norm(motor - pivot)
        u = (motor - pivot) / max(length, 1e-9)
        style = dict(color=CAD_COLOUR, fontsize=8, fontweight="bold")
        # Kept clear of the plate and prop so they stay readable.
        ax.text(*(pivot + 0.3 * length * u - 2 * HOLE_SPACING_MM * root_dir),
                f"arm roll {arm.arm_roll:.1f}°", **style)
        ax.text(*(motor + 1.5 * HOLE_SPACING_MM * (u + mount_dir)),
                f"motor pitch {arm.motor_pitch:.1f}°", **style)

    def _draw_normal(self, extent_m, elev, azim):
        ax = self.normal_ax
        ax.cla()
        if self.normal_visualizer is None:
            ax.set_axis_off()
            ax.text2D(0.5, 0.5, f"DroneVisualizer unavailable:\n{_DRONE_VISUALIZER_ERROR}",
                      transform=ax.transAxes, ha="center", va="center")
            return
        self.normal_visualizer.plot_3d(self.genome, ax=ax, elevation=elev, azimuth=azim,
                                       circle_radius=self.prop_radius / 1000.0)
        # Same fixed limits as the CAD panel (DroneVisualizer otherwise
        # rescales to fit the motors on every redraw).
        ax.set_xlim(-extent_m, extent_m)
        ax.set_ylim(-extent_m, extent_m)
        ax.set_zlim(-extent_m, extent_m)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlabel("X right [m]")
        ax.set_ylabel("Y front [m]")
        ax.set_zlabel("Z up [m]")
        ax.set_title("Drawn from the genome by DroneVisualizer\n"
                     "(red = thrust, magenta arc = spin)", color=GENOME_COLOUR)


def load_genome(path):
    genome = np.load(path, allow_pickle=True)
    genome = np.atleast_2d(np.asarray(genome, dtype=float))
    if genome.shape[-1] != 6:
        raise ValueError(f"expected an (n_arms, 6) spherical genome, got shape {genome.shape}")
    return genome[~np.isnan(genome).any(axis=1)]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--genome", help=".npy file with an (n_arms, 6) spherical genome")
    parser.add_argument("--plate-radius", type=float, default=DEFAULT_PLATE_RADIUS_MM,
                        help="plate radius in mm (default: %(default)s)")
    parser.add_argument("--prop-radius", type=float, default=DEFAULT_PROP_RADIUS_MM,
                        help="drawn prop radius in mm (default: %(default)s; real 2\" props: 12.7)")
    parser.add_argument("--n-slots", type=int, default=DEFAULT_N_SLOTS,
                        help="attachment slots around the plate rim (default: %(default)s)")
    parser.add_argument("--slot-offset", type=float, default=DEFAULT_SLOT_OFFSET_DEG,
                        help="azimuth of slot 0 in degrees from +X (default: %(default)s)")
    parser.add_argument("--csv", help="where the 'print / save CSV' button writes the table")
    args = parser.parse_args()

    genome = load_genome(args.genome) if args.genome else DEFAULT_GENOME
    print(format_table(genome_to_fusion_params(genome, args.plate_radius,
                                               args.n_slots, args.slot_offset)))
    Viewer(genome, args.plate_radius, args.csv, args.prop_radius,
           args.n_slots, args.slot_offset)
    plt.show()


if __name__ == "__main__":
    main()
