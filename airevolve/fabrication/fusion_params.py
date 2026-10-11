"""
Genome → Fusion 360 arm parameters.

The physical drone is a flat central plate of radius R with a fixed ring of
screw holes. Arms bolt on at discrete attachment slots on the plate rim (at
the plate mid-plane), and every motor bolts to a mount at the arm tip. Per
arm the CAD needs:

    slot            which attachment slot on the rim (slot_angle = its azimuth)
    arm_yaw_offset  arm azimuth relative to the radial direction at the slot
    arm_pitch       arm elevation out of the plate, +90° = straight up
    arm_length      slot pivot → motor centre, in mm
    arm_roll        roll of the arm-root mounting-hole pair about the arm axis
    motor_pitch     angle of the motor-mount hole pair (tilt toward the tip)

Slots
-----
``n_slots`` slots (default 16: 32 core holes, 2-hole mounts that do not share
holes, 22.5° apart). Slot k sits at ``slot_offset + k·360/n_slots`` degrees,
measured from +X (right), counter-clockwise seen from above — so with the
defaults slot 4 is the front (+Y). Each arm gets its own slot (globally
optimal assignment on the slot → motor distance). The motor position is kept
exactly, so an arm whose genome azimuth is off-grid leaves its slot at a
non-zero ``arm_yaw_offset``.

Frames
------
Genome rows are ``[magnitude (m), arm_yaw, arm_pitch, motor_pitch, motor_yaw,
direction]`` written in a body frame G with x = right, y = **forward**,
z = up. ``genome_to_propellers`` maps G to the simulator's NED body frame via
NED = (G_y, G_x, −G_z), which is a proper rotation (no mirroring). All CAD
parameters here are expressed in G, so in Fusion:

    plate in the XY plane, Z up, front of the drone (FC arrow) = +Y.

In G a genome motor sits at ``m·(cos e cos a, cos e sin a, sin e)`` and
thrusts along ``−(cos y sin p, sin y sin p, cos p)``; motor_pitch = π gives
thrust straight up. Genome motor angles are absolute body-frame angles.

Arm kinematics (motor position preserved exactly)
-------------------------------------------------
    pivot      A = R(cos ψ_k, sin ψ_k, 0),  d = P − A
    azimuth    α = atan2(d_y, d_x),          arm_yaw_offset = α − ψ_k
    arm_pitch  θ = atan2(d_z, |d_xy|)
    arm_length L = |d|

An offset beyond ±90° means the arm runs back over the plate; this is flagged
in ``warnings``.

Mounting holes (arm roll, motor pitch)
--------------------------------------
Arm frame: u = cos θ â + sin θ ẑ (along the arm, â = horizontal arm
direction), w = −sin θ â + cos θ ẑ (arm "up"), t̂ = ẑ × â (side). (u, t̂, w)
is right-handed.

* Arm roll φ: the arm-root hole pair, default (0°) two holes along w —
  vertically along z for a flat arm. Rolling the pair by φ about u
  (right-hand rule about the outward arm axis) puts it along
  ``w' = cos φ w − sin φ t̂``.
* Motor pitch τ: the motor-mount hole pair always lies across the arm, along
  the rolled side axis ``t̂' = cos φ t̂ + sin φ w`` (perpendicular to the
  arm). The motor mount pitches about the axis through these two holes; at
  0° the motor faces w' (perpendicular to the arm), and at τ it faces

      f = cos τ w' + sin τ u          (positive τ = toward the arm tip)

A flat quad gives zero roll and pitch on every arm.

Pusher flip (CAD only)
----------------------
The prop sits on the side the motor faces. If the thrust points back along
the arm (f · u < 0, i.e. negative motor pitch) the arm would be on the prop
side, so the motor is mounted the other way round: facing −f
(pitch → −pitch, roll → roll + 180°), with its spin reversed. The prop keeps
the same orientation and rotation in the world, so thrust and yaw torque are
unchanged — but it is now mounted upside down on the shaft (keep the
original prop type, mount it flipped, and reverse the motor direction).
The genome and simulator are not affected; ``motor_flipped`` records the flip
and ``fusion_params_to_propellers`` undoes it.

Spin is always given as seen looking at the motor from its shaft (prop) end.
For an unflipped motor that is the simulator's convention (ccw about the
thrust, seen from the side the thrust points to); for a flipped motor it is
the opposite label.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

DEFAULT_PLATE_RADIUS_MM = 30.0
DEFAULT_N_SLOTS = 16
DEFAULT_SLOT_OFFSET_DEG = 0.0

# Below this horizontal slot → motor distance (mm) the arm azimuth is
# ill-defined (motor straight above/below the slot) and the radial direction
# is used instead.
_AZIMUTH_EPS_MM = 1e-6
# |f · u| above this means the motor faces (almost) along the arm: roll is
# then ill-defined (gimbal lock) and is set to 0.
_GIMBAL_LOCK_COS = np.cos(np.radians(0.5))
# Motors perpendicular to the arm (f · u = 0 up to rounding) are not flipped.
_FLIP_EPS = 1e-9


@dataclass
class ArmFusionParams:
    """CAD parameters for one arm. Angles in degrees, lengths in mm."""
    slot: int
    slot_angle: float
    arm_yaw_offset: float          # arm azimuth − slot_angle
    arm_pitch: float
    arm_length: float
    arm_roll: float                # arm-root hole pair, 0 = along arm "up"
    motor_pitch: float             # motor-mount hole pair, 0 = along the arm
    spin: str                      # "cw" / "ccw" seen from the motor's shaft end
    motor_flipped: bool = False    # mounted as a pusher (shaft = −thrust)
    warnings: List[str] = field(default_factory=list)

    @property
    def arm_yaw(self) -> float:
        """Absolute azimuth of the arm in degrees."""
        return self.slot_angle + self.arm_yaw_offset

    @property
    def sim_spin(self) -> str:
        """Spin in the simulator's convention (about the thrust direction)."""
        if not self.motor_flipped:
            return self.spin
        return "ccw" if self.spin == "cw" else "cw"


def slot_angles(n_slots: int = DEFAULT_N_SLOTS,
                slot_offset_deg: float = DEFAULT_SLOT_OFFSET_DEG) -> np.ndarray:
    """Azimuth of every attachment slot in degrees, in [0, 360)."""
    return (slot_offset_deg + 360.0 / n_slots * np.arange(n_slots)) % 360.0


def genome_motor_in_g(row) -> tuple[np.ndarray, np.ndarray]:
    """Motor position (mm) and unit thrust direction of a genome row, in frame G."""
    magnitude, arm_yaw, arm_pitch, mot_pitch, mot_yaw = (float(v) for v in row[:5])
    pos = 1000.0 * magnitude * np.array([
        np.cos(arm_pitch) * np.cos(arm_yaw),
        np.cos(arm_pitch) * np.sin(arm_yaw),
        np.sin(arm_pitch),
    ])
    thrust = -np.array([
        np.cos(mot_yaw) * np.sin(mot_pitch),
        np.sin(mot_yaw) * np.sin(mot_pitch),
        np.cos(mot_pitch),
    ])
    return pos, thrust


def _arm_frame(yaw: float, pitch: float):
    """(â, u, t̂, w) for arm azimuth/pitch in radians."""
    a_hat = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    z_hat = np.array([0.0, 0.0, 1.0])
    u = np.cos(pitch) * a_hat + np.sin(pitch) * z_hat
    w = -np.sin(pitch) * a_hat + np.cos(pitch) * z_hat
    t_hat = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
    return a_hat, u, t_hat, w


def _slot_pivot(slot_angle_rad: float, plate_radius_mm: float) -> np.ndarray:
    return plate_radius_mm * np.array([np.cos(slot_angle_rad), np.sin(slot_angle_rad), 0.0])


def _assign_slots(motors_mm: Sequence[np.ndarray], plate_radius_mm: float,
                  angles_deg: np.ndarray) -> np.ndarray:
    """Give each arm a unique slot, minimising the total slot → motor distance."""
    if len(motors_mm) > len(angles_deg):
        raise ValueError(f"{len(motors_mm)} arms but only {len(angles_deg)} attachment slots")
    pivots = np.array([_slot_pivot(np.radians(a), plate_radius_mm) for a in angles_deg])
    cost = np.linalg.norm(np.asarray(motors_mm)[:, None, :] - pivots[None, :, :], axis=2)
    _, slots = linear_sum_assignment(cost)
    return slots


def _wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def genome_to_fusion_params(
    genome,
    plate_radius_mm: float = DEFAULT_PLATE_RADIUS_MM,
    n_slots: int = DEFAULT_N_SLOTS,
    slot_offset_deg: float = DEFAULT_SLOT_OFFSET_DEG,
    flip_motors: bool = True,
) -> List[ArmFusionParams]:
    """Convert a spherical-angular genome (n_arms × 6) to per-arm CAD parameters.

    Rows containing NaN are skipped, matching ``genome_to_propellers``. Each
    arm is given its own of ``n_slots`` attachment slots (``ValueError`` if
    there are more arms than slots). With ``flip_motors`` any motor whose
    thrust points back along its arm is mounted as a pusher (see module
    docstring).
    """
    genome = np.atleast_2d(np.asarray(genome, dtype=float))
    genome = genome[~np.isnan(genome).any(axis=1)]
    R = float(plate_radius_mm)
    if len(genome) == 0:
        return []

    motors = [genome_motor_in_g(row) for row in genome]
    angles = slot_angles(n_slots, slot_offset_deg)
    slots = _assign_slots([P for P, _ in motors], R, angles)

    arms = []
    for row, (P, f), slot in zip(genome, motors, slots):
        warnings = []
        psi = np.radians(angles[slot])
        d = P - _slot_pivot(psi, R)

        horiz = np.hypot(d[0], d[1])
        yaw = np.arctan2(d[1], d[0]) if horiz > _AZIMUTH_EPS_MM else psi
        yaw_offset = _wrap_deg(np.degrees(yaw - psi))
        pitch = np.arctan2(d[2], horiz)
        if abs(yaw_offset) > 90.0:
            warnings.append(f"arm yaw offset {yaw_offset:.1f}° runs back over the plate")

        _, u, t_hat, w = _arm_frame(yaw, pitch)
        spin = "cw" if row[5] > 0.5 else "ccw"
        # Thrust pointing back along the arm puts the arm on the prop side:
        # turn the motor round (shaft = −thrust) and reverse its spin.
        flipped = flip_motors and float(np.dot(f, u)) < -_FLIP_EPS
        if flipped:
            f = -f
            spin = "ccw" if spin == "cw" else "cw"
        fu = float(np.dot(f, u))
        motor_pitch = np.arcsin(np.clip(fu, -1.0, 1.0))
        if abs(fu) > _GIMBAL_LOCK_COS:
            roll = 0.0
            warnings.append("motor faces along the arm: arm roll is arbitrary (set to 0)")
        else:
            roll = np.arctan2(-np.dot(f, t_hat), np.dot(f, w))

        arms.append(ArmFusionParams(
            slot=int(slot),
            slot_angle=float(angles[slot]),
            arm_yaw_offset=float(yaw_offset),
            arm_pitch=float(np.degrees(pitch)),
            arm_length=float(np.linalg.norm(d)),
            arm_roll=float(np.degrees(roll)),
            motor_pitch=float(np.degrees(motor_pitch)),
            spin=spin,
            motor_flipped=bool(flipped),
            warnings=warnings,
        ))
    return arms


def _arm_axes(arm: ArmFusionParams):
    """(u, w', t̂', shaft) of one arm: arm axis, rolled arm "up", rolled side
    axis and motor facing."""
    _, u, t_hat, w = _arm_frame(np.radians(arm.arm_yaw), np.radians(arm.arm_pitch))
    roll, tilt = np.radians(arm.arm_roll), np.radians(arm.motor_pitch)
    w_rolled = np.cos(roll) * w - np.sin(roll) * t_hat
    t_rolled = np.cos(roll) * t_hat + np.sin(roll) * w
    shaft = np.cos(tilt) * w_rolled + np.sin(tilt) * u
    return u, w_rolled, t_rolled, shaft


def arm_geometry_in_g(arm: ArmFusionParams, plate_radius_mm: float = DEFAULT_PLATE_RADIUS_MM):
    """Forward kinematics of one arm in frame G.

    Returns (pivot_mm, motor_mm, shaft_unit, thrust_unit) — built purely from
    the CAD parameters, independently of the genome. ``shaft`` is the way the
    motor (and prop) faces; ``thrust`` equals it unless the motor is flipped.
    """
    u, _, _, shaft = _arm_axes(arm)
    pivot = _slot_pivot(np.radians(arm.slot_angle), plate_radius_mm)
    motor = pivot + arm.arm_length * u
    thrust = -shaft if arm.motor_flipped else shaft
    return pivot, motor, shaft, thrust


def hole_pair_directions(arm: ArmFusionParams):
    """Unit directions (in G) of the arm-root and motor-mount hole pairs.

    The arm-root pair lies along the rolled arm "up" w' (its angle sets the
    arm roll); the motor-mount pair lies across the arm along the rolled side
    axis t̂', which is the axis the motor pitches about.
    """
    _, w_rolled, t_rolled, _ = _arm_axes(arm)
    return w_rolled, t_rolled


def g_to_ned(v) -> np.ndarray:
    """Frame G (x right, y forward, z up) → simulator NED body frame."""
    v = np.asarray(v, dtype=float)
    return np.array([v[1], v[0], -v[2]])


def fusion_params_to_propellers(
    arms: Sequence[ArmFusionParams],
    plate_radius_mm: float = DEFAULT_PLATE_RADIUS_MM,
    propsize=2,
) -> list:
    """Rebuild simulator propeller dicts (NED, metres) from CAD parameters.

    The output has the same format as ``genome_to_propellers`` so the two can
    be compared directly. Pusher flips are undone (thrust and spin back in the
    simulator's convention).
    """
    propellers = []
    for arm in arms:
        _, motor, _, thrust = arm_geometry_in_g(arm, plate_radius_mm)
        loc = g_to_ned(motor / 1000.0)
        d = g_to_ned(thrust)
        propellers.append({
            "loc": loc.tolist(),
            "dir": [*d.tolist(), arm.sim_spin],
            "propsize": propsize,
        })
    return propellers


_COLUMNS = ["arm", "slot", "slot_angle_deg", "arm_yaw_offset_deg", "arm_pitch_deg",
            "arm_length_mm", "arm_roll_deg", "motor_pitch_deg", "motor_spin_from_shaft",
            "motor_flipped", "warnings"]


def _rows(arms: Sequence[ArmFusionParams]):
    for i, a in enumerate(arms):
        yield [i, a.slot, round(a.slot_angle, 2), round(a.arm_yaw_offset, 2),
               round(a.arm_pitch, 2), round(a.arm_length, 2), round(a.arm_roll, 2),
               round(a.motor_pitch, 2), a.spin, int(a.motor_flipped), "; ".join(a.warnings)]


def to_csv(arms: Sequence[ArmFusionParams], path) -> None:
    """Write one row per arm (degrees / mm) for Fusion's parameter table."""
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(_COLUMNS)
        writer.writerows(_rows(arms))


def format_table(arms: Sequence[ArmFusionParams]) -> str:
    lines = [f"{'arm':>3} {'slot':>4} {'slot°':>7} {'yawoff°':>8} {'pitch°':>7} "
             f"{'len mm':>7} {'roll°':>7} {'mpitch°':>8} {'spin':>4} flip"]
    for i, a in enumerate(arms):
        lines.append(f"{i:>3} {a.slot:>4} {a.slot_angle:>7.2f} {a.arm_yaw_offset:>8.2f} "
                     f"{a.arm_pitch:>7.2f} {a.arm_length:>7.2f} {a.arm_roll:>7.2f} "
                     f"{a.motor_pitch:>8.2f} {a.spin:>4} {'yes' if a.motor_flipped else '':>4}")
        lines.extend(f"    ! {w}" for w in a.warnings)
    lines.append("roll: arm-root hole pair about the arm (0 = holes along z)")
    lines.append("mpitch: motor about its mount holes (0 = ⟂ arm, + = to tip)")
    lines.append("spin seen from the motor's shaft end")
    return "\n".join(lines)


def print_table(arms: Sequence[ArmFusionParams]) -> None:
    print(format_table(arms))
