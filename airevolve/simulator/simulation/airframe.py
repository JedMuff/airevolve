"""Airframe mass model: everything on the drone except the motors and props.

`mass_properties` builds mass, centre of gravity and inertia from rigid
parts: a central hub (box), the battery (box), one straight arm per motor
from the hub rim to the motor (uniform rod), and the motor + prop as a point
mass at its location. The arms are what scale with an evolved morphology;
the hub and battery are fixed.

Airframes
---------
LEGACY      the original model: 13.6 g flight controller block, 43 g battery
            2 cm below, carbon tubes (34 g/m) from the centre to each motor.
REAL_HEX    the flight-verified hexacopter, from weighing its parts:
            battery 41.11 g; central part 40 g (plate, ESCs, flight
            controller, Raspberry Pi, screws) + 8.90 g wiring; 12.5 g printed
            arms from the 30 mm plate rim to the motor (0.25 kg/m). With its
            motors (4.6 g + 0.996 g prop at 8 cm) this sums to the measured
            198.59 g. `inertia_scale` is fitted so that its roll/pitch
            inertia equals the inertia implied by the hex's sysid (see
            sysid_presets.register_hex3_motor); it absorbs what the part
            model misses (mass placement within the parts, battery position).

A prop-library entry selects its airframe with an "airframe" key (the
"hex3" motors use "real_hex"); entries without one use LEGACY.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np

from .propeller_data import BATTERY_MASS, BEAM_DENSITY, CONTROLLER_MASS


@dataclass(frozen=True)
class Airframe:
    name: str
    hub_mass: float             # kg, at the origin
    hub_size: tuple             # m, box dimensions (x, y, z)
    battery_mass: float         # kg
    battery_pos: tuple          # m, body frame (NED)
    battery_size: tuple         # m, box dimensions (0 = point mass)
    arm_density: float          # kg/m
    arm_root_radius: float      # m, arms start this far out (hub rim)
    inertia_scale: float = 1.0  # applied to the whole inertia matrix

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in d.items()})


LEGACY_AIRFRAME = Airframe(
    name="legacy",
    hub_mass=CONTROLLER_MASS, hub_size=(0.105, 0.036, 0.035),
    battery_mass=BATTERY_MASS, battery_pos=(0.0, 0.0, 0.02), battery_size=(0.0, 0.0, 0.0),
    arm_density=BEAM_DENSITY, arm_root_radius=0.0,
)

# Unscaled; sysid_presets registers the calibrated copy as "real_hex".
REAL_HEX_AIRFRAME = Airframe(
    name="real_hex",
    hub_mass=0.040 + 0.008904,
    # 60 mm plate (a box with the same in-plane inertia as the disk) and a
    # 20 mm stack.
    hub_size=(0.052, 0.052, 0.020),
    battery_mass=0.04111, battery_pos=(0.0, 0.0, 0.0), battery_size=(0.058, 0.031, 0.019),
    arm_density=0.0125 / 0.050, arm_root_radius=0.030,
)

AIRFRAMES = {"legacy": LEGACY_AIRFRAME}


def register_airframe(airframe: Airframe):
    AIRFRAMES[airframe.name] = airframe


def get_airframe(name):
    if name is None:
        return LEGACY_AIRFRAME
    if name not in AIRFRAMES:
        raise ValueError(f"unknown airframe {name!r}; registered: {list(AIRFRAMES)}")
    return AIRFRAMES[name]


def _box_inertia(mass, size):
    x, y, z = size
    return mass / 12.0 * np.diag([y * y + z * z, x * x + z * z, x * x + y * y])


def arm_segments(locs, airframe: Airframe):
    """(root, tip) of each arm: from the hub rim, in the motor's horizontal
    direction, to the motor."""
    segments = []
    for loc in np.asarray(locs, dtype=float):
        horizontal = np.array([loc[0], loc[1], 0.0])
        r = np.linalg.norm(horizontal)
        root = horizontal / r * airframe.arm_root_radius if r > 1e-9 else np.zeros(3)
        segments.append((root, loc))
    return segments


def mass_properties(locs, motor_masses, airframe: Airframe = LEGACY_AIRFRAME):
    """Mass (kg), centre of gravity (3,) and inertia about it (3, 3).

    Args:
        locs: (n, 3) motor positions in the body frame.
        motor_masses: (n,) motor + prop masses.
        airframe: the rest of the drone.
    """
    # Parts as (mass, centre, inertia about own centre).
    parts = [
        (airframe.hub_mass, np.zeros(3), _box_inertia(airframe.hub_mass, airframe.hub_size)),
        (airframe.battery_mass, np.asarray(airframe.battery_pos, dtype=float),
         _box_inertia(airframe.battery_mass, airframe.battery_size)),
    ]
    for (root, tip), motor_mass in zip(arm_segments(locs, airframe), motor_masses):
        d = tip - root
        length = np.linalg.norm(d)
        arm_mass = airframe.arm_density * length
        if length > 1e-9:
            u = d / length
            rod = arm_mass * length**2 / 12.0 * (np.eye(3) - np.outer(u, u))
            parts.append((arm_mass, root + d / 2, rod))
        parts.append((float(motor_mass), tip, np.zeros((3, 3))))

    mass = sum(m for m, _, _ in parts)
    cg = sum(m * c for m, c, _ in parts) / mass
    inertia = np.zeros((3, 3))
    for m, c, own in parts:
        r = c - cg
        inertia += own + m * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
    return mass, cg, airframe.inertia_scale * inertia


def scaled(airframe: Airframe, inertia_scale: float) -> Airframe:
    return replace(airframe, inertia_scale=float(inertia_scale))
