"""Identified real drones and motor calibration.

`PARAMS_HEX3` is the system identification of the real hexacopter whose
RL policies are flight-verified (optimal_quad_control_RL/randomization.py,
`params_hex3`, trained with w_max_n = 4700). `real_hex3_propellers` is its
layout inferred from the coefficient signs and ratios, in flight-controller
motor order, so that

    DroneSimulator(real_hex3_propellers(), sysid_params=PARAMS_HEX3)

reproduces the reference env exactly. `PARAMS_VUHEX` is a later
identification (`params_vuhex`) with the same motor order and spin pattern,
so it uses the same layout. `SYSID_DRONES` maps the `--drone` names to
(sysid, w_norm_max).

`calibrate_prop_constants` backs the per-motor physical constants (k_f, k_m,
motor curve, drag, ...) out of an identified drone so that custom
morphologies built from the same motors and props can be simulated in
physics mode (`propsize="hex3"`).
"""
from __future__ import annotations

import warnings

import numpy as np

from .dynamics_params import INFLOW_COEF
from .propeller_data import PROPELLER_LIBRARY, register_propeller

# optimal_quad_control_RL/randomization.py: params_hex3.
# k_wz is -k_w from the fit (thrust acts along -z); drag k_x, k_y kept from
# the previous hex sets.
PARAMS_HEX3 = {
    'k_wz': 6.09e-07, 'k_z': 0, 'k_wx': 0, 'k_x': -6.51e-05, 'k_wy': 0, 'k_y': -6.32e-05,
    'k_p1': -1.46e-05, 'k_p2': -1.46e-05, 'k_p3': 1.40e-05, 'k_p4': 1.40e-05, 'k_p5': -2.89e-05, 'k_p6': 2.84e-05,
    'k_pd1': 0, 'k_pd2': 0, 'k_pd3': 0, 'k_pd4': 0, 'k_pd5': 0, 'k_pd6': 0,
    'k_q1': -2.40e-05, 'k_q2': 2.53e-05, 'k_q3': -2.40e-05, 'k_q4': 2.53e-05, 'k_q5': 6.44e-07, 'k_q6': 6.44e-07,
    'k_qd1': 0, 'k_qd2': 0, 'k_qd3': 0, 'k_qd4': 0, 'k_qd5': 0, 'k_qd6': 0,
    'k_r1': 2.09e-06, 'k_r2': 2.09e-06, 'k_r3': -2.09e-06, 'k_r4': -2.09e-06, 'k_r5': -2.09e-06, 'k_r6': 2.09e-06,
    'k_rd1': 5.42e-04, 'k_rd2': 5.42e-04, 'k_rd3': -5.42e-04, 'k_rd4': -5.42e-04, 'k_rd5': -5.42e-04, 'k_rd6': 5.42e-04,
    'w_min': 448.13, 'w_max': 4449.64, 'k': 0.83, 'tau': 0.038
}

# Motor-speed normalization the hex3 policy was trained and flown with.
HEX3_W_NORM_MAX = 4700.0

# optimal_quad_control_RL/randomization.py: params_vuhex (verbatim).
PARAMS_VUHEX = {
    'k_wz': 5.92e-07, 'k_z': 0, 'k_wx': 0, 'k_x': -6.10e-07, 'k_wy': 0, 'k_y': -5.28e-07,
    'k_p1': -1.09e-05, 'k_p2': -1.09e-05, 'k_p3': 1.09e-05, 'k_p4': 1.09e-05, 'k_p5': -2.17e-05, 'k_p6': 2.17e-05,
    'k_pd1': 0, 'k_pd2': 0, 'k_pd3': 0, 'k_pd4': 0, 'k_pd5': 0, 'k_pd6': 0,
    'k_q1': -1.10e-05, 'k_q2': 1.10e-05, 'k_q3': -1.10e-05, 'k_q4': 1.10e-05, 'k_q5': 0, 'k_q6': 0,
    'k_qd1': 0, 'k_qd2': 0, 'k_qd3': 0, 'k_qd4': 0, 'k_qd5': 0, 'k_qd6': 0,
    'k_r1': 8.90e-07, 'k_r2': 8.90e-07, 'k_r3': -8.90e-07, 'k_r4': -8.90e-07, 'k_r5': -8.90e-07, 'k_r6': 8.90e-07,
    'k_rd1': 7.42e-04, 'k_rd2': 7.42e-04, 'k_rd3': -7.42e-04, 'k_rd4': -7.42e-04, 'k_rd5': -7.42e-04, 'k_rd6': 7.42e-04,
    'w_min': 433.41, 'w_max': 4537.87, 'k': 0.85, 'tau': 0.04
}

# w_max_n in optimal_quad_control_RL/real_quad_env.py when vuhex was added.
VUHEX_W_NORM_MAX = 4000.0

SYSID_DRONES = {
    "real_hex3": (PARAMS_HEX3, HEX3_W_NORM_MAX),
    "real_vuhex": (PARAMS_VUHEX, VUHEX_W_NORM_MAX),
}

# Placeholder scale of the real hex. The sysid only fixes k_f/m and the
# k_f·r/I ratios; absolute k_f, k_m and inertia need the measured mass and
# arm length. Replace these with measurements before trusting custom
# morphologies built with propsize="hex3". (Sysid mode is unaffected.)
HEX3_MASS = 0.19859     # kg, all-up weight with battery
HEX3_ARM_LENGTH = 0.08  # m, motor distance from the centre
_PLACEHOLDER_MASS = 0.30
_PLACEHOLDER_ARM_LENGTH = 0.10


def real_hex3_propellers(arm_length=None, propsize=3):
    """Hex-X layout of the real hex in flight-controller motor order.

    Inferred from PARAMS_HEX3 (k_p = -y·k_f/Ixx, k_q = x·k_f/Iyy, k_r ∝
    spin): M1 rear-right, M2 front-right, M3 rear-left, M4 front-left,
    M5 right, M6 left. |k_p5| ≈ 2·|k_p1| puts M1-M4 at ±30° from the x axis
    and M5/M6 on the y axis. Spin follows the sign of k_r (+ → ccw).
    """
    R = arm_length or HEX3_ARM_LENGTH or _PLACEHOLDER_ARM_LENGTH
    layout = [  # (angle from +x towards +y in degrees, spin)
        (150, "ccw"),   # M1 rear-right
        (30, "ccw"),    # M2 front-right
        (-150, "cw"),   # M3 rear-left
        (-30, "cw"),    # M4 front-left
        (90, "cw"),     # M5 right
        (-90, "ccw"),   # M6 left
    ]
    return [
        {"loc": [R * np.cos(np.radians(a)), R * np.sin(np.radians(a)), 0.0],
         "dir": [0.0, 0.0, -1.0, spin], "propsize": propsize}
        for a, spin in layout
    ]


def calibrate_prop_constants(sysid, propellers, mass, motor_mass=None):
    """Per-motor constants of an identified drone, for physics mode.

    Args:
        sysid: identified coefficients (randomization.py format), flat motors.
        propellers: the identified drone's layout in sysid motor order.
        mass: its measured mass (kg).
        motor_mass: mass of one motor + prop (kg), used by DroneConfiguration
            for custom morphologies. Defaults to the 3-inch library value.

    Returns:
        (specs, inertia): a prop-library entry and the inertia diagonal
        implied by the sysid (fit by least squares over all motors).
    """
    n = len(propellers)
    k_w = sysid["k_wz"]
    k_f = k_w * mass
    loc = np.array([p["loc"] for p in propellers], dtype=float)
    per_motor = lambda name: np.array([sysid[f"{name}{i + 1}"] for i in range(n)], dtype=float)
    k_p, k_q, k_r, k_rd = per_motor("k_p"), per_motor("k_q"), per_motor("k_r"), per_motor("k_rd")

    # k_p_i = -y_i·k_f/Ixx and k_q_i = x_i·k_f/Iyy: least-squares 1/I.
    lsq = lambda lever, coef: float(lever @ coef / (lever @ lever))
    inv_Ixx = lsq(-loc[:, 1] * k_f, k_p)
    inv_Iyy = lsq(loc[:, 0] * k_f, k_q)
    # Izz is not observable from roll/pitch; use the flat-frame identity
    # Izz ≈ Ixx + Iyy (point masses in the motor plane).
    Ixx, Iyy = 1.0 / inv_Ixx, 1.0 / inv_Iyy
    Izz = Ixx + Iyy

    k_m = float(np.mean(np.abs(k_r))) * Izz
    specs = {
        "constants": [k_f, k_m],
        "wmax": float(sysid["w_max"]),
        "mass": motor_mass if motor_mass is not None else PROPELLER_LIBRARY["prop3"]["mass"],
        "w_min": float(sysid["w_min"]),
        "k": float(sysid["k"]),
        "tau": float(sysid["tau"]),
        "k_r_react": float(np.mean(np.abs(k_rd))),
        "react_izz_ref": Izz,
        "k_x_drag": -float(sysid["k_x"]),
        "k_y_drag": -float(sysid["k_y"]),
        "drag_mass_ref": float(mass),
        "inflow_coef": INFLOW_COEF,
    }
    return specs, np.array([Ixx, Iyy, Izz])


def register_hex3_motor():
    """Register the real hex's motor/prop as prop-library entry "hex3"."""
    if HEX3_MASS is None or HEX3_ARM_LENGTH is None:
        warnings.warn(
            "hex3 motor constants use a PLACEHOLDER mass/arm length "
            f"({_PLACEHOLDER_MASS} kg, {_PLACEHOLDER_ARM_LENGTH} m). Set HEX3_MASS and "
            "HEX3_ARM_LENGTH in sysid_presets.py to the real hex's measurements.",
            stacklevel=2,
        )
    mass = HEX3_MASS or _PLACEHOLDER_MASS
    specs, _ = calibrate_prop_constants(PARAMS_HEX3, real_hex3_propellers(), mass)
    register_propeller("hex3", specs)
    return specs
