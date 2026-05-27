"""Reference-form dynamics parameter derivation.

Maps an airevolve propeller configuration into the reference dynamics form
(matching optimal_quad_control_RL/randomization.py:5-10's `params_5inch`
schema, but with per-motor coefficients computed from the actual airevolve
morphology rather than sysid'd from flight data).

This is the airevolve runtime equivalent of
`experimentation/reference_drone_sim.py:derive_params_2inch_from_airevolve`.
The runtime version generalizes to any prop size via
`get_extended_prop_params` (Phase 1) and to any N-motor morphology with
arbitrarily tilted thrust vectors.

Sign convention: we bake position, thrust-direction, and spin signs into
the per-motor coefficients themselves, so the downstream symbolic build is
morphology-agnostic.

**Per-motor force coefficients** (accounts for tilted thrust directions):
  * `k_fx_signed[i] = dx_i · k_f / m`  — body-x force accel per W²
  * `k_fy_signed[i] = dy_i · k_f / m`  — body-y force accel per W²
  * `k_fz_signed[i] = dz_i · k_f / m`  — body-z force accel per W²
  where (dx_i, dy_i, dz_i) is the normalized thrust direction of motor i.

**Per-motor moment coefficients** (full cross-product r × F_dir):
  * `k_p_signed[i] = (y_i·dz_i - z_i·dy_i) · k_f / Ixx`  — roll moment
  * `k_q_signed[i] = (z_i·dx_i - x_i·dz_i) · k_f / Iyy`  — pitch moment
  * `k_r_signed[i] = spin_i · 2 · k_m · W_hover / Izz`  — yaw (linearized)
  * `k_r_react_signed[i] = spin_i · k_r_react`  — yaw from dW

For standard quadrotors with dir=[0,0,-1] on all motors, these reduce to
the original formulas: k_fx=k_fy=0, k_fz=-k_f/m, k_p=-y·k_f/Ixx, etc.

See `experimentation/RUNTIME_DYNAMICS_MIGRATION.md` Phase 2.1.
"""
from __future__ import annotations

import numpy as np

from .propeller_data import get_extended_prop_params


def _spin_sign(rotation: str) -> float:
    """+1 for ccw, -1 for cw. Consistent with the reference's convention
    (verified against optimal_quad_control_RL via the V0 parity test)."""
    if rotation == "ccw":
        return 1.0
    if rotation == "cw":
        return -1.0
    raise ValueError(f"unknown rotation direction: {rotation!r}")


def _normalize_thrust_dir(prop_dir):
    """Extract and normalize the thrust direction vector from a prop dir spec.

    Args:
        prop_dir: [dx, dy, dz, rotation_str] from propeller config.

    Returns:
        (dx, dy, dz) as floats, unit-normalized.
    """
    d = np.array([float(prop_dir[0]), float(prop_dir[1]), float(prop_dir[2])])
    mag = np.linalg.norm(d)
    if mag < 1e-12:
        raise ValueError(f"Zero-length thrust direction: {prop_dir[:3]}")
    d /= mag
    return float(d[0]), float(d[1]), float(d[2])


def derive_reference_params(
    propellers: list,
    mass: float,
    inertia: np.ndarray,
    prop_size,
    gravity: float = 9.81,
) -> dict:
    """Derive a reference-form parameter dict for an airevolve drone config.

    Args:
        propellers: list of propeller dicts (`loc`, `dir`, `propsize`).
        mass: total drone mass in kg (from DroneConfiguration.mass).
        inertia: 3x3 inertia matrix in body frame.
        prop_size: prop size for fetching aerodynamic constants
            (k_drag, k_r_react, k, w_min). All propellers assumed same size.
        gravity: m/s².

    Returns:
        dict with keys:
            n_motors (int): number of motors
            k_w (float): legacy scalar thrust coeff (kept for backward compat)
            k_fx_signed, k_fy_signed, k_fz_signed (list[float]):
                per-motor force accel coefficients in body x/y/z
            k_x, k_y (float): body-frame drag accel coefficients
            k_p_signed (list[float]): per-motor roll-moment coefficient
            k_q_signed (list[float]): per-motor pitch-moment coefficient
            k_r_signed (list[float]): per-motor yaw-moment coefficient (linear W)
            k_r_react_signed (list[float]): per-motor yaw-moment from dW
            tau, k, w_min, w_max (float): motor model parameters

    Notes:
        * Tilted thrust directions are fully supported via per-motor
          `k_fx/k_fy/k_fz_signed` and moment coefficients computed from
          the cross product of position × thrust direction.
        * For standard drones with dir=[0,0,-1], this reduces to the
          original formulas (k_fx=k_fy=0, k_fz=-k_f/m).
        * `k_w` is kept for backward compatibility but is NOT used by the
          dynamics. It equals `k_f / m` (same as before).
    """
    n = len(propellers)
    if n == 0:
        raise ValueError("derive_reference_params: no propellers in config")

    extended = get_extended_prop_params(prop_size)
    k_f, k_m = extended["constants"]
    w_max = float(extended["wmax"])

    Ixx = float(inertia[0, 0])
    Iyy = float(inertia[1, 1])
    Izz = float(inertia[2, 2])
    m = float(mass)

    # Hover motor speed (per motor) for linearizing the yaw torque term.
    F_hover_per_motor = m * gravity / n
    if F_hover_per_motor <= 0 or k_f <= 0:
        raise ValueError(
            f"derive_reference_params: invalid hover-thrust calc "
            f"(F_hover={F_hover_per_motor}, k_f={k_f})"
        )
    W_hover = float(np.sqrt(F_hover_per_motor / k_f))

    k_fx_signed = []
    k_fy_signed = []
    k_fz_signed = []
    k_p_signed = []
    k_q_signed = []
    k_r_signed = []
    k_r_react_signed = []
    for prop in propellers:
        x_i = float(prop["loc"][0])
        y_i = float(prop["loc"][1])
        z_i = float(prop["loc"][2])
        spin = _spin_sign(prop["dir"][3])
        dx_i, dy_i, dz_i = _normalize_thrust_dir(prop["dir"])

        # Per-motor force acceleration coefficients (body frame).
        # F_body_i = k_f · W_i² · d_hat_i  →  accel_i = (k_f / m) · W_i² · d_hat_i
        k_fx_signed.append(dx_i * k_f / m)
        k_fy_signed.append(dy_i * k_f / m)
        k_fz_signed.append(dz_i * k_f / m)

        # Per-motor moment coefficients from thrust (full cross product).
        # M_i = r_i × F_i = r_i × (k_f · W_i² · d_hat_i)
        # Angular accel = M / I (using diagonal inertia approximation).
        # cross(r, d) = (y·dz - z·dy, z·dx - x·dz, x·dy - y·dx)
        cp_x = y_i * dz_i - z_i * dy_i  # roll moment arm
        cp_y = z_i * dx_i - x_i * dz_i  # pitch moment arm
        k_p_signed.append(cp_x * k_f / Ixx)
        k_q_signed.append(cp_y * k_f / Iyy)

        # M_z (steady, linearized at hover): spin_i · 2 · k_m · W_hover / Izz
        k_r_signed.append(spin * 2.0 * k_m * W_hover / Izz)
        # M_z (motor-acceleration reaction): spin_i · k_r_react
        k_r_react_signed.append(spin * float(extended["k_r_react"]))

    return {
        "n_motors": n,
        "k_w": k_f / m,  # legacy, kept for backward compat
        "k_fx_signed": k_fx_signed,
        "k_fy_signed": k_fy_signed,
        "k_fz_signed": k_fz_signed,
        "k_x": float(extended["k_x_drag"]),
        "k_y": float(extended["k_y_drag"]),
        "k_p_signed": k_p_signed,
        "k_q_signed": k_q_signed,
        "k_r_signed": k_r_signed,
        "k_r_react_signed": k_r_react_signed,
        "tau": float(extended["tau"]),
        "k": float(extended["k"]),
        "w_min": float(extended["w_min"]),
        "w_max": w_max,
    }


# Reference's normalization constants (verified against
# optimal_quad_control_RL/quad_race_env.py:41-42 via the V0 parity test).
# These are intentionally independent of physical w_max; the motor state
# `w_i ∈ [-1, 1]` represents `W_i ∈ [W_MIN_N, W_MAX_N] = [0, 3000]` rad/s.
# At full throttle, Wc may exceed W_MAX_N; the state can go outside [-1, 1]
# during transients. This is intentional in the reference.
W_MIN_N = 0.0
W_MAX_N = 3000.0
