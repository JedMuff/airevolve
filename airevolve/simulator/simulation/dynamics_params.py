from __future__ import annotations
import numpy as np
from .propeller_data import get_extended_prop_params

def _spin_sign(rotation: str) -> float:
    if rotation == "ccw":
        return 1.0
    if rotation == "cw":
        return -1.0
    raise ValueError(f"unknown rotation direction: {rotation!r}")

def _normalize_thrust_dir(prop_dir):
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

        k_fx_signed.append(dx_i * k_f / m)
        k_fy_signed.append(dy_i * k_f / m)
        k_fz_signed.append(dz_i * k_f / m)

        cp_x = y_i * dz_i - z_i * dy_i  
        cp_y = z_i * dx_i - x_i * dz_i  
        cp_z = x_i * dy_i - y_i * dx_i
        
        k_p_signed.append((cp_x * k_f + spin * k_m * dx_i) / Ixx)
        k_q_signed.append((cp_y * k_f + spin * k_m * dy_i) / Iyy)
        k_r_signed.append((cp_z * k_f + spin * k_m * dz_i) / Izz)
        k_r_react_signed.append(spin * float(extended["k_r_react"]))

    return {
        "n_motors": n,
        "k_w": k_f / m,
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

W_MIN_N = 0.0
W_MAX_N = 3000.0