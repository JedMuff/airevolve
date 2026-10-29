"""Reference-form dynamics parameters.

The runtime dynamics (DroneSimulator) use the per-motor form of
`optimal_quad_control_RL/real_quad_env.py`, the model whose policies are
flight-verified on the real hexacopter. For motor i with unit thrust
direction d_i (body frame), speed W_i (rad/s) and body velocity v_b:

    vd_i = d_i · v_b                  (axial inflow speed)
    vp_i = v_b - vd_i · d_i           (in-plane speed)

per body axis j, exactly as the reference writes it:

    F_j = Σ_i f_coef_ij · (W_i² - inflow_coef · vd_i · W_i)
          + k_drag_j · Σ_i vp_ij · W_i
    M   = Σ_i m_coef_i · W_i² + Σ_i m_dw_coef_i · dW_i/dt

F and M are accelerations (mass and inertia are absorbed into the
coefficients). The motor model maps the action u ∈ [-1, 1] to U = (u+1)/2,
then Wc = (w_max - w_min)·sqrt(k·U² + (1-k)·U) + w_min with a first-order
lag of time constant tau.

There are two ways to get the coefficients:

* `derive_reference_params` (physics mode, any morphology): from the
  propeller layout, mass, inertia and per-prop constants:

      f_coef_i  = k_f / m · d_i
      m_coef_i  = I⁻¹ ⊙ (r_i × k_f d_i + spin_i · k_m · a_i)
      m_dw_coef_i = spin_i · k_r_react · (Izz_ref / Izz) · a_i
      k_drag    = -(k_x, k_y, 0) · (m_ref / m)

  with a_i = -d_i the prop axis, r_i = loc_i - cg, spin_i = +1 for "ccw"
  and -1 for "cw".

* `sysid_to_reference_params` (sysid mode): maps a real-repo
  `randomization.py`-style dict (k_wz, k_x, k_p1..N, k_rd1..N, ...)
  directly. For flat motors this reproduces `real_quad_env.f_func`.

`params_to_vector` / `ParamLayout` flatten a parameter dict into the vector
consumed by the lambdified dynamics, so every env can carry its own
(domain-randomized) parameters.
"""
from __future__ import annotations

import numpy as np

from .propeller_data import get_extended_prop_params

# Axial inflow coefficient of the reference model (2π / 5-inch pitch in m).
# real_quad_env.py hardcodes it for every drone.
INFLOW_COEF = 2 * np.pi / 0.127

# Motor-speed normalization: the state stores w ∈ [-1, 1] for
# W ∈ [W_MIN_N, W_MAX_N] rad/s. It sets the scale of the motor-speed policy
# inputs and must match `w_max_n` in the deployed C controller (each run
# stores its value in run_config.json). 4000 matches the current
# real_quad_env.py; the flown hex3 policies used 4700.
W_MIN_N = 0.0
W_MAX_N = 4000.0


def _spin_sign(rotation: str) -> float:
    """+1 for ccw, -1 for cw."""
    if rotation == "ccw":
        return 1.0
    if rotation == "cw":
        return -1.0
    raise ValueError(f"unknown rotation direction: {rotation!r}")


def _unit_axes(propellers) -> np.ndarray:
    d = np.array([prop["dir"][:3] for prop in propellers], dtype=float)
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def derive_reference_params(
    propellers: list,
    mass: float,
    inertia: np.ndarray,
    prop_size,
    gravity: float = 9.81,
    cg=None,
) -> dict:
    """Derive reference-form coefficients from a morphology (physics mode).

    Args:
        propellers: list of propeller dicts (`loc`, `dir`, `propsize`).
        mass: total drone mass in kg.
        inertia: 3x3 inertia matrix in body frame (diagonal is used).
        prop_size: prop-library key for the motor/prop constants. All
            propellers are assumed to share it.
        gravity: m/s² (unused by the coefficients, kept for API compat).
        cg: centre of gravity in body frame; lever arms are measured from
            it. Defaults to the origin.

    Returns:
        Reference-form parameter dict (see module docstring), plus `k_w`
        (= k_f/m, the flat-motor thrust coefficient) for convenience.
    """
    n = len(propellers)
    if n == 0:
        raise ValueError("derive_reference_params: no propellers in config")

    ext = get_extended_prop_params(prop_size)
    k_f, k_m = ext["constants"]
    m = float(mass)
    I_diag = np.array([inertia[0, 0], inertia[1, 1], inertia[2, 2]], dtype=float)
    I_inv = 1.0 / I_diag
    cg = np.zeros(3) if cg is None else np.asarray(cg, dtype=float)

    # Constants calibrated on a specific drone carry its mass/Izz so they
    # rescale to the new morphology; library entries without them are
    # already per-drone accelerations and are used as-is.
    react_scale = ext.get("react_izz_ref", I_diag[2]) / I_diag[2]
    drag_scale = ext.get("drag_mass_ref", m) / m

    axes = _unit_axes(propellers)
    f_coef, m_coef, m_dw_coef = [], [], []
    for prop, d in zip(propellers, axes):
        spin = _spin_sign(prop["dir"][3])
        a = -d  # prop axis; (0, 0, 1) for a flat motor
        r = np.asarray(prop["loc"], dtype=float) - cg
        f_coef.append(k_f / m * d)
        m_coef.append(I_inv * (np.cross(r, k_f * d) + spin * k_m * a))
        m_dw_coef.append(spin * ext["k_r_react"] * react_scale * a)

    return {
        "n_motors": n,
        "axis": axes.tolist(),
        "f_coef": np.array(f_coef).tolist(),
        "m_coef": np.array(m_coef).tolist(),
        "m_dw_coef": np.array(m_dw_coef).tolist(),
        "k_drag": [-ext["k_x_drag"] * drag_scale, -ext["k_y_drag"] * drag_scale, 0.0],
        "inflow_coef": float(ext.get("inflow_coef", INFLOW_COEF)),
        "tau": float(ext["tau"]),
        "k": float(ext["k"]),
        "w_min": float(ext["w_min"]),
        "w_max": float(ext["wmax"]),
        "k_w": k_f / m,
    }


def sysid_to_reference_params(sysid: dict, propellers: list) -> dict:
    """Map a real-repo sysid dict onto the reference form (sysid mode).

    `sysid` uses the `optimal_quad_control_RL/randomization.py` keys:
    k_wz, k_z, k_wx, k_x, k_wy, k_y, k_p1..N, k_pd1..N, k_q1..N, k_qd1..N,
    k_r1..N, k_rd1..N, tau, k, w_min, w_max. Motor i of `propellers` is
    motor i+1 of the sysid; only the thrust axes are taken from
    `propellers` (positions and spins are already in the coefficients).
    """
    axes = _unit_axes(propellers)
    layout = ParamLayout(len(propellers))
    params = layout.to_dict(sysid_param_matrix(sysid, axes)[0])
    params["k_w"] = float(sysid["k_wz"])
    return params


def sysid_param_matrix(sysid: dict, axes) -> np.ndarray:
    """Parameter vectors from sysid values, one row per env.

    `sysid` values may be scalars or (n_envs,) arrays (domain-randomized
    draws); returns an (n_envs, P) array in `ParamLayout` order.
    """
    axes = np.asarray(axes, dtype=float)
    n = len(axes)
    layout = ParamLayout(n)
    get = lambda key: np.atleast_1d(np.asarray(sysid[key], dtype=float))
    n_envs = len(get("k_wz"))
    per_motor = lambda name: np.stack([get(f"{name}{i + 1}") for i in range(n)], axis=1)  # (envs, N)

    vec = np.zeros((n_envs, layout.size))
    vec[:, layout.slices["axis"]] = axes.ravel()
    k_wvec = np.stack([get("k_wx"), get("k_wy"), get("k_wz")], axis=1)  # (envs, 3)
    vec[:, layout.slices["f_coef"]] = (axes[None] * k_wvec[:, None, :]).reshape(n_envs, 3 * n)
    vec[:, layout.slices["m_coef"]] = np.stack(
        [per_motor("k_p"), per_motor("k_q"), per_motor("k_r")], axis=2).reshape(n_envs, 3 * n)
    vec[:, layout.slices["m_dw_coef"]] = np.stack(
        [per_motor("k_pd"), per_motor("k_qd"), per_motor("k_rd")], axis=2).reshape(n_envs, 3 * n)
    vec[:, layout.slices["k_drag"]] = np.stack([get("k_x"), get("k_y"), get("k_z")], axis=1)
    vec[:, layout.slices["inflow_coef"]] = INFLOW_COEF
    for name in ("tau", "k", "w_min", "w_max"):
        vec[:, layout.slices[name]] = get(name)[:, None]
    return vec


class ParamLayout:
    """Order of the flat parameter vector for an N-motor drone.

    Per-motor blocks are row-major (motor, xyz). `randomize_mask` marks the
    entries that domain randomization scales: everything except the
    thrust axes (geometry) and the inflow constant, which the reference
    hardcodes.
    """

    BLOCKS = [
        ("axis", 3, True), ("f_coef", 3, True), ("m_coef", 3, True), ("m_dw_coef", 3, True),
        ("k_drag", 3, False), ("inflow_coef", 1, False),
        ("tau", 1, False), ("k", 1, False), ("w_min", 1, False), ("w_max", 1, False),
    ]
    FIXED = ("axis", "inflow_coef")

    def __init__(self, n_motors: int):
        self.n = n_motors
        self.slices = {}
        start = 0
        for name, width, per_motor in self.BLOCKS:
            size = width * (n_motors if per_motor else 1)
            self.slices[name] = slice(start, start + size)
            start += size
        self.size = start
        self.randomize_mask = np.ones(self.size, dtype=bool)
        for name in self.FIXED:
            self.randomize_mask[self.slices[name]] = False
        self.k_index = self.slices["k"].start

    def to_vector(self, params: dict) -> np.ndarray:
        vec = np.zeros(self.size)
        for name, _, _ in self.BLOCKS:
            vec[self.slices[name]] = np.ravel(params[name])
        return vec

    def to_dict(self, vec) -> dict:
        vec = np.asarray(vec, dtype=float)
        out = {"n_motors": self.n}
        for name, width, per_motor in self.BLOCKS:
            block = vec[self.slices[name]]
            if per_motor:
                out[name] = block.reshape(self.n, width).tolist()
            elif width > 1:
                out[name] = block.tolist()
            else:
                out[name] = float(block[0])
        return out


def params_to_vector(params: dict) -> np.ndarray:
    return ParamLayout(params["n_motors"]).to_vector(params)
