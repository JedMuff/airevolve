"""Drop-in replacement for DroneSimulator that implements
optimal_quad_control_RL's sysid'd 5-inch dynamics verbatim.

Used by reference_dynamics_env.ReferenceDynamicsGateEnv for the session 4
parity experiment in RL_TRAINING_FIXES.md. Lives under experimentation/ and
is not imported by airevolve's runtime path.

The dynamics is a faithful port of
optimal_quad_control_RL/quad_race_env.py:22-103 with constants from
optimal_quad_control_RL/randomization.py:5-10 (params_5inch). Differences
from airevolve's DroneSimulator:

* No mass / inertia in the equations of motion. F and M are treated as
  accelerations directly (the reference is a normalized sysid'd model).
  `mass` and `inertia` are kept as decorative attributes for env logging
  and observation construction.
* `dynamics_func` takes the *full* 16-dim state and the 4-dim action;
  motor speed dynamics are baked into the symbolic equations (action ->
  Wc -> first-order lag -> W).
* Per-motor sysid'd moment coefficients (k_p1..4, k_q1..4, k_r1..8)
  including the Mz reaction-torque derivative term.
"""
from __future__ import annotations

import numpy as np
from sympy import (
    Array,
    Matrix,
    cos,
    lambdify,
    sin,
    sqrt,
    symbols,
    tan,
)


# fixed_params_5inch from optimal_quad_control_RL/randomization.py:5-10
PARAMS_5INCH = {
    "k_w": 2.49e-06,
    "k_x": 4.85e-05,
    "k_y": 7.28e-05,
    "k_p1": 6.55e-05,
    "k_p2": 6.61e-05,
    "k_p3": 6.36e-05,
    "k_p4": 6.67e-05,
    "k_q1": 5.28e-05,
    "k_q2": 5.86e-05,
    "k_q3": 5.05e-05,
    "k_q4": 5.89e-05,
    "k_r1": 1.07e-02,
    "k_r2": 1.07e-02,
    "k_r3": 1.07e-02,
    "k_r4": 1.07e-02,
    "k_r5": 1.97e-03,
    "k_r6": 1.97e-03,
    "k_r7": 1.97e-03,
    "k_r8": 1.97e-03,
    "w_min": 238.49,
    "w_max": 3295.50,
    "k": 0.95,
    "tau": 0.04,
}

# fixed_params_3inch from optimal_quad_control_RL/randomization.py:29-35
PARAMS_3INCH = {
    "k_w": 6.00e-07,
    "k_x": 3.36e-05,
    "k_y": 3.73e-05,
    "k_p1": 2.57e-05,
    "k_p2": 2.51e-05,
    "k_p3": 2.72e-05,
    "k_p4": 2.00e-05,
    "k_q1": 9.10e-06,
    "k_q2": 9.96e-06,
    "k_q3": 1.17e-05,
    "k_q4": 8.21e-06,
    "k_r1": 9.64e-03,
    "k_r2": 9.64e-03,
    "k_r3": 9.64e-03,
    "k_r4": 9.64e-03,
    "k_r5": 1.14e-03,
    "k_r6": 1.14e-03,
    "k_r7": 1.14e-03,
    "k_r8": 1.14e-03,
    "w_min": 305.40,
    "w_max": 4887.57,
    "k": 0.84,
    "tau": 0.04,
}

# Decorative — not used by dynamics_func, only by env code that reads
# self.drone_sim.mass / self.drone_sim.inertia for logging or observation.
# 5-inch sysid values picked per user (see plan).
DECORATIVE_MASS = 0.6
DECORATIVE_INERTIA = np.diag([0.0025, 0.0021, 0.0043])


def derive_params_2inch_from_airevolve(arm_length: float = 0.11) -> tuple[dict, float, np.ndarray]:
    """Map airevolve/dronehover prop2 hardware specs into the reference's
    22-parameter sysid form. No real 2-inch sysid set exists anywhere — this
    is the best-effort substitute, mixing parameters we *can* derive from
    airevolve's static hardware specs (k_f, k_m, mass, inertia, geometry)
    with parameters we *cannot* derive (because airevolve doesn't model the
    underlying physics — see below) which we borrow from
    optimal_quad_control_RL's 3-inch sysid (closest validated reference).

    Returns (params_dict, mass, inertia_matrix). The mass/inertia are
    airevolve's actual computed values for the canonical 2-inch quad and
    are used as the env's decorative drone_sim.mass / drone_sim.inertia.

    Derivation per parameter:

    * `k_w = k_f / m` — F = k_w·sum(W²) is *acceleration* in the reference,
      airevolve has F = k_f·sum(W²) as a *force* and divides by mass later.
    * `k_p1..4 = k_f · arm_y / Ixx`, `k_q1..4 = k_f · arm_x / Iyy` —
      symmetric X-quad, all four motors at the same |arm|. Result includes
      the implicit /Ixx, /Iyy that the reference's per-axis dp = Mx
      formulation requires.
    * `k_r1..4 = 2 · k_m · W_hover / Izz` — airevolve has Mz ∝ W² (from
      k_m·sum(spin·W²)); the reference has Mz ∝ W (linear). Linearize at
      hover throttle: dMz/dW ≈ 2·k_m·W_hover, then divide by Izz.
    * `tau = 0.04`, `w_max = 5000` — directly from prop2 specs / matched.

    Borrowed from 3-inch sysid (airevolve doesn't model these):

    * `k_x, k_y` — body drag aerodynamics. Airevolve has k_xn=0.16,
      k_yn=0.24 hardcoded as session-3 calibration but they aren't physically
      grounded for 2-inch. 3-inch values (3.36e-5, 3.73e-5) are real sysid.
    * `k_r5..8` — motor reaction torque on Mz. Airevolve absent.
    * `k` — sqrt-poly motor-response nonlinearity. Airevolve uses k=0.
    * `w_min` — idle motor speed (props always spinning). Airevolve uses 0.
    """
    # Lazy import — keeps reference_drone_sim.py importable even if airevolve's
    # internals aren't on sys.path.
    from airevolve.simulator.simulation.drone_configuration import DroneConfiguration
    from airevolve.simulator.simulation.propeller_data import (
        create_standard_propeller_config,
    )

    propellers = create_standard_propeller_config(
        "quad", arm_length=arm_length, prop_size=2
    )
    cfg = DroneConfiguration(propellers)

    m = float(cfg.mass)
    Ixx = float(cfg.inertia_matrix[0, 0])
    Iyy = float(cfg.inertia_matrix[1, 1])
    Izz = float(cfg.inertia_matrix[2, 2])

    k_f, k_m = cfg.propellers[0]["constants"]
    w_max = float(cfg.propellers[0]["wmax"])

    # Symmetric X-quad: |x| = |y| for all motors at arm_length × {cos, sin}(45°)
    arm_x = abs(float(cfg.propellers[0]["loc"][0]))
    arm_y = abs(float(cfg.propellers[0]["loc"][1]))

    # Hover motor speed (per motor). Linearization point for the Mz term.
    g = 9.81
    F_hover_per_motor = m * g / 4.0
    W_hover = (F_hover_per_motor / k_f) ** 0.5

    k_w = k_f / m
    k_p_all = k_f * arm_y / Ixx
    k_q_all = k_f * arm_x / Iyy
    k_r_linear = 2.0 * k_m * W_hover / Izz

    # Borrowed-from-3-inch:
    k_x_borrow = PARAMS_3INCH["k_x"]
    k_y_borrow = PARAMS_3INCH["k_y"]
    k_r_react_borrow = PARAMS_3INCH["k_r5"]  # k_r5..8 all equal in sysid
    k_nonlin_borrow = PARAMS_3INCH["k"]
    w_min_borrow = PARAMS_3INCH["w_min"]

    params = {
        "k_w": k_w,
        "k_x": k_x_borrow,
        "k_y": k_y_borrow,
        "k_p1": k_p_all, "k_p2": k_p_all, "k_p3": k_p_all, "k_p4": k_p_all,
        "k_q1": k_q_all, "k_q2": k_q_all, "k_q3": k_q_all, "k_q4": k_q_all,
        "k_r1": k_r_linear, "k_r2": k_r_linear,
        "k_r3": k_r_linear, "k_r4": k_r_linear,
        "k_r5": k_r_react_borrow, "k_r6": k_r_react_borrow,
        "k_r7": k_r_react_borrow, "k_r8": k_r_react_borrow,
        "tau": 0.04,
        "k": k_nonlin_borrow,
        "w_min": w_min_borrow,
        "w_max": w_max,
    }
    return params, m, np.array(cfg.inertia_matrix)


# Decorative defaults for 3-inch (no canonical mass; rough 5-inch/3-inch ratio).
DECORATIVE_MASS_3INCH = 0.3
DECORATIVE_INERTIA_3INCH = np.diag([0.0010, 0.0009, 0.0017])


def get_variant(variant: str) -> tuple[dict, float, np.ndarray]:
    """Return (params, decorative_mass, decorative_inertia) for a named
    variant. Centralizes the variant->config mapping so the env subclass and
    CLI both agree.
    """
    if variant == "5inch":
        return PARAMS_5INCH, DECORATIVE_MASS, DECORATIVE_INERTIA
    if variant == "3inch":
        return PARAMS_3INCH, DECORATIVE_MASS_3INCH, DECORATIVE_INERTIA_3INCH
    if variant == "2inch_derived":
        return derive_params_2inch_from_airevolve()
    raise ValueError(
        f"unknown reference-params variant: {variant!r}. "
        "Pick one of: 5inch, 3inch, 2inch_derived"
    )

# Reference's normalization constants (quad_race_env.py:41-42)
W_MIN_N = 0.0
W_MAX_N = 3000.0


def _build_dynamics_func(params: dict) -> callable:
    """Build the lambdified 16-dim state derivative function.

    Verbatim port of optimal_quad_control_RL/quad_race_env.py:22-103. The
    23-parameter dict is substituted symbolically so the returned function
    only takes (state[16], action[4]).
    """
    state = symbols("x y z v_x v_y v_z phi theta psi p q r w1 w2 w3 w4")
    x, y, z, vx, vy, vz, phi, theta, psi, p, q, r, w1, w2, w3, w4 = state
    control = symbols("U_1 U_2 U_3 U_4")
    u1, u2, u3, u4 = control

    g = 9.81

    Rx = Matrix([[1, 0, 0], [0, cos(phi), -sin(phi)], [0, sin(phi), cos(phi)]])
    Ry = Matrix([[cos(theta), 0, sin(theta)], [0, 1, 0], [-sin(theta), 0, cos(theta)]])
    Rz = Matrix([[cos(psi), -sin(psi), 0], [sin(psi), cos(psi), 0], [0, 0, 1]])
    R = Rz * Ry * Rx

    vbx, vby, _vbz = R.T @ Matrix([vx, vy, vz])

    W1 = (w1 + 1) / 2 * (W_MAX_N - W_MIN_N) + W_MIN_N
    W2 = (w2 + 1) / 2 * (W_MAX_N - W_MIN_N) + W_MIN_N
    W3 = (w3 + 1) / 2 * (W_MAX_N - W_MIN_N) + W_MIN_N
    W4 = (w4 + 1) / 2 * (W_MAX_N - W_MIN_N) + W_MIN_N

    U1 = (u1 + 1) / 2
    U2 = (u2 + 1) / 2
    U3 = (u3 + 1) / 2
    U4 = (u4 + 1) / 2

    k_x = params["k_x"]
    k_y = params["k_y"]
    k_w = params["k_w"]
    k_p = [params[f"k_p{i}"] for i in (1, 2, 3, 4)]
    k_q = [params[f"k_q{i}"] for i in (1, 2, 3, 4)]
    k_r = [params[f"k_r{i}"] for i in (1, 2, 3, 4, 5, 6, 7, 8)]
    tau = params["tau"]
    k = params["k"]
    w_min = params["w_min"]
    w_max = params["w_max"]

    Wc1 = (w_max - w_min) * sqrt(k * U1**2 + (1 - k) * U1) + w_min
    Wc2 = (w_max - w_min) * sqrt(k * U2**2 + (1 - k) * U2) + w_min
    Wc3 = (w_max - w_min) * sqrt(k * U3**2 + (1 - k) * U3) + w_min
    Wc4 = (w_max - w_min) * sqrt(k * U4**2 + (1 - k) * U4) + w_min

    d_W1 = (Wc1 - W1) / tau
    d_W2 = (Wc2 - W2) / tau
    d_W3 = (Wc3 - W3) / tau
    d_W4 = (Wc4 - W4) / tau

    d_w1 = d_W1 / (W_MAX_N - W_MIN_N) * 2
    d_w2 = d_W2 / (W_MAX_N - W_MIN_N) * 2
    d_w3 = d_W3 / (W_MAX_N - W_MIN_N) * 2
    d_w4 = d_W4 / (W_MAX_N - W_MIN_N) * 2

    T = -k_w * (W1**2 + W2**2 + W3**2 + W4**2)
    Dx = -k_x * vbx * (W1 + W2 + W3 + W4)
    Dy = -k_y * vby * (W1 + W2 + W3 + W4)

    Mx = -k_p[0] * W1**2 - k_p[1] * W2**2 + k_p[2] * W3**2 + k_p[3] * W4**2
    My = -k_q[0] * W1**2 + k_q[1] * W2**2 - k_q[2] * W3**2 + k_q[3] * W4**2
    Mz = (
        -k_r[0] * W1
        + k_r[1] * W2
        + k_r[2] * W3
        - k_r[3] * W4
        - k_r[4] * d_W1
        + k_r[5] * d_W2
        + k_r[6] * d_W3
        - k_r[7] * d_W4
    )

    d_x = vx
    d_y = vy
    d_z = vz

    accel = Matrix([0, 0, g]) + R @ Matrix([Dx, Dy, T])
    d_vx, d_vy, d_vz = accel

    d_phi = p + q * sin(phi) * tan(theta) + r * cos(phi) * tan(theta)
    d_theta = q * cos(phi) - r * sin(phi)
    d_psi = q * sin(phi) / cos(theta) + r * cos(phi) / cos(theta)

    d_p = Mx
    d_q = My
    d_r = Mz

    f = [
        d_x, d_y, d_z,
        d_vx, d_vy, d_vz,
        d_phi, d_theta, d_psi,
        d_p, d_q, d_r,
        d_w1, d_w2, d_w3, d_w4,
    ]
    return lambdify((Array(state), Array(control)), Array(f), "numpy")


class ReferenceDroneSim:
    """Mirrors DroneSimulator's interface but swaps in the reference's
    sysid'd dynamics. Constructor is not parameterized by morphology; the
    dynamics is fixed by `params` (default: PARAMS_5INCH).

    Args:
        dt: integration timestep (seconds).
        params: 22-parameter sysid dict (see PARAMS_5INCH for the schema).
            Defaults to PARAMS_5INCH.
        mass: decorative drone mass for env logging/observation. Defaults
            to DECORATIVE_MASS (5-inch). Not consumed by dynamics_func.
        inertia: decorative inertia matrix (3x3). Defaults to
            DECORATIVE_INERTIA. Not consumed by dynamics_func.
    """

    def __init__(
        self,
        dt: float = 0.01,
        params: dict | None = None,
        mass: float | None = None,
        inertia: np.ndarray | None = None,
    ):
        self.params = dict(params) if params is not None else dict(PARAMS_5INCH)
        self.dt = dt
        self.g = 9.81
        self.num_motors = 4

        # Decorative interface-compat attributes. Not consumed by dynamics_func.
        self.mass = float(mass) if mass is not None else DECORATIVE_MASS
        self.inertia = (
            np.array(inertia) if inertia is not None else DECORATIVE_INERTIA.copy()
        )
        self.center_of_gravity = np.zeros(3)
        # Bf / Bm have no meaning in the reference dynamics (per-motor
        # moment coefficients replace the allocation matrix). Provide zero
        # placeholders so callers reading these attributes don't blow up.
        self.Bf = np.zeros((3, self.num_motors))
        self.Bm = np.zeros((3, self.num_motors))

        self.dynamics_func = _build_dynamics_func(self.params)
