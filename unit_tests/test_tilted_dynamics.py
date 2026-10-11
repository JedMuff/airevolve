"""Tilt-aware reference-form dynamics.

The reference-form dynamics originally assumed every motor thrusts along
body -z, so yaw could only come from spin direction. Evolved morphologies
often cant their motors and spin all props the same way, steering yaw with
the lever arm of the tilted thrust instead. These tests check that:

1. For flat motors the tilt-aware coefficients reduce exactly to the
   original scalar ones (the parity test covers the full dynamics_func).
2. A tangentially canted motor produces lateral force and the r × F yaw
   moment.
3. An evolved all-ccw hexacopter (top-ranked NEAT figure-8 individual from
   the March 2026 runs) can hold zero angular acceleration at hover thrust,
   while the same morphology with flat motors cannot cancel its yaw.

Usage: /home/jed/miniconda3/envs/isaaclab/bin/python unit_tests/test_tilted_dynamics.py
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from airevolve.simulator.simulation.drone_simulator import DroneSimulator
from airevolve.simulator.simulation.dynamics_params import W_MAX_N, W_MIN_N
from airevolve.simulator.simulation.propeller_data import (
    create_standard_propeller_config,
    genome_to_propellers,
    get_extended_prop_params,
)

# [magnitude, arm_yaw, arm_pitch, motor_pitch, motor_yaw, direction]
MARCH_TOP1_GENOME = np.array([
    [0.13206512817875496, -2.5629557009052686, 0.7386619582705095, 2.701704076598013, 2.7636683495737207, 0.0],
    [0.10370805329360139, -0.8585726869419439, -0.2918436824491207, 2.756957514853005, 2.8258811743548136, 0.0],
    [0.16043994955940882, -2.4994473627558413, 1.0406342043408074, 2.520113508065138, -0.14604556990988232, 0.0],
    [0.1584540168912637, -1.0928429708014804, 1.3707038996930059, 2.9693388086202708, -0.8152555310919941, 0.0],
    [0.12189064470924209, 1.4957111814803943, -0.7035688148438511, 2.9289732693517374, 3.1257781332483487, 0.0],
    [0.1476405296968997, 1.082791598942516, -0.4597251870897445, 2.6222762228682885, -0.23753858819667287, 0.0],
])


def march_propellers(flatten=False):
    """Propellers for MARCH_TOP1_GENOME, optionally with all motors flat."""
    props = genome_to_propellers(MARCH_TOP1_GENOME)
    if flatten:
        for prop in props:
            prop["dir"][:3] = [0.0, 0.0, -1.0]
    return props


def steady_derivative(sim, W):
    """State derivative at level hover with motors held steady at speeds W."""
    p = sim.params
    w_min, w_max, k = p["w_min"], p["w_max"], p["k"]
    # Invert Wc = (w_max - w_min)·sqrt(k·U² + (1-k)·U) + w_min so dW = 0.
    s = ((W - w_min) / (w_max - w_min)) ** 2
    U = (-(1 - k) + np.sqrt((1 - k) ** 2 + 4 * k * s)) / (2 * k)
    state = np.zeros(12 + sim.num_motors)
    state[12:] = 2 * (W - W_MIN_N) / (W_MAX_N - W_MIN_N) - 1
    return np.asarray(sim.dynamics_func(state, 2 * U - 1), dtype=float)


def best_hover_residual(sim):
    """Min over motor speeds of |[d_vz, d_p, d_q, d_r]|."""
    p = sim.params
    w_hover = np.sqrt(sim.g / sim.num_motors / -p["f_coef"][0][2])
    res = least_squares(
        lambda W: steady_derivative(sim, W)[[5, 9, 10, 11]],
        x0=np.full(sim.num_motors, w_hover),
        bounds=(p["w_min"], p["w_max"]),
        xtol=1e-12, ftol=1e-12, gtol=1e-12,
    )
    return steady_derivative(sim, res.x)


def test_flat_coefficients_match_scalar_form():
    sim = DroneSimulator(
        propellers=create_standard_propeller_config("quad", arm_length=0.11, prop_size=2)
    )
    p = sim.params
    for i in range(sim.num_motors):
        np.testing.assert_allclose(p["f_coef"][i], [0, 0, -p["k_w"]], atol=1e-15)
        np.testing.assert_allclose(
            p["m_thr_coef"][i], [p["k_p_signed"][i], p["k_q_signed"][i], 0], rtol=1e-12, atol=1e-15
        )
        np.testing.assert_allclose(p["m_spin_coef"][i], [0, 0, p["k_r_signed"][i]], atol=1e-15)
        np.testing.assert_allclose(p["m_react_coef"][i], [0, 0, p["k_r_react_signed"][i]], atol=1e-15)


def test_canted_motor_gives_lateral_force_and_lever_yaw():
    props = create_standard_propeller_config("quad", arm_length=0.11, prop_size=2)
    loc = np.array(props[0]["loc"], dtype=float)
    tangent = np.cross([0.0, 0.0, 1.0], loc)
    tangent /= np.linalg.norm(tangent)
    tilt = np.radians(20)
    d = np.cos(tilt) * np.array([0.0, 0.0, -1.0]) + np.sin(tilt) * tangent
    props[0]["dir"] = list(d) + [props[0]["dir"][3]]
    sim = DroneSimulator(propellers=props)
    p = sim.params

    k_f = p["k_w"] * sim.mass
    np.testing.assert_allclose(p["f_coef"][0], k_f / sim.mass * d, rtol=1e-12)
    r = loc - np.asarray(sim.center_of_gravity)
    expected_mz = k_f * np.cross(r, d)[2] / sim.inertia[2, 2]
    assert abs(expected_mz) > 0
    np.testing.assert_allclose(p["m_thr_coef"][0][2], expected_mz, rtol=1e-12)


def test_same_spin_hover_yaw_torque_is_physical():
    """At hover speed each prop's drag torque is k_m·W_hover², not twice that."""
    props = create_standard_propeller_config("quad", arm_length=0.11, prop_size=2)
    for prop in props:
        prop["dir"][3] = "ccw"
    sim = DroneSimulator(propellers=props)
    k_f, k_m = get_extended_prop_params(2)["constants"]
    W_hover = np.sqrt(sim.mass * sim.g / sim.num_motors / k_f)

    sd = steady_derivative(sim, np.full(sim.num_motors, W_hover))
    expected = sim.num_motors * k_m * W_hover**2 / sim.inertia[2, 2]
    np.testing.assert_allclose(sd[11], expected, rtol=1e-9)


def test_evolved_same_spin_drone_can_cancel_yaw():
    tilted = DroneSimulator(propellers=march_propellers())
    sd = best_hover_residual(tilted)
    assert abs(sd[5]) < 1e-4, f"cannot hold hover thrust: d_vz={sd[5]:.3e}"
    assert np.max(np.abs(sd[9:12])) < 1e-4, f"cannot balance moments: {sd[9:12]}"

    # Same arms and spins but flat motors: all-ccw drag can't be cancelled.
    flat = DroneSimulator(propellers=march_propellers(flatten=True))
    sd_flat = best_hover_residual(flat)
    assert abs(sd_flat[11]) > 1.0, f"flat same-spin drone unexpectedly balanced yaw: {sd_flat[11]:.3e}"


if __name__ == "__main__":
    test_flat_coefficients_match_scalar_form()
    test_canted_motor_gives_lateral_force_and_lever_yaw()
    test_same_spin_hover_yaw_torque_is_physical()
    test_evolved_same_spin_drone_can_cancel_yaw()
    print("tilted dynamics OK")
