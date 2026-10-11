"""Lee controller with tilted motors.

Tilted motors push sideways as well as up. The plain mixer only controls
body-z thrust and the three torques, so such drones drift off position. The
thrust-axis allocation in LeeGeometricControl aligns the drone's real thrust
axis with the commanded force and suppresses the sideways force. These tests
check that:

1. Flat drones keep the plain mixer and an identity attitude correction, so
   their behaviour is unchanged.
2. A tilted drone (evolved figure-8 hexacopter whose thrust axis is ~32° off
   body −z) holds a hover setpoint with the fix and drifts without it.
3. Commanded thrust and torques are reproduced by the bounded allocation.

Usage: .venv/bin/python unit_tests/test_lee_tilted_allocation.py
"""
from __future__ import annotations

import numpy as np

from airevolve.controllers.lee_control.lee_controller import LeeGeometricControl
from airevolve.controllers.utils.wind_model import Wind
from airevolve.simulator.simulation import DroneInterface
from airevolve.simulator.simulation.propeller_data import genome_to_propellers

# [magnitude, arm_yaw, arm_pitch, motor_pitch, motor_yaw, direction]
FLAT_QUAD = np.array([[0.09, np.pi / 4 + k * np.pi / 2, 0.0, np.pi, 0.0, k % 2] for k in range(4)])
FLAT_HEX = np.array([[0.11, k * np.pi / 3, 0.0, np.pi, 0.0, k % 2] for k in range(6)])
# figure8 spherical rep10 gen42 ind0 (airevolve030326/v2 elites).
TILTED_HEX = np.array([
    [0.08646461925119353, 3.0723510807368912, 1.004914214973649e-16, 1.7129426820307687, -2.829817719083386, 0.0],
    [0.08149770806113024, 1.4658921539735976, 3.077457198631987, -1.4417267612649514, -0.6772066389956457, 0.0],
    [0.06820866303973412, 1.4726291414507209, 1.0470896506666824e-16, 2.1450043154407314, -1.4316243381667004, 0.0],
    [0.06250396319107676, 2.201529767073069, 5.105419638288982e-17, 2.8600109399354157, 2.9292383508380953, 0.0],
    [0.060248895138481494, -1.3971908112398679, 4.454179348537767e-17, -3.12152886243051, -1.063207078437407, 1.0],
    [0.0734402687607353, -0.2594901739852049, 8.211849269873757e-17, -2.5164153163016882, -0.7466863431247774, 1.0],
])


def _make(genome):
    quad = DroneInterface(0, propellers=genome_to_propellers(genome))
    return quad, LeeGeometricControl(quad, yawType=1, orient="NED", auto_scale_gains=True)


def _disable_fix(ctrl, quad):
    ctrl.bounded_allocation = False
    ctrl.mixer_inv = quad.params["mixerFMinv"]
    for c in (ctrl.position_controller, ctrl.velocity_controller, ctrl.attitude_controller):
        c.thrust_axis_correction = np.array([0.0, 0.0, 0.0, 1.0])


def _hover_drift(genome, fix=True, seconds=8.0, settle=1.0, dt=0.005):
    """Mean distance (m) from a fixed position setpoint over the last
    ``settle`` seconds of a hover (after the start-up transient: the drone
    starts level and has to tip onto its thrust axis)."""
    quad, ctrl = _make(genome)
    if not fix:
        _disable_fix(ctrl, quad)
    start = np.zeros(3)
    quad.drone_sim.set_state(position=start, velocity=np.zeros(3),
                             attitude=np.zeros(3), angular_velocity=np.zeros(3))
    quad._update_state_variables()
    sDes = np.zeros(19)
    sDes[0:3] = start
    wind = Wind('None', 2.0, 90, -15)
    ctrl.controller(sDes, quad, "xyz_pos", dt)
    t, errors = 0.0, []
    for _ in range(int(seconds / dt)):
        quad.update(t, dt, ctrl.w_cmd, wind)
        t += dt
        ctrl.controller(sDes, quad, "xyz_pos", dt)
        err = float(np.linalg.norm(quad.pos - start))
        if not np.isfinite(err) or err > 50:
            return np.inf
        if t > seconds - settle:
            errors.append(err)
    return float(np.mean(errors))


def test_flat_drones_keep_plain_mixer():
    for genome in (FLAT_QUAD, FLAT_HEX):
        quad, ctrl = _make(genome)
        assert ctrl.mixer_inv is quad.params["mixerFMinv"]
        assert not ctrl.bounded_allocation
        np.testing.assert_array_equal(ctrl.thrust_axis_body, [0.0, 0.0, -1.0])
        for c in (ctrl.position_controller, ctrl.velocity_controller, ctrl.attitude_controller):
            np.testing.assert_array_equal(c.thrust_axis_correction, [0.0, 0.0, 0.0, 1.0])


def test_tilted_drone_thrust_axis_and_trim():
    quad, ctrl = _make(TILTED_HEX)
    assert ctrl.can_trim and ctrl.bounded_allocation
    tilt = np.degrees(np.arccos(-ctrl.thrust_axis_body[2]))
    assert 20.0 < tilt < 45.0, tilt
    # The attitude correction maps the thrust axis onto body −z.
    from airevolve.controllers.lee_control.lee_math_utils import quat_rotate
    q_c = ctrl.position_controller.thrust_axis_correction
    np.testing.assert_allclose(quat_rotate(q_c, ctrl.thrust_axis_body), [0, 0, -1], atol=1e-9)


def test_bounded_allocation_reproduces_hover_wrench():
    quad, ctrl = _make(TILTED_HEX)
    Bf, Bm = quad.drone_sim.Bf, quad.drone_sim.Bm
    weight = quad.params["mB"] * quad.params["g"]
    w2 = ctrl._allocate_bounded(np.array([weight, 0.0, 0.0, 0.0]))
    assert np.all(w2 >= ctrl._alloc_lo - 1e-12) and np.all(w2 <= ctrl._alloc_hi + 1e-12)
    force, torque = Bf @ w2, Bm @ w2
    np.testing.assert_allclose(force @ ctrl.thrust_axis_body, weight, rtol=1e-3)
    assert np.linalg.norm(torque) < 1e-4 * weight
    # Sideways force at hover is (almost) gone; the plain mixer leaves ~27 % of the weight.
    sideways = np.linalg.norm(force - (force @ ctrl.thrust_axis_body) * ctrl.thrust_axis_body)
    plain = Bf @ (quad.params["mixerFMinv"] @ np.array([weight, 0, 0, 0]))
    assert sideways < 0.01 * weight
    assert np.linalg.norm(plain[:2]) > 0.2 * weight


def test_tilted_drone_holds_hover_only_with_fix():
    with_fix = _hover_drift(TILTED_HEX, fix=True)
    without = _hover_drift(TILTED_HEX, fix=False)
    assert with_fix < 0.1, f"drifted {with_fix:.3f} m with the fix"
    assert without > 5 * with_fix, f"plain mixer drift {without:.3f} m vs {with_fix:.3f} m"


def test_flat_drone_hover_unchanged():
    assert _hover_drift(FLAT_QUAD) < 0.01


if __name__ == "__main__":
    test_flat_drones_keep_plain_mixer()
    test_tilted_drone_thrust_axis_and_trim()
    test_bounded_allocation_reproduces_hover_wrench()
    test_tilted_drone_holds_hover_only_with_fix()
    test_flat_drone_hover_unchanged()
    print("lee tilted allocation OK")
