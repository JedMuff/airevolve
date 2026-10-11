"""Genome → Fusion 360 arm parameters.

Checks that the CAD parameters from ``genome_to_fusion_params`` rebuild
exactly the motor positions and thrust directions the simulator uses
(``genome_to_propellers``), and that the angle conventions have the
documented meaning and signs.

Usage: /home/jed/miniconda3/envs/isaaclab/bin/python unit_tests/test_fusion_params.py
"""
from __future__ import annotations

import numpy as np

from airevolve.fabrication.fusion_params import (
    arm_geometry_in_g,
    fusion_params_to_propellers,
    genome_to_fusion_params,
    hole_pair_directions,
)
from airevolve.simulator.simulation.drone_simulator import DroneSimulator
from airevolve.simulator.simulation.propeller_data import genome_to_propellers

R_MM = 30.0

# Flat X-quad, motors thrusting up (motor_pitch = π), alternating spin.
STANDARD_QUAD = np.array([
    [0.1, np.pi / 4 + k * np.pi / 2, 0.0, np.pi, 0.0, k % 2] for k in range(4)
])


def _assert_matches_simulator(genome, plate_radius_mm=R_MM):
    arms = genome_to_fusion_params(genome, plate_radius_mm)
    rebuilt = fusion_params_to_propellers(arms, plate_radius_mm)
    reference = genome_to_propellers(genome)
    assert len(rebuilt) == len(reference)
    for got, want in zip(rebuilt, reference):
        np.testing.assert_allclose(got["loc"], want["loc"], atol=1e-9)
        np.testing.assert_allclose(got["dir"][:3], want["dir"][:3], atol=1e-9)
        assert got["dir"][3] == want["dir"][3]
    return arms


def test_random_genomes_round_trip_to_simulator():
    rng = np.random.default_rng(0)
    # Limits of SphericalAngularDroneGenomeHandler (magnitude in metres).
    low = np.array([0.055, -np.pi, -np.pi / 2, -np.pi, -np.pi, 0.0])
    high = np.array([0.17, np.pi, np.pi / 2, np.pi, np.pi, 1.0])
    for _ in range(200):
        genome = rng.uniform(low, high, size=(rng.integers(3, 9), 6))
        genome[:, 5] = np.round(genome[:, 5])
        _assert_matches_simulator(genome)


def test_nan_rows_are_skipped():
    genome = np.vstack([STANDARD_QUAD, np.full((2, 6), np.nan)])
    assert len(_assert_matches_simulator(genome)) == 4


def test_standard_quad_has_zero_motor_angles():
    arms = _assert_matches_simulator(STANDARD_QUAD)
    for k, arm in enumerate(arms):
        assert arm.slot == 2 + 4 * k
        assert np.isclose(arm.slot_angle, 45 + 90 * k)
        assert np.isclose(arm.arm_yaw_offset, 0.0, atol=1e-9)
        assert np.isclose(arm.arm_pitch, 0.0)
        assert np.isclose(arm.arm_length, 100.0 - R_MM)
        assert np.isclose(arm.arm_roll, 0.0, atol=1e-9)
        assert np.isclose(arm.motor_pitch, 0.0, atol=1e-9)
        assert not arm.warnings
    for prop in fusion_params_to_propellers(arms, R_MM):
        np.testing.assert_allclose(prop["dir"][:3], [0, 0, -1], atol=1e-12)


def test_simulator_force_points_up_for_standard_quad():
    # dir is the force direction in the dynamics: an upright quad must get
    # an upward (−z in NED) force from every motor.
    sim = DroneSimulator(propellers=fusion_params_to_propellers(
        genome_to_fusion_params(STANDARD_QUAD, R_MM), R_MM))
    f_coef = np.array(sim.params["f_coef"])
    assert np.all(f_coef[:, 2] < 0)
    np.testing.assert_allclose(f_coef[:, :2], 0.0, atol=1e-12)


def test_tilt_toward_arm_tip_is_positive():
    # Arm along +x; thrust tipped 30° outward (toward +x) from vertical.
    # thrust = −(cos y sin p, sin y sin p, cos p) → p = π + 30°, y = 0.
    genome = np.array([[0.1, 0.0, 0.0, np.pi + np.radians(30), 0.0, 0]])
    (arm,) = _assert_matches_simulator(genome)
    assert np.isclose(arm.motor_pitch, 30.0)
    assert np.isclose(arm.arm_roll, 0.0, atol=1e-9)


def test_roll_follows_right_hand_rule_about_arm():
    # Arm along +x; thrust tipped 30° toward +y. A positive rotation about +x
    # takes +z toward −y, so this is a roll of −30°.
    genome = np.array([[0.1, 0.0, 0.0, np.pi + np.radians(30), np.pi / 2, 0]])
    (arm,) = _assert_matches_simulator(genome)
    assert np.isclose(arm.arm_roll, -30.0)
    assert np.isclose(arm.motor_pitch, 0.0, atol=1e-9)


def test_raised_arm_tilt_is_measured_from_rim_arm():
    # Arm raised 30°, motor thrust tilted 30° back toward the plate so it is
    # perpendicular to the genome arm. The real arm starts at the rim, so it
    # is steeper than 30° and the motor shows the difference as tilt.
    genome = np.array([[0.1, 0.0, np.radians(30), np.pi - np.radians(30), 0.0, 0]])
    (arm,) = _assert_matches_simulator(genome)
    pz, px = 100 * np.sin(np.radians(30)), 100 * np.cos(np.radians(30))
    assert np.isclose(arm.arm_pitch, np.degrees(np.arctan2(pz, px - R_MM)))
    expected_tilt = np.degrees(np.radians(30) - np.radians(arm.arm_pitch))
    assert np.isclose(arm.motor_pitch, -expected_tilt)
    assert np.isclose(arm.arm_roll, 0.0, atol=1e-9)


def _prop_rotation(spin, axis):
    """Prop angular-velocity direction: ccw = right-handed about the axis."""
    return (1.0 if spin == "ccw" else -1.0) * np.asarray(axis)


def test_motor_thrusting_back_along_arm_is_flipped():
    # Arm along +x; thrust tipped 40° back toward the plate (−x).
    genome = np.array([[0.1, 0.0, 0.0, np.pi - np.radians(40), 0.0, 0]])
    (arm,) = _assert_matches_simulator(genome)   # sim thrust/spin still reproduced
    assert arm.motor_flipped
    assert np.isclose(arm.motor_pitch, 40.0)       # shaft now leans toward the tip
    assert np.isclose(abs(arm.arm_roll), 180.0)  # and points down
    assert arm.spin == "cw" and arm.sim_spin == "ccw"

    (unflipped,) = genome_to_fusion_params(genome, R_MM, flip_motors=False)
    assert not unflipped.motor_flipped
    assert np.isclose(unflipped.motor_pitch, -40.0)
    assert unflipped.spin == "ccw"


def test_flipped_props_turn_the_same_way_in_the_world():
    rng = np.random.default_rng(1)
    low = np.array([0.055, -np.pi, -np.pi / 2, -np.pi, -np.pi, 0.0])
    high = np.array([0.17, np.pi, np.pi / 2, np.pi, np.pi, 1.0])
    flipped = []
    for _ in range(20):
        genome = rng.uniform(low, high, size=(16, 6))
        genome[:, 5] = np.round(genome[:, 5])
        arms = _assert_matches_simulator(genome)
        flipped += [a.motor_flipped for a in arms]
        for arm, prop in zip(arms, genome_to_propellers(genome)):
            _, _, shaft, thrust = arm_geometry_in_g(arm, R_MM)
            assert arm.motor_pitch >= -1e-6               # prop never on the arm side
            sim_thrust = np.array([prop["dir"][1], prop["dir"][0], -prop["dir"][2]])
            np.testing.assert_allclose(thrust, sim_thrust, atol=1e-9)
            np.testing.assert_allclose(_prop_rotation(arm.spin, shaft),
                                       _prop_rotation(prop["dir"][3], sim_thrust), atol=1e-9)
    assert any(flipped) and not all(flipped)


def test_arm_running_back_over_plate_is_flagged():
    # Motor almost straight above the plate rim, inside it: the arm has to
    # leave its slot pointing back toward the centre.
    genome = np.array([[0.06, 0.0, np.radians(80), np.pi, 0.0, 0]])
    (arm,) = _assert_matches_simulator(genome)
    assert abs(arm.arm_yaw_offset) > 90
    assert any("back over the plate" in w for w in arm.warnings)


def test_slots_are_discrete_and_unique():
    rng = np.random.default_rng(2)
    low = np.array([0.055, -np.pi, -np.pi / 2, -np.pi, -np.pi, 0.0])
    high = np.array([0.17, np.pi, np.pi / 2, np.pi, np.pi, 1.0])
    for _ in range(50):
        genome = rng.uniform(low, high, size=(rng.integers(3, 17), 6))
        genome[:, 5] = np.round(genome[:, 5])
        arms = _assert_matches_simulator(genome)
        slots = [a.slot for a in arms]
        assert len(set(slots)) == len(slots)
        for a in arms:
            assert np.isclose(a.slot_angle, 22.5 * a.slot)


def test_off_grid_yaw_gets_yaw_offset():
    genome = np.array([[0.1, np.radians(10), 0.0, np.pi, 0.0, 0]])
    (arm,) = _assert_matches_simulator(genome)
    assert arm.slot == 0
    motor = 100 * np.array([np.cos(np.radians(10)), np.sin(np.radians(10))])
    expected = np.degrees(np.arctan2(motor[1], motor[0] - R_MM))
    assert np.isclose(arm.arm_yaw_offset, expected)
    assert np.isclose(arm.arm_length, np.hypot(motor[0] - R_MM, motor[1]))
    # Thrust is straight up, which is perpendicular to the (flat) arm.
    assert np.isclose(arm.arm_roll, 0.0, atol=1e-9)
    assert np.isclose(arm.motor_pitch, 0.0, atol=1e-9)


def test_hole_pairs_are_perpendicular_to_shaft():
    rng = np.random.default_rng(3)
    low = np.array([0.055, -np.pi, -np.pi / 2, -np.pi, -np.pi, 0.0])
    high = np.array([0.17, np.pi, np.pi / 2, np.pi, np.pi, 1.0])
    genome = rng.uniform(low, high, size=(16, 6))
    for arm in genome_to_fusion_params(genome, R_MM):
        pivot, motor, shaft, _ = arm_geometry_in_g(arm, R_MM)
        u = (motor - pivot) / np.linalg.norm(motor - pivot)
        root, mount = hole_pair_directions(arm)
        assert np.isclose(root @ u, 0.0, atol=1e-9)      # root holes across the arm
        assert np.isclose(mount @ u, 0.0, atol=1e-9)     # mount holes across the arm
        assert np.isclose(mount @ root, 0.0, atol=1e-9)  # ... on the rolled side axis
        assert np.isclose(mount @ shaft, 0.0, atol=1e-9)  # motor pitches about them
        # The motor pitch is the shaft's rotation from w' about the mount-hole axis.
        assert np.isclose(np.degrees(np.arctan2(np.cross(root, shaft) @ mount, root @ shaft)),
                          arm.motor_pitch)


def test_standard_quad_hole_pairs_are_default():
    for arm in genome_to_fusion_params(STANDARD_QUAD, R_MM):
        root, mount = hole_pair_directions(arm)
        np.testing.assert_allclose(root, [0, 0, 1], atol=1e-12)
        _, motor, _, _ = arm_geometry_in_g(arm, R_MM)
        radial = np.r_[motor[:2], 0] / np.linalg.norm(motor[:2])
        side = np.cross([0, 0, 1], radial)
        np.testing.assert_allclose(mount, side, atol=1e-12)


def test_too_many_arms_raises():
    genome = np.array([[0.1, k * 2 * np.pi / 17, 0.0, np.pi, 0.0, 0] for k in range(17)])
    try:
        genome_to_fusion_params(genome, R_MM)
    except ValueError:
        return
    raise AssertionError("expected ValueError for 17 arms on 16 slots")


if __name__ == "__main__":
    test_random_genomes_round_trip_to_simulator()
    test_nan_rows_are_skipped()
    test_standard_quad_has_zero_motor_angles()
    test_simulator_force_points_up_for_standard_quad()
    test_tilt_toward_arm_tip_is_positive()
    test_roll_follows_right_hand_rule_about_arm()
    test_raised_arm_tilt_is_measured_from_rim_arm()
    test_motor_thrusting_back_along_arm_is_flipped()
    test_flipped_props_turn_the_same_way_in_the_world()
    test_arm_running_back_over_plate_is_flagged()
    test_slots_are_discrete_and_unique()
    test_off_grid_yaw_gets_yaw_offset()
    test_hole_pairs_are_perpendicular_to_shaft()
    test_standard_quad_hole_pairs_are_default()
    test_too_many_arms_raises()
    print("fusion params OK")
