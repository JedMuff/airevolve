"""Test that genome-to-DroneInterface conversion matches known-good reference."""
import numpy as np
from airevolve.evolution_tools.inspection_tools.utils import convert_to_cartesian, ENU_to_NED
from airevolve.evolution_tools.inspection_tools.morphological_descriptors.hovering_info import orientation_to_unit_vector


def _genome_to_propellers(genome):
    """Replicate the conversion logic from simulate_with_gains."""
    propellers = []
    for arm in genome:
        r, theta, phi, motor_pitch, motor_yaw, direction = arm
        ex, ey, ez = convert_to_cartesian(r, theta, phi)
        x, y, z = ENU_to_NED(ex, ey, ez)
        rot = "ccw" if direction < 0.5 else "cw"
        motor_dir = orientation_to_unit_vector(0.0, motor_pitch, motor_yaw)
        propellers.append({
            "loc": [float(x), float(y), float(z)],
            "dir": [float(motor_dir[0]), float(motor_dir[1]), float(motor_dir[2]), rot],
            "propsize": 2
        })
    return propellers


def test_standard_quad_genome_matches_create_2inch_quad():
    """A standard quad genome should produce the same propellers as create_2inch_quad()."""
    # Standard 2-inch quad as genome (ENU spherical coords)
    mag = 0.06 * np.sqrt(2)  # arm length to match [0.06, 0.06, 0] in NED
    genome = np.array([
        [mag,  np.pi/4,    0, 0, 0, 0],    # NED: (+0.06, +0.06, 0), CCW
        [mag, -np.pi/4,    0, 0, 0, 1],    # NED: (-0.06, +0.06, 0), CW
        [mag, -3*np.pi/4,  0, 0, 0, 0],    # NED: (-0.06, -0.06, 0), CCW
        [mag,  3*np.pi/4,  0, 0, 0, 1],    # NED: (+0.06, -0.06, 0), CW
    ])

    # Reference: orientation_to_unit_vector(0,0,0) = [0,0,1] in NED (motor axis).
    # This matches the hover check convention (hovering_info.get_sim).
    # Note: create_2inch_quad() uses [0,0,-1] which is the negated convention —
    # the genome conversion intentionally uses the non-negated convention.
    expected_propellers = [
        {"loc": [0.06, 0.06, 0], "dir": [0, 0, 1, "ccw"], "propsize": 2},
        {"loc": [-0.06, 0.06, 0], "dir": [0, 0, 1, "cw"], "propsize": 2},
        {"loc": [-0.06, -0.06, 0], "dir": [0, 0, 1, "ccw"], "propsize": 2},
        {"loc": [0.06, -0.06, 0], "dir": [0, 0, 1, "cw"], "propsize": 2},
    ]

    propellers = _genome_to_propellers(genome)

    for actual, expected in zip(propellers, expected_propellers):
        np.testing.assert_allclose(actual["loc"], expected["loc"], atol=1e-10)
        np.testing.assert_allclose(actual["dir"][:3], expected["dir"][:3], atol=1e-10)
        assert actual["dir"][3] == expected["dir"][3]
        assert actual["propsize"] == expected["propsize"]


def test_propsize_always_2():
    """Propsize should always be 2, regardless of arm length."""
    for r in [0.01, 0.05, 0.1, 0.5, 1.0]:
        genome = np.array([[r, 0, 0, 0, 0, 0]])
        propellers = _genome_to_propellers(genome)
        assert propellers[0]["propsize"] == 2


def test_direction_threshold():
    """direction < 0.5 -> ccw, direction >= 0.5 -> cw."""
    genome_ccw = np.array([[0.06, 0, 0, 0, 0, 0.0]])
    genome_cw = np.array([[0.06, 0, 0, 0, 0, 1.0]])
    genome_boundary = np.array([[0.06, 0, 0, 0, 0, 0.5]])

    assert _genome_to_propellers(genome_ccw)[0]["dir"][3] == "ccw"
    assert _genome_to_propellers(genome_cw)[0]["dir"][3] == "cw"
    assert _genome_to_propellers(genome_boundary)[0]["dir"][3] == "cw"


def test_zero_motor_angles_give_default_direction():
    """Zero motor pitch/yaw should give motor_dir = [0, 0, 1] (NED convention)."""
    genome = np.array([[0.06, 0, 0, 0, 0, 0]])
    propellers = _genome_to_propellers(genome)
    np.testing.assert_allclose(propellers[0]["dir"][:3], [0, 0, 1], atol=1e-10)
