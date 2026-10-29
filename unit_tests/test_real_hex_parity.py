"""Parity with the flight-verified hex env of optimal_quad_control_RL.

Policies trained in optimal_quad_control_RL/real_quad_env.py with
params_hex3 (w_max_n = 4700) fly on the real hexacopter. These tests check
that airevolve reproduces that env, so policies trained here deploy through
the same C code:

1. Dynamics: sysid-mode DroneSimulator vs real_quad_env.f_func on random
   states/actions (frozen fixture, plus the live reference when the repo is
   checked out next to this one).
2. Env: DroneGateEnv / DroneHoverEnv vs Quadcopter3DGates / Quadcopter3DHover
   step for step with the same seed and actions, with and without ±30%
   domain randomization (live reference only).
3. Calibration: the motor constants backed out of the sysid re-derive the
   sysid coefficients in physics mode.

Usage: python unit_tests/test_real_hex_parity.py [--regenerate]
"""
from __future__ import annotations

import os
import sys

import numpy as np

from airevolve.evolution_tools.evaluators.drone_gate_env import DroneGateEnv, DroneHoverEnv
from airevolve.simulator.simulation.drone_simulator import DroneSimulator
from airevolve.simulator.simulation.dynamics_params import sysid_param_matrix
from airevolve.simulator.simulation.propeller_data import register_propeller
from airevolve.simulator.simulation.sysid_presets import (
    HEX3_MASS, HEX3_W_NORM_MAX, PARAMS_HEX3, PARAMS_VUHEX, calibrate_prop_constants, real_hex3_propellers,
)

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE = os.path.join(HERE, "data", "real_hex3_dynamics.npz")
REFERENCE_REPO = os.environ.get(
    "OPTIMAL_QUAD_CONTROL_RL",
    os.path.join(os.path.dirname(os.path.dirname(HERE)), "optimal_quad_control_RL"),
)


def load_reference():
    """real_quad_env and randomization modules, or None if not checked out."""
    if not os.path.isfile(os.path.join(REFERENCE_REPO, "real_quad_env.py")):
        return None
    if REFERENCE_REPO not in sys.path:
        sys.path.insert(0, REFERENCE_REPO)
    import randomization as ref_rand
    import real_quad_env as ref
    return ref, ref_rand


# Sysid sets checked against the live reference, with its current w_max_n.
REFERENCE_SETS = {"hex3": PARAMS_HEX3, "vuhex": PARAMS_VUHEX}


def hex3_sim(sysid=PARAMS_HEX3, w_norm_max=HEX3_W_NORM_MAX):
    return DroneSimulator(real_hex3_propellers(), sysid_params=sysid, dt=0.01,
                          w_norm_max=w_norm_max)


def random_inputs(n, seed=0):
    rng = np.random.default_rng(seed)
    states = np.concatenate([
        rng.uniform(-3, 3, (n, 6)),     # position, velocity
        rng.uniform(-1, 1, (n, 3)),     # euler angles
        rng.uniform(-5, 5, (n, 3)),     # body rates
        rng.uniform(-1, 1, (n, 6)),     # normalized motor speeds
    ], axis=1)
    return states, rng.uniform(-1, 1, (n, 6))


def airevolve_derivatives(states, actions, sysid=PARAMS_HEX3, w_norm_max=HEX3_W_NORM_MAX):
    sim = hex3_sim(sysid, w_norm_max)
    params = np.tile(sim.param_vector[:, None], (1, len(states)))
    return np.asarray(sim.dynamics_func_p(states.T, actions.T, params)).T


def reference_derivatives(ref, states, actions, sysid=PARAMS_HEX3):
    params = np.array([sysid[p.name] for p in ref.params])
    return np.asarray(ref.f_func(states.T, actions.T, np.tile(params[:, None], (1, len(states))))).T


def assert_close(a, b):
    np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-9)


def test_dynamics_match_fixture():
    data = np.load(FIXTURE)
    assert_close(airevolve_derivatives(data["states"], data["actions"]), data["derivatives"])


def test_dynamics_match_reference():
    loaded = load_reference()
    if loaded is None:
        print("  (skipped: optimal_quad_control_RL not found)")
        return
    ref = loaded[0]
    states, actions = random_inputs(2000, seed=1)
    for sysid in REFERENCE_SETS.values():
        assert_close(airevolve_derivatives(states, actions, sysid, ref.w_max_n),
                     reference_derivatives(ref, states, actions, sysid))


def _run_pair(make_mine, make_ref, ref_param_names, num_envs=20, steps=1500):
    """Step both envs with their own global-RNG streams and identical actions."""
    np.random.seed(3); mine = make_mine(); rng_mine = np.random.get_state()
    np.random.seed(3); theirs = make_ref(); rng_ref = np.random.get_state()

    def call(state, fn, *args):
        np.random.set_state(state)
        out = fn(*args)
        return out, np.random.get_state()

    (obs_a, rng_mine) = call(rng_mine, mine.reset)
    (obs_b, rng_ref) = call(rng_ref, theirs.reset)
    np.testing.assert_allclose(obs_a, obs_b, atol=1e-6)
    actions_rng = np.random.default_rng(1)
    ended = 0
    for _ in range(steps):
        act = np.clip(actions_rng.normal(-0.3, 0.4, (num_envs, 6)), -1, 1)
        mine.step_async(act)
        theirs.step_async(act)
        (obs_a, rew_a, done_a, _), rng_mine = call(rng_mine, mine.step_wait)
        (obs_b, rew_b, done_b, _), rng_ref = call(rng_ref, theirs.step_wait)
        np.testing.assert_array_equal(done_a, done_b)
        np.testing.assert_allclose(obs_a, obs_b, atol=1e-5)
        np.testing.assert_allclose(rew_a, rew_b, atol=1e-6)
        ref_params = sysid_param_matrix(
            {name: theirs.params[:, i] for i, name in enumerate(ref_param_names)},
            mine.drone_sim.params["axis"])
        np.testing.assert_array_equal(mine.params, ref_params)
        # Re-sync so float rounding (cse reorders the arithmetic) is not
        # amplified by the chaotic random-action trajectories.
        mine.world_states = theirs.world_states.copy()
        ended += done_a.sum()
    assert ended > 50, "too few episode ends to exercise resets"


def test_env_matches_reference():
    loaded = load_reference()
    if loaded is None:
        print("  (skipped: optimal_quad_control_RL not found)")
        return
    ref, ref_rand = loaded
    names = [p.name for p in ref.params]
    n = 20
    cases = [(sysid, fraction, getattr(ref_rand, ref_name.format(name)))
             for name, sysid in REFERENCE_SETS.items()
             for fraction, ref_name in [(0.0, "randomization_fixed_params_{}"),
                                        (0.3, "randomization_{}_30_percent")]]
    for sysid, fraction, ref_randomization in cases:
        sim = hex3_sim(sysid, ref.w_max_n)
        common = dict(drone_sim=sim, randomization=fraction, seed=3, device="cpu")
        _run_pair(lambda: DroneGateEnv(n, **common),
                  lambda: ref.Quadcopter3DGates(n, ref_randomization, seed=3), names)
        _run_pair(lambda: DroneGateEnv(n, initialize_at_random_gates=False, **common),
                  lambda: ref.Quadcopter3DGates(n, ref_randomization, seed=3, initialize_at_random_gates=False),
                  names)
        _run_pair(lambda: DroneHoverEnv(n, **common),
                  lambda: ref.Quadcopter3DHover(n, ref_randomization, seed=3), names)


def test_calibrated_motor_rederives_sysid():
    mass, arm = 0.3, 0.1
    props = real_hex3_propellers(arm)
    specs, inertia = calibrate_prop_constants(PARAMS_HEX3, props, mass)
    register_propeller("hex3_test", specs)
    physics = DroneSimulator([dict(p, propsize="hex3_test") for p in props], dt=0.01,
                             mass=mass, inertia=np.diag(inertia)).params
    sysid = hex3_sim().params

    np.testing.assert_allclose(physics["f_coef"], sysid["f_coef"], rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(physics["k_drag"], sysid["k_drag"], rtol=1e-12)
    for key in ("tau", "k", "w_min", "w_max", "inflow_coef"):
        np.testing.assert_allclose(physics[key], sysid[key], rtol=1e-12)
    # Roll/pitch: the sysid has small left/right and M5/M6 asymmetries the
    # symmetric layout can't express.
    m_phys, m_sys = np.array(physics["m_coef"]), np.array(sysid["m_coef"])
    np.testing.assert_allclose(m_phys[:4, :2], m_sys[:4, :2], rtol=0.06)
    np.testing.assert_allclose(m_phys[4:, 0], m_sys[4:, 0], rtol=0.06)
    # Yaw: identical up to the mean over motors, with the same signs.
    np.testing.assert_allclose(m_phys[:, 2], m_sys[:, 2], rtol=1e-9)
    np.testing.assert_allclose(physics["m_dw_coef"], sysid["m_dw_coef"], rtol=1e-9, atol=1e-15)


def test_real_hex_airframe_reproduces_sysid():
    """The hex3 layout built in the custom setup (hex3 motors + real_hex
    airframe) has the measured mass and the sysid's dynamics."""
    physics = DroneSimulator(real_hex3_propellers(propsize="hex3"), dt=0.01, w_norm_max=HEX3_W_NORM_MAX)
    assert physics.config.airframe.name == "real_hex"
    np.testing.assert_allclose(physics.mass, HEX3_MASS, rtol=1e-12)
    p, q = physics.params, hex3_sim().params
    for key in ("f_coef", "k_drag", "m_dw_coef", "tau", "k", "w_min", "w_max"):
        np.testing.assert_allclose(p[key], q[key], rtol=1e-9, atol=1e-15)
    m_phys, m_sys = np.array(p["m_coef"]), np.array(q["m_coef"])
    np.testing.assert_allclose(m_phys[:, 2], m_sys[:, 2], rtol=1e-9)
    np.testing.assert_allclose(m_phys[:4, :2], m_sys[:4, :2], rtol=0.05)
    np.testing.assert_allclose(m_phys[4:, 0], m_sys[4:, 0], rtol=0.05)


def regenerate_fixture():
    loaded = load_reference()
    if loaded is None:
        raise SystemExit("optimal_quad_control_RL not found; cannot regenerate the fixture")
    states, actions = random_inputs(200)
    os.makedirs(os.path.dirname(FIXTURE), exist_ok=True)
    np.savez(FIXTURE, states=states, actions=actions,
             derivatives=reference_derivatives(loaded[0], states, actions))
    print(f"wrote {FIXTURE}")


if __name__ == "__main__":
    if "--regenerate" in sys.argv:
        regenerate_fixture()
    test_dynamics_match_fixture()
    test_dynamics_match_reference()
    test_env_matches_reference()
    test_calibrated_motor_rederives_sysid()
    test_real_hex_airframe_reproduces_sysid()
    print("real hex parity OK")
