"""Per-step parity check: airevolve runtime DroneSimulator vs the
experimentation ReferenceDroneSim for the canonical 4-motor 2-inch quad.

This is the V0 gate for Phase 2 of RUNTIME_DYNAMICS_MIGRATION.md. It
verifies that the runtime's reference-form dynamics produces the same
state derivatives as the experimentation reference (which is itself
verified at machine precision against optimal_quad_control_RL/quad_race_env.py
by experimentation/validate_reference_dynamics.py).

Why a permutation: the reference's hardcoded sign pattern in
`_build_dynamics_func` (reference_drone_sim.py:280-291) labels motor 1
at front-left (x<0, y>0, cw) and motor 2 at front-right. Airevolve's
`create_standard_propeller_config("quad")` labels motor 1 at front-right
(angle π/4) and motor 2 at front-left (3π/4). The two systems are
related by a swap of motor[0] and motor[1] (motors 3, 4 already match).

The runtime intentionally derives per-motor signs from actual positions
(see dynamics_params.derive_reference_params) so it generalizes to
asymmetric morphologies (Phase 4). For the symmetric 4-motor canonical
quad, this differs from the reference's hardcoded pattern by exactly the
above permutation.

Pass criterion: <1e-6 abs error and <1e-9 rel error after permutation
(matches experimentation/validate_reference_dynamics.py).

Usage: /home/jed/miniconda3/envs/isaaclab/bin/python unit_tests/test_dynamics_parity.py
"""
from __future__ import annotations

import os
import sys

from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from airevolve.simulator.simulation.drone_simulator import DroneSimulator  # noqa: E402
from airevolve.simulator.simulation.propeller_data import (  # noqa: E402
    create_standard_propeller_config,
)
from experimentation.reference_drone_sim import (  # noqa: E402
    ReferenceDroneSim,
    derive_params_2inch_from_airevolve,
)


# Reference's motor numbering ↔ airevolve's:
#   reference[0] = airevolve[1] (front-left, cw)
#   reference[1] = airevolve[0] (front-right, ccw)
#   reference[2] = airevolve[2] (back-left, ccw)
#   reference[3] = airevolve[3] (back-right, cw)
# So airevolve_index = REF_TO_AEV[ref_index]:
REF_TO_AEV = np.array([1, 0, 2, 3])
# Inverse for unpermuting: AEV_TO_REF[aev_index] = ref_index
AEV_TO_REF = np.argsort(REF_TO_AEV)


def _permute_state_motors(state: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """Reorder the motor-speed slice [12:16] of a (B, 16) state."""
    out = state.copy()
    out[:, 12:16] = state[:, 12 + perm]
    return out


def main():
    propellers = create_standard_propeller_config(
        "quad", arm_length=0.11, prop_size=2
    )
    sim_runtime = DroneSimulator(propellers=propellers, dt=0.01)

    ref_params, ref_mass, ref_inertia = derive_params_2inch_from_airevolve()
    sim_ref = ReferenceDroneSim(
        dt=0.01, params=ref_params, mass=ref_mass, inertia=ref_inertia
    )

    # Sanity: airevolve's actual mass/inertia match what derive_params used.
    assert np.isclose(sim_runtime.mass, ref_mass), (
        f"mass mismatch: {sim_runtime.mass} vs {ref_mass}"
    )
    assert np.allclose(sim_runtime.inertia, ref_inertia), "inertia mismatch"

    rng = np.random.default_rng(0)
    n_envs = 256
    n_trials = 20

    max_abs_overall = 0.0
    max_rel_overall = 0.0
    for trial in range(n_trials):
        state = np.zeros((n_envs, 16), dtype=np.float64)
        state[:, 0:3] = rng.uniform(-5, 5, size=(n_envs, 3))
        state[:, 3:6] = rng.uniform(-12, 12, size=(n_envs, 3))
        state[:, 6:8] = rng.uniform(-np.pi / 3, np.pi / 3, size=(n_envs, 2))
        state[:, 8] = rng.uniform(-np.pi, np.pi, size=n_envs)
        state[:, 9:12] = rng.uniform(-5, 5, size=(n_envs, 3))
        state[:, 12:16] = rng.uniform(-1, 1, size=(n_envs, 4))
        action = rng.uniform(-1, 1, size=(n_envs, 4))

        sd_runtime = np.asarray(sim_runtime.dynamics_func(state.T, action.T)).T

        # Run reference with permuted motor assignment, then unpermute the
        # motor-speed derivatives.
        state_for_ref = _permute_state_motors(state, REF_TO_AEV)
        action_for_ref = action[:, REF_TO_AEV]
        sd_ref_perm = np.asarray(
            sim_ref.dynamics_func(state_for_ref.T, action_for_ref.T)
        ).T
        sd_ref = sd_ref_perm.copy()
        sd_ref[:, 12:16] = sd_ref_perm[:, 12 + AEV_TO_REF]

        abs_err = float(np.max(np.abs(sd_runtime - sd_ref)))
        denom = np.maximum(np.abs(sd_ref), 1.0)
        rel_err = float(np.max(np.abs(sd_runtime - sd_ref) / denom))
        max_abs_overall = max(max_abs_overall, abs_err)
        max_rel_overall = max(max_rel_overall, rel_err)

    print(
        f"over {n_trials * n_envs:_} (state, action) samples: "
        f"max abs err={max_abs_overall:.3e}, max rel err={max_rel_overall:.3e}"
    )
    assert max_abs_overall < 1e-6, (
        f"V0 RUNTIME FAIL: abs dynamics err {max_abs_overall:.3e}"
    )
    assert max_rel_overall < 1e-9, (
        f"V0 RUNTIME FAIL: rel dynamics err {max_rel_overall:.3e}"
    )
    print(
        "V0 RUNTIME OK: airevolve runtime DroneSimulator matches "
        "ReferenceDroneSim (modulo motor[0]↔motor[1] permutation)"
    )


if __name__ == "__main__":
    main()
