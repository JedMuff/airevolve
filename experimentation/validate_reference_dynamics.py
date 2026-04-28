"""Numerical parity check for ReferenceDroneSim against
optimal_quad_control_RL's f_func.

Compares the *state-derivative* output of both functions across a sweep of
random (state, action) inputs. We check the per-call agreement (not a long
integration) because the equations of motion contain Euler singularities
near theta = ±π/2 where tan(theta) blows up; once a trajectory passes
through such a region, sub-ULP floating-point differences amplify into
visibly different states. That's expected numerical chaos, not a bug.

Usage: /home/jed/miniconda3/envs/isaaclab/bin/python validate_reference_dynamics.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

# Add optimal_quad_control_RL to sys.path so we can import its f_func
OPTIMAL_DIR = "/home/jed/workspaces/optimal_quad_control_RL"
if OPTIMAL_DIR not in sys.path:
    sys.path.insert(0, OPTIMAL_DIR)

# Importing quad_race_env initializes torch (default device) — keep that
# noise out of the assertions but otherwise let it be.
import quad_race_env as ref  # noqa: E402

from reference_drone_sim import PARAMS_5INCH, ReferenceDroneSim  # noqa: E402


def main():
    rng = np.random.default_rng(0)
    sim = ReferenceDroneSim(dt=0.01)

    # Reference's f_func wants a 23-element params vector ordered the same as
    # the symbols defined in quad_race_env.py:28. Rebuild that order from
    # the PARAMS_5INCH dict.
    param_order = [p.name for p in ref.params]
    p_vec = np.array([PARAMS_5INCH[n] for n in param_order], dtype=np.float64)

    n_envs = 256
    n_trials = 20  # fresh random (state, action) per trial

    max_err_overall = 0.0
    max_relerr_overall = 0.0
    for trial in range(n_trials):
        state = np.zeros((n_envs, 16), dtype=np.float64)
        state[:, 0:3] = rng.uniform(-5, 5, size=(n_envs, 3))
        state[:, 3:6] = rng.uniform(-12, 12, size=(n_envs, 3))
        # Roll/pitch bounded away from ±π/2 to avoid Euler singularities
        # (which are real, but inflating tan(theta) is not what we're testing).
        state[:, 6:8] = rng.uniform(-np.pi / 3, np.pi / 3, size=(n_envs, 2))
        state[:, 8] = rng.uniform(-np.pi, np.pi, size=n_envs)
        state[:, 9:12] = rng.uniform(-5, 5, size=(n_envs, 3))
        state[:, 12:16] = rng.uniform(-1, 1, size=(n_envs, 4))
        action = rng.uniform(-1, 1, size=(n_envs, 4))

        sd_a = np.asarray(sim.dynamics_func(state.T, action.T)).T
        sd_b = np.asarray(
            ref.f_func(state.T, action.T, np.tile(p_vec[:, None], (1, n_envs)))
        ).T

        abs_err = np.max(np.abs(sd_a - sd_b))
        # Component-wise relative error, with an absolute floor so we don't
        # divide by ~0 for components that happen to be near zero.
        denom = np.maximum(np.abs(sd_b), 1.0)
        rel_err = np.max(np.abs(sd_a - sd_b) / denom)
        max_err_overall = max(max_err_overall, abs_err)
        max_relerr_overall = max(max_relerr_overall, rel_err)

    print(
        f"over {n_trials * n_envs:_} (state, action) samples: "
        f"max abs err={max_err_overall:.3e}, max rel err={max_relerr_overall:.3e}"
    )
    assert max_err_overall < 1e-6, f"V0 FAIL: abs dynamics err {max_err_overall:.3e}"
    assert max_relerr_overall < 1e-9, f"V0 FAIL: rel dynamics err {max_relerr_overall:.3e}"
    print("V0 OK: ReferenceDroneSim matches optimal_quad_control_RL.f_func")


if __name__ == "__main__":
    main()
