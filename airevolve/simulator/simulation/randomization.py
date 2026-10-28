"""Domain randomization of the dynamics parameters.

Same semantics as optimal_quad_control_RL/randomization.py: every
coefficient is drawn independently from U(v·(1-p), v·(1+p)) on each reset,
with the motor-curve exponent k capped at 1. Coefficients that are zero
stay zero.

For a sysid-mode simulator the draws are made on the sysid dict, in its key
order, exactly like the reference (so seeded runs match it draw for draw);
in physics mode they are made on the derived coefficient vector, leaving
the geometry (thrust axes) and the inflow constant fixed.
"""
from __future__ import annotations

import numpy as np

from .dynamics_params import sysid_param_matrix


def make_randomization(sim, fraction=0.0):
    """Return `draw(n) -> (n, P)` parameter vectors for `sim`.

    Args:
        sim: DroneSimulator providing the nominal parameters.
        fraction: relative half-width of the uniform distribution
            (0.3 = ±30%). 0 gives the nominal parameters without drawing
            random numbers.
    """
    fraction = float(fraction)
    if fraction < 0:
        raise ValueError("randomization fraction must be >= 0")

    if sim.sysid_params is not None:
        sysid = sim.sysid_params
        axes = sim.params["axis"]

        def draw(n):
            if fraction == 0:
                values = {key: np.repeat(value, n) for key, value in sysid.items()}
            else:
                values = {
                    key: np.random.uniform(value * (1 - fraction), value * (1 + fraction), n)
                    if key != "k" else
                    np.random.uniform(value * (1 - fraction), min(value * (1 + fraction), 1), n)
                    for key, value in sysid.items()
                }
            return sysid_param_matrix(values, axes)
        return draw

    nominal = sim.param_vector
    mask = sim.param_layout.randomize_mask
    k_index = sim.param_layout.k_index

    def draw(n):
        out = np.repeat(nominal[None], n, axis=0)
        if fraction == 0:
            return out
        scale = np.random.uniform(1 - fraction, 1 + fraction, size=(n, mask.sum()))
        out[:, mask] *= scale
        out[:, k_index] = np.minimum(out[:, k_index], 1.0)
        return out
    return draw
