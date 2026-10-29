"""Run configuration shared by RL training, viewing and C code export.

A run config is a JSON-serializable dict saved next to the checkpoints
(`run_config.json`). It pins down everything needed to rebuild the exact
drone and env a policy was trained on:

    {
      "drone": {
        "name": "real_hex3",
        "propellers": [...],            # motor i = flight-controller motor i+1
        "sysid_params": {...} | null,   # sysid mode when given
        "prop_library": {...},          # custom prop-library entries used
        "airframes": {...},             # airframes those entries name
        "mass": null, "inertia": null,  # measured overrides (physics mode)
        "w_norm_max": 4000.0
      },
      "task": "gates" | "hover",
      "yaw_rate_penalty": 0.05,         # hover only (0 = reference reward)
      "track_radius": 1.5,
      "gates_ahead": 1,
      "motor_limit": 1.0,
      "randomization": 0.3,
      "dt": 0.01,
      "num_state_history": 0, "num_action_history": 0, "history_step_size": 1
    }
"""
from __future__ import annotations

import json
import os

import numpy as np

from airevolve.simulator.simulation.drone_simulator import DroneSimulator
from airevolve.simulator.simulation.dynamics_params import W_MAX_N
from airevolve.simulator.simulation.propeller_data import (
    create_standard_propeller_config, genome_to_propellers, get_propeller_specs, register_propeller,
)
from airevolve.simulator.simulation.sysid_presets import (
    SYSID_DRONES, real_hex3_propellers,
)
from airevolve.simulator.simulation.airframe import Airframe, get_airframe, register_airframe

CONFIG_NAME = "run_config.json"
STANDARD_DRONES = ("quad", "hex", "tri", "octo")
# The real drones' layout in physics mode (the evolution simulator), with
# their motors calibrated from the sysid: --drone name -> propsize.
PHYSICS_DRONES = {"physics_hex3": "hex3", "physics_vuhex": "vuhex"}


def drone_config(name="real_hex3", genome=None, propellers=None, propsize=None,
                 arm_length=0.11, w_norm_max=None, mass=None, inertia=None):
    """Drone section of a run config.

    Args:
        name: "real_hex3" (sysid mode, the flight-verified hex),
            "real_vuhex" (sysid mode, same layout), "physics_hex3" /
            "physics_vuhex" (that layout in physics mode with the motors
            calibrated from the sysid), a standard layout ("quad", "hex", "tri", "octo"), or a label for `genome` /
            `propellers`.
        genome: spherical-angular genome array (rows with NaN are skipped).
        propellers: explicit propeller list.
        propsize: prop-library entry for physics mode (2-8, "matched", or
            "hex3" for the real hex's motors calibrated from its sysid).
        arm_length: arm length for the standard layouts.
        w_norm_max: motor-speed normalization; defaults to the preset's
            4000 (the flown hex3 policies used 4700).
        mass, inertia: measured overrides for physics mode.
    """
    sysid = None
    if genome is not None:
        propellers = genome_to_propellers(np.asarray(genome), propsize=propsize or 2)
    elif propellers is not None:
        propellers = [dict(p) for p in propellers]
        if propsize is not None:
            for p in propellers:
                p["propsize"] = propsize
    elif name in SYSID_DRONES:
        propellers = real_hex3_propellers()
        params, preset_w_norm_max = SYSID_DRONES[name]
        sysid = dict(params)
        w_norm_max = w_norm_max or preset_w_norm_max
    elif name in PHYSICS_DRONES:
        propellers = real_hex3_propellers(propsize=propsize or PHYSICS_DRONES[name])
    elif name in STANDARD_DRONES:
        propellers = create_standard_propeller_config(name, arm_length=arm_length, prop_size=propsize or 2)
    else:
        raise ValueError(f"unknown drone {name!r}: use {tuple(SYSID_DRONES)}, {tuple(PHYSICS_DRONES)}, {STANDARD_DRONES}, a genome or propellers")

    prop_library, airframes = {}, {}
    if sysid is None:
        for size in {p["propsize"] for p in propellers}:
            specs = get_propeller_specs(size)
            if isinstance(size, str) and size != "matched":
                prop_library[size] = specs
            if specs.get("airframe"):
                airframes[specs["airframe"]] = get_airframe(specs["airframe"]).to_dict()

    clean = [{"loc": [float(v) for v in p["loc"]],
              "dir": [float(v) for v in p["dir"][:3]] + [p["dir"][3]],
              "propsize": p["propsize"]} for p in propellers]
    return {
        "name": name,
        "propellers": clean,
        "sysid_params": sysid,
        "prop_library": prop_library,
        "airframes": airframes,
        "mass": mass,
        "inertia": None if inertia is None else np.asarray(inertia).tolist(),
        "w_norm_max": float(w_norm_max or W_MAX_N),
    }


def build_drone(drone: dict, dt=0.01) -> DroneSimulator:
    """DroneSimulator for the drone section of a run config."""
    for airframe in drone.get("airframes", {}).values():
        register_airframe(Airframe.from_dict(airframe))
    for size, specs in drone.get("prop_library", {}).items():
        register_propeller(size, specs)
    return DroneSimulator(
        propellers=[dict(p, dir=list(p["dir"])) for p in drone["propellers"]],
        dt=dt,
        sysid_params=drone.get("sysid_params"),
        mass=drone.get("mass"),
        inertia=drone.get("inertia"),
        w_norm_max=drone.get("w_norm_max", W_MAX_N),
    )


def make_env(config: dict, num_envs: int, **overrides):
    """Env for a run config; `overrides` are passed to the env constructor."""
    from airevolve.evolution_tools.evaluators.drone_gate_env import (
        DroneGateEnv, DroneHoverEnv, figure8_track,
    )

    kwargs = dict(
        drone_sim=build_drone(config["drone"], dt=config["dt"]),
        motor_limit=config["motor_limit"],
        randomization=config["randomization"],
        num_state_history=config["num_state_history"],
        num_action_history=config["num_action_history"],
        history_step_size=config["history_step_size"],
        dt=config["dt"],
    )
    if config["task"] == "gates":
        gates, yaws, start = figure8_track(config["track_radius"])
        kwargs.update(gates_pos=gates, gate_yaw=yaws, start_pos=start, gates_ahead=config["gates_ahead"])
        kwargs.update(overrides)
        return DroneGateEnv(num_envs, **kwargs)
    if config["task"] == "hover":
        kwargs["yaw_rate_penalty"] = config.get("yaw_rate_penalty", 0.0)
        kwargs.update(overrides)
        return DroneHoverEnv(num_envs, **kwargs)
    raise ValueError(f"unknown task {config['task']!r}")


def run_config(drone, task="gates", track_radius=1.5, gates_ahead=1, motor_limit=1.0,
               randomization=0.0, dt=0.01, num_state_history=0, num_action_history=0,
               history_step_size=1, yaw_rate_penalty=0.0):
    return {
        "drone": drone,
        "task": task,
        "track_radius": float(track_radius),
        "gates_ahead": int(gates_ahead) if task == "gates" else 0,
        "yaw_rate_penalty": float(yaw_rate_penalty) if task == "hover" else 0.0,
        "motor_limit": float(motor_limit),
        "randomization": float(randomization),
        "dt": float(dt),
        "num_state_history": int(num_state_history),
        "num_action_history": int(num_action_history),
        "history_step_size": int(history_step_size),
    }


def save_config(config: dict, folder: str):
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, CONFIG_NAME), "w") as f:
        json.dump(config, f, indent=2)


def load_config(path: str) -> dict:
    """Load a run config from its file, or from the checkpoint folder."""
    if os.path.isdir(path):
        path = os.path.join(path, CONFIG_NAME)
    with open(path) as f:
        return json.load(f)


def latest_checkpoint(folder: str) -> str:
    """Path of the checkpoint with the most timesteps in `folder`."""
    checkpoints = [f for f in os.listdir(folder) if f.endswith(".zip") and f[:-4].isdigit()]
    if not checkpoints:
        raise FileNotFoundError(f"no <timesteps>.zip checkpoints in {folder}")
    return os.path.join(folder, max(checkpoints, key=lambda f: int(f[:-4])))
