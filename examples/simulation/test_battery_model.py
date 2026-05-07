#!/usr/bin/env python3
"""
Standalone test and plotting script for LiPoBatteryModel.

Simulates a 3-phase hexacopter mission and plots voltage, power,
energy, and State of Charge over time with phase-transition markers.

Usage (from project root):
    python examples/simulation/test_battery_model.py
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

# Allow running from any directory by ensuring the project root is importable
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import importlib.util as _ilu
import pathlib as _pl

_bm_path = _pl.Path(_PROJECT_ROOT) / "airevolve" / "simulator" / "simulation" / "battery_model.py"
_spec = _ilu.spec_from_file_location("battery_model", _bm_path)
_mod  = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
LiPoBatteryModel = _mod.LiPoBatteryModel

# ------------------------------------------------------------------ #
# Mission parameters                                                   #
# ------------------------------------------------------------------ #
DT = 0.005  # seconds — matches DroneSimulator default dt

# Phase definitions
# Current draws are representative of a 250-class hexacopter:
#   Hover:          ~8 A  (motors spinning at moderate throttle to maintain altitude)
#   Forward flight: ~18 A (higher throttle, forward tilt, aerodynamic drag)
PHASES = [
    {"name": "Phase 1 — Hover",          "duration": 30.0,  "current": 8.0},
    {"name": "Phase 2 — Forward Flight",  "duration": 60.0,  "current": 18.0},
    {"name": "Phase 3 — Return Hover",    "duration": 30.0,  "current": 8.0},
]

# ------------------------------------------------------------------ #
# Run simulation                                                        #
# ------------------------------------------------------------------ #

battery = LiPoBatteryModel()

print(f"Battery : {battery.BATTERY_NAME}")
print(f"Capacity: {battery.CAPACITY_MAH} mAh  |  "
      f"Max current: {battery.MAX_CURRENT} A  |  "
      f"Mass: {battery.MASS_G} g  |  "
      f"Dims: {battery.DIMENSIONS_MM} mm")
print()

phase_start_times = []
for phase in PHASES:
    phase_start_times.append(battery._total_time_s)
    steps = int(phase["duration"] / DT)
    current = phase["current"]
    print(f"  {phase['name']}: {phase['duration']:.0f}s @ {current}A — {steps} steps")

    for _ in range(steps):
        battery.step(DT, current)
        if battery.is_depleted:
            print(f"    *** Battery depleted at t={battery._total_time_s:.2f}s ***")
            break

    if battery.is_depleted:
        break

print()
print("Simulation complete.")
print(f"  Final SoC     : {battery.soc * 100:.2f}%")
print(f"  Final voltage : {battery.voltage:.3f} V")
total_j = battery.get_total_energy_consumed()
print(f"  Total energy  : {total_j:.1f} J  ({total_j / 3600:.4f} Wh)")
print(f"  Depleted      : {battery.is_depleted}")
print(f"  RL obs        : {battery.get_rl_observations()}")
print(f"  Mass & dims   : {battery.get_mass_and_dimensions()}")
print()

# ------------------------------------------------------------------ #
# Retrieve history                                                     #
# ------------------------------------------------------------------ #
hist = battery.get_history()

# Normalise to numpy arrays regardless of pandas availability
if hasattr(hist, "to_numpy"):
    # pandas DataFrame
    time_arr    = hist["time"].to_numpy()
    soc_arr     = hist["soc"].to_numpy()
    voltage_arr = hist["voltage"].to_numpy()
    power_arr   = hist["power"].to_numpy()
    energy_arr  = hist["total_energy_j"].to_numpy()
else:
    time_arr    = np.array(hist["time"])
    soc_arr     = np.array(hist["soc"])
    voltage_arr = np.array(hist["voltage"])
    power_arr   = np.array(hist["power"])
    energy_arr  = np.array(hist["total_energy_j"])

# ------------------------------------------------------------------ #
# Plotting                                                             #
# ------------------------------------------------------------------ #

# Phase transition x-positions (skip the first which is t=0)
phase_lines = phase_start_times[1:]  # [30.0, 90.0]
phase_line_labels = ["Phase 2\nStart", "Phase 3\nStart"]

_LINE_KW = dict(color="gray", linestyle="--", linewidth=1.2, alpha=0.8)


def _add_phase_lines(ax):
    """Mark phase transitions with dashed vertical lines and labels."""
    for t, label in zip(phase_lines, phase_line_labels):
        ax.axvline(x=t, **_LINE_KW)
        # get_xaxis_transform(): x in data coords, y in axes-fraction — immune to y-range
        ax.text(t + 0.4, 0.97, label, fontsize=7, va="top", ha="left",
                color="gray", transform=ax.get_xaxis_transform(), clip_on=False)


fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    f"LiPo Battery Discharge — {battery.BATTERY_NAME}\n"
    "3-Phase Hexacopter Mission  (Hover 30 s / Forward 60 s / Return 30 s)",
    fontsize=13, fontweight="bold",
)

# ── Subplot 1: Voltage vs Time ────────────────────────────────────── #
ax = axes[0, 0]
ax.plot(time_arr, voltage_arr, color="royalblue", linewidth=1.5, label="Terminal Voltage")
ax.axhline(
    y=battery.VOLTAGE_DEPLETION_THRESHOLD, color="red",
    linestyle=":", linewidth=1.2,
    label=f"Depletion threshold ({battery.VOLTAGE_DEPLETION_THRESHOLD} V)",
)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Terminal Voltage (V)")
ax.set_title("Voltage vs Time")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
_add_phase_lines(ax)

# ── Subplot 2: Instantaneous Power vs Time ───────────────────────── #
ax = axes[0, 1]
ax.plot(time_arr, power_arr, color="darkorange", linewidth=1.5)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Instantaneous Power (W)")
ax.set_title("Instantaneous Power vs Time")
ax.grid(True, alpha=0.3)
_add_phase_lines(ax)

# ── Subplot 3: Cumulative Energy Consumed vs Time ────────────────── #
ax = axes[1, 0]
ax.plot(time_arr, energy_arr / 3600.0, color="forestgreen", linewidth=1.5)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Energy Consumed (Wh)")
ax.set_title("Total Energy Consumed vs Time")
ax.grid(True, alpha=0.3)
_add_phase_lines(ax)

# ── Subplot 4: State of Charge vs Time ──────────────────────────── #
ax = axes[1, 1]
ax.plot(time_arr, soc_arr * 100.0, color="crimson", linewidth=1.5, label="SoC")
ax.axhline(
    y=battery.SOC_DEPLETION_THRESHOLD * 100, color="red",
    linestyle=":", linewidth=1.2,
    label=f"Depletion threshold ({battery.SOC_DEPLETION_THRESHOLD*100:.0f}%)",
)
ax.set_xlabel("Time (s)")
ax.set_ylabel("State of Charge (%)")
ax.set_title("State of Charge vs Time")
ax.set_ylim([0, 105])
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
_add_phase_lines(ax)

plt.tight_layout()
plt.show()
