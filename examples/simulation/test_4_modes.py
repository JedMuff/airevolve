#!/usr/bin/env python3
"""
test_4_modes.py — ECM battery 4-mode live validation on a figure-8 rollout.

Instantiates a sub-250 g hexacopter (Flywoo GN405 FC, EMAX ECO 1404 3700KV × 6,
Gemfan 2609 props, Tattu R-Line 750mAh 14.8V 4S 95C) inside DroneGateEnv, runs a
randomly-initialised (or pre-trained) PPO policy through NUM_EPISODES figure-8
episodes, and pipes the kinematic state to LiPoBatteryModel.step() each tick.

Live visualiser  : 3D flight view + 5 live battery strips  (plt.ion)
Final figure     : 5 stacked subplots — Mode / Voltage / SoC / Capacity / Power

Usage:
    python examples/simulation/test_4_modes.py
    python examples/simulation/test_4_modes.py --episodes 20 --max-steps 1200
    python examples/simulation/test_4_modes.py --policy-path __data__/rl/policy.zip
    python examples/simulation/test_4_modes.py --no-viz --no-show --save-fig out.png
"""
from __future__ import annotations

# ═══════════════════════════════════════════════════════════════════════════════
# TASK 1 — UNIVERSAL SYS.PATH BOOTSTRAP
# ═══════════════════════════════════════════════════════════════════════════════
#
# THE PROBLEM
# -----------
# Running  `python examples/simulation/test_4_modes.py`  from any directory fails
# with ModuleNotFoundError because Python only adds the CWD (or the script's own
# directory, depending on invocation) to sys.path — not the repository root.
#
# THE ROOT-MARKER APPROACH  (used here)
# --------------------------------------
# Walk upward from this file's resolved path until we find a known root marker
# (setup.py).  Then insert that directory at sys.path[0].
#   + Works from any CWD on macOS, Linux, or Windows.
#   + Works through symlinks (Path.resolve() follows them).
#   + Depth-agnostic: moving the script deeper or shallower never breaks it.
#   + Falls back to a hardcoded "2 levels up" if the marker is absent.
#
# LONG-TERM RECOMMENDATION
# -------------------------
# Run once on every machine (laptop and compute server):
#
#     pip install -e .          # or:  pip install -e . --no-deps
#
# This registers `airevolve` in site-packages via a .pth pointer, so every
# Python process on that environment can `import airevolve` without any
# sys.path magic.  The -e (editable) flag means changes to the source are
# reflected immediately without re-installing.  After this one-time step,
# you can delete the bootstrap block below from every script.

import sys
from pathlib import Path


def _bootstrap_repo_root(marker: str = "setup.py") -> Path:
    """
    Walk from this file upward until 'marker' is found; insert root into sys.path.

    Returns the repo root as a Path so callers can reference it if needed.
    """
    here = Path(__file__).resolve()          # absolute, symlinks resolved
    for candidate in [here, *here.parents]:
        if (candidate / marker).exists():
            root = candidate
            break
    else:
        # Hardcoded fallback: examples/simulation/test_4_modes.py → root 3 levels up
        root = here.parent.parent.parent

    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


REPO_ROOT = _bootstrap_repo_root()


# ───────────────────────────────────────────────────────────────────────────────
import argparse
import math
import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

import torch
from stable_baselines3 import PPO

from airevolve.evolution_tools.evaluators.drone_gate_env import (
    DroneGateEnv,
    gate_pos as DEFAULT_GATE_POS,
    gate_yaw as DEFAULT_GATE_YAW,
)
from airevolve.simulator.simulation.propeller_data import create_standard_propeller_config
from airevolve.simulator.simulation.battery_model import LiPoBatteryModel


# ═══════════════════════════════════════════════════════════════════════════════
# HARDWARE / SIMULATION CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
# Physical hexacopter (sub-250 g):
#   FC      : Flywoo GN405
#   Motors  : EMAX ECO 1404 3700KV  ×6
#   Props   : Gemfan 2609  (2.6" diameter — nearest prop library size = 3")
#   Battery : Tattu R-Line 750mAh 14.8V 4S 95C

ARM_LENGTH   = 0.09    # metres — typical arm for a 1404-class hex
PROP_SIZE    = 3       # Gemfan 2609 ≈ 2.6" → nearest library size = 3"
DT           = 0.01    # seconds — DroneGateEnv default timestep
NUM_EPISODES = 20      # figure-8 episodes to simulate
MAX_STEPS    = 1200    # steps per episode → 12 s @ dt=0.01
DEVICE       = "cpu"

# Set to a .zip from run_rl_figure8.py to use a trained policy.
# None → randomly-initialised PPO.  A random policy is ideal for testing all
# 4 modes because unconstrained motor commands cause large velocity excursions.
POLICY_PATH: str | None = None  # e.g. "__data__/rl/figure8_prop3_final.zip"

# Battery drain strategy (documented choice):
#   False (default) → CONTINUOUS drain across all 20 episodes.
#                     Gives a single realistic discharge curve; the battery
#                     may reach the 15% SoC safety cutoff and stop.
#   True            → RESET battery at the start of each episode.
#                     Useful for comparing per-episode energy budgets.
RESET_BATTERY_EACH_EPISODE = False

# Live visualiser: skip every this many steps between screen redraws.
# Lower = smoother, higher = faster simulation.  30 ≈ 3 Hz at dt=0.01.
VIZ_UPDATE_INTERVAL = 30

# Auto-detect whether a graphical display is available.
# On headless Linux compute nodes (no $DISPLAY / $WAYLAND_DISPLAY), skip live viz.
_HAS_DISPLAY = (
    sys.platform in ("darwin", "win32")
    or bool(os.environ.get("DISPLAY"))
    or bool(os.environ.get("WAYLAND_DISPLAY"))
)


# ═══════════════════════════════════════════════════════════════════════════════
# CATEGORICAL MODE MAP  (shared by loop, live viz, and final plotter)
# ═══════════════════════════════════════════════════════════════════════════════
_MODES       = ["Ground", "Takeoff", "Hover", "Move"]
_MODE_INT    = {m: i for i, m in enumerate(_MODES)}
_MODE_COLORS = ["steelblue", "tomato", "mediumseagreen", "darkorange"]


# ═══════════════════════════════════════════════════════════════════════════════
# LIVE VISUALISER
# ═══════════════════════════════════════════════════════════════════════════════
class LiveVisualizer:
    """
    Real-time matplotlib monitor for the battery / flight test.

    Figure layout (18×10 window, 2-column GridSpec):
    ┌─────────────────────────┬──────────────────┐
    │  3D Flight View         │  Flight Mode     │
    │  (drone position,       │  Voltage         │
    │   gate markers,         │  SoC             │
    │   trajectory trail)     │  Capacity        │
    │                         │  Power           │
    └─────────────────────────┴──────────────────┘

    Design decisions:
    • set_data_3d() / set_xdata+set_ydata used instead of ax.clear() so that
      statically-plotted gate markers never need to be redrawn.
    • Battery internal lists accessed directly (_hist_*) — avoids allocating
      a full pandas DataFrame on every update tick.
    • Fill-between mode patches are stored as handles and removed/replaced
      each redraw so old bands do not accumulate.
    • plt.pause(0.001) keeps the GUI event loop alive; on headless servers
      the caller should set live_viz=False to skip this class entirely.
    """

    _TRAIL_LEN   = 300    # history positions shown in 3D trail
    _TRAIL_PRUNE = 600    # prune internal buffer when it exceeds this

    def __init__(
        self,
        gate_pos: np.ndarray,
        battery: LiPoBatteryModel,
        title: str = "",
    ) -> None:
        self._battery = battery
        self._gate_pos = gate_pos

        # Position trail (ENU convention for 3D display: z_up = -z_ned)
        self._trail_x: list[float] = []
        self._trail_y: list[float] = []
        self._trail_z: list[float] = []

        # Accumulated fill-between handles for the mode strip
        self._mode_fills: list = []

        plt.ion()
        matplotlib.rcParams.update({
            "axes.titlesize":  9,
            "axes.labelsize":  8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
        })

        self._fig = plt.figure(
            figsize=(18, 10),
            num="Airevolve — Live Battery Monitor",
        )
        gs = gridspec.GridSpec(
            5, 2, figure=self._fig,
            left=0.06, right=0.98, top=0.93, bottom=0.07,
            hspace=0.60, wspace=0.30,
        )

        # ── 3D flight axis (left column, all 5 rows) ──────────────────────────
        self._ax3 = self._fig.add_subplot(gs[:, 0], projection="3d")
        self._ax3.set_title("3D Flight View  (NED frame, z-axis flipped for display)")

        # Gate markers — static, plotted once
        gx, gy = gate_pos[:, 0], gate_pos[:, 1]
        gz     = -gate_pos[:, 2]     # NED z → display z_up
        self._ax3.scatter(gx, gy, gz, c="cyan", s=120, marker="o",
                          depthshade=False, label="Gates", zorder=5)
        for i, (xi, yi, zi) in enumerate(zip(gx, gy, gz)):
            self._ax3.text(xi, yi, zi + 0.1, str(i), fontsize=7, color="cyan")

        # Dynamic drone objects (initialised at gate-0 position)
        x0, y0, z0 = float(gx[0]), float(gy[0]), float(gz[0])
        self._dot, = self._ax3.plot(
            [x0], [y0], [z0], "ro", markersize=9, label="Drone", zorder=10)
        self._trail, = self._ax3.plot(
            [], [], [], "b-", alpha=0.35, linewidth=1.0)

        # Mode + battery HUD text (top-left of 3D axis)
        self._hud = self._ax3.text2D(
            0.02, 0.97, "Mode: Ground",
            transform=self._ax3.transAxes,
            fontsize=9, color="white", va="top",
            bbox=dict(facecolor="navy", alpha=0.75, pad=3),
        )

        # Set 3D axis limits from gate bounding box
        pad = 1.5
        r_xy = max(np.abs(gx).max(), np.abs(gy).max()) + pad
        r_z  = gz.max() + pad
        self._ax3.set_xlim(-r_xy, r_xy)
        self._ax3.set_ylim(-r_xy, r_xy)
        self._ax3.set_zlim(0, r_z)
        self._ax3.set_xlabel("X (m)")
        self._ax3.set_ylabel("Y (m)")
        self._ax3.set_zlabel("Z_up (m)")
        self._ax3.legend(fontsize=7, loc="upper right")

        # ── Right column: 5 mini-strip axes ───────────────────────────────────
        self._ax_mode = self._fig.add_subplot(gs[0, 1])
        self._ax_v    = self._fig.add_subplot(gs[1, 1])
        self._ax_soc  = self._fig.add_subplot(gs[2, 1])
        self._ax_cap  = self._fig.add_subplot(gs[3, 1])
        self._ax_pow  = self._fig.add_subplot(gs[4, 1])

        # Mode strip
        self._ax_mode.set_title("Flight Mode")
        self._ax_mode.set_yticks(range(len(_MODES)))
        self._ax_mode.set_yticklabels(_MODES)
        self._ax_mode.set_ylim(-0.65, len(_MODES) - 0.35)
        self._ax_mode.grid(axis="x", alpha=0.25)
        self._mode_line, = self._ax_mode.step(
            [], [], where="post", color="slategray", linewidth=1.2, zorder=3)

        # Voltage strip
        self._ax_v.set_title(
            f"Voltage (V)  ·  cutoff {battery.VOLTAGE_DEPLETION_THRESHOLD} V")
        self._ax_v.axhline(battery.VOLTAGE_DEPLETION_THRESHOLD,
                            color="red", linestyle=":", linewidth=1.0)
        self._ax_v.set_ylim(
            battery.VOLTAGE_DEPLETED - 0.3, battery.VOLTAGE_FULL + 0.3)
        self._ax_v.grid(alpha=0.25)
        self._v_line, = self._ax_v.plot([], [], color="royalblue", lw=1.2)

        # SoC strip
        self._ax_soc.set_title(
            f"SoC (%)  ·  reserve {battery.SOC_DEPLETION_THRESHOLD*100:.0f}%")
        self._ax_soc.axhline(battery.SOC_DEPLETION_THRESHOLD * 100,
                              color="red", linestyle=":", linewidth=1.0)
        self._ax_soc.set_ylim(0, 105)
        self._ax_soc.grid(alpha=0.25)
        self._soc_line, = self._ax_soc.plot([], [], color="crimson", lw=1.2)

        # Capacity strip
        res_mah = battery.CAPACITY_MAH * battery.SOC_DEPLETION_THRESHOLD
        self._ax_cap.set_title(
            f"Remaining Capacity (mAh)  ·  reserve {res_mah:.0f} mAh")
        self._ax_cap.axhline(res_mah, color="red", linestyle=":", linewidth=1.0)
        self._ax_cap.set_ylim(0, battery.CAPACITY_MAH + 10)
        self._ax_cap.grid(alpha=0.25)
        self._cap_line, = self._ax_cap.plot([], [], color="forestgreen", lw=1.2)

        # Power strip
        p_max = battery.FLIGHT_MODE_CURRENTS["Takeoff"] * battery.VOLTAGE_FULL
        self._ax_pow.set_title("Instantaneous Power (W)")
        self._ax_pow.set_ylim(0, p_max * 1.05)
        self._ax_pow.grid(alpha=0.25)
        self._ax_pow.set_xlabel("Simulation Time (s)")
        self._pow_line, = self._ax_pow.plot(
            [], [], color="darkorchid", lw=0.9, alpha=0.85)

        if title:
            self._fig.suptitle(title, fontsize=10, fontweight="bold")

        self._fig.canvas.draw()
        plt.pause(0.02)

    def update(self, kinematic_state: np.ndarray, step_num: int) -> None:
        """
        Refresh all live axes from the current drone state and battery history.

        Parameters
        ----------
        kinematic_state : (12+N,) NED state from DroneGateEnv.world_states[0]
        step_num        : global step counter (used in terminal output only)
        """
        # ── 3D drone position and trail ────────────────────────────────────────
        x  = float(kinematic_state[0])
        y  = float(kinematic_state[1])
        zu = float(-kinematic_state[2])   # NED → z_up

        self._trail_x.append(x)
        self._trail_y.append(y)
        self._trail_z.append(zu)

        # Prune buffers to avoid unbounded memory growth
        if len(self._trail_x) > self._TRAIL_PRUNE:
            self._trail_x = self._trail_x[-self._TRAIL_LEN:]
            self._trail_y = self._trail_y[-self._TRAIL_LEN:]
            self._trail_z = self._trail_z[-self._TRAIL_LEN:]

        tx = self._trail_x[-self._TRAIL_LEN:]
        ty = self._trail_y[-self._TRAIL_LEN:]
        tz = self._trail_z[-self._TRAIL_LEN:]

        self._trail.set_data_3d(tx, ty, tz)
        self._dot.set_data_3d([x], [y], [zu])

        # HUD text and colour
        mode  = self._battery._last_mode or "—"
        soc_p = self._battery.soc * 100.0
        volt  = self._battery.voltage
        hud_col = _MODE_COLORS[_MODE_INT.get(mode, 0)]
        self._hud.set_text(
            f"Mode: {mode}  |  SoC: {soc_p:.1f}%  |  {volt:.2f} V")
        self._hud.get_bbox_patch().set_facecolor(hud_col)

        # ── Battery strips ─────────────────────────────────────────────────────
        # Access internal lists directly — avoids allocating a DataFrame each tick.
        t_list  = self._battery._hist_time
        m_list  = self._battery._hist_mode
        v_list  = self._battery._hist_voltage
        s_list  = self._battery._hist_soc
        c_list  = self._battery._hist_capacity_mah
        pw_list = self._battery._hist_power

        if not t_list:
            return

        t  = np.asarray(t_list)
        mi = np.array([_MODE_INT.get(mo, 0) for mo in m_list], dtype=int)

        # Mode step line
        self._mode_line.set_xdata(t)
        self._mode_line.set_ydata(mi)

        # Mode fill-between bands: remove stale patches, redraw fresh ones
        for fill in self._mode_fills:
            fill.remove()
        self._mode_fills.clear()
        for idx, colour in enumerate(_MODE_COLORS):
            mask = mi == idx
            if np.any(mask):
                fill = self._ax_mode.fill_between(
                    t, idx - 0.4, idx + 0.4,
                    where=mask, step="post",
                    color=colour, alpha=0.30, linewidth=0, zorder=2,
                )
                self._mode_fills.append(fill)
        self._ax_mode.set_xlim(t[0], max(t[-1], t[0] + 0.5))

        # Simple strips — extend line data and let the axis auto-scale x
        strips = [
            (self._v_line,   np.asarray(v_list)),
            (self._soc_line, np.asarray(s_list) * 100.0),
            (self._cap_line, np.asarray(c_list)),
            (self._pow_line, np.asarray(pw_list)),
        ]
        for line, ydata in strips:
            line.set_xdata(t)
            line.set_ydata(ydata)

        x_lim = (t[0], max(t[-1], t[0] + 0.5))
        for ax in (self._ax_v, self._ax_soc, self._ax_cap, self._ax_pow):
            ax.set_xlim(*x_lim)

        # Auto-scale power axis (the others have stable fixed y-limits)
        pw_arr = np.asarray(pw_list)
        if pw_arr.size:
            self._ax_pow.set_ylim(0, max(float(pw_arr.max()) * 1.05, 5.0))

        self._fig.canvas.draw_idle()
        plt.pause(0.001)

    def finalize(self) -> None:
        """Freeze the live figure; the final static figure will appear next."""
        plt.ioff()
        # Leave the window open — user closes manually.  The final 5-subplot
        # figure will open as a separate window via plt.show() below.


# ═══════════════════════════════════════════════════════════════════════════════
# ARG PARSER
# ═══════════════════════════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--policy-path", default=POLICY_PATH,
                   help="path to a trained PPO .zip policy (optional)")
    p.add_argument("--episodes", type=int, default=NUM_EPISODES,
                   help=f"figure-8 episodes (default {NUM_EPISODES})")
    p.add_argument("--max-steps", type=int, default=MAX_STEPS,
                   help=f"steps per episode (default {MAX_STEPS})")
    p.add_argument("--no-viz", action="store_true",
                   help="skip live 3D visualiser (use on headless servers)")
    p.add_argument("--no-show", action="store_true",
                   help="save final figure to file instead of displaying it")
    p.add_argument("--save-fig", default="battery_4modes.png",
                   help="output path when --no-show is set")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    args = parse_args()
    live_viz = _HAS_DISPLAY and not args.no_viz

    # ── 1. Hexacopter morphology (no EA — direct propeller instantiation) ──────
    # create_standard_propeller_config("hex", …) builds 6 propeller dicts in a
    # flat 60° symmetric ring — the canonical sub-250 g hexacopter layout.
    propellers = create_standard_propeller_config(
        "hex", arm_length=ARM_LENGTH, prop_size=PROP_SIZE
    )
    print(
        f"Hexacopter morphology: {len(propellers)} motors | "
        f"arm = {ARM_LENGTH*100:.0f} cm | prop = {PROP_SIZE}\""
    )

    # ── 2. DroneGateEnv — figure-8 default track ──────────────────────────────
    # The default track has 8 gates at ±1.5 m radius, z = -1.5 m NED (1.5 m AGL).
    # render_mode=None: we implement our own live visualiser above — the env's
    # built-in render() only returns a state dict, not a graphical output.
    # Wide bounds (±50 m) prevent out-of-bounds termination with a random policy,
    # ensuring each episode runs for the full max_steps = 12 s.
    env = DroneGateEnv(
        num_envs=1,
        propellers=propellers,
        gates_ahead=1,
        num_state_history=0,
        num_action_history=0,
        render_mode=None,
        device=DEVICE,
        dt=DT,
        initialize_at_random_gates=False,   # deterministic start at gate 0
        seed=42,
        max_steps=args.max_steps,
        x_bounds=[-50, 50],
        y_bounds=[-50, 50],
        z_bounds=[-50, 10],
    )
    print(
        f"DroneGateEnv: {env.num_motors} motors | "
        f"obs = {env.observation_space.shape} | "
        f"dt = {DT} s | max_steps = {args.max_steps}"
    )

    # ── 3. RL policy ──────────────────────────────────────────────────────────
    # Load a trained policy if supplied; otherwise use a randomly-initialised PPO.
    # We never call model.learn() — the reward function is untouched.
    # We only call model.predict() and read world_states for battery logging.
    policy_kwargs = dict(
        activation_fn=torch.nn.ReLU,
        net_arch=dict(pi=[64, 64], vf=[64, 64]),
    )
    if args.policy_path and os.path.isfile(args.policy_path):
        print(f"Loading trained policy: {args.policy_path}")
        model = PPO.load(args.policy_path, env=env, device=DEVICE)
    else:
        if args.policy_path:
            print(f"[WARN] Policy file not found: {args.policy_path!r} — using random policy.")
        else:
            print("Using randomly-initialised PPO policy (no --policy-path supplied).")
        # n_steps must be ≥ max_steps for PPO's rollout buffer — no training happens.
        model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs,
                    n_steps=args.max_steps, device=DEVICE, verbose=0)

    # ── 4. Battery model ──────────────────────────────────────────────────────
    # DRAIN STRATEGY: continuous (RESET_BATTERY_EACH_EPISODE = False).
    # The battery is NOT reset between episodes.  Over 20 × 12 s = 240 s of
    # flight, a random policy draws roughly 8–18 A (chaotic motor commands
    # produce large velocity excursions → Takeoff / Move modes dominate).
    # Expected depletion: ~200–350 s at mixed modes, well within the run.
    battery = LiPoBatteryModel()
    print(
        f"Battery: {battery.BATTERY_NAME} | "
        f"{battery.CAPACITY_MAH} mAh | {battery.VOLTAGE_FULL} V full | "
        f"R_int = {battery.INTERNAL_RESISTANCE} Ω | "
        f"drain = {'per-episode reset' if RESET_BATTERY_EACH_EPISODE else 'continuous'}\n"
    )
    print(f"Mode current draws: {battery.FLIGHT_MODE_CURRENTS}")
    total_sim_s = args.episodes * args.max_steps * DT
    print(
        f"\nSimulation plan: {args.episodes} ep × {args.max_steps} steps × {DT}s "
        f"= {total_sim_s:.0f} s total sim time\n"
    )

    # ── 5. Live visualiser ────────────────────────────────────────────────────
    viz: LiveVisualizer | None = None
    if live_viz:
        try:
            viz = LiveVisualizer(
                gate_pos=DEFAULT_GATE_POS,
                battery=battery,
                title=(
                    f"Airevolve — ECM Live Battery Monitor  ·  "
                    f"{battery.BATTERY_NAME}  ·  "
                    f"{len(propellers)}-motor hexacopter  ·  "
                    f"{args.episodes} episodes"
                ),
            )
            print("Live visualiser started.  Close the window or press Ctrl-C to stop early.\n")
        except Exception as exc:
            print(f"[WARN] Live visualiser unavailable ({exc}); continuing headless.\n")
            viz = None

    # ── 6. Simulation loop ────────────────────────────────────────────────────
    obs           = env.reset()
    episode_count = 0
    global_step   = 0
    episode_end_times: list[float] = []    # battery time at each episode boundary

    # Safety cap prevents runaway loops if the episode done signal misbehaves.
    max_global_steps = args.episodes * args.max_steps * 2

    try:
        for _ in range(max_global_steps):

            # 6a. PPO step — deterministic=False adds exploration noise.
            #     Both a random and a trained policy work identically here.
            action, _ = model.predict(obs, deterministic=False)
            env.step_async(action)
            obs, _rewards, dones, _infos = env.step_wait()
            global_step += 1

            # 6b. Battery step — MODAL path.
            #     world_states[0] layout (DroneGateEnv NED frame):
            #       [0:3]   position   x, y, z_NED   (z < 0 when airborne)
            #       [3:6]   velocity   vx, vy, vz     (vz < 0 when ascending)
            #       [6:9]   attitude   phi, theta, psi
            #       [9:12]  ang. rate  p, q, r
            #       [12:]   motor speeds (normalised)
            #
            #     _classify_mode() reads [2] (altitude) and [3:6] (velocity):
            #       altitude ≤ 0.05 m → Ground   (on ground / just after reset)
            #       |vz|     > 0.30 m/s → Takeoff (fast vertical motion)
            #       √(vx²+vy²) > 0.30 m/s → Move (horizontal flight)
            #       otherwise           → Hover
            kinematic_state = env.world_states[0]
            battery.step(DT, kinematic_state)

            # 6c. Live visualiser refresh (every VIZ_UPDATE_INTERVAL steps)
            if viz is not None and (global_step % VIZ_UPDATE_INTERVAL == 0):
                viz.update(kinematic_state, global_step)

            # 6d. Episode bookkeeping
            if dones[0]:
                episode_count += 1
                t_now = battery._total_time_s
                episode_end_times.append(t_now)

                print(
                    f"  ep {episode_count:3d}/{args.episodes} | "
                    f"t = {t_now:8.2f} s | "
                    f"SoC = {battery.soc*100:5.1f}% | "
                    f"V = {battery.voltage:.3f} V | "
                    f"mode = {battery.mode!r:9s} | "
                    f"{'>>> DEPLETED <<<' if battery.is_depleted else ''}"
                )

                # Stop the simulation when the battery safety reserve is hit
                if battery.is_depleted:
                    print(
                        f"\n  Battery depleted at t = {t_now:.2f} s "
                        f"(after {episode_count} episodes).  "
                        f"Stopping simulation.\n"
                    )
                    break

                # Optionally reset battery between episodes
                if RESET_BATTERY_EACH_EPISODE:
                    battery.reset()
                    print(f"  → Battery reset for episode {episode_count + 1}.")

                if episode_count >= args.episodes:
                    break

    except KeyboardInterrupt:
        print("\n[Interrupted by user — plotting collected data.]\n")

    # ── 7. Final state summary ─────────────────────────────────────────────────
    print(f"\nFinal battery state: {battery}")
    total_j = battery.get_total_energy_consumed()
    print(
        f"Total energy consumed: {total_j:.1f} J  "
        f"({total_j / 3600:.4f} Wh)"
    )

    if viz is not None:
        viz.finalize()

    # ── 8. Extract history arrays ─────────────────────────────────────────────
    hist = battery.get_history()
    if hasattr(hist, "to_numpy"):
        time_arr    = hist["time"].to_numpy()
        mode_arr    = hist["mode"].to_numpy()
        voltage_arr = hist["voltage"].to_numpy()
        soc_arr     = hist["soc"].to_numpy()
        cap_mah_arr = hist["capacity_mah"].to_numpy()
        power_arr   = hist["power"].to_numpy()
    else:
        time_arr    = np.array(hist["time"])
        mode_arr    = np.array(hist["mode"])
        voltage_arr = np.array(hist["voltage"])
        soc_arr     = np.array(hist["soc"])
        cap_mah_arr = np.array(hist["capacity_mah"])
        power_arr   = np.array(hist["power"])

    mode_int_arr = np.array([_MODE_INT.get(m, 0) for m in mode_arr], dtype=int)

    # ── 9. Final 5-subplot figure ─────────────────────────────────────────────
    fig, axes = plt.subplots(
        5, 1, figsize=(14, 16), sharex=True,
        gridspec_kw={"hspace": 0.07},
    )
    fig.suptitle(
        f"LiPo Battery — ECM 4-Mode Full Analysis\n"
        f"{battery.BATTERY_NAME}  ·  "
        f"{len(propellers)}-motor hexacopter  ·  "
        f"{episode_count} episodes completed  ·  dt = {DT} s",
        fontsize=13, fontweight="bold",
    )

    _GK  = dict(alpha=0.25)
    _VLK = dict(color="dimgray", linestyle="--", linewidth=1.0, alpha=0.7)

    def _ep_lines(ax, label_top: bool = False) -> None:
        """Mark episode boundaries with dashed vertical lines."""
        for i, t in enumerate(episode_end_times[:-1]):
            ax.axvline(t, **_VLK)
            if label_top:
                ax.text(t + 0.15, 0.97, f" ep{i+2}",
                        transform=ax.get_xaxis_transform(),
                        fontsize=6.5, va="top", color="dimgray")

    # Subplot 1 — Flight Mode (categorical step + shaded bands)
    ax = axes[0]
    ax.step(time_arr, mode_int_arr, where="post",
            color="slategray", lw=1.0, zorder=3)
    for i, (mn, col) in enumerate(zip(_MODES, _MODE_COLORS)):
        mask = mode_int_arr == i
        if np.any(mask):
            ax.fill_between(time_arr, i - 0.45, i + 0.45,
                            where=mask, step="post",
                            color=col, alpha=0.35, lw=0, zorder=2)
    ax.set_yticks(range(len(_MODES)))
    ax.set_yticklabels(_MODES, fontsize=9)
    ax.set_ylim(-0.7, len(_MODES) - 0.3)
    ax.set_ylabel("Flight Mode")
    ax.set_title("Flight Mode", loc="left", pad=3)
    ax.grid(axis="x", **_GK)
    _ep_lines(ax, label_top=True)
    ax.legend(
        handles=[mpatches.Patch(facecolor=c, alpha=0.65, label=m)
                 for m, c in zip(_MODES, _MODE_COLORS)],
        fontsize=8, ncol=4, loc="upper right",
    )

    # Subplot 2 — Terminal Voltage
    ax = axes[1]
    ax.plot(time_arr, voltage_arr, color="royalblue", lw=1.4)
    ax.axhline(battery.VOLTAGE_DEPLETION_THRESHOLD,
               color="red", ls=":", lw=1.2,
               label=f"Cutoff ({battery.VOLTAGE_DEPLETION_THRESHOLD} V)")
    ax.set_ylabel("Voltage (V)")
    ax.set_title("Terminal Voltage (ECM)", loc="left", pad=3)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(**_GK)
    _ep_lines(ax)

    # Subplot 3 — State of Charge
    ax = axes[2]
    ax.plot(time_arr, soc_arr * 100.0, color="crimson", lw=1.4)
    ax.axhline(battery.SOC_DEPLETION_THRESHOLD * 100,
               color="red", ls=":", lw=1.2,
               label=f"Reserve ({battery.SOC_DEPLETION_THRESHOLD*100:.0f}%)")
    ax.set_ylabel("SoC (%)")
    ax.set_title("State of Charge", loc="left", pad=3)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(**_GK)
    _ep_lines(ax)

    # Subplot 4 — Remaining Capacity
    ax = axes[3]
    ax.plot(time_arr, cap_mah_arr, color="forestgreen", lw=1.4)
    res = battery.CAPACITY_MAH * battery.SOC_DEPLETION_THRESHOLD
    ax.axhline(res, color="red", ls=":", lw=1.2,
               label=f"Reserve ({res:.0f} mAh)")
    ax.set_ylabel("Capacity (mAh)")
    ax.set_title("Remaining Capacity", loc="left", pad=3)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(**_GK)
    _ep_lines(ax)

    # Subplot 5 — Instantaneous Power
    ax = axes[4]
    ax.plot(time_arr, power_arr, color="darkorchid", lw=1.0, alpha=0.85)
    # Annotate steady-state power at 100% SoC for each mode as reference lines
    for mn, i_m in battery.FLIGHT_MODE_CURRENTS.items():
        v_ref = max(
            battery.VOLTAGE_DEPLETED,
            13.0 + (16.8 - 13.0) * math.sqrt(1.0) - i_m * battery.INTERNAL_RESISTANCE,
        )
        p_ref = v_ref * i_m
        ax.axhline(p_ref, ls=":", lw=0.9, alpha=0.55,
                   label=f"{mn} @ full SoC ({p_ref:.1f} W)")
    ax.set_ylabel("Power (W)")
    ax.set_xlabel("Simulation Time (s)")
    ax.set_title("Instantaneous Power", loc="left", pad=3)
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    ax.grid(**_GK)
    _ep_lines(ax)

    plt.tight_layout()

    if args.no_show:
        fig.savefig(args.save_fig, dpi=150, bbox_inches="tight")
        print(f"Figure saved: {args.save_fig}")
    else:
        plt.ioff()   # leave interactive mode before final blocking show()
        plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
