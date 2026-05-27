from __future__ import annotations
import argparse
import os
import sys
from dataclasses import dataclass
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.save_util import load_from_zip_file

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from airevolve.evolution_tools.evaluators.drone_gate_env import DroneGateEnv
from airevolve.evolution_tools.evaluators.drone_gate_env_power import PowerAwareDroneEnv
from airevolve.simulator.visualization.animation import (
    create_camera, create_gate_geometry, draw_drone_and_forces,
    COLORS_BGR, DEFAULT_WIDTH, DEFAULT_HEIGHT,
)
from airevolve.simulator.visualization.drone_visualization import (
    create_grid, create_path, create_drone, set_thrust,
)
from airevolve.simulator.simulation.propeller_data import (
    create_standard_propeller_config, PROPELLER_LIBRARY,
)
from airevolve.simulator.simulation.battery_model import LiPoBatteryModel

VIDEOS_DIR = "__data__/evolved_videos"

GENOME_PATHS = {
    "ind0903": (
        "/Users/mikolajduchlinski/Desktop/results_folder_update/standard_ppo_power_ea/exp_standard_ppo_power_ea/rl_logs/generation_28/individual_0903/genome.npy"
    ),
}

MOTOR_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#795548"]
GRID_SIZE = 20
DRONE_BOX = [0.02, 0.02, 0.02]
PROP_RADIUS = 0.0254
THRUST_SCALE = 0.2
THRUST_BASE_LEN = 0.0

VIEW_W = 640
VIEW_H = 400
PLOT_W = 640
PLOT_H_MID = 800
PLOT_H_RIGHT = 800
BAT_H = 280

OUT_W = 1920
OUT_H = 1080
FPS = 50
RENDER_EVERY_N_STEPS = 2
DPI = 100

@dataclass
class Target:
    morph: str
    reward: str
    policy_zip: str
    train_seed: int
    env_seed: int
    expected_gates: int

TARGETS = [
    Target(
        "ind0903", "finalgate",
        "/Users/mikolajduchlinski/Desktop/results_folder_update/standard_ppo_power_ea/exp_standard_ppo_power_ea/rl_logs/generation_28/individual_0903/policy.zip",
        5, 1001, 31,
    ),
]

def _load_genome_arms(path: str) -> np.ndarray:
    obj = np.load(path, allow_pickle=True)
    inner = obj.item() if obj.dtype == object else obj
    arms = inner.arms if hasattr(inner, "arms") else inner
    return np.asarray(arms, dtype=float)

def _build_env(morph: str, env_seed: int, device: str, is_power_aware: bool) -> DroneGateEnv:
    common = dict(
        num_envs=1, gates_ahead=1, num_state_history=0, num_action_history=0,
        history_step_size=1, render_mode=None, device=device, dt=0.01,
        initialize_at_random_gates=True, seed=env_seed,
    )
    if is_power_aware:
        common["strict_voltage_kill"] = False
        common["randomize_soc"] = False
        EnvClass = PowerAwareDroneEnv
    else:
        EnvClass = DroneGateEnv

    if morph == "quad":
        return EnvClass(propellers=create_standard_propeller_config("quad", 0.11, 2), **common)
    if morph == "hex":
        return EnvClass(propellers=create_standard_propeller_config("hex", 0.11, 2), **common)
    if morph in GENOME_PATHS:
        arms = _load_genome_arms(os.path.join(REPO_ROOT, GENOME_PATHS[morph]))
        return EnvClass(individual=arms, **common)
    raise ValueError(f"unknown morph: {morph}")

def _get_wmax(propellers: list) -> float:
    prop_key = f"prop{propellers[0]['propsize']}"
    return float(PROPELLER_LIBRARY[prop_key]["wmax"])

class ViewRenderer:
    def __init__(self, view_type: str, propellers: list, gate_pos, gate_yaw):
        self.cam = create_camera(view_type, VIEW_W, VIEW_H)
        
        # Zoom out to see the full scene
        if view_type == 'top':
            self.cam.r[0] = -11.0
            # To move drone down-right, we move the camera target up-left
            self.center_offset = np.array([-1.0, 1.0, 0.0])
        else:
            self.cam.r[0] = -11.5
            self.center_offset = np.array([-2.0, 2.0, 0.0])
            
        self.gate_pos = gate_pos
        self.gate_yaw = gate_yaw
        self.num_arms = len(propellers)
        motor_colors_bgr = [
            COLORS_BGR.get(c, COLORS_BGR["white"])
            for c in ["red", "blue", "green", "orange", "purple", "brown"]
        ]
        self.drone, self.forces = create_drone(
            propellers, box_size=DRONE_BOX, prop_radius=PROP_RADIUS,
            scale=1, motor_colors=motor_colors_bgr,
        )
        self.grid = create_grid(GRID_SIZE, GRID_SIZE, 1)
        self.gate, _ = create_gate_geometry()
        self.path_pts: list[np.ndarray] = []

    def render(self, ws: np.ndarray, u: np.ndarray, gates_passed: int, follow: bool = True) -> np.ndarray:
        pos = ws[0:3]
        ori = ws[6:9]
        self.path_pts.append(pos.copy())

        if follow:
            self.cam.set_center(pos + self.center_offset)
        else:
            self.cam.set_center(np.zeros(3) + self.center_offset)

        frame = 255 * np.ones((VIEW_H, VIEW_W, 3), dtype=np.uint8)
        self.grid.draw(frame, self.cam, color=COLORS_BGR["gray"], pt=1)

        if len(self.path_pts) > 2:
            path_obj = create_path(self.path_pts[::5])
            path_obj.draw(frame, self.cam, color=(0, 200, 100), pt=1)

        state_dict = {"x": pos[0], "y": pos[1], "z": pos[2],
                      "phi": ori[0], "theta": ori[1], "psi": ori[2]}
        for i in range(self.num_arms):
            state_dict[f"u{i+1}"] = float(u[i]) if i < len(u) else 0.0

        pos_b = pos[np.newaxis, :]
        ori_b = ori[np.newaxis, :]
        u_b = u[np.newaxis, :]

        draw_drone_and_forces(
            self.drone, self.forces, pos_b[0], ori_b[0], u_b[0],
            state_dict, True, THRUST_SCALE, frame, self.cam,
        )

        for gpos, gyaw in zip(self.gate_pos, self.gate_yaw):
            self.gate.translate(gpos - self.gate.pos)
            self.gate.rotate([0, 0, gyaw])
            self.gate.draw(frame, self.cam, color=(0, 140, 255), pt=4)

        label = f"Gates: {gates_passed}"
        cv2.putText(frame, label, (10, VIEW_H - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS_BGR["black"], 2)

        return frame

class DashboardPlotter:
    def __init__(self, num_motors: int, total_steps: int,
                 soc_threshold: float, v_threshold: float):
        self.num_motors = num_motors
        self.total_steps = total_steps
        self.soc_thr = soc_threshold
        self.v_thr = v_threshold

        self.steps: list[int] = []
        self.speeds: list[float] = []
        self.ang_speeds: list[float] = []
        self.motor_actions: list[list[float]] = [[] for _ in range(num_motors)]
        self.soc_hist: list[float] = []
        self.volt_hist: list[float] = []
        self.curr_hist: list[float] = []
        self.gate_steps: list[int] = []

        self._build_figure()

    def _build_figure(self):
        n = self.num_motors
        
        # Left bottom figure for Current
        self.fig_left_bat = plt.figure(figsize=(VIEW_W / DPI, BAT_H / DPI), dpi=DPI)
        self.fig_left_bat.patch.set_facecolor("white")
        gs_left = gridspec.GridSpec(1, 1, figure=self.fig_left_bat,
                                    top=0.82, bottom=0.25,
                                    left=0.14, right=0.97)
        self.ax_curr = self.fig_left_bat.add_subplot(gs_left[0])
        _style_ax(self.ax_curr)
        self.ax_curr.set_title("Current Drawn  [A]", color="black", fontsize=11, pad=5)
        self.ax_curr.set_xlim(0, self.total_steps)
        self.ax_curr.set_ylim(0, 80)
        self.ax_curr.set_xlabel("Time Step", color="black", fontsize=9)
        self.ax_curr.set_ylabel("Current (A)", color="black", fontsize=9)
        self.line_curr, = self.ax_curr.plot([], [], color="#d84315", lw=1.5)

        self.fig_mid = plt.figure(figsize=(PLOT_W / DPI, (PLOT_H_MID + BAT_H) / DPI), dpi=DPI)
        self.fig_mid.patch.set_facecolor("white")
        gs_mid = gridspec.GridSpec(3, 1, figure=self.fig_mid,
                                   height_ratios=[2, 2, 1],
                                   hspace=0.45, top=0.94, bottom=0.07,
                                   left=0.14, right=0.97)
        self.ax_speed = self.fig_mid.add_subplot(gs_mid[0])
        self.ax_ang = self.fig_mid.add_subplot(gs_mid[1])
        self.ax_soc = self.fig_mid.add_subplot(gs_mid[2])

        for ax in [self.ax_speed, self.ax_ang, self.ax_soc]:
            _style_ax(ax)

        self.ax_speed.set_title("Speed Over Time with Gate Passages", color="black", fontsize=11, pad=5)
        self.ax_speed.set_xlim(0, self.total_steps)
        self.ax_speed.set_ylim(0, 12)
        self.ax_speed.set_xlabel("Time Step", color="black", fontsize=9)
        self.ax_speed.set_ylabel("Speed", color="black", fontsize=9)
        self.line_speed, = self.ax_speed.plot([], [], color="blue", lw=1.5, label="Speed")
        self.ax_speed.legend(loc="lower right")

        self.ax_ang.set_title("Angular Speed Over Time with Gate Passages", color="black", fontsize=11, pad=5)
        self.ax_ang.set_xlim(0, self.total_steps)
        self.ax_ang.set_ylim(0, 12)
        self.ax_ang.set_xlabel("Time Step", color="black", fontsize=9)
        self.ax_ang.set_ylabel("Angular Speed", color="black", fontsize=9)
        self.line_ang, = self.ax_ang.plot([], [], color="blue", lw=1.5, label="Angular Speed")
        self.ax_ang.legend(loc="lower right")

        self.ax_soc.set_title("State of Charge  [%]", color="black", fontsize=11, pad=5)
        self.ax_soc.set_xlim(0, self.total_steps)
        self.ax_soc.set_ylim(0, 105)
        self.ax_soc.set_xlabel("Time Step", color="black", fontsize=9)
        self.ax_soc.set_ylabel("SoC (%)", color="black", fontsize=9)
        self.ax_soc.axhline(self.soc_thr * 100, color="red", lw=1.2, ls="--",
                             label=f"Min {self.soc_thr*100:.0f}%")
        self.ax_soc.legend(fontsize=8, loc="lower left", facecolor="white", labelcolor="black")
        self.line_soc, = self.ax_soc.plot([], [], color="green", lw=1.5)

        self.fig_right = plt.figure(
            figsize=(PLOT_W / DPI, (PLOT_H_RIGHT + BAT_H) / DPI), dpi=DPI)
        self.fig_right.patch.set_facecolor("white")
        self.fig_right.text(
            0.5, 0.97, "Motor Actions Over Time with Gate Passages",
            ha="center", va="top", color="black", fontsize=12,
        )
        gs_right = gridspec.GridSpec(n + 1, 1, figure=self.fig_right,
                                     height_ratios=[1] * n + [1],
                                     hspace=0.55, top=0.92, bottom=0.06,
                                     left=0.14, right=0.97)
        self.ax_motors: list[plt.Axes] = []
        self.line_motors: list = []
        for i in range(n):
            ax = self.fig_right.add_subplot(gs_right[i])
            _style_ax(ax)
            ax.set_xlim(0, self.total_steps)
            ax.set_ylim(-0.05, 1.05)
            ax.set_yticks([0, 1])
            ax.set_ylabel(f"Motor {i+1}", color="black",
                          fontsize=9, rotation=90, labelpad=5)
            if i < n - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("", color="black", fontsize=9)
            line, = ax.plot([], [], color=MOTOR_COLORS[i % len(MOTOR_COLORS)], lw=1.1)
            self.ax_motors.append(ax)
            self.line_motors.append(line)

        self.ax_volt = self.fig_right.add_subplot(gs_right[n])
        _style_ax(self.ax_volt)
        self.ax_volt.set_title("V_terminal  [V]", color="black", fontsize=11, pad=5)
        self.ax_volt.set_xlim(0, self.total_steps)
        self.ax_volt.set_ylim(12.0, 17.5)
        self.ax_volt.set_xlabel("Time Step", color="black", fontsize=9)
        self.ax_volt.set_ylabel("Voltage (V)", color="black", fontsize=9)
        self.ax_volt.axhline(self.v_thr, color="red", lw=1.2, ls="--",
                              label=f"Min {self.v_thr:.1f} V")
        self.ax_volt.legend(fontsize=8, loc="lower left", facecolor="white", labelcolor="black")
        self.line_volt, = self.ax_volt.plot([], [], color="purple", lw=1.5)

    def push(self, step: int, ws: np.ndarray, actions: np.ndarray,
             battery_data: dict, gate_passed: bool):
        vel = ws[3:6]
        rates = ws[9:12]
        self.steps.append(step)
        self.speeds.append(float(np.linalg.norm(vel)))
        self.ang_speeds.append(float(np.linalg.norm(rates)))
        for i in range(self.num_motors):
            normalized = (float(actions[i]) + 1.0) / 2.0
            self.motor_actions[i].append(max(0.0, min(1.0, normalized)))
        self.soc_hist.append(battery_data["soc"] * 100.0)
        self.volt_hist.append(battery_data["voltage"])
        self.curr_hist.append(battery_data["current"])
        if gate_passed:
            self.gate_steps.append(step)

    def render_left_bat(self) -> np.ndarray:
        xs = self.steps
        self.line_curr.set_data(xs, self.curr_hist)
        _redraw_gate_lines(self.ax_curr, self.gate_steps)
        
        if self.curr_hist:
            m = max(self.curr_hist) * 1.15
            self.ax_curr.set_ylim(0, max(80, m))

        self.fig_left_bat.canvas.draw()
        buf = np.frombuffer(self.fig_left_bat.canvas.buffer_rgba(), dtype=np.uint8)
        h = int(self.fig_left_bat.get_figheight() * DPI)
        w = int(self.fig_left_bat.get_figwidth() * DPI)
        img = buf.reshape(h, w, 4)[:, :, :3]
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def render_mid(self) -> np.ndarray:
        xs = self.steps
        self.line_speed.set_data(xs, self.speeds)
        self.line_ang.set_data(xs, self.ang_speeds)
        self.line_soc.set_data(xs, self.soc_hist)

        for ax in [self.ax_speed, self.ax_ang]:
            _redraw_gate_lines(ax, self.gate_steps)

        if self.speeds:
            m = max(self.speeds) * 1.15
            self.ax_speed.set_ylim(0, max(12, m))
        if self.ang_speeds:
            m = max(self.ang_speeds) * 1.15
            self.ax_ang.set_ylim(0, max(12, m))

        self.fig_mid.canvas.draw()
        buf = np.frombuffer(self.fig_mid.canvas.buffer_rgba(), dtype=np.uint8)
        h = int(self.fig_mid.get_figheight() * DPI)
        w = int(self.fig_mid.get_figwidth() * DPI)
        img = buf.reshape(h, w, 4)[:, :, :3]
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def render_right(self) -> np.ndarray:
        xs = self.steps
        for i, (line, ax) in enumerate(zip(self.line_motors, self.ax_motors)):
            line.set_data(xs, self.motor_actions[i])
            _redraw_gate_lines(ax, self.gate_steps)

        self.line_volt.set_data(xs, self.volt_hist)
        _redraw_gate_lines(self.ax_volt, self.gate_steps)

        self.fig_right.canvas.draw()
        buf = np.frombuffer(self.fig_right.canvas.buffer_rgba(), dtype=np.uint8)
        h = int(self.fig_right.get_figheight() * DPI)
        w = int(self.fig_right.get_figwidth() * DPI)
        img = buf.reshape(h, w, 4)[:, :, :3]
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def _style_ax(ax: plt.Axes):
    ax.set_facecolor("white")
    ax.tick_params(colors="black", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#aaaaaa")
    ax.grid(True, color="#dddddd", lw=0.8, alpha=0.9)

_line_cache: dict[int, set] = {}

def _redraw_gate_lines(ax: plt.Axes, gate_steps: list[int]):
    aid = id(ax)
    if aid not in _line_cache:
        _line_cache[aid] = set()
    for gs in gate_steps:
        if gs not in _line_cache[aid]:
            ax.axvline(gs, color='red', linestyle='--', alpha=0.6, lw=1.2, zorder=0)
            _line_cache[aid].add(gs)

def render_target(t: Target, device: str, steps: int) -> str:
    # First check observation space size from zip
    data, _, _ = load_from_zip_file(t.policy_zip)
    obs_dim = data["observation_space"].shape[0]
    is_power_aware = (obs_dim == 25)

    env = _build_env(t.morph, t.env_seed, device, is_power_aware)
    model = PPO.load(t.policy_zip, env=env, device=device)

    propellers = env.drone_sim.config.propellers
    num_motors = len(propellers)
    wmax = _get_wmax(propellers)

    out_dir = os.path.join(REPO_ROOT, VIDEOS_DIR)
    os.makedirs(out_dir, exist_ok=True)
    out_name = (f"dashboard_{t.morph}_{t.reward}_seed{t.train_seed}"
                f"_env{t.env_seed}_{t.expected_gates}gates.mp4")
    out_path = os.path.join(out_dir, out_name)
    print(f"Rendering dashboard -> {out_path}", flush=True)

    battery = LiPoBatteryModel(strict_voltage_kill=False)
    iso_view = ViewRenderer("iso", propellers, env.gate_pos, env.gate_yaw)
    top_view = ViewRenderer("top", propellers, env.gate_pos, env.gate_yaw)
    dashboard = DashboardPlotter(
        num_motors=num_motors, total_steps=steps,
        soc_threshold=LiPoBatteryModel.SOC_DEPLETION_THRESHOLD,
        v_threshold=LiPoBatteryModel.VOLTAGE_DEPLETION_THRESHOLD,
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, FPS, (OUT_W, OUT_H))

    obs = env.reset()
    battery.reset()

    for step in range(steps):
        actions, _ = model.predict(obs, deterministic=True)
        obs, _, dones, infos = env.step(actions)

        ws = env.world_states[0]
        prev_u = env.prev_actions[0]
        gates_passed = int(env.num_gates_passed[0])
        gate_just_passed = bool(infos[0].get("gate_passed", False))

        w_norm = np.clip(ws[12:12 + num_motors], -1.0, 1.0)
        motor_rpms = ((w_norm + 1.0) / 2.0) * wmax
        bat = battery.step(dt=env.dt, motor_rpms=motor_rpms, max_rpm=wmax)

        dashboard.push(step, ws, prev_u, bat, gate_just_passed)

        if step % RENDER_EVERY_N_STEPS == 0:
            iso_frame = iso_view.render(ws, prev_u, gates_passed)
            top_frame = top_view.render(ws, prev_u, gates_passed)
            curr_img = dashboard.render_left_bat()

            left_col = np.vstack([iso_frame, top_frame, curr_img])

            mid_img = dashboard.render_mid()
            right_img = dashboard.render_right()

            frame = np.hstack([left_col, mid_img, right_img])

            cv2.putText(frame, f"Step {step:04d}/{steps}", (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

            out.write(frame)

        if step % 100 == 0:
            print(f"  step {step}/{steps}  gates={gates_passed}", flush=True)

    out.release()
    plt.close("all")
    _line_cache.clear()
    print(f"Saved -> {out_path}", flush=True)
    return out_path

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--device",
                   default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--steps", type=int, default=1200)
    return p.parse_args()

def main() -> None:
    args = parse_args()
    for t in TARGETS:
        render_target(t, args.device, args.steps)
    print("done", flush=True)

if __name__ == "__main__":
    main()