from __future__ import annotations
import argparse
import os
import sys
import cv2
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.save_util import load_from_zip_file

# Resolve project root (two levels up from examples/videos)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from airevolve.evolution_tools.evaluators.drone_gate_env import DroneGateEnv
from airevolve.evolution_tools.evaluators.drone_gate_env_power import PowerAwareDroneEnv
from airevolve.evolution_tools.evaluators.gate_train import (
    figure8, backandforth, circle, slalom,
)
from airevolve.simulator.visualization.animation import (
    create_camera, create_gate_geometry, draw_drone_and_forces,
    COLORS_BGR
)
from airevolve.simulator.visualization.drone_visualization import (
    create_grid, create_path, create_drone
)
from airevolve.simulator.simulation.propeller_data import (
    create_standard_propeller_config
)

VIDEOS_DIR = "__data__/evolved_videos"

GRID_SIZE = 20
DRONE_BOX = [0.02, 0.02, 0.02]
PROP_RADIUS = 0.0254
THRUST_SCALE = 0.2

VIEW_W = 1920
VIEW_H = 1080

OUT_W = VIEW_W * 2
OUT_H = VIEW_H * 2
FPS = 50
RENDER_EVERY_N_STEPS = 2

_TRACK_CFGS = {
    "figure8": figure8,
    "backandforth": backandforth,
    "circle": circle,
    "slalom": slalom,
}

def _load_genome_arms(path: str) -> np.ndarray:
    obj = np.load(path, allow_pickle=True)
    inner = obj.item() if obj.dtype == object else obj
    arms = inner.arms if hasattr(inner, "arms") else inner
    return np.asarray(arms, dtype=float)

def _build_env(ind_dir: str, env_seed: int, device: str, is_power_aware: bool,
               gates_ahead: int = 1, gate_cfg: str = "figure8") -> DroneGateEnv:
    track = _TRACK_CFGS[gate_cfg]
    common = dict(
        num_envs=1, gates_ahead=gates_ahead, num_state_history=0, num_action_history=0,
        history_step_size=1, render_mode=None, device=device, dt=0.01,
        initialize_at_random_gates=False, seed=env_seed,
        gates_pos=track.gate_pos,
        gate_yaw=track.gate_yaw,
        start_pos=track.starting_pos,
        x_bounds=track.x_bounds,
        y_bounds=track.y_bounds,
        z_bounds=track.z_bounds,
    )
    if is_power_aware:
        common["strict_voltage_kill"] = False
        common["randomize_soc"] = False
        EnvClass = PowerAwareDroneEnv
    else:
        EnvClass = DroneGateEnv

    basename = os.path.basename(os.path.normpath(ind_dir))
    if basename in ["quad", "hex"]:
        return EnvClass(propellers=create_standard_propeller_config(basename, 0.11, 2), **common)
    else:
        genome_path = os.path.join(ind_dir, "genome.npy")
        arms = _load_genome_arms(genome_path)
        return EnvClass(individual=arms, **common)

class ViewRenderer:
    def __init__(self, view_type: str, propellers: list, gate_pos, gate_yaw):
        self.cam = create_camera(view_type, VIEW_W, VIEW_H)
        
        if view_type == 'top':
            self.cam.r[0] = -5.0
            self.center_offset = np.array([1.0, -1.0, 0.0])
        else:
            self.cam.r[0] = -5.0
            self.center_offset = np.array([-2.0, -1.0, 0.0])
            
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

    def render(self, ws: np.ndarray, u: np.ndarray, gates_passed: int, energy_j: float, follow: bool = True) -> np.ndarray:
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

        label_energy = f"Energy: {energy_j:.1f} J"
        (tw, th), _ = cv2.getTextSize(label_energy, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.putText(frame, label_energy, (VIEW_W - tw - 10, VIEW_H - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS_BGR["black"], 2)

        return frame

def load_drone(ind_dir: str, env_seed: int, device: str, gate_cfg: str):
    policy_zip = os.path.join(ind_dir, "policy.zip")
    if not os.path.exists(policy_zip):
        raise FileNotFoundError(f"Cannot find policy.zip at {policy_zip}")
        
    data, _, _ = load_from_zip_file(policy_zip)
    obs_dim = int(data["observation_space"].shape[0])

    probe_env = _build_env(ind_dir, env_seed, device, is_power_aware=False, gate_cfg=gate_cfg)
    num_motors = len(probe_env.drone_sim.config.propellers)

    base = obs_dim - 18 - (num_motors - 6)
    is_power_aware = (base % 4) == 3
    gates_ahead = (base - (3 if is_power_aware else 0)) // 4
    if gates_ahead < 1:
        raise ValueError(f"Cannot infer a valid gates_ahead from obs_dim={obs_dim} for {ind_dir}")

    env = _build_env(ind_dir, env_seed, device, is_power_aware, gates_ahead, gate_cfg=gate_cfg)
    model = PPO.load(policy_zip, env=env, device=device)
    return env, model, env.drone_sim.config.propellers

def render_comparison(dir1: str, dir2: str, gate_cfg: str, device: str, steps: int) -> str:
    env_seed = 1001

    env1, model1, prop1 = load_drone(dir1, env_seed, device, gate_cfg)
    env2, model2, prop2 = load_drone(dir2, env_seed, device, gate_cfg)

    out_dir = os.path.join(REPO_ROOT, VIDEOS_DIR)
    os.makedirs(out_dir, exist_ok=True)
    name1 = os.path.basename(os.path.normpath(dir1))
    name2 = os.path.basename(os.path.normpath(dir2))
    out_name = f"compare_{name1}_vs_{name2}_{gate_cfg}.mp4"
    out_path = os.path.join(out_dir, out_name)
    print(f"Rendering comparison -> {out_path}", flush=True)

    top_view1 = ViewRenderer("top", prop1, env1.gate_pos, env1.gate_yaw)
    iso_view1 = ViewRenderer("iso", prop1, env1.gate_pos, env1.gate_yaw)
    
    top_view2 = ViewRenderer("top", prop2, env2.gate_pos, env2.gate_yaw)
    iso_view2 = ViewRenderer("iso", prop2, env2.gate_pos, env2.gate_yaw)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, FPS, (OUT_W, OUT_H))

    obs1 = env1.reset()
    obs2 = env2.reset()

    cum_energy1 = 0.0
    prev_energy1 = 0.0
    cum_energy2 = 0.0
    prev_energy2 = 0.0

    for step in range(steps):
        actions1, _ = model1.predict(obs1, deterministic=True)
        obs1, _, _, _ = env1.step(actions1)
        ws1 = env1.world_states[0]
        prev_u1 = env1.prev_actions[0]
        gates_passed1 = int(env1.num_gates_passed[0])

        curr_e1 = float(env1.bat_energy_j[0])
        if curr_e1 < prev_energy1:
            cum_energy1 += prev_energy1
        prev_energy1 = curr_e1
        total_e1 = cum_energy1 + curr_e1

        actions2, _ = model2.predict(obs2, deterministic=True)
        obs2, _, _, _ = env2.step(actions2)
        ws2 = env2.world_states[0]
        prev_u2 = env2.prev_actions[0]
        gates_passed2 = int(env2.num_gates_passed[0])

        curr_e2 = float(env2.bat_energy_j[0])
        if curr_e2 < prev_energy2:
            cum_energy2 += prev_energy2
        prev_energy2 = curr_e2
        total_e2 = cum_energy2 + curr_e2

        if step % RENDER_EVERY_N_STEPS == 0:
            top_frame1 = top_view1.render(ws1, prev_u1, gates_passed1, total_e1)
            iso_frame1 = iso_view1.render(ws1, prev_u1, gates_passed1, total_e1)
            
            top_frame2 = top_view2.render(ws2, prev_u2, gates_passed2, total_e2)
            iso_frame2 = iso_view2.render(ws2, prev_u2, gates_passed2, total_e2)

            # Left side: top video is top view, bottom video is iso view
            left_col = np.vstack([top_frame1, iso_frame1])
            
            # Right side: top video is top view, bottom video is iso view
            right_col = np.vstack([top_frame2, iso_frame2])

            frame = np.hstack([left_col, right_col])

            # Draw labels
            cv2.putText(frame, f"{name1} - Step {step:04d}/{steps}", (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            cv2.putText(frame, f"{name2} - Step {step:04d}/{steps}", (VIEW_W + 8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

            out.write(frame)

    out.release()
    print(f"Saved -> {out_path}", flush=True)
    return out_path

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render side-by-side comparison of 2 drones")
    p.add_argument("dir1", type=str, help="Directory for drone 1 (left side)")
    p.add_argument("dir2", type=str, help="Directory for drone 2 (right side)")
    p.add_argument("--gate_cfg", type=str, default="figure8", choices=["figure8", "backandforth", "circle", "slalom"],
                   help="Track/gate configuration to evaluate on.")
    p.add_argument("--device",
                   default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--steps", type=int, default=1200)
    return p.parse_args()

def main() -> None:
    args = parse_args()
    
    dir1 = os.path.abspath(args.dir1)
    dir2 = os.path.abspath(args.dir2)
    
    render_comparison(dir1, dir2, args.gate_cfg, args.device, args.steps)
    print("done", flush=True)

if __name__ == "__main__":
    main()
