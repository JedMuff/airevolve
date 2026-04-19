"""
Quick benchmark to estimate per-individual evaluation time.
Generates one valid individual, then compares default threading vs single-thread per worker.
"""
import sys, os, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
from multiprocessing import Pool

# ── Step 1: Generate a valid individual ──────────────────────────────────────
print("=" * 70)
print("Step 1: Generating a valid individual...")
print("=" * 70)

from airevolve.evolution_tools.genome_handlers.spherical_angular_genome_handler import SphericalAngularDroneGenomeHandler
from airevolve.evolution_tools.genome_handlers.repair_workflow import (
    stage1_optimization_repair, stage2_hover_check, stage3_hover_repair
)
from airevolve.evolution_tools.genome_handlers.operators.optimization_repair_operator import OptimizationRepairConfig

spherical_params = np.array([
    [0.055, 0.17],           # magnitude
    [-np.pi, np.pi],         # arm yaw (azimuth)
    [-np.pi / 2, np.pi / 2], # arm pitch (elevation)
    [-np.pi, np.pi],         # motor pitch
    [-np.pi, np.pi],         # motor yaw
    [0, 1],                  # direction
])

handler_kwargs = {
    'min_max_narms': (4, 4),
    'append_arm_chance': 0.0,
    'parameter_limits': spherical_params,
    'bilateral_plane_for_symmetry': None,
    'repair': False,
}

def try_gen(idx):
    h = SphericalAngularDroneGenomeHandler(**handler_kwargs, rnd=np.random.default_rng(idx))
    ind = h.random_population(1)[0]
    ok, _ = stage2_hover_check(ind, verbose=False, allow_spinning=False)
    if not ok:
        return None
    rep, _ = stage1_optimization_repair(ind, coordinate_system='spherical',
                                         config=OptimizationRepairConfig(fixed_params=[3,4]), verbose=False)
    if rep is None:
        return None
    final, _ = stage3_hover_repair(rep, coordinate_system='spherical', verbose=False)
    return final

gen_start = time.time()
valid_ind = None
batch = 0
while valid_ind is None:
    seeds = list(range(batch * 5000, (batch + 1) * 5000))
    with Pool(32) as pool:
        for result in pool.imap_unordered(try_gen, seeds, chunksize=50):
            if result is not None:
                valid_ind = result
                break
    batch += 1

gen_time = time.time() - gen_start
print(f"Found valid individual in {gen_time:.1f}s")
print(f"Individual shape: {valid_ind.shape}")
print()

# ── Benchmark helper ─────────────────────────────────────────────────────────
from airevolve.evolution_tools.evaluators.drone_hover_env import DroneHoverEnv
from airevolve.evolution_tools.evaluators.drone_gate_env import DroneGateEnv
from airevolve.evolution_tools.evaluators import gate_train
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor

DEVICE = 'cpu'
NUM_ENVS = 100
SAMPLE_STEPS = 100_000

policy_kwargs = dict(
    activation_fn=torch.nn.ReLU,
    net_arch=dict(pi=[64,64,64], vf=[64,64,64]),
    log_std_init=0.041
)

def run_hover_bench(label):
    os.makedirs('/tmp/benchmark_eval', exist_ok=True)
    env = DroneHoverEnv(
        num_envs=NUM_ENVS, individual=valid_ind,
        target_pos=np.array([0.0, 0.0, -1.5], dtype=np.float32),
        x_bounds=[-5,5], y_bounds=[-5,5], z_bounds=[-5,0],
        position_tolerance=0.5, velocity_tolerance=0.3, motor_limit=1.0, device=DEVICE
    )
    env = VecMonitor(env, filename=f'/tmp/benchmark_eval/hover_{label}')
    model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=0,
                learning_rate=1e-4, n_steps=256, batch_size=128, n_epochs=22,
                gamma=0.9965, gae_lambda=0.894, clip_range=0.156,
                ent_coef=0.0044, vf_coef=0.55, max_grad_norm=3.19, device=DEVICE)
    t = time.time()
    model.learn(total_timesteps=SAMPLE_STEPS, log_interval=None)
    elapsed = time.time() - t
    env.close()
    return SAMPLE_STEPS / elapsed

def run_gate_bench(label):
    os.makedirs('/tmp/benchmark_eval', exist_ok=True)
    env = DroneGateEnv(
        num_envs=NUM_ENVS, individual=valid_ind,
        gates_pos=gate_train.figure8.gate_pos, gate_yaw=gate_train.figure8.gate_yaw,
        start_pos=gate_train.figure8.starting_pos,
        x_bounds=gate_train.figure8.x_bounds, y_bounds=gate_train.figure8.y_bounds,
        z_bounds=gate_train.figure8.z_bounds,
        gates_ahead=1, motor_limit=1.0, device=DEVICE
    )
    env = VecMonitor(env, filename=f'/tmp/benchmark_eval/gate_{label}')
    model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=0,
                learning_rate=1e-4, n_steps=256, batch_size=128, n_epochs=22,
                gamma=0.9965, gae_lambda=0.894, clip_range=0.156,
                ent_coef=0.0044, vf_coef=0.55, max_grad_norm=3.19, device=DEVICE)
    t = time.time()
    model.learn(total_timesteps=SAMPLE_STEPS, log_interval=None)
    elapsed = time.time() - t
    env.close()
    return SAMPLE_STEPS / elapsed

# ── Test A: Default threads (no limits) ──────────────────────────────────────
print("=" * 70)
print("Test A: Default threading (no limits)")
print(f"  torch threads: {torch.get_num_threads()}")
print("=" * 70)

hover_rate_a = run_hover_bench("default")
gate_rate_a = run_gate_bench("default")
print(f"  Hover: {hover_rate_a:,.0f} steps/sec")
print(f"  Gate:  {gate_rate_a:,.0f} steps/sec")

# ── Test B: Single thread (simulates parallel worker) ────────────────────────
print()
print("=" * 70)
print("Test B: Single thread (simulates 1 of 12 parallel workers)")
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
torch.set_num_threads(1)
print(f"  torch threads: {torch.get_num_threads()}")
print("=" * 70)

hover_rate_b = run_hover_bench("single")
gate_rate_b = run_gate_bench("single")
print(f"  Hover: {hover_rate_b:,.0f} steps/sec")
print(f"  Gate:  {gate_rate_b:,.0f} steps/sec")

# ── Comparison ───────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("Comparison (device=cpu, num_envs=100)")
print("=" * 70)

HOVER_STEPS = 2_000_000
GATE_STEPS = 10_000_000

for label, hr, gr in [("Default threads", hover_rate_a, gate_rate_a),
                       ("Single thread",   hover_rate_b, gate_rate_b)]:
    h_est = HOVER_STEPS / hr
    g_est = GATE_STEPS / gr
    total = h_est + g_est
    print(f"\n  {label}:")
    print(f"    Hover  (2M):  {h_est:6.1f}s ({h_est/60:5.1f} min) @ {hr:,.0f} sps")
    print(f"    Gate  (10M):  {g_est:6.1f}s ({g_est/60:5.1f} min) @ {gr:,.0f} sps")
    print(f"    Total:        {total:6.1f}s ({total/60:5.1f} min) per individual")
    print(f"    12 workers:   {total/60:5.1f} min per generation")

ratio = hover_rate_b / hover_rate_a
print(f"\n  Single-thread is {ratio:.2f}x vs default per core")
print(f"  But 12 single-thread workers = {12*ratio:.1f}x total throughput")
print()

# Cleanup
import shutil
shutil.rmtree('/tmp/benchmark_eval', ignore_errors=True)
