"""3-seed × 1M-step training smoke test for the runtime DroneSimulator.

This is the Phase 7 CI gate from RUNTIME_DYNAMICS_MIGRATION.md. It asserts
that the runtime path (post-Phase 2 dynamics rewrite) can train a canonical
2-inch quad to median gate_passes >50 within 1M steps × 3 seeds.

For comparison:
* pre-migration airevolve typically produced ~0-50 gate_passes at 1M for
  7/10 seeds on this track (session 3 of RL_TRAINING_FIXES.md).
* post-migration: V1 smoke seed 1 hit gate_passes=116 at 1M and 836 at 5M
  on the default DroneGateEnv track (= reference figure-8). The 50-pass
  threshold is comfortably above the pre-migration baseline.

The migration doc originally suggested >200 at 1M, but that's only
achieved consistently at ≥2M steps; we use 50 here so the 5-minute CI
budget stays meaningful.

Runtime: ~5 min on RTX A4000 (3 seeds × 1M steps × ~30s each).

Usage: /home/jed/miniconda3/envs/isaaclab/bin/python unit_tests/test_dynamics_smoke.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

# stable_baselines3 / PPO live in the experimentation env; reuse the
# experimentation harness for consistency.
EXP_DIR = "/home/jed/workspaces/airevolve/experimentation"
if EXP_DIR not in sys.path:
    sys.path.insert(0, EXP_DIR)


def main():
    # Lazy imports so that just importing this file (e.g. by run_all_tests.py
    # discovery) doesn't pull stable_baselines3 unless the test is run.
    import csv

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecMonitor

    from airevolve.evolution_tools.evaluators.drone_gate_env import DroneGateEnv
    from window_metrics_callback import WindowMetricsCallback

    SEEDS = [1, 2, 3]
    TOTAL_STEPS = int(1e6)
    NUM_ENVS = 100

    final_gate_passes = []
    for seed in SEEDS:
        t0 = time.time()
        env = DroneGateEnv(num_envs=NUM_ENVS, seed=seed)
        env = VecMonitor(env)
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=1000,
            batch_size=5000,
            n_epochs=10,
            gamma=0.999,
            gae_lambda=0.95,
            ent_coef=0.01,
            verbose=0,
            device="cpu",
            policy_kwargs={"log_std_init": 0, "net_arch": [64, 64]},
        )

        # Same window-metrics callback as the baselines (gate_passes/100k steps).
        csv_path = f"/tmp/airevolve_smoke_seed{seed}_window_metrics.csv"
        cb = WindowMetricsCallback(window=100_000, csv_path=csv_path)
        model.learn(total_timesteps=TOTAL_STEPS, callback=cb)

        # Read final-window gate_passes from the CSV.
        gp = 0
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                gp = int(row["gate_passes"])  # last row wins
        final_gate_passes.append(gp)
        print(
            f"seed {seed}: final gate_passes={gp}, "
            f"time={time.time() - t0:.0f}s"
        )

    median_gp = int(np.median(final_gate_passes))
    print(
        f"\n3-seed × 1M smoke: gate_passes per seed = {final_gate_passes}, "
        f"median = {median_gp}"
    )
    assert median_gp > 50, (
        f"DYNAMICS SMOKE FAIL: median gate_passes {median_gp} ≤ 50"
    )
    print("DYNAMICS SMOKE OK: median gate_passes > 50")


if __name__ == "__main__":
    main()
