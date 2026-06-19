import os
import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
torch.set_num_threads(1)

REPO_ROOT = Path("/projects/prjs2127/airevolve")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "examples" / "evolution"))

from stable_baselines3 import PPO
from airevolve.evolution_tools.evaluators.drone_gate_env_power import PowerAwareDroneEnv
from airevolve.evolution_tools.evaluators.gate_train import (
    backandforth, figure8, circle, slalom,
)
from airevolve.evolution_tools.selectors.nsga2_utils import (
    fast_non_dominated_sort,
    calculate_crowding_distance,
)

warnings.filterwarnings("ignore")

BASE_DIR = Path("/projects/prjs2127/airevolve/results")

RUN_DIRS = [
    "exp_standard_ppo_power_ea_figure8_rep1",
    "exp_standard_ppo_power_ea_figure8_rep2",
    "exp_standard_ppo_power_ea_figure8_rep3",
    "exp_standard_ppo_power_ea_figure8_rep4",
    "exp_standard_ppo_power_ea_figure8_rep5",
    "exp_standard_ppo_power_ea_shuttlerun_rep1",
    "exp_standard_ppo_power_ea_shuttlerun_rep2",
    "exp_standard_ppo_power_ea_shuttlerun_rep3",
    "exp_standard_ppo_power_ea_shuttlerun_rep4",
    "exp_standard_ppo_power_ea_shuttlerun_rep5",
]

GATE_CFG_MAP = {
    "backandforth": backandforth,
    "figure8": figure8,
    "circle": circle,
    "slalom": slalom,
}

MAX_STEPS = 1200
DT = 0.01
FAIL_ENERGY = 1e9


def get_gate_cfg_for_run(run_dir: Path) -> str:
    config_path = run_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)["gate_cfg"]
    if "shuttlerun" in run_dir.name:
        return "backandforth"
    return "figure8"


def create_eval_env(genome: np.ndarray, gate_cfg: str) -> PowerAwareDroneEnv:
    cfg = GATE_CFG_MAP[gate_cfg]
    env = PowerAwareDroneEnv(
        num_envs=1,
        individual=genome,
        gates_pos=cfg.gate_pos,
        gate_yaw=cfg.gate_yaw,
        start_pos=cfg.starting_pos,
        x_bounds=cfg.x_bounds,
        y_bounds=cfg.y_bounds,
        z_bounds=cfg.z_bounds,
        initialize_at_random_gates=False,
        gates_ahead=2,
        num_state_history=0,
        num_action_history=0,
        history_step_size=1,
        render_mode=None,
        device="cpu",
        max_steps=MAX_STEPS,
        experiment_type=2,
        penalty_weights={},
        randomize_soc=False,
        strict_voltage_kill=True,
    )
    return env


def evaluate_individual(genome: np.ndarray, model_path: str, gate_cfg: str) -> tuple:
    env = create_eval_env(genome, gate_cfg)
    try:
        model = PPO.load(model_path, device="cpu")
        obs = env.reset()

        max_gates = 0
        total_true_energy = 0.0

        for _ in range(MAX_STEPS):
            actions, _ = model.predict(obs, deterministic=True)
            obs, _rewards, _dones, infos = env.step(actions)

            total_true_energy += float(env.bat_power[0]) * DT

            current_gates = int(infos[0]["num_gates_passed"][0])
            if current_gates > max_gates:
                max_gates = current_gates

        return float(max_gates), total_true_energy
    finally:
        env.close()


def load_parent_ids_map(run_dir: Path) -> dict:
    parent_map = {}
    snapshots_dir = run_dir / "snapshots"
    if not snapshots_dir.exists():
        return parent_map
    for gen_dir in sorted(snapshots_dir.glob("gen_*")):
        csv_path = gen_dir / "pareto_info.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            if "id" in df.columns and "parent_ids" in df.columns:
                for _, row in df.iterrows():
                    parent_map[str(row["id"])] = str(row["parent_ids"])
    return parent_map


def compute_nsga2_ranks(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rank"] = -1
    df["crowding_distance"] = 0.0

    for gen, group in df.groupby("generation"):
        fitnesses = group[["gates_passed", "total_energy_j"]].values
        fronts = fast_non_dominated_sort(fitnesses)

        for rank_idx, front_indices in enumerate(fronts):
            original_indices = group.index[front_indices]
            df.loc[original_indices, "rank"] = rank_idx

            front_fitnesses = fitnesses[front_indices]
            cd = calculate_crowding_distance(front_fitnesses)
            df.loc[original_indices, "crowding_distance"] = cd

    return df


def process_run(run_name: str):
    run_dir = BASE_DIR / run_name
    rl_logs_dir = run_dir / "rl_logs"

    if not rl_logs_dir.exists():
        print(f"  [SKIP] No rl_logs directory found for {run_name}")
        return

    gate_cfg = get_gate_cfg_for_run(run_dir)
    parent_map = load_parent_ids_map(run_dir)

    print(f"  Gate config: {gate_cfg}")
    print(f"  Parent IDs loaded: {len(parent_map)} entries")

    records = []
    gen_dirs = sorted(rl_logs_dir.glob("generation_*"))

    for gen_dir in gen_dirs:
        gen_num = int(gen_dir.name.split("_")[-1])
        ind_dirs = sorted(gen_dir.glob("individual_*"))

        for ind_dir in ind_dirs:
            ind_id = ind_dir.name.split("_")[-1]

            genome_path = ind_dir / "genome.npy"
            if not genome_path.exists():
                print(f"    [WARN] No genome.npy for {gen_dir.name}/{ind_dir.name}, skipping")
                continue

            best_model = ind_dir / "best_model.zip"
            final_model = ind_dir / "final_model.zip"
            policy_model = ind_dir / "policy.zip"

            if best_model.exists():
                model_path = str(best_model)
            elif final_model.exists():
                model_path = str(final_model)
            elif policy_model.exists():
                model_path = str(policy_model)
            else:
                gates_passed = 0.0
                total_energy_j = FAIL_ENERGY
                parent_ids = parent_map.get(ind_id, "[]")
                records.append({
                    "id": ind_id,
                    "generation": gen_num,
                    "gates_passed": gates_passed,
                    "total_energy_j": total_energy_j,
                    "parent_ids": parent_ids,
                })
                print(f"    {gen_dir.name}/{ind_dir.name}: NO MODEL (non-hoverable) -> gates=0, energy=FAIL")
                continue

            genome = np.load(genome_path, allow_pickle=True).astype(np.float32)

            try:
                gates_passed, total_energy_j = evaluate_individual(genome, model_path, gate_cfg)
            except Exception as e:
                print(f"    [ERROR] {gen_dir.name}/{ind_dir.name}: {e}")
                gates_passed = 0.0
                total_energy_j = FAIL_ENERGY

            if gates_passed < 1.0:
                total_energy_j = FAIL_ENERGY

            parent_ids = parent_map.get(ind_id, "[]")

            records.append({
                "id": ind_id,
                "generation": gen_num,
                "gates_passed": gates_passed,
                "total_energy_j": total_energy_j,
                "parent_ids": parent_ids,
            })

            print(f"    {gen_dir.name}/{ind_dir.name}: gates={gates_passed:.2f}, energy={total_energy_j:.2f} J")

    if not records:
        print(f"  [SKIP] No records collected for {run_name}")
        return

    df = pd.DataFrame(records)
    df = compute_nsga2_ranks(df)
    df["fitness"] = df.apply(
        lambda row: f"({row['gates_passed']}, {row['total_energy_j']})", axis=1
    )

    out_csv = run_dir / "evolution_data_corrected.csv"
    df.to_csv(out_csv, index=False)
    print(f"  -> Saved {len(df)} rows to {out_csv}")

    last_gen = int(df["generation"].max())
    pareto_final = df[(df["generation"] == last_gen) & (df["rank"] == 0)]
    pareto_csv = run_dir / "pareto_front_final_corrected.csv"
    pareto_final.to_csv(pareto_csv, index=False)
    print(f"  -> Saved {len(pareto_final)} Pareto-optimal individuals to {pareto_csv}")


def main():
    print("=" * 70)
    print("CORRECTED RE-EVALUATION PIPELINE (PARALLEL)")
    print(f"Base directory: {BASE_DIR}")
    print(f"Eval steps: {MAX_STEPS} (dt={DT}, total={MAX_STEPS * DT:.1f}s)")
    print("=" * 70)

    # Check if a specific array task ID was passed from SLURM
    if len(sys.argv) > 1:
        task_id = int(sys.argv[1])
        if task_id < 0 or task_id >= len(RUN_DIRS):
            print(f"Error: Task ID {task_id} is out of bounds for RUN_DIRS.")
            sys.exit(1)
        
        target_run = RUN_DIRS[task_id]
        print(f"\n[Worker {task_id}] Processing exclusively: {target_run}...")
        process_run(target_run)
    else:
        for i, run_name in enumerate(RUN_DIRS):
            print(f"\n[{i + 1}/{len(RUN_DIRS)}] Processing {run_name}...")
            process_run(run_name)

    print("\n" + "=" * 70)
    print("ALL DONE")
    print("=" * 70)

if __name__ == "__main__":
    main()