"""Plotter for best-individual evaluation results.

Loads evaluation results from a directory structure of
design/task/rep_X/ and produces:
  - progress_heatmap: completion count by design x task
  - fitness_by_design: box plot of fitness per design
  - fitness_by_task: box plot of fitness per task
  - fitness_heatmap: mean fitness by design x task
  - duration_by_design: box plot of evaluation duration per design

When detailed=True, additionally produces:
  - lap_time_by_design / lap_time_by_task
  - max_reward_by_design / max_reward_by_task
"""

import os
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import torch
    from stable_baselines3 import PPO
    from airevolve.evolution_tools.inspection_tools.morphological_descriptors.hovering_info import get_sim
    from airevolve.evolution_tools.evaluators.drone_gate_env import DroneGateEnv
    from airevolve.evolution_tools.evaluators.gate_train import backandforth, circle, slalom, figure8
    SIMULATION_AVAILABLE = True
except ImportError:
    SIMULATION_AVAILABLE = False

from experimentation.config import (
    BASE_DIR, TASKS, GENOTYPES, COLORS, GENOTYPE_LABELS, TASK_LABELS,
    figures_dir,
)
from experimentation.plot_utils import (
    setup_style, save_figure, apply_axis_style,
    print_analysis_header, print_summary_stats, print_stat_tests,
    print_analysis_footer,
)

warnings.filterwarnings("ignore")

PLOTTER_NAME = "best_evaluations"

# Gate configurations for lap time calculation
GATE_COUNTS = {
    "circle": 4,
    "slalom": 4,
    "backandforth": 4,
    "figure8": 8,
}


# ── Data helpers ─────────────────────────────────────────────────────────────

def _calculate_stats(gate_times_sec, n_gates):
    """Calculate lap times, segment times, and their statistics."""
    segment_durations = np.diff(gate_times_sec)
    segment_indices = np.arange(len(segment_durations)) % n_gates

    avg_times, std_times = [], []
    for i in range(n_gates):
        segment_times = segment_durations[segment_indices == i]
        avg_times.append(np.mean(segment_times))
        std_times.append(np.std(segment_times))

    n_laps = len(gate_times_sec) // n_gates
    lap_times = []
    for lap_idx in range(n_laps):
        start_idx = lap_idx * n_gates
        end_idx = start_idx + n_gates - 1
        lap_times.append(gate_times_sec[end_idx] - gate_times_sec[start_idx])

    lap_times = np.array(lap_times)
    return {
        "avg_segment_times": avg_times,
        "std_segment_times": std_times,
        "lap_times": lap_times,
        "avg_lap_time": np.mean(lap_times),
        "std_lap_time": np.std(lap_times),
        "total_laps": n_laps,
    }


def _extract_simulation_data(individual, policy_file, gate_cfg, device):
    """Run a forward simulation and return positions, velocities, actions, etc."""
    sim = get_sim(individual)
    sim.compute_hover(verbose=False)

    gate_configs = {
        "backandforth": backandforth,
        "circle": circle,
        "slalom": slalom,
        "figure8": figure8,
    }
    if gate_cfg not in gate_configs:
        raise ValueError(f"Invalid gate configuration: {gate_cfg}")

    gate_config = gate_configs[gate_cfg]
    env = DroneGateEnv(
        num_envs=1,
        Bf=sim.Bf, Bm=sim.Bm,
        gates_pos=gate_config.gate_pos,
        gate_yaw=gate_config.gate_yaw,
        start_pos=gate_config.starting_pos,
        x_bounds=gate_config.x_bounds,
        y_bounds=gate_config.y_bounds,
        z_bounds=gate_config.z_bounds,
        initialize_at_random_gates=False,
        gates_ahead=1,
        num_state_history=0,
        num_action_history=0,
        history_step_size=1,
        render_mode=None,
        device=device,
    )

    policy_kwargs = dict(
        activation_fn=torch.nn.ReLU,
        net_arch=[dict(pi=[64, 64, 64], vf=[64, 64, 64])],
        log_std_init=0,
    )
    model = PPO(
        "MlpPolicy", env,
        policy_kwargs=policy_kwargs,
        verbose=0, n_steps=1000, batch_size=5000,
        n_epochs=10, gamma=0.999, device=device,
    )
    try:
        model = PPO.load(policy_file)
    except Exception:
        model = PPO.load(policy_file[:-4])

    env.reset()
    positions, velocities, angular_velocities, gate_passes, actions = [], [], [], [], []
    for _ in range(1200):
        action, _ = model.predict(env.states, deterministic=True)
        states, rewards, dones, infos = env.step(action)
        positions.append(env.world_states[0, 0:3])
        velocities.append(env.world_states[0, 3:6])
        angular_velocities.append(env.world_states[0, 9:12])
        gate_passes.append(infos[0]["gate_passed"])
        actions.append(action[0] * 0.5 + 0.5)

    return {
        "positions": np.array(positions[:-1]),
        "velocities": np.array(velocities[:-1]),
        "angular_velocities": np.array(angular_velocities[:-1]),
        "gate_passes": np.array(gate_passes[:-1]),
        "actions": np.array(actions[:-1]),
    }


def _read_monitor_file(monitor_file):
    """Read training monitor CSV and return (episode_rewards, timesteps)."""
    try:
        data = pd.read_csv(monitor_file, skiprows=1)
        episode_rewards = data["r"]
        time_steps = data["t"]
        if isinstance(episode_rewards.iloc[-1], str):
            episode_rewards = episode_rewards[:-1]
            time_steps = time_steps[:-1]
        return np.array(episode_rewards, dtype=float), np.array(time_steps, dtype=int)
    except Exception:
        return None, None


def _analyze_detailed_performance(exp_dir, design_name, task_name, device="cpu"):
    """Analyse detailed performance including lap times and max rewards."""
    result = {
        "avg_lap_time": None,
        "std_lap_time": None,
        "total_laps": None,
        "max_reward": None,
    }
    if not SIMULATION_AVAILABLE:
        return result

    try:
        policy_files = list(exp_dir.glob("*.zip")) + list(exp_dir.glob("*.pkl"))
        if not policy_files:
            return result
        individual_files = list(exp_dir.glob("individual.npy"))
        if not individual_files:
            return result

        individual = np.load(individual_files[0])
        sim_data = _extract_simulation_data(individual, str(policy_files[0]), task_name, device)

        gate_passes = sim_data["gate_passes"]
        n_gates = GATE_COUNTS.get(task_name, 4)
        gate_times = [i * 0.01 for i, passed in enumerate(gate_passes) if passed]

        if len(gate_times) >= n_gates:
            lap_stats = _calculate_stats(np.array(gate_times), n_gates)
            result["avg_lap_time"] = lap_stats["avg_lap_time"]
            result["std_lap_time"] = lap_stats["std_lap_time"]
            result["total_laps"] = lap_stats["total_laps"]

        monitor_files = list(exp_dir.glob("*monitor.csv")) + list(exp_dir.glob("monitor.csv"))
        if monitor_files:
            episode_rewards, _ = _read_monitor_file(monitor_files[0])
            if episode_rewards is not None and len(episode_rewards) > 0:
                result["max_reward"] = float(np.max(episode_rewards))
    except Exception:
        pass

    return result


def _load_results(base_dir, detailed=False, device="cpu"):
    """Load all evaluation results from the directory tree."""
    results = []
    base_path = Path(base_dir)
    if not base_path.exists():
        return pd.DataFrame()

    for design_dir in base_path.iterdir():
        if not design_dir.is_dir():
            continue
        design_name = design_dir.name
        for task_dir in design_dir.iterdir():
            if not task_dir.is_dir():
                continue
            task_name = task_dir.name
            for rep_dir in task_dir.iterdir():
                if not rep_dir.is_dir() or not rep_dir.name.startswith("rep_"):
                    continue
                rep_num = int(rep_dir.name.split("_")[1])

                entry = {
                    "design": design_name,
                    "task": task_name,
                    "repetition": rep_num,
                    "completed": False,
                    "fitness": None,
                    "duration": None,
                    "error": None,
                    "avg_lap_time": None,
                    "std_lap_time": None,
                    "total_laps": None,
                    "max_reward": None,
                }

                results_file = rep_dir / "results.pkl"
                fitness_file = rep_dir / "fitness.txt"

                if results_file.exists():
                    try:
                        with open(results_file, "rb") as f:
                            rd = pickle.load(f)
                        entry.update({
                            "completed": rd.get("success", False),
                            "fitness": rd.get("fitness"),
                            "duration": rd.get("duration"),
                            "error": rd.get("error"),
                        })
                    except Exception as e:
                        entry["error"] = f"Error loading results: {e}"
                elif fitness_file.exists():
                    try:
                        with open(fitness_file, "r") as f:
                            entry["fitness"] = float(f.read().strip())
                        entry["completed"] = True
                    except Exception as e:
                        entry["error"] = f"Error loading fitness: {e}"
                else:
                    entry["error"] = "No results found"

                if detailed and entry["completed"]:
                    entry.update(
                        _analyze_detailed_performance(rep_dir, design_name, task_name, device)
                    )

                results.append(entry)

    return pd.DataFrame(results)


# ── Plotting ─────────────────────────────────────────────────────────────────

def _plot_progress_heatmap(df, out_dir):
    """Completion-count heatmap (design x task)."""
    fig, ax = plt.subplots(figsize=(12, 8))
    matrix = df.pivot_table(
        values="completed", index="design", columns="task", aggfunc="sum", fill_value=0,
    )
    sns.heatmap(matrix, annot=True, fmt="d", cmap="RdYlGn",
                cbar_kws={"label": "Completed Experiments"}, ax=ax)
    apply_axis_style(ax, xlabel="Task", ylabel="Design")
    save_figure(fig, out_dir, "progress_heatmap")


def _plot_fitness_by_design(completed_df, out_dir):
    """Box plot of fitness grouped by design."""
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.boxplot(data=completed_df, x="design", y="fitness", ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    apply_axis_style(ax, xlabel="Design", ylabel="Fitness")
    save_figure(fig, out_dir, "fitness_by_design")


def _plot_fitness_by_task(completed_df, out_dir):
    """Box plot of fitness grouped by task."""
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.boxplot(data=completed_df, x="task", y="fitness", ax=ax)
    apply_axis_style(ax, xlabel="Task", ylabel="Fitness")
    save_figure(fig, out_dir, "fitness_by_task")


def _plot_fitness_heatmap(completed_df, out_dir):
    """Mean fitness heatmap (design x task)."""
    fig, ax = plt.subplots(figsize=(12, 8))
    matrix = completed_df.pivot_table(values="fitness", index="design", columns="task", aggfunc="mean")
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="viridis",
                cbar_kws={"label": "Mean Fitness"}, ax=ax)
    apply_axis_style(ax, xlabel="Task", ylabel="Design")
    save_figure(fig, out_dir, "fitness_heatmap")


def _plot_duration_by_design(completed_df, out_dir):
    """Box plot of evaluation duration grouped by design."""
    if "duration" not in completed_df.columns or completed_df["duration"].isna().all():
        return
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.boxplot(data=completed_df, x="design", y="duration", ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    apply_axis_style(ax, xlabel="Design", ylabel="Duration (seconds)")
    save_figure(fig, out_dir, "duration_by_design")


def _plot_lap_time_by_design(lap_df, out_dir):
    """Box plot of average lap time grouped by design."""
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.boxplot(data=lap_df, x="design", y="avg_lap_time", ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    apply_axis_style(ax, xlabel="Design", ylabel="Average Lap Time (s)")
    save_figure(fig, out_dir, "lap_time_by_design")


def _plot_lap_time_by_task(lap_df, out_dir):
    """Box plot of average lap time grouped by task."""
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.boxplot(data=lap_df, x="task", y="avg_lap_time", ax=ax)
    apply_axis_style(ax, xlabel="Task", ylabel="Average Lap Time (s)")
    save_figure(fig, out_dir, "lap_time_by_task")


def _plot_max_reward_by_design(reward_df, out_dir):
    """Box plot of max reward grouped by design."""
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.boxplot(data=reward_df, x="design", y="max_reward", ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    apply_axis_style(ax, xlabel="Design", ylabel="Maximum Reward")
    save_figure(fig, out_dir, "max_reward_by_design")


def _plot_max_reward_by_task(reward_df, out_dir):
    """Box plot of max reward grouped by task."""
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.boxplot(data=reward_df, x="task", y="max_reward", ax=ax)
    apply_axis_style(ax, xlabel="Task", ylabel="Maximum Reward")
    save_figure(fig, out_dir, "max_reward_by_task")


# ── Analysis helpers ─────────────────────────────────────────────────────────

def _print_fitness_analysis(completed_df):
    """Print fitness summary statistics and pairwise tests by design."""
    if completed_df.empty:
        return

    data_by_design = {}
    for design in completed_df["design"].unique():
        vals = completed_df[completed_df["design"] == design]["fitness"].dropna().values
        if len(vals) > 0:
            data_by_design[design] = vals

    print_analysis_header("Fitness by Design")
    print_summary_stats(data_by_design, metric_label="fitness")
    print_stat_tests(data_by_design, test_name="Mann-Whitney U")
    print_analysis_footer("(cross-task fitness)")


def _print_task_fitness_analysis(completed_df):
    """Print fitness summary statistics and pairwise tests by task."""
    if completed_df.empty:
        return

    data_by_task = {}
    for task in completed_df["task"].unique():
        vals = completed_df[completed_df["task"] == task]["fitness"].dropna().values
        if len(vals) > 0:
            data_by_task[task] = vals

    print_analysis_header("Fitness by Task")
    print_summary_stats(data_by_task, metric_label="fitness")
    print_stat_tests(data_by_task, test_name="Mann-Whitney U")
    print_analysis_footer("(cross-task fitness)")


# ── Entry point ──────────────────────────────────────────────────────────────

def run(base_dir, tasks, genotypes, output_base=None, detailed=False, device="cpu"):
    """Run all best-evaluation plots and analyses.

    Args:
        base_dir: root data directory (evaluation results tree)
        tasks: list of task names (used for filtering / labels)
        genotypes: list of genotype names (unused here but kept for interface consistency)
        output_base: override for output root (defaults to config.output_dir)
        detailed: perform detailed analysis (lap times, max rewards) via simulation
        device: torch device for detailed simulation
    """
    setup_style()

    out_dir = figures_dir(base_dir, task="cross_task", plotter_name=PLOTTER_NAME)

    df = _load_results(base_dir, detailed=detailed, device=device)
    if df.empty:
        print(f"No evaluation results found in {base_dir}")
        return

    # Progress heatmap (all experiments)
    _plot_progress_heatmap(df, out_dir)

    # Completed experiments with fitness
    completed_df = df[df["completed"] & df["fitness"].notna()]
    if len(completed_df) == 0:
        print("No completed experiments with fitness data.")
        return

    _plot_fitness_by_design(completed_df, out_dir)
    _plot_fitness_by_task(completed_df, out_dir)
    _plot_fitness_heatmap(completed_df, out_dir)
    _plot_duration_by_design(completed_df, out_dir)

    # Analytical output
    _print_fitness_analysis(completed_df)
    _print_task_fitness_analysis(completed_df)

    # Best performing experiment
    best = completed_df.loc[completed_df["fitness"].idxmin()]
    print_analysis_header("Best Performing Experiment")
    print(f"  Design: {best['design']}")
    print(f"  Task:   {best['task']}")
    print(f"  Fitness: {best['fitness']:.2f}")
    print_analysis_footer(out_dir)

    # Detailed plots (lap times, rewards)
    if detailed:
        lap_df = df[df["completed"] & df["avg_lap_time"].notna()]
        if len(lap_df) > 0:
            _plot_lap_time_by_design(lap_df, out_dir)
            _plot_lap_time_by_task(lap_df, out_dir)

        reward_df = df[df["completed"] & df["max_reward"].notna()]
        if len(reward_df) > 0:
            _plot_max_reward_by_design(reward_df, out_dir)
            _plot_max_reward_by_task(reward_df, out_dir)


if __name__ == "__main__":
    run(BASE_DIR, TASKS, GENOTYPES)
