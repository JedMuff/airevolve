#!/usr/bin/env python3
"""Plot reward curves for hover and gate stages of an individual."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_monitor_csv(path: Path) -> pd.DataFrame:
    """Load a stable-baselines3 monitor CSV (first line is JSON comment)."""
    return pd.read_csv(path, skiprows=1)


def plot_reward_curves(individual_dir: str, window: int = 100):
    individual_dir = Path(individual_dir)
    hover_path = individual_dir / "stage1_hover.monitor.csv"
    gate_path = individual_dir / "stage2_gate.monitor.csv"

    stages = []
    if hover_path.exists():
        stages.append(("Stage 1: Hover", hover_path))
    if gate_path.exists():
        stages.append(("Stage 2: Gate", gate_path))

    if not stages:
        print(f"No monitor CSVs found in {individual_dir}", file=sys.stderr)
        sys.exit(1)

    fig, axes = plt.subplots(1, len(stages), figsize=(7 * len(stages), 5), squeeze=False)
    axes = axes[0]

    for ax, (title, path) in zip(axes, stages):
        df = load_monitor_csv(path)
        episodes = np.arange(len(df))
        rewards = df["r"].values

        ax.plot(episodes, rewards, alpha=0.15, color="steelblue", linewidth=0.5)

        if len(rewards) >= window:
            smoothed = pd.Series(rewards).rolling(window, min_periods=1).mean()
            ax.plot(episodes, smoothed, color="steelblue", linewidth=2,
                    label=f"rolling mean (w={window})")
            ax.legend()

        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.grid(True, alpha=0.3)

    fig.suptitle(individual_dir.name, fontsize=14, fontweight="bold")
    fig.tight_layout()
    out_path = individual_dir / "reward_curves.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("individual_dir", help="Path to an individual's directory")
    parser.add_argument("-w", "--window", type=int, default=100,
                        help="Rolling-mean window size (default: 100)")
    args = parser.parse_args()
    plot_reward_curves(args.individual_dir, window=args.window)
