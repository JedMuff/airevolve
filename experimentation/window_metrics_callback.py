"""Per-window CSV callback. Logs gate-passes, mean episode reward, mean
episode length, and episode count once every `window` env steps. A copy
lives in both airevolve/experimentation/ and optimal_quad_control_RL/
because the two repos do not share sys.path."""
import csv
import os
from stable_baselines3.common.callbacks import BaseCallback


class WindowMetricsCallback(BaseCallback):
    def __init__(self, window: int = 100_000, csv_path: str = "window_metrics.csv", verbose: int = 0):
        super().__init__(verbose)
        self.window = window
        self.csv_path = csv_path
        self.last_log = 0
        self.gates_in_window = 0
        self.ep_rewards: list[float] = []
        self.ep_lens: list[int] = []
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["timestep", "gate_passes", "mean_ep_reward", "mean_ep_length", "n_episodes"]
            )

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if info.get("gate_passed", False):
                self.gates_in_window += 1
            ep = info.get("episode")
            if ep is not None:
                self.ep_rewards.append(float(ep["r"]))
                self.ep_lens.append(int(ep["l"]))
        if self.num_timesteps - self.last_log >= self.window:
            mean_r = sum(self.ep_rewards) / len(self.ep_rewards) if self.ep_rewards else float("nan")
            mean_l = sum(self.ep_lens) / len(self.ep_lens) if self.ep_lens else float("nan")
            with open(self.csv_path, "a", newline="") as f:
                csv.writer(f).writerow(
                    [self.num_timesteps, self.gates_in_window, mean_r, mean_l, len(self.ep_rewards)]
                )
            self.logger.record("custom/gate_passes_per_window", self.gates_in_window)
            self.gates_in_window = 0
            self.ep_rewards.clear()
            self.ep_lens.clear()
            self.last_log = self.num_timesteps
        return True
