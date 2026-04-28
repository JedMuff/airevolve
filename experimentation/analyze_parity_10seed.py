"""Aggregate 10-seed parity results and produce comparison plots.

Default root: comparison_results/. Override with --root-dir for sessions
that write to a sibling directory (e.g., comparison_results_session4_refdyn).
"""
import argparse
import glob
import os

import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_ROOT = "/home/jed/workspaces/airevolve/experimentation/comparison_results"


def load(root: str, framework: str) -> pd.DataFrame:
    rows = []
    for csv_path in sorted(glob.glob(f"{root}/{framework}/seed_*/window_metrics.csv")):
        seed = int(os.path.basename(os.path.dirname(csv_path)).split("_")[1])
        df = pd.read_csv(csv_path)
        df["seed"] = seed
        df["framework"] = framework
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root-dir", default=DEFAULT_ROOT,
                   help="dir containing airevolve/seed_*/ and optimal/seed_*/ subdirs")
    args = p.parse_args()
    root = args.root_dir

    aev = load(root, "airevolve")
    opt = load(root, "optimal")
    if aev.empty or opt.empty:
        raise SystemExit(f"missing data: airevolve={len(aev)} rows, optimal={len(opt)} rows under {root}")
    dfs = pd.concat([aev, opt], ignore_index=True)

    plots = [
        ("mean_ep_reward", "mean episode reward"),
        ("mean_ep_length", "mean episode length (steps)"),
        ("gate_passes", "gate passes per 100k env steps"),
    ]
    for metric, ylabel in plots:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for fw, color in [("airevolve", "tab:blue"), ("optimal", "tab:orange")]:
            sub = dfs[dfs.framework == fw]
            agg = sub.groupby("timestep")[metric].agg(
                median="median",
                q25=lambda x: x.quantile(0.25),
                q75=lambda x: x.quantile(0.75),
            )
            ax.plot(agg.index, agg["median"], color=color, label=f"{fw} (median)", linewidth=2)
            ax.fill_between(agg.index, agg["q25"], agg["q75"], color=color, alpha=0.2,
                            label=f"{fw} (IQR)")
        ax.set_xlabel("training timestep")
        ax.set_ylabel(ylabel)
        ax.set_title(f"airevolve vs optimal_quad_control_RL — {ylabel}")
        ax.legend(loc="best")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        out = os.path.join(root, f"compare_{metric}.png")
        fig.savefig(out, dpi=130)
        print(f"wrote {out}")
        plt.close(fig)

    print("\n=== final-window summary (median across seeds) ===")
    final = dfs.sort_values("timestep").groupby(["framework", "seed"]).tail(1)
    summary = final.groupby("framework")[["gate_passes", "mean_ep_reward", "mean_ep_length"]].median()
    print(summary.to_string())

    print("\n=== per-seed final-window values ===")
    print(final[["framework", "seed", "timestep", "gate_passes",
                 "mean_ep_reward", "mean_ep_length"]].to_string(index=False))


if __name__ == "__main__":
    main()
