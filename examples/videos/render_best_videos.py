"""Render videos for representative individuals per (task, genotype).

For each (task, genotype) in the config, selects:
  - Top K: best-fitness individuals, one per run (distinct reps)
  - Middle K: median-performing, distinct reps
  - Bottom K: lowest-fitness individuals that still flew at least one gate,
    distinct reps

Only individuals whose hover_breakdown.json status == "full_evaluation" are
eligible — they're the ones with `tuning_results.json` (required by
examples/videos/make_lee_video.py).

Usage:
    # Dry run: print the plan, no rendering
    python examples/videos/render_best_videos.py \
        --config experimentation/experiment_config_neat.yaml --dry-run

    # Render one (top of first task/geno) to verify the pipeline
    python examples/videos/render_best_videos.py \
        --config experimentation/experiment_config_neat.yaml --test-one

    # Full batch
    python examples/videos/render_best_videos.py \
        --config experimentation/experiment_config_neat.yaml
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from experimentation.config import load_config

MAKE_VIDEO_SCRIPT = REPO_ROOT / "examples" / "videos" / "make_lee_video.py"

_REP_RE = re.compile(r"^rep(\d+)$")


def gather_valid_individuals(run_dir: Path):
    """Yield (fitness, gates_passed, ind_path) for individuals eligible for
    video rendering (tuning_results.json present, full_evaluation)."""
    for gen_dir in sorted(run_dir.glob("generation_*")):
        if not gen_dir.is_dir():
            continue
        for ind_dir in sorted(gen_dir.glob("individual_*")):
            if not (ind_dir / "tuning_results.json").exists():
                continue
            hb_path = ind_dir / "hover_breakdown.json"
            if not hb_path.exists():
                continue
            try:
                with open(hb_path) as f:
                    hb = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            if hb.get("status") != "full_evaluation":
                continue
            yield (
                float(hb.get("total_fitness", 0.0)),
                int(hb.get("gates_passed", 0)),
                ind_dir,
            )


def pick_representatives(valid_by_rep, k=3, min_gates_for_bottom=1):
    """Return (top, middle, bottom) lists of (rep_name, fitness, gates, path).

    Each list has up to k entries, all from distinct reps, non-overlapping.
    Bottom is filtered to individuals with gates_passed >= min_gates_for_bottom
    ("still flies").
    """
    rep_bests = {}
    for rep_name, inds in valid_by_rep.items():
        if not inds:
            continue
        fit, gates, path = max(inds, key=lambda x: x[0])
        rep_bests[rep_name] = (fit, gates, path)

    if not rep_bests:
        return [], [], []

    sorted_reps = sorted(
        rep_bests.items(), key=lambda kv: kv[1][0], reverse=True
    )

    top = [(r, *v) for r, v in sorted_reps[:k]]
    top_reps = {r for r, *_ in top}

    fliers = [(r, v) for r, v in sorted_reps if v[1] >= min_gates_for_bottom]
    fliers_for_bottom = [kv for kv in fliers if kv[0] not in top_reps]
    bottom = [(r, *v) for r, v in fliers_for_bottom[-k:][::-1]]
    bottom_reps = {r for r, *_ in bottom}

    remaining = [
        (r, v) for r, v in sorted_reps
        if r not in top_reps and r not in bottom_reps
    ]
    n = len(remaining)
    if n == 0:
        middle = []
    else:
        mid_start = max(0, (n - k) // 2)
        middle = [(r, *v) for r, v in remaining[mid_start:mid_start + k]]

    return top, middle, bottom


def run_make_video(ind_path: Path, task: str, dry_run=False,
                   action_smooth_ms: float = 50.0,
                   overlay_scale: float = 6.0,
                   iso_overlay_pos: str = "lower right",
                   draw_forces: bool = True):
    cmd = [
        sys.executable, str(MAKE_VIDEO_SCRIPT),
        str(ind_path), "--gate-cfg", task,
        "--action-smooth-ms", str(action_smooth_ms),
        "--overlay-scale", str(overlay_scale),
        "--iso-overlay-pos", iso_overlay_pos,
    ]
    if not draw_forces:
        cmd.append("--no-forces")
    print("    $ " + " ".join(cmd))
    if dry_run:
        return 0
    return subprocess.call(cmd)


def collate_video(ind_path: Path, collated_dir: Path, task: str, genotype: str,
                  label: str, rep: str, fitness: float, dry_run=False) -> Path | None:
    """Copy <ind_path>/videos/combined_output.mp4 to collated_dir with a
    descriptive name. Returns the destination path, or None if the source is
    missing."""
    src = ind_path / "videos" / "combined_output.mp4"
    dest_name = f"{task}_{genotype}_{label}_{rep}_fit{fitness:.2f}.mp4"
    dest = collated_dir / dest_name
    if dry_run:
        print(f"    -> would collate {dest}")
        return dest
    if not src.exists():
        print(f"    !! missing {src}, skipping collation")
        return None
    collated_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    print(f"    -> collated {dest}")
    return dest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True,
                        help="Path to experiment YAML config")
    parser.add_argument("--k", type=int, default=1,
                        help="Number of representatives per bucket (default 1)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the plan without rendering")
    parser.add_argument("--test-one", action="store_true",
                        help="Render only the very first TOP pick (sanity check)")
    parser.add_argument("--min-gates-for-bottom", type=int, default=1,
                        help="Min gates_passed to qualify as 'still flies' (default 1)")
    parser.add_argument("--collated-dir", default=str(REPO_ROOT / "tmp" / "videos_collated"),
                        help="Directory to collate combined_output.mp4 files into")
    parser.add_argument("--top-only", action="store_true",
                        help="Only process the TOP bucket (skip MID and BOT)")
    parser.add_argument("--collate-only", action="store_true",
                        help="Skip rendering; only copy existing combined_output.mp4 files")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip rendering when combined_output.mp4 already exists "
                             "(still collates it). Lets you resume an interrupted run.")
    args = parser.parse_args()

    collated_dir = Path(args.collated_dir)

    cfg = load_config(args.config)
    base = Path(cfg["base_dir"])
    tasks = cfg["tasks"]
    genotypes = cfg["genotypes"]

    print(f"Base: {base}")
    print(f"Tasks: {tasks}")
    print(f"Genotypes: {genotypes}")
    print(f"K per bucket: {args.k}")
    print(f"Collated dir: {collated_dir}")
    mode = ('DRY RUN' if args.dry_run else
            'COLLATE-ONLY' if args.collate_only else
            'TEST-ONE' if args.test_one else 'RENDER')
    print(f"Mode: {mode}{' (TOP only)' if args.top_only else ''}")
    print()

    rendered = 0
    for geno in genotypes:
        for task in tasks:
            tg_dir = base / task / geno
            if not tg_dir.exists():
                continue

            valid_by_rep = {}
            for rep_dir in sorted(tg_dir.iterdir()):
                if not _REP_RE.match(rep_dir.name):
                    continue
                inds = list(gather_valid_individuals(rep_dir))
                if inds:
                    valid_by_rep[rep_dir.name] = inds

            top, mid, bot = pick_representatives(
                valid_by_rep, k=args.k,
                min_gates_for_bottom=args.min_gates_for_bottom,
            )

            print(f"=== {task} / {geno} ===")
            print(f"    reps with valid individuals: {len(valid_by_rep)}/"
                  f"{sum(1 for _ in tg_dir.iterdir() if _REP_RE.match(_.name))}")

            buckets = (("TOP", top),) if args.top_only else \
                      (("TOP", top), ("MID", mid), ("BOT", bot))
            for label, group in buckets:
                if not group:
                    print(f"    [{label}] (none)")
                    continue
                for rep, fit, gates, path in group:
                    print(f"    [{label}] {rep} fitness={fit:7.2f} gates={gates} path={path}")
                    already_rendered = (path / "videos" / "combined_output.mp4").exists()
                    if args.collate_only or (args.skip_existing and already_rendered):
                        if args.skip_existing and already_rendered and not args.collate_only:
                            print("    (skip-existing) combined_output.mp4 present, skipping render")
                    else:
                        run_make_video(path, task, dry_run=args.dry_run)
                    collate_video(path, collated_dir, task, geno, label, rep,
                                  fit, dry_run=args.dry_run)
                    rendered += 1
                    if args.test_one:
                        print(f"\n--test-one: stopping after 1 render.")
                        print(f"Rendered: {rendered}")
                        return
            print()

    print(f"Total video commands {'dry-ran' if args.dry_run else 'invoked'}: {rendered}")


if __name__ == "__main__":
    main()
