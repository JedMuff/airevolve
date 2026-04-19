"""Create symlinks mapping NEAT experiment output to the standard pipeline layout.

NEAT experiments are stored as flat directories:
    {source}/neat_combined_hg_{task}_{genotype}_rep{N}_{seed}_{idx}/
        neat_combined_hover_gate_runs/{timestamp_dir}/

The pipeline expects:
    {target}/{task}/{neat_genotype}/{run_name}/evolution_data.csv

This script creates symlinks to bridge the two layouts.
"""

import argparse
import os
import re
from collections import defaultdict

from experimentation.config import BASE_DIR


# Map raw genotype names from directory to pipeline genotype names
GENOTYPE_MAP = {
    "cppn": "neat_cppn",
    "hybrid-cppn": "neat_hybrid_cppn",
    "spherical": "neat_spherical",
}

# Regex to parse NEAT experiment directory names
DIR_PATTERN = re.compile(
    r"^neat_combined_hg_(?P<task>\w+?)_(?P<genotype>cppn|hybrid-cppn|spherical)"
    r"_rep(?P<rep>\d+)_(?P<seed>\d+)_(?P<idx>\d+)$"
)


def find_run_dir(rep_dir):
    """Find the actual run directory containing evolution_data.csv.

    Descends through the intermediate neat_combined_hover_gate_runs/ layer.
    """
    runs_parent = os.path.join(rep_dir, "neat_combined_hover_gate_runs")
    if not os.path.isdir(runs_parent):
        return None
    for entry in os.listdir(runs_parent):
        candidate = os.path.join(runs_parent, entry)
        if os.path.isdir(candidate) and os.path.exists(
            os.path.join(candidate, "evolution_data.csv")
        ):
            return candidate
    return None


def count_generations(run_dir):
    """Count generation directories in a run."""
    count = 0
    for entry in os.listdir(run_dir):
        if re.match(r"generation_\d+$", entry):
            count += 1
    return count


def discover_experiments(source_dir):
    """Scan source directory and group experiments by (task, genotype, rep).

    Returns dict: (task, genotype, rep) -> list of (dir_path, seed, run_dir).
    For duplicates, all candidates are returned for later selection.
    """
    experiments = defaultdict(list)

    for entry in sorted(os.listdir(source_dir)):
        m = DIR_PATTERN.match(entry)
        if not m:
            continue

        full_path = os.path.join(source_dir, entry)
        if not os.path.isdir(full_path):
            continue

        run_dir = find_run_dir(full_path)
        if run_dir is None:
            print(f"  SKIP {entry}: no evolution_data.csv found")
            continue

        task = m.group("task")
        genotype = m.group("genotype")
        rep = int(m.group("rep"))
        seed = int(m.group("seed"))

        key = (task, genotype, rep)
        experiments[key].append((full_path, seed, run_dir))

    return experiments


def create_symlinks(experiments, target_dir, dry_run=False):
    """Create symlinks from target layout to actual run directories.

    When multiple candidates share the same (task, genotype, rep) key
    (e.g. two batches that each numbered reps 0-9), all candidates are
    kept: the "best" candidate (most generations, highest seed as
    tiebreaker) keeps the original rep number, and extras are offset
    by ``stride = max_original_rep + 1`` so a second batch of rep0-9
    becomes rep10-19.
    """
    max_rep = max((rep for (_, _, rep) in experiments.keys()), default=-1)
    stride = max_rep + 1

    created = 0
    skipped = 0

    for (task, genotype, rep), candidates in sorted(experiments.items()):
        mapped_genotype = GENOTYPE_MAP[genotype]

        ranked = sorted(
            candidates,
            key=lambda c: (count_generations(c[2]), c[1]),
            reverse=True,
        )

        if len(ranked) > 1:
            seeds = [c[1] for c in ranked]
            rep_nums = [rep + i * stride for i in range(len(ranked))]
            print(f"  MULTI {task}/{genotype}/rep{rep}: "
                  f"{len(ranked)} candidates, assigning reps "
                  f"{rep_nums} for seeds {seeds}")

        for i, (_, _, run_dir) in enumerate(ranked):
            rep_num = rep + i * stride
            link_dir = os.path.join(target_dir, task, mapped_genotype)
            link_path = os.path.join(link_dir, f"rep{rep_num}")

            if os.path.lexists(link_path):
                if os.path.islink(link_path) and os.readlink(link_path) == run_dir:
                    skipped += 1
                    continue
                elif os.path.islink(link_path):
                    print(f"  UPDATE {link_path} -> {run_dir}")
                    if not dry_run:
                        os.unlink(link_path)
                else:
                    print(f"  WARN {link_path} exists and is not a symlink, skipping")
                    skipped += 1
                    continue

            if dry_run:
                print(f"  DRY RUN: {link_path} -> {run_dir}")
            else:
                os.makedirs(link_dir, exist_ok=True)
                os.symlink(run_dir, link_path)
                print(f"  LINK {link_path} -> {run_dir}")
            created += 1

    return created, skipped


def main():
    parser = argparse.ArgumentParser(description="Reorganize NEAT data into pipeline layout")
    parser.add_argument("--source", default=os.path.join(os.path.dirname(__file__), "..", "tmp"),
                        help="Source directory containing NEAT experiment dirs")
    parser.add_argument("--target", default=BASE_DIR,
                        help="Target base directory for symlinks")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be done without creating symlinks")
    args = parser.parse_args()

    source = os.path.abspath(args.source)
    target = os.path.abspath(args.target)

    print(f"Source: {source}")
    print(f"Target: {target}")
    print()

    experiments = discover_experiments(source)
    print(f"\nFound {len(experiments)} unique (task, genotype, rep) combinations")

    created, skipped = create_symlinks(experiments, target, dry_run=args.dry_run)
    print(f"\nDone: {created} created, {skipped} skipped")


if __name__ == "__main__":
    main()
