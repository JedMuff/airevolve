"""
Sort experiment data from tmp/airevolve_data_180226/ to organized structure
on the external drive /media/jed/My Passport/airevolve030326/.

Dry-run by default. Use --execute to actually copy files.
"""

import argparse
import os
import re
import shutil


# Source directory
SRC_DIR = os.path.expanduser("~/workspaces/airevolve/tmp/combined_hg/")
# Target directory on external drive
DST_DIR = "/media/jed/My Passport/airevolve030326"


def parse_dir_name(name):
    """Parse a directory name to determine genome type and target subdirectory.

    Returns (target_subdir, description) or (None, reason) if unrecognized.
    """
    # lee_tuning_spherical_rep{N}_{job}_{task}
    m = re.match(r"lee_tuning_spherical_rep(\d+)_\d+_\d+$", name)
    if m:
        return f"spherical/rep{m.group(1)}", f"spherical rep{m.group(1)}"

    # lee_tuning_cppn_rep{N}_{job}_{task}
    m = re.match(r"lee_tuning_cppn_rep(\d+)_\d+_\d+$", name)
    if m:
        return f"cppn/rep{m.group(1)}", f"cppn rep{m.group(1)}"

    # lee_tuning_hybrid-cppn_rep{N}_{job}_{task}
    m = re.match(r"lee_tuning_hybrid-cppn_rep(\d+)_\d+_\d+$", name)
    if m:
        return f"hybrid_cppn/rep{m.group(1)}", f"hybrid-cppn rep{m.group(1)}"

    # lee_tuning_{job}_{task}  (no genome label → spherical, use run_{job}_{task})
    m = re.match(r"lee_tuning_(\d+)_(\d+)$", name)
    if m:
        subdir = f"spherical/run_{m.group(1)}_{m.group(2)}"
        return subdir, f"spherical run_{m.group(1)}_{m.group(2)}"

    # combined_hg_{task}_{genotype}_rep{N}_{jobid}_{arrayid}
    m = re.match(r"combined_hg_(\w+?)_(spherical|cppn|hybrid-cppn)_rep(\d+)_\d+_\d+$", name)
    if m:
        task = m.group(1)
        genotype = m.group(2).replace("-", "_")
        rep = m.group(3)
        # Return group dir (without rep) so main() can assign the next available rep
        return f"v2/{task}/{genotype}", f"v2 {task}/{genotype} rep{rep}"

    return None, f"unrecognized pattern: {name}"


def next_available_rep(group_dir, assigned_reps):
    """Return the next available rep number for a group directory.

    Scans group_dir for existing rep{N} subdirectories, also considers
    reps already assigned in this run (tracked in assigned_reps dict),
    and returns the next free number.
    """
    existing = set(assigned_reps.get(group_dir, []))
    if os.path.isdir(group_dir):
        for d in os.listdir(group_dir):
            m = re.match(r"rep(\d+)$", d)
            if m:
                existing.add(int(m.group(1)))
    next_rep = 0
    while next_rep in existing:
        next_rep += 1
    assigned_reps.setdefault(group_dir, []).append(next_rep)
    return next_rep


def find_inner_run(src_path):
    """Unwrap the lee_tuning_runs/ or combined_hover_gate_runs/ nesting.

    Returns the path to the innermost run directory, or None.
    """
    for runs_subdir in ("lee_tuning_runs", "combined_hover_gate_runs"):
        runs_dir = os.path.join(src_path, runs_subdir)
        if not os.path.isdir(runs_dir):
            continue

        subdirs = [d for d in os.listdir(runs_dir)
                   if os.path.isdir(os.path.join(runs_dir, d))]
        if len(subdirs) == 1:
            return os.path.join(runs_dir, subdirs[0])
        elif len(subdirs) > 1:
            subdirs.sort()
            return os.path.join(runs_dir, subdirs[0])
    return None


def main():
    parser = argparse.ArgumentParser(description="Sort experiment data to external drive")
    parser.add_argument("--execute", action="store_true",
                        help="Actually copy files (default is dry-run)")
    parser.add_argument("--src", default=SRC_DIR, help="Source directory")
    parser.add_argument("--dst", default=DST_DIR, help="Destination directory")
    args = parser.parse_args()

    if not os.path.isdir(args.src):
        print(f"Source directory not found: {args.src}")
        return

    if not os.path.isdir(args.dst):
        print(f"Destination directory not found: {args.dst}")
        return

    mode = "EXECUTE" if args.execute else "DRY-RUN"
    print(f"Mode: {mode}")
    print(f"Source: {args.src}")
    print(f"Destination: {args.dst}")
    print()

    entries = sorted(os.listdir(args.src))
    copied = 0
    skipped = 0
    errors = 0
    assigned_reps = {}  # track reps assigned in this run to avoid collisions

    for entry in entries:
        src_path = os.path.join(args.src, entry)
        if not os.path.isdir(src_path):
            continue

        target_subdir, description = parse_dir_name(entry)
        if target_subdir is None:
            print(f"  SKIP {entry}: {description}")
            skipped += 1
            continue

        dst_path = os.path.join(args.dst, target_subdir)

        # For v2 combined_hg entries, assign the next available rep number
        if target_subdir.startswith("v2/"):
            next_rep = next_available_rep(dst_path, assigned_reps)
            target_subdir = f"{target_subdir}/rep{next_rep}"
            dst_path = os.path.join(args.dst, target_subdir)

        # Check if target already has data
        if os.path.exists(os.path.join(dst_path, "evolution_data.csv")):
            print(f"  EXISTS {entry} -> {target_subdir}")
            skipped += 1
            continue

        # Find the inner run directory
        inner_run = find_inner_run(src_path)
        if inner_run is None:
            print(f"  ERROR {entry}: no lee_tuning_runs/ or combined_hover_gate_runs/ subdirectory found")
            errors += 1
            continue

        print(f"  COPY {entry} -> {target_subdir}")
        print(f"       from: {inner_run}")

        if args.execute:
            os.makedirs(dst_path, exist_ok=True)
            shutil.copytree(inner_run, dst_path, dirs_exist_ok=True)
            copied += 1
        else:
            copied += 1

    print()
    print(f"Summary: {copied} to copy, {skipped} skipped, {errors} errors")
    if not args.execute:
        print("(dry-run — use --execute to actually copy)")


if __name__ == "__main__":
    main()
