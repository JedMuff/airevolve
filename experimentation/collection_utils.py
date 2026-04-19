"""Shared utilities for experiment data collection scripts."""

import json
import os
import pickle
import re
import sys
import types

import numpy as np
import pandas as pd

from experimentation.config import PARAMETER_LIMITS, get_min_max  # noqa: F401 — re-export for backwards compat


def iter_experiments(base_dir, tasks, genotypes):
    """Yield (task, genotype, experiment_dir) for each valid experiment."""
    for task in tasks:
        for genotype in genotypes:
            experiment_dir = os.path.join(base_dir, task, genotype)
            if os.path.isdir(experiment_dir):
                yield task, genotype, experiment_dir


def iter_runs(experiment_dir):
    """Yield (run_name, run_path) for dirs containing evolution_data.csv."""
    if not os.path.isdir(experiment_dir):
        return
    for entry in sorted(os.listdir(experiment_dir)):
        run_path = os.path.join(experiment_dir, entry)
        if not os.path.isdir(run_path):
            continue
        if os.path.exists(os.path.join(run_path, "evolution_data.csv")):
            yield entry, run_path


def iter_generations(run_path):
    """Yield (gen_idx, gen_path) sorted by generation index."""
    gens = []
    for entry in os.listdir(run_path):
        m = re.match(r"generation_(\d+)$", entry)
        if m:
            gen_path = os.path.join(run_path, entry)
            if os.path.isdir(gen_path):
                gens.append((int(m.group(1)), gen_path))
    gens.sort(key=lambda x: x[0])
    yield from gens


def iter_individuals(gen_path):
    """Yield (ind_id, ind_path) sorted by individual ID."""
    inds = []
    for entry in os.listdir(gen_path):
        if entry.startswith("individual_"):
            ind_path = os.path.join(gen_path, entry)
            if os.path.isdir(ind_path):
                ind_id = entry.replace("individual_", "")
                inds.append((ind_id, ind_path))
    inds.sort(key=lambda x: x[0])
    yield from inds


def load_phenotype(ind_path):
    """Load phenotype.npy (fallback genome.npy) from individual directory."""
    for fname in ("phenotype.npy", "genome.npy"):
        fpath = os.path.join(ind_path, fname)
        if os.path.exists(fpath):
            try:
                arr = np.load(fpath)
                if arr.ndim == 2 and not np.isnan(arr).all():
                    return arr
            except Exception:
                pass
    return None


def load_genotype(ind_path):
    """Load genotype.pkl via pickle. Returns object or None."""
    fpath = os.path.join(ind_path, "genotype.pkl")
    if not os.path.exists(fpath):
        return None
    try:
        with open(fpath, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def load_hover_breakdown(ind_path):
    """Load hover_breakdown.json. Returns dict or None."""
    fpath = os.path.join(ind_path, "hover_breakdown.json")
    if not os.path.exists(fpath):
        return None
    try:
        with open(fpath, "r") as f:
            return json.load(f)
    except Exception:
        return None


def load_learning_curve(ind_path):
    """Load learning_curve.json. Returns dict or None."""
    fpath = os.path.join(ind_path, "learning_curve.json")
    if not os.path.exists(fpath):
        return None
    try:
        with open(fpath, "r") as f:
            return json.load(f)
    except Exception:
        return None


def load_evolution_csv(run_path):
    """Load evolution_data.csv as DataFrame."""
    fpath = os.path.join(run_path, "evolution_data.csv")
    if os.path.exists(fpath):
        return pd.read_csv(fpath, dtype={"id": str})
    return None


def build_fitness_lookup(run_path):
    """Build a (generation, individual_id) -> fitness dict from evolution_data.csv."""
    lookup = {}
    evo_df = load_evolution_csv(run_path)
    if evo_df is not None:
        for _, row in evo_df.iterrows():
            lookup[(int(row["generation"]), str(row["id"]))] = float(row["fitness"])
    return lookup


# ── Deduplicated helpers ─────────────────────────────────────────────────────

def convert_str_to_nparray(s):
    """Convert string representation of numpy array back to numpy array.

    Handles numpy's default str() output including nan, spacing, etc.
    """
    s = s.replace("nan", "np.nan")
    s = s.replace("0. ", "0.0")
    s = re.sub(r"(-?\d|np\.nan)\s+(?=-?\d|np\.nan)", r"\1, ", s)
    s = re.sub(r"(\d\.)\s+(?=\d)", r"\1, ", s)
    s = re.sub(r"\]\s+\[", "], [", s)
    s = s.replace(". ", " ")
    s = s.replace(".,", ",")
    arr = np.array(eval(s))
    return arr


def lazy_import_cppn():
    """Lazy import CPPN network + topological_sort to avoid fcl dependency at module level.

    Returns (CPPNNetwork, topological_sort).
    """
    if "fcl" not in sys.modules:
        sys.modules["fcl"] = types.ModuleType("fcl")
    from airevolve.evolution_tools.genome_handlers.cppn.network import CPPNNetwork
    from airevolve.evolution_tools.genome_handlers.cppn.evaluation import topological_sort
    return CPPNNetwork, topological_sort


def lazy_import_cppn_compat():
    """Lazy import cppn_compatibility_distance to avoid fcl dependency at module level.

    Returns cppn_compatibility_distance function.
    """
    if "fcl" not in sys.modules:
        sys.modules["fcl"] = types.ModuleType("fcl")
    from airevolve.evolution_tools.genome_handlers.cppn.compatibility import cppn_compatibility_distance
    return cppn_compatibility_distance
