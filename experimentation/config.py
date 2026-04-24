"""Shared experiment configuration — single source of truth for paths, tasks, genotypes, and colors."""

import os
from pathlib import Path

import numpy as np
import yaml


# ── Defaults ──────────────────────────────────────────────────────────────────

BASE_DIR = "/path/to/data/airevolve030326/v2"

TASKS = ["backandforth", "figure8", "circle", "slalom"]
TASK_LABELS = {
    "backandforth": "Shuttle Run",
    "figure8": "Figure 8",
    "circle": "Circle",
    "slalom": "Slalom",
}

GENOTYPES = ["spherical", "cppn", "hybrid_cppn"]
GENOTYPE_LABELS = {
    "spherical": "Direct",
    "cppn": "CPPN",
    "hybrid_cppn": "Hybrid",
    "neat_spherical": "Direct",
    "neat_cppn": "CPPN",
    "neat_hybrid_cppn": "Hybrid",
}
COLORS = {
    "spherical": "#4477AA",
    "cppn": "#EE6677",
    "hybrid_cppn": "#228833",
    "neat_spherical": "#4477AA",
    "neat_cppn": "#EE6677",
    "neat_hybrid_cppn": "#228833",
}

# Phenotype parameter limits: [min, max] per column
PARAMETER_LIMITS = np.array([
    [0.09, 0.4],       # magnitude
    [0, 2 * np.pi],    # arm_yaw
    [0, 2 * np.pi],    # arm_pitch
    [0, 2 * np.pi],    # motor_pitch
    [0, 2 * np.pi],    # motor_yaw
    [0, 1],            # direction
])

# Default style settings
DEFAULT_STYLE = {
    "axis_label_fontsize": 30,
    "tick_label_fontsize": 25,
    "legend_fontsize": 14,
    "annotation_fontsize": 12,
    "dpi": 300,
}

DEFAULT_Y_LIMITS = {}

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "experiment_config.yaml")

# Configurable subdirectory name under output/ for figures (set via load_config)
_figures_subdir = "figures"

# Configurable y-axis limits per plot (set via load_config)
Y_LIMITS = dict(DEFAULT_Y_LIMITS)


def set_figures_subdir(name):
    """Override the default figures subdirectory name (e.g. 'figures_NEAT')."""
    global _figures_subdir
    _figures_subdir = name


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_min_max():
    """Return (min_vals, max_vals) arrays from PARAMETER_LIMITS."""
    return PARAMETER_LIMITS[:, 0], PARAMETER_LIMITS[:, 1]


def output_dir(base_dir=None, task=None, subdirectory=None):
    """Build standardised output path: {base_dir}/output/[{task}/][{subdirectory}/].

    Creates directory if it doesn't exist. Returns the path.
    """
    base = base_dir or BASE_DIR
    parts = [base, "output"]
    if task:
        parts.append(task)
    if subdirectory:
        parts.append(subdirectory)
    path = os.path.join(*parts)
    os.makedirs(path, exist_ok=True)
    return path


def csv_dir(base_dir=None, task=None, genotype=None):
    """Build CSV output path: {base_dir}/output/csv/{task}/{genotype}/."""
    parts = ["csv"]
    if task:
        parts.append(task)
    if genotype:
        parts.append(genotype)
    return output_dir(base_dir, subdirectory=os.path.join(*parts))


def figures_dir(base_dir=None, task=None, plotter_name=None, figures_subdir=None):
    """Build figures output path: {base_dir}/output/{figures_subdir}/{task}/{plotter_name}/.

    *figures_subdir* defaults to the module-level ``_figures_subdir`` (set via
    ``load_config`` / ``set_figures_subdir``), which itself defaults to
    ``"figures"``.
    """
    parts = [figures_subdir or _figures_subdir]
    if task:
        parts.append(task)
    if plotter_name:
        parts.append(plotter_name)
    return output_dir(base_dir, subdirectory=os.path.join(*parts))


def load_config(path=None):
    """Load YAML experiment config, merging with defaults.

    Returns a dict with keys: base_dir, tasks, genotypes, n_workers,
    style, collectors, plotters.
    """
    defaults = {
        "base_dir": BASE_DIR,
        "tasks": list(TASKS),
        "genotypes": list(GENOTYPES),
        "n_workers": 24,
        "figures_subdir": "figures",
        "style": dict(DEFAULT_STYLE),
        "y_limits": dict(DEFAULT_Y_LIMITS),
        "collectors": [],
        "plotters": [],
    }

    config_path = path or DEFAULT_CONFIG_PATH
    if not os.path.exists(config_path):
        return defaults

    with open(config_path, "r") as f:
        user = yaml.safe_load(f) or {}

    # Merge top-level keys
    for key in ("base_dir", "tasks", "genotypes", "n_workers", "figures_subdir",
                 "collectors", "plotters"):
        if key in user:
            defaults[key] = user[key]

    # Merge style dict (partial overrides allowed)
    if "style" in user and isinstance(user["style"], dict):
        defaults["style"].update(user["style"])

    # Merge y_limits dict (partial overrides allowed)
    if "y_limits" in user and isinstance(user["y_limits"], dict):
        defaults["y_limits"].update(user["y_limits"])

    # Apply figures subdirectory globally so plotters pick it up
    set_figures_subdir(defaults["figures_subdir"])

    # Apply y-axis limits globally so plotters pick them up
    global Y_LIMITS
    Y_LIMITS.update(defaults["y_limits"])

    return defaults
