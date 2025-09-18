#!/usr/bin/env python3
"""
Shared test utilities for AirEvolve2 unit tests.

This module contains common helper functions and utilities used across
multiple test files to avoid code duplication.
"""

import numpy as np
from typing import List


def extract_positions(population) -> np.ndarray:
    """Extract all motor positions from a population."""
    all_positions = []
    for individual in population:
        positions = individual.get_motor_positions()
        all_positions.extend(positions)
    return np.array(all_positions)


def extract_orientations(population) -> np.ndarray:
    """Extract all motor orientations from a population."""
    all_orientations = []
    for individual in population:
        orientations = individual.get_motor_orientations()
        all_orientations.extend(orientations)
    return np.array(all_orientations)


def extract_directions(population) -> np.ndarray:
    """Extract all propeller directions from a population."""
    all_directions = []
    for individual in population:
        directions = individual.get_propeller_directions()
        all_directions.extend(directions)
    return np.array(all_directions)