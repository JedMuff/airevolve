"""Bi-objective fitness wrapper for NSGA-II gate-racing evolution.

Returns a tuple (waypoints: int, total_energy_j: float) where:
  - waypoints      is the number of gates passed during the 12-second RL eval,
                   to be MAXIMISED.
  - total_energy_j is the Joules consumed during that same 12-second flight,
                   to be MINIMISED.

Non-hoverable / un-repairable morphologies return (0, FAIL_ENERGY) so they are
always dominated by any morphology that passes at least one gate.

Only the RL brain is supported (brain='rl').  Lee tuning does not expose the
battery model through the same interface.
"""

import os
import pickle

import numpy as np

from airevolve.evolution_tools.evaluators.hover_fitness import continuous_hover_fitness
from airevolve.evolution_tools.genome_handlers.repair_workflow import stage2_hover_check

# Large but finite sentinel — clearly beyond any realistic 12-second flight.
# Max possible energy: 18 A × 16.8 V × 12 s ≈ 3629 J.
# Using 1e9 J (1 GJ) ensures failed morphologies are dominated.
_FAIL_ENERGY: float = 1e9


class BiObjectiveFitness:
    """Picklable bi-objective fitness wrapper for NSGA-II.

    Designed to be used as the `fitness_function` argument to
    `evolve_nsga2()` from `strategies.nsga2_strategy`.

    Parameters
    ----------
    brain              : 'rl' (only RL is supported)
    hover_gradient     : if True, non-hoverable drones receive a gradient
                         (int(hover_score), FAIL_ENERGY) instead of (0, FAIL_ENERGY).
    per_individual_repair : apply 3-stage repair before every evaluation.
    is_indirect        : True for CPPN / hybrid-cppn genomes.
    handler_class      : genome handler class (for indirect decoding).
    handler_kwargs     : kwargs forwarded to handler_class().
    coordinate_system  : 'spherical' | 'cartesian' | 'cppn' | 'hybrid-cppn'.
    brain_kwargs       : forwarded to gate_train_power.evaluate_individual():
        gate_cfg         : str  — 'figure8', 'circle', etc.
        training_ts      : int  — PPO training timesteps.
        num_envs         : int  — parallel envs during training.
        device           : str  — torch device string.
        max_steps        : int  — evaluation length in env steps (default 1200 → 12 s).
        experiment_type  : int  — 0=baseline, 1=dense, 2=sparse (default 0).
        penalty_weights  : dict — e.g. {"dense_weight": 1e-5} or {"sparse_weight": 1e-3}.
    """

    def __init__(
        self,
        *,
        brain: str,
        hover_gradient: bool,
        per_individual_repair: bool,
        is_indirect: bool,
        handler_class,
        handler_kwargs: dict,
        coordinate_system: str,
        brain_kwargs: dict = None,
    ) -> None:
        if brain != "rl":
            raise ValueError(
                f"BiObjectiveFitness only supports brain='rl', got {brain!r}"
            )
        self.brain                = brain
        self.hover_gradient       = hover_gradient
        self.per_individual_repair = per_individual_repair
        self.is_indirect          = is_indirect
        self.handler_class        = handler_class
        self.handler_kwargs       = handler_kwargs
        self.coordinate_system    = coordinate_system
        self.brain_kwargs         = brain_kwargs or {}

    # ── Callable interface ────────────────────────────────────────────────────

    def __call__(self, genome, ind_save_dir) -> tuple:
        """Evaluate one genome; returns (waypoints: int, total_energy_j: float)."""

        # 1. Persist genome for reproducibility
        if ind_save_dir is not None:
            os.makedirs(ind_save_dir, exist_ok=True)
            if self.is_indirect:
                with open(os.path.join(ind_save_dir, "genotype.pkl"), "wb") as fh:
                    pickle.dump(genome, fh)
            else:
                np.save(os.path.join(ind_save_dir, "genome.npy"), genome)

        # 2. Decode to phenotype (arm-matrix in spherical coords)
        if self.is_indirect:
            handler   = self.handler_class(genome=genome, **self.handler_kwargs)
            phenotype = handler.get_phenotype()
            repair_coord = "spherical"
        else:
            arms      = genome.arms if hasattr(genome, "arms") else genome
            phenotype = np.asarray(arms, dtype=np.float64)
            repair_coord = self.coordinate_system

        # 3. Optional 3-stage repair
        if self.per_individual_repair:
            raw = phenotype
            phenotype = self._repair(phenotype, repair_coord)
            if phenotype is None:
                return self._fail_result(raw)

        # 4. Hover feasibility check (fast filter before expensive RL training)
        can_hover, _ = stage2_hover_check(phenotype, verbose=False, allow_spinning=False)
        if not can_hover:
            return self._fail_result(phenotype)

        # 5. RL training + 12-second evaluation
        from airevolve.evolution_tools.evaluators import gate_train_power
        return gate_train_power.evaluate_individual(
            phenotype,
            ind_save_dir,
            self.brain_kwargs["training_ts"],
            self.brain_kwargs["num_envs"],
            self.brain_kwargs["gate_cfg"],
            self.brain_kwargs["device"],
            max_steps=self.brain_kwargs.get("max_steps", 1200),
            experiment_type=self.brain_kwargs.get("experiment_type", 0),
            penalty_weights=self.brain_kwargs.get("penalty_weights", None),
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _fail_result(self, phenotype) -> tuple:
        if self.hover_gradient:
            hover_score = continuous_hover_fitness(phenotype)
            return (int(hover_score), _FAIL_ENERGY)
        return (0, _FAIL_ENERGY)

    def _repair(self, phenotype, repair_coord):
        """3-stage repair; returns None on failure."""
        from airevolve.evolution_tools.genome_handlers.repair_workflow import (
            stage1_optimization_repair,
            stage3_hover_repair,
        )
        from airevolve.evolution_tools.genome_handlers.operators.optimization_repair_operator import (
            OptimizationRepairConfig,
        )
        can_hover, _ = stage2_hover_check(phenotype, verbose=False, allow_spinning=False)
        if not can_hover:
            return None
        config = OptimizationRepairConfig(fixed_params=[3, 4])
        repaired, _ = stage1_optimization_repair(
            phenotype, coordinate_system=repair_coord,
            config=config, verbose=False,
        )
        if repaired is None:
            return None
        final, _ = stage3_hover_repair(repaired, coordinate_system=repair_coord, verbose=False)
        return final
