# Evolution Examples Unification Plan

## Goal

Replace the three near-duplicate evolution example scripts with one unified
`examples/evolution/run_evolution.py`, and port the `continuous_hover_fitness`
gradient signal from the `ppsn_2026_submission` branch
(`experimentation/run_combined_hover_gate_evolution.py`) so that non-hoverable
drones can receive a [0, 3] gradient signal instead of a hard zero.

Today's scripts (all in `examples/evolution/`):
- `run_evolution.py` — RL brain (`gate_train`), spherical/cartesian, naive `static_success` init-pop filter.
- `run_evolution_with_optimization_repair.py` — RL brain, single-process 3-stage repair init-pop, no per-individual repair.
- `run_evolution_with_lee_tuning.py` — Lee CMA-ES brain, all four genomes, parallel repair+CMA-ES init-pop, per-individual repair.

After this refactor only the unified script remains.

---

## CLI surface

```
python examples/evolution/run_evolution.py \
  --brain {rl, lee} \
  --genome {spherical, cartesian, cppn, hybrid-cppn} \
  --fitness {gate, pure_hover, edit_distance, zero} \
  --init-pop-mode {random, hover_repair} \
  --per-individual-repair \
  --hover-gradient \
  ... (existing per-brain knobs: --max-evals, --training-timesteps, etc.)
```

### Axis semantics

| Flag | Effect |
|---|---|
| `--brain` | Picks the gate evaluator: `rl` → `gate_train.evaluate_individual`, `lee` → `lee_tune_evaluator.evaluate_individual_with_tuning`. Ignored (with warning) when `--fitness` is not `gate`. |
| `--genome` | Picks the genome handler. CPPN/hybrid-cppn require a decode shim before the brain runs (see Decode Shim section). |
| `--fitness` | Picks the fitness function. Only `gate` consults `--brain` and `--hover-gradient`. |
| `--init-pop-mode random` | Plain `genome_handler.random_population(pop_size)`, no filtering. |
| `--init-pop-mode hover_repair` | Parallel sampling using `_try_generate_individual` / `_try_generate_cppn_individual` from `examples/evolution/run_evolution_with_lee_tuning.py:156` and `:222` (hover check → optimization repair → hover repair). **No CMA-ES tuning stage** — Phase 2 of `generate_initial_pop_parallel` is dropped. |
| `--per-individual-repair` | Apply 3-stage repair to every individual the EA produces before fitness evaluation. Independent of init-pop mode. |
| `--hover-gradient` | Only meaningful with `--fitness gate`. Adds the ppsn short-circuit logic (see below). |

### `--hover-gradient` semantics (option a, ppsn behavior)

For each individual:
1. Decode (if indirect) and apply repair (if `--per-individual-repair`).
2. Compute `hover_fit = continuous_hover_fitness(phenotype)` ∈ [0, 3].
3. Run `stage2_hover_check(phenotype, allow_spinning=False)`.
4. If hover check **fails** → return `hover_fit` and **skip the brain entirely** (no PPO training, no CMA-ES tuning).
5. If hover check **passes** → run the brain to get `gates_passed`, return `hover_fit + gates_passed`.

Hoverable drones always score > 3, non-hoverable drones score in [0, 3]. This
matches `_CombinedHoverGateFitness.__call__` in
`experimentation/run_combined_hover_gate_evolution.py:380-442` (ppsn branch).

### Brain × fitness compatibility matrix

| Fitness | Uses brain? | `--hover-gradient` honored? |
|---|---|---|
| `gate` | yes (`--brain` decides) | yes |
| `pure_hover` | no | n/a |
| `edit_distance` | no | n/a |
| `zero` | no | n/a |

If `--brain` or `--hover-gradient` is set with a non-`gate` fitness, log a
warning and ignore.

### Other CLI flags to keep

From the existing scripts, keep:
- `--population-size` (default 16), `--generations` (default 50)
- `--num-mutate` (default = population_size), `--num-crossover` (default 0)
- `--strategy-type {plus, comma}` (default `plus`)
- `--gate-cfg {figure8, circle, slalom, backandforth}` (default `figure8`) — only consulted when `--fitness gate`
- `--log-dir` (default `./.data`)
- `--show-plot`, `--save-all-plots` (post-run plotting)
- `--num-workers` (parallelism for `evaluate_population`; **default 1 if `--brain rl`**, else 32 — see Risks)
- `--min-narms` (default 6), `--max-narms` (default 6)
- CPPN-specific: `--num-segments` (default 8), `--initial-hidden-nodes` (default 0)
- Lee-specific (only when `--brain lee`): `--max-evals` (500), `--cma-workers` (1), `--sim-time` (20.0), `--dt` (0.005), `--timeout` (30.0)
- RL-specific (only when `--brain rl`): `--training-timesteps` (1e6), `--num-envs` (2), `--device` (`cuda:0`)

### CLI flags to **drop**

- `--symmetry` — only ever worked on `run_evolution.py`'s spherical/cartesian path; absent from the lee-tuning script and not requested for the unified version. Drop.
- `--repair-verbose` — replaced by global `--verbose` if needed.
- `--init-pop-max-evals`, `--init-pop-gates-threshold`, `--init-pop-tuning-workers`, `--skip-init-tuning` — these all pertain to Phase 2 CMA-ES tuning of the init-pop, which is dropped by user decision.

### Experiment dir naming

```
{log_dir}/{fitness}_{brain}_{genome}_{init_pop_mode}{_repair}{_hg}_{gate_cfg}_{timestamp}
```

E.g. `./.data/gate_lee_cppn_hover_repair_hg_figure8_20260429_143012`. Suffix
flags: `_repair` if `--per-individual-repair`, `_hg` if `--hover-gradient`.
This makes log directories self-describing and prevents collisions with old
runs.

---

## Reference locations (cite these when implementing)

### Source of `continuous_hover_fitness`
- File: `experimentation/run_combined_hover_gate_evolution.py` (branch `ppsn_2026_submission` only)
- Function body: lines 283–325
- Constants: line 279 (`G = 9.81`), line 280 (`C_AUTHORITY = 300.0`)
- To read it: `git show ppsn_2026_submission:experimentation/run_combined_hover_gate_evolution.py | sed -n '270,330p'`

### Source of `_CombinedHoverGateFitness` (the wrapper template)
- File: `experimentation/run_combined_hover_gate_evolution.py` (ppsn branch)
- Class: lines 333–442

### Source of init-pop helpers (parallel sampling + repair)
- File: `examples/evolution/run_evolution_with_lee_tuning.py` (already on `main`)
- `_try_generate_individual` — lines 156–219 (direct encoding)
- `_try_generate_cppn_individual` — lines 222–286 (CPPN/hybrid-cppn)
- `generate_initial_pop_parallel` — lines 414–687 (drop the Phase 2 CMA-ES block, lines 561–638)

### Source of `_RepairAndEvaluateFitness` (per-individual repair template)
- File: `examples/evolution/run_evolution_with_lee_tuning.py`
- Class: lines 768 onward (read to end of class for full behavior)

### Source of genome handler config dispatch
- File: `examples/evolution/run_evolution_with_lee_tuning.py`
- Function: `get_genome_handler_config` at line 690 (already covers all 4 encodings)

### Repair workflow imports
```python
from airevolve.evolution_tools.genome_handlers.repair_workflow import (
    stage1_optimization_repair,  # at airevolve/.../repair_workflow.py:276
    stage2_hover_check,           # at airevolve/.../repair_workflow.py:332
    stage3_hover_repair,          # at airevolve/.../repair_workflow.py:416
)
from airevolve.evolution_tools.genome_handlers.operators.optimization_repair_operator import (
    OptimizationRepairConfig,
)
```

`stage2_hover_check(individual, verbose=False, allow_spinning=False)` returns
`(can_hover: bool, message: str)`. Use `allow_spinning=False` everywhere to
match the ppsn semantics.

### Hover-info simulator
```python
from airevolve.evolution_tools.inspection_tools.morphological_descriptors.hovering_info import get_sim
# sim = get_sim(individual); sim has .Bf (3, n_props), .Bm (3, n_props)
```

`continuous_hover_fitness` reads `sim.Bf` and `sim.Bm`.

---

## Interfaces

### `evolve()` contract — `airevolve/evolution_tools/strategies/mu_lambda.py:13`

```python
def evolve(
    fitness_function: Callable[[np.ndarray, str], float],   # (genome, log_dir) -> fitness
    population_size: int,
    num_generations: int,
    num_mutate: int,
    num_crossover: int,
    mutate_after_crossover: bool,
    strategy_type: str,                # 'plus' | 'comma'
    parent_selection: Callable,
    initial_population: Optional[List[np.ndarray]] = None,
    reevaluate_old: bool = False,      # see "Elite re-evaluation" note
    log_dir: str = "./logs",
    genome_handler: Optional[object] = GenomeHandler,
    verbose: bool = True,
    num_workers: int = 1,
)
```

The fitness function the unified script passes to `evolve()` must have
signature `(genome, log_dir) -> float`. The `UnifiedFitness.__call__`
satisfies this.

### Brain entry points

```python
# airevolve/evolution_tools/evaluators/gate_train.py
def evaluate_individual(individual, ind_save_dir, training_ts, num_envs,
                        gate_cfg, device="cuda:0", num=None) -> int:
    ...  # returns gates_passed; 0 if static hover check fails

# airevolve/evolution_tools/evaluators/lee_tune_evaluator.py:934
def evaluate_individual_with_tuning(individual, ind_save_dir, gate_cfg='circle',
                                    max_evals=100, num_workers=4, sim_time=20.0,
                                    dt=0.005, timeout=30.0, num=None) -> int:
    ...  # returns gates_passed; 0 if tuning failed
```

Both return an int (gates passed). Both expect `individual` as a numpy array
shaped `(n_arms, 6)` (the spherical-encoded phenotype).

**Note**: both brains *internally* re-do their own hover check and return 0
if it fails. With `--hover-gradient`, the wrapper short-circuits *before*
calling the brain so this internal check never runs on non-hoverable drones —
that's fine, just be aware they're not idempotent if you skip the wrapper.

### Other fitness signatures (don't match `(genome, log_dir)` as-is — wrapper normalizes)

```python
# airevolve/evolution_tools/evaluators/edit_distance.py
def evaluate_individual(individual: GenomeHandler, log_dir, target,
                        min_vals, max_vals) -> float:
    return -compute_edit_distance(target, individual.body, ...)

# airevolve/evolution_tools/evaluators/zero.py
def fitness_function(member: GenomeHandler, log_dir=None) -> float:
    return 0
```

Note `edit_distance.evaluate_individual` takes a `GenomeHandler` (not raw
genome) and needs a `target`/`min_vals`/`max_vals`. The wrapper bridges this.
**See "Open decisions"** for what target to use.

`zero.fitness_function` also takes a `GenomeHandler` — wrapper just returns 0.

### `_run_brain` dispatch the wrapper needs

```python
def _run_brain(self, phenotype, ind_save_dir):
    if self.brain == 'rl':
        return gate_train.evaluate_individual(
            phenotype, ind_save_dir,
            self.brain_kwargs['training_ts'],
            self.brain_kwargs['num_envs'],
            self.brain_kwargs['gate_cfg'],
            self.brain_kwargs['device'],
        )
    elif self.brain == 'lee':
        return lee_tune_evaluator.evaluate_individual_with_tuning(
            phenotype, ind_save_dir,
            gate_cfg=self.brain_kwargs['gate_cfg'],
            max_evals=self.brain_kwargs['max_evals'],
            num_workers=self.brain_kwargs['cma_workers'],
            sim_time=self.brain_kwargs['sim_time'],
            dt=self.brain_kwargs['dt'],
            timeout=self.brain_kwargs['timeout'],
        )
```

---

## New code

### `airevolve/evolution_tools/evaluators/hover_fitness.py` (new)

```python
"""Continuous hover fitness — gradient signal in [0, 3] for non-hoverable drones.

Ported from experimentation/run_combined_hover_gate_evolution.py
(ppsn_2026_submission branch, lines 279-325).
"""
import numpy as np
from numpy.linalg import norm, eig

G = 9.81
C_AUTHORITY = 300.0  # sqrt(lambda_min) half-saturation constant


def continuous_hover_fitness(phenotype) -> float:
    """Continuous hover fitness in [0, 3]. Higher = closer to hoverable.

    Sum of three [0, 1] terms:
      - rank feasibility:  (rank(Bf) + rank(Bm)) / 6
      - force capability:  min(||Bf @ 1|| / G, 2) / 2
      - torque balance:    (lambda_min / lambda_max) * sqrt(lambda_min) / (sqrt(lambda_min) + C)
    """
    from airevolve.evolution_tools.inspection_tools.morphological_descriptors.hovering_info import get_sim

    sim = get_sim(phenotype)
    if sim is None:
        return 0.0

    Bf = sim.Bf  # (3, n_props)
    Bm = sim.Bm  # (3, n_props)

    rank_f = np.linalg.matrix_rank(Bf)
    rank_m = np.linalg.matrix_rank(Bm)
    f_rank = (rank_f + rank_m) / 6.0

    n_props = Bf.shape[1]
    eta_max = np.ones(n_props)
    f_vec = Bf @ eta_max
    thrust_ratio = norm(f_vec) / G
    f_force = min(thrust_ratio, 2.0) / 2.0

    gram_m = Bm @ Bm.T
    eigs = np.real(eig(gram_m)[0])
    eigs = np.maximum(eigs, 0.0)

    lambda_min = np.min(eigs)
    lambda_max = np.max(eigs)

    condition = lambda_min / (lambda_max + 1e-12)
    authority = np.sqrt(lambda_min)
    authority_sat = authority / (authority + C_AUTHORITY)
    f_torque = condition * authority_sat

    return float(f_rank + f_force + f_torque)
```

### `airevolve/evolution_tools/evaluators/unified_fitness.py` (new)

Picklable wrapper class. Owns: indirect-encoding decode, optional repair,
fitness-mode dispatch, hover-gradient short-circuit. Module-level (not nested)
so multiprocessing can pickle it.

```python
class UnifiedFitness:
    def __init__(self, *, brain, fitness_mode, hover_gradient,
                 per_individual_repair, is_indirect, handler_class,
                 handler_kwargs, brain_kwargs, coordinate_system,
                 edit_distance_target=None,
                 edit_distance_min_vals=None,
                 edit_distance_max_vals=None):
        self.brain = brain                                # 'rl' | 'lee' | None
        self.fitness_mode = fitness_mode                  # 'gate' | 'pure_hover' | 'edit_distance' | 'zero'
        self.hover_gradient = hover_gradient              # bool
        self.per_individual_repair = per_individual_repair # bool
        self.is_indirect = is_indirect                    # cppn / hybrid-cppn
        self.handler_class = handler_class                # for re-decoding indirect genomes
        self.handler_kwargs = handler_kwargs
        self.brain_kwargs = brain_kwargs
        self.coordinate_system = coordinate_system        # 'spherical' | 'cartesian'
        self.edit_distance_target = edit_distance_target
        self.edit_distance_min_vals = edit_distance_min_vals
        self.edit_distance_max_vals = edit_distance_max_vals

    def __call__(self, genome, ind_save_dir):
        # 1. Save genome (npy or pkl based on encoding).
        if ind_save_dir is not None:
            os.makedirs(ind_save_dir, exist_ok=True)
            if self.is_indirect:
                with open(os.path.join(ind_save_dir, 'genotype.pkl'), 'wb') as f:
                    pickle.dump(genome, f)
            else:
                np.save(os.path.join(ind_save_dir, 'genome.npy'), genome)

        # 2. Decode indirect → phenotype.
        if self.is_indirect:
            handler = self.handler_class(genome=genome, **self.handler_kwargs)
            phenotype = handler.get_phenotype()
            repair_coord = 'spherical'  # CPPN decodes to spherical
        else:
            phenotype = genome
            repair_coord = self.coordinate_system

        # 3. Per-individual repair (3-stage). On failure return 0 immediately.
        if self.per_individual_repair:
            phenotype = self._repair(phenotype, repair_coord)
            if phenotype is None:
                return 0.0

        # 4. Dispatch by fitness_mode.
        if self.fitness_mode == 'zero':
            return 0.0
        if self.fitness_mode == 'pure_hover':
            return continuous_hover_fitness(phenotype)
        if self.fitness_mode == 'edit_distance':
            from airevolve.evolution_tools.evaluators.edit_distance import compute_edit_distance
            return -compute_edit_distance(
                self.edit_distance_target, phenotype,
                self.edit_distance_min_vals, self.edit_distance_max_vals,
            )
        if self.fitness_mode == 'gate':
            if self.hover_gradient:
                hover_fit = continuous_hover_fitness(phenotype)
                can_hover, _ = stage2_hover_check(phenotype, allow_spinning=False)
                if not can_hover:
                    return hover_fit
                return hover_fit + self._run_brain(phenotype, ind_save_dir)
            return self._run_brain(phenotype, ind_save_dir)
        raise ValueError(f"Unknown fitness_mode: {self.fitness_mode}")

    def _repair(self, phenotype, repair_coord):
        # 3-stage: hover_check → stage1_opt_repair → stage3_hover_repair
        # Returns None if any stage fails.
        ...

    def _run_brain(self, phenotype, ind_save_dir):
        ...  # see Interfaces section
```

This wrapper subsumes `_RepairAndEvaluateFitness` (`run_evolution_with_lee_tuning.py:768`)
and `_CombinedHoverGateFitness` (`experimentation/run_combined_hover_gate_evolution.py:333`).
Delete both after the wrapper lands.

### `examples/evolution/run_evolution.py` — REPLACE

Build it from these existing pieces:
- `get_genome_handler_config` — copy from `run_evolution_with_lee_tuning.py:690` (covers all 4 encodings).
- Init-pop generation:
  - `--init-pop-mode random` → `WrappedHandler().random_population(pop_size)`.
  - `--init-pop-mode hover_repair` → call into a stripped version of `generate_initial_pop_parallel` that retains only Phase 1.
- Fitness function: `UnifiedFitness(...)` instance.
- `evolve()` invocation, CSV save, fitness plot, optional diversity/summary plots — copy the tail of `run_evolution_with_lee_tuning.py`.

### `examples/evolution/run_evolution_with_optimization_repair.py` — DELETE
### `examples/evolution/run_evolution_with_lee_tuning.py` — DELETE

Lift `_try_generate_individual`, `_try_generate_cppn_individual`, and the
trimmed `generate_initial_pop_parallel` into either the new `run_evolution.py`
or a helper module under `airevolve/evolution_tools/` first.

### `examples/evolution/optimization_repair_demo.py` — KEEP

Standalone demo of the repair operator, not an evolution runner. Untouched.

### Genome handlers

Always pass `repair=False` on the handler. The unified wrapper owns repair via
`--per-individual-repair`.

---

## Implementation order

1. **Add `hover_fitness.py`** — pure port. Smoke test:
   ```bash
   python -c "
   import numpy as np
   from airevolve.evolution_tools.evaluators.hover_fitness import continuous_hover_fitness
   # Standard 4-arm quadcopter genome — should hover, score should be high
   genome = np.array([
       [0.13,  np.pi/4, 0, np.pi, 0, 0],
       [0.13, 3*np.pi/4, 0, np.pi, 0, 1],
       [0.13, -3*np.pi/4, 0, np.pi, 0, 0],
       [0.13, -np.pi/4, 0, np.pi, 0, 1],
   ])
   print(continuous_hover_fitness(genome))  # expect close to 3
   "
   ```
2. **Add `UnifiedFitness` wrapper** — start with `fitness_mode='gate'` only,
   no hover-gradient. Verify spherical + lee end-to-end with population_size=4,
   generations=2.
3. **Add the decode shim** — extend the wrapper for CPPN/hybrid-cppn for both
   brains. Verify CPPN + lee, then CPPN + rl.
4. **Add hover-gradient mode** — implement the short-circuit. Verify a known
   non-hoverable drone (e.g., 2-arm coplanar) returns `hover_fit ∈ [0, 3]`
   without the brain running (instrument with a print to confirm).
5. **Add `pure_hover`, `edit_distance`, `zero`** modes.
6. **Lift init-pop helpers** — port `_try_generate_individual`,
   `_try_generate_cppn_individual`, trimmed `generate_initial_pop_parallel`
   (Phase 1 only) into the new script or a helper module.
7. **Write the new `run_evolution.py`** — CLI parsing, dispatch, plotting.
8. **Smoke-test matrix** — see "Smoke tests" below.
9. **Delete old scripts and update README**.

---

## Smoke tests

Run before deleting old scripts:

```bash
# 1. Lee × spherical × random × gate (matches old run_evolution.py with --brain lee)
python examples/evolution/run_evolution.py \
  --brain lee --genome spherical --fitness gate \
  --init-pop-mode random \
  --population-size 4 --generations 2 --max-evals 50 --cma-workers 1

# 2. Lee × cppn × hover_repair × gate (matches old run_evolution_with_lee_tuning.py)
python examples/evolution/run_evolution.py \
  --brain lee --genome cppn --fitness gate \
  --init-pop-mode hover_repair --per-individual-repair \
  --population-size 4 --generations 2 --max-evals 50 --cma-workers 1

# 3. Lee × spherical × random × gate + hover-gradient (the new mode)
python examples/evolution/run_evolution.py \
  --brain lee --genome spherical --fitness gate --hover-gradient \
  --init-pop-mode random \
  --population-size 4 --generations 2 --max-evals 50 --cma-workers 1
# Expect: most non-hoverable drones return fitness in [0, 3], hoverables > 3.

# 4. pure_hover (no brain)
python examples/evolution/run_evolution.py \
  --genome spherical --fitness pure_hover \
  --init-pop-mode random \
  --population-size 4 --generations 2

# 5. RL × cppn (verifies decode shim on RL path)
python examples/evolution/run_evolution.py \
  --brain rl --genome cppn --fitness gate \
  --init-pop-mode random \
  --population-size 4 --generations 2 --training-timesteps 1000 --num-envs 2 --device cpu
```

Pass criteria for each: script completes without exception, `evolution_data.csv`
written, fitness plot rendered.

---

## Resolved decisions

1. **`edit_distance` target = standard hexacopter** — 6 arms equally spaced at
   60°, magnitude 0.13, motors pointing up (`motor_pitch=π` per the memory
   note "motor_pitch=pi gives upward thrust in NED"), alternating CCW/CW
   directions. Define as a module-level constant in `unified_fitness.py`:

   ```python
   STANDARD_HEXACOPTER = np.array([
       [0.13,           0.0, 0.0, np.pi, 0.0, 0],  #   0°
       [0.13,    np.pi/3.0, 0.0, np.pi, 0.0, 1],  #  60°
       [0.13,  2*np.pi/3.0, 0.0, np.pi, 0.0, 0],  # 120°
       [0.13,         np.pi, 0.0, np.pi, 0.0, 1],  # 180°
       [0.13, -2*np.pi/3.0, 0.0, np.pi, 0.0, 0],  # 240°
       [0.13,   -np.pi/3.0, 0.0, np.pi, 0.0, 1],  # 300°
   ])
   ```

   Use the `shared_params` rows from `get_genome_handler_config` for
   `min_vals` / `max_vals`:

   ```python
   EDIT_DISTANCE_MIN = np.array([0.055, -np.pi, -np.pi/2, -np.pi, -np.pi, 0])
   EDIT_DISTANCE_MAX = np.array([0.17,   np.pi,  np.pi/2,  np.pi,  np.pi, 1])
   ```

   No CLI flag needed — the target is fixed.

2. **Repair-before-eval is universal.** When `--per-individual-repair` is set,
   the 3-stage repair runs before *every* fitness mode (`gate`, `pure_hover`,
   `edit_distance`, `zero`) — repaired phenotype is what gets scored. When
   the flag is off, no repair runs and the raw decoded phenotype is scored.
   This is already what the wrapper sketch does (step 3 of `__call__` runs
   before the dispatch in step 4) — flagging it here so the contract is
   explicit.

## Things still to verify during implementation

- **Hybrid-cppn `coordinate_system`.** `repair_coord` is hardcoded to
  `'spherical'` for indirect encodings in the wrapper. Verify this matches
  `HybridCPPNDroneGenomeHandler.get_phenotype()` output (the existing
  `_RepairAndEvaluateFitness` does the same thing, so this should be safe,
  but worth a sanity check on first hybrid-cppn run).

---

## Risks (from earlier discussion + new ones)

- **`--num-workers` interaction with PPO.** PPO already runs vectorized envs;
  parallelizing over individuals on top of that may oversubscribe the GPU.
  Default `--num-workers 1` for `--brain rl`, leave 32 for `--brain lee`.
- **`stage2_hover_check` is on `main`** (`airevolve/evolution_tools/genome_handlers/repair_workflow.py:332`).
  Confirmed signature: `(individual, verbose=False, allow_spinning=False) → (bool, str)`.
- **CPPN-decoded phenotype shape compatibility.** Both brains assume
  `(n_arms, 6)` spherical layout. CPPN/hybrid-cppn `get_phenotype()` returns
  this shape (confirmed by reading `_RepairAndEvaluateFitness` which already
  does this for the Lee path), but the RL brain has never seen one. First CPPN
  + RL run might surface a column-order or dtype issue — verify in step 3.
- **`README.md` references the old scripts** at lines 39 and 46-48. Update both
  (the quick-start command line and the examples directory description) when
  deleting.
- **`examples/visualization/visualize_initial_bspline.py`** mentions
  `run_evolution_with_lee_tuning.py` in comments only (lines 33, 364). Update
  comments, no code change.
- **No tests directly import the deleted scripts** (verified via
  `grep -rn run_evolution unit_tests/`). Safe to delete.
- **Elite re-evaluation is intentional** (per
  `~/.claude/projects/-home-jed-workspaces-airevolve/memory/MEMORY.md`). The
  `evolve()` function has a `reevaluate_old: bool = False` parameter — leave
  it at the default unless told otherwise. The unified script should not flip
  it on for "efficiency" reasons.
