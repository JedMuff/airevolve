# Lee Tuning All Genotypes Experiment

**Script:** `experimentation/slurm_cluster/lee_tuning_all_genotypes_slurm.sh`
**Entry point:** `examples/evolution/run_evolution_with_lee_tuning.py`

## Experiment Overview

This experiment co-evolves drone morphologies and tunes their controllers across 3 genotype representations and 4 trajectory tasks. Each drone morphology is evaluated by how many gates it can navigate after its Lee flight controller is optimized via a 2-stage CMA-ES pipeline.

**SLURM job array:** 120 jobs = 10 repetitions x 3 genotypes x 4 tasks
**Resources per job:** 32 CPUs, 31 GB RAM, 100-hour wall time

### Experimental Factors

| Factor | Levels |
|--------|--------|
| Genotype | `spherical`, `cppn`, `hybrid-cppn` |
| Task | `backandforth`, `figure8`, `circle`, `slalom` |
| Repetitions | 10 per condition |

---

## Genotype Representations

All conditions use a **fixed 6 arms** (`--min-narms 6 --max-narms 6`).

### Spherical Angular (direct encoding)

Each genome is an `(max_narms, 6)` array. Per-arm parameters:

| Parameter | Description | Bounds |
|-----------|-------------|--------|
| `r` | Radial distance from center | [0.055, 0.105] m |
| `theta` | Arm azimuth angle | [-pi, pi] |
| `phi` | Arm elevation angle | [-pi/2, pi/2] |
| `motor_pitch` | Motor tilt pitch | angular |
| `motor_yaw` | Motor tilt yaw | angular |
| `direction` | Spin direction | 0 (CCW) or 1 (CW) |

**Random generation:**
1. Number of arms drawn uniformly from `[min_narms, max_narms]` (fixed at 6 here). Unused rows are NaN.
2. `r`, `theta`, `motor_pitch`, `motor_yaw` are sampled uniformly within their bounds.
3. `phi` uses **arcsin sampling** for uniform spatial distribution on the sphere: `sin(phi)` is drawn uniformly in `[sin(phi_min), sin(phi_max)]`, then inverted with `arcsin`. This avoids the polar bias that uniform angular sampling would produce.
4. `direction` is a random binary choice (0 or 1).

### CPPN (indirect encoding)

A Compositional Pattern Producing Network (CPPN) that is decoded into the same `(max_narms, 6)` spherical phenotype. Configured with `--num-segments 8` and `--initial-hidden-nodes 0`.

**Network topology:**
- **Inputs (2):** `seg_normalized` (segment index normalized to [-1, 1]) and `bias` (constant 1.0).
- **Outputs (7):** `arm_present`, `magnitude`, `arm_yaw`, `arm_pitch`, `motor_yaw`, `motor_pitch`, `direction`. All use SIN activation.
- Initial topology is fully connected (2 x 7 = 14 connections) with no hidden nodes (`--initial-hidden-nodes 0`).

**Random generation:**
1. All connection weights drawn uniformly from `[-weight_range, weight_range]`.
2. All output node biases drawn uniformly from `[-bias_range, bias_range]`.
3. When `initial_hidden_nodes > 0` (not in this experiment), hidden nodes are inserted by splitting random existing connections. Hidden activations are drawn from {SIN, COS, GAUSSIAN, TANH, ABS}.

**Decoding to phenotype:**
The CPPN is evaluated once per azimuthal segment (8 segments, each spanning 2*pi/8 = 45 degrees). For each segment:
1. If `arm_present` output >= 0 (or if remaining segments <= remaining arm slots), an arm is placed.
2. `arm_yaw` is constrained to the segment's angular range: `segment_center + output * half_width`.
3. `magnitude`, `arm_pitch`, `motor_pitch`, `motor_yaw` are mapped from tanh range [-1, 1] to their parameter bounds.
4. `direction` is thresholded at 0 (negative = 0/CCW, positive = 1/CW).

### Hybrid CPPN (indirect encoding)

Splits the genome into a **direct part** (arm geometry) and a **CPPN part** (motor parameters). Configured with `--initial-hidden-nodes 0`.

**Direct part** -- `(n_arms, 3)` array:
- Column 0: `magnitude` -- uniform in bounds.
- Column 1: `arm_yaw` -- uniform in bounds.
- Column 2: `arm_pitch` -- arcsin sampling (same as spherical).

**CPPN part** -- 4-input, 3-output network:
- **Inputs (4):** normalized magnitude, normalized arm_yaw, normalized arm_pitch, bias (constant 1.0). The direct arm parameters are normalized to [-1, 1] and fed as CPPN inputs.
- **Outputs (3):** `motor_yaw`, `motor_pitch`, `direction`. All use TANH activation.
- Initial topology: fully connected (4 x 3 = 12 connections), no hidden nodes.

**Random generation:**
1. Direct parameters generated the same as spherical (uniform + arcsin for pitch).
2. CPPN weights and biases drawn uniformly from their respective ranges.
3. Hidden node insertion follows the same procedure as the pure CPPN.

**Decoding to phenotype:**
The direct part provides columns 0-2 (`magnitude`, `arm_yaw`, `arm_pitch`). The CPPN is evaluated per-arm with the normalized direct parameters as input, producing columns 3-5 (`motor_pitch`, `motor_yaw`, `direction`). TANH outputs are mapped to parameter bounds; direction is thresholded at 0.

---

## Initial Population Generation (2-Phase)

The initial population is generated through a demanding 2-phase pipeline that ensures every individual in generation 0 is both physically valid and capable of navigating gates.

### Phase 1: Parallel Sampling with Repair Pipeline

Batches of `population_size * 1000` candidate genomes are generated in parallel (32 workers). Each candidate passes through a 3-stage repair pipeline:

**Stage 1 -- Hover Check** (`stage2_hover_check`):
- Uses the `drone-hover` library to solve for static hover equilibrium (motor thrusts that balance gravity).
- Requires all motor thrusts in [0, 1] (physically feasible).
- Strict mode: no spinning allowed.
- ~Most candidates fail here (expected pass rate ~0.2%).

**Stage 2 -- Collision Repair** (`stage1_optimization_repair`):
- Formulates collision removal as a constrained optimization (SLSQP).
- Minimizes deviation from the original genome while enforcing:
  - No cylinder-cylinder arm collisions.
  - No intersection with the central core sphere (radius 0.05 m).
  - Arms attach to the disc base (radius 0.15 m).
  - Propeller clearance (propeller radius 0.0254 m).
- Motor orientation parameters (pitch, yaw) are held fixed (`fixed_params=[3, 4]`).

**Stage 3 -- Hover Repair** (`stage3_hover_repair`):
- Computes the net thrust vector from the hover solution.
- Rotates the entire drone so net thrust aligns with gravity [0, 0, -1].
- Preserves the drone's structure; only adjusts orientation.

Sampling repeats in batches (up to 100 iterations) until `population_size` survivors are collected.

### Phase 2: CMA-ES Controller Pre-Tuning

Each Phase 1 survivor undergoes controller tuning before entering the initial population:

- **Budget:** 500 CMA-ES evaluations per drone (`--init-pop-max-evals 500`)
- **Acceptance threshold:** Must navigate at least 8 gates (`--init-pop-gates-threshold 8`)
- **Parallelism:** 32 tuning workers (`--init-pop-tuning-workers 32`)
- Individuals that fail the threshold are discarded and replaced by new Phase 1 survivors.

This ensures generation 0 starts with morphologies that are already demonstrably capable.

---

## Evolutionary Algorithm

### Strategy

**(mu + lambda) evolution strategy** (`--strategy-type plus`, the default):
- Parents survive into the next generation alongside offspring.
- Combined pool (parents + offspring) is sorted by fitness; top `population_size` are kept.

### Parameters

| Parameter | Value |
|-----------|-------|
| Population size (mu) | 16 |
| Generations | 50 |
| Mutation offspring (lambda) | 16 |
| Crossover offspring | 0 (disabled) |

### Selection
**Tournament selection** with tournament size 3. Selects `num_mutate` parents for mutation each generation.

### Mutation
Three mutation types, selected probabilistically per individual:

1. **Parameter mutation** (primary): Gaussian noise added to continuous parameters (std = 5% of parameter range). Angular parameters are wrapped to valid ranges. The `direction` bit is flipped randomly.
2. **Add arm**: Insert a new randomly-generated arm (probability governed by `append_arm_chance`; effectively 0 in this experiment since arms are fixed at 6).
3. **Remove arm**: Delete a random arm (same probability; effectively 0 here).

### Crossover
Disabled (`--num-crossover 0`). When enabled, it performs arm-wise uniform crossover: for each arm slot, randomly pick from parent 1 or parent 2.

### Repair During Evolution

The 3-stage repair pipeline is re-applied to every offspring **before** fitness evaluation. If any repair stage fails, the individual receives fitness = 0.

---

## Fitness Evaluation: 2-Stage CMA-ES Controller Tuning

Each individual's fitness is determined by tuning a Lee geometric tracking controller on the drone morphology and measuring gate passage.

### Fitness Metric
`gates_passed + normalized_distance_bonus_to_next_gate`

Where the distance bonus is a fractional value in [0, 1) rewarding progress toward the next uncleared gate.

### Stage 1: Gains + Timing Optimization (40% of budget = 200 evals)

Optimizes **7 parameters**:

| Parameter | Description | Bounds | Initial Guess |
|-----------|-------------|--------|---------------|
| `pos_P` | Position control gain | [10.0, 25.0] | 14.3 |
| `vel_P` | Velocity control gain | [0.1, 15.0] | 9.0 |
| `att_P` | Attitude control gain | [0.1, 10.0] | 2.9 |
| `rate_P` | Rate control gain | [-1.0, -0.01] | -0.02 |
| `total_time` | Total trajectory time | [5.0, 30.0] | 12.7 |
| `velocity_scale` | Velocity scaling factor | [0.5, 10.0] | 4.6 |
| `startup_time` | Pre-flight hover time | [0.1, 5.0] | 1.9 |

- Initial CMA-ES sigma: 1.5
- Early stopping: if best result reaches the gates threshold, Stage 2 is skipped.

### Stage 2: Gains + Timing + Trajectory Optimization (60% of budget = 300 evals)

Optimizes **7 + n_gates * 3 parameters**:
- Same 7 controller gains/timing parameters from Stage 1.
- Plus **3 offset control points per gate** defining a B-spline trajectory (`BSplineGateTrajectory`), allowing the optimizer to adjust the flight path around each gate.

- Seeded from Stage 1's best result.
- Initial CMA-ES sigma: 0.3 (fine-tuning).

### Simulation Parameters
- Simulation time: 20.0 s
- Time step: 0.005 s (200 Hz)
- Evaluation timeout: 30.0 s
- CMA-ES workers per drone: 1 (parallelism is across drones, not within CMA-ES)
- Evolution workers: 32 (parallel drone evaluations)

---

## Gate Configurations (Tasks)

### BackAndForth (4 gates)
- 4 gates along the X-axis at x = [2, 8, 8, 2], y = 0, z = 0.
- Drone flies forward and back. Yaw alternates between 0 and pi.
- Start position: [0, 0, 0].

### Figure8 (8 gates)
- 8 gates forming a figure-8 pattern in the XY plane.
- Spans approximately 6 m wide (X: -3 to 3) by 3 m deep (Y: -1.5 to 1.5).
- Start position: [0, -1.5, 0].

### Circle (4 gates)
- 4 gates arranged in a circle of radius 1.5 m.
- Positions: [0, -1.5], [1.5, 0], [0, 1.5], [-1.5, 0].
- Start position: [-1.5, -1.5, 0].

### Slalom (20 gates)
- 20 gates in a straight-line slalom with 2 m spacing along X.
- Y alternates between +1 and -1 m.
- Total course length: ~38 m.
- Start position: [0, -1, 0].

---

## Data Output

Results are written to a temporary directory on the compute node, then moved to persistent scratch storage:

```
/scratch/jed/airevolve_data_180226/lee_tuning_{task}_{genotype}_rep{rep}_{jobid}_{arrayid}/
```

Each run's log directory contains the full evolutionary history (per-generation fitness, genomes, and tuning results).
