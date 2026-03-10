# Continuous Hover Fitness Function for CPPN Evolution

## Problem

The current approach to finding hoverable drones is binary: generate a morphology, run the full hover check (SLSQP optimisation), pass or fail. This is expensive (~0.2% success rate for random CPPNs) and provides zero gradient signal — a drone that *nearly* hovers gets the same score as one that's completely wrong.

## Goal

Design a **continuous fitness function** that measures *how close* a decoded phenotype is to being a hoverable drone. This enables guided search (hill climbing, (1+1)-ES, etc.) instead of blind random mutation.

## Background: What Makes a Drone Hoverable

The drone-hover package (`dronehover/optimization.py`) checks two constraints simultaneously:

### Force Balance

The total thrust from all propellers must equal gravity:

```
η^T @ A @ η = G²

where:
  A = Bf^T @ Bf           (force Gram matrix, 3×3 → scalar via η)
  Bf = M^{-1} @ F_cols     (force effectiveness matrix, 3 × n_props)
  η = [ŵ₁², ŵ₂², ..., ŵₙ²]  (squared normalised motor speeds)
  G = 9.81 m/s²
```

Each column of Bf represents one propeller's contribution to linear acceleration:

```
Bf[:, i] = (1/mass) × k_f[i] × ω_max[i]² × thrust_dir[i]

where:
  k_f[i]     = force constant (8.12e-08 for 2-inch props)
  ω_max[i]   = max RPM (5000 for 2-inch props)
  thrust_dir[i] = unit vector of thrust direction in NED frame
```

### Torque Balance

**Static hover**: net torque must be zero:
```
η^T @ Bm^T @ Bm @ η = 0
```

**Spinning hover** (relaxed): torque must be aligned with net force:
```
‖cross(Bf @ η, Bm @ η)‖ = 0
```

Each column of Bm represents one propeller's contribution to angular acceleration:

```
Bm[:, i] = I^{-1} @ (cross(r_i - cg, k_f[i] × ω_max[i]² × thrust_dir[i])
                     + k_m[i] × ω_max[i]² × rot_dir[i] × thrust_dir[i])

where:
  I           = 3×3 inertia tensor
  r_i         = propeller position in NED frame
  cg          = centre of gravity in NED frame
  k_m[i]      = moment constant (6.40e-10 for 2-inch props)
  rot_dir[i]  = +1 (CW) or -1 (CCW)
```

### What the Hover Check Actually Does

The `Hover.static()` method runs SLSQP to minimise input cost (`η^T @ η`) subject to:
- Force constraint: `η^T @ A @ η = G²`
- Moment constraint: `η^T @ Bm^T @ Bm @ η = 0`
- Bounds: `0.02² ≤ η[i] ≤ 1.0²` (motor speed limits)

If SLSQP converges → static hover. If not, tries spinning hover (relaxed torque constraint). If neither works → cannot hover.

## Empirical Calibration Data

Computed on 53 known-hoverable drones (from repaired genomes) and 500 random phenotypes (192 hoverable, 308 non-hoverable):

### Known-hoverable drones (53 repaired genomes)

| Metric | Min | Max | Mean | Median |
|--------|-----|-----|------|--------|
| σ_min(Bf) | 1.50 | 4.23 | 2.78 | 2.94 |
| σ_min(Bm) | 104.8 | 603.8 | 303.7 | 311.7 |
| thrust_ratio | 1.00 | 1.86 | 1.36 | 1.22 |
| condition (λ_min/λ_max) | 0.008 | 0.392 | 0.206 | 0.229 |
| authority √λ_min | 104.8 | 603.8 | 303.7 | 311.7 |

### Random hoverable vs non-hoverable (500 random phenotypes)

| Metric | Hoverable (192) mean | Non-hoverable (308) mean |
|--------|---------------------|--------------------------|
| σ_min(Bf) | 2.64 | 2.77 |
| σ_min(Bm) | 394.6 | 417.4 |
| thrust_ratio | 1.64 | 1.54 |
| condition | 0.139 | 0.152 |
| authority | 394.6 | 417.4 |

### Important Caveat

The distributions **overlap heavily** between hoverable and non-hoverable drones. These metrics alone don't cleanly discriminate at the hover boundary — whether a drone can actually hover depends on whether a feasible motor speed vector η exists within bounds, which is a global property of the constrained optimisation, not just a matrix property.

**However**, for the empty-init experiment these metrics are still valuable: an empty CPPN (no connections) produces Bf = 0 and Bm = 0, so all values start at zero and must climb to the ranges above. The fitness function provides strong gradient signal during the construction phase, even if it becomes noisy near the hover boundary.

## Proposed Continuous Fitness Function

### Overview

The fitness decomposes into **three additive terms**, each measuring a different aspect of hoverability. All are cheap to compute (matrix operations, no simulation). Each term is normalised to [0, 1].

```
fitness(phenotype) = f_struct + f_force + f_torque     ∈ [0, 3]
```

### Term 1: Rank Feasibility (`f_rank`)

The most fundamental requirement — can the motors even *in principle* produce forces and torques in all 3 axes?

```
rank_f = rank(Bf)                    # Force rank (0–3)
rank_m = rank(Bm)                    # Moment rank (0–3)

f_rank = (rank_f + rank_m) / 6.0     ∈ [0, 1]
```

- **f_rank = 1.0**: Both matrices full rank — hovering is geometrically possible
- **f_rank < 1.0**: Missing a control axis — fundamentally cannot hover
- **Cost**: O(n³) for rank computation — negligible

This is a discrete signal (only 7 possible values) but captures meaningful structural milestones as the CPPN builds up connections. Between rank changes, Terms 2 and 3 provide continuous signal.

**Continuity**: No — rank is discrete. Acceptable because rank changes correspond to structural mutations (adding connections/nodes) which are inherently discrete events. ✗

### Term 2: Force Capability (`f_force`)

Measures how much vertical thrust the motors can produce relative to gravity.

```
f_max = ‖Bf @ η_max‖     where η_max = [1, 1, ..., 1]
thrust_ratio = f_max / G

f_force = min(thrust_ratio, 2.0) / 2.0     ∈ [0, 1]
```

**Normalisation**: Linear mapping clipped at 2× gravity. A drone needs thrust_ratio ≥ 1.0 to hover (f_force ≥ 0.5). Hoverable drones typically have thrust_ratio 1.0–1.9.

**Continuity**: `‖Bf @ η‖` is continuous in Bf (matrix norm of a linear map). The `min(·, 2.0)` introduces a kink at exactly 2.0 but remains continuous (not differentiable at the kink, but continuous). ✓

### Term 3: Torque Balance (`f_torque`)

Measures how balanced the drone's torque authority is across all rotation axes.

```
gram_m = Bm @ Bm^T                         # 3×3 moment Gram matrix
eigs = eigenvalues(gram_m)                  # 3 non-negative eigenvalues
λ_min = min(eigs)
λ_max = max(eigs)

condition = λ_min / (λ_max + ε)             # ∈ [0, 1], how balanced across axes
authority = √λ_min / (√λ_min + c_a)         # ∈ [0, 1), overall torque capability

f_torque = condition × authority             ∈ [0, 1]
```

**Normalisation**:
- `condition = λ_min / (λ_max + ε)` is naturally in [0, 1] (ratio of smallest to largest eigenvalue, with ε = 1e-12 for numerical safety). Value of 1.0 = perfectly isotropic torque authority.
- `authority` uses the same half-saturation form as Term 1.

**Calibration constant**:
```
c_a = 300.0   # √λ_min median ≈ 312 for hoverable drones
```

The product means both conditions matter: balanced but weak torque (low authority) scores low, and strong but imbalanced torque (low condition) also scores low.

**Continuity**: Eigenvalues of a symmetric matrix are continuous functions of its entries. The ratio λ_min/(λ_max + ε) is continuous (ε prevents division by zero; when λ_max = 0, both eigenvalues are 0 so the ratio is 0). The half-saturation function is smooth. ✓

## Implementation

```python
import numpy as np
from numpy.linalg import norm, eig

from airevolve.evolution_tools.inspection_tools.morphological_descriptors.hovering_info import (
    get_sim,
)

G = 9.81

# Half-saturation constant (from empirical calibration on hoverable drones)
C_AUTHORITY = 300.0  # √λ_min(Bm@Bm.T) half-saturation


def continuous_hover_fitness(phenotype: np.ndarray) -> float:
    """Compute continuous hover fitness for a decoded phenotype.

    Args:
        phenotype: Array of shape (max_narms, 6) with NaN for unused rows.
                   Columns: [magnitude, arm_yaw, arm_pitch, motor_pitch,
                             motor_yaw, direction]

    Returns:
        Fitness score in [0, 3]. Higher = closer to hoverable.
        Each sub-term is in [0, 1].
    """
    sim = get_sim(phenotype)
    if sim is None:
        return 0.0

    Bf = sim.Bf  # (3, n_props)
    Bm = sim.Bm  # (3, n_props)

    # --- Term 1: Rank feasibility [0, 1] ---
    rank_f = np.linalg.matrix_rank(Bf)
    rank_m = np.linalg.matrix_rank(Bm)
    f_rank = (rank_f + rank_m) / 6.0

    # --- Term 2: Force capability [0, 1] ---
    n_props = Bf.shape[1]
    eta_max = np.ones(n_props)
    f_vec = Bf @ eta_max
    thrust_ratio = norm(f_vec) / G
    f_force = min(thrust_ratio, 2.0) / 2.0

    # --- Term 3: Torque balance [0, 1] ---
    gram_m = Bm @ Bm.T
    eigs = np.real(eig(gram_m)[0])
    eigs = np.maximum(eigs, 0.0)

    lambda_min = np.min(eigs)
    lambda_max = np.max(eigs)

    condition = lambda_min / (lambda_max + 1e-12)
    authority = np.sqrt(lambda_min)
    authority_sat = authority / (authority + C_AUTHORITY)
    f_torque = condition * authority_sat

    return f_rank + f_force + f_torque
```

## Expected Values

### On known-hoverable drones (53 repaired genomes)

| Term | Expected range | Reasoning |
|------|---------------|-----------|
| f_rank | 1.0 | All hoverable drones have full rank Bf and Bm (rank 3 + 3 = 6) |
| f_force | 0.50–0.93 | thrust_ratio 1.0–1.86, mapped linearly to [0.5, 0.93] |
| f_torque | 0.001–0.20 | condition is low (0.008–0.39) × authority_sat (~0.5) |
| **total** | **1.5–2.1** | Sum of three terms |

### On empty CPPN (no connections)

| Term | Value | Reasoning |
|------|-------|-----------|
| f_rank | ? | Bf and Bm are non-zero (arms are placed by forcing rule) but may not be full rank |
| f_force | ~0.5 | All output biases=0, SIN(0)=0, but arm forcing rule places 6 arms with default params → some thrust |
| f_torque | ~0.0 | Identical arm configurations → torques likely cancel poorly |

The fitness should climb toward ~1.5+ as the CPPN builds diverse structure.

## How This Enables Guided CPPN Evolution

### Current approach (random walk)
```
for each mutation:
    mutate CPPN
    decode to phenotype
    if hover_check_passes(phenotype):    # binary: 0 or 1
        accept
```

### Proposed approach (hill climbing)
```
network = create_empty_cppn()
best_fitness = continuous_hover_fitness(decode(network))

for each mutation:
    candidate = copy(network)
    mutate(candidate)
    phenotype = decode(candidate)
    fitness = continuous_hover_fitness(phenotype)    # continuous: 0.0 to 3.0

    if fitness > best_fitness:
        network = candidate
        best_fitness = fitness

    if best_fitness > HOVER_THRESHOLD:
        run_full_hover_check(phenotype)    # only when fitness is high
```

### Expected Benefits

1. **Sample efficiency**: Every mutation gets a gradient signal, not just pass/fail
2. **Speed**: Fitness computation is ~100× cheaper than the full SLSQP hover check (no iterative optimisation, just matrix ops)
3. **Progressive construction**: Empty CPPNs start at low fitness and climb toward hoverability — we can track the trajectory
4. **Threshold gating**: Only run the expensive full pipeline (hover check + repair + CMA-ES) when continuous fitness exceeds a threshold — dramatically reduces wasted computation

### Limitations

- The fitness function does **not predict hoverability** at the boundary — hoverable and non-hoverable drones have overlapping metric distributions (see calibration data above)
- The real hover check involves finding a feasible η within motor speed bounds, which is a global constraint satisfaction problem that these local matrix properties don't capture
- The fitness is most useful for **guiding structure from nothing** (empty CPPN → reasonable morphology), less useful for fine-tuning near the hover boundary

## Physics Reference

### Phenotype to Propeller Conversion

Each arm row `[mag, arm_yaw, arm_pitch, mot_pitch, mot_yaw, direction]` becomes:

**Position** (spherical → Cartesian → NED):
```
x_enu = mag × cos(arm_pitch) × cos(arm_yaw)
y_enu = mag × cos(arm_pitch) × sin(arm_yaw)
z_enu = mag × sin(arm_pitch)

x_ned = y_enu,  y_ned = x_enu,  z_ned = -z_enu
```

**Thrust direction** (motor orientation → unit vector in NED):
```
R = R_z(mot_yaw) @ R_y(-mot_pitch)
thrust_enu = R @ [0, 0, -1]
thrust_ned = ENU_to_NED(thrust_enu)
```

**Constants** (for 2-inch propellers):
```
k_f = 8.12e-08    (force constant, N/rad²s²)
k_m = 6.40e-10    (moment constant, Nm/rad²s²)
ω_max = 5000 RPM  (max motor speed)
mass_motor = 0.0046 kg
```

### Mass Model
```
mass = core_mass + n_arms × motor_mass + arm_mass_coeff × Σ(arm_lengths)

where:
  core_mass = 0.25 kg (controller + battery + frame)
  motor_mass = 0.05 kg per arm
  arm_mass_coeff = 0.01 kg/unit_length
```

### Key Files
- `dronehover/optimization.py` — `Hover` class, Bf/Bm construction, SLSQP hover check
- `dronehover/bodies/custom_bodies.py` — `Custombody`, inertia computation
- `dronehover/__init__.py` — Propeller library (k_f, k_m, ω_max, mass)
- `airevolve/.../hovering_info.py` — `get_sim()`, `drone_info()`, phenotype → propeller conversion
- `airevolve/.../utils.py` — `convert_to_cartesian()`, `ENU_to_NED()`
- `airevolve/.../mass.py` — `compute_total_mass()`
- `airevolve/.../inertia.py` — `inertia()` via parallel axis theorem
