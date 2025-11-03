# TimedLR Task (Task B) Guide

## Overview

The `timedlr` task is a challenging flight task inspired by the "Task B" from AirframeOptimization2. It features dynamically generated gates with variable positioning, making it suitable for testing drone adaptability and learning capabilities.

## Task Characteristics

### Gate Configuration

- **Number of gates**: 100 gates
- **Gate spacing**: 0.25m between gates (along X-axis)
- **Gate radius range**: 0.5m to 0.7m from centerline
- **Vertical variation**: ±0.1m in Z-axis
- **Gate width**: 0.5m
- **Left/Right probability**: 5% chance for gates to be positioned left or right

### Course Layout

The task generates a straight course along the X-axis with gates that can be:
- **Center gates**: Most gates (90%) are centered on the X-axis
- **Left gates**: 5% probability, positioned at random radius between 0.5-0.7m to the left
- **Right gates**: 5% probability, positioned at random radius between 0.5-0.7m to the right
- **Vertical variation**: All gates have random Z offset between -0.1m and +0.1m

### Flight Boundaries

- **X bounds**: [0, 27] meters (covers all 100 gates plus buffer)
- **Y bounds**: [-1.7, 1.7] meters 
- **Z bounds**: [-0.6, 0.6] meters

### Starting Position

- **Position**: [0.0, 0.0, 0.0]
- **Orientation**: Aligned with positive X-axis (forward)

## Key Differences from Other Tasks

| Feature | figure8 | circle | slalom | timedlr |
|---------|---------|--------|--------|---------|
| Gates | 8 | 4 | 41 | 100 |
| Layout | Fixed loop | Fixed circle | Fixed zigzag | **Dynamic straight** |
| Difficulty | Medium | Easy | Medium-Hard | **Hard** |
| Randomization | None | None | None | **Yes (seed-based)** |
| Gate spacing | Variable | Regular | 2m | **0.25m** |
| Course length | ~12m | ~9.4m | ~82m | **~25m** |

## Usage Examples

### Basic Training

```bash
# Quick test with default quadcopter (100k timesteps)
python examples/run_learning_evaluation.py \
    --task timedlr \
    --task-seed 42 \
    --timesteps 1e5 \
    --num-envs 10

# Production training (100M timesteps)
python examples/run_learning_evaluation.py \
    --task timedlr \
    --task-seed 42 \
    --production
```

### With Custom Parameters

```python
from airevolve.evolution_tools.evaluators.gate_train import timedlr

# Customize gate parameters
timedlr.gate_d = 0.3  # Wider spacing
timedlr.gate_r_min = 0.6  # Larger minimum radius
timedlr.gate_r_max = 0.9  # Larger maximum radius
timedlr.gate_prob_lr = 0.1  # More left/right gates

# Generate gates
gate_pos, gate_yaw = timedlr.generate_gates(seed=123)
```

### Reproducibility

The task uses seed-based generation for reproducibility:

```python
# Same seed = same gates
gates1, yaw1 = timedlr.generate_gates(seed=42)
gates2, yaw2 = timedlr.generate_gates(seed=42)
assert np.allclose(gates1, gates2)

# Different seed = different gates
gates3, yaw3 = timedlr.generate_gates(seed=99)
assert not np.allclose(gates1, gates3)
```

## Integration with Evolution

### In Evolution Loop

```python
from airevolve.evolution_tools.evaluators.gate_train import evaluate_individual
import numpy as np

# Your evolved individual
individual = np.array([...])  # 6x6 genome array

# Evaluate on timedlr task
fitness = evaluate_individual(
    individual=individual,
    ind_save_dir="./results/gen_0/ind_0",
    training_ts=1e7,
    num_envs=50,
    gate_cfg="timedlr",
    device="cuda:0",
    task_seed=42  # Use consistent seed for fair comparison
)
```

### Cross-Task Evaluation

Evaluate the same individual on multiple tasks:

```python
tasks = ['circle', 'figure8', 'slalom', 'timedlr']
fitness_scores = {}

for task in tasks:
    if task == 'timedlr':
        fitness = evaluate_individual(
            individual, save_dir, 1e7, 50, task,
            task_seed=42
        )
    else:
        fitness = evaluate_individual(
            individual, save_dir, 1e7, 50, task
        )
    fitness_scores[task] = fitness

print(f"Fitness scores: {fitness_scores}")
```

## Performance Metrics

The task measures:
- **Gates passed**: Total number of gates successfully navigated
- **Episode length**: Number of timesteps before completion or failure
- **Crash rate**: Percentage of episodes ending in crashes vs bounds violations
- **Speed**: Average velocity through the course

## Configuration Parameters

All parameters can be modified before generating gates:

```python
from airevolve.evolution_tools.evaluators.gate_train import timedlr

# Course parameters
timedlr.num_gates = 150          # More gates
timedlr.gate_d = 0.2             # Tighter spacing

# Gate positioning
timedlr.gate_r_min = 0.4         # Minimum offset radius
timedlr.gate_r_max = 0.8         # Maximum offset radius
timedlr.gate_z_min = -0.15       # Lower bound
timedlr.gate_z_max = 0.15        # Upper bound

# Gate characteristics  
timedlr.gate_width = 0.6         # Wider gates
timedlr.gate_prob_lr = 0.15      # More lateral variation

# Generate with new parameters
gates, yaw = timedlr.generate_gates(seed=42)
```

## Testing

Run the test script to verify the implementation:

```bash
python examples/test_timedlr_task.py
```

Expected output:
- Gate generation statistics
- Verification of spacing and bounds
- Reproducibility check
- Comparison with other tasks

## Troubleshooting

### Issue: Gates are not reproducible
**Solution**: Ensure you're passing the same `task_seed` parameter consistently.

### Issue: Training is too slow
**Solution**: Reduce `num_gates` or increase `gate_d` for a shorter course during testing.

### Issue: Gates are too difficult
**Solution**: 
- Increase `gate_width` (make gates larger)
- Decrease `gate_prob_lr` (fewer lateral gates)
- Reduce vertical variation (`gate_z_min`, `gate_z_max` closer to 0)

### Issue: Gates are too easy
**Solution**:
- Decrease `gate_width` (make gates smaller)
- Increase `gate_prob_lr` (more lateral gates)
- Increase vertical variation
- Decrease `gate_d` (tighter spacing)

## Comparison with AirframeOptimization2

The `timedlr` task in airevolve is equivalent to "Task B" in AirframeOptimization2 with these characteristics:

| Parameter | AirframeOptimization2 | airevolve |
|-----------|----------------------|-----------|
| Task name | `timedlr` | `timedlr` |
| Gate distance | 0.25m | 0.25m ✓ |
| Gate radius | 0.5-0.7m | 0.5-0.7m ✓ |
| Gate z variation | ±0.1m | ±0.1m ✓ |
| Gate width | 0.5m | 0.5m ✓ |
| LR probability | 0.05 | 0.05 ✓ |
| Number of gates | 100 | 100 ✓ |
| Seed-based generation | Yes | Yes ✓ |

## Future Enhancements

Potential improvements:
- [ ] Add gate size variation
- [ ] Implement wind disturbances
- [ ] Add moving gates
- [ ] Support gate orientation changes
- [ ] Add time-based rewards
- [ ] Implement multi-lap courses
