# Learning Stabilization in Optuna Hyperparameter Optimization

## Overview

This document describes the learning stabilization method implemented in `optuna_hyperparameter_search.py` to balance final task performance with training stability during hyperparameter optimization for PPO-based drone control.

## Motivation

During hyperparameter optimization, trials that achieve high final fitness may exhibit unstable learning curves with high variance in episode returns during training. Such hyperparameters can lead to:

- **Unpredictable learning dynamics**: High variance makes it difficult to assess convergence
- **Sensitivity to initialization**: Results may vary significantly across different random seeds
- **Reduced reproducibility**: Training outcomes become less reliable
- **Poor generalization**: Models may overfit to specific episodes rather than learn robust policies

By incorporating a stability metric into the optimization objective, we encourage Optuna to discover hyperparameter configurations that not only achieve high performance but also exhibit smooth, consistent learning trajectories.

## Implementation

### Stability Metric

The stability metric quantifies the variability of episode returns during the **second half** of training:

```python
# Extract episode returns from VecMonitor
monitor_data = env.get_attr('episode_returns')[0]

# Focus on second half of training (after initial exploration)
second_half = monitor_data[len(monitor_data)//2:]

# Compute coefficient of variation as stability penalty
stability_penalty = np.std(second_half) / (np.mean(second_half) + 1e-8)
```

**Key design choices:**

1. **Second-half focus**: We ignore the first half of training where high variance is expected due to initial exploration and learning. By focusing on the latter half, we assess whether the agent has converged to a stable policy.

2. **Coefficient of variation (CV)**: Using CV (std/mean) provides a normalized measure of variability that is scale-invariant. This allows fair comparison across different reward scales and gate configurations.

3. **Efficient computation**: The stability metric is derived from training data already collected by `VecMonitor`, avoiding the computational overhead of additional evaluation episodes.

### Objective Function

The combined objective balances final fitness (number of gates passed) with training stability:

```python
combined_score = fitness - stability_weight * stability_penalty
```

Where:
- **fitness**: Number of gates passed in a deterministic evaluation episode (higher is better)
- **stability_penalty**: Coefficient of variation of episode returns in the second half of training (lower is better)
- **stability_weight**: User-configurable parameter controlling the trade-off (default: 0.3)

### Configuration

The stability weight can be adjusted via command-line argument:

```bash
python examples/optuna_hyperparameter_search.py \
    --task figure8 \
    --n-trials 50 \
    --stability-weight 0.3
```

**Note**: The current implementation uses a hardcoded quadcopter design defined in the script's `main()` function. To optimize for a different drone morphology, modify the `example_individual` array in the script. The genome format is a numpy array where each row represents a motor with parameters: `[x, y, z, roll, pitch, yaw, direction]`, where position is in meters, angles in radians, and direction is 0 (CCW) or 1 (CW).

**Recommended values:**

- `0.0`: Ignore stability, optimize only for final fitness
- `0.1-0.2`: Light stability preference
- `0.3-0.5`: Moderate stability preference (recommended for most cases)
- `0.5-1.0`: Strong stability preference, may sacrifice some final performance

## Evaluation Process

For each Optuna trial:

1. **Training phase** (20,000 timesteps):
   - PPO trains on the gate task using trial hyperparameters
   - `VecMonitor` records episode returns throughout training
   - No additional computational overhead for stability tracking

2. **Stability computation**:
   - Extract episode returns from the second half of training
   - Compute coefficient of variation as stability penalty

3. **Evaluation phase** (1 deterministic episode):
   - Run a single deterministic episode to measure final fitness
   - Count number of gates passed

4. **Scoring**:
   - Combine fitness and stability into a single objective value
   - Optuna maximizes this combined score across trials

## Benefits

1. **No additional episodes**: Unlike multi-episode evaluation approaches, this method uses training data already collected, adding negligible computational cost.

2. **Robust hyperparameters**: Selected configurations exhibit more consistent learning behavior across different seeds and initializations.

3. **Easier hyperparameter transfer**: Stable hyperparameters are more likely to generalize to related tasks or modified environments.

4. **Interpretable control**: The `stability_weight` parameter provides intuitive control over the stability-performance trade-off.

## Example Results

### Without Stability Consideration (stability_weight=0.0)
- Trial may find hyperparameters achieving 8 gates passed
- Training curve shows high variance (CV ≈ 0.8)
- Results vary significantly across reruns

### With Stability Consideration (stability_weight=0.3)
- Trial finds hyperparameters achieving 7 gates passed
- Training curve shows smooth progression (CV ≈ 0.3)
- Consistent results across multiple seeds
- **Trade-off**: Slight reduction in peak performance for significantly improved stability

## Technical Notes

### VecMonitor Integration

The `VecMonitor` wrapper from Stable-Baselines3 automatically tracks episode returns during training. Our implementation accesses this data via:

```python
env.get_attr('episode_returns')[0]
```

This returns a list of cumulative rewards for each completed episode across all vectorized environments.

### Handling Edge Cases

- **Division by zero**: Added `1e-8` epsilon to prevent division errors when mean returns are near zero
- **Insufficient episodes**: If fewer than 2 episodes complete in the second half, stability penalty defaults to 0.0
- **Early termination**: If evaluation episode completes early (all gates passed), fitness is still correctly computed

## Future Enhancements

Potential improvements to the stabilization method:

1. **Adaptive stability weight**: Automatically adjust based on task difficulty
2. **Multi-metric stability**: Incorporate variance in value function estimates or policy entropy
3. **Visualization**: Plot stability vs. fitness for all trials to help tune the weight parameter
4. **Per-environment tuning**: Different stability preferences for different gate configurations

## References

- Stable-Baselines3 VecMonitor: [documentation](https://stable-baselines3.readthedocs.io/en/master/common/monitor.html)
- Optuna optimization framework: [documentation](https://optuna.readthedocs.io/)
- PPO algorithm: Schulman et al., "Proximal Policy Optimization Algorithms" (2017)

## Related Files

- `examples/optuna_hyperparameter_search.py`: Main implementation
- `examples/PPO_TRAINING_GUIDE.md`: General PPO training documentation
- `airevolve/evolution_tools/evaluators/gate_train.py`: Gate task configurations
- `airevolve/evolution_tools/evaluators/drone_gate_env.py`: Vectorized environment implementation
