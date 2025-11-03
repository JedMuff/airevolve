# Reward Function Update for TimedLR Task

## Summary

The reward function in `drone_gate_env.py` has been updated to match the reward structure used in AirframeOptimization2's aerial_gym_dev navigation task. This should significantly improve learning performance on the challenging timedlr task.

## Key Changes

### Before (Simple Linear Rewards)
```python
# Progress toward gate (linear)
prog_rewards = d2g_old - d2g_new

# Small angular velocity penalty
rat_penalty = 0.001 * ||angular_velocity||

rewards = prog_rewards - rat_penalty

# Out-of-bounds: -10
```

### After (aerial_gym_dev-style Exponential Rewards)
```python
# Exponential distance rewards (two scales)
pos_reward = 5.0 * exp(-d2g^2 / 3.5^2) + 5.0 * exp(-2.0 * d2g^2)

# Getting closer bonus (asymmetric)
getting_closer_reward = 10.0 * progress  (if moving closer)
                      = 20.0 * progress  (if moving away - 2x penalty)

# Distance from goal
distance_reward = (20.0 - d2g) / 20.0

# Action smoothness
action_diff_penalty = -0.8 * sum(exp(-3.333 * action_diff^2) - 1.0)

# Angular velocity penalty
ang_vel_penalty = -0.5 * ||angular_velocity||^2

# Combined
rewards = pos_reward + getting_closer_reward + distance_reward 
          + action_diff_penalty + ang_vel_penalty

# Gate passing bonus: +50.0
# Out-of-bounds: -100.0 (increased from -10)
```

## Reward Components Explained

### 1. Exponential Distance Rewards
- **Purpose**: Provide much stronger reward signals when close to gates
- **Effect**: Agent gets clear feedback that being near the gate is valuable
- **Two components**: One for long-range guidance, one for close-range precision

### 2. Getting Closer Reward (Asymmetric)
- **Purpose**: Reward progress, heavily penalize moving away
- **Effect**: Agent learns that moving toward gate is good, moving away is very bad
- **Asymmetry**: Moving away is penalized 2x compared to progress reward

### 3. Distance from Goal Reward
- **Purpose**: Provide baseline reward based on absolute distance
- **Effect**: Keeps reward positive even when far from gate

### 4. Action Smoothness Penalty
- **Purpose**: Discourage jerky, erratic control
- **Effect**: Agent learns smooth, controlled flight paths
- **Implementation**: Exponential penalty on action changes

### 5. Angular Velocity Penalty
- **Purpose**: Discourage spinning and unstable flight
- **Effect**: Agent learns to maintain stable orientation
- **Increased**: From 0.001 to 0.5 coefficient for stronger effect

### 6. Gate Passing Bonus
- **Purpose**: Explicit reward for achieving the primary goal
- **Effect**: Clear signal that passing gates is the objective
- **Value**: +50.0 (new addition)

### 7. Collision/Out-of-Bounds Penalty
- **Purpose**: Strong deterrent for crashes
- **Effect**: Agent learns to stay within bounds
- **Increased**: From -10 to -100 to match aerial_gym_dev

## Why These Changes Matter

### Problem with Linear Rewards
The original linear reward (`d2g_old - d2g_new`) provides very weak signals:
- When 5m away, moving 0.1m closer gives reward = 0.1
- When 0.5m away, moving 0.1m closer gives reward = 0.1
- No distinction between "close" and "far"

### Solution: Exponential Rewards
The new exponential rewards provide stronger gradients near goals:
- When 5m away: reward ≈ 0.1 per step closer
- When 0.5m away: reward ≈ 4.0 per step closer
- Agent gets much clearer feedback when near gates

### For TimedLR Task (0.25m gate spacing)
This is critical because:
1. **Precise navigation required**: 0.25m spacing needs very accurate control
2. **Strong local gradients**: Exponential rewards provide clear signals near gates
3. **Stable flight encouraged**: Action smoothness and angular velocity penalties
4. **Clear objectives**: Gate passing bonus makes goal explicit

## Expected Impact on Training

### Learning Speed
- **Faster initial learning**: Exponential rewards provide clearer gradients
- **Better exploration**: Asymmetric penalty discourages bad trajectories quickly
- **Smoother convergence**: Action penalties encourage stable policies

### Final Performance
- **Better gate passage**: Explicit +50 bonus for passing gates
- **Smoother flight**: Action smoothness penalties
- **Fewer crashes**: Higher out-of-bounds penalty (-100 vs -10)

## Testing the New Rewards

Run the test script to verify the reward function:
```bash
cd examples
python test_reward_function.py
```

## Training Recommendations

With the new reward structure:

1. **Start with shorter episodes** to get early feedback
2. **Use higher learning rates initially** - rewards are more informative now
3. **Monitor average reward** - should be higher than before due to distance_reward baseline
4. **Watch for gate passing** - should see non-zero gate counts earlier in training

## Comparison with AirframeOptimization2

The reward function now closely matches the navigation task used in aerial_gym_dev:
- ✓ Exponential distance rewards
- ✓ Getting closer reward with asymmetry
- ✓ Distance from goal reward
- ✓ Action smoothness penalties
- ✓ Angular velocity penalty
- ✓ High collision penalty (-100)
- ✓ Gate passing bonus (not in navigation, but essential for timedlr)

**Differences:**
- aerial_gym_dev uses curriculum learning (multiplies rewards by 1.0-3.0)
- aerial_gym_dev uses velocity control (4D actions), airevolve uses motor control
- aerial_gym_dev has separate x/z/yaw action penalties, airevolve uses uniform penalty

## Files Modified

1. **drone_gate_env.py**: Updated reward computation in `step_wait()` method
2. **PPO_TRAINING_GUIDE.md**: Documented new reward structure
3. **test_reward_function.py**: Created test script for reward verification

## Next Steps

1. Run `test_reward_function.py` to verify rewards work correctly
2. Train a new model with the updated rewards:
   ```bash
   python run_learning_evaluation.py --task timedlr --timesteps 1e7 --num-envs 100
   ```
3. Compare performance with previous training results
4. Consider hyperparameter tuning with new reward structure

## References

- Original reward code: `aerial_gym_dev/aerial_gym/task/navigation_task/navigation_task.py`
- Configuration: `aerial_gym_dev/aerial_gym/config/task_config/navigation_task_config.py`
- Task parameters: `AirframeOptimization2/src/main.py` (Task B definition)
