# Curriculum Learning Guide

## Overview

**Curriculum learning** is an optional training strategy that progressively increases task difficulty during reinforcement learning. In airevolve, it's designed to help agents learn challenging tasks by starting with easier rewards and gradually shifting to harder objectives.

This guide explains:
- What curriculum learning is
- When it's used in airevolve
- How it works under the hood
- How to customize it for your experiments

---

## ⚠️ Important Note: Experimental Status

**Curriculum learning and advanced rewards are currently in EXPERIMENTAL status**. While the implementation is complete and the code has been tested for correctness, **there are no published results demonstrating that training with advanced rewards successfully completes the timedlr task**. 

Use this feature for research and experimentation. If you encounter issues:
1. Check the Troubleshooting section below
2. Consider using simpler tasks (lrcontinuous, figure8) first
3. Report results and issues to help improve the implementation

---

## What Is Advanced Reward?

Before diving into curriculum learning, it's important to understand **advanced rewards**, since curriculum learning only works with them.

### Simple vs Advanced Rewards

Airevolve supports two reward structures:

**Simple Reward** (used for most tasks):
```
reward = (distance_to_previous_gate - distance_to_next_gate)  # Progress toward goal
       - 0.001 × angular_velocity                             # Small penalty for rotation
```
- **2 components**: distance progress + angular velocity penalty
- **Static**: same rewards throughout training
- **Linear**: reward scales linearly with gate progress
- **Used for**: figure8, circle, slalom, backandforth, lrcontinuous

**Advanced Reward** (used for timedlr with curriculum):
```
reward = position_reward                                      # Exponential distance rewards
       + getting_closer_bonus                                 # 10x progress or -20x if moving away
       + distance_from_goal_reward                            # Bonus for proximity
       + action_smoothness_penalty                            # Discourage jerky control
       + angular_velocity_penalty                             # Encourage stable flight
       - 100.0 × curriculum_progress × (out_of_bounds)        # Dynamic boundary penalty
       + 50.0 × curriculum_progress × (gate_passed)           # Dynamic gate bonus
```
- **7+ components**: multiple reward signals combined
- **Curriculum-scaled**: 3 components change over time based on curriculum progress
- **Exponential**: distance rewards use exponential functions for stronger signals when close
- **Used for**: timedlr (Task B) only

### Relationship to AirframeOptimization2

⚠️ **Important**: Airevolve's advanced reward is **inspired by but NOT identical to** AirframeOptimization2's reward function. Both use curriculum learning and multiple reward components, but the specific design differs:

| Feature | AirframeOptimization2 | Airevolve | Status |
|---------|----------------------|-----------|--------|
| Position reward | Linear (d_prev - d) | Exponential (two scales) | Different |
| Action smoothness | ❌ No | ✅ Yes | Airevolve only |
| Angular velocity | Threshold (>10 rad/s): -2.0 | Always: -0.5 × norm² | Different |
| Tilt penalty | ✅ Yes (>90°): -2.0 | ❌ No | AirframeOptimization2 only |
| Yaw penalty | ✅ Yes: -0.01 × error | ❌ No | AirframeOptimization2 only |
| Safety penalty | ✅ Yes (gate geometry) | ❌ No | AirframeOptimization2 only |
| Gate bonus | +10.0 | +50.0 | Different magnitude |
| Out-of-bounds | Tracked separately | -100.0 (curriculum-scaled) | Different |
| Curriculum | ✅ Yes (epochs 300-800) | ✅ Yes (epochs 300-800) | Same |

**Key Insight**: Airevolve simplified some aspects (removed tilt/yaw/safety penalties) but added action smoothness, resulting in a leaner but still effective reward structure for the timedlr task.

### Why Advanced Rewards?

The timedlr task is extremely difficult (0.25m gate spacing). Advanced rewards provide:
- **Richer learning signal**: Multiple reward components guide the agent better
- **Exponential proximity signals**: Stronger rewards for being close to gates
- **Action smoothness**: Penalizes jittery control, encouraging stable flight
- **Curriculum compatibility**: 3 components scale with difficulty progression
- **Curriculum learning**: Same progression schedule as AirframeOptimization2 (epochs 300-800)

---

## Quick Start

### Automatic Activation

Curriculum learning **automatically activates** when training on the **timedlr (Task B)** task:

```bash
# Curriculum is automatically enabled for timedlr
python examples/run_learning_evaluation.py \
    --task timedlr \
    --task-seed 42 \
    --timesteps 1e7 \
    --num-envs 50
```

### Manual Control

To enable curriculum learning programmatically:

```python
from airevolve.evolution_tools.evaluators.drone_gate_env import DroneGateEnv

# Curriculum is automatically enabled when use_advanced_reward=True
env = DroneGateEnv(
    num_envs=10,
    individual=your_drone_design,
    gates_pos=gate_positions,
    gate_yaw=gate_orientations,
    use_advanced_reward=True,  # Enables advanced reward + curriculum
)
```

To disable curriculum (even with advanced rewards):

```python
# Note: You would need to modify DroneGateEnv initialization
# Currently, advanced_reward automatically enables curriculum
# This is by design for the timedlr task
```

---

## How Curriculum Learning Works

### Core Concept

Curriculum learning uses a **difficulty schedule** that progresses over training time. The schedule is measured in "epochs" (groups of environment steps), and curriculum progress ranges from 0 (easy) to 1 (hard).

### Training Phases

```
Epoch Range        Curriculum Progress    Phase
─────────────────────────────────────────────────────────
0 - 299           0.0 (constant)         Warm-up / Learning basics
300 - 800         0.0 → 1.0 (linear)     Curriculum progression
800+              1.0 (constant)         Full difficulty / Fine-tuning
```

### Mathematical Formula

```
epoch = total_training_steps / curriculum_horizon_length

if curriculum_enabled:
    curriculum_progress = clip(
        (epoch - 300) / (800 - 300),
        min=0.0,
        max=1.0
    )
else:
    curriculum_progress = 1.0
```

where `curriculum_horizon_length = 16` (standard value, ~16 environment steps = 1 epoch).

---

## Reward Function with Curriculum

### Why Curriculum Matters

The **timedlr** task is extremely challenging:
- 100 gates with only 0.25m spacing
- Requires very slow, precise flight
- Standard rewards alone would be too difficult

### Three Components Scaled by Curriculum

#### 1. **Proximity Rewards** (Scaled by `1 - curriculum_progress`)

**Early training (epochs 0-300)**: Full proximity rewards
```
reward = large_bonus_for_getting_close_to_gate
```

**Mid training (epochs 300-800)**: Gradually reduced
```
reward = bonus_for_getting_close × (1 - 0.5) = 50% of full reward
```

**Late training (epochs 800+)**: Zeroed out
```
reward = bonus_for_getting_close × (1 - 1.0) = 0
```

**Why?** Proximity rewards are "shaping" signals that help agents learn basic navigation. Once they understand the task, these rewards are removed to force the agent to master precise gate-passing.

#### 2. **Out-of-Bounds Penalty** (Scaled by `curriculum_progress`)

**Early training**: No penalty for leaving bounds
```
penalty = -100.0 × 0.0 = 0
```

**Mid training**: Partial penalty
```
penalty = -100.0 × 0.5 = -50.0
```

**Late training**: Full penalty
```
penalty = -100.0 × 1.0 = -100.0
```

**Why?** Early exploration is encouraged. As training progresses, agents must learn to stay within boundaries.

#### 3. **Gate Passing Bonus** (Scaled by `curriculum_progress`)

**Early training**: No bonus
```
reward = +50.0 × 0.0 = 0
```

**Mid training**: Partial bonus
```
reward = +50.0 × 0.5 = +25.0
```

**Late training**: Full bonus
```
reward = +50.0 × 1.0 = +50.0
```

**Why?** Initially, the agent focuses on general navigation. As it learns, passing gates becomes the primary objective.

---

## Example Training Timeline

Suppose training with 1e7 timesteps and 100 parallel environments:

```
Wall-Clock Time    Total Steps    Epoch    Curriculum    Phase
────────────────────────────────────────────────────────────────
0 min              0              0        0.0           Warm-up
5 min              480k            30       0.0           Warm-up
10 min             960k            60       0.0           Warm-up
30 min             2.88M           180      0.0           Warm-up
45 min             4.32M           270      0.0           Warm-up
60 min             5.76M           360      0.12          Early progression
90 min             8.64M           540      0.48          Mid progression
120 min            11.52M          720      0.84          Late progression
150 min            14.4M+          900+     1.0           Full difficulty
```

### What This Means

- **First 45 minutes**: Agent learns to navigate through gates with full proximity rewards (easy)
- **45-120 minutes**: Curriculum gradually activates, making task harder
- **120+ minutes**: Full task difficulty, agent must be precise with gates

---

## Customization

### Configuration Parameters

All curriculum parameters are defined in `DroneGateEnv.__init__()`:

```python
# Location: airevolve/evolution_tools/evaluators/drone_gate_env.py (lines 103-107)

self.curriculum_enabled = True if self.use_advanced_reward else False
self.curriculum_horizon_length = 16              # Steps per "epoch"
self.curriculum_start_epoch = 300                # When curriculum begins
self.curriculum_end_epoch = 800                  # When curriculum completes
self._total_steps = 0                            # Internal counter
```

### Modifying Curriculum Schedule

To experiment with different schedules, modify `drone_gate_env.py`:

```python
# Make curriculum progress faster (reach full difficulty by epoch 500)
self.curriculum_end_epoch = 500

# Delay curriculum start (begin at epoch 400 instead of 300)
self.curriculum_start_epoch = 400

# Change epoch length (32 steps = slower epoch progression)
self.curriculum_horizon_length = 32
```

### Disabling Curriculum Entirely

Currently, curriculum is **automatically enabled** whenever `use_advanced_reward=True`. To disable it:

1. Set `use_advanced_reward=False` (but this also uses simple rewards)
2. Or modify line 106 in `drone_gate_env.py`:
   ```python
   # Before:
   self.curriculum_enabled = True if self.use_advanced_reward else False
   
   # After:
   self.curriculum_enabled = False  # Always off
   ```

---

## Monitoring Curriculum Progress

### During Training

Curriculum progress is logged to the info dict each step:

```python
# In your evaluation loop
for step in range(num_steps):
    actions, _ = model.predict(states)
    states, rewards, dones, infos = env.step(actions)
    
    # Curriculum info is available when curriculum is enabled
    if 'curriculum_progress' in infos[0]:
        progress = infos[0]['curriculum_progress']
        epoch = infos[0]['curriculum_epoch']
        print(f"Epoch {epoch}: curriculum_progress = {progress:.2f}")
```

### From Saved Monitor Data

The `monitor.csv` file saved during training contains episode rewards but **not curriculum progress**. To track curriculum:

```python
# You can estimate curriculum progress from wall-clock time:
import numpy as np

# Assume 100 parallel envs, horizon_length=16
# After t seconds, estimate steps completed
steps_per_second = 100 * (your_fps)  # depends on simulation speed
epoch = (total_seconds * steps_per_second) // 16
curriculum_progress = np.clip((epoch - 300) / 500, 0, 1)
```

---

## Comparison: With vs Without Curriculum

### Without Curriculum (lrcontinuous, figure8, etc.)

```
Training Reward Curve:
│     ╱╱╱╱╱ ← steady improvement, but initially harder
│  ╱╱╱
│╱╱
└─────────────────────→ Steps
```

**Characteristics:**
- Immediate full difficulty
- Longer initial learning plateau
- May converge to suboptimal policies
- Simpler reward signal, fewer rewards to tune

### With Curriculum (timedlr)

```
Training Reward Curve:
│              ╱╱╱╱╱ ← fast improvement in difficult phase
│    ╱╱╱╱╱╱╱
│  ╱╱╱
│╱╱ ← easier early phase
└───────────────────── → Steps
```

**Characteristics:**
- Early learning is easier (proximity rewards)
- Faster convergence overall
- Higher final performance potential
- More complex reward tuning required

---

## Best Practices

### 1. Use Curriculum for Challenging Tasks

✅ **Good**: Tasks with <0.5m gate spacing (timedlr)
```bash
python run_learning_evaluation.py --task timedlr --production
```

❌ **Not needed**: Tasks with adequate spacing
```bash
python run_learning_evaluation.py --task lrcontinuous --production
```

### 2. Train Long Enough

Curriculum needs time to progress. Minimum recommendations:

- **Quick testing**: 1e6 timesteps (100 envs = ~100 epochs covered)
- **Production**: 1e7 timesteps (100 envs = ~1000 epochs, full progression + convergence)
- **Thorough**: 1e8 timesteps (100 envs = ~10000 epochs, extensive fine-tuning)

```bash
# Minimum for meaningful curriculum progression
python run_learning_evaluation.py \
    --task timedlr \
    --timesteps 1e7 \
    --num-envs 100
```

### 3. Adjust Hyperparameters for Advanced Rewards

The advanced reward structure (with curriculum) may benefit from different hyperparameters than simple rewards:

```bash
# Use Optuna to find task-specific hyperparameters
python optuna_hyperparameter_search.py \
    --task timedlr \
    --n-trials 30 \
    --timesteps-per-trial 1e6 \
    --num-envs 20
```

### 4. Seed Management

For reproducible curriculum experiments, use consistent seeds:

```bash
# Always use same seed for gate generation
python run_learning_evaluation.py \
    --task timedlr \
    --task-seed 42 \
    --timesteps 1e7 \
    --num-envs 50
```

---

## Troubleshooting

### Issue: Learning is too slow initially

**Cause**: Warm-up phase (epochs 0-300) with easy rewards might be too short or too easy

**Solution**:
- Reduce `curriculum_start_epoch` to extend warm-up
- Increase `curriculum_end_epoch` to slow progression
- Run longer training (1e8+ timesteps)

### Issue: Learning plateaus mid-training

**Cause**: Curriculum shift at epoch 300 is too abrupt

**Solution**:
- Reduce `curriculum_start_epoch` (e.g., 250 instead of 300)
- Increase `curriculum_end_epoch` (e.g., 1000 instead of 800)
- Adjust learning rate downward for stability

### Issue: Agent crashes frequently in late training

**Cause**: Out-of-bounds penalty (scaled by curriculum) becomes too harsh

**Solution**:
- Increase `curriculum_horizon_length` to slow epoch progression
- Use higher learning rate (0.001 → 0.0005)
- Increase network capacity (medium → large architecture)

### Issue: Gates passed plateaus at low numbers

**Cause**: Task may be inherently difficult for your drone design

**Solution**:
- Verify drone can hover: `python examples/draw_blueprint.py`
- Check gate spacing is reasonable
- Try on easier task first (lrcontinuous or figure8)
- Consider evolving better morphologies

---

## Advanced: Custom Curriculum Schedules

For research purposes, you can implement custom schedules by modifying `drone_gate_env.py` (lines 453-458):

```python
# Default linear schedule
if self.curriculum_enabled:
    epoch = self._total_steps // self.curriculum_horizon_length
    denom = max(self.curriculum_end_epoch - self.curriculum_start_epoch, 1)
    curriculum_progress = np.clip((epoch - self.curriculum_start_epoch) / denom, 0.0, 1.0)
else:
    curriculum_progress = 1.0

# Alternative: Exponential schedule (faster early, slower late)
curriculum_progress = np.clip(
    1.0 - np.exp(-3.0 * (epoch - self.curriculum_start_epoch) / denom),
    0.0, 1.0
)

# Alternative: Step-based schedule (abrupt jumps)
steps = [0.2, 0.5, 0.8, 1.0]
thresholds = [300, 400, 600, 800]
curriculum_progress = next(
    (s for t, s in zip(thresholds, steps) if epoch >= t),
    1.0
)
```

---

## Related Documentation

- **[TIMEDLR_TASK_GUIDE.md](TIMEDLR_TASK_GUIDE.md)**: Task B specification and parameters
- **[PPO_TRAINING_GUIDE.md](PPO_TRAINING_GUIDE.md)**: General PPO training workflow
- **[TASK_COMPARISON_AIRFRAMEOPT2_VS_AIREVOLVE.md](TASK_COMPARISON_AIRFRAMEOPT2_VS_AIREVOLVE.md)**: Detailed reward function comparison

---

## Summary Table

| Aspect | Details |
|--------|---------|
| **Enabled for** | timedlr (Task B) only |
| **Activation** | Automatic when `use_advanced_reward=True` |
| **Schedule** | Epochs 300-800 (500 epoch progression) |
| **Components Scaled** | Proximity rewards (1-progress), penalties (progress), bonuses (progress) |
| **Typical Duration** | 45-120 minutes (1e7 timesteps, 100 envs) |
| **Key Benefit** | Enables learning of very difficult tasks (0.25m gate spacing) |
| **Customizable** | Yes (see Customization section) |
| **Disableable** | Yes (set `use_advanced_reward=False` or modify code) |

