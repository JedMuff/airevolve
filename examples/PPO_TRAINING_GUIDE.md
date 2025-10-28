# PPO Training with Production Settings and Hyperparameter Optimization

This directory contains enhanced scripts for training drone morphologies with PPO reinforcement learning.

## Scripts

### 1. `run_learning_evaluation.py` - Enhanced with Production Settings

Train a single drone design with configurable parameters.

#### Quick Start Examples:

**Debug mode (fast testing):**
```bash
python run_learning_evaluation.py --timesteps 1e5 --num-envs 1
```

**Production mode (full training):**
```bash
python run_learning_evaluation.py --production
```
This automatically uses: 1e8 timesteps, 100 parallel environments, CPU device

**Custom configuration:**
```bash
python run_learning_evaluation.py \
    --task circle \
    --timesteps 1e7 \
    --num-envs 50 \
    --device cpu \
    --output-dir my_results
```

#### Available Options:

- `--task`: Flight task (circle, figure8, slalom, backandforth)
- `--timesteps`: Training timesteps (e.g., 1e5, 1e7, 1e8)
- `--num-envs`: Number of parallel environments
- `--device`: Training device (cpu, cuda:0, cuda:1) - **cpu recommended for MLP policies**
- `--output-dir`: Directory to save results
- `--no-videos`: Disable video creation
- `--production`: Use production settings (1e8 timesteps, 100 envs, cpu)

---

### 2. `optuna_hyperparameter_search.py` - Hyperparameter Optimization

Find optimal PPO hyperparameters for a given task using Optuna.

#### Quick Start:

**Fast search (10 trials, short training):**
```bash
python optuna_hyperparameter_search.py \
    --task figure8 \
    --n-trials 10 \
    --timesteps-per-trial 5e5 \
    --num-envs 10
```

**Production search (50 trials, longer training):**
```bash
python optuna_hyperparameter_search.py \
    --task figure8 \
    --n-trials 50 \
    --timesteps-per-trial 5e6 \
    --num-envs 50 \
    --device cpu
```

**Parallel optimization (multiple trials simultaneously):**
```bash
python optuna_hyperparameter_search.py \
    --task circle \
    --n-trials 100 \
    --n-jobs 4 \
    --storage sqlite:///optuna_study.db
```

#### Hyperparameters Searched:

The Optuna search optimizes 13 key PPO hyperparameters. Here's what each one controls:

**1. Learning Rate** (1e-5 to 1e-2, log scale)
- Controls how big steps the algorithm takes when updating the policy
- Too high: Training becomes unstable, policy changes too drastically  
- Too low: Training is very slow, may get stuck
- Think of it as the "step size" when learning

**2. Network Architecture** (small/medium/large)
- Small (32x32): Fast but limited learning capacity
- Medium (64x64): Good balance  
- Large (128x128x128): More powerful but slower
- Controls the "brain size" - bigger brains learn more complex behaviors

**3. N Steps** (256, 512, 1000, 2048)
- How many actions the drone takes before updating its policy
- Low values: Updates frequently, more reactive to recent experiences
- High values: Updates less frequently, considers longer sequences
- Like how long you practice before reflecting on what you learned

**4. Batch Size** (32, 64, 128, 256, 512, 1000)  
- How many experiences to process together when learning
- Small: More frequent updates, less stable
- Large: More stable updates, but slower
- Like studying from 10 examples vs 1000 examples at once

**5. N Epochs** (3 to 30)
- How many times to review the same batch of experiences  
- Low: Quick learning, might miss patterns
- High: Thorough learning, but risk of overthinking
- Like how many times you re-read your notes when studying

**6. Gamma** (0.9 to 0.9999, log scale)
- How much the drone cares about future rewards vs immediate rewards
- Low (0.9): Focuses on immediate rewards
- High (0.9999): Considers long-term consequences  
- High gamma = "I'll accept small costs now for big future rewards"

**7. GAE Lambda** (0.8 to 0.99)
- Balances between actual observed rewards and estimated future rewards
- Low: Trusts actual observations more
- High: Trusts the AI's predictions more
- Do you trust what actually happened or your guess about the future?

**8. Clip Range** (0.1 to 0.4)
- How much the policy can change in one update (prevents dramatic changes)
- Low: Very conservative updates, stable but slow
- High: Allows bigger changes, faster but less stable
- Speed limit for policy changes - prevents "jerky" learning

**9. Entropy Coefficient** (1e-8 to 0.1, log scale)
- How much the drone should explore vs exploit what it knows
- Low: Stick to what works (exploitation)
- High: Try new things more often (exploration)
- Controls curiosity level - high entropy = "let me try something different"

**10. Value Function Coefficient** (0.1 to 1.0)
- Importance of predicting future rewards vs learning actions
- Low: Focus more on learning actions
- High: Balance action learning with reward prediction
- How much effort to spend on "imagining outcomes" vs "learning actions"

**11. Max Gradient Norm** (0.3 to 5.0)
- Prevents neural network from making too large updates at once
- Low: Very gradual learning, very stable
- High: Allows faster learning, but less stable  
- Maximum allowed "correction size" when AI realizes it made a mistake

**12. Log Std Init** (-1.0 to 1.0)
- How "random" or "confident" the drone starts out
- Negative: Starts very confident, less exploration
- Positive: Starts uncertain, more exploration
- Initial confidence level affects early exploration behavior

**13. Network Architecture Type** (derived from net_arch)
- The complexity of the drone's "decision-making brain"
- Simple: Fast training, limited capability
- Complex: Slow training, high capability
- Like simple calculator vs powerful computer for making decisions

#### How These Work Together:

**Exploration vs Exploitation Balance:**
- `entropy_coef` and `log_std_init` control how much the drone experiments

**Learning Stability:**  
- `learning_rate`, `clip_range`, and `max_grad_norm` prevent dramatic changes

**Memory and Planning:**
- `gamma` and `gae_lambda` control future consequence consideration
- `n_steps` controls experience collection before learning

**Learning Efficiency:**
- `batch_size` and `n_epochs` control thoroughness of learning
- `net_arch` controls capacity for complex behaviors

**For Gate Navigation:**
- High `gamma` is usually good (long-term planning for course completion)
- Moderate `entropy_coef` helps explore different flight paths  
- Appropriate `n_steps` helps learn sequences like "approach → align → fly through"

#### Available Options:

- `--task`: Flight task to optimize for
- `--n-trials`: Number of trials to run (more = better search, slower)
- `--timesteps-per-trial`: Training timesteps per trial (shorter = faster search)
- `--num-envs`: Parallel environments per trial
- `--device`: Training device (cpu recommended)
- `--study-name`: Name for the Optuna study
- `--storage`: Database URL for persistent storage (e.g., sqlite:///optuna.db)
- `--n-jobs`: Number of parallel jobs (parallel optimization)
- `--output-dir`: Directory to save results

#### Output Files:

- `best_hyperparameters.pkl`: Python pickle with all results
- `best_hyperparameters.txt`: Human-readable text file
- `optimization_history.png`: Plot of fitness over trials (requires plotly/kaleido)
- `param_importances.png`: Parameter importance plot (requires plotly/kaleido)

---

## Workflow Recommendations

### 1. Quick Testing
```bash
# Test that everything works (1-2 minutes)
python run_learning_evaluation.py --timesteps 1e5 --num-envs 1
```

### 2. Baseline Training (Before Optimization)
```bash
# Train with default hyperparameters to establish baseline (~20-40 minutes)
python run_learning_evaluation.py \
    --task figure8 \
    --timesteps 1e7 \
    --num-envs 50 \
    --output-dir baseline_results
```

### 3. Hyperparameter Search
```bash
# Find good hyperparameters (~3-6 hours)
python optuna_hyperparameter_search.py \
    --task figure8 \
    --n-trials 30 \
    --timesteps-per-trial 1e6 \
    --num-envs 20
```

### 4. Optimized Training (After Optimization)
```bash
# Train with optimized hyperparameters (~20-40 minutes)
python run_learning_evaluation_optimized.py \
    --task figure8 \
    --timesteps 1e7 \
    --num-envs 50 \
    --output-dir optimized_results \
    --optuna-results optuna_results
```

### 5. Compare Results
```bash
# Compare baseline vs optimized performance (~1-2 minutes)
python compare_training_results.py \
    --baseline baseline_results \
    --optimized optimized_results \
    --output comparison_results
```

---

## ⏱️ Detailed Time Estimates

### Complete Workflow Timing:
- **Total time**: 4-7 hours (conservative) | 2-4 hours (optimistic)
- **With parallel Optuna**: 1.5-3 hours (using `--n-jobs 4`)

### Step-by-Step Breakdown:

#### Step 1: Quick Testing (1-2 minutes)
- **Purpose**: Verify installation and basic functionality
- **Configuration**: 1e5 timesteps, 1 environment
- **Very fast**, just a sanity check

#### Step 2: Baseline Training (20-40 minutes)
- **Configuration**: 1e7 timesteps, 50 parallel environments
- **Factors affecting time**:
  - CPU cores (more cores = faster with parallel envs)
  - Task complexity (figure8 < slalom < circle < backandforth)  
  - Hardware (newer CPUs significantly faster)
- **Typical range**: 20-40 minutes on modern multi-core CPU

#### Step 3: Hyperparameter Search (3-6 hours)
- **Configuration**: 30 trials × 1e6 timesteps each, 20 parallel environments
- **Breakdown**:
  - ~6-12 minutes per trial
  - Trials run sequentially by default
  - Early trials may fail faster (bad hyperparameters)
- **Speed up options**:
  - Use `--n-jobs 4` for parallel trials: reduces to ~1-2 hours
  - Reduce `--timesteps-per-trial` to 5e5: ~1.5-3 hours  
  - Use fewer trials (20 instead of 30): ~2-4 hours
- **Memory requirement**: ~2-4GB per parallel job

#### Step 4: Optimized Training (20-40 minutes)
- **Configuration**: Same as baseline (1e7 timesteps, 50 environments)
- **Duration**: Identical to baseline since same training configuration
- **Note**: The hyperparameters affect learning quality, not training speed

#### Step 5: Results Comparison (1-2 minutes)
- **Operations**: Load data, generate plots, create statistical report
- **Very fast**: Mostly file I/O and matplotlib plotting
- **Output**: 4-panel comparison plot + detailed text report

### ⚡ Quick Testing Workflow (~30 minutes total):

```bash
# Quick baseline (2-3 minutes)
python run_learning_evaluation.py \
    --task figure8 --timesteps 1e6 --num-envs 10 \
    --output-dir quick_baseline

# Quick hyperparameter search (20-25 minutes)
python optuna_hyperparameter_search.py \
    --task figure8 --n-trials 10 \
    --timesteps-per-trial 2e5 --num-envs 10

# Quick optimized training (2-3 minutes)  
python run_learning_evaluation_optimized.py \
    --task figure8 --timesteps 1e6 --num-envs 10 \
    --output-dir quick_optimized \
    --optuna-results optuna_results

# Compare results (<1 minute)
python compare_training_results.py \
    --baseline quick_baseline \
    --optimized quick_optimized \
    --output quick_comparison
```

### 🏭 Production Workflow (~8-24 hours total):

```bash
# Production baseline (1-2 hours)
python run_learning_evaluation.py --production \
    --output-dir production_baseline

# Comprehensive hyperparameter search (6-20 hours)
python optuna_hyperparameter_search.py \
    --task figure8 --n-trials 100 \
    --timesteps-per-trial 5e6 --num-envs 50

# Production optimized training (1-2 hours)
python run_learning_evaluation_optimized.py --production \
    --output-dir production_optimized \
    --optuna-results optuna_results
```

### 💡 Time Optimization Tips:

1. **Start with quick testing** to verify everything works
2. **Use parallel Optuna** (`--n-jobs`) if you have enough RAM (4GB+ per job)
3. **Reduce timesteps-per-trial** initially to explore hyperparameter space faster
4. **Run longer trials** only after identifying promising regions  
5. **Use CPU** for MLP policies (GPU won't provide significant speedup)
6. **More parallel environments** = faster training (up to your CPU core count)
7. **Consider task complexity**: figure8 trains faster than slalom or complex courses

### 🖥️ Hardware Requirements:

- **Minimum**: 4 CPU cores, 8GB RAM
- **Recommended**: 8+ CPU cores, 16GB+ RAM  
- **For parallel Optuna**: Additional 4GB RAM per parallel job
- **GPU**: Optional, CPU recommended for MLP policies

---

## Performance Tips

1. **Use CPU for MLP policies**: PPO with MLP policies performs better on CPU than GPU
2. **More parallel environments = faster training**: Use 50-100 for production
3. **Optuna search**: Start with 10-20 trials with short training to get rough estimates
4. **Longer training per trial**: Once you narrow down good regions, increase timesteps-per-trial
5. **Parallel search**: Use `--n-jobs` with `--storage` for faster hyperparameter search

---

## Expected Runtimes

### run_learning_evaluation.py
- Debug (1e5 steps, 1 env): ~1 minute
- Medium (1e7 steps, 50 envs): ~15-30 minutes
- Production (1e8 steps, 100 envs): ~1-2 hours

### optuna_hyperparameter_search.py
- Quick (10 trials, 5e5 steps/trial, 10 envs): ~30 minutes
- Medium (30 trials, 1e6 steps/trial, 20 envs): ~3-5 hours
- Production (50 trials, 5e6 steps/trial, 50 envs): ~10-20 hours
- With parallel jobs (n_jobs=4): Divide by ~3-4

---

## Requirements

All required packages should be installed in your conda environment:
- numpy
- torch
- stable-baselines3
- optuna
- matplotlib
- pandas
- python-fcl

Optional for visualizations:
- plotly
- kaleido (for saving Optuna plots)

Install optional packages:
```bash
conda run -n python3.9 pip install plotly kaleido
```

---

## Comparing Before vs After Optimization

### Method 1: Using Training Curves

Both `run_learning_evaluation.py` and the optimized version automatically save training curves as:
- `monitor.csv`: Episode rewards over time
- `figure.png`: Reward plot

**Key metrics to compare:**
- **Final performance**: Last 100 episodes average reward
- **Learning speed**: How quickly rewards improve
- **Stability**: Variance in rewards during training
- **Sample efficiency**: Performance at same timestep count

### Method 2: Using the Comparison Script

The `compare_training_results.py` script (see below) automatically:
1. Loads training data from both runs
2. Creates side-by-side comparison plots
3. Calculates statistical comparisons
4. Generates a summary report

### Method 3: Multiple Evaluation Runs

For robust comparison, run multiple training sessions:

```bash
# Baseline (3 runs)
for i in {1..3}; do
    python run_learning_evaluation.py \
        --output-dir baseline_run_$i \
        --task figure8 --timesteps 1e7
done

# Optimized (3 runs)  
for i in {1..3}; do
    python run_learning_evaluation_optimized.py \
        --output-dir optimized_run_$i \
        --task figure8 --timesteps 1e7
done

# Compare all runs
python compare_multiple_runs.py \
    --baseline-dirs baseline_run_1 baseline_run_2 baseline_run_3 \
    --optimized-dirs optimized_run_1 optimized_run_2 optimized_run_3
```

### What to Look For

**Successful Optimization Shows:**
- Higher final fitness scores
- Faster convergence (reaches good performance sooner)
- More stable training (less variance)
- Better sample efficiency (same performance with fewer timesteps)

**Example Improvements:**
```
Baseline:     Final fitness = 2.3 ± 0.8 gates (after 10M steps)
Optimized:    Final fitness = 4.1 ± 0.3 gates (after 10M steps)
Improvement:  +78% performance, +62% stability
```
