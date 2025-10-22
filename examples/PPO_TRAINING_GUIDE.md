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

- **Learning rate**: 1e-5 to 1e-2 (log scale)
- **Network architecture**: small (32x2), medium (64x2), large (128x3)
- **N steps**: 256, 512, 1000, 2048
- **Batch size**: 32, 64, 128, 256, 512, 1000
- **N epochs**: 3 to 30
- **Gamma**: 0.9 to 0.9999 (log scale)
- **GAE lambda**: 0.8 to 0.99
- **Clip range**: 0.1 to 0.4
- **Entropy coefficient**: 1e-8 to 0.1 (log scale)
- **Value function coefficient**: 0.1 to 1.0
- **Max gradient norm**: 0.3 to 5.0
- **Log std init**: -1.0 to 1.0

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

### 2. Hyperparameter Search
```bash
# Find good hyperparameters (several hours)
python optuna_hyperparameter_search.py \
    --task figure8 \
    --n-trials 30 \
    --timesteps-per-trial 1e6 \
    --num-envs 20
```

### 3. Production Training
```bash
# Train with best hyperparameters (update code with best params)
python run_learning_evaluation.py --production
```

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
