# AirEvolve

An evolutionary algorithm framework for optimizing drone morphology and control. AirEvolve uses evolutionary computation techniques to evolve drone designs that can navigate through gate courses.

## Installation

### Prerequisites

- Python 3.7 or higher
- CUDA-capable GPU (recommended for reinforcement learning training)

### Install from source

```bash
git clone ---
cd airevolve
pip install -e .
```

### Additional dependency: dronehover

AirEvolve requires the `dronehover` package for hover feasibility checks and motor thrust computation. This package is not on PyPI and must be installed manually:

```bash
pip install git+<dronehover-repo-url>
```

For trimesh-based 3D visualisation, install the optional extra:

```bash
pip install -e ".[vis]"
```

## Quick Start: Reproducing the Research Paper

The following examples demonstrate how to reproduce the experiments described in our paper using the strictly Darwinian, bi-objective NSGA-II evolutionary algorithm.

### 1. Run a Single Experiment (Local/Interactive)

To run a single repetition of the standard PPO + Power-Aware NSGA-II evaluation matching the paper's exact setup (μ=24, λ=32, 1×10^7 training steps):

```bash
python experimentation/run_exp_standard_ppo_power_ea.py \
    --population-size 24 \
    --num-mutate 32 \
    --generations 40 \
    --training-timesteps 10000000 \
    --gate-cfg figure8 \
    --num-workers 24
```
*Note: This script initializes every individual from scratch (pure Darwinian approach) and leaves power/energy constraints exclusively to the evolutionary grading phase.*

### 2. Run the Full Experiment Suite (SLURM Cluster)

To execute the full, rigorous experiment suite (5 repetitions of `figure8` and 5 repetitions of `backandforth`) on a SLURM cluster, submit the provided bash script:

```bash
sbatch scripts/run_5_experiment_slurm.sh
```
*This uses an array job (1-10) of 10 nodes to parallelize repetitions across nodes. Expected computation time 115 hours for an array*

### 3. Standalone Training for Evaluation

To train a single morphology manually (when evaluating learned controllers of indivudals independently from the evolutionary loop), this script can be used with the battery model enabled (but controller not aware of the battery consumption, just the environment models the physics of electronics):

```bash
python airevolve/evolution_tools/evaluators/gate_train.py \
    --gate-cfg figure8 \
    --training-timesteps 100000000 \
    --num-envs 5
```

*On a Macbook M2 pro computation time ~30-35 minutes with 50 000 it/s, because the environment has been vectorized to numpy arrays*
## Visualization & Analysis

Once you have generated results, you can use the following scripts to analyze the population, plot pareto fronts, and generate flight videos. 

### Plotting Experimental Results

The analysis pipeline parses the `results/` folder for `evolution_data.csv` and generates learning metrics, morphological diversity plots, and pareto fronts. Run it directly from the repository root:

```bash
python analyze_results.py
```

### Rendering Flight Videos

You can generate MP4 videos of a trained policy navigating the track. The video scripts require a directory containing a morphology (`genome.npy`) and a trained model (`policy.zip`).

**Single Drone Flight:**
To generate a flight video (top-down view, isometric view, and real-time battery constraints) of a specific evolved individual:

```bash
python examples/videos/evolved_trained_video.py \
    path/to/results/exp_standard_ppo_power_ea_figure8_rep1/rl_logs/generation_40/individual_1234 \
    --task figure8

python examples/videos/evolved_trained_video.py \
    path/to/results/exp_standard_ppo_power_ea_figure8_rep1/rl_logs/generation_40/individual_1234 \
    --task backandforth
```

**Side-by-Side Comparison:**
To compare two different trained morphologies flying the track simultaneously:

```bash
python examples/videos/compare_trained_video.py \
    path/to/individual_A \
    path/to/individual_B \
    --gate_cfg figure8
```

## Architecture

### Core Components

- **Evolution Tools**: Core evolutionary algorithm implementations
  - `strategies/`: Evolution strategies (μ+λ, μ,λ)
  - `selectors/`: Parent selection methods (tournament, top-k)
  - `genome_handlers/`: Genome representation and operators
  - `evaluators/`: Fitness evaluation functions

- **Simulator**: Physics-based drone simulation
  - `simulation/`: Core simulation engine with propeller physics
  - `visualization/`: 3D visualization and animation tools

- **Analysis Tools**: Post-evolution analysis and visualization
  - `inspection_tools/`: Fitness plotting, diversity analysis, morphological descriptors, learning descriptors
  - `behavioural_analysis/`: Trajectory analysis and performance metrics

- **Phenotype Assembly**: Physical fabrication from evolved genomes
  - Converts a genome into printable STL / editable STEP files
  - See [`airevolve/phenotype_assembly/README.md`](airevolve/phenotype_assembly/README.md) for full documentation

- **Experimentation Tools**
  - `experimentation/`: Research scripts and data collection tools

### Genome Representations

1. **Spherical Angular**: `[magnitude, arm_rotation, arm_pitch, motor_rotation, motor_pitch, direction]` (Used in the my research paper)
2. **Cartesian Euler**: Standard 3D Cartesian coordinates with Euler angles

## Configuration Options

### Evolution Parameters

- `--genome-handler`: Choose between 'spherical' or 'cartesian' representations
- `--population-size`: Number of individuals per generation
- `--generations`: Number of evolutionary generations
- `--num-mutate`: Number of mutation operations per generation
- `--strategy-type`: Evolution strategy ('plus' or 'comma')
- `--symmetry`: Bilateral symmetry plane ('xy', 'xz', 'yz', 'none')
- `--init-pop-mode`: Random or hover repair

### Gate Training Parameters

- `--gate-cfg`: Gate configuration ('backandforth', 'figure8', 'circle', 'slalom')
- `--training-timesteps`: RL training duration per individual
- `--num-envs`: Number of parallel training environments
- `--device`: Training device ('cuda:0', 'cpu')

## Testing

Run the test suite:

```bash
python unit_tests/run_all_tests.py
```

Individual test modules are available in the `unit_tests/` directory.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

# TODO
- Add remaining parts to assembly as optional parameter for full aesthetics: landing legs, battery, control board+stand offs, rasperry pi holder and raspberry pi, motor intermediary part, motors, propellers.