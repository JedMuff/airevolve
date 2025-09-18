# AirEvolve

An evolutionary algorithm framework for optimizing drone morphology and control. AirEvolve uses evolutionary computation techniques to evolve drone designs that can navigate through complex gate courses and perform various flight tasks.

## Features

- **Multiple Genome Representations**: Support for both spherical angular and Cartesian Euler coordinate systems
- **Evolutionary Strategies**: μ+λ and μ,λ evolution strategies with configurable selection pressure
- **Morphological Optimization**: Evolve drone arm configurations, motor placements, and propeller orientations
- **Gate Navigation**: Built-in support for training drones to navigate through gate courses
- **Symmetry Constraints**: Optional bilateral symmetry enforcement (XY, XZ, YZ planes)
- **Visualization Tools**: Comprehensive plotting and animation capabilities for analysis
- **Flexible Evaluation**: Multiple evaluation environments including gate navigation, hover tasks, and custom scenarios

## Installation

### Prerequisites

- Python 3.7 or higher
- CUDA-capable GPU (recommended for reinforcement learning training)

### Install from source

```bash
git clone https://github.com/JedMuff/airevolve.git
cd airevolve
pip install -e .
```

## Quick Start

Run a complete evolution experiment:

```bash
python examples/run_evolution.py --genome-handler spherical --population-size 50 --generations 100 --gate-cfg figure8
```
## Examples

The `examples/` directory contains several demonstration scripts:

- `run_evolution.py`: Complete evolution pipeline with gate training
- `genome_visualizer_demo.py`: Visualize drone morphologies
- `make_video.py`: Create animations of drone behavior
- `sample_genomes.py`: Generate and analyze random drone designs

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

- **Experimentation Tools**
  - `experimentation/`: Research scripts and data collection tools

### Genome Representations

1. **Spherical Angular**: `[magnitude, arm_rotation, arm_pitch, motor_rotation, motor_pitch, direction]`
2. **Cartesian Euler**: Standard 3D Cartesian coordinates with Euler angles

## Configuration Options

### Evolution Parameters

- `--genome-handler`: Choose between 'spherical' or 'cartesian' representations
- `--population-size`: Number of individuals per generation
- `--generations`: Number of evolutionary generations
- `--num-mutate`: Number of mutation operations per generation
- `--strategy-type`: Evolution strategy ('plus' or 'comma')
- `--symmetry`: Bilateral symmetry plane ('xy', 'xz', 'yz', 'none')

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

## Citation

If you use AirEvolve in your research, please cite:

```bibtex
@software{airevolve2025,
  title={Unconventional Hexacopters via Evolution and Learning: Performance Gains and New Insights},
  author={Jed Muff},
  year={2025},
  url={https://github.com/JedMuff/airevolve}
}
```