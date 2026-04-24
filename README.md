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

## Quick Start

Run a complete evolution experiment:

```bash
python examples/evolution/run_evolution.py --genome-handler spherical --population-size 50 --generations 100 --gate-cfg figure8
```
## Examples

The `examples/` directory is organised by purpose:

- `examples/evolution/` — end-to-end evolution runners (`run_evolution.py`,
  `run_evolution_with_lee_tuning.py`, `run_evolution_with_optimization_repair.py`).
- `examples/simulation/` — single-drone simulation demos
  (`run_3D_simulation_lee_ctrl.py`).
- `examples/tuning/` — controller and repair tuning
  (`tune_lee_controller_gates.py`, `tune_lee_controller_gates_matched.py`,
  `optimization_repair_demo.py`).
- `examples/videos/` — render flight videos
  (`make_video.py`, `make_lee_video.py`).
- `examples/visualization/` — morphology / genome visualisation
  (`genome_visualizer_demo.py`, `sample_genomes.py`, `draw_blueprint.py`,
  `visualize_cppn_genome.py`, `visualize_initial_bspline.py`,
  `generate_drone_stl_from_genome.py`).

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
  author={---},
  year={2025},
  url={---}
}
```

# TODO
- Add remaining parts to assembly as optional parameter for full aesthetics: landing legs, battery, control board+stand offs, rasperry pi holder and raspberry pi, motor intermediary part, motors, propellers.