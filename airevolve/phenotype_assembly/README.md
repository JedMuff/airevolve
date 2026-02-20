# Phenotype Assembly

Converts evolved drone genomes into physical 3D models (STL / STEP files) ready for 3D printing or inspection in CAD software.

## Quick Start

```python
from airevolve.evolution_tools.genome_handlers import SphericalAngularDroneGenomeHandler
from airevolve.phenotype_assembly import generate_stl_files

handler = SphericalAngularDroneGenomeHandler(...)
# ... evolve or load genome ...

result = generate_stl_files(handler, output_dir="./my_drone")
print(result.assembly_file)   # Path to full_drone_assembly.stl
```

See `examples/generate_drone_stl_from_genome.py` for complete worked examples.

---

## Module Structure

```
airevolve/phenotype_assembly/
├── __init__.py          — public API
├── config.py            — physical constants (plate diameter, screw sizes, …)
├── models.py            — dataclasses: ArmCADParameters, DroneCADParameters,
│                          AssemblyConfig, STLGenerationResult
├── genome_adapter.py    — genome → DroneCADParameters conversion
├── assembler.py         — places parts onto the plate; assemble_drone()
├── generator.py         — orchestrates the pipeline; file I/O
└── parts/
    ├── core_plate.py    — hub-and-spoke central plate
    ├── arm_mount.py     — sphere-clamp that grips the plate rim
    └── motor_arm.py     — arm tube + motor-mounting disc
```

---

## API Reference

### `generate_stl_files()`

```python
from airevolve.phenotype_assembly import generate_stl_files, AssemblyConfig

result = generate_stl_files(
    genome_handler,                      # SphericalAngularDroneGenomeHandler
    output_dir="./drone_stls",           # where to write files
    include_assembly=True,               # write full_drone_assembly.stl
    include_individual_parts=True,       # write core_plate.stl + arm_N.stl
    include_step_files=True,             # write STEP equivalents
    include_landing_leg=False,           # not yet implemented
    distribute_arms_evenly=False,        # True = ignore genome arm_rotation
    magnitude_to_length_scale=100.0,     # genome magnitude → mm
    assembly_config=AssemblyConfig(),    # physical dimensions (see below)
)
```

Returns `STLGenerationResult` with fields:
- `output_dir` — `Path` to the output directory
- `core_plate_file` — `Path` or `None`
- `arm_files` — `List[Path]`, one per arm
- `assembly_file` — `Path` or `None`
- `step_files` — `List[Path]`

### `quick_visualize_genome()`

```python
from airevolve.phenotype_assembly import quick_visualize_genome

path = quick_visualize_genome(handler, output_file="drone.stl")
```

Generates a single combined STL with all defaults.

### `AssemblyConfig`

All physical dimensions in one typed dataclass (replaces the old `part_config` dict).

```python
from airevolve.phenotype_assembly import AssemblyConfig

cfg = AssemblyConfig(
    sphere_radius=15,    # mm — larger pivot sphere (default 12)
    disc_diameter=25,    # mm — larger motor disc   (default 23)
    disc_thickness=4,    # mm — thicker motor disc  (default 3)
)

result = generate_stl_files(handler, assembly_config=cfg)
```

Key fields and defaults:

| Field | Default | Description |
|---|---|---|
| `plate_diameter` | 60 mm | Core plate outer diameter |
| `plate_thickness` | 2 mm | Core plate thickness |
| `outer_ring_width` | 7.5 mm | Width of outer annular ring |
| `sphere_radius` | 12 mm | Pivot sphere radius |
| `clamp_inset` | 10.5 mm | How far sphere centre sits inside the plate rim |
| `cylinder_inner_radius` | 4 mm | Arm tube inner radius |
| `wall_thickness` | 1 mm | Arm tube wall thickness |
| `disc_diameter` | 23 mm | Motor mount disc diameter |
| `disc_thickness` | 3 mm | Motor mount disc thickness |
| `motor_screw_count` | 4 | Number of motor mounting screws |

---

## Genome → CAD Coordinate Mapping

The genome uses spherical coordinates. The mapping to physical assembly parameters is:

| Genome column | Range | CAD field | Conversion |
|---|---|---|---|
| `magnitude` | 0.5–2.0 | `arm_length` (mm) | `× magnitude_to_length_scale` |
| `arm_rotation` (θ) | 0–2π rad | `attachment_angle` (°) | `degrees(θ)` |
| `arm_pitch` (φ) | 0–π rad | `arm_elevation` (°) | `90 − degrees(φ)` |
| `motor_rotation` | 0–2π rad | `motor_azimuth` (°) | `degrees(motor_rotation)` |
| `motor_pitch` | 0–π rad | `motor_tilt` (°) | `90 − degrees(motor_pitch)` |
| `direction` | {0, 1} | `direction` | unchanged |

`arm_elevation = 0°` → arm lies horizontal. `arm_elevation = 90°` → arm points straight up.

---

## Output Files

| File | Contents |
|---|---|
| `core_plate.stl` | Central hub-and-spoke plate |
| `arm_N.stl` | Combined arm mount + motor arm for arm N |
| `full_drone_assembly.stl` | All parts in assembled position |
| `core_plate.step` | Core plate with full BREP geometry |
| `full_drone_assembly.step` | Full assembly with named, coloured parts |

---

## Advanced Usage

### Accessing low-level functions

```python
from airevolve.phenotype_assembly import (
    genome_to_cad_parameters,
    place_arm_on_plate,
    assemble_drone,
    create_core_plate,
    create_arm_mount,
    create_motor_arm,
    AssemblyConfig,
)

cfg = AssemblyConfig()
cad_params = genome_to_cad_parameters(handler)

# Place one arm
arm_parts = place_arm_on_plate(cad_params.arms[0], cfg)
# arm_parts is a dict: {sphere, upper_jaw, lower_jaw, motor_arm} → cq.Workplane

# Build the full compound
compound, named = assemble_drone(cad_params.arms, cfg, create_core_plate())
```

### Inspecting per-arm parameters

```python
cad_params = genome_to_cad_parameters(handler)
for i, arm in enumerate(cad_params.arms):
    print(f"Arm {i+1}:")
    print(f"  attachment_angle : {arm.attachment_angle:.1f}°")
    print(f"  arm_elevation    : {arm.arm_elevation:.1f}°")
    print(f"  arm_length       : {arm.arm_length:.1f} mm")
    print(f"  motor_tilt       : {arm.motor_tilt:.1f}°")
    print(f"  motor_azimuth    : {arm.motor_azimuth:.1f}°")
    print(f"  direction        : {'CW' if arm.direction else 'CCW'}")
```

---

## Dependencies

- `cadquery` — CAD solid modelling
- `numpy` — numerical operations
- `airevolve.evolution_tools.genome_handlers` — genome format

---

## Troubleshooting

**No STL files generated** — check that the genome has at least one valid arm: `handler.get_arm_count()`.

**CAD errors / geometry failures** — motor tilt values very close to ±90° can cause singularities; the code nudges them by 0.001° automatically. If problems persist, try a slightly different `motor_tilt` value in `AssemblyConfig`.

**Arms look wrong in the STL** — use `distribute_arms_evenly=False` (the default) so attachment angles come directly from the genome's `arm_rotation` values, matching what the simulator sees.
