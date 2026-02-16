# Phenotype Assembly Module

This module converts evolved drone genomes into physical 3D models (STL files) that can be 3D printed or visualized.

## Quick Start

```python
from airevolve.evolution_tools.genome_handlers import SphericalAngularDroneGenomeHandler
from airevolve.phenotype_assembly import generate_stl_files

# Create or load an evolved individual
handler = SphericalAngularDroneGenomeHandler(...)
# ... evolve or load genome ...

# Generate STL files
result = generate_stl_files(
    genome_handler=handler,
    output_dir="./my_drone"
)

print(f"Assembly file: {result.assembly_file}")
```

## Main API

### `generate_stl_files()`

The primary function for generating STL files from a genome.

**Parameters:**
- `genome_handler`: `SphericalAngularDroneGenomeHandler` instance with evolved genome
- `output_dir`: Directory to save STL files (default: `"./drone_stls"`)
- `include_assembly`: Generate full assembled drone STL (default: `True`)
- `include_landing_leg`: Generate landing leg STL (default: `False`)
- `include_individual_parts`: Generate separate STL for each arm (default: `True`)
- `include_step_files`: Save STEP files for CAD editing (default: `True`)
- `distribute_arms_evenly`: Distribute arms evenly around disc (default: `True`)
- `magnitude_to_length_scale`: Scale factor (mm per unit magnitude, default: `100.0`)
- `part_config`: Optional dict to override default part parameters

**Returns:** `STLGenerationResult` with paths to all generated files

### `quick_visualize_genome()`

Generate a single STL file for quick visualization.

```python
from airevolve.phenotype_assembly import quick_visualize_genome

output_file = quick_visualize_genome(handler, output_file="drone.stl")
```

## Genome to CAD Mapping

The module converts `SphericalAngularDroneGenomeHandler` genome format to CAD parameters:

### Genome Format
- Shape: `(max_narms, 6)` with NaN masking
- Columns: `[magnitude, arm_rotation, arm_pitch, motor_rotation, motor_pitch, direction]`
- Angles in radians, spherical coordinates

### CAD Parameters
- `arm_length`: Magnitude × `magnitude_to_length_scale` (mm)
- `arm_attachment_angle`: Position around disc (0-360°)
- `arm_tilt`: Orientation from vertical (degrees)
- `motor_tilt`, `motor_azimuth`: Motor disc orientation (degrees)

## Module Structure

```
airevolve/phenotype_assembly/
├── __init__.py              # Public API exports
├── README.md                # This file
├── config.py                # Default configuration parameters
├── genome_adapter.py        # Genome ↔ CAD parameter conversion
├── stl_generator.py         # Main STL generation API
├── core_plate.py            # Core plate CAD generation
├── part_generators.py       # Arm and landing leg generation
├── full_drone_assembly.py   # Original assembly script (legacy)
├── main.py                  # Original batch script (legacy)
└── [other legacy files]     # Preserved for backward compatibility
```

## Examples

See `/examples/generate_drone_stl_from_genome.py` for complete examples including:
- Basic STL generation
- Loading evolved individuals from files
- Custom part configurations
- Quick visualization

## Advanced Usage

### Custom Part Configuration

```python
custom_config = {
    'sphere_radius': 15,       # Larger sphere (default: 12mm)
    'disc_diameter': 25,       # Larger motor disc (default: 23mm)
    'disc_thickness': 4,       # Thicker disc (default: 3mm)
    'clamp_jaw_length': 13,    # Custom clamp size
    # ... see part_generators.py for all options
}

result = generate_stl_files(
    genome_handler=handler,
    output_dir="./custom_drone",
    part_config=custom_config
)
```

### Using Individual Part Generators

```python
from airevolve.phenotype_assembly import (
    create_core_plate,
    create_arm_assembly,
    create_landing_leg
)

# Generate individual parts
core_plate = create_core_plate(plate_diameter=70)

arm_parts = create_arm_assembly(
    arm_tilt=-30,
    arm_azimuth=0,
    motor_tilt=-90,
    motor_azimuth=0,
    arm_length=60
)

landing_leg = create_landing_leg()
```

### Genome Conversion Functions

```python
from airevolve.phenotype_assembly import (
    genome_to_cad_parameters,
    cad_parameters_to_assembly_vector
)

# Convert genome to CAD parameters
cad_params = genome_to_cad_parameters(handler)

# Access individual arm parameters
for arm in cad_params.arms:
    print(f"Arm: length={arm.arm_length}mm, angle={arm.arm_attachment_angle}°")

# Convert to assembly vector format
assembly_vector = cad_parameters_to_assembly_vector(cad_params)
```

## Output Files

The module generates:

1. **STL files** (`.stl`) - For 3D printing and visualization
   - `core_plate.stl` - Central mounting plate
   - `arm_1.stl`, `arm_2.stl`, ... - Individual arms
   - `full_drone_assembly.stl` - Complete assembled drone
   - `landing_leg.stl` - Optional landing gear

2. **STEP files** (`.step`) - For CAD editing (if `include_step_files=True`)
   - Preserve individual parts and colors
   - Editable in CAD software (FreeCAD, Fusion 360, etc.)

## Dependencies

- `cadquery` - CAD modeling
- `numpy` - Numerical operations
- Parent module: `airevolve.evolution_tools.genome_handlers`

## Backward Compatibility

Legacy scripts are preserved:
- `full_drone_assembly.py` - Can still be run as standalone script
- `main.py` - Original batch generation script
- Individual part scripts - Still functional

## Configuration

Default parameters are in `config.py`:
- Screw sizes and tolerances
- Plate dimensions
- Structural parameters

These can be overridden via `part_config` parameter or by modifying `config.py`.

## Coordinate Systems

The module handles coordinate system conversions:
- **Genome**: Spherical coordinates (r, θ, φ) + motor angles
- **CAD**: Cartesian with Euler angles (degrees)
- **STL**: Right-handed coordinate system (Z-up)

## Troubleshooting

### Issue: No STL files generated
- Check that genome has valid (non-NaN) arms: `handler.get_arm_count()`
- Verify output directory permissions

### Issue: Arms not positioned correctly
- Adjust `distribute_arms_evenly` parameter
- Check `magnitude_to_length_scale` for reasonable arm lengths

### Issue: CAD errors
- Ensure CadQuery is installed: `pip install cadquery`
- Some parameter combinations may cause geometric issues (e.g., motor_tilt=±90°)

## Future Enhancements

Potential improvements:
- [ ] Support for more genome types
- [ ] Automatic collision detection in STL
- [ ] Assembly instructions generation
- [ ] Bill of materials (BOM) export
- [ ] Integration with FEA analysis tools
