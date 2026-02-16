"""
Phenotype Assembly Module

This module provides tools for generating physical drone phenotypes (STL files)
from evolved genomes using CAD (CadQuery).

Main API:
    - generate_stl_files: Generate STL files from a genome handler
    - quick_visualize_genome: Quick single-file generation for visualization

Example:
    >>> from airevolve.evolution_tools.genome_handlers import SphericalAngularDroneGenomeHandler
    >>> from airevolve.phenotype_assembly import generate_stl_files
    >>>
    >>> # Create and evolve individual
    >>> handler = SphericalAngularDroneGenomeHandler(...)
    >>> # ... evolution ...
    >>>
    >>> # Generate STL files
    >>> result = generate_stl_files(handler, output_dir="./my_drone")
"""

from .stl_generator import (
    generate_stl_files,
    quick_visualize_genome,
    STLGenerationResult
)

from .genome_adapter import (
    genome_to_cad_parameters,
    cad_parameters_to_assembly_vector,
    cad_parameters_to_individual_vector,
    ArmCADParameters,
    DroneCADParameters
)

from .core_plate import create_core_plate
from .part_generators import create_arm_assembly, create_landing_leg

__all__ = [
    # Main API
    'generate_stl_files',
    'quick_visualize_genome',
    'STLGenerationResult',
    # Genome conversion
    'genome_to_cad_parameters',
    'cad_parameters_to_assembly_vector',
    'cad_parameters_to_individual_vector',
    'ArmCADParameters',
    'DroneCADParameters',
    # Part generation functions (advanced usage)
    'create_core_plate',
    'create_arm_assembly',
    'create_landing_leg',
]

__version__ = '0.1.0'
